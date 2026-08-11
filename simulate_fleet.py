#!/usr/bin/env python3
"""
simulate_fleet.py — Multi-node & heterogeneous-storage fleet simulator.

Boots N simulated Swarm nodes on a single host using distinct loopback ports.
Node 0 uses real O_DIRECT reads (DirectFileExpertStore or GGUFExpertStore);
nodes 1..N-1 use SimulatedExpertStore with bandwidth caps to model slower peers.

Usage:
    from simulate_fleet import SimulatedFleet

    fleet = SimulatedFleet(num_nodes=3, gguf_path="/tmp/model.gguf",
                           node_bandwidths=[3200, 280, 280])
    await fleet.start()
    # ... run inference ...
    await fleet.stop()

Design rules:
  - Python 3.11+, standard library only, plus all existing Swarm modules.
  - Each "node" is a fully-fledged async stack: NodeIdentity, RpcServer,
    RpcClient, FailoverCoordinator, PipelineCoordinator, GangSync.
  - Uses real FleetTable (not fake) but populates it directly rather than
    via UDP broadcast, since loopback multicast has known WSL2 limitations.
  - All background tasks strongly referenced.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from node_identity import FleetTable, NodeDescriptor
from rpc import RpcClient, RpcServer
from sharding import NodeCapability, ShardAssignment, compute_assignment
from failover import FailoverCoordinator
from gang_sync import GangSync
from pipeline import PipelineCoordinator
from storage_io import ExpertStore, SimulatedExpertStore, DirectFileExpertStore

logger = logging.getLogger("swarm.simulate_fleet")


def _contextlib_suppress(*exceptions):
    return contextlib.suppress(*exceptions)


# ── GGUFExpertStore ──────────────────────────────────────────────────────

class GGUFExpertStore(ExpertStore):
    """ExpertStore that reads tensors from a GGUF file via os.pread.

    Uses ``gguf_stream_reader`` for tensor offset lookups, then issues
    O_DIRECT-aligned reads against the backing file.  Falls back to buffered
    reads on platforms/filesystems that don't support O_DIRECT.

    Tensor naming convention:
      ``blk.{layer}.expert.{expert}.weight``
    """

    _O_DIRECT_AVAILABLE = hasattr(os, "O_DIRECT")
    _DEFAULT_ALIGNMENT = 4096

    def __init__(
        self,
        gguf_path: str | Path,
        tensor_name_pattern: str = "blk.{layer}.expert.{expert}.weight",
    ) -> None:
        from gguf_stream_reader import GGUFReader

        self._path = str(gguf_path)
        self._reader = GGUFReader(self._path)
        self._pattern = tensor_name_pattern

        # Use buffered I/O for GGUF reads.  O_DIRECT requires offset/size
        # alignment to the filesystem block size.  GGUF tensor offsets are
        # determined by the file header and are unlikely to be aligned.
        # For the production path, DirectFileExpertStore handles O_DIRECT
        # properly with mmap-backed aligned buffers; GGUFExpertStore is
        # for integration testing where correctness trumps raw throughput.
        self._fd = os.open(self._path, os.O_RDONLY)
        self._use_odirect = False

        # Cache tensor offsets for fast lookup
        self._offsets: dict[tuple[int, int], tuple[int, int, str]] = {}
        self._tensor_names: set[str] = set(self._reader.list_tensors())

        logger.info(
            "GGUFExpertStore: %s, %d tensors",
            self._path, len(self._tensor_names),
        )

    async def read_expert(self, layer: int, expert: int) -> bytes:
        key = (layer, expert)
        if key not in self._offsets:
            name = self._pattern.format(layer=layer, expert=expert)
            offset, size, dtype = self._reader.get_tensor_offset(name)
            self._offsets[key] = (offset, size, dtype)

        offset, size, _dtype = self._offsets[key]
        return await asyncio.to_thread(self._buffered_pread, offset, size)

    async def close(self) -> None:
        if hasattr(self, "_fd"):
            os.close(self._fd)

    def _buffered_pread(self, offset: int, size: int) -> bytes:
        data = os.pread(self._fd, size, offset)
        # os.pread can return fewer bytes than requested on some
        # filesystems; loop until we have the full amount.
        while len(data) < size:
            more = os.pread(self._fd, size - len(data), offset + len(data))
            if not more:
                break
            data += more
        return data

    def _aligned_pread(self, offset: int, size: int) -> bytes:
        alignment = self._DEFAULT_ALIGNMENT
        aligned_offset = (offset // alignment) * alignment
        offset_skip = offset - aligned_offset
        read_size = ((offset_skip + size + alignment - 1) // alignment) * alignment

        buf = os.pread(self._fd, read_size, aligned_offset)
        return buf[offset_skip : offset_skip + size]


# ── SimulatedFleet ───────────────────────────────────────────────────────


class SimulatedFleet:
    """Boot an N-node Swarm cluster on a single host.

    Parameters
    ----------
    num_nodes:
        Number of simulated nodes (at least 1).
    base_port:
        Starting TCP port for RPC servers.  Node i gets port base_port + i.
    expert_store_factories:
        List of callables ``(node_index) -> ExpertStore``, one per node.
        If shorter than *num_nodes*, remaining nodes get a default
        SimulatedExpertStore at 280 MB/s (eMMC class).
    hardware_gen:
        Hardware generation tag for fleet discovery.
    own_storage_bandwidths:
        Per-node storage bandwidth in Mbps, used for sharding weights.
    num_experts:
        Number of expert slots (default 64 for OLMoE).
    num_layers:
        Number of MoE layers for dense pipeline mode.
    settle_window:
        Failover settle window in seconds (default 0.3 — fast for local).
    """

    def __init__(
        self,
        *,
        num_nodes: int = 3,
        base_port: int = 24600,
        expert_store_factories: list[Callable[[int], ExpertStore]] | None = None,
        hardware_gen: str = "simulated-rk3588",
        own_storage_bandwidths: list[int] | None = None,
        num_experts: int = 64,
        num_layers: int = 40,
        settle_window: float = 0.3,
    ) -> None:
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")

        self.num_nodes = num_nodes
        self.base_port = base_port
        self.hardware_gen = hardware_gen
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.settle_window = settle_window

        # Node IDs
        self.node_ids = [
            f"sim-node-{i:02d}-aaaa-bbbb-cccc-dddddddddddd"
            for i in range(num_nodes)
        ]

        # Bandwidths
        if own_storage_bandwidths is None:
            self.bandwidths = [3200] + [280] * (num_nodes - 1)
        else:
            self.bandwidths = own_storage_bandwidths
            if len(self.bandwidths) < num_nodes:
                self.bandwidths += [280] * (num_nodes - len(self.bandwidths))

        # Expert store factories
        if expert_store_factories is None:
            # Default: node 0 gets nothing (caller must provide),
            # others get simulated stores
            self._store_factories: list[Callable[[int], ExpertStore]] = []

            def _make_simulated(idx: int) -> SimulatedExpertStore:
                bw = self.bandwidths[idx] if idx < len(self.bandwidths) else 280
                return SimulatedExpertStore(
                    expert_size_bytes=4 * 1024,  # small for sim
                    num_layers=num_layers,
                    num_experts=num_experts,
                    bandwidth_mbps=float(bw),
                    latency_ms=0.1 if bw > 1000 else 0.5,
                    seed=idx * 1000,
                )

            for i in range(num_nodes):
                self._store_factories.append(_make_simulated)
        else:
            self._store_factories = list(expert_store_factories)
            while len(self._store_factories) < num_nodes:
                idx = len(self._store_factories)
                bw = self.bandwidths[idx] if idx < len(self.bandwidths) else 280

                def _make_sim(idx: int, bandwidth: float) -> SimulatedExpertStore:
                    return SimulatedExpertStore(
                        expert_size_bytes=4 * 1024,
                        num_layers=num_layers,
                        num_experts=num_experts,
                        bandwidth_mbps=bandwidth,
                        latency_ms=0.1 if bandwidth > 1000 else 0.5,
                        seed=idx * 1000,
                    )

                self._store_factories.append(
                    lambda i=idx, b=bw: _make_sim(i, b)
                )

        # Per-node components (populated in start())
        self._expert_stores: list[ExpertStore] = []
        self._fleet_tables: list[FleetTable] = []
        self._rpc_servers: list[RpcServer] = []
        self._rpc_clients: list[RpcClient] = []
        self._failovers: list[FailoverCoordinator] = []
        self._gang_syncs: list[GangSync] = []
        self._pipelines: list[PipelineCoordinator] = []
        self._started = False

        # Background task tracking
        self._background_tasks: set[asyncio.Task[Any]] = set()

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def peers(self) -> dict[str, tuple[str, int]]:
        """Mapping from node_id → (host, port) for all nodes."""
        host = "127.0.0.1"
        return {
            nid: (host, self.base_port + i)
            for i, nid in enumerate(self.node_ids)
        }

    def get_store(self, node_index: int) -> ExpertStore:
        """Return the ExpertStore for node *node_index*."""
        return self._expert_stores[node_index]

    def set_store_factory(self, node_index: int,
                          factory: Callable[[int], ExpertStore]) -> None:
        """Replace the expert store factory for one node."""
        while len(self._store_factories) <= node_index:
            self._store_factories.append(
                lambda i=node_index: SimulatedExpertStore(
                    expert_size_bytes=4 * 1024,
                    num_layers=self.num_layers,
                    num_experts=self.num_experts,
                    bandwidth_mbps=float(self.bandwidths[node_index]),
                    seed=node_index * 1000,
                )
            )
        self._store_factories[node_index] = factory

    async def start(self) -> None:
        """Boot all nodes: stores, fleet tables, RPC, failover, gang, pipeline.

        Idempotent.
        """
        if self._started:
            return
        self._started = True

        # ── 1. Create expert stores ─────────────────────────────────
        for i in range(self.num_nodes):
            store = self._store_factories[i](i)
            self._expert_stores.append(store)

        # ── 2. Create FleetTables ───────────────────────────────────
        for nid in self.node_ids:
            ft = FleetTable(own_node_id=nid, stale_timeout=60.0)
            self._fleet_tables.append(ft)

        # ── 3. Create RpcClients ────────────────────────────────────
        for nid in self.node_ids:
            client = RpcClient(own_node_id=nid)
            self._rpc_clients.append(client)

        # ── 4. Create GangSync instances (before RpcServers since
        #       handler needs the gang) ──────────────────────────────
        all_peers = self.peers
        for i, nid in enumerate(self.node_ids):
            client = self._rpc_clients[i]
            gang = GangSync(
                own_node_id=nid,
                rpc_client=client,
                peers=all_peers,
            )
            self._gang_syncs.append(gang)

        # ── 5. Create PipelineCoordinators ──────────────────────────
        for i, nid in enumerate(self.node_ids):
            client = self._rpc_clients[i]
            pipe = PipelineCoordinator(
                own_node_id=nid,
                rpc_client=client,
                peers=all_peers,
                compute_stage=None,
                assignment=None,
            )
            self._pipelines.append(pipe)

        # ── 6. Create RpcServers (handler is gang_sync's handler) ──
        # We'll use a combined handler that dispatches to both gang and pipeline
        for i, nid in enumerate(self.node_ids):
            port = self.base_port + i
            gang = self._gang_syncs[i]
            pipe = self._pipelines[i]

            async def combined_handler(conn, frame,
                                       g=gang, p=pipe) -> None:
                from rpc import Frame, MessageType as MT
                # Try gang first (MoE mode), then pipeline (dense mode)
                # Both ignore non-matching messages
                await g.handle_frame(conn, frame)
                await p.handle_frame(conn, frame)

            server = RpcServer(
                own_node_id=nid,
                port=port,
                handler=combined_handler,
                bind_ip="127.0.0.1",
            )
            self._rpc_servers.append(server)

        # ── 7. Start RpcServers ────────────────────────────────────
        for srv in self._rpc_servers:
            await srv.start()
        await asyncio.sleep(0.05)  # let listeners bind

        # ── 8. Create FailoverCoordinators ──────────────────────────
        for i, nid in enumerate(self.node_ids):
            ft = self._fleet_tables[i]
            failover = FailoverCoordinator(
                fleet_table=ft,
                own_node_id=nid,
                num_experts=self.num_experts,
                own_storage_bandwidth_mbps=self.bandwidths[i],
                settle_window=self.settle_window,
            )
            self._failovers.append(failover)

        # ── 9. Start failovers ─────────────────────────────────────
        for fc in self._failovers:
            await fc.start()

        # ── 10. Populate FleetTables (direct injection, no UDP) ────
        host = "127.0.0.1"
        for i, nid in enumerate(self.node_ids):
            for j, other_nid in enumerate(self.node_ids):
                if i == j:
                    continue
                desc = NodeDescriptor(
                    node_id=other_nid,
                    hostname=f"sim-{j:02d}",
                    ip=host,
                    port=self.base_port + j,
                    ram_total_mb=8192,
                    ram_available_mb=6000,
                    storage_bandwidth_mbps=self.bandwidths[j],
                    hardware_gen=self.hardware_gen,
                    load=0.1,
                    uptime_seconds=1.0,
                    timestamp=time.time(),
                )
                is_new = await self._fleet_tables[i].update(desc)
                if is_new:
                    # Fire join manually (real FleetTable doesn't auto-fire)
                    await self._fleet_tables[i]._fire_join(desc)

        # ── 11. Start fleet eviction sweeps ────────────────────────
        for ft in self._fleet_tables:
            await ft.start_eviction_sweep(interval=10.0)  # long — no real eviction

        # ── 12. Wait for settle + first assignment ─────────────────
        await asyncio.sleep(self.settle_window + 0.2)

        # ── 13. Update PipelineCoordinators with assignments ───────
        for i in range(self.num_nodes):
            assignment = self._failovers[i].current_assignment()
            if assignment is not None:
                self._pipelines[i].set_assignment(assignment)

        logger.info(
            "SimulatedFleet: %d nodes ready on ports %d–%d",
            self.num_nodes, self.base_port,
            self.base_port + self.num_nodes - 1,
        )

    async def stop(self) -> None:
        """Shut down all nodes gracefully."""
        self._started = False

        # Close RPC clients first (pooled connections)
        for client in self._rpc_clients:
            with _contextlib_suppress(Exception):
                await client.close()

        # Stop RPC servers
        for srv in self._rpc_servers:
            with _contextlib_suppress(Exception):
                await srv.stop()

        # Stop fleet eviction
        for ft in self._fleet_tables:
            with _contextlib_suppress(Exception):
                await ft.stop()

        # Close expert stores
        for store in self._expert_stores:
            with _contextlib_suppress(Exception):
                await store.close()

        # Clear all lists
        self._expert_stores.clear()
        self._rpc_servers.clear()
        self._rpc_clients.clear()
        self._failovers.clear()
        self._gang_syncs.clear()
        self._pipelines.clear()
        self._fleet_tables.clear()

        logger.info("SimulatedFleet: all nodes stopped")

    def get_assignment(self, node_index: int = 0) -> ShardAssignment | None:
        """Return the current shard assignment for node *node_index*."""
        if 0 <= node_index < len(self._failovers):
            return self._failovers[node_index].current_assignment()
        return None

    def get_pipeline(self, node_index: int = 0) -> PipelineCoordinator:
        """Return the PipelineCoordinator for node *node_index*."""
        return self._pipelines[node_index]

    def get_gang_sync(self, node_index: int = 0) -> GangSync:
        """Return the GangSync for node *node_index*."""
        return self._gang_syncs[node_index]

    def get_failover(self, node_index: int = 0) -> FailoverCoordinator:
        """Return the FailoverCoordinator for node *node_index*."""
        return self._failovers[node_index]

    def get_rpc_client(self, node_index: int = 0) -> RpcClient:
        """Return the RpcClient for node *node_index*."""
        return self._rpc_clients[node_index]


# ── Self-test ─────────────────────────────────────────────────────────────


async def _self_test() -> None:
    """Boot a 3-node simulated fleet and verify it converges."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("── simulate_fleet self-test ──")

    fleet = SimulatedFleet(
        num_nodes=3,
        base_port=24700,
        num_experts=64,
        num_layers=40,
        settle_window=0.3,
    )

    await fleet.start()

    # Verify convergence
    assignment = fleet.get_assignment(0)
    assert assignment is not None, "No assignment after start"
    print(f"  Fleet converged: {len(assignment.node_counts)} nodes")
    print(f"  Fleet hash: {assignment.fleet_hash[:16]}…")

    # Verify all nodes have the same assignment
    for i in range(1, 3):
        a = fleet.get_assignment(i)
        assert a is not None, f"Node {i} has no assignment"
        assert a.fleet_hash == assignment.fleet_hash, (
            f"Node {i} hash mismatch"
        )
    print("  All 3 nodes agree on shard assignment ✓")

    # Verify pipeline order
    pipe = fleet.get_pipeline(0)
    pipe_order = pipe._get_pipeline_order(assignment)
    print(f"  Pipeline order: {len(pipe_order)} nodes")
    assert len(pipe_order) == 3, f"Expected 3 nodes in pipeline, got {len(pipe_order)}"

    await fleet.stop()
    print("── simulate_fleet self-test passed ──")


if __name__ == "__main__":
    asyncio.run(_self_test())
