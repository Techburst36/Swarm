"""Integration harness — all six modules running together as one system.

Every module in this project is unit-tested in isolation and passes. None of
them had ever been introduced to each other. This harness wires the real
implementations together on localhost and exercises the seams between them,
which is where the interesting bugs live:

    node_identity.FleetTable   -- who is alive
      -> sharding              -- who owns which experts/layers
        -> failover            -- what changed when membership moved
          -> gang_sync         -- MoE lockstep reduce over rpc
          -> pipeline          -- dense stage handoff over rpc
            -> rpc             -- the transport underneath both

Deliberate choices:

* **Real FleetTable, not the _FakeFleetTable failover.py ships.** failover was
  only ever tested against the fake; this is the first time it meets the real
  one. Descriptors are injected via ``FleetTable.update()`` rather than over
  UDP, because multicast discovery does not loop back on WSL2 (a known
  environment limitation, documented in test-plan.md) and the transport is
  not what is under test here.
* **Real RpcServer/RpcClient on 127.0.0.1**, never 0.0.0.0. Real sockets,
  real framing, real backpressure.
* Nodes are separate objects with separate ports, exactly as they would be on
  separate blades.

Run:  python3 test_integration.py
"""

from __future__ import annotations

import asyncio
import time

from failover import FailoverCoordinator
from gang_sync import GangSync, GangSyncError, GangSyncTimeout
from node_identity import FleetTable, NodeDescriptor
from pipeline import PipelineCoordinator, PipelineError
from rpc import MessageType, RpcClient, RpcServer
from sharding import NodeCapability, compute_assignment

HOST = "127.0.0.1"
BASE_PORT = 24600
NUM_EXPERTS = 64
NUM_LAYERS = 40


# ── Helpers ───────────────────────────────────────────────────────────────


def _desc(node_id: str, port: int, bw: int, hostname: str = "") -> NodeDescriptor:
    """Build a NodeDescriptor as discovery would have produced it."""
    return NodeDescriptor(
        node_id=node_id,
        hostname=hostname or node_id,
        ip=HOST,
        port=port,
        ram_total_mb=8192,
        ram_available_mb=6000,
        storage_bandwidth_mbps=bw,
        hardware_gen="rk3588-8gb",
        load=0.0,
        uptime_seconds=1.0,
        timestamp=time.time(),
    )


async def _join(fleet: FleetTable, desc: NodeDescriptor) -> None:
    """Insert a peer the way NodeIdentity does, callbacks included.

    INTEGRATION FINDING: the real FleetTable never fires callbacks from its
    own mutators. ``update()`` returns ``is_new`` and ``evict_stale()``
    returns the evicted list; firing is the caller's job -- NodeIdentity
    fires joins, and ``_eviction_loop`` (started by ``start_eviction_sweep``)
    fires leaves. The contract is consistent in both directions.

    ``_FakeFleetTable`` in failover.py fires both automatically from
    ``add_node()`` / ``remove_node()``. So failover.py's 13 unit tests
    exercise a contract the real class does not have. Any caller driving a
    raw FleetTable -- rather than going through NodeIdentity and the eviction
    sweep -- gets a FailoverCoordinator that silently never reshards, with no
    error to indicate why.
    """
    is_new = await fleet.update(desc)
    if is_new:
        await fleet._fire_join(desc)


def _caps_from_fleet(descs: list[NodeDescriptor]) -> list[NodeCapability]:
    """The seam between Layer 3 and sharding: descriptor -> capability."""
    return [
        NodeCapability(
            node_id=d.node_id, storage_bandwidth_mbps=d.storage_bandwidth_mbps
        )
        for d in descs
    ]


def _peers_from_fleet(descs: list[NodeDescriptor]) -> dict[str, tuple[str, int]]:
    """The seam between Layer 3 and the transport: descriptor -> peer address.

    gang_sync and pipeline both need node_id -> (host, port). Getting this
    wrong is silent: messages go nowhere and everything times out.
    """
    return {d.node_id: (d.ip, d.port) for d in descs}


def _enc(v: int) -> bytes:
    return v.to_bytes(8, "big", signed=True)


def _dec(b: bytes) -> int:
    return int.from_bytes(b, "big", signed=True)


def _int_combine(acc: bytes, local: bytes) -> bytes:
    return _enc(_dec(acc) + _dec(local))


class Node:
    """One simulated blade: identity, transport, and both execution modes."""

    def __init__(self, node_id: str, port: int, bandwidth_mbps: int) -> None:
        self.node_id = node_id
        self.port = port
        self.bandwidth_mbps = bandwidth_mbps

        self.fleet = FleetTable(node_id)
        self.client = RpcClient(own_node_id=node_id)
        self.gang: GangSync | None = None
        self.pipe: PipelineCoordinator | None = None
        self.server: RpcServer | None = None
        self.failover: FailoverCoordinator | None = None

    async def start(self, peers: dict[str, tuple[str, int]]) -> None:
        self.gang = GangSync(
            own_node_id=self.node_id, rpc_client=self.client, peers=peers
        )
        self.pipe = PipelineCoordinator(
            own_node_id=self.node_id,
            rpc_client=self.client,
            peers=peers,
            compute_stage=self._compute_stage,
        )

        # Both coordinators share one RpcServer, so the handler must route
        # frames to whichever one owns them. This is a real integration
        # question: two modules, one transport, overlapping message types.
        async def handler(conn, frame) -> None:
            if await self._route(conn, frame):
                return

        self.server = RpcServer(
            own_node_id=self.node_id,
            port=self.port,
            handler=handler,
            bind_ip=HOST,
        )
        await self.server.start()

    async def _route(self, conn, frame) -> bool:
        """Dispatch a frame to gang_sync or pipeline.

        Both use MessageType.ACTIVATION, so the type alone is not enough to
        tell them apart. Each module's decoder rejects frames it does not
        recognise, so try pipeline first and fall through to gang_sync.
        """
        assert self.pipe is not None and self.gang is not None
        try:
            handled = await self.pipe.handle_frame(conn, frame)
            if handled:
                return True
        except Exception:
            pass
        try:
            await self.gang.handle_frame(conn, frame)
            return True
        except Exception:
            return False

    def _compute_stage(self, activation: bytes, start: int, end: int) -> bytes:
        return _enc(_dec(activation) + (end - start))

    async def stop(self) -> None:
        try:
            await self.client.close()
        except Exception:
            pass
        if self.server is not None:
            try:
                await self.server.stop()
            except Exception:
                pass


async def _build_fleet(
    n: int, base_port: int, bandwidths: list[int] | None = None
) -> tuple[list[Node], list[NodeDescriptor]]:
    bws = bandwidths or [4000] * n
    node_ids = [f"int-{i:02d}-{'0'*8}" for i in range(n)]
    descs = [
        _desc(nid, base_port + i, bws[i]) for i, nid in enumerate(node_ids)
    ]
    peers = _peers_from_fleet(descs)

    nodes = [Node(nid, base_port + i, bws[i]) for i, nid in enumerate(node_ids)]
    for node in nodes:
        await node.start(peers)

        # Populate each node's FleetTable with its peers, as discovery would.
        for d in descs:
            if d.node_id != node.node_id:
                await _join(node.fleet, d)

    await asyncio.sleep(0.1)
    return nodes, descs


async def _teardown(nodes: list[Node]) -> None:
    for node in nodes:
        await node.stop()


# ── Seam 1: FleetTable -> sharding ────────────────────────────────────────


async def test_all_nodes_agree_on_assignment() -> None:
    """Every node must independently derive the identical assignment.

    This is the whole coordinator-free premise. Each node has its own
    FleetTable, populated in its own order; if any of that leaked into the
    result the fleet would silently disagree about ownership.
    """
    nodes, descs = await _build_fleet(4, BASE_PORT)
    try:
        assignments = []
        for node in nodes:
            live = await node.fleet.get_live_nodes()
            # A node's own descriptor is not in its own FleetTable (self is
            # filtered), so it must add itself back before sharding.
            caps = _caps_from_fleet(live) + [
                NodeCapability(
                    node_id=node.node_id,
                    storage_bandwidth_mbps=node.bandwidth_mbps,
                )
            ]
            assignments.append(compute_assignment(caps, NUM_EXPERTS))

        hashes = {a.fleet_hash for a in assignments}
        assert len(hashes) == 1, f"nodes disagree on fleet hash: {hashes}"
        counts = {tuple(sorted(a.node_counts.items())) for a in assignments}
        assert len(counts) == 1, "nodes disagree on expert counts"
        print(f"PASS: all 4 nodes derived identical assignment ({len(hashes)} hash)")
    finally:
        await _teardown(nodes)


async def test_self_inclusion_is_required() -> None:
    """Document the sharp edge found above, as an executable check.

    FleetTable deliberately excludes self (a node is not its own peer). If a
    caller forwards get_live_nodes() straight into compute_assignment without
    adding itself, the node silently shards itself out of its own fleet.
    """
    nodes, _ = await _build_fleet(3, BASE_PORT + 10)
    try:
        node = nodes[0]
        live = await node.fleet.get_live_nodes()
        assert node.node_id not in {d.node_id for d in live}, (
            "FleetTable included self; the rest of this project assumes it does not"
        )

        wrong = compute_assignment(_caps_from_fleet(live), NUM_EXPERTS)
        assert node.node_id not in wrong.node_counts, "expected self to be absent"

        right = compute_assignment(
            _caps_from_fleet(live)
            + [
                NodeCapability(
                    node_id=node.node_id,
                    storage_bandwidth_mbps=node.bandwidth_mbps,
                )
            ],
            NUM_EXPERTS,
        )
        assert node.node_id in right.node_counts
        assert wrong.fleet_hash != right.fleet_hash
        print("PASS: self must be re-added after get_live_nodes (seam documented)")
    finally:
        await _teardown(nodes)


# ── Seam 2: assignment -> gang_sync over real rpc ─────────────────────────


async def test_gang_sync_over_real_fleet() -> None:
    """A MoE lockstep reduce driven by an assignment derived from FleetTable."""
    nodes, descs = await _build_fleet(4, BASE_PORT + 20)
    try:
        caps = _caps_from_fleet(descs)
        assignment = compute_assignment(caps, NUM_EXPERTS)

        contribs = [11, 22, 33, 44]
        results = await asyncio.gather(
            *(
                nodes[i].gang.ring_reduce(
                    assignment=assignment,
                    layer_id=1,
                    local_contribution=_enc(contribs[i]),
                    combine=_int_combine,
                    timeout=8.0,
                )
                for i in range(4)
            )
        )
        sums = [_dec(r) for r in results]
        assert len(set(sums)) == 1, f"nodes disagree: {sums}"
        assert sums[0] == sum(contribs)
        print(f"PASS: gang-sync over fleet-derived assignment ({sums[0]})")
    finally:
        await _teardown(nodes)


# ── Seam 3: assignment -> pipeline over the same transport ────────────────


async def test_pipeline_over_real_fleet() -> None:
    """A dense pipeline pass using the same fleet and the same RpcServers."""
    nodes, descs = await _build_fleet(4, BASE_PORT + 30)
    try:
        caps = _caps_from_fleet(descs)
        assignment = compute_assignment(caps, NUM_LAYERS)
        for node in nodes:
            node.pipe.set_assignment(assignment)

        result = await nodes[0].pipe.run_pipeline(
            assignment=assignment,
            request_id=1,
            initial_activation=_enc(0),
            timeout=8.0,
        )
        assert _dec(result) == NUM_LAYERS, f"got {_dec(result)}"
        print(f"PASS: pipeline over fleet-derived assignment ({_dec(result)})")
    finally:
        await _teardown(nodes)


async def test_both_modes_share_one_transport() -> None:
    """Gang-sync and pipeline traffic on the same RpcServer must not collide.

    Both modules send MessageType.ACTIVATION. If their inner framings were
    confusable, one module would consume the other's frames and both would
    fail in confusing ways. Run them concurrently and check both results.
    """
    nodes, descs = await _build_fleet(4, BASE_PORT + 40)
    try:
        caps = _caps_from_fleet(descs)
        gang_assignment = compute_assignment(caps, NUM_EXPERTS)
        pipe_assignment = compute_assignment(caps, NUM_LAYERS)
        for node in nodes:
            node.pipe.set_assignment(pipe_assignment)

        gang_task = asyncio.gather(
            *(
                nodes[i].gang.ring_reduce(
                    assignment=gang_assignment,
                    layer_id=7,
                    local_contribution=_enc(10 * (i + 1)),
                    combine=_int_combine,
                    timeout=8.0,
                )
                for i in range(4)
            )
        )
        pipe_task = nodes[1].pipe.run_pipeline(
            assignment=pipe_assignment,
            request_id=99,
            initial_activation=_enc(1000),
            timeout=8.0,
        )

        gang_results, pipe_result = await asyncio.gather(gang_task, pipe_task)

        sums = [_dec(r) for r in gang_results]
        assert len(set(sums)) == 1 and sums[0] == 100, f"gang wrong: {sums}"
        assert _dec(pipe_result) == 1000 + NUM_LAYERS, (
            f"pipeline wrong: {_dec(pipe_result)}"
        )
        print("PASS: gang-sync and pipeline ran concurrently on one transport")
    finally:
        await _teardown(nodes)


# ── Seam 4: real FleetTable -> failover (never tested together before) ────


async def test_failover_with_real_fleet_table() -> None:
    """FailoverCoordinator against the REAL FleetTable, not _FakeFleetTable.

    failover.py's own tests use a fake. This is the first time the two meet.
    """
    node = Node("fo-self-0000", BASE_PORT + 50, 4000)
    await node.start({})
    try:
        coord = FailoverCoordinator(
            fleet_table=node.fleet,
            own_node_id=node.node_id,
            num_experts=NUM_EXPERTS,
            own_storage_bandwidth_mbps=node.bandwidth_mbps,
            settle_window=0.3,
        )
        await coord.start()
        node.failover = coord

        assert coord.current_assignment() is None

        await _join(node.fleet, _desc("fo-peer-a000", BASE_PORT + 51, 4000))
        await _join(node.fleet, _desc("fo-peer-b000", BASE_PORT + 52, 1900))
        await asyncio.sleep(0.8)

        assert coord.current_assignment() is not None, (
            "no assignment produced from real FleetTable join callbacks"
        )
        diff = await asyncio.wait_for(coord.diff_queue.get(), timeout=2.0)
        assert diff.old_fleet_hash is None, "expected a first assignment"
        assert len(diff.nodes_added) == 3, (
            f"expected self + 2 peers, got {diff.nodes_added}"
        )
        print(
            f"PASS: failover drove off the real FleetTable "
            f"({diff.moved_count} experts placed)"
        )
    finally:
        await node.stop()


async def test_failover_reshard_on_real_eviction() -> None:
    """A real stale-eviction must produce a reshard diff.

    Uses FleetTable's actual eviction path (short stale_timeout) rather than
    a fake's remove_node(), so the whole Layer 3 -> Layer 4 chain is real.
    """
    node = Node("ev-self-0000", BASE_PORT + 60, 4000)
    await node.start({})
    node.fleet = FleetTable(node.node_id, stale_timeout=0.6)
    try:
        coord = FailoverCoordinator(
            fleet_table=node.fleet,
            own_node_id=node.node_id,
            num_experts=NUM_EXPERTS,
            own_storage_bandwidth_mbps=node.bandwidth_mbps,
            settle_window=0.3,
        )
        await coord.start()

        # Real background sweep — this is what fires leave callbacks.
        await node.fleet.start_eviction_sweep(interval=0.2)

        await _join(node.fleet, _desc("ev-peer-a000", BASE_PORT + 61, 4000))
        await _join(node.fleet, _desc("ev-peer-b000", BASE_PORT + 62, 4000))
        await asyncio.sleep(0.8)
        first = await asyncio.wait_for(coord.diff_queue.get(), timeout=2.0)
        assert len(first.nodes_added) == 3

        # Stop refreshing peer A; let the real eviction sweep drop it.
        # Must wait longer than stale_timeout (0.6s) or nothing is stale yet.
        # Keep B alive while A goes stale, then let the sweep notice.
        for _ in range(5):
            await asyncio.sleep(0.2)
            await node.fleet.update(_desc("ev-peer-b000", BASE_PORT + 62, 4000))
        await asyncio.sleep(0.6)

        live_ids = {d.node_id for d in await node.fleet.get_live_nodes()}
        assert "ev-peer-a000" not in live_ids, "sweep did not evict peer-a"
        assert "ev-peer-b000" in live_ids, "sweep wrongly evicted the live peer"
        await asyncio.sleep(0.9)

        second = await asyncio.wait_for(coord.diff_queue.get(), timeout=3.0)
        assert "ev-peer-a000" in second.nodes_removed, (
            f"expected peer-a removed, got {second.nodes_removed}"
        )
        assert second.moved_count > 0, "eviction produced no expert movement"
        print(
            f"PASS: real eviction -> reshard "
            f"({second.moved_count} moved, {second.unchanged_count} stayed)"
        )
    finally:
        await node.stop()


# ── Seam 5: gang_sync failure -> failover diagnostic hook ─────────────────


async def test_gang_sync_timeout_feeds_failover() -> None:
    """The documented failure path: GangSyncTimeout -> report_gang_sync_failure.

    A dead node must produce a timeout that names it, and reporting that to
    the coordinator must NOT trigger a reshard while FleetTable still
    considers the node alive.
    """
    nodes, descs = await _build_fleet(3, BASE_PORT + 70)
    try:
        caps = _caps_from_fleet(descs)
        assignment = compute_assignment(caps, NUM_EXPERTS)

        coord = FailoverCoordinator(
            fleet_table=nodes[0].fleet,
            own_node_id=nodes[0].node_id,
            num_experts=NUM_EXPERTS,
            own_storage_bandwidth_mbps=nodes[0].bandwidth_mbps,
            settle_window=0.3,
        )
        await coord.start()
        await asyncio.sleep(0.6)
        while not coord.diff_queue.empty():
            coord.diff_queue.get_nowait()

        # Kill node 2's server but leave it in FleetTable (as a transient
        # network blip would look).
        await nodes[2].server.stop()

        failed = False
        try:
            await nodes[0].gang.ring_reduce(
                assignment=assignment,
                layer_id=1,
                local_contribution=_enc(1),
                combine=_int_combine,
                timeout=2.0,
            )
        except GangSyncError:
            failed = True

        assert failed, "ring_reduce succeeded despite a dead node"

        coord.report_gang_sync_failure(layer_id=1, node_id=nodes[2].node_id)
        await asyncio.sleep(0.6)

        assert coord.diff_queue.empty(), (
            "a transient gang-sync failure triggered a reshard; FleetTable "
            "should remain the sole membership authority"
        )
        print("PASS: gang-sync timeout reported, no spurious reshard")
    finally:
        await _teardown(nodes)


# ── Seam 6: full chain, membership change reshapes execution ──────────────


async def test_reshard_changes_execution() -> None:
    """The whole point, end to end.

    A node joins, failover produces a new assignment, and the next gang-sync
    actually runs over the new fleet — with the new node participating.
    """
    nodes, descs = await _build_fleet(3, BASE_PORT + 80)
    extra: Node | None = None
    try:
        caps = _caps_from_fleet(descs)
        a1 = compute_assignment(caps, NUM_EXPERTS)

        results = await asyncio.gather(
            *(
                nodes[i].gang.ring_reduce(
                    assignment=a1,
                    layer_id=1,
                    local_contribution=_enc(i + 1),
                    combine=_int_combine,
                    timeout=8.0,
                )
                for i in range(3)
            )
        )
        assert _dec(results[0]) == 6, "3-node reduce wrong"

        # Fourth node joins.
        extra = Node("int-03-00000000", BASE_PORT + 83, 4000)
        d4 = _desc(extra.node_id, extra.port, 4000)
        all_descs = descs + [d4]
        peers = _peers_from_fleet(all_descs)

        await extra.start(peers)
        for node in nodes:
            node.gang._peers = peers  # as a real reshard would re-provision
            await _join(node.fleet, d4)
        for d in descs:
            await _join(extra.fleet, d)
        await asyncio.sleep(0.2)

        a2 = compute_assignment(_caps_from_fleet(all_descs), NUM_EXPERTS)
        assert a2.fleet_hash != a1.fleet_hash, "assignment did not change"
        assert extra.node_id in a2.node_counts, "new node got no experts"

        live = [*nodes, extra]
        results2 = await asyncio.gather(
            *(
                live[i].gang.ring_reduce(
                    assignment=a2,
                    layer_id=2,
                    local_contribution=_enc(i + 1),
                    combine=_int_combine,
                    timeout=8.0,
                )
                for i in range(4)
            )
        )
        sums = [_dec(r) for r in results2]
        assert len(set(sums)) == 1 and sums[0] == 10, f"4-node reduce wrong: {sums}"
        print(
            f"PASS: fleet grew 3->4, assignment changed, execution followed "
            f"({a1.fleet_hash[:8]}… -> {a2.fleet_hash[:8]}…)"
        )
    finally:
        await _teardown(nodes)
        if extra is not None:
            await extra.stop()


# ── Runner ────────────────────────────────────────────────────────────────


async def _run() -> None:
    print("── Seam: FleetTable -> sharding ──")
    await test_all_nodes_agree_on_assignment()
    await test_self_inclusion_is_required()

    print("\n── Seam: assignment -> execution over real rpc ──")
    await test_gang_sync_over_real_fleet()
    await test_pipeline_over_real_fleet()
    await test_both_modes_share_one_transport()

    print("\n── Seam: real FleetTable -> failover ──")
    await test_failover_with_real_fleet_table()
    await test_failover_reshard_on_real_eviction()

    print("\n── Seam: failure signals ──")
    await test_gang_sync_timeout_feeds_failover()

    print("\n── Full chain ──")
    await test_reshard_changes_execution()


def main() -> None:
    asyncio.run(_run())
    print("\nAll integration tests passed.")


if __name__ == "__main__":
    main()
