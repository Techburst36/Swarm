#!/usr/bin/env python3
"""
node_identity.py — Zero-coordinator peer discovery for the Swarm distributed
inference fleet.

Design constraints
------------------
**No fixed roles.** There is no "leader," "coordinator," "primary," or "worker"
encoded in this module.  Every node broadcasts identically, listens identically,
and maintains a symmetrical view of the fleet.  Role assignment (scheduling,
expert sharding, pipeline staging) lives in Layer 4 (the distributed scheduler);
this module must not reach ahead and build any of that logic here.

**Standard library only.**  Runs on a minimal ARM Linux image with nothing
beyond Python 3.11+.  No pip, no venv, no third-party packages.

**Discovery by UDP broadcast** on the LAN, not a central registry, not mDNS /
Bonjour, not a coordinator node.  A multicast fallback mode (``--multicast-group``)
lets you run multiple instances on a single machine for testing.

**Thread-safe via asyncio.**  All I/O is async.  `FleetTable` is protected by
an `asyncio.Lock`.  No raw `threading` module is used anywhere.

Architecture
------------
Three classes, layered bottom-up:

1. **`NodeDescriptor`** — an immutable snapshot of a node's capabilities at a
   point in time.  Serialises to/from compact JSON over UDP.

2. **`FleetTable`** — an async-safe container mapping ``node_id → (descriptor,
   last_seen)``.  Receives descriptors from the network, evicts stale nodes on
   a background sweep, and fires join/leave callbacks.

3. **`NodeIdentity`** — the main entry point for a single node.  Owns this
   node's persistent identity, runs the broadcast loop (periodic UDP sends)
   and the discovery listener (incoming UDP datagrams), and exposes a
   ``.fleet`` attribute for higher layers to consume.

Usage (production)
------------------
.. code-block:: python

    import asyncio
    from node_identity import NodeIdentity

    async def main():
        node = NodeIdentity(
            port=9877,                  # TCP port for inter-node data (advertised only)
            discovery_port=9876,        # UDP port for discovery
            hardware_gen="rk3588-8gb",  # free-form hardware tag
            storage_bandwidth_mbps=4000,
        )
        node.fleet.on_join(lambda desc: print(f"JOIN:  {desc.node_id[:8]}… {desc.hostname}"))
        node.fleet.on_leave(lambda nid, desc: print(f"LEAVE: {nid[:8]}…"))

        await node.start()
        try:
            await asyncio.Event().wait()  # run forever
        finally:
            await node.stop()

    asyncio.run(main())

Usage (local testing with multicast)
------------------------------------
Run three terminals on one machine::

    python node_identity.py --port 9001 --node-name alpha   --multicast-group 239.255.42.99
    python node_identity.py --port 9002 --node-name beta    --multicast-group 239.255.42.99
    python node_identity.py --port 9003 --node-name gamma   --multicast-group 239.255.42.99

IPv4-only.  IPv6 is not supported — the broadcast address 255.255.255.255
and multicast groups like 239.x.x.x are IPv4 concepts.  If you need IPv6,
this module needs a redesign (likely link-local multicast on ff02::/16).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import platform
import socket
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_DISCOVERY_PORT: int = 9876
DEFAULT_MULTICAST_GROUP: str = "239.255.42.99"
DEFAULT_BROADCAST_INTERVAL: float = 2.0
DEFAULT_STALE_TIMEOUT: float = 8.0  # ~4 missed broadcasts
DEFAULT_EVICTION_SWEEP_INTERVAL: float = 2.0
NODE_ID_DIR: Path = Path.home() / ".swarm"
NODE_ID_FILE: str = "node_id.json"

DESCRIPTOR_JSON_VERSION: int = 1

logger = logging.getLogger("swarm.node_identity")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _detect_ip() -> str:
    """Return the best-guess LAN IPv4 address of this host.

    Strategy (in order):
      1. UDP-"connect" to a dummy address — the OS picks the default-route
         source address without sending any packets.
      2. ``gethostname()`` → ``gethostbyname()``.
      3. ``"127.0.0.1"`` with a warning logged.

    Callers can override via ``NodeIdentity(ip=...)`` if auto-detection picks
    the wrong interface.
    """
    # Strategy 1: default-route source address
    for dummy in ("1.1.1.1", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((dummy, 1))
                ip = s.getsockname()[0]
                if ip and ip != "127.0.0.1":
                    return ip
        except OSError:
            continue

    # Strategy 2: gethostname → gethostbyname
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip != "127.0.0.1":
            return ip
    except OSError:
        pass

    # Strategy 3: surrender
    logger.warning(
        "Could not auto-detect LAN IP — defaulting to 127.0.0.1. "
        "Set `ip=` on NodeIdentity to override."
    )
    return "127.0.0.1"


def _get_ram_info() -> tuple[int, int]:
    """Return ``(total_mb, available_mb)`` from the OS.

    Reads ``/proc/meminfo`` on Linux; returns ``(0, 0)`` on other platforms.
    """
    try:
        with open("/proc/meminfo", "r") as fh:
            meminfo = fh.read()
    except OSError:
        return (0, 0)

    total = 0
    available = 0
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            total = int(parts[1]) // 1024  # kB → MB
        elif line.startswith("MemAvailable:"):
            parts = line.split()
            available = int(parts[1]) // 1024  # kB → MB
    return (total, available)


def _load_or_create_node_id(path: Path | None = None) -> str:
    """Return the persisted node UUID, creating one if it doesn't exist.

    The file is a tiny JSON blob: ``{"node_id": "..."}``.
    If the file is corrupted or unreadable, a new UUID is generated and the
    file is overwritten.
    """
    filepath = (path or NODE_ID_DIR) / NODE_ID_FILE
    try:
        if filepath.exists():
            data = json.loads(filepath.read_text())
            nid = data.get("node_id", "")
            if nid:
                # Basic validation: must look like a UUID
                uuid.UUID(nid)
                return nid
    except (json.JSONDecodeError, ValueError, OSError):
        logger.warning("Corrupted node_id file at %s — regenerating.", filepath)

    # Create new
    nid = str(uuid.uuid4())
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps({"node_id": nid}))
    logger.info("Created new node_id: %s", nid)
    return nid


# ── NodeDescriptor ─────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class NodeDescriptor:
    """Immutable snapshot of a node's capabilities at a moment in time.

    Serialised as compact JSON for UDP transport.  The ``v`` field is a
    schema version for forward compatibility — consumers should check it
    before interpreting unknown fields.
    """

    node_id: str
    hostname: str
    ip: str
    port: int
    ram_total_mb: int
    ram_available_mb: int
    storage_bandwidth_mbps: int
    hardware_gen: str
    load: float
    uptime_seconds: float
    timestamp: float

    def to_json(self) -> str:
        """Serialise to a compact JSON string for UDP broadcast."""
        return json.dumps(
            {
                "v": DESCRIPTOR_JSON_VERSION,
                "node_id": self.node_id,
                "hostname": self.hostname,
                "ip": self.ip,
                "port": self.port,
                "ram_total_mb": self.ram_total_mb,
                "ram_available_mb": self.ram_available_mb,
                "storage_bandwidth_mbps": self.storage_bandwidth_mbps,
                "hardware_gen": self.hardware_gen,
                "load": self.load,
                "uptime_seconds": self.uptime_seconds,
                "ts": self.timestamp,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, data: str) -> NodeDescriptor | None:
        """Parse a JSON descriptor string.

        Returns ``None`` on malformed input, missing required fields, or
        an unknown/unexpected schema version.  The caller should silently
        drop the datagram — this is a noisy UDP environment and bad packets
        are expected.
        """
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return None

        if not isinstance(obj, dict):
            return None

        version = obj.get("v")
        if version != DESCRIPTOR_JSON_VERSION:
            # Future: could handle migration here.  For now, reject.
            return None

        try:
            return cls(
                node_id=obj["node_id"],
                hostname=obj["hostname"],
                ip=obj["ip"],
                port=obj["port"],
                ram_total_mb=obj["ram_total_mb"],
                ram_available_mb=obj["ram_available_mb"],
                storage_bandwidth_mbps=obj["storage_bandwidth_mbps"],
                hardware_gen=obj["hardware_gen"],
                load=obj["load"],
                uptime_seconds=obj["uptime_seconds"],
                timestamp=obj["ts"],
            )
        except (KeyError, TypeError, ValueError):
            return None


# ── FleetTable ─────────────────────────────────────────────────────────────────


class FleetTable:
    """Async-safe container for the live view of peer nodes.

    Maintained purely from received UDP broadcasts.  A node never adds itself
    to the fleet — ``own_node_id`` is used to filter out the node's own
    packets.

    A background eviction sweep (``start_eviction_sweep``) periodically
    removes nodes not heard from within ``stale_timeout`` seconds and fires
    ``on_leave`` callbacks.

    Callbacks are async (or sync — both accepted) and are called *outside*
    the internal lock so they can safely interact with the table.

    Parameters
    ----------
    own_node_id:
        The stable UUID of *this* node.  Any descriptor with a matching
        ``node_id`` is silently dropped — a node should not appear in its
        own peer table.
    stale_timeout:
        Seconds before a node is considered dead (default 8.0).
    """

    def __init__(self, own_node_id: str, stale_timeout: float = DEFAULT_STALE_TIMEOUT) -> None:
        self._own_node_id = own_node_id
        self._stale_timeout = stale_timeout
        self._lock = asyncio.Lock()
        # node_id → (NodeDescriptor, last_seen_monotonic)
        self._nodes: dict[str, tuple[NodeDescriptor, float]] = {}
        self._eviction_task: asyncio.Task[None] | None = None
        self._join_callbacks: list[Callable[[NodeDescriptor], Awaitable[None] | None]] = []
        self._leave_callbacks: list[Callable[[str, NodeDescriptor], Awaitable[None] | None]] = []

    # ── Public API ────────────────────────────────────────────────────────

    async def update(self, descriptor: NodeDescriptor) -> bool:
        """Insert or refresh a peer descriptor.

        Returns ``True`` if this is a **new** node (join event), ``False`` if
        it is a refresh of an already-known node.  The caller should fire a
        join callback when ``True`` is returned.

        Descriptors matching ``own_node_id`` are silently discarded.
        """
        if descriptor.node_id == self._own_node_id:
            return False

        now = time.monotonic()
        async with self._lock:
            is_new = descriptor.node_id not in self._nodes
            self._nodes[descriptor.node_id] = (descriptor, now)
            return is_new

    async def get_live_nodes(self) -> list[NodeDescriptor]:
        """Return all currently-known peer descriptors.

        The caller gets a snapshot — the list is a copy and safe to iterate
        without holding the lock.
        """
        async with self._lock:
            return [desc for desc, _ in self._nodes.values()]

    async def evict_stale(self) -> list[tuple[str, NodeDescriptor]]:
        """Remove nodes not seen within ``stale_timeout``.

        Returns a list of ``(node_id, last_descriptor)`` for each evicted
        node, so the caller can fire leave callbacks.
        """
        cutoff = time.monotonic() - self._stale_timeout
        evicted: list[tuple[str, NodeDescriptor]] = []
        async with self._lock:
            stale_ids = [
                nid
                for nid, (_, last_seen) in self._nodes.items()
                if last_seen < cutoff
            ]
            for nid in stale_ids:
                desc, _ = self._nodes.pop(nid)
                evicted.append((nid, desc))
        return evicted

    # ── Background eviction ───────────────────────────────────────────────

    async def start_eviction_sweep(
        self, interval: float = DEFAULT_EVICTION_SWEEP_INTERVAL
    ) -> None:
        """Start a background asyncio task that periodically evicts stale nodes.

        Safe to call multiple times — only one sweep runs at a time.

        Parameters
        ----------
        interval:
            Seconds between eviction checks (default 2.0).
        """
        if self._eviction_task is not None and not self._eviction_task.done():
            return
        self._eviction_task = asyncio.create_task(self._eviction_loop(interval))

    async def stop(self) -> None:
        """Cancel the eviction sweep task (if running)."""
        if self._eviction_task is not None:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
            self._eviction_task = None

    async def _eviction_loop(self, interval: float) -> None:
        """Internal: periodic stale-node eviction."""
        while True:
            await asyncio.sleep(interval)
            evicted = await self.evict_stale()
            for node_id, desc in evicted:
                await self._fire_leave(node_id, desc)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def on_join(
        self, callback: Callable[[NodeDescriptor], Awaitable[None] | None]
    ) -> None:
        """Register a callback fired when a new peer is discovered.

        The callback receives the peer's ``NodeDescriptor``.  It may be
        a plain function or an async function.
        """
        self._join_callbacks.append(callback)

    def on_leave(
        self, callback: Callable[[str, NodeDescriptor], Awaitable[None] | None]
    ) -> None:
        """Register a callback fired when a peer is evicted as stale.

        The callback receives ``(node_id, last_known_descriptor)``.  It may
        be a plain function or an async function.
        """
        self._leave_callbacks.append(callback)

    async def _fire_join(self, descriptor: NodeDescriptor) -> None:
        """Fire all registered join callbacks for *descriptor*."""
        for cb in self._join_callbacks:
            try:
                result = cb(descriptor)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Join callback %r raised", cb)

    async def _fire_leave(self, node_id: str, descriptor: NodeDescriptor) -> None:
        """Fire all registered leave callbacks for *node_id*."""
        for cb in self._leave_callbacks:
            try:
                result = cb(node_id, descriptor)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Leave callback %r raised", cb)


# ── Discovery protocol (internal) ──────────────────────────────────────────────


class _DiscoveryProtocol(asyncio.DatagramProtocol):
    """asyncio UDP protocol that funnels received datagrams to a callback.

    The callback receives raw bytes and the sender address.  It is scheduled
    as a background task so it can safely do async work (like acquiring
    FleetTable's lock).
    """

    def __init__(
        self,
        on_datagram: Callable[[bytes, tuple[str, int]], Awaitable[None]],
    ) -> None:
        self.transport: asyncio.DatagramTransport | None = None
        self._on_datagram = on_datagram
        # Strong references to in-flight tasks. asyncio.create_task() only
        # holds a weak reference via the event loop; without this set, a
        # task can be garbage-collected mid-flight and any exception inside
        # it is silently lost -- no traceback, no log line.
        self._background_tasks: set[asyncio.Task] = set()

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        # Schedule async handling without blocking the protocol.
        # Keep a strong reference until the task completes, then discard it.
        task = asyncio.create_task(self._on_datagram(data, addr))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def error_received(self, exc: Exception) -> None:
        logger.debug("UDP error: %s", exc)

    def connection_lost(self, exc: Exception | None) -> None:
        if exc is not None:
            logger.warning("Discovery UDP connection lost: %s", exc)


# ── NodeIdentity ───────────────────────────────────────────────────────────────


class NodeIdentity:
    """Per-node entry point for zero-coordinator peer discovery.

    Owns this node's persistent identity, periodically broadcasts a capability
    descriptor over UDP, listens for peers' descriptors, and maintains a
    ``FleetTable`` view that Layer 4 consumes.

    **No roles are assigned here.**  This module's job stops at "here is the
    live fleet."  Scheduling, expert sharding, pipeline-staging — all of that
    belongs in the distributed scheduler layer, which reads ``.fleet``.

    Parameters
    ----------
    port:
        The TCP port this node will listen on for inter-node data traffic.
        Advertised in the descriptor; nothing listens on it in this module.
    discovery_port:
        UDP port for broadcast/multicast discovery (default 9876).
    hardware_gen:
        Free-form hardware generation tag, e.g. ``"rk3588-8gb"``.  Nodes of
        different generations coexist in the same fleet by design.
    storage_bandwidth_mbps:
        Advertised storage bandwidth.  Defaults to 0 ("unmeasured") — a
        separate benchmarking module should update this via
        ``update_storage_bandwidth()``.
    broadcast_interval:
        Seconds between descriptor broadcasts (default 2.0).
    stale_timeout:
        Seconds before a peer is evicted from the fleet table (default 8.0).
    multicast_group:
        If set, use IP multicast (e.g. ``"239.255.42.99"``) instead of
        subnet broadcast.  Required for multi-instance testing on a single
        machine.
    bind_ip:
        If set, bind the UDP socket to this specific local IP.  Defaults to
        ``""`` (INADDR_ANY), which listens on all interfaces.
    ip:
        Override the advertised IP address.  Defaults to auto-detection.
    hostname:
        Override the advertised hostname.  Defaults to ``socket.gethostname()``.
    node_id:
        Override the stable node UUID.  Defaults to a persisted UUID in
        ``~/.swarm/node_id.json``.
    """

    def __init__(
        self,
        *,
        port: int,
        discovery_port: int = DEFAULT_DISCOVERY_PORT,
        hardware_gen: str = "unknown",
        storage_bandwidth_mbps: int = 0,
        broadcast_interval: float = DEFAULT_BROADCAST_INTERVAL,
        stale_timeout: float = DEFAULT_STALE_TIMEOUT,
        multicast_group: str | None = None,
        bind_ip: str = "0.0.0.0",
        ip: str | None = None,
        hostname: str | None = None,
        node_id: str | None = None,
    ) -> None:
        # ── Identity ──────────────────────────────────────────────────────
        self._node_id = node_id or _load_or_create_node_id()
        self._hostname = hostname or socket.gethostname()
        self._ip = ip or _detect_ip()
        self._port = port
        self._hardware_gen = hardware_gen

        ram_total, _ = _get_ram_info()
        self._ram_total_mb = ram_total

        # ── Volatile state (updated across broadcasts) ────────────────────
        self._load: float = 0.0
        self._storage_bandwidth_mbps: int = storage_bandwidth_mbps
        self._start_time: float | None = None  # set in start()

        # ── Network config ────────────────────────────────────────────────
        self._discovery_port = discovery_port
        self._broadcast_interval = broadcast_interval
        self._multicast_group = multicast_group
        self._bind_ip = bind_ip

        # ── Sub-components ────────────────────────────────────────────────
        self.fleet = FleetTable(self._node_id, stale_timeout=stale_timeout)

        # ── Runtime handles ───────────────────────────────────────────────
        self._transport: asyncio.DatagramTransport | None = None
        self._broadcast_task: asyncio.Task[None] | None = None
        self._running: bool = False

    # ── Public API ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start discovery: create the UDP endpoint, begin broadcasting, begin
        the eviction sweep.

        Idempotent — calling ``start()`` on an already-running node is a no-op.
        """
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()

        loop = asyncio.get_running_loop()

        # ── Create UDP endpoint ───────────────────────────────────────────
        local_addr = (self._bind_ip, self._discovery_port)
        transport, _protocol = await loop.create_datagram_endpoint(
            lambda: _DiscoveryProtocol(on_datagram=self._on_datagram_received),
            local_addr=local_addr,
            reuse_port=True,  # allow multiple listeners on same port
        )
        self._transport = transport

        # ── Configure multicast if requested ──────────────────────────────
        if self._multicast_group is not None:
            sock = transport.get_extra_info("socket")
            if sock is not None:
                try:
                    # Interface to join/send multicast on. On WSL2 and other
                    # virtualised network stacks, the default route interface
                    # does not loop multicast back to local listeners, so allow
                    # forcing loopback via SWARM_MCAST_IF=127.0.0.1.
                    mcast_if = os.environ.get("SWARM_MCAST_IF", "0.0.0.0")
                    mreq = socket.inet_aton(self._multicast_group) + socket.inet_aton(
                        mcast_if
                    )
                    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                    sock.setsockopt(
                        socket.IPPROTO_IP,
                        socket.IP_MULTICAST_IF,
                        socket.inet_aton(mcast_if),
                    )
                    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
                    # Enable loopback so we can hear our own multicast on the
                    # same machine (we filter by node_id, so this is safe).
                    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
                except OSError as exc:
                    logger.warning("Multicast setup partially failed: %s", exc)

        # ── Enable broadcast on the socket ────────────────────────────────
        sock = transport.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            except OSError:
                pass  # not fatal; may already be set

        # ── Start background tasks ────────────────────────────────────────
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        await self.fleet.start_eviction_sweep()

        logger.info(
            "NodeIdentity started: %s (%s) on %s:%d",
            self._node_id[:8],
            self._hostname,
            self._ip,
            self._port,
        )

    async def stop(self) -> None:
        """Stop discovery: cancel broadcast, close UDP, stop eviction sweep."""
        self._running = False

        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None

        if self._transport is not None:
            self._transport.close()
            self._transport = None

        await self.fleet.stop()
        logger.info("NodeIdentity stopped: %s", self._node_id[:8])

    def update_load(self, load: float) -> None:
        """Set the current load (0.0–1.0) advertised in the next broadcast."""
        self._load = max(0.0, min(1.0, load))

    def update_storage_bandwidth(self, mbps: int) -> None:
        """Set the measured storage bandwidth advertised in the next broadcast."""
        self._storage_bandwidth_mbps = mbps

    def _make_descriptor(self) -> NodeDescriptor:
        """Build a fresh descriptor for the current moment."""
        now = time.time()
        uptime = (
            now - (self._start_time or now)
            if self._start_time is not None
            else 0.0
        )
        ram_available = 0
        ram_total, _ = _get_ram_info()
        if ram_total == 0:
            ram_total = self._ram_total_mb
        # Re-read available RAM each broadcast
        _, ram_available = _get_ram_info()

        return NodeDescriptor(
            node_id=self._node_id,
            hostname=self._hostname,
            ip=self._ip,
            port=self._port,
            ram_total_mb=ram_total or self._ram_total_mb,
            ram_available_mb=ram_available,
            storage_bandwidth_mbps=self._storage_bandwidth_mbps,
            hardware_gen=self._hardware_gen,
            load=self._load,
            uptime_seconds=uptime,
            timestamp=now,
        )

    # ── Broadcast loop ────────────────────────────────────────────────────

    async def _broadcast_loop(self) -> None:
        """Periodically send this node's descriptor over UDP."""
        dest = (self._multicast_group or "255.255.255.255", self._discovery_port)
        while self._running:
            try:
                desc = self._make_descriptor()
                payload = desc.to_json().encode("utf-8")
                if self._transport is not None:
                    self._transport.sendto(payload, dest)
            except Exception:
                logger.exception("Broadcast failed")
            await asyncio.sleep(self._broadcast_interval)

    # ── Ingress ───────────────────────────────────────────────────────────

    async def _on_datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Handle an incoming discovery datagram.

        Parses the JSON, updates the fleet table, and fires join callbacks
        for newly-seen nodes.
        """
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return

        descriptor = NodeDescriptor.from_json(text)
        if descriptor is None:
            return

        # Reject descriptors from non-peers (loopback, obviously wrong)
        if descriptor.ip in ("127.0.0.1", "::1") and descriptor.node_id != self._node_id:
            return

        is_new = await self.fleet.update(descriptor)
        if is_new:
            logger.info(
                "New peer: %s (%s) @ %s:%d [%s]",
                descriptor.node_id[:8],
                descriptor.hostname,
                descriptor.ip,
                descriptor.port,
                descriptor.hardware_gen,
            )
            await self.fleet._fire_join(descriptor)


# ── CLI test harness ───────────────────────────────────────────────────────────


async def _amain() -> None:
    parser = argparse.ArgumentParser(
        description="Swarm node discovery — test harness for local development."
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="TCP port this node advertises (9999, etc.)",
    )
    parser.add_argument(
        "--node-name",
        type=str,
        default=None,
        help="Override hostname in the descriptor (useful for multi-instance testing)",
    )
    parser.add_argument(
        "--discovery-port",
        type=int,
        default=DEFAULT_DISCOVERY_PORT,
        help=f"UDP discovery port (default: {DEFAULT_DISCOVERY_PORT})",
    )
    parser.add_argument(
        "--multicast-group",
        type=str,
        default=DEFAULT_MULTICAST_GROUP,
        help=(
            "IP multicast group for discovery "
            f"(default: {DEFAULT_MULTICAST_GROUP}). "
            "Use this for multi-instance testing on one machine."
        ),
    )
    parser.add_argument(
        "--hardware-gen",
        type=str,
        default="dev-machine",
        help='Hardware generation tag (default: "dev-machine")',
    )
    parser.add_argument(
        "--stale-timeout",
        type=float,
        default=DEFAULT_STALE_TIMEOUT,
        help=f"Seconds before a peer is evicted (default: {DEFAULT_STALE_TIMEOUT})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress the noisy multicast warning when not on real hardware
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    node = NodeIdentity(
        port=args.port,
        discovery_port=args.discovery_port,
        hardware_gen=args.hardware_gen,
        multicast_group=args.multicast_group,
        stale_timeout=args.stale_timeout,
        hostname=args.node_name,
    )

    # ── Callbacks (demonstrate the API) ───────────────────────────────────
    node.fleet.on_join(
        lambda desc: print(f"  >>> JOIN:  {desc.hostname:<20} {desc.node_id[:8]}…  {desc.hardware_gen}")
    )
    node.fleet.on_leave(
        lambda nid, desc: print(f"  <<< LEAVE: {desc.hostname:<20} {nid[:8]}…")
    )

    await node.start()

    # ── Print fleet table every second ────────────────────────────────────
    try:
        while True:
            await asyncio.sleep(1.0)
            live = await node.fleet.get_live_nodes()
            desc = node._make_descriptor()

            # Clear screen-ish header
            print(f"\n{'─' * 70}")
            print(f"  SELF  {desc.hostname:<20} {desc.node_id[:8]}…  "
                  f"ip={desc.ip}  port={desc.port}  load={desc.load:.1f}")
            print(f"  Peers: {len(live)}")
            if live:
                for peer in sorted(live, key=lambda d: d.hostname):
                    age = time.time() - peer.timestamp
                    print(
                        f"    {peer.hostname:<20} {peer.node_id[:8]}…  "
                        f"ip={peer.ip}:{peer.port}  "
                        f"load={peer.load:.1f}  "
                        f"hw={peer.hardware_gen:<14}  "
                        f"bw={peer.storage_bandwidth_mbps} Mbps  "
                        f"age={age:.1f}s"
                    )
            else:
                print("    (none — waiting for peers…)")
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        await node.stop()


def main() -> None:
    """Entry point for the test harness."""
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
