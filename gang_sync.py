#!/usr/bin/env python3
"""
gang_sync.py — Sequential ring-reduce for the Swarm distributed MoE scheduler.

Layer 4 of the software stack.  This module makes "gang mode" work:
in MoE mode every node computes the same layer in lockstep.  No node may
start layer N+1 until every node's contribution to layer N has been summed
and distributed back to all of them.

Algorithm: sequential ring-reduce + fan-out
-------------------------------------------
This is deliberately NOT a full bandwidth-optimal two-phase ring-allreduce.

At the target scale (4 to 6 nodes, ~12 KB per layer boundary — see
architecture.md §3.1) the latency of a few sequential hops is far smaller
than the bandwidth term in a full allreduce, and the simple version avoids
the second ring pass, the recursive-doubling coordination, and the
associated state-machine complexity.

The algorithm:

1.  Each node starts with its own local contribution (bytes) for layer N.
2.  Node 0 sends its contribution to node 1.
3.  Node 1 combines (combine() callback) its own contribution with what it
    received, forwards the result to node 2.  Continues around the ring.
4.  The last node now holds the complete sum.  It sends the final result
    directly to every other node (fan-out, not a second ring pass).
5.  ring_reduce() returns only once every node has the final summed result.

Ring order: coordinator-free, deterministic
--------------------------------------------
Ring order is computed by sorting the node_ids from the ShardAssignment
(``sorted(assignment.node_counts.keys())``).  This is the same sort order
that sharding.py uses internally for its own determinism.  Every node
computes the identical ring independently from the same assignment.
There is no leader election, no negotiation, no voting.

This property is load-bearing: two nodes that disagree on ring order will
send messages to the wrong peers, produce different sums, or hang.  The
fleet_hash from ShardAssignment is embedded in every ring message so that
such a disagreement is detected and surfaced as a GangSyncError rather than
a silent wrong result.

What this module does NOT do
-----------------------------
- **No tensor/numpy assumptions.**  Payload is opaque ``bytes``.  The caller
  supplies a ``combine(bytes, bytes) -> bytes`` callback.
- **No failover.**  If a node is unresponsive past *timeout*, raise
  ``GangSyncTimeout`` and stop.  Do not reform the ring or retry.  Failover
  is a separate, not-yet-built module.
- **No full two-phase allreduce.**  Unnecessary at this node count and
  data size (see above).

Message framing
---------------
Ring messages use rpc.py ``MessageType.ACTIVATION``.  The inner framing
(prepended to the caller's payload inside ``Frame.payload``):

    ┌──────────────┬──────────────┬────────────────────┬───────────┐
    │  layer_id    │ ring_msg_type│ fleet_hash_raw     │ payload   │
    │  uint32 BE   │  uint8       │  32 bytes (binary) │ variable  │
    │    4 B       │   1 B        │    32 B            │           │
    └──────────────┴──────────────┴────────────────────┴───────────┘

``ring_msg_type`` values:
  - 0x01  RING_HOP      — accumulated sum being passed around the ring
  - 0x02  FAN_OUT_RESULT — final result broadcast from the last node

``fleet_hash_raw`` is ``bytes.fromhex(assignment.fleet_hash)``, i.e. the
raw 32-byte SHA-256.  A node rejects any message whose fleet_hash does not
match its own ShardAssignment for that layer_id, preventing cross-talk
between two nodes that computed sharding from different fleet snapshots.

Design rules (do not break)
----------------------------
- Python 3.11+, standard library only, plus rpc.py and sharding.py.
- All background tasks strongly referenced (asyncio.create_task pitfall —
  this has been a real bug twice already in this project).
- No coordinator, no leader election, no voting.
- Timeout on every wait; raise GangSyncTimeout, never hang.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import struct
from collections.abc import Callable

from rpc import Frame, MessageType, RpcClient, RpcConnection, RpcError, RpcServer
from sharding import ShardAssignment

logger = logging.getLogger("swarm.gang_sync")

# ── Constants ──────────────────────────────────────────────────────────────────

# Ring message types embedded in our inner framing (NOT rpc.py MessageType).
RING_HOP: int = 0x01
FAN_OUT_RESULT: int = 0x02

# Inner framing layout (does NOT include the caller's payload — that follows).
# Format: "!IB32s" = uint32 layer_id, uint8 ring_msg_type, 32s fleet_hash_raw
_RING_FRAME_HEADER_FORMAT: str = "!IB32s"
_RING_FRAME_HEADER_SIZE: int = struct.calcsize(_RING_FRAME_HEADER_FORMAT)  # 37

# Default timeout for waiting on any ring message.
DEFAULT_RING_TIMEOUT: float = 5.0

# Sanity cap on a single ring payload.  rpc.py enforces its own
# max_frame_bytes at the transport layer; this is a second, cheaper check
# at the application layer so an oversized payload is rejected with a
# gang-sync-specific error naming the layer, rather than surfacing as an
# opaque transport failure.  A combine() callback that returns a runaway
# result (a bug in caller code, not an attack) is caught here before it is
# forwarded around the whole ring and amplified N times.
MAX_RING_PAYLOAD_BYTES: int = 64 * 1024 * 1024  # 64 MB


# ── Exceptions ─────────────────────────────────────────────────────────────────


class GangSyncError(Exception):
    """Raised for gang-sync failures that are not timeouts.

    Covers: fleet-hash mismatch (two nodes disagree on sharding),
    send failure during ring forwarding, and internal protocol violations.
    """


class GangSyncTimeout(GangSyncError):
    """A ring-reduce operation timed out waiting for an expected message.

    Attributes
    ----------
    layer_id:
        The layer whose ring pass timed out.
    expected_from:
        Human description of which node or hop was expected.
    """

    def __init__(self, layer_id: int, expected_from: str) -> None:
        self.layer_id = layer_id
        self.expected_from = expected_from
        super().__init__(
            f"Layer {layer_id}: timed out waiting for message from {expected_from}"
        )


# ── Ring message framing ──────────────────────────────────────────────────────


def _encode_ring_message(
    *,
    layer_id: int,
    ring_msg_type: int,
    fleet_hash: str,
    payload: bytes,
) -> bytes:
    """Encode a ring message: inner framing header + caller payload.

    Parameters
    ----------
    layer_id:
        Monotonic layer identifier.  Must match across all nodes in the ring.
    ring_msg_type:
        ``RING_HOP`` or ``FAN_OUT_RESULT``.
    fleet_hash:
        64-character hex SHA-256 from ``ShardAssignment.fleet_hash``.
        Converted to raw 32 bytes on the wire.
    payload:
        The accumulated sum bytes (caller's data).
    """
    if len(payload) > MAX_RING_PAYLOAD_BYTES:
        raise GangSyncError(
            f"Layer {layer_id}: ring payload is {len(payload)} bytes, "
            f"over the {MAX_RING_PAYLOAD_BYTES} byte cap. This usually "
            f"means the combine() callback returned a runaway result."
        )
    fleet_hash_raw = bytes.fromhex(fleet_hash)
    if len(fleet_hash_raw) != 32:
        raise ValueError(
            f"fleet_hash must be a 64-char hex string (32 bytes raw), "
            f"got {len(fleet_hash)} chars"
        )
    header = struct.pack(
        _RING_FRAME_HEADER_FORMAT,
        layer_id,
        ring_msg_type,
        fleet_hash_raw,
    )
    return header + payload


def _decode_ring_message(data: bytes) -> tuple[int, int, str, bytes] | None:
    """Decode a ring message from raw bytes.

    Returns ``(layer_id, ring_msg_type, fleet_hash, payload)`` on success,
    or ``None`` if the data is too short to contain a valid header.

    The returned ``fleet_hash`` is the 64-character hex string, reconstructed
    from the 32 raw bytes on the wire, for direct comparison with
    ``ShardAssignment.fleet_hash``.
    """
    if len(data) < _RING_FRAME_HEADER_SIZE:
        return None
    layer_id, ring_msg_type, fleet_hash_raw = struct.unpack(
        _RING_FRAME_HEADER_FORMAT,
        data[:_RING_FRAME_HEADER_SIZE],
    )
    fleet_hash = fleet_hash_raw.hex()
    payload = data[_RING_FRAME_HEADER_SIZE:]
    return layer_id, ring_msg_type, fleet_hash, payload


# ── Internal state ─────────────────────────────────────────────────────────────


@dataclasses.dataclass
class _RingState:
    """Per-layer state for an in-progress ring-reduce call.

    Created by ``ring_reduce()``, consumed by the handler, cleaned up when
    the call returns (success or failure).
    """

    layer_id: int
    fleet_hash: str
    ring_order: list[str]          # node_ids, sorted
    own_position: int              # index into ring_order
    own_node_id: str
    local_contribution: bytes
    combine: Callable[[bytes, bytes], bytes]
    result_future: asyncio.Future[bytes]


# ── GangSync ───────────────────────────────────────────────────────────────────


class GangSync:
    """Per-node coordinator for gang-mode ring-reduce.

    One instance per node.  Owns the handler registered with the node's
    ``RpcServer`` and orchestrates ring-reduce calls.

    Parameters
    ----------
    own_node_id:
        Stable UUID of *this* node.  Must appear in every
        ``ShardAssignment`` passed to ``ring_reduce()``.
    rpc_client:
        An ``RpcClient`` connected (or connectable) to all peer nodes.
        Used to send ring-hop and fan-out messages.
    peers:
        Mapping from ``node_id`` to ``(host, port)`` for every node in the
        fleet.  Needed because ``ShardAssignment`` carries only
        ``node_id``, not network addresses, and ring forwarding requires
        knowing where to send.
    """

    def __init__(
        self,
        *,
        own_node_id: str,
        rpc_client: RpcClient,
        peers: dict[str, tuple[str, int]],
    ) -> None:
        self._own_node_id = own_node_id
        self._rpc_client = rpc_client
        self._peers = peers

        # layer_id → _RingState for in-progress ring-reduce calls.
        self._pending: dict[int, _RingState] = {}

        # Early-arrival buffer.  A ring-hop may land before ring_reduce()
        # has been called on this node (concurrent asyncio.gather launch).
        # We buffer one message per layer_id so it can be replayed when
        # ring_reduce() sets up state.
        # layer_id → (ring_msg_type, fleet_hash, payload)
        self._early_buffer: dict[int, tuple[int, str, bytes]] = {}
        # Bounded to _MAX_EARLY_BUFFER entries to prevent unbounded growth
        # from stale messages for layers that will never be called.
        self._MAX_EARLY_BUFFER: int = 16

        # Strong references to background tasks (see module docstring).
        self._background_tasks: set[asyncio.Task] = set()

    # ── Public API ─────────────────────────────────────────────────────────

    async def ring_reduce(
        self,
        assignment: ShardAssignment,
        layer_id: int,
        local_contribution: bytes,
        combine: Callable[[bytes, bytes], bytes],
        *,
        timeout: float = DEFAULT_RING_TIMEOUT,
    ) -> bytes:
        """Run one ring-reduce pass for layer *layer_id*.

        Blocks until every node in the ring has received the summed result,
        or until *timeout* seconds elapse waiting for an expected message.

        Parameters
        ----------
        assignment:
            The current ``ShardAssignment``.  Ring order is derived from
            ``sorted(assignment.node_counts.keys())``.  ``fleet_hash`` is
            embedded in every outbound ring message and checked on every
            inbound one.
        layer_id:
            Monotonic layer identifier.  Must be identical across all nodes.
            Messages with a different ``layer_id`` are silently dropped.
        local_contribution:
            This node's contribution for the current layer (opaque bytes).
        combine:
            Callback ``(accumulated: bytes, local: bytes) -> bytes`` that
            performs elementwise addition.  ``accumulated`` is the
            already-summed contribution from prior ring nodes; ``local`` is
            this node's ``local_contribution``.
        timeout:
            Seconds to wait for any expected ring message before raising
            ``GangSyncTimeout``.

        Returns
        -------
        bytes
            The complete summed result, identical on every node in the ring.

        Raises
        ------
        ValueError
            If ``own_node_id`` is not in *assignment*, or ``fleet_hash`` is
            not a valid 64-character hex string.
        GangSyncTimeout
            If an expected ring message does not arrive within *timeout*.
        GangSyncError
            On fleet-hash mismatch (peer computed sharding from a different
            fleet snapshot) or send failure during forwarding.
        """
        if self._own_node_id not in assignment.node_counts:
            raise ValueError(
                f"own_node_id {self._own_node_id[:8]}… is not in the "
                f"ShardAssignment (fleet has {len(assignment.node_counts)} nodes)"
            )

        # ── Compute ring order ─────────────────────────────────────────
        # Same sort order sharding.py uses internally for determinism.
        ring_order = sorted(assignment.node_counts.keys())
        # dict keys are unique by construction, but assert the invariant
        # the ring depends on rather than trusting it silently: a duplicate
        # node_id would make ring_order.index() return the wrong position
        # and route messages in a cycle that never terminates.
        if len(set(ring_order)) != len(ring_order):
            raise GangSyncError(
                f"Layer {layer_id}: ShardAssignment contains duplicate "
                f"node_ids; ring order would be ambiguous"
            )
        own_position = ring_order.index(self._own_node_id)
        is_first = own_position == 0
        is_last = own_position == len(ring_order) - 1

        # Single-node ring: nothing to combine, return immediately.
        if len(ring_order) == 1:
            return local_contribution

        # ── Create state ───────────────────────────────────────────────
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[bytes] = loop.create_future()

        # Reject a concurrent second call for the same layer_id.  Without
        # this, the second call silently overwrites the first's state and
        # the first caller's future is orphaned -- it never resolves, so
        # that caller blocks until its own timeout with no indication why.
        # A duplicate layer_id in flight is a caller bug; surface it.
        if layer_id in self._pending:
            raise GangSyncError(
                f"Layer {layer_id}: a ring_reduce for this layer_id is "
                f"already in progress on this node. Concurrent calls with "
                f"the same layer_id are not supported -- each layer must "
                f"complete before it is reused."
            )

        state = _RingState(
            layer_id=layer_id,
            fleet_hash=assignment.fleet_hash,
            ring_order=ring_order,
            own_position=own_position,
            own_node_id=self._own_node_id,
            local_contribution=local_contribution,
            combine=combine,
            result_future=result_future,
        )
        self._pending[layer_id] = state

        try:
            # ── Drain early-buffer if a message arrived before we were ready ─
            early = self._early_buffer.pop(layer_id, None)
            if early is not None:
                ring_msg_type, fleet_hash_in_msg, payload = early
                self._handle_ring_message(
                    state, ring_msg_type, fleet_hash_in_msg, payload
                )

            # ── First node initiates the ring ──────────────────────────
            if is_first:
                successor = ring_order[1]
                successor_addr = self._peers.get(successor)
                if successor_addr is None:
                    raise GangSyncError(
                        f"No peer address for successor {successor[:8]}… "
                        f"in layer {layer_id}"
                    )
                logger.debug(
                    "Layer %d: node 0 sending initial hop to %s…",
                    layer_id,
                    successor[:8],
                )
                try:
                    await self._rpc_client.send(
                        successor_addr,
                        MessageType.ACTIVATION,
                        _encode_ring_message(
                            layer_id=layer_id,
                            ring_msg_type=RING_HOP,
                            fleet_hash=assignment.fleet_hash,
                            payload=local_contribution,
                        ),
                    )
                except (RpcError, ConnectionError, OSError) as exc:
                    raise GangSyncError(
                        f"Layer {layer_id}: failed to send initial ring-hop "
                        f"to {successor[:8]}…: {exc}"
                    ) from exc

            # ── Wait for the result ─────────────────────────────────────
            expected_desc = (
                "fan-out from last node"
                if not is_last
                else "ring-hop from predecessor (last node combines, then "
                      "self-sets result)"
            )
            try:
                return await asyncio.wait_for(result_future, timeout=timeout)
            except asyncio.TimeoutError:
                raise GangSyncTimeout(layer_id, expected_desc) from None

        finally:
            self._cleanup_state(layer_id)

    # ── Handler (register with RpcServer) ──────────────────────────────────

    async def handle_frame(self, conn: RpcConnection, frame: Frame) -> None:
        """Handler for inbound ring messages.  Register with ``RpcServer``.

        Usage::

            gang = GangSync(own_node_id=..., rpc_client=..., peers=...)
            server = RpcServer(own_node_id=..., port=..., handler=gang.handle_frame)

        Only ``MessageType.ACTIVATION`` frames are inspected.  All other
        message types are silently ignored (they belong to other layers of
        the stack).
        """
        if frame.msg_type != MessageType.ACTIVATION:
            return

        if frame.payload is None:
            logger.debug("ACTIVATION frame with no payload — ignoring")
            return

        decoded = _decode_ring_message(frame.payload)
        if decoded is None:
            # Payload too short to be one of ours — not a ring message.
            return

        layer_id, ring_msg_type, fleet_hash_in_msg, payload = decoded

        state = self._pending.get(layer_id)
        if state is None:
            # No in-progress ring_reduce for this layer_id.  Buffer in
            # case ring_reduce hasn't been called yet (concurrent launch).
            # Enforce a bound on the early buffer so stale messages
            # for layers that will never be called don't leak memory.
            if len(self._early_buffer) >= self._MAX_EARLY_BUFFER:
                # Evict the oldest entry (smallest layer_id — best-effort LRU).
                oldest = min(self._early_buffer.keys())
                logger.debug(
                    "Early buffer full (%d entries) — evicting layer %d",
                    len(self._early_buffer),
                    oldest,
                )
                del self._early_buffer[oldest]
            logger.debug(
                "Layer %d: message arrived before ring_reduce — buffering",
                layer_id,
            )
            self._early_buffer[layer_id] = (ring_msg_type, fleet_hash_in_msg, payload)
            return

        # Delegate to the inner handler with state context.
        self._handle_ring_message(state, ring_msg_type, fleet_hash_in_msg, payload)

    # ── Internal: message handling ─────────────────────────────────────────

    def _handle_ring_message(
        self,
        state: _RingState,
        ring_msg_type: int,
        fleet_hash_in_msg: str,
        payload: bytes,
    ) -> None:
        """Process a decoded ring message against an active _RingState.

        This is synchronous (no awaits) so it can be called from either
        the async handler or the early-buffer drain path without
        double-await confusion.  Any async work (sending to peers) is
        spawned as a background task with a strong reference kept.
        """
        # ── Fleet-hash gate ────────────────────────────────────────────
        # This is the "fail loudly" check: two nodes with different
        # fleet snapshots cannot produce a valid result.
        if fleet_hash_in_msg != state.fleet_hash:
            msg = (
                f"Layer {state.layer_id}: fleet_hash mismatch from peer. "
                f"Expected {state.fleet_hash[:16]}…, "
                f"got {fleet_hash_in_msg[:16]}…. "
                f"This node computed sharding from a different fleet "
                f"snapshot than the sending peer."
            )
            logger.error(msg)
            if not state.result_future.done():
                state.result_future.set_exception(GangSyncError(msg))
            return

        if ring_msg_type == RING_HOP:
            self._handle_hop(state, payload)
        elif ring_msg_type == FAN_OUT_RESULT:
            self._handle_fan_out(state, payload)
        else:
            logger.warning(
                "Layer %d: unknown ring_msg_type 0x%02X — ignoring",
                state.layer_id,
                ring_msg_type,
            )

    def _handle_hop(self, state: _RingState, accumulated: bytes) -> None:
        """Process a RING_HOP: combine, forward or fan-out.

        Called from the handler task (has access to event loop).
        """
        layer_id = state.layer_id
        ring_order = state.ring_order
        own_pos = state.own_position
        is_last = own_pos == len(ring_order) - 1

        # Combine received + local.
        try:
            combined = state.combine(accumulated, state.local_contribution)
        except Exception as exc:
            logger.exception("Layer %d: combine() callback raised", layer_id)
            if not state.result_future.done():
                state.result_future.set_exception(
                    GangSyncError(
                        f"Layer {layer_id}: combine() failed: {exc}"
                    )
                )
            return

        if is_last:
            # ── Last node: fan-out to all others ───────────────────────
            logger.debug(
                "Layer %d: last node — fanning out result to %d peers",
                layer_id,
                len(ring_order) - 1,
            )
            # Set own result immediately (we already have it).
            if not state.result_future.done():
                state.result_future.set_result(combined)

            # Fan-out to every other node.  Fire-and-forget; spawn as a
            # background task so a slow peer doesn't block the handler slot
            # longer than necessary.  A failed fan-out means the peer will
            # time out and raise GangSyncTimeout — that's their problem,
            # not ours.
            task = asyncio.create_task(
                self._fan_out_to_peers(
                    layer_id=layer_id,
                    fleet_hash=state.fleet_hash,
                    combined=combined,
                    ring_order=ring_order,
                    exclude_self=True,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        else:
            # ── Forward to successor ───────────────────────────────────
            successor = ring_order[own_pos + 1]
            successor_addr = self._peers.get(successor)
            if successor_addr is None:
                if not state.result_future.done():
                    state.result_future.set_exception(
                        GangSyncError(
                            f"Layer {layer_id}: no peer address for "
                            f"successor {successor[:8]}…"
                        )
                    )
                return

            logger.debug(
                "Layer %d: forwarding combined to successor %s…",
                layer_id,
                successor[:8],
            )
            # Spawn as background task so send failure is handled cleanly.
            task = asyncio.create_task(
                self._forward_to_successor(
                    layer_id=layer_id,
                    fleet_hash=state.fleet_hash,
                    combined=combined,
                    successor=successor,
                    successor_addr=successor_addr,
                    state=state,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    def _handle_fan_out(self, state: _RingState, payload: bytes) -> None:
        """Process a FAN_OUT_RESULT: the final sum has arrived."""
        if not state.result_future.done():
            state.result_future.set_result(payload)

    # ── Internal: network I/O (run as background tasks) ────────────────────

    async def _forward_to_successor(
        self,
        *,
        layer_id: int,
        fleet_hash: str,
        combined: bytes,
        successor: str,
        successor_addr: tuple[str, int],
        state: _RingState,
    ) -> None:
        """Send the combined result to the next node in the ring.

        Runs as a background task.  On failure, sets the result future
        with a GangSyncError so the caller sees the failure immediately
        rather than waiting for a timeout.
        """
        try:
            await self._rpc_client.send(
                successor_addr,
                MessageType.ACTIVATION,
                _encode_ring_message(
                    layer_id=layer_id,
                    ring_msg_type=RING_HOP,
                    fleet_hash=fleet_hash,
                    payload=combined,
                ),
            )
        except (RpcError, ConnectionError, OSError) as exc:
            msg = (
                f"Layer {layer_id}: failed to forward ring-hop to "
                f"{successor[:8]}…: {exc}"
            )
            logger.error(msg)
            if not state.result_future.done():
                state.result_future.set_exception(GangSyncError(msg))

    async def _fan_out_to_peers(
        self,
        *,
        layer_id: int,
        fleet_hash: str,
        combined: bytes,
        ring_order: list[str],
        exclude_self: bool,
    ) -> None:
        """Send the final summed result to every other node in the ring.

        Runs as a background task spawned by the last node's handler.
        Individual send failures are logged but do not prevent other
        fan-outs from proceeding — the affected peers will time out and
        raise GangSyncTimeout on their own.
        """
        encoded = _encode_ring_message(
            layer_id=layer_id,
            ring_msg_type=FAN_OUT_RESULT,
            fleet_hash=fleet_hash,
            payload=combined,
        )
        # Fan out concurrently, not sequentially.  A sequential loop means
        # one slow or unreachable peer delays every peer after it in the
        # list -- with a 5 s connect timeout and 5 nodes, the last peer
        # could wait 20 s for a result that was ready immediately, and
        # would likely hit its own ring timeout first.  Sending in
        # parallel bounds total fan-out time by the slowest single peer
        # rather than the sum of all of them.
        async def _send_one(node_id: str) -> None:
            addr = self._peers.get(node_id)
            if addr is None:
                logger.error(
                    "Layer %d: no peer address for %s… during fan-out",
                    layer_id,
                    node_id[:8],
                )
                return
            try:
                await self._rpc_client.send(
                    addr,
                    MessageType.ACTIVATION,
                    encoded,
                )
            except (RpcError, ConnectionError, OSError) as exc:
                logger.error(
                    "Layer %d: fan-out to %s… failed: %s",
                    layer_id,
                    node_id[:8],
                    exc,
                )

        targets = [
            nid for nid in ring_order
            if not (exclude_self and nid == self._own_node_id)
        ]
        # return_exceptions=True: _send_one already swallows the expected
        # network errors, so anything reaching here is unexpected -- log it
        # rather than letting it vanish into the background task.
        results = await asyncio.gather(
            *(_send_one(nid) for nid in targets),
            return_exceptions=True,
        )
        for nid, res in zip(targets, results):
            if isinstance(res, BaseException):
                logger.error(
                    "Layer %d: unexpected error fanning out to %s…: %r",
                    layer_id,
                    nid[:8],
                    res,
                )

    # ── Internal: cleanup ──────────────────────────────────────────────────

    def _cleanup_state(self, layer_id: int) -> None:
        """Remove state for *layer_id* and cancel any lingering future."""
        state = self._pending.pop(layer_id, None)
        if state is not None and not state.result_future.done():
            state.result_future.cancel()
        # Also purge early buffer entry if ring_reduce never picked it up.
        self._early_buffer.pop(layer_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════


async def _demo() -> None:
    """Run a self-contained demo of gang-mode ring-reduce.

    Spins up 4 real RpcServer/RpcClient nodes on localhost, gives each a
    synthetic integer contribution, runs ring_reduce concurrently, and
    verifies every node gets the identical correct sum.

    Also demonstrates the timeout path: kills one node mid-ring and
    confirms survivors raise GangSyncTimeout without hanging.
    """
    import time

    from rpc import RpcServer
    from sharding import NodeCapability, ShardAssignment, compute_assignment

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("swarm.gang_sync").setLevel(logging.INFO)

    # ── Synthetic fleet: 4 nodes, equal bandwidth ──────────────────────
    NUM_NODES = 4
    BASE_PORT = 21000
    NODE_IDS = [f"node-{i:02d}-aaaa-bbbb-cccc-dddddddddddd" for i in range(NUM_NODES)]
    HOST = "127.0.0.1"

    peers: dict[str, tuple[str, int]] = {
        nid: (HOST, BASE_PORT + i) for i, nid in enumerate(NODE_IDS)
    }

    # Build a ShardAssignment for this fleet (64 experts, equal split).
    caps = [NodeCapability(node_id=nid, storage_bandwidth_mbps=4000) for nid in NODE_IDS]
    assignment: ShardAssignment = compute_assignment(caps, 64)

    # Each node's "contribution" is a 4-byte big-endian integer.
    contributions = [10, 20, 30, 40]
    EXPECTED_SUM = sum(contributions)  # 100

    # ── Create per-node servers + GangSync instances ────────────────────
    servers: list[RpcServer] = []
    clients: list[RpcClient] = []
    gangs: list[GangSync] = []

    for i, nid in enumerate(NODE_IDS):
        port = BASE_PORT + i
        client = RpcClient(own_node_id=nid)
        clients.append(client)

        gang = GangSync(
            own_node_id=nid,
            rpc_client=client,
            peers=peers,
        )
        gangs.append(gang)

        server = RpcServer(
            own_node_id=nid,
            port=port,
            handler=gang.handle_frame,
            bind_ip=HOST,
        )
        servers.append(server)

    # Start all servers.
    for srv in servers:
        await srv.start()
    await asyncio.sleep(0.1)  # let listeners bind

    try:
        # ── Happy-path demo ────────────────────────────────────────────
        print("=" * 72)
        print("  Gang-sync demo — 4 nodes, integer addition ring-reduce")
        print("=" * 72)
        print(f"  Contributions: {contributions}")
        print(f"  Expected sum:   {EXPECTED_SUM}\n")

        def int_combine(acc: bytes, local: bytes) -> bytes:
            """Elementwise integer addition of two big-endian int32 values."""
            a = int.from_bytes(acc, "big", signed=True)
            b = int.from_bytes(local, "big", signed=True)
            return (a + b).to_bytes(4, "big", signed=True)

        async def run_one(node_index: int) -> bytes:
            contrib = contributions[node_index].to_bytes(4, "big", signed=True)
            return await gangs[node_index].ring_reduce(
                assignment=assignment,
                layer_id=1,
                local_contribution=contrib,
                combine=int_combine,
                timeout=5.0,
            )

        t0 = time.monotonic()
        results = await asyncio.gather(*(run_one(i) for i in range(NUM_NODES)))
        elapsed = time.monotonic() - t0

        # All results must be identical and correct.
        sums = [int.from_bytes(r, "big", signed=True) for r in results]
        all_same = len(set(sums)) == 1
        correct = sums[0] == EXPECTED_SUM

        print(f"  Results: {sums}")
        print(f"  All nodes agree: {'✓' if all_same else '✗ FAIL'}")
        print(f"  Sum correct ({EXPECTED_SUM}): {'✓' if correct else '✗ FAIL'}")
        print(f"  Round-trip time: {elapsed*1000:.1f} ms")
        print()

        if not all_same or not correct:
            print("  HAPPY-PATH DEMO FAILED — aborting")
            return

        # ── Second call (different layer_id) confirms state cleanup ─────
        print("  ── Second layer (confirms state cleanup) ──")
        contribs2 = [1, 2, 3, 4]
        EXPECTED_SUM2 = sum(contribs2)

        async def run_one_2(node_index: int) -> bytes:
            contrib = contribs2[node_index].to_bytes(4, "big", signed=True)
            return await gangs[node_index].ring_reduce(
                assignment=assignment,
                layer_id=2,
                local_contribution=contrib,
                combine=int_combine,
                timeout=5.0,
            )

        results2 = await asyncio.gather(*(run_one_2(i) for i in range(NUM_NODES)))
        sums2 = [int.from_bytes(r, "big", signed=True) for r in results2]
        ok2 = len(set(sums2)) == 1 and sums2[0] == EXPECTED_SUM2
        print(f"  Results: {sums2}")
        print(f"  Sum correct ({EXPECTED_SUM2}): {'✓' if ok2 else '✗ FAIL'}")
        print()

        # ── Timeout demo: kill node 2 (last node), run ring with 3 ─────
        print("  ── Timeout demo (node 2 absent) ──")
        # Stop node 2's server — it won't receive the ring-hop, so
        # node 1's forward will time out at the TCP level, and nodes 0
        # and 1 will time out waiting for fan-out.
        await servers[2].stop()
        # Also close node 2's client so it doesn't accept new connections.
        await clients[2].close()

        async def run_timeout(node_index: int) -> bytes:
            contrib = contributions[node_index].to_bytes(4, "big", signed=True)
            return await gangs[node_index].ring_reduce(
                assignment=assignment,
                layer_id=3,
                local_contribution=contrib,
                combine=int_combine,
                timeout=2.0,  # short timeout for quick demo
            )

        # Run nodes 0, 1, and 3 only.  Node 2 is dead.
        t0 = time.monotonic()
        gathered = await asyncio.gather(
            run_timeout(0),
            run_timeout(1),
            run_timeout(3),
            return_exceptions=True,
        )
        elapsed = time.monotonic() - t0

        timeout_count = sum(1 for r in gathered if isinstance(r, GangSyncTimeout))
        error_count = sum(1 for r in gathered if isinstance(r, GangSyncError) and not isinstance(r, GangSyncTimeout))
        success_count = sum(1 for r in gathered if isinstance(r, bytes))

        print(f"  Node 0 got: {type(gathered[0]).__name__}")
        print(f"  Node 1 got: {type(gathered[1]).__name__}")
        print(f"  Node 3 got: {type(gathered[2]).__name__}")
        print(f"  Timeouts: {timeout_count}, Errors: {error_count}, Success: {success_count}")
        print(f"  Elapsed: {elapsed*1000:.1f} ms (should be near 2 s timeout)")
        print(f"  No hangs: {'✓' if elapsed < 5.0 else '✗ HUNG'}")

        # Every surviving node should have timed out or errored.
        all_failed = success_count == 0
        print(f"  All survivors failed cleanly: {'✓' if all_failed else '✗ (got a result from dead ring)'}")

        # ── Fleet-hash mismatch demo ────────────────────────────────────
        print("\n  ── Fleet-hash mismatch demo ──")
        # Build a different assignment (same nodes, different bandwidth).
        caps_mismatch = [
            NodeCapability(node_id=NODE_IDS[0], storage_bandwidth_mbps=1000),
            NodeCapability(node_id=NODE_IDS[1], storage_bandwidth_mbps=2000),
            NodeCapability(node_id=NODE_IDS[2], storage_bandwidth_mbps=3000),
            NodeCapability(node_id=NODE_IDS[3], storage_bandwidth_mbps=4000),
        ]
        assignment_mismatch = compute_assignment(caps_mismatch, 64)
        # Restart node 2 so the ring can form, but give node 3 the WRONG assignment.
        clients[2] = RpcClient(own_node_id=NODE_IDS[2])
        gangs[2] = GangSync(own_node_id=NODE_IDS[2], rpc_client=clients[2], peers=peers)
        servers[2] = RpcServer(
            own_node_id=NODE_IDS[2],
            port=BASE_PORT + 2,
            handler=gangs[2].handle_frame,
            bind_ip=HOST,
        )
        await servers[2].start()
        await asyncio.sleep(0.1)

        async def run_mismatch(node_index: int) -> bytes:
            contrib = contributions[node_index].to_bytes(4, "big", signed=True)
            # Node 3 uses the wrong assignment!
            assn = assignment_mismatch if node_index == 3 else assignment
            return await gangs[node_index].ring_reduce(
                assignment=assn,
                layer_id=4,
                local_contribution=contrib,
                combine=int_combine,
                timeout=3.0,
            )

        gathered2 = await asyncio.gather(
            *(run_mismatch(i) for i in range(NUM_NODES)),
            return_exceptions=True,
        )

        mismatch_errors = sum(
            1 for r in gathered2
            if isinstance(r, GangSyncError) and not isinstance(r, GangSyncTimeout)
        )
        print(f"  Fleet-hash mismatch detected: {'✓' if mismatch_errors > 0 else '✗ NOT DETECTED'}")
        for i, r in enumerate(gathered2):
            if isinstance(r, Exception):
                print(f"    Node {i}: {type(r).__name__}: {r}")
            else:
                print(f"    Node {i}: result={int.from_bytes(r, 'big', signed=True) if isinstance(r, bytes) else r}")

        print("\n" + "=" * 72)
        print("  Demo complete.")
        print("=" * 72)

    finally:
        for c in clients:
            try:
                await c.close()
            except Exception:
                pass
        for srv in servers:
            try:
                await srv.stop()
            except Exception:
                pass


def main() -> None:
    """Entry point: run the demo."""
    asyncio.run(_demo())


if __name__ == "__main__":
    main()
