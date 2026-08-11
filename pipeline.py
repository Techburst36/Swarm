#!/usr/bin/env python3
"""
pipeline.py — Dense pipeline execution mode for the Swarm distributed scheduler.

Layer 4 of the software stack.  This module is the complement to gang_sync.py:
instead of every node working the same layer on different experts (MoE gang
mode), each node owns a contiguous range of layers.  An activation tensor is
handed down the chain — node A computes layers 0–9, passes its output to
node B, which computes layers 10–19, and so on.  The last node produces the
final output and returns it to whichever node originated the request.

Why sharding.py is reused for layer assignment
-----------------------------------------------
Dense pipelining assigns layers to nodes in contiguous blocks, weighted by
measured storage bandwidth.  ``compute_assignment(nodes, num_layers)`` does
exactly that: largest-remainder apportionment → contiguous block assignment
in sorted node_id order.  Each "expert slot index" in the resulting
``ShardAssignment`` is reinterpreted as a "layer index."  This is not a hack —
the apportionment problem is identical, and reusing it means the
coordinator-free determinism property (every node independently computes the
same assignment from the same fleet view) comes along for free.  Do not write
a second apportionment implementation.

What this module does NOT do
-----------------------------
- **No tensor/numpy assumptions.**  Payload is opaque ``bytes``.  The caller
  supplies a ``compute_stage(activation, layer_start, layer_end) -> bytes``
  callback.  This module never inspects the activation's numeric format.
- **No microbatching or intra-request pipeline parallelism.**  Multiple
  independent requests may be in flight at different stages simultaneously
  (that falls out naturally and is useful).  Splitting a single request into
  microbatches to keep every node busy is real pipeline-parallel complexity
  that the batch-1 target regime does not need yet.
- **No failover.**  A node dying mid-request fails that request loudly and
  specifically (naming which stage/node failed), not rerouting around it.
  failover.py handles membership changes separately.
- **No coordinator/leader election.**  Every node derives the same pipeline
  order independently from the same ``ShardAssignment``.

Message framing
---------------
Pipeline messages use rpc.py ``MessageType.ACTIVATION``.  The inner framing
(prepended to the caller's payload):

    ┌──────────────┬──────────────┬──────────┬────────────────────┬───────────────┐
    │  request_id  │ stage_index  │ msg_type │ fleet_hash_raw     │ originator_len│
    │  uint32 BE   │  uint32 BE   │ uint8    │  32 bytes (binary) │  uint16 BE    │
    │    4 B       │    4 B       │  1 B     │    32 B            │   2 B         │
    └──────────────┴──────────────┴──────────┴────────────────────┴───────────────┘
    └─ 43 bytes fixed ───────────────────────────────────────────────────────────┘
    ... followed by originator_node_id (UTF-8, originator_len bytes) ...
    ... followed by payload (activation bytes) ...

``msg_type`` values:
  - 0x01  PIPELINE_FORWARD  — activation being forwarded to the next stage
  - 0x02  PIPELINE_RESULT   — final result sent back to the originating node
  - 0x03  PIPELINE_ERROR    — error propagated back to the originating node

``fleet_hash_raw`` is ``bytes.fromhex(assignment.fleet_hash)``, i.e. the
raw 32-byte SHA-256.  A node rejects any message whose fleet_hash does not
match its own ``ShardAssignment``, preventing cross-talk between two nodes
that computed sharding from different fleet snapshots.

The flow, precisely
--------------------
1. Any node may originate a request by calling ``run_pipeline()``.
2. If the originating node owns stage 0, it computes its own layer range
   first; otherwise it forwards the initial activation to whichever node
   owns stage 0.
3. Each stage: receive activation → call ``compute_stage()`` for that node's
   layer range → forward the result to the next stage's node.
4. The last stage sends its output directly back to the originating node
   (not back down the chain).
5. ``run_pipeline`` on the originating node returns the final output once
   it arrives.  Every other node's participation happens through its
   inbound message handler, not through a blocking call.

Design rules (do not break)
----------------------------
- Python 3.11+, standard library only, plus the four existing project modules.
- All background tasks strongly referenced (asyncio.create_task pitfall —
  this makes five times across this project).
- No coordinator, no leader election, no voting.
- Pipeline order is deterministic, derived from
  ``sorted(assignment.node_counts.keys())`` — the same ordering sharding.py
  and gang_sync.py already use.  This is load-bearing.
- Timeout on every wait; raise PipelineTimeout, never hang.
- Bounded in-flight state; reject over-cap requests loudly.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import struct
import time
from collections.abc import Callable
from typing import Any

from rpc import Frame, MessageType, RpcClient, RpcConnection, RpcError
from sharding import ShardAssignment

logger = logging.getLogger("swarm.pipeline")

# ── Constants ──────────────────────────────────────────────────────────────────

# Pipeline message types embedded in our inner framing (NOT rpc.py MessageType).
PIPELINE_FORWARD: int = 0x01
PIPELINE_RESULT: int = 0x02
PIPELINE_ERROR: int = 0x03

# Inner framing layout (does NOT include the caller's payload or the
# variable-length originator_node_id — those follow the fixed header).
# Format: "!IIB32sH" = request_id(u32), stage_index(u32), msg_type(u8),
#                      fleet_hash_raw(32s), originator_len(u16)
_PIPELINE_FRAME_HEADER_FORMAT: str = "!IIB32sH"
_PIPELINE_FRAME_HEADER_SIZE: int = struct.calcsize(_PIPELINE_FRAME_HEADER_FORMAT)  # 43

# Default timeout for waiting on pipeline completion.
DEFAULT_PIPELINE_TIMEOUT: float = 60.0

# Sanity cap on a single pipeline payload.  rpc.py enforces its own
# max_frame_bytes at the transport layer; this is a second, cheaper check
# at the application layer so an oversized activation from a compute_stage()
# bug is caught at the first hop, not amplified down the whole chain.
DEFAULT_MAX_PAYLOAD_BYTES: int = 64 * 1024 * 1024  # 64 MB

# Maximum concurrently tracked requests.  Rejecting over this cap prevents
# a slow memory leak from orphaned or never-completed requests.
DEFAULT_MAX_CONCURRENT_REQUESTS: int = 64

# Maximum early-buffer entries.  A PIPELINE_RESULT may arrive before
# run_pipeline() sets up state for its request_id; we buffer it here.
# Bounded to prevent unbounded growth from stale results.
_MAX_EARLY_BUFFER: int = 16


# ── Exceptions ─────────────────────────────────────────────────────────────────


class PipelineError(Exception):
    """Raised for pipeline failures that are not timeouts.

    Covers: fleet-hash mismatch (two nodes disagree on sharding),
    send failure during forwarding, invalid stage index, payload
    size cap exceeded, duplicate request_id, and internal protocol
    violations.
    """


class PipelineTimeout(PipelineError):
    """A pipeline operation timed out waiting for the final result.

    Attributes
    ----------
    request_id:
        The request that timed out.
    stage_description:
        Human description of which stage or node was expected.
    """

    def __init__(self, request_id: int, stage_description: str) -> None:
        self.request_id = request_id
        self.stage_description = stage_description
        super().__init__(
            f"Request {request_id}: timed out waiting for {stage_description}"
        )


# ── Pipeline message framing ──────────────────────────────────────────────────


def _encode_pipeline_message(
    *,
    request_id: int,
    stage_index: int,
    msg_type: int,
    fleet_hash: str,
    originator_node_id: str,
    payload: bytes,
) -> bytes:
    """Encode a pipeline message: inner framing header + originator + payload.

    Parameters
    ----------
    request_id:
        Unique request identifier.  Must match across all nodes involved.
    stage_index:
        Which stage in the pipeline this activation is destined for.
    msg_type:
        ``PIPELINE_FORWARD``, ``PIPELINE_RESULT``, or ``PIPELINE_ERROR``.
    fleet_hash:
        64-character hex SHA-256 from ``ShardAssignment.fleet_hash``.
        Converted to raw 32 bytes on the wire.
    originator_node_id:
        Stable UUID of the node that originated the request.  Needed so
        the last stage knows where to send the result, and so any node
        can route errors back to the originator.
    payload:
        The activation bytes (or error message for PIPELINE_ERROR).
    """
    fleet_hash_raw = bytes.fromhex(fleet_hash)
    if len(fleet_hash_raw) != 32:
        raise ValueError(
            f"fleet_hash must be a 64-char hex string (32 bytes raw), "
            f"got {len(fleet_hash)} chars"
        )
    originator_bytes = originator_node_id.encode("utf-8")
    if len(originator_bytes) > 65535:
        raise ValueError(
            f"originator_node_id is {len(originator_bytes)} bytes, "
            f"over the 65535-byte cap"
        )
    header = struct.pack(
        _PIPELINE_FRAME_HEADER_FORMAT,
        request_id,
        stage_index,
        msg_type,
        fleet_hash_raw,
        len(originator_bytes),
    )
    return header + originator_bytes + payload


def _decode_pipeline_message(
    data: bytes,
) -> tuple[int, int, int, str, str, bytes] | None:
    """Decode a pipeline message from raw bytes.

    Returns ``(request_id, stage_index, msg_type, fleet_hash,
    originator_node_id, payload)`` on success, or ``None`` if the data is
    too short to contain a valid header or the originator field.

    The returned ``fleet_hash`` is the 64-character hex string, reconstructed
    from the 32 raw bytes on the wire, for direct comparison with
    ``ShardAssignment.fleet_hash``.
    """
    if len(data) < _PIPELINE_FRAME_HEADER_SIZE:
        return None
    request_id, stage_index, msg_type, fleet_hash_raw, originator_len = struct.unpack(
        _PIPELINE_FRAME_HEADER_FORMAT,
        data[:_PIPELINE_FRAME_HEADER_SIZE],
    )
    fleet_hash = fleet_hash_raw.hex()
    originator_start = _PIPELINE_FRAME_HEADER_SIZE
    originator_end = originator_start + originator_len
    if len(data) < originator_end:
        return None
    originator_node_id = data[originator_start:originator_end].decode("utf-8")
    payload = data[originator_end:]
    return request_id, stage_index, msg_type, fleet_hash, originator_node_id, payload


# ── Internal state ─────────────────────────────────────────────────────────────


@dataclasses.dataclass
class _PipelineState:
    """Per-request state tracked on the originating node.

    Created by ``run_pipeline()``, consumed by the handler when a
    PIPELINE_RESULT or PIPELINE_ERROR arrives, cleaned up when the
    call returns (success or failure).
    """

    request_id: int
    fleet_hash: str
    pipeline_order: list[str]  # node_ids, sorted, nodes with >0 layers only
    own_node_id: str
    result_future: asyncio.Future[bytes]


# ── PipelineCoordinator ────────────────────────────────────────────────────────


class PipelineCoordinator:
    """Per-node coordinator for dense pipeline execution.

    One instance per node.  Owns the handler registered with the node's
    ``RpcServer`` and orchestrates pipeline runs.

    Parameters
    ----------
    own_node_id:
        Stable UUID of *this* node.  Must appear in every
        ``ShardAssignment`` passed to ``run_pipeline()`` and
        ``set_assignment()``.
    rpc_client:
        An ``RpcClient`` connected (or connectable) to all peer nodes.
        Used to forward activations and send results.
    peers:
        Mapping from ``node_id`` to ``(host, port)`` for every node in the
        fleet.  Needed because ``ShardAssignment`` carries only
        ``node_id``, not network addresses.
    compute_stage:
        Default callback ``(activation, layer_start, layer_end) -> bytes``.
        May be overridden per-request via ``run_pipeline()``.  If neither
        the constructor nor ``run_pipeline`` provides one, calls to
        ``run_pipeline`` raise ``PipelineError``.  Both sync and async
        callbacks are accepted.
    assignment:
        Optional initial ``ShardAssignment``.  Must be set (via constructor
        or ``set_assignment()``) before the handler can process messages.
    max_concurrent_requests:
        Maximum number of requests tracked simultaneously (default 64).
    max_payload_bytes:
        Maximum activation size in bytes forwarded between stages
        (default 64 MB).  A ``compute_stage()`` that returns a result
        over this cap is caught at the first hop.
    """

    def __init__(
        self,
        *,
        own_node_id: str,
        rpc_client: RpcClient,
        peers: dict[str, tuple[str, int]],
        compute_stage: Callable[[bytes, int, int], Any] | None = None,
        assignment: ShardAssignment | None = None,
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        self._own_node_id = own_node_id
        self._rpc_client = rpc_client
        self._peers = peers
        self._compute_stage = compute_stage
        self._assignment: ShardAssignment | None = assignment
        self._max_payload_bytes = max_payload_bytes

        # request_id → _PipelineState for in-progress requests.
        self._pending: dict[int, _PipelineState] = {}

        # Concurrency cap — reject over this limit rather than growing
        # without bound.  A node that accumulates orphaned request state
        # has a slow memory leak that only shows up after days of uptime.
        self._max_concurrent = max_concurrent_requests

        # Early-arrival buffer.  A PIPELINE_RESULT may land before
        # run_pipeline() has set up state (e.g. asyncio.gather launch).
        # Buffer one message per request_id so it can be replayed.
        # request_id → (msg_type, fleet_hash, originator_node_id, payload)
        self._early_buffer: dict[int, tuple[int, str, str, bytes]] = {}
        # Bounded to _MAX_EARLY_BUFFER entries.

        # Strong references to background tasks (see module docstring for why).
        self._background_tasks: set[asyncio.Task[Any]] = set()

    # ── Public API ─────────────────────────────────────────────────────────

    def set_assignment(self, assignment: ShardAssignment) -> None:
        """Update the current pipeline assignment.

        Called by the higher layer whenever the fleet view changes and a
        new sharding assignment is computed.  The handler validates inbound
        messages against this assignment's fleet_hash.

        All nodes in the fleet should receive the same assignment (via
        sharding.py's coordinator-free determinism) before any pipeline
        requests are dispatched.
        """
        self._assignment = assignment
        logger.info(
            "Pipeline assignment updated: fleet_hash=%s…, %d nodes",
            assignment.fleet_hash[:16],
            len(assignment.node_counts),
        )

    def _get_pipeline_order(self, assignment: ShardAssignment) -> list[str]:
        """Compute the pipeline order from a ShardAssignment.

        Nodes with zero layers are excluded — they have no work to do in
        the pipeline.  Order is ``sorted(node_counts.keys())``, the same
        ordering sharding.py and gang_sync.py use.  This is load-bearing:
        two nodes that disagree on pipeline order will route activations
        to the wrong peers.
        """
        return [
            nid
            for nid in sorted(assignment.node_counts.keys())
            if assignment.node_counts.get(nid, 0) > 0
        ]

    def layer_range_for_node(
        self, assignment: ShardAssignment, node_id: str
    ) -> tuple[int, int]:
        """Return the ``(layer_start, layer_end)`` half-open interval for *node_id*.

        ``layer_start`` is the first layer index this node owns (inclusive).
        ``layer_end`` is one past the last layer (exclusive), suitable for
        use with Python's ``range()``.

        Nodes with no layers return ``(0, 0)``.

        This is a convenience for the caller to know which layers it is
        responsible for computing.  It is not called internally — the
        stage handler derives layer ranges directly from the assignment.
        """
        experts = assignment.node_experts.get(node_id, [])
        if not experts:
            return (0, 0)
        return (experts[0], experts[-1] + 1)

    async def run_pipeline(
        self,
        *,
        assignment: ShardAssignment,
        request_id: int,
        initial_activation: bytes,
        compute_stage: Callable[[bytes, int, int], Any] | None = None,
        timeout: float = DEFAULT_PIPELINE_TIMEOUT,
    ) -> bytes:
        """Run one pipeline pass for *request_id*.

        Blocks until the final stage produces a result and sends it back,
        or until *timeout* seconds elapse.

        Parameters
        ----------
        assignment:
            The current ``ShardAssignment``.  Pipeline order is derived from
            ``sorted(assignment.node_counts.keys())``.  ``fleet_hash`` is
            embedded in every outbound message and checked on every inbound
            one.  This assignment is also stored for the handler to use on
            inbound messages for this request.
        request_id:
            Unique integer identifier for this request.  Must not duplicate
            an in-flight request on this node.  Carried in every pipeline
            message so concurrent requests are never confused.
        initial_activation:
            The input activation bytes for the first layer (stage 0).
        compute_stage:
            Callback ``(activation, layer_start, layer_end) -> bytes`` for
            this request.  If provided, overrides the constructor's default.
            Both sync and async callbacks are accepted (await if needed).
        timeout:
            Seconds to wait for the final result before raising
            ``PipelineTimeout``.

        Returns
        -------
        bytes
            The final output activation from the last pipeline stage.

        Raises
        ------
        ValueError
            If ``own_node_id`` is not in *assignment*, or the assignment
            has not been set (and *assignment* is ``None``).
        PipelineError
            If *request_id* duplicates an in-flight request, or
            ``compute_stage`` is not available.
        PipelineTimeout
            If the final result does not arrive within *timeout*.
        """
        if assignment is None:
            assignment = self._assignment
        if assignment is None:
            raise PipelineError(
                "No ShardAssignment available — call set_assignment() "
                "or pass assignment= to run_pipeline()"
            )
        # Store for the handler to use on inbound messages for this fleet.
        if self._assignment is None:
            self._assignment = assignment

        if self._own_node_id not in assignment.node_counts:
            raise ValueError(
                f"own_node_id {self._own_node_id[:8]}… is not in the "
                f"ShardAssignment (fleet has {len(assignment.node_counts)} nodes)"
            )

        # ── Choose compute_stage callback ─────────────────────────────
        cb = compute_stage if compute_stage is not None else self._compute_stage
        if cb is None:
            raise PipelineError(
                "No compute_stage callback available — provide one to "
                "the PipelineCoordinator constructor or to run_pipeline()"
            )

        # ── Compute pipeline order ────────────────────────────────────
        pipeline_order = self._get_pipeline_order(assignment)
        if len(set(pipeline_order)) != len(pipeline_order):
            raise PipelineError(
                f"Request {request_id}: ShardAssignment contains duplicate "
                f"node_ids; pipeline order would be ambiguous"
            )

        # Single-node pipeline: compute and return immediately.
        if len(pipeline_order) == 1 and pipeline_order[0] == self._own_node_id:
            layer_start, layer_end = self.layer_range_for_node(
                assignment, self._own_node_id
            )
            result = cb(initial_activation, layer_start, layer_end)
            if asyncio.iscoroutine(result):
                result = await result
            if not isinstance(result, (bytes, bytearray)):
                raise PipelineError(
                    f"Request {request_id}: compute_stage must return bytes, "
                    f"got {type(result).__name__}"
                )
            if len(result) > self._max_payload_bytes:
                raise PipelineError(
                    f"Request {request_id}: compute_stage returned "
                    f"{len(result)} bytes, over the "
                    f"{self._max_payload_bytes} byte cap"
                )
            return bytes(result)

        own_position = self._own_position_in_pipeline(pipeline_order)
        is_stage_0 = own_position == 0

        # ── Duplicate request_id guard ────────────────────────────────
        if request_id in self._pending:
            raise PipelineError(
                f"Request {request_id}: a run_pipeline for this request_id "
                f"is already in progress on this node. Concurrent calls "
                f"with the same request_id are not supported — each "
                f"request must complete before its id is reused."
            )

        # ── Concurrency cap ───────────────────────────────────────────
        if len(self._pending) >= self._max_concurrent:
            raise PipelineError(
                f"Too many concurrent pipeline requests ({len(self._pending)}); "
                f"cap is {self._max_concurrent}.  Rejecting request {request_id}."
            )

        # ── Create state ──────────────────────────────────────────────
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[bytes] = loop.create_future()

        state = _PipelineState(
            request_id=request_id,
            fleet_hash=assignment.fleet_hash,
            pipeline_order=pipeline_order,
            own_node_id=self._own_node_id,
            result_future=result_future,
        )
        self._pending[request_id] = state

        try:
            # ── Drain early-buffer if a result arrived before we were ready ─
            early = self._early_buffer.pop(request_id, None)
            if early is not None:
                msg_type, fleet_hash_in_msg, originator, payload = early
                self._handle_pipeline_message(
                    msg_type, fleet_hash_in_msg, originator, payload, request_id
                )

            # ── Initiate the pipeline ─────────────────────────────────
            if is_stage_0:
                # Originator owns stage 0: compute locally, then forward.
                layer_start, layer_end = self.layer_range_for_node(
                    assignment, self._own_node_id
                )
                try:
                    stage_result = cb(initial_activation, layer_start, layer_end)
                    if asyncio.iscoroutine(stage_result):
                        stage_result = await stage_result
                    if not isinstance(stage_result, (bytes, bytearray)):
                        raise PipelineError(
                            f"Request {request_id}: compute_stage must return "
                            f"bytes, got {type(stage_result).__name__}"
                        )
                    stage_result = bytes(stage_result)
                except PipelineError:
                    raise
                except Exception as exc:
                    raise PipelineError(
                        f"Request {request_id}: compute_stage for "
                        f"stage 0 (layers [{layer_start}, {layer_end})) "
                        f"raised: {exc}"
                    ) from exc

                if len(stage_result) > self._max_payload_bytes:
                    raise PipelineError(
                        f"Request {request_id}: compute_stage returned "
                        f"{len(stage_result)} bytes, over the "
                        f"{self._max_payload_bytes} byte cap"
                    )

                # Forward to stage 1 as a background task.
                task = asyncio.create_task(
                    self._forward_to_stage(
                        request_id=request_id,
                        source_stage=0,
                        next_stage=1,
                        fleet_hash=assignment.fleet_hash,
                        originator_node_id=self._own_node_id,
                        payload=stage_result,
                        pipeline_order=pipeline_order,
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            else:
                # Originator does NOT own stage 0: forward initial activation
                # to the node that does.
                stage_0_node = pipeline_order[0]
                stage_0_addr = self._peers.get(stage_0_node)
                if stage_0_addr is None:
                    raise PipelineError(
                        f"Request {request_id}: no peer address for "
                        f"stage-0 node {stage_0_node[:8]}…"
                    )
                logger.debug(
                    "Request %d: forwarding initial activation to stage 0 "
                    "(%s…)",
                    request_id,
                    stage_0_node[:8],
                )
                encoded = _encode_pipeline_message(
                    request_id=request_id,
                    stage_index=0,
                    msg_type=PIPELINE_FORWARD,
                    fleet_hash=assignment.fleet_hash,
                    originator_node_id=self._own_node_id,
                    payload=initial_activation,
                )
                try:
                    await self._rpc_client.send(
                        stage_0_addr,
                        MessageType.ACTIVATION,
                        encoded,
                    )
                except (RpcError, ConnectionError, OSError) as exc:
                    raise PipelineError(
                        f"Request {request_id}: failed to send initial "
                        f"activation to stage-0 node {stage_0_node[:8]}…: {exc}"
                    ) from exc

            # ── Wait for the result ────────────────────────────────────
            stage_desc = f"PIPELINE_RESULT for request {request_id}"
            try:
                return await asyncio.wait_for(result_future, timeout=timeout)
            except asyncio.TimeoutError:
                raise PipelineTimeout(request_id, stage_desc) from None

        finally:
            self._cleanup_state(request_id)

    # ── Handler (register with RpcServer) ──────────────────────────────────

    async def handle_frame(self, conn: RpcConnection, frame: Frame) -> None:
        """Handler for inbound pipeline messages.  Register with ``RpcServer``.

        Usage::

            pipe = PipelineCoordinator(own_node_id=..., rpc_client=..., peers=...)
            server = RpcServer(own_node_id=..., port=..., handler=pipe.handle_frame)

        Only ``MessageType.ACTIVATION`` frames are inspected.  All other
        message types are silently ignored (they belong to other layers of
        the stack).
        """
        if frame.msg_type != MessageType.ACTIVATION:
            return

        if frame.payload is None:
            logger.debug("ACTIVATION frame with no payload — ignoring")
            return

        decoded = _decode_pipeline_message(frame.payload)
        if decoded is None:
            # Payload too short to be one of ours — not a pipeline message.
            return

        request_id, stage_index, msg_type, fleet_hash_in_msg, originator, payload = (
            decoded
        )

        if msg_type == PIPELINE_FORWARD:
            # Spawn as a background task so a slow compute_stage doesn't
            # block the handler slot.  Multiple requests arriving at the
            # same stage concurrently will run in parallel — that's the
            # caller's responsibility to make safe.
            task = asyncio.create_task(
                self._process_stage(
                    request_id=request_id,
                    stage_index=stage_index,
                    fleet_hash=fleet_hash_in_msg,
                    originator_node_id=originator,
                    payload=payload,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        elif msg_type in (PIPELINE_RESULT, PIPELINE_ERROR):
            self._handle_pipeline_message(
                msg_type, fleet_hash_in_msg, originator, payload, request_id
            )
        else:
            logger.warning(
                "Unknown pipeline msg_type 0x%02X for request %d — ignoring",
                msg_type,
                request_id,
            )

    # ── Internal: message handling (synchronous, no awaits) ────────────────

    def _handle_pipeline_message(
        self,
        msg_type: int,
        fleet_hash_in_msg: str,
        originator: str,
        payload: bytes,
        request_id: int,
    ) -> None:
        """Process a PIPELINE_RESULT or PIPELINE_ERROR against pending state.

        Synchronous (no awaits) so it can be called from either the async
        handler or the early-buffer drain path without double-await confusion.
        """
        state = self._pending.get(request_id)
        if state is None:
            # No in-progress run_pipeline for this request_id.  Buffer in
            # case run_pipeline hasn't been called yet (concurrent launch).
            # Enforce a bound so stale messages don't leak memory.
            if len(self._early_buffer) >= _MAX_EARLY_BUFFER:
                oldest = min(self._early_buffer.keys())
                logger.debug(
                    "Early buffer full (%d entries) — evicting request %d",
                    len(self._early_buffer),
                    oldest,
                )
                del self._early_buffer[oldest]
            logger.debug(
                "Request %d: message arrived before run_pipeline — buffering",
                request_id,
            )
            self._early_buffer[request_id] = (
                msg_type,
                fleet_hash_in_msg,
                originator,
                payload,
            )
            return

        # ── Fleet-hash gate ───────────────────────────────────────────
        if fleet_hash_in_msg != state.fleet_hash:
            msg = (
                f"Request {request_id}: fleet_hash mismatch from peer. "
                f"Expected {state.fleet_hash[:16]}…, "
                f"got {fleet_hash_in_msg[:16]}…. "
                f"This node computed sharding from a different fleet "
                f"snapshot than the sending peer."
            )
            logger.error(msg)
            if not state.result_future.done():
                state.result_future.set_exception(PipelineError(msg))
            return

        if msg_type == PIPELINE_RESULT:
            if not state.result_future.done():
                state.result_future.set_result(payload)
        elif msg_type == PIPELINE_ERROR:
            error_text = payload.decode("utf-8", errors="replace")
            if not state.result_future.done():
                state.result_future.set_exception(
                    PipelineError(
                        f"Request {request_id}: pipeline error from peer: "
                        f"{error_text}"
                    )
                )

    # ── Internal: stage processing (background task) ───────────────────────

    async def _process_stage(
        self,
        *,
        request_id: int,
        stage_index: int,
        fleet_hash: str,
        originator_node_id: str,
        payload: bytes,
    ) -> None:
        """Process one pipeline stage: compute, then forward or return result.

        Runs as a background task spawned from the handler.  On any failure,
        propagates an error back to the originator (unless this node IS the
        originator, in which case the future is resolved directly).
        """
        assignment = self._assignment
        if assignment is None:
            await self._send_pipeline_error(
                request_id=request_id,
                fleet_hash=fleet_hash,
                originator_node_id=originator_node_id,
                error_text="No ShardAssignment set on this node",
                pipeline_order=[],
            )
            return

        # ── Fleet-hash gate ───────────────────────────────────────────
        if fleet_hash != assignment.fleet_hash:
            await self._send_pipeline_error(
                request_id=request_id,
                fleet_hash=fleet_hash,
                originator_node_id=originator_node_id,
                error_text=(
                    f"fleet_hash mismatch: expected {assignment.fleet_hash[:16]}…, "
                    f"got {fleet_hash[:16]}…"
                ),
                pipeline_order=[],
            )
            return

        pipeline_order = self._get_pipeline_order(assignment)

        # Validate stage_index.
        if stage_index >= len(pipeline_order):
            await self._send_pipeline_error(
                request_id=request_id,
                fleet_hash=fleet_hash,
                originator_node_id=originator_node_id,
                error_text=(
                    f"stage_index {stage_index} out of range "
                    f"(pipeline has {len(pipeline_order)} stages)"
                ),
                pipeline_order=pipeline_order,
            )
            return

        node_id = pipeline_order[stage_index]
        if node_id != self._own_node_id:
            await self._send_pipeline_error(
                request_id=request_id,
                fleet_hash=fleet_hash,
                originator_node_id=originator_node_id,
                error_text=(
                    f"PIPELINE_FORWARD for stage {stage_index} arrived at "
                    f"{self._own_node_id[:8]}… but stage {stage_index} is "
                    f"{node_id[:8]}… — pipeline order mismatch"
                ),
                pipeline_order=pipeline_order,
            )
            return

        # ── Call compute_stage ────────────────────────────────────────
        cb = self._compute_stage
        if cb is None:
            await self._send_pipeline_error(
                request_id=request_id,
                fleet_hash=fleet_hash,
                originator_node_id=originator_node_id,
                error_text="No compute_stage callback configured",
                pipeline_order=pipeline_order,
            )
            return

        layer_start, layer_end = self.layer_range_for_node(assignment, node_id)

        try:
            result = cb(payload, layer_start, layer_end)
            if asyncio.iscoroutine(result):
                result = await result
            if not isinstance(result, (bytes, bytearray)):
                raise PipelineError(
                    f"compute_stage must return bytes, got {type(result).__name__}"
                )
            result = bytes(result)
        except PipelineError as exc:
            await self._send_pipeline_error(
                request_id=request_id,
                fleet_hash=fleet_hash,
                originator_node_id=originator_node_id,
                error_text=str(exc),
                pipeline_order=pipeline_order,
            )
            return
        except Exception as exc:
            await self._send_pipeline_error(
                request_id=request_id,
                fleet_hash=fleet_hash,
                originator_node_id=originator_node_id,
                error_text=(
                    f"compute_stage for layers [{layer_start}, {layer_end}) "
                    f"raised: {exc}"
                ),
                pipeline_order=pipeline_order,
            )
            return

        # ── Payload size cap ──────────────────────────────────────────
        if len(result) > self._max_payload_bytes:
            await self._send_pipeline_error(
                request_id=request_id,
                fleet_hash=fleet_hash,
                originator_node_id=originator_node_id,
                error_text=(
                    f"compute_stage returned {len(result)} bytes, over the "
                    f"{self._max_payload_bytes} byte cap"
                ),
                pipeline_order=pipeline_order,
            )
            return

        # ── Forward or return result ──────────────────────────────────
        is_last = stage_index == len(pipeline_order) - 1

        if is_last:
            # Last stage: send result back to originator.
            if originator_node_id == self._own_node_id:
                # Originator IS the last stage — resolve locally.
                state = self._pending.get(request_id)
                if state is not None and not state.result_future.done():
                    state.result_future.set_result(result)
                else:
                    logger.warning(
                        "Request %d: last stage is originator but no pending "
                        "state found — result dropped",
                        request_id,
                    )
            else:
                # Send PIPELINE_RESULT to originator.
                originator_addr = self._peers.get(originator_node_id)
                if originator_addr is None:
                    logger.error(
                        "Request %d: no peer address for originator %s… — "
                        "cannot deliver result",
                        request_id,
                        originator_node_id[:8],
                    )
                    return
                encoded = _encode_pipeline_message(
                    request_id=request_id,
                    stage_index=stage_index,
                    msg_type=PIPELINE_RESULT,
                    fleet_hash=fleet_hash,
                    originator_node_id=originator_node_id,
                    payload=result,
                )
                try:
                    await self._rpc_client.send(
                        originator_addr,
                        MessageType.ACTIVATION,
                        encoded,
                    )
                    logger.debug(
                        "Request %d: PIPELINE_RESULT sent to originator %s…",
                        request_id,
                        originator_node_id[:8],
                    )
                except (RpcError, ConnectionError, OSError) as exc:
                    logger.error(
                        "Request %d: failed to send PIPELINE_RESULT to "
                        "originator %s…: %s",
                        request_id,
                        originator_node_id[:8],
                        exc,
                    )
        else:
            # Forward to next stage.
            await self._forward_to_stage(
                request_id=request_id,
                source_stage=stage_index,
                next_stage=stage_index + 1,
                fleet_hash=fleet_hash,
                originator_node_id=originator_node_id,
                payload=result,
                pipeline_order=pipeline_order,
            )

    async def _forward_to_stage(
        self,
        *,
        request_id: int,
        source_stage: int,
        next_stage: int,
        fleet_hash: str,
        originator_node_id: str,
        payload: bytes,
        pipeline_order: list[str],
    ) -> None:
        """Forward an activation to the next pipeline stage.

        Runs as a background task.  On send failure, propagates the error
        back to the originator if possible, or sets the future exception
        if this node is the originator.
        """
        next_node = pipeline_order[next_stage]
        next_addr = self._peers.get(next_node)
        if next_addr is None:
            error_text = (
                f"No peer address for stage-{next_stage} node {next_node[:8]}…"
            )
            logger.error("Request %d: %s", request_id, error_text)
            await self._propagate_error_to_originator(
                request_id=request_id,
                fleet_hash=fleet_hash,
                originator_node_id=originator_node_id,
                error_text=error_text,
                pipeline_order=pipeline_order,
            )
            return

        encoded = _encode_pipeline_message(
            request_id=request_id,
            stage_index=next_stage,
            msg_type=PIPELINE_FORWARD,
            fleet_hash=fleet_hash,
            originator_node_id=originator_node_id,
            payload=payload,
        )
        try:
            await self._rpc_client.send(
                next_addr,
                MessageType.ACTIVATION,
                encoded,
            )
            logger.debug(
                "Request %d: forwarded stage %d → %d (%s…)",
                request_id,
                source_stage,
                next_stage,
                next_node[:8],
            )
        except (RpcError, ConnectionError, OSError) as exc:
            error_text = (
                f"Failed to forward stage {source_stage}→{next_stage} "
                f"to {next_node[:8]}…: {exc}"
            )
            logger.error("Request %d: %s", request_id, error_text)
            await self._propagate_error_to_originator(
                request_id=request_id,
                fleet_hash=fleet_hash,
                originator_node_id=originator_node_id,
                error_text=error_text,
                pipeline_order=pipeline_order,
            )

    async def _send_pipeline_error(
        self,
        *,
        request_id: int,
        fleet_hash: str,
        originator_node_id: str,
        error_text: str,
        pipeline_order: list[str],
    ) -> None:
        """Send a PIPELINE_ERROR back to the originator.

        If this node IS the originator, resolve the future directly instead.
        """
        if originator_node_id == self._own_node_id:
            state = self._pending.get(request_id)
            if state is not None and not state.result_future.done():
                state.result_future.set_exception(
                    PipelineError(f"Request {request_id}: {error_text}")
                )
            return

        await self._propagate_error_to_originator(
            request_id=request_id,
            fleet_hash=fleet_hash,
            originator_node_id=originator_node_id,
            error_text=error_text,
            pipeline_order=pipeline_order,
        )

    async def _propagate_error_to_originator(
        self,
        *,
        request_id: int,
        fleet_hash: str,
        originator_node_id: str,
        error_text: str,
        pipeline_order: list[str],
    ) -> None:
        """Send a PIPELINE_ERROR to the originator node over the network.

        If this node IS the originator, resolve the future directly.
        If the originator is unreachable, log the error — the originator
        will time out eventually.
        """
        if originator_node_id == self._own_node_id:
            state = self._pending.get(request_id)
            if state is not None and not state.result_future.done():
                state.result_future.set_exception(
                    PipelineError(f"Request {request_id}: {error_text}")
                )
            return

        originator_addr = self._peers.get(originator_node_id)
        if originator_addr is None:
            logger.error(
                "Request %d: cannot send PIPELINE_ERROR to originator %s… — "
                "no peer address; error was: %s",
                request_id,
                originator_node_id[:8],
                error_text,
            )
            return

        encoded = _encode_pipeline_message(
            request_id=request_id,
            stage_index=0,  # unused for PIPELINE_ERROR
            msg_type=PIPELINE_ERROR,
            fleet_hash=fleet_hash,
            originator_node_id=originator_node_id,
            payload=error_text.encode("utf-8"),
        )
        try:
            await self._rpc_client.send(
                originator_addr,
                MessageType.ACTIVATION,
                encoded,
            )
            logger.debug(
                "Request %d: PIPELINE_ERROR sent to originator %s…",
                request_id,
                originator_node_id[:8],
            )
        except (RpcError, ConnectionError, OSError) as exc:
            logger.error(
                "Request %d: failed to send PIPELINE_ERROR to originator "
                "%s…: %s (original error: %s)",
                request_id,
                originator_node_id[:8],
                exc,
                error_text,
            )

    # ── Internal: helpers ──────────────────────────────────────────────────

    def _own_position_in_pipeline(self, pipeline_order: list[str]) -> int:
        """Return this node's index in *pipeline_order*.

        Raises ``PipelineError`` if this node is not in the order
        (should not happen — caller validates).
        """
        try:
            return pipeline_order.index(self._own_node_id)
        except ValueError:
            raise PipelineError(
                f"own_node_id {self._own_node_id[:8]}… is not in the "
                f"pipeline order ({len(pipeline_order)} nodes)"
            )

    def _cleanup_state(self, request_id: int) -> None:
        """Remove state for *request_id* and cancel any lingering future."""
        state = self._pending.pop(request_id, None)
        if state is not None and not state.result_future.done():
            state.result_future.cancel()
        # Also purge early buffer entry if run_pipeline never picked it up.
        self._early_buffer.pop(request_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════




async def _demo_dead_node() -> None:
    """Demo 4: Dead node mid-pipeline."""
    from rpc import RpcServer
    from sharding import NodeCapability, compute_assignment

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("swarm.pipeline").setLevel(logging.INFO)

    NUM_NODES = 4
    NUM_LAYERS = 40
    BASE_PORT = 22100
    NODE_IDS = [f"dead-{i:02d}-aaaa-bbbb-cccc-dddddddddddd" for i in range(NUM_NODES)]
    HOST = "127.0.0.1"

    peers = {nid: (HOST, BASE_PORT + i) for i, nid in enumerate(NODE_IDS)}
    caps = [NodeCapability(node_id=nid, storage_bandwidth_mbps=4000) for nid in NODE_IDS]
    assignment = compute_assignment(caps, NUM_LAYERS)

    def _compute_stage(activation: bytes, layer_start: int, layer_end: int) -> bytes:
        val = int.from_bytes(activation, "big", signed=True)
        return (val + layer_end - layer_start).to_bytes(8, "big", signed=True)

    servers = []
    clients = []
    pipes = []

    for i, nid in enumerate(NODE_IDS):
        port = BASE_PORT + i
        client = RpcClient(own_node_id=nid)
        clients.append(client)
        pipe = PipelineCoordinator(
            own_node_id=nid,
            rpc_client=client,
            peers=peers,
            compute_stage=_compute_stage,
            assignment=assignment,
        )
        pipes.append(pipe)
        server = RpcServer(own_node_id=nid, port=port, handler=pipe.handle_frame, bind_ip=HOST)
        servers.append(server)

    for srv in servers:
        await srv.start()
    await asyncio.sleep(0.1)

    try:
        print("=" * 72)
        print("  Dead-node timeout demo")
        print("=" * 72)

        # Kill node B (stage 1) and node C (stage 2) — make the pipeline fail
        # at the first hop from stage 0 to stage 1.
        print("  Stopping node B (stage 1)…")
        await servers[1].stop()
        await clients[1].close()

        t0 = time.monotonic()
        try:
            await pipes[0].run_pipeline(
                assignment=assignment,
                request_id=30,
                initial_activation=(0).to_bytes(8, "big", signed=True),
                timeout=3.0,
            )
            print("  ✗ FAIL: run_pipeline returned instead of raising")
        except PipelineTimeout as exc:
            elapsed = time.monotonic() - t0
            print(f"  ✓ PipelineTimeout raised: {exc}")
            print(f"  Elapsed: {elapsed*1000:.1f} ms (should be near timeout)")
        except Exception as exc:
            elapsed = time.monotonic() - t0
            # PipelineError from a failed forward is also acceptable —
            # it means the error was detected quickly rather than via timeout.
            print(f"  ✓ PipelineError raised (not a hang): {type(exc).__name__}: {exc}")
            print(f"  Elapsed: {elapsed*1000:.1f} ms")

        print()

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


async def _demo_fleet_hash_mismatch() -> None:
    """Demo 5: Fleet-hash mismatch."""
    from rpc import RpcServer
    from sharding import NodeCapability, compute_assignment

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("swarm.pipeline").setLevel(logging.INFO)

    NUM_NODES = 4
    NUM_LAYERS = 40
    BASE_PORT = 22200
    NODE_IDS = [f"hash-{i:02d}-aaaa-bbbb-cccc-dddddddddddd" for i in range(NUM_NODES)]
    HOST = "127.0.0.1"

    peers = {nid: (HOST, BASE_PORT + i) for i, nid in enumerate(NODE_IDS)}
    caps = [NodeCapability(node_id=nid, storage_bandwidth_mbps=4000) for nid in NODE_IDS]
    assignment = compute_assignment(caps, NUM_LAYERS)

    # A different assignment (different bandwidth → different hash).
    caps_mismatch = [
        NodeCapability(node_id=NODE_IDS[0], storage_bandwidth_mbps=1000),
        NodeCapability(node_id=NODE_IDS[1], storage_bandwidth_mbps=2000),
        NodeCapability(node_id=NODE_IDS[2], storage_bandwidth_mbps=3000),
        NodeCapability(node_id=NODE_IDS[3], storage_bandwidth_mbps=4000),
    ]
    assignment_mismatch = compute_assignment(caps_mismatch, NUM_LAYERS)

    def _compute_stage(activation: bytes, layer_start: int, layer_end: int) -> bytes:
        val = int.from_bytes(activation, "big", signed=True)
        return (val + layer_end - layer_start).to_bytes(8, "big", signed=True)

    servers = []
    clients = []
    pipes = []

    for i, nid in enumerate(NODE_IDS):
        port = BASE_PORT + i
        client = RpcClient(own_node_id=nid)
        clients.append(client)
        # Node B (i=1) gets the WRONG assignment.
        assn = assignment_mismatch if i == 1 else assignment
        pipe = PipelineCoordinator(
            own_node_id=nid,
            rpc_client=client,
            peers=peers,
            compute_stage=_compute_stage,
            assignment=assn,
        )
        pipes.append(pipe)
        server = RpcServer(own_node_id=nid, port=port, handler=pipe.handle_frame, bind_ip=HOST)
        servers.append(server)

    for srv in servers:
        await srv.start()
    await asyncio.sleep(0.1)

    try:
        print("=" * 72)
        print("  Fleet-hash mismatch demo")
        print("=" * 72)
        print(f"  Node A (hash-00) fleet_hash: {assignment.fleet_hash[:16]}… (correct)")
        print(f"  Node B (hash-01) fleet_hash: {assignment_mismatch.fleet_hash[:16]}… (WRONG)")
        print()

        t0 = time.monotonic()
        try:
            await pipes[0].run_pipeline(
                assignment=assignment,
                request_id=40,
                initial_activation=(0).to_bytes(8, "big", signed=True),
                timeout=5.0,
            )
            print("  ✗ FAIL: run_pipeline returned instead of raising")
        except PipelineError as exc:
            elapsed = time.monotonic() - t0
            print(f"  ✓ PipelineError raised (fleet-hash mismatch):")
            print(f"    {exc}")
            print(f"  Elapsed: {elapsed*1000:.1f} ms")
        except PipelineTimeout as exc:
            elapsed = time.monotonic() - t0
            print(f"  ⚠ PipelineTimeout raised (mismatch node didn't send error):")
            print(f"    {exc}")
            print(f"  Elapsed: {elapsed*1000:.1f} ms")
        print()

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
    """Entry point: run all demos."""
    asyncio.run(_run_all_demos())


async def _run_all_demos() -> None:
    """Run demos 1–3, then the failure-mode demos."""
    # The happy-path demos (1–3) run in _demo().
    # Failure-mode demos run separately to avoid interference.
    from rpc import RpcServer
    from sharding import NodeCapability, compute_assignment

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("swarm.pipeline").setLevel(logging.INFO)

    # ── Happy path: demos 1, 2, 3 ─────────────────────────────────────
    NUM_NODES = 4
    NUM_LAYERS = 40
    BASE_PORT = 23000
    NODE_IDS = [f"full-{i:02d}-aaaa-bbbb-cccc-dddddddddddd" for i in range(NUM_NODES)]
    HOST = "127.0.0.1"

    peers = {nid: (HOST, BASE_PORT + i) for i, nid in enumerate(NODE_IDS)}
    caps = [NodeCapability(node_id=nid, storage_bandwidth_mbps=4000) for nid in NODE_IDS]
    assignment = compute_assignment(caps, NUM_LAYERS)

    def _compute_stage(activation: bytes, layer_start: int, layer_end: int) -> bytes:
        val = int.from_bytes(activation, "big", signed=True)
        return (val + layer_end - layer_start).to_bytes(8, "big", signed=True)

    servers = []
    clients = []
    pipes = []

    for i, nid in enumerate(NODE_IDS):
        port = BASE_PORT + i
        client = RpcClient(own_node_id=nid)
        clients.append(client)
        pipe = PipelineCoordinator(
            own_node_id=nid,
            rpc_client=client,
            peers=peers,
            compute_stage=_compute_stage,
            assignment=assignment,
        )
        pipes.append(pipe)
        server = RpcServer(own_node_id=nid, port=port, handler=pipe.handle_frame, bind_ip=HOST)
        servers.append(server)

    for srv in servers:
        await srv.start()
    await asyncio.sleep(0.1)

    try:
        print("=" * 72)
        print("  Swarm Pipeline — Full Demo")
        print("=" * 72)
        print()

        # Show pipeline topology.
        pipeline_order = pipes[0]._get_pipeline_order(assignment)
        print("  Pipeline order and layer ranges:")
        for idx, nid in enumerate(pipeline_order):
            start, end = pipes[0].layer_range_for_node(assignment, nid)
            print(f"    Stage {idx}: {nid[:10]}…  layers [{start}, {end})  "
                  f"({end - start} layers)")
        print()

        # ── Demo 1: Request from stage 0 ──────────────────────────────
        print("── Demo 1: Request originating from stage 0 (node A) ──")
        initial = 0
        expected = initial + NUM_LAYERS

        t0 = time.monotonic()
        result = await pipes[0].run_pipeline(
            assignment=assignment,
            request_id=100,
            initial_activation=initial.to_bytes(8, "big", signed=True),
            timeout=5.0,
        )
        elapsed = time.monotonic() - t0
        result_val = int.from_bytes(result, "big", signed=True)
        ok = result_val == expected
        print(f"  Initial: {initial}, Result: {result_val}, Expected: {expected}")
        print(f"  {'✓' if ok else '✗ FAIL'}  ({elapsed*1000:.1f} ms)")
        print()

        # ── Demo 2: Request from middle stage ─────────────────────────
        print("── Demo 2: Request originating from middle stage (node C) ──")
        initial_2 = 100
        expected_2 = initial_2 + NUM_LAYERS

        t0 = time.monotonic()
        result_2 = await pipes[2].run_pipeline(
            assignment=assignment,
            request_id=101,
            initial_activation=initial_2.to_bytes(8, "big", signed=True),
            timeout=5.0,
        )
        elapsed_2 = time.monotonic() - t0
        result_val_2 = int.from_bytes(result_2, "big", signed=True)
        ok_2 = result_val_2 == expected_2
        print(f"  Initial: {initial_2}, Result: {result_val_2}, Expected: {expected_2}")
        print(f"  {'✓' if ok_2 else '✗ FAIL'}  ({elapsed_2*1000:.1f} ms)")
        print()

        # ── Demo 3: Two concurrent requests ───────────────────────────
        print("── Demo 3: Two concurrent requests in flight ──")
        init_a, init_b = 0, 500
        exp_a, exp_b = init_a + NUM_LAYERS, init_b + NUM_LAYERS

        async def run_one(req_id: int, init_val: int) -> tuple[int, int]:
            res = await pipes[0].run_pipeline(
                assignment=assignment,
                request_id=req_id,
                initial_activation=init_val.to_bytes(8, "big", signed=True),
                timeout=5.0,
            )
            return req_id, int.from_bytes(res, "big", signed=True)

        t0 = time.monotonic()
        r3 = await asyncio.gather(run_one(200, init_a), run_one(201, init_b))
        elapsed_3 = time.monotonic() - t0

        ok_3a = r3[0][1] == exp_a
        ok_3b = r3[1][1] == exp_b
        print(f"  Request 200: {init_a} → {r3[0][1]} (expected {exp_a}) "
              f"{'✓' if ok_3a else '✗'}")
        print(f"  Request 201: {init_b} → {r3[1][1]} (expected {exp_b}) "
              f"{'✓' if ok_3b else '✗'}")
        print(f"  Both correct: {'✓' if ok_3a and ok_3b else '✗ FAIL'}")
        print(f"  Concurrent: {elapsed_3*1000:.1f} ms")
        print()

        # ── Demo 3b: Duplicate request_id guard ───────────────────────
        print("── Demo 3b: Duplicate request_id guard ──")
        # Use a slow per-request compute_stage (on originator=stage 0)
        # so the first request stays in _pending long enough for the
        # duplicate check to catch it.
        hold_dup: asyncio.Event = asyncio.Event()

        async def _slow_dup(activation: bytes, ls: int, le: int) -> bytes:
            await hold_dup.wait()
            return _compute_stage(activation, ls, le)

        task_a = asyncio.create_task(
            pipes[0].run_pipeline(
                assignment=assignment,
                request_id=250,
                initial_activation=(0).to_bytes(8, "big", signed=True),
                compute_stage=_slow_dup,
                timeout=5.0,
            )
        )
        await asyncio.sleep(0.05)  # let it reach the await

        try:
            await pipes[0].run_pipeline(
                assignment=assignment,
                request_id=250,
                initial_activation=(0).to_bytes(8, "big", signed=True),
                timeout=5.0,
            )
            print("  ✗ FAIL: duplicate request_id was not rejected")
        except PipelineError as exc:
            print(f"  ✓ Duplicate rejected: {exc}")
        finally:
            hold_dup.set()
            await task_a
        print()

        # ── Concurrency cap ───────────────────────────────────────────
        print("── Demo 3c: Concurrency cap ──")
        hold_cap: asyncio.Event = asyncio.Event()

        async def _slow_cap(activation: bytes, ls: int, le: int) -> bytes:
            await hold_cap.wait()
            return _compute_stage(activation, ls, le)

        capped = PipelineCoordinator(
            own_node_id=NODE_IDS[0],
            rpc_client=clients[0],
            peers=peers,
            compute_stage=_compute_stage,
            assignment=assignment,
            max_concurrent_requests=2,
        )
        # Fill the cap with 2 slow requests held on the originator side.
        c1 = asyncio.create_task(
            capped.run_pipeline(
                assignment=assignment,
                request_id=301,
                initial_activation=(0).to_bytes(8, "big", signed=True),
                compute_stage=_slow_cap,
                timeout=10.0,
            )
        )
        c2 = asyncio.create_task(
            capped.run_pipeline(
                assignment=assignment,
                request_id=302,
                initial_activation=(0).to_bytes(8, "big", signed=True),
                compute_stage=_slow_cap,
                timeout=10.0,
            )
        )
        await asyncio.sleep(0.05)

        try:
            await capped.run_pipeline(
                assignment=assignment,
                request_id=303,
                initial_activation=(0).to_bytes(8, "big", signed=True),
                compute_stage=_compute_stage,
                timeout=10.0,
            )
            print("  ✗ FAIL: over-cap request was not rejected")
        except PipelineError as exc:
            print(f"  ✓ Over-cap rejected: {exc}")
        finally:
            hold_cap.set()
            await asyncio.gather(c1, c2, return_exceptions=True)
        print()

        if not all([ok, ok_2, ok_3a, ok_3b]):
            print("  HAPPY-PATH DEMOS FAILED — aborting")
            return

    finally:
        for c in clients:
            try:
                await c.close()
            except Exception:                pass
        for srv in servers:
            try:
                await srv.stop()
            except Exception:
                pass

    # ── Demo 4: Dead node ─────────────────────────────────────────────
    await _demo_dead_node()

    # ── Demo 5: Fleet-hash mismatch ───────────────────────────────────
    await _demo_fleet_hash_mismatch()

    print("=" * 72)
    print("  All pipeline demos complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
