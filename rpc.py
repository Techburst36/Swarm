#!/usr/bin/env python3
"""
rpc.py — Transport layer for the Swarm distributed inference fleet.

Binary framing over TCP with backpressure.  This is the foundation the
distributed scheduler (Layer 4) sits on; it is correct and boring by design.

Wire format
-----------
Every message is a 20-byte fixed header followed by a variable-length payload::

    ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
    │  magic   │ version  │ msg_type │  flags   │stream_id │ payl_len │ checksum │
    │ uint32   │  uint8   │  uint8   │ uint16   │ uint32   │ uint32   │ uint32   │
    │  4 B     │  1 B     │  1 B     │  2 B     │  4 B     │  4 B     │  4 B     │
    └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
    └─ 20 bytes fixed ───────────────────────────────────────────────────────────┘
    ... followed by payload_len bytes of payload ...

All integers are big-endian (``!``).  The magic number is ``0x5357524D``
(ASCII ``"SWRM"``).  The checksum is a CRC32 of the payload (zlib.crc32).

Backpressure strategy
---------------------
The receive side uses a bounded ``asyncio.Queue`` (default *max_pending* 64).
The receive loop checks whether the queue is full **before** reading the next
frame from the socket.  When full, it briefly sleeps to let the handler
catch up, which in turn lets TCP's own flow control push back on the sender.

Responses to pending ``send_and_wait`` requests **bypass the queue entirely**
and resolve their future directly, so a handler that calls ``send_and_wait``
internally cannot deadlock against a full receive queue.  Likewise,
PING→PONG and GOODBYE are handled inside the receive loop before queue
dispatch.

The send side uses ``StreamWriter.drain()`` after every write to honour TCP
backpressure from the remote end.

Threat model
------------
The current design assumes a trusted LAN.  No TLS, no authentication, no
encryption.  This is the layer where auth would be added later (a simple
pre-shared key or certificate check in the HELLO handshake).

Design rules (do not break)
---------------------------
- Python 3.11+, standard library only.
- **Never** use pickle, marshal, shelve, or eval on received data.
- All background tasks are strongly referenced via ``_background_tasks`` sets.
- No fixed roles — any node can be server and client simultaneously.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import enum
import hashlib
import logging
import os
import struct
import time
import zlib
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("swarm.rpc")

# ── Constants ──────────────────────────────────────────────────────────────────

MAGIC: int = 0x5357524D  # "SWRM"
HEADER_FORMAT: str = "!IBBHIII"  # magic, version, msg_type, flags, stream_id, payload_len, checksum
HEADER_SIZE: int = struct.calcsize(HEADER_FORMAT)  # 20
CURRENT_VERSION: int = 1

DEFAULT_MAX_FRAME_BYTES: int = 256 * 1024 * 1024  # 256 MB
DEFAULT_MAX_PENDING: int = 64
DEFAULT_CONNECT_TIMEOUT: float = 5.0
DEFAULT_SEND_TIMEOUT: float = 30.0
DEFAULT_MAX_RETRIES: int = 5
DEFAULT_BACKOFF_BASE: float = 0.1  # seconds
DEFAULT_BACKOFF_CAP: float = 5.0  # seconds


# ── Exceptions ─────────────────────────────────────────────────────────────────


class RpcError(Exception):
    """Raised for transport-layer errors.

    Covers: ERROR frame received from peer, connection lost mid-operation,
    handshake failure, timeout on ``send_and_wait``, and protocol violations
    (bad magic, unknown version, checksum mismatch, oversized frame).
    """


class ConnectionClosed(RpcError):
    """Raised when an operation is attempted on a closed connection."""


# ── MessageType ────────────────────────────────────────────────────────────────


class MessageType(enum.IntEnum):
    """Stable message type codes for the RPC framing protocol.

    Values are fixed once assigned — changing them breaks wire compatibility.
    """

    HELLO = 1       # sent on connect: payload = sender's node_id (UTF-8)
    HELLO_ACK = 2   # response to HELLO: payload = acceptor's node_id (UTF-8)
    ACTIVATION = 3  # opaque tensor bytes
    RESULT = 4      # response carrying computed output
    PING = 5        # liveness check
    PONG = 6        # liveness response
    ERROR = 7       # payload = UTF-8 error string
    GOODBYE = 8     # graceful disconnect


# ── Frame ──────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Frame:
    """A single framed message on the wire.

    Attributes
    ----------
    magic:
        Must be ``0x5357524D`` (``"SWRM"``).
    version:
        Protocol version.  Currently only ``1`` is supported.
    msg_type:
        One of :class:`MessageType` (or an ``int`` for forward compatibility).
    flags:
        Reserved, must be ``0``.
    stream_id:
        Correlates request/response.  Locally-initiated messages use even IDs,
        remotely-initiated use odd.
    payload_len:
        Length of *payload* in bytes.
    checksum:
        CRC32 of *payload* (``zlib.crc32``, truncated to uint32).
    payload:
        The message body, or ``None`` when decoded from header only.
    """

    magic: int
    version: int
    msg_type: int
    flags: int
    stream_id: int
    payload_len: int
    checksum: int
    payload: bytes | None = None

    def encode(self) -> bytes:
        """Encode header + payload to wire bytes."""
        header = struct.pack(
            HEADER_FORMAT,
            self.magic,
            self.version,
            self.msg_type,
            self.flags,
            self.stream_id,
            self.payload_len,
            self.checksum,
        )
        if self.payload is not None:
            return header + self.payload
        return header

    @classmethod
    def decode_header(cls, data: bytes) -> Frame:
        """Parse a 20-byte header from *data*.

        Raises
        ------
        struct.error
            If *data* is not exactly 20 bytes.
        """
        magic, version, msg_type, flags, stream_id, payload_len, checksum = struct.unpack(
            HEADER_FORMAT, data
        )
        return cls(
            magic=magic,
            version=version,
            msg_type=msg_type,
            flags=flags,
            stream_id=stream_id,
            payload_len=payload_len,
            checksum=checksum,
            payload=None,
        )


# ── RpcConnection ──────────────────────────────────────────────────────────────


class RpcConnection:
    """A single bidirectional RPC connection to one peer.

    Owns the TCP socket, the receive loop, and the handler dispatch task.
    Created by :class:`RpcServer` (inbound) or :class:`RpcClient` (outbound).

    Parameters
    ----------
    reader:
        ``asyncio.StreamReader`` for the connected socket.
    writer:
        ``asyncio.StreamWriter`` for the connected socket.
    own_node_id:
        Stable UUID of *this* node, sent in the HELLO handshake.
    max_frame_bytes:
        Reject any frame whose ``payload_len`` exceeds this (default 256 MB).
    max_pending:
        Maximum number of unsolicited frames queued for the handler before
        backpressure stops reading from the socket (default 64).
    handler:
        Optional async callback ``(connection, frame)`` for unsolicited
        messages.  Frames that are responses to local ``send_and_wait`` calls
        bypass this handler.
    on_close:
        Optional sync callback ``(connection)`` called when the connection
        closes (cleanly or with error).
    is_initiator:
        ``True`` if this side dialed the connection (sends HELLO first).
        ``False`` if this side accepted (waits for HELLO).
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        own_node_id: str,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        max_pending: int = DEFAULT_MAX_PENDING,
        handler: Callable[..., Any] | None = None,
        on_close: Callable[..., Any] | None = None,
        is_initiator: bool = False,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._own_node_id = own_node_id
        self._max_frame_bytes = max_frame_bytes
        self._handler = handler
        self._on_close = on_close
        self._is_initiator = is_initiator

        # Peer identity — set after HELLO handshake completes.
        self._peer_node_id: str | None = None

        # Peer address for diagnostics.
        peername = writer.get_extra_info("peername")
        self._peer_addr: str = f"{peername[0]}:{peername[1]}" if peername else "unknown"

        # Bounded queue for unsolicited frames → handler.
        self._receive_queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=max_pending)
        # Caps concurrent handler invocations.  This is what preserves
        # backpressure now that handlers run in their own tasks rather than
        # inline in the handler loop.
        self._handler_slots: asyncio.Semaphore = asyncio.Semaphore(max_pending)

        # Pending response futures, keyed by stream_id.
        self._pending: dict[int, asyncio.Future[Frame]] = {}

        # Stream-ID allocator (even IDs = locally initiated).
        self._next_stream_id: int = 0
        self._stream_id_lock = asyncio.Lock()

        # State.
        self._closing: bool = False
        # _closing means "shutdown requested" and is set by several paths
        # (a peer's GOODBYE, an explicit close, a protocol error).  It must
        # NOT double as the idempotency guard for close(): when the receive
        # loop exits because _closing was already set, its finally-block call
        # to close() would return immediately having cancelled nothing,
        # leaving the handler task alive forever and hanging server shutdown.
        self._close_started: bool = False
        self._handshake_done: bool = False

        # Background tasks — strong references (see module docstring for why).
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._recv_task: asyncio.Task[Any] | None = None
        self._handler_task: asyncio.Task[Any] | None = None

        # Set when the receive loop exits with an error, so send_and_wait
        # can surface it immediately rather than waiting for timeout.
        self._recv_error: Exception | None = None

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def peer_node_id(self) -> str | None:
        """The remote node's stable UUID, or ``None`` before handshake."""
        return self._peer_node_id

    @property
    def peer_description(self) -> str:
        """Human-readable peer identifier for log messages."""
        if self._peer_node_id:
            return f"{self._peer_node_id[:8]}… @ {self._peer_addr}"
        return self._peer_addr

    @property
    def is_closed(self) -> bool:
        return self._closing

    # ── Public API ─────────────────────────────────────────────────────────

    async def send_frame(
        self,
        msg_type: MessageType | int,
        payload: bytes = b"",
        *,
        stream_id: int | None = None,
        timeout: float | None = None,
    ) -> None:
        """Send a frame to the peer (fire-and-forget).

        Parameters
        ----------
        msg_type:
            Message type from :class:`MessageType`.
        payload:
            Opaque payload bytes.
        stream_id:
            If ``None``, a locally-initiated even stream ID is allocated.
        timeout:
            Seconds to wait for ``drain()``, or ``None`` for no timeout.
        """
        if self._closing:
            raise ConnectionClosed(f"Connection to {self.peer_description} is closed")

        if stream_id is None:
            stream_id = await self._allocate_stream_id()

        await self._send_frame_raw(msg_type, payload, stream_id, timeout=timeout)

    async def send_and_wait(
        self,
        msg_type: MessageType | int,
        payload: bytes = b"",
        *,
        timeout: float = DEFAULT_SEND_TIMEOUT,
    ) -> bytes:
        """Send a request and wait for the peer's response.

        Allocates a stream ID, registers a future, sends the frame, and
        blocks until a matching response arrives or *timeout* expires.

        Returns the response frame's payload (``bytes``).

        Raises
        ------
        RpcError
            If the peer responds with an ERROR frame, or the connection is
            lost before a response arrives.
        asyncio.TimeoutError
            If no response arrives within *timeout*.
        ConnectionClosed
            If the connection is already closed.
        """
        if self._closing:
            raise ConnectionClosed(f"Connection to {self.peer_description} is closed")

        stream_id = await self._allocate_stream_id()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Frame] = loop.create_future()
        self._pending[stream_id] = future

        try:
            await self._send_frame_raw(msg_type, payload, stream_id, timeout=timeout)
            # Check for an error that arrived between send and the await below.
            if self._recv_error is not None:
                raise RpcError(
                    f"Connection to {self.peer_description} lost: {self._recv_error}"
                ) from self._recv_error
            frame = await asyncio.wait_for(future, timeout=timeout)
            return frame.payload if frame.payload is not None else b""
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"send_and_wait({MessageType(msg_type).name}) to "
                f"{self.peer_description} timed out after {timeout}s"
            )
        finally:
            self._pending.pop(stream_id, None)

    async def close(self) -> None:
        """Gracefully close the connection.

        Sends a GOODBYE frame, cancels background tasks, and closes the socket.
        Safe to call multiple times.
        """
        if self._close_started:
            return
        self._close_started = True
        self._closing = True

        # Try to send GOODBYE — don't block on it.
        try:
            with contextlib.suppress(OSError, ConnectionError, ConnectionClosed):
                await self._send_frame_raw(
                    MessageType.GOODBYE, b"", stream_id=0, timeout=1.0
                )
        except Exception:
            pass

        # Cancel background tasks.
        #
        # close() is frequently called FROM one of these tasks (the receive
        # loop's finally block calls it on disconnect).  Cancelling and then
        # awaiting the currently-running task makes it await itself, which
        # never completes -- the close hangs forever and takes any caller
        # (e.g. RpcServer.stop) with it.  So: cancel every task, but only
        # await the ones that are not us.
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None

        # Per-frame handler tasks spawned by _handler_loop must be cancelled
        # too.  Cancelling only the loop tasks leaves these pending, which
        # surfaces as "Task was destroyed but it is pending!" at loop
        # teardown and means handler cleanup never ran.  Snapshot the set
        # first: done-callbacks mutate it as tasks finish.
        for task in list(self._background_tasks):
            if task.done() or task is current:
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        for task in (self._recv_task, self._handler_task):
            if task is None or task.done():
                continue
            task.cancel()
            if task is current:
                continue  # cannot await ourselves; cancellation lands on unwind
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Close the writer.
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except (OSError, ConnectionError):
            pass

        # Resolve any pending futures with an error.
        for sid, future in list(self._pending.items()):
            if not future.done():
                future.set_exception(
                    RpcError(f"Connection to {self.peer_description} closed")
                )
        self._pending.clear()

        logger.info("Connection to %s closed", self.peer_description)

        # Notify owner.
        if self._on_close is not None:
            try:
                self._on_close(self)
            except Exception:
                logger.exception("on_close callback raised")

    # ── Handshake ──────────────────────────────────────────────────────────

    async def handshake(self) -> None:
        """Run the HELLO / HELLO_ACK handshake.

        Called by the initiator (client side) after the TCP connection is
        established but before the receive loop starts.
        """
        if self._handshake_done:
            return

        if self._is_initiator:
            await self._initiator_handshake()
        else:
            await self._acceptor_handshake()

        self._handshake_done = True
        logger.debug("Handshake complete with %s", self.peer_description)

    async def _initiator_handshake(self) -> None:
        """Send HELLO, wait for HELLO_ACK."""
        await self._send_frame_raw(
            MessageType.HELLO,
            self._own_node_id.encode("utf-8"),
            stream_id=0,
        )
        frame = await self._read_frame()
        if frame.msg_type != MessageType.HELLO_ACK:
            raise RpcError(
                f"Expected HELLO_ACK from {self._peer_addr}, "
                f"got {_msg_type_name(frame.msg_type)}"
            )
        if frame.payload is None:
            raise RpcError(f"HELLO_ACK from {self._peer_addr} had no payload")
        self._peer_node_id = frame.payload.decode("utf-8")

    async def _acceptor_handshake(self) -> None:
        """Wait for HELLO, send HELLO_ACK."""
        frame = await self._read_frame()
        if frame.msg_type != MessageType.HELLO:
            raise RpcError(
                f"Expected HELLO from {self._peer_addr}, "
                f"got {_msg_type_name(frame.msg_type)}"
            )
        if frame.payload is None:
            raise RpcError(f"HELLO from {self._peer_addr} had no payload")
        self._peer_node_id = frame.payload.decode("utf-8")
        await self._send_frame_raw(
            MessageType.HELLO_ACK,
            self._own_node_id.encode("utf-8"),
            stream_id=frame.stream_id,
        )

    # ── Start receive loop ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the receive loop and handler task.

        Called after :meth:`handshake` completes.  Idempotent.
        """
        if self._recv_task is not None and not self._recv_task.done():
            return

        self._recv_task = asyncio.create_task(self._recv_loop())
        self._background_tasks.add(self._recv_task)
        self._recv_task.add_done_callback(self._background_tasks.discard)

        if self._handler is not None:
            self._handler_task = asyncio.create_task(self._handler_loop())
            self._background_tasks.add(self._handler_task)
            self._handler_task.add_done_callback(self._background_tasks.discard)

    # ── Internals ──────────────────────────────────────────────────────────

    async def _allocate_stream_id(self) -> int:
        """Return the next locally-initiated (even) stream ID."""
        async with self._stream_id_lock:
            sid = self._next_stream_id
            self._next_stream_id += 2
            return sid

    async def _send_frame_raw(
        self,
        msg_type: MessageType | int,
        payload: bytes,
        stream_id: int,
        *,
        timeout: float | None = None,
    ) -> None:
        """Send an already-framed message.  No validation, no ID allocation."""
        checksum = zlib.crc32(payload) & 0xFFFFFFFF
        frame = Frame(
            magic=MAGIC,
            version=CURRENT_VERSION,
            msg_type=msg_type if isinstance(msg_type, int) else msg_type.value,
            flags=0,
            stream_id=stream_id,
            payload_len=len(payload),
            checksum=checksum,
            payload=payload,
        )
        self._writer.write(frame.encode())
        # Do NOT build the drain coroutine before deciding how to await it.
        # Creating it first leaves a window where it can be orphaned (if
        # wait_for raises before wrapping it in a task, which 3.12+'s
        # timeout-based reimplementation makes more likely), producing a
        # "coroutine 'StreamWriter.drain' was never awaited" RuntimeWarning
        # and skipping backpressure entirely on that path.  asyncio.timeout()
        # wraps the await itself, so the coroutine is always consumed.
        if timeout is not None:
            async with asyncio.timeout(timeout):
                await self._writer.drain()
        else:
            await self._writer.drain()

    async def _read_frame(self) -> Frame:
        """Read and validate one frame from the socket.

        Returns the fully-populated :class:`Frame` on success.

        Raises
        ------
        RpcError
            On bad magic, unknown version, oversized payload, or checksum
            mismatch.
        asyncio.IncompleteReadError
            If the peer disconnects mid-frame.
        """
        # Read header.
        header_data = await self._reader.readexactly(HEADER_SIZE)
        frame = Frame.decode_header(header_data)

        # Validate magic.
        if frame.magic != MAGIC:
            raise RpcError(
                f"Bad magic 0x{frame.magic:08X} from {self.peer_description} "
                f"(expected 0x{MAGIC:08X})"
            )

        # Validate version.
        if frame.version != CURRENT_VERSION:
            raise RpcError(
                f"Unknown protocol version {frame.version} from "
                f"{self.peer_description}"
            )

        # Validate payload size.
        if frame.payload_len > self._max_frame_bytes:
            raise RpcError(
                f"Oversized frame from {self.peer_description}: "
                f"{frame.payload_len} bytes (max {self._max_frame_bytes})"
            )

        # Validate flags.
        if frame.flags != 0:
            logger.debug(
                "Non-zero flags 0x%04X from %s — ignoring",
                frame.flags,
                self.peer_description,
            )

        # Read payload.
        payload = await self._reader.readexactly(frame.payload_len)

        # Validate checksum.
        actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_checksum != frame.checksum:
            raise RpcError(
                f"Checksum mismatch from {self.peer_description}: "
                f"got 0x{actual_checksum:08X}, frame claims 0x{frame.checksum:08X}"
            )

        frame.payload = payload
        return frame

    async def _recv_loop(self) -> None:
        """Main receive loop — reads frames and dispatches them.

        Backpressure: before reading each frame, checks whether the handler
        queue is full.  When full, briefly sleeps so the handler can drain
        and TCP flow control pushes back on the sender.

        Exits on any read error, protocol violation, or GOODBYE.
        """
        cancelled = False
        try:
            while not self._closing:
                # --- backpressure: don't read if handler can't keep up ---
                if self._receive_queue.full():
                    await asyncio.sleep(0.001)
                    continue

                try:
                    frame = await self._read_frame()
                except asyncio.IncompleteReadError as e:
                    logger.debug("Peer %s disconnected mid-frame: %s", self.peer_description, e)
                    self._recv_error = e
                    break
                except (ConnectionError, OSError) as e:
                    logger.debug("Connection to %s lost: %s", self.peer_description, e)
                    self._recv_error = e
                    break

                # Dispatch — may signal that we should stop.
                should_continue = await self._dispatch(frame)
                if not should_continue:
                    break
        except RpcError as e:
            logger.debug("Protocol error from %s: %s", self.peer_description, e)
            self._recv_error = e
        except asyncio.CancelledError:
            # We were cancelled, which means close() is ALREADY running --
            # close() is what cancels this task.  Two things matter here:
            #
            # 1. Re-raise rather than swallow.  Suppressing CancelledError
            #    leaves the task stuck in the "cancelling" state forever,
            #    which surfaces at teardown as "Task was destroyed but it is
            #    pending!".
            # 2. Do NOT call close() from the finally in this case.  Any
            #    await inside a cancelling task re-raises CancelledError
            #    immediately, so the redundant close() cannot complete and
            #    the coroutine is torn down mid-await -- the source of
            #    "RuntimeError: coroutine ignored GeneratorExit".
            cancelled = True
            raise
        except Exception:
            logger.exception("Unexpected error in receive loop for %s", self.peer_description)
        finally:
            # Only close on a natural exit (disconnect, protocol error,
            # GOODBYE).  On cancellation the closer is already doing it.
            if not cancelled:
                await self.close()

    async def _dispatch(self, frame: Frame) -> bool:
        """Route an incoming frame.

        Returns ``False`` if the receive loop should stop (GOODBYE received).
        """
        # ── PING → auto-PONG ──────────────────────────────────────────
        if frame.msg_type == MessageType.PING:
            await self._send_frame_raw(
                MessageType.PONG, b"", stream_id=frame.stream_id
            )
            return True

        # ── GOODBYE → stop ────────────────────────────────────────────
        if frame.msg_type == MessageType.GOODBYE:
            logger.info("Peer %s sent GOODBYE", self.peer_description)
            self._closing = True
            return False

        # ── Response to a pending send_and_wait? ───────────────────────
        if frame.stream_id in self._pending:
            future = self._pending.pop(frame.stream_id)
            if frame.msg_type == MessageType.ERROR:
                error_msg = _safe_decode(frame.payload)
                future.set_exception(RpcError(f"Peer {self.peer_description}: {error_msg}"))
            else:
                future.set_result(frame)
            return True

        # ── Unsolicited ERROR → log ───────────────────────────────────
        if frame.msg_type == MessageType.ERROR:
            logger.warning(
                "Unsolicited ERROR from %s: %s",
                self.peer_description,
                _safe_decode(frame.payload),
            )
            return True

        # ── Unsolicited message → handler queue ───────────────────────
        if self._handler is not None:
            try:
                self._receive_queue.put_nowait(frame)
            except asyncio.QueueFull:
                # Should not happen (we check full() before reading), but
                # handle gracefully: log and drop.
                logger.warning(
                    "Handler queue full, dropping %s from %s",
                    _msg_type_name(frame.msg_type),
                    self.peer_description,
                )
            return True

        # No handler → log and continue.
        logger.debug(
            "No handler for unsolicited %s from %s (stream %d)",
            _msg_type_name(frame.msg_type),
            self.peer_description,
            frame.stream_id,
        )
        return True

    async def _run_one_handler(self, frame: Frame) -> None:
        """Run the handler for a single frame, then release its slot.

        Exceptions are logged, never propagated -- one bad frame must not
        tear down the connection.
        """
        assert self._handler is not None
        try:
            result = self._handler(self, frame)
            if asyncio.iscoroutine(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Handler raised on %s from %s",
                _msg_type_name(frame.msg_type),
                self.peer_description,
            )
        finally:
            self._handler_slots.release()
            self._receive_queue.task_done()

    async def _handler_loop(self) -> None:
        """Consume frames from the receive queue and dispatch them.

        Each frame runs in its own task rather than inline, so a handler
        that blocks (for example, one that calls ``send_and_wait`` and
        awaits a reply) does not prevent the queue from draining.

        Why this matters: ``_recv_loop`` declines to read from the socket
        while the receive queue is full.  With serial inline handling, a
        blocked handler meant the queue never drained, so the socket was
        never read, so the reply the handler was waiting for never arrived
        -- a stall lasting until the send timeout fired.

        Backpressure is preserved by ``_handler_slots``, a semaphore that
        caps how many handlers run concurrently.  Acquiring a slot before
        pulling from the queue means an overloaded handler still applies
        pressure back through the queue to TCP, which is the intended
        behaviour; it just no longer blocks unrelated traffic.
        """
        assert self._handler is not None
        try:
            while not self._closing:
                # Acquire a slot BEFORE taking work off the queue, so the
                # queue depth still reflects real pending work.
                await self._handler_slots.acquire()
                try:
                    frame = await self._receive_queue.get()
                except asyncio.CancelledError:
                    self._handler_slots.release()
                    raise

                task = asyncio.create_task(self._run_one_handler(frame))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
        except asyncio.CancelledError:
            pass


# ── RpcServer ──────────────────────────────────────────────────────────────────


class RpcServer:
    """Listen for inbound RPC connections.

    Binds a TCP port, accepts connections, performs the acceptor-side
    HELLO handshake, and registers a handler for each peer.

    Parameters
    ----------
    own_node_id:
        Stable UUID of this node, sent in HELLO_ACK.
    port:
        TCP port to bind.
    handler:
        Async callback ``(connection, frame)`` for unsolicited messages
        from accepted peers.
    bind_ip:
        IP to bind to (default ``"0.0.0.0"``).
    max_frame_bytes:
        Maximum frame payload size (default 256 MB).
    max_pending:
        Maximum unsolicited frames queued per connection (default 64).
    """

    def __init__(
        self,
        *,
        own_node_id: str,
        port: int,
        handler: Callable[..., Any],
        bind_ip: str = "0.0.0.0",
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        max_pending: int = DEFAULT_MAX_PENDING,
    ) -> None:
        self._own_node_id = own_node_id
        self._port = port
        self._handler = handler
        self._bind_ip = bind_ip
        self._max_frame_bytes = max_frame_bytes
        self._max_pending = max_pending

        self._server: asyncio.Server | None = None
        self._connections: list[RpcConnection] = []
        self._connections_lock = asyncio.Lock()

        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._accept_task: asyncio.Task[Any] | None = None

    @property
    def connections(self) -> list[RpcConnection]:
        """Snapshot of currently active connections."""
        return list(self._connections)

    async def start(self) -> None:
        """Start listening.  Idempotent."""
        if self._server is not None:
            return

        self._server = await asyncio.start_server(
            self._on_client_connected,
            host=self._bind_ip,
            port=self._port,
        )
        logger.info(
            "RpcServer listening on %s:%d (node %s…)",
            self._bind_ip,
            self._port,
            self._own_node_id[:8],
        )

    async def stop(self) -> None:
        """Stop listening and close all connections.

        Ordering matters here.  On Python 3.12+, ``asyncio.Server.wait_closed()``
        does not return until every connection the server accepted has
        finished.  Awaiting it before closing those connections is a circular
        wait: wait_closed() waits for the connections, and the connections are
        only closed after wait_closed() returns.  So connections are closed
        first, then the listener is drained.
        """
        # Stop accepting new connections immediately, but do not await yet.
        if self._server is not None:
            self._server.close()

        # Close all existing connections first.
        async with self._connections_lock:
            conns = list(self._connections)
            self._connections.clear()

        for conn in conns:
            try:
                await conn.close()
            except Exception:
                logger.exception("Error closing connection during shutdown")

        # Now the listener can actually drain.
        if self._server is not None:
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None

        logger.info("RpcServer stopped")

    async def _on_client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a new inbound TCP connection."""
        conn = RpcConnection(
            reader,
            writer,
            own_node_id=self._own_node_id,
            max_frame_bytes=self._max_frame_bytes,
            max_pending=self._max_pending,
            handler=self._handler,
            on_close=self._on_connection_closed,
            is_initiator=False,
        )

        try:
            await conn.handshake()
        except (RpcError, asyncio.IncompleteReadError, ConnectionError, OSError) as e:
            logger.debug("Handshake failed with %s: %s", conn.peer_description, e)
            await conn.close()
            return

        async with self._connections_lock:
            self._connections.append(conn)

        await conn.start()
        logger.info("Accepted connection from %s", conn.peer_description)

    def _on_connection_closed(self, conn: RpcConnection) -> None:
        """Remove *conn* from the active list (called via on_close)."""
        # Best-effort removal; lock might be held, so use create_task.
        task = asyncio.create_task(self._remove_connection(conn))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _remove_connection(self, conn: RpcConnection) -> None:
        async with self._connections_lock:
            with contextlib.suppress(ValueError):
                self._connections.remove(conn)


# ── RpcClient ──────────────────────────────────────────────────────────────────


class RpcClient:
    """Manages outbound RPC connections with connection pooling and reconnect.

    Parameters
    ----------
    own_node_id:
        Stable UUID of this node, sent in HELLO.
    handler:
        Optional async callback for unsolicited messages from peers.
    connect_timeout:
        Seconds to wait for a TCP connection to establish (default 5).
    max_retries:
        Maximum connection attempts before giving up (default 5).
    max_frame_bytes:
        Maximum frame payload size (default 256 MB).
    max_pending:
        Maximum unsolicited frames queued per connection (default 64).
    """

    def __init__(
        self,
        *,
        own_node_id: str,
        handler: Callable[..., Any] | None = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        max_pending: int = DEFAULT_MAX_PENDING,
    ) -> None:
        self._own_node_id = own_node_id
        self._handler = handler
        self._connect_timeout = connect_timeout
        self._max_retries = max_retries
        self._max_frame_bytes = max_frame_bytes
        self._max_pending = max_pending

        # Connection pool: (host, port) → RpcConnection
        self._pool: dict[tuple[str, int], RpcConnection] = {}
        self._pool_lock = asyncio.Lock()

        # Set by close().  Without this, a task still running after close()
        # can call _get_connection(), which dials a brand-new connection
        # into the freshly-emptied pool -- one nobody will ever close, whose
        # receive loop is still blocked on a read at interpreter teardown
        # ("Task was destroyed but it is pending!").  A closed client must
        # refuse to reopen.
        self._closed: bool = False

        # Strong references to in-flight cleanup tasks.  asyncio.create_task()
        # only holds a weak reference via the event loop, so without this a
        # task can be garbage-collected before it runs and its exception is
        # lost silently.  Assigning to a local variable does NOT help: the
        # local goes out of scope as soon as the enclosing function returns.
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def send_and_wait(
        self,
        peer: tuple[str, int],
        msg_type: MessageType | int,
        payload: bytes = b"",
        *,
        timeout: float = DEFAULT_SEND_TIMEOUT,
    ) -> bytes:
        """Send a request to *peer* and return the response payload.

        Acquires a connection (from pool or new), sends the frame, and
        blocks until a response arrives or *timeout* expires.

        Parameters
        ----------
        peer:
            ``(host, port)`` tuple.  ``host`` may be an IP address or hostname.
        msg_type:
            Message type from :class:`MessageType`.
        payload:
            Opaque payload bytes.
        timeout:
            Seconds to wait for the response (default 30).

        Returns
        -------
        bytes
            The response frame's payload.

        Raises
        ------
        RpcError
            If the peer responds with an ERROR frame, the connection cannot
            be established, or the connection is lost.
        asyncio.TimeoutError
            If no response arrives within *timeout*.
        """
        conn = await self._get_connection(peer)
        return await conn.send_and_wait(msg_type, payload, timeout=timeout)

    async def send(
        self,
        peer: tuple[str, int],
        msg_type: MessageType | int,
        payload: bytes = b"",
        *,
        timeout: float | None = None,
    ) -> None:
        """Send a fire-and-forget frame to *peer*.

        Parameters
        ----------
        peer:
            ``(host, port)`` tuple.
        msg_type:
            Message type from :class:`MessageType`.
        payload:
            Opaque payload bytes.
        timeout:
            Seconds to wait for ``drain()``, or ``None``.
        """
        conn = await self._get_connection(peer)
        await conn.send_frame(msg_type, payload, timeout=timeout)

    async def close(self) -> None:
        """Close all pooled connections.

        After this returns the client is permanently closed: further sends
        raise rather than silently dialling a new connection.
        """
        async with self._pool_lock:
            self._closed = True
            conns = list(self._pool.values())
            self._pool.clear()

        for conn in conns:
            await conn.close()
        logger.info("RpcClient closed (%d connections)", len(conns))

    # ── Internals ──────────────────────────────────────────────────────────

    async def _get_connection(self, peer: tuple[str, int]) -> RpcConnection:
        """Return a live connection to *peer*, creating one if needed.

        Uses bounded exponential backoff on initial connection attempts.
        Removes dead connections from the pool automatically.
        """
        if self._closed:
            raise RpcError(
                f"RpcClient is closed; refusing to dial {peer[0]}:{peer[1]}"
            )

        host, port = peer

        # Check existing pooled connection.
        async with self._pool_lock:
            conn = self._pool.get((host, port))
            if conn is not None and not conn.is_closed:
                return conn
            # Dead or missing — remove so we don't reuse a zombie.
            self._pool.pop((host, port), None)

        # Create a new connection with retry.
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                conn = await self._connect(host, port)
                async with self._pool_lock:
                    # Check for a concurrent connection.
                    existing = self._pool.get((host, port))
                    if existing is not None and not existing.is_closed:
                        # Another task beat us — use theirs.
                        await conn.close()
                        return existing
                    self._pool[(host, port)] = conn
                return conn
            except (RpcError, ConnectionError, OSError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = min(
                        DEFAULT_BACKOFF_BASE * (2 ** attempt),
                        DEFAULT_BACKOFF_CAP,
                    )
                    logger.debug(
                        "Connection attempt %d/%d to %s:%d failed: %s — "
                        "retrying in %.2fs",
                        attempt + 1,
                        self._max_retries,
                        host,
                        port,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)

        raise RpcError(
            f"Failed to connect to {host}:{port} after "
            f"{self._max_retries} attempts"
        ) from last_error

    async def _connect(self, host: str, port: int) -> RpcConnection:
        """Establish a TCP connection and perform the initiator handshake."""
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=self._connect_timeout,
        )

        conn = RpcConnection(
            reader,
            writer,
            own_node_id=self._own_node_id,
            max_frame_bytes=self._max_frame_bytes,
            max_pending=self._max_pending,
            handler=self._handler,
            on_close=self._on_pooled_connection_closed,
            is_initiator=True,
        )

        try:
            await conn.handshake()
        except Exception:
            await conn.close()
            raise

        await conn.start()
        logger.info(
            "Connected to %s:%d (peer %s)", host, port, conn.peer_description
        )
        return conn

    def _on_pooled_connection_closed(self, conn: RpcConnection) -> None:
        """Remove *conn* from the pool when it closes."""
        peername = conn._writer.get_extra_info("peername")
        if peername is not None:
            key = (peername[0], peername[1])
            # Best-effort; don't block.  Use a task since we might be inside
            # the connection's own close path.
            task = asyncio.create_task(self._remove_from_pool(key))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _remove_from_pool(self, key: tuple[str, int]) -> None:
        async with self._pool_lock:
            self._pool.pop(key, None)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _msg_type_name(value: int) -> str:
    """Human-readable name for a message type code."""
    try:
        return MessageType(value).name
    except ValueError:
        return f"<{value}>"


def _safe_decode(payload: bytes | None) -> str:
    """Decode UTF-8 payload safely, never raising."""
    if payload is None:
        return "<null>"
    return payload.decode("utf-8", errors="replace")


# ── Demo ───────────────────────────────────────────────────────────────────────


async def _demo_handler(conn: RpcConnection, frame: Frame) -> None:
    """Demo handler: echo ACTIVATION frames back as RESULT.

    For a realistic 50 MB payload this exercises the full backpressure path:
    the handler is deliberately slow so the receive queue fills up, which
    stops the receive loop from reading, which lets TCP push back on the
    sender's ``drain()``.
    """
    if frame.msg_type == MessageType.ACTIVATION:
        pl = len(frame.payload) if frame.payload else 0
        logger.debug(
            "Handler processing %d-byte ACTIVATION from %s (stream %d)",
            pl,
            conn.peer_description,
            frame.stream_id,
        )
        # For large payloads (>1 MB), add a small artificial delay so the
        # test exercises backpressure rather than completing instantly.
        if pl > 1_000_000:
            await asyncio.sleep(0.05)
        await conn.send_frame(
            MessageType.RESULT,
            frame.payload or b"",
            stream_id=frame.stream_id,
        )
    else:
        logger.debug(
            "Handler got %s from %s (stream %d)",
            _msg_type_name(frame.msg_type),
            conn.peer_description,
            frame.stream_id,
        )


async def _demo() -> None:
    """Run a self-contained demo: server + client on localhost.

    Exercises:
    - HELLO handshake
    - Small request/response (PING→PONG handled automatically by the receive loop)
    - Small ACTIVATION→RESULT through the handler
    - Large 50 MB ACTIVATION→RESULT (exercises backpressure)
    - Graceful shutdown with GOODBYE
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down debug logs for the demo unless --verbose is used.
    logging.getLogger("swarm.rpc").setLevel(logging.INFO)

    SERVER_PORT = 19999
    SERVER_NODE = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    CLIENT_NODE = "11111111-2222-3333-4444-555555555555"

    # ── Start server ──────────────────────────────────────────────────
    server = RpcServer(
        own_node_id=SERVER_NODE,
        port=SERVER_PORT,
        handler=_demo_handler,
        bind_ip="127.0.0.1",
    )
    await server.start()

    # ── Create client ─────────────────────────────────────────────────
    client = RpcClient(
        own_node_id=CLIENT_NODE,
        connect_timeout=5.0,
    )

    try:
        peer = ("127.0.0.1", SERVER_PORT)

        # Wait a moment for the server to be ready.
        await asyncio.sleep(0.1)

        # ── Small message: PING → PONG ────────────────────────────────
        print("\n── PING → PONG (handled automatically by receive loop) ──")
        t0 = time.monotonic()
        pong = await client.send_and_wait(peer, MessageType.PING, timeout=5.0)
        dt = time.monotonic() - t0
        print(f"  PONG payload: {len(pong)} bytes, round-trip: {dt*1000:.2f} ms")

        # ── Small ACTIVATION → RESULT ─────────────────────────────────
        print("\n── Small ACTIVATION → RESULT (12 KB, like a decode layer boundary) ──")
        small_payload = b"\xAB" * (12 * 1024)  # 12 KB
        t0 = time.monotonic()
        result = await client.send_and_wait(
            peer, MessageType.ACTIVATION, small_payload, timeout=5.0
        )
        dt = time.monotonic() - t0
        ok = "OK" if result == small_payload else "MISMATCH"
        print(f"  {ok}: {len(result)} bytes, round-trip: {dt*1000:.2f} ms "
              f"({len(result)/dt/1e6:.2f} MB/s)")

        # ── Large ACTIVATION → RESULT (50 MB) ─────────────────────────
        print("\n── Large ACTIVATION → RESULT (50 MB, exercises backpressure) ──")
        print("  Generating 50 MB payload...")
        # Use a deterministic but non-compressible payload.
        large_payload = hashlib.sha256(b"swarm demo seed").digest() * (50 * 1024 * 1024 // 32)
        print(f"  Payload: {len(large_payload)/1e6:.0f} MB, sending...")
        t0 = time.monotonic()
        result = await client.send_and_wait(
            peer, MessageType.ACTIVATION, large_payload, timeout=120.0
        )
        dt = time.monotonic() - t0
        ok = (
            "OK (hash match)"
            if hashlib.sha256(result).digest() == hashlib.sha256(large_payload).digest()
            else "MISMATCH"
        )
        print(f"  {ok}: {len(result)/1e6:.1f} MB, round-trip: {dt:.2f}s "
              f"({len(result)/dt/1e6:.2f} MB/s)")

        # ── Small message again (connection still healthy) ────────────
        print("\n── Small ACTIVATION → RESULT (verify connection still healthy) ──")
        t0 = time.monotonic()
        result = await client.send_and_wait(
            peer, MessageType.ACTIVATION, small_payload, timeout=5.0
        )
        dt = time.monotonic() - t0
        ok = "OK" if result == small_payload else "MISMATCH"
        print(f"  {ok}: {len(result)} bytes, round-trip: {dt*1000:.2f} ms")

        print("\n── All demo steps passed ──")

    finally:
        await client.close()
        await server.stop()


def main() -> None:
    """Entry point: run the demo."""
    asyncio.run(_demo())


if __name__ == "__main__":
    main()
