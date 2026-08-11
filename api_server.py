#!/usr/bin/env python3
"""
api_server.py — OpenAI-compatible HTTP API for the Swarm distributed
inference fleet.

Layer 5 of the software stack.  This module runs on one node in the fleet
and exposes a standard chat-completions endpoint, a model list, and a
health check.  It consumes ``FailoverCoordinator.diff_queue`` to track the
current ``ShardAssignment``, and rejects requests while the fleet is
unconverged.

Design rules (do not break)
----------------------------
- Python 3.11+, standard library only, plus the six existing project modules.
  No FastAPI, no Flask, no uvicorn, no http.server — raw ``asyncio.start_server``.
- Bind to 127.0.0.1 by default.  No 0.0.0.0 without an explicit ``--bind``.
- No authentication (trusted-LAN threat model, consistent with rpc.py).
  AUTHENTICATION_INSERTION_POINT marks where auth would go.
- All background tasks strongly referenced (asyncio.create_task pitfall — this
  makes it six times across the project).
- Bounded everything: concurrency cap, body size, generate timeout.
- Reshard mid-request: complete against the old (immutable) assignment if
  nodes are still alive; fail loudly via gang-sync/pipeline timeout if nodes
  have left.  Never silently switch assignments under a running request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from failover import (
    DEFAULT_SETTLE_WINDOW,
    FailoverCoordinator,
    ReshardDiff,
    _FakeFleetTable,
    _make_desc,
)
from node_identity import FleetTable, NodeDescriptor
from sharding import NodeCapability, ShardAssignment, compute_assignment

logger = logging.getLogger("swarm.api_server")

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_BIND: str = "127.0.0.1"
DEFAULT_PORT: int = 8000
DEFAULT_MAX_CONCURRENT: int = 32
DEFAULT_MAX_BODY_BYTES: int = 1 * 1024 * 1024  # 1 MB
DEFAULT_GENERATE_TIMEOUT: float = 300.0  # 5 minutes
DEFAULT_RESHARD_GRACE: float = 0.2  # seconds after diff before marking ready
DEFAULT_MAX_HEADERS_BYTES: int = 64 * 1024  # 64 KB

# ── AUTHENTICATION_INSERTION_POINT ─────────────────────────────────────────────
# The current threat model assumes a trusted LAN (see rpc.py for the full
# rationale).  To add auth later:
#
#   1. Choose a scheme (pre-shared bearer token is simplest — the same key
#      every node knows at provision time).
#   2. Check for "Authorization: Bearer <token>" in every request handler.
#   3. Return 401 with a WWW-Authenticate header on mismatch or absence.
#   4. Add a ``--auth-token`` CLI flag and an env-var fallback.
#   5. The health endpoint (/health) should remain unauthenticated so load
#      balancers and monitoring can reach it.
#
# Do NOT half-implement auth.  It is either fully enforced across every
# data-plane endpoint, or absent everywhere except /health.  A partial
# gate is a false sense of security that costs the same implementation
# effort as a real one.
# ────────────────────────────────────────────────────────────────────────────────


# ── Exceptions ─────────────────────────────────────────────────────────────────


class ApiError(Exception):
    """Raised for API-level errors that produce a structured HTTP response.

    Attributes
    ----------
    status:
        HTTP status code.
    message:
        Human-readable error message.
    error_type:
        OpenAI-compatible error type string.
    """

    def __init__(self, status: int, message: str, error_type: str = "server_error") -> None:
        self.status = status
        self.message = message
        self.error_type = error_type
        super().__init__(message)


# ── Type aliases ───────────────────────────────────────────────────────────────

# A generate function receives (messages, params, assignment) and returns
# either a plain string (non-streaming) or an async iterable of strings
# (streaming, one chunk per token).
GenerateFn = Callable[
    [list[dict[str, Any]], dict[str, Any], ShardAssignment],
    Awaitable[str] | AsyncIterator[str],
]


# ── InstanceManager ────────────────────────────────────────────────────────────


class InstanceManager:
    """Tracks the current fleet assignment and loaded model.

    Consumes ``FailoverCoordinator.diff_queue`` via a background task so
    it stays current as the fleet changes.  Marks the fleet as not-ready
    briefly after each reshard, then re-enables serving once the new
    assignment is settled.

    Parameters
    ----------
    failover:
        The ``FailoverCoordinator`` whose ``diff_queue`` this manager consumes.
    model_name:
        The name of the model this instance serves.  Reported in
        ``/v1/models`` and validated in chat-completions requests.
    generate_fn:
        Callback ``(messages, params, assignment) -> text``.  Called on
        every chat-completions request.  Both sync and async callbacks
        are accepted; streaming is detected via ``__aiter__`` on the
        returned value.
    reshard_grace:
        Seconds to reject requests after a new assignment arrives before
        marking the fleet as ready again (default 0.2).
    """

    def __init__(
        self,
        *,
        failover: FailoverCoordinator,
        model_name: str,
        generate_fn: GenerateFn,
        reshard_grace: float = DEFAULT_RESHARD_GRACE,
    ) -> None:
        self._failover = failover
        self._model_name = model_name
        self._generate_fn = generate_fn
        self._reshard_grace = reshard_grace

        self._ready: bool = False
        self._assignment: ShardAssignment | None = None
        self._start_time: float | None = None  # set in start()

        # Guard for the ready-state transition.
        self._ready_lock = asyncio.Lock()

        # Background task handles.
        self._consumer_task: asyncio.Task[None] | None = None
        self._grace_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def assignment(self) -> ShardAssignment | None:
        return self._assignment

    @property
    def fleet_size(self) -> int:
        if self._assignment is None:
            return 0
        return len(self._assignment.node_counts)

    @property
    def uptime_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    # ── Public API ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Begin consuming the failover diff queue.

        Idempotent — calling ``start()`` on an already-started manager is a
        no-op.
        """
        if self._consumer_task is not None and not self._consumer_task.done():
            return

        self._start_time = time.monotonic()

        # Check if an assignment already exists (failover may have produced
        # one before start() was called).
        existing = self._failover.current_assignment()
        if existing is not None:
            self._assignment = existing
            self._ready = True

        self._consumer_task = asyncio.create_task(self._consume_diffs())
        self._background_tasks.add(self._consumer_task)
        self._consumer_task.add_done_callback(self._background_tasks.discard)

        logger.info(
            "InstanceManager started: model=%s, ready=%s",
            self._model_name,
            self._ready,
        )

    async def stop(self) -> None:
        """Stop consuming diffs and cancel grace timer."""
        for task in list(self._background_tasks):
            if task.done():
                continue
            task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await task
        self._background_tasks.clear()

        if self._grace_task is not None:
            self._grace_task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await self._grace_task
            self._grace_task = None

        self._consumer_task = None
        self._ready = False

    async def generate(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> str | AsyncIterator[str]:
        """Run the generate callback against the current assignment.

        Captures *assignment* before calling the callback, so if a reshard
        lands mid-generation the in-flight request completes against the
        old (immutable) assignment.  This is the "complete safely, never
        silently switch" guarantee.

        Raises
        ------
        ApiError(503)
            If the fleet is not ready (no assignment yet, or mid-reshard).
        """
        assignment = self._get_assignment_for_request()
        if assignment is None:
            raise ApiError(
                503,
                "Fleet is not ready — no shard assignment available. "
                "Wait for node discovery to converge.",
                error_type="service_unavailable",
            )

        fn = self._generate_fn
        try:
            result = fn(messages, params, assignment)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except ApiError:
            raise
        except Exception as exc:
            logger.exception("generate_fn raised")
            raise ApiError(500, f"Inference error: {exc}") from exc

    def _get_assignment_for_request(self) -> ShardAssignment | None:
        """Return the current assignment if the fleet is ready.

        Must be called from within an async context (the ready flag is
        protected by a lock, but reads of a bool are atomic in CPython
        and the worst case is a false 503 that resolves on retry).
        """
        if not self._ready:
            return None
        return self._assignment

    # ── Diff consumer ──────────────────────────────────────────────────────

    async def _consume_diffs(self) -> None:
        """Background task: read ReshardDiffs from failover and update state."""
        try:
            while True:
                diff: ReshardDiff = await self._failover.diff_queue.get()
                await self._apply_diff(diff)
        except asyncio.CancelledError:
            pass

    async def _apply_diff(self, diff: ReshardDiff) -> None:
        """Update state for a new reshard diff.

        Marks the fleet as not-ready, updates the assignment, and schedules
        a grace-period task to re-enable serving.
        """
        async with self._ready_lock:
            self._ready = False
            self._assignment = diff.new_assignment

            logger.info(
                "Reshard applied: fleet_hash=%s…, %d nodes, "
                "%d experts moved. Fleet marked not-ready for %.1fs.",
                diff.new_fleet_hash[:16],
                len(diff.new_assignment.node_counts),
                diff.moved_count,
                self._reshard_grace,
            )

        # Cancel any in-progress grace task (consecutive reshards).
        if self._grace_task is not None:
            self._grace_task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await self._grace_task

        self._grace_task = asyncio.create_task(self._grace_then_ready())
        self._background_tasks.add(self._grace_task)
        self._grace_task.add_done_callback(self._background_tasks.discard)

    async def _grace_then_ready(self) -> None:
        """Wait the reshard grace period, then mark the fleet as ready."""
        try:
            await asyncio.sleep(self._reshard_grace)
        except asyncio.CancelledError:
            return
        async with self._ready_lock:
            self._ready = True
        logger.info("Fleet ready — serving requests against new assignment")


# ── HTTP helpers ───────────────────────────────────────────────────────────────


# contextlib.suppress is cleaner but requires an import we only need here.
import contextlib


def contextlib_suppress(*exceptions: type[BaseException]) -> Any:
    """Local alias so the module docstring's no-dependency claim is honest.

    We only use ``contextlib.suppress`` in two places (stop() and
    _apply_diff()); writing it out keeps the import list minimal.
    """
    return contextlib.suppress(*exceptions)


class _HttpRequest:
    """Parsed HTTP/1.1 request."""

    __slots__ = ("method", "path", "headers", "body", "version")

    def __init__(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        version: str = "HTTP/1.1",
    ) -> None:
        self.method = method
        self.path = path
        self.headers = headers
        self.body = body
        self.version = version


def _make_response(
    status: int,
    body: bytes,
    content_type: str = "application/json",
    *,
    extra_headers: dict[str, str] | None = None,
) -> bytes:
    """Build a complete HTTP/1.1 response."""
    reason = {
        200: "OK",
        201: "Created",
        400: "Bad Request",
        401: "Unauthorized",
        404: "Not Found",
        413: "Content Too Large",
        422: "Unprocessable Entity",
        429: "Too Many Requests",
        500: "Internal Server Error",
        503: "Service Unavailable",
    }.get(status, "Unknown")

    lines = [f"HTTP/1.1 {status} {reason}"]
    lines.append(f"Content-Type: {content_type}")
    lines.append(f"Content-Length: {len(body)}")
    lines.append("Date: " + time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()))
    lines.append("Server: swarm-api")

    if extra_headers:
        for key, value in extra_headers.items():
            lines.append(f"{key}: {value}")
    else:
        lines.append("Connection: keep-alive")

    lines.append("")
    lines.append("")
    return "\r\n".join(lines).encode("utf-8") + body


def _json_response(data: Any, status: int = 200) -> bytes:
    """Encode *data* as a JSON HTTP response."""
    body = json.dumps(data, ensure_ascii=False, indent=None).encode("utf-8")
    return _make_response(status, body)


def _openai_error(
    status: int,
    message: str,
    error_type: str = "server_error",
) -> bytes:
    """Build an OpenAI-compatible error response."""
    return _json_response(
        {
            "error": {
                "message": message,
                "type": error_type,
                "code": status,
            }
        },
        status=status,
    )


async def _read_http_request(
    reader: asyncio.StreamReader,
    max_body_bytes: int,
    max_headers_bytes: int,
) -> _HttpRequest | None:
    """Parse one HTTP request from *reader*.

    Returns ``None`` if the client closed the connection cleanly before
    sending a request.
    """
    # ── Read request line ──────────────────────────────────────────────
    try:
        request_line = await asyncio.wait_for(
            reader.readline(), timeout=30.0
        )
    except asyncio.TimeoutError:
        return None

    if not request_line:
        # EOF — client disconnected cleanly.
        return None

    request_line = request_line.rstrip(b"\r\n")
    if not request_line:
        return None

    parts = request_line.split(b" ", 2)
    if len(parts) != 3:
        return None

    method = parts[0].decode("ascii", errors="replace").upper()
    path = parts[1].decode("ascii", errors="replace")
    version = parts[2].decode("ascii", errors="replace")

    # ── Read headers ──────────────────────────────────────────────────
    headers: dict[str, str] = {}
    total_header_bytes = 0
    while True:
        line = await reader.readline()
        total_header_bytes += len(line)
        if total_header_bytes > max_headers_bytes:
            return None
        line = line.rstrip(b"\r\n")
        if not line:
            break
        if b":" in line:
            key_bytes, value_bytes = line.split(b":", 1)
            key = key_bytes.decode("ascii", errors="replace").strip().lower()
            value = value_bytes.decode("ascii", errors="replace").strip()
            headers[key] = value

    # ── Read body ─────────────────────────────────────────────────────
    content_length_str = headers.get("content-length", "0")
    try:
        content_length = int(content_length_str)
    except ValueError:
        content_length = 0

    if content_length > max_body_bytes:
        # Read and discard the body so the connection stays clean.
        await reader.readexactly(content_length)
        raise ApiError(
            413,
            f"Request body too large ({content_length} bytes, "
            f"max {max_body_bytes})",
            error_type="invalid_request_error",
        )

    body = b""
    if content_length > 0:
        body = await reader.readexactly(content_length)

    return _HttpRequest(method=method, path=path, headers=headers, body=body, version=version)


# ── OpenAI response builders ───────────────────────────────────────────────────


def _build_chat_response(
    content: str,
    model: str,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat completion response object."""
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "fp_swarm_stub_0001",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": max(1, len(content) // 4),
            "total_tokens": 0,
        },
    }


def _build_chat_chunk(
    content: str,
    model: str,
    *,
    finish_reason: str | None = None,
    role: str | None = None,
    index: int = 0,
) -> dict[str, Any]:
    """Build an OpenAI-compatible streaming chat chunk."""
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content:
        delta["content"] = content

    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "fp_swarm_stub_0001",
        "choices": [
            {
                "index": index,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


# ── ApiServer ──────────────────────────────────────────────────────────────────


class ApiServer:
    """OpenAI-compatible HTTP API fronting the Swarm fleet.

    Binds an HTTP/1.1 server on *bind*:*port* and exposes three endpoints:

    - ``POST /v1/chat/completions`` — chat completions (OpenAI-compatible)
    - ``GET /v1/models`` — list loaded models
    - ``GET /health`` — fleet status and readiness

    Parameters
    ----------
    instance:
        The ``InstanceManager`` that tracks fleet state and runs generation.
    bind:
        IP address to bind to.  Defaults to ``"127.0.0.1"`` — must be
        explicitly changed to ``"0.0.0.0"`` to listen on all interfaces.
    port:
        TCP port to bind to (default 8000).
    max_concurrent:
        Maximum concurrent requests before returning 503 (default 32).
    max_body_bytes:
        Maximum request body size in bytes (default 1 MB).
    generate_timeout:
        Seconds before a generate call is cancelled (default 300).
    """

    def __init__(
        self,
        *,
        instance: InstanceManager,
        bind: str = DEFAULT_BIND,
        port: int = DEFAULT_PORT,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        generate_timeout: float = DEFAULT_GENERATE_TIMEOUT,
    ) -> None:
        self._instance = instance
        self._bind = bind
        self._port = port
        self._max_concurrent = max_concurrent
        self._max_body_bytes = max_body_bytes
        self._generate_timeout = generate_timeout

        self._server: asyncio.Server | None = None
        self._concurrency_sem: asyncio.Semaphore = asyncio.Semaphore(max_concurrent)
        self._background_tasks: set[asyncio.Task[Any]] = set()

    # ── Public API ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the HTTP server.  Idempotent."""
        if self._server is not None:
            return

        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self._bind,
            port=self._port,
        )
        logger.info(
            "ApiServer listening on %s:%d (model=%s, max_concurrent=%d)",
            self._bind,
            self._port,
            self._instance.model_name,
            self._max_concurrent,
        )

    async def stop(self) -> None:
        """Stop the HTTP server and cancel all in-flight handlers."""
        if self._server is not None:
            self._server.close()
            with contextlib_suppress(Exception):
                await self._server.wait_closed()
            self._server = None

        for task in list(self._background_tasks):
            if task.done():
                continue
            task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await task
        self._background_tasks.clear()

        logger.info("ApiServer stopped")

    # ── Connection handling ────────────────────────────────────────────────

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle one TCP connection (may carry multiple HTTP/1.1 requests)."""
        peername = writer.get_extra_info("peername")
        peer_str = f"{peername[0]}:{peername[1]}" if peername else "unknown"
        logger.debug("Connection from %s", peer_str)

        keep_alive = True
        try:
            while keep_alive:
                try:
                    request = await _read_http_request(
                        reader, self._max_body_bytes, DEFAULT_MAX_HEADERS_BYTES
                    )
                except ApiError as exc:
                    writer.write(_openai_error(exc.status, exc.message, exc.error_type))
                    await writer.drain()
                    break
                except Exception:
                    break

                if request is None:
                    break

                # The client may request close.
                connection_hdr = request.headers.get("connection", "").lower()
                if connection_hdr == "close":
                    keep_alive = False

                response = await self._dispatch(request)

                # Override Connection header if the client asked to close.
                if not keep_alive:
                    # Strip the trailing \r\n\r\n so we can re-encode.
                    body_start = response.find(b"\r\n\r\n") + 4
                    header_part = response[:body_start].decode("utf-8")
                    # Replace Connection: keep-alive with Connection: close
                    header_part = header_part.replace(
                        "Connection: keep-alive", "Connection: close"
                    )
                    response = header_part.encode("utf-8") + response[body_start:]

                writer.write(response)
                try:
                    await writer.drain()
                except (ConnectionError, OSError):
                    break
        except (ConnectionError, OSError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    # ── Dispatch ───────────────────────────────────────────────────────────

    async def _dispatch(self, request: _HttpRequest) -> bytes:
        """Route an HTTP request to the appropriate handler.

        All responses go through this method, so we can add auth, logging,
        and metrics in one place (see AUTHENTICATION_INSERTION_POINT above).
        """
        t0 = time.monotonic()

        # ── Route ──────────────────────────────────────────────────────
        if request.method == "POST" and request.path == "/v1/chat/completions":
            response = await self._handle_chat_completions(request)
        elif request.method == "GET" and request.path == "/v1/models":
            response = await self._handle_models()
        elif request.method == "GET" and request.path == "/health":
            response = await self._handle_health()
        elif request.method == "GET" and request.path == "/":
            response = _make_response(
                200,
                b'{"service":"swarm-api","version":"0.1.0","endpoints":["/v1/chat/completions","/v1/models","/health"]}\n',
            )
        else:
            response = _openai_error(404, "Not found", "invalid_request_error")

        elapsed = time.monotonic() - t0
        logger.debug(
            "%s %s → %d (%.1f ms)",
            request.method,
            request.path,
            _status_from_response(response),
            elapsed * 1000,
        )
        return response

    # ── Handler: POST /v1/chat/completions ─────────────────────────────────

    async def _handle_chat_completions(self, request: _HttpRequest) -> bytes:
        """Handle a chat completions request."""
        # ── Concurrency gate ───────────────────────────────────────────
        if self._concurrency_sem.locked() or not await self._try_acquire_slot():
            return _openai_error(
                503,
                "Server at capacity — too many concurrent requests. "
                f"Limit is {self._max_concurrent}. Retry later.",
                error_type="rate_limit_exceeded",
            )
        try:
            return await self._handle_chat_completions_impl(request)
        finally:
            self._concurrency_sem.release()

    async def _try_acquire_slot(self) -> bool:
        """Try to acquire a concurrency slot without blocking.

        ``Semaphore.acquire()`` always suspends; we check ``locked()``
        as a fast path and use a tiny timeout as the actual non-blocking
        acquire.
        """
        try:
            async with asyncio.timeout(0.001):
                await self._concurrency_sem.acquire()
            return True
        except asyncio.TimeoutError:
            return False

    async def _handle_chat_completions_impl(self, request: _HttpRequest) -> bytes:
        # ── Fleet ready check ──────────────────────────────────────────
        if not self._instance.ready:
            return _openai_error(
                503,
                "Fleet is not converged — a reshard is in progress. "
                "Retry in a moment.",
                error_type="service_unavailable",
            )

        # ── Parse JSON body ────────────────────────────────────────────
        try:
            body_json = json.loads(request.body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return _openai_error(
                400,
                f"Invalid JSON in request body: {exc}",
                "invalid_request_error",
            )

        if not isinstance(body_json, dict):
            return _openai_error(
                400,
                "Request body must be a JSON object",
                "invalid_request_error",
            )

        # ── Validate model ─────────────────────────────────────────────
        model_requested = body_json.get("model", "")
        if model_requested and model_requested != self._instance.model_name:
            return _openai_error(
                404,
                f"Model '{model_requested}' is not loaded. "
                f"Available: '{self._instance.model_name}'",
                "invalid_request_error",
            )

        # ── Extract messages ──────────────────────────────────────────
        messages: list[dict[str, Any]] = body_json.get("messages", [])
        if not messages or not isinstance(messages, list):
            return _openai_error(
                400,
                "Missing or empty 'messages' array",
                "invalid_request_error",
            )

        # ── Extract params ─────────────────────────────────────────────
        params: dict[str, Any] = {
            "max_tokens": body_json.get("max_tokens", 256),
            "temperature": body_json.get("temperature", 0.7),
            "top_p": body_json.get("top_p", 1.0),
            "stream": body_json.get("stream", False),
        }
        streaming = bool(params["stream"])

        # ── Call generate ──────────────────────────────────────────────
        try:
            async with asyncio.timeout(self._generate_timeout):
                result = await self._instance.generate(messages, params)
        except asyncio.TimeoutError:
            return _openai_error(
                504,
                f"Generation timed out after {self._generate_timeout:.0f}s",
                error_type="server_error",
            )
        except ApiError as exc:
            return _openai_error(exc.status, exc.message, exc.error_type)

        # ── Build response ─────────────────────────────────────────────
        model_name = self._instance.model_name
        if streaming and hasattr(result, "__aiter__"):
            return await self._build_streaming_response(result, model_name)
        elif isinstance(result, str):
            return _json_response(_build_chat_response(result, model_name))
        else:
            return _openai_error(
                500,
                f"generate_fn returned unexpected type: {type(result).__name__}",
            )

    async def _build_streaming_response(
        self, chunks: Any, model: str
    ) -> bytes:
        """Build a complete SSE streaming response.

        Because we buffer the whole stream before writing, this is not true
        streaming — but it avoids the complexity of holding a writer open
        while generating, which interacts badly with concurrency limits
        and connection keep-alive.  True chunked transfer can come later.
        """
        sse_parts: list[bytes] = []

        # Role-assignment chunk first (OpenAI convention: role in the first
        # delta, empty content).
        sse_parts.append(
            b"data: "
            + json.dumps(
                _build_chat_chunk("", model, finish_reason=None, role="assistant"),
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n\n"
        )

        async for token in chunks:
            sse_parts.append(
                b"data: "
                + json.dumps(
                    _build_chat_chunk(
                        str(token),
                        model,
                        finish_reason=None,
                    ),
                    ensure_ascii=False,
                ).encode("utf-8")
                + b"\n\n"
            )
        # Final chunk with finish_reason.
        sse_parts.append(
            b"data: "
            + json.dumps(
                _build_chat_chunk("", model, finish_reason="stop"),
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n\n"
        )
        sse_parts.append(b"data: [DONE]\n\n")

        body = b"".join(sse_parts)
        return _make_response(
            200,
            body,
            content_type="text/event-stream",
            extra_headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ── Handler: GET /v1/models ────────────────────────────────────────────

    async def _handle_models(self) -> bytes:
        """List the loaded model."""
        model_name = self._instance.model_name
        return _json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "swarm",
                    }
                ],
            }
        )

    # ── Handler: GET /health ───────────────────────────────────────────────

    async def _handle_health(self) -> bytes:
        """Report fleet status."""
        assignment = self._instance.assignment
        return _json_response(
            {
                "status": "ok" if self._instance.ready else "degraded",
                "fleet_size": self._instance.fleet_size,
                "assignment_hash": assignment.fleet_hash if assignment else None,
                "converged": self._instance.ready,
                "uptime_seconds": self._instance.uptime_seconds,
                "model": self._instance.model_name,
            }
        )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _status_from_response(data: bytes) -> int:
    """Extract the HTTP status code from a raw response."""
    try:
        end = data.index(b"\r\n")
        line = data[:end].decode("ascii")
        return int(line.split(" ")[1])
    except (ValueError, IndexError):
        return 0


# ── Stub generate function ─────────────────────────────────────────────────────


async def stub_generate(
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    assignment: ShardAssignment,
) -> str | AsyncIterator[str]:
    """Stub generate function — returns clearly-marked placeholder text.

    This exists so the API server is runnable before layers 1 and 2
    (inference core, expert streaming) are built.  Every response is
    prefixed with ``[STUB]`` and carries an unambiguous disclaimer.

    Callers can programmatically distinguish a stub response by checking
    for the ``[STUB]`` prefix at the start of ``choices[0].message.content``
    (non-streaming) or the first ``delta.content`` (streaming).
    """
    streaming = bool(params.get("stream", False))

    # Extract the user's last message for a vaguely relevant echo.
    user_text = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_text = str(msg.get("content", ""))
    user_text = user_text.strip()

    total = params.get("max_tokens", 256)

    prefix = (
        f"[STUB — no inference engine loaded] "
        f"Heard: \"{user_text[:120]}\". "
        f"Params: max_tokens={total}, temperature={params.get('temperature', 0.7):.1f}. "
        f"Fleet: {len(assignment.node_counts)} nodes, "
        f"model layer would be sharded across "
        f"{len([c for c in assignment.node_counts.values() if c > 0])} "
        f"of them. "
        f"This is placeholder text — layers 1 and 2 (inference core, "
        f"expert streaming) are not built yet. "
        f"Prompt length: {len(user_text)} chars."
    )

    if streaming:
        async def _stream() -> AsyncIterator[str]:
            words = prefix.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                await asyncio.sleep(0.01)  # simulate inter-token latency
        return _stream()
    else:
        return prefix


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════


async def _demo() -> None:
    """Run a self-contained demo of the API server.

    Simulates a 3-node fleet on localhost (2 peers + self), starts the
    API server, and prints a curl command the user can run to verify
    it works.
    """
    import signal

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("swarm.api_server").setLevel(logging.INFO)

    # ── Simulated fleet ────────────────────────────────────────────────
    OWN_ID = "api-node-aabb-cccc-dddd-eeeeeeeeeeee"
    PEER_A = "peer-aaaa-0000-0000-0000-000000000000"
    PEER_B = "peer-bbbb-0000-0000-0000-000000000000"

    fleet = _FakeFleetTable(own_node_id=OWN_ID)

    failover = FailoverCoordinator(
        fleet_table=fleet,
        own_node_id=OWN_ID,
        num_experts=64,
        own_storage_bandwidth_mbps=4000,
        settle_window=0.3,
    )
    await failover.start()

    instance = InstanceManager(
        failover=failover,
        model_name="glm-5.2-stub",
        generate_fn=stub_generate,
        reshard_grace=0.1,
    )
    await instance.start()

    # Add two peers so the fleet has 3 total (self + 2 peers).
    await fleet.add_node(_make_desc(PEER_A, "alpha", 4000))
    await fleet.add_node(_make_desc(PEER_B, "bravo", 1900))
    await asyncio.sleep(0.8)  # let settle window expire + reshard land

    # ── Start API server ───────────────────────────────────────────────
    server = ApiServer(
        instance=instance,
        bind="127.0.0.1",
        port=8000,
        max_concurrent=32,
    )
    await server.start()

    print("=" * 72)
    print("  Swarm API Server — demo")
    print("=" * 72)
    print(f"  Model:       {instance.model_name}")
    print(f"  Fleet:       3 nodes (self + 2 simulated peers)")
    print(f"  Converged:   {instance.ready}")
    print(f"  Assignment:  {instance.assignment.fleet_hash[:16] if instance.assignment else '(none)'}…")
    print(f"  Listening:   http://127.0.0.1:8000")
    print()
    print("  Try these curl commands:")
    print()
    print("    # Health check")
    print('    curl -s http://127.0.0.1:8000/health | python3 -m json.tool')
    print()
    print("    # List models")
    print('    curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool')
    print()
    print("    # Chat completion")
    print('    curl -s http://127.0.0.1:8000/v1/chat/completions \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -d \'{"model":"glm-5.2-stub","messages":[{"role":"user","content":"Hello, Swarm!"}],"max_tokens":100}\' | python3 -m json.tool')
    print()
    print("  Press Ctrl+C to stop.")
    print("=" * 72)

    # ── Run until interrupted ──────────────────────────────────────────
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            # Windows — signal handlers not supported; Ctrl+C raises
            # KeyboardInterrupt in the main task instead.
            pass

    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        pass

    print("\nShutting down…")
    await server.stop()
    await instance.stop()
    print("Done.")


def main() -> None:
    """Entry point: run the demo or start a production server.

    Use ``--demo`` for the interactive demo, or pass ``--bind`` and
    ``--port`` to run as a standalone server.
    """
    parser = argparse.ArgumentParser(
        description="Swarm API server — OpenAI-compatible HTTP endpoint."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the self-contained demo (simulated fleet on localhost).",
    )
    parser.add_argument(
        "--bind",
        type=str,
        default=DEFAULT_BIND,
        help=f"IP address to bind to (default: {DEFAULT_BIND}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port to listen on (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help=f"Maximum concurrent requests (default: {DEFAULT_MAX_CONCURRENT}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.demo:
        asyncio.run(_demo())
    else:
        print(
            "Production mode: requires a real FleetTable + FailoverCoordinator. "
            "Use --demo for the self-contained demo, or wire this module into "
            "your own startup script."
        )


if __name__ == "__main__":
    main()
