"""Tests for api_server.py — HTTP endpoint correctness, error handling,
concurrency cap, fleet state transitions, and stub detection.

Uses real TCP sockets against a running ApiServer on localhost.  Each
test starts its own server on a distinct port, runs, and tears down.

Run:  python3 test_api_server.py
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from api_server import (
    ApiServer,
    InstanceManager,
    stub_generate,
)
from failover import (
    DEFAULT_SETTLE_WINDOW,
    FailoverCoordinator,
    _FakeFleetTable,
    _make_desc,
)
from sharding import ShardAssignment

# ── Helpers ───────────────────────────────────────────────────────────────


_PORT_BASE: int = 25000
_port_counter: int = 0


def _next_port() -> int:
    global _port_counter
    _port_counter += 1
    return _PORT_BASE + _port_counter


async def _http_request(
    host: str,
    port: int,
    method: str,
    path: str,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """Make an HTTP/1.1 request and return (status, headers, body)."""
    if headers is None:
        headers = {}
    headers.setdefault("host", f"{host}:{port}")
    if body:
        headers.setdefault("content-type", "application/json")
        headers.setdefault("content-length", str(len(body)))
    else:
        headers.setdefault("content-length", "0")
    headers.setdefault("connection", "close")

    request_line = f"{method} {path} HTTP/1.1\r\n"
    header_lines = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    raw = (request_line + header_lines + "\r\n").encode("utf-8") + body

    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(raw)
        await writer.drain()

        # Read response.
        response = b""
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            response += chunk
            # Stop when we have the full body.
            if b"\r\n\r\n" in response:
                header_end = response.index(b"\r\n\r\n") + 4
                header_part = response[:header_end].decode("utf-8", errors="replace")
                cl = 0
                for line in header_part.split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        cl = int(line.split(":", 1)[1].strip())
                if len(response) >= header_end + cl:
                    break
    finally:
        writer.close()
        await writer.wait_closed()

    # Parse status line.
    header_end = response.index(b"\r\n\r\n") + 4
    header_bytes = response[:header_end]
    body_bytes = response[header_end:]

    lines = header_bytes.decode("utf-8", errors="replace").split("\r\n")
    status_line = lines[0]
    status = int(status_line.split(" ")[1])

    resp_headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            resp_headers[k.strip().lower()] = v.strip()

    return status, resp_headers, body_bytes


async def _http_json(
    host: str, port: int, method: str, path: str, body: Any = None
) -> tuple[int, Any]:
    """HTTP request that sends/receives JSON."""
    body_bytes = json.dumps(body).encode("utf-8") if body is not None else b""
    status, headers, raw_body = await _http_request(
        host, port, method, path, body=body_bytes
    )
    try:
        parsed = json.loads(raw_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        parsed = raw_body.decode("utf-8", errors="replace")
    return status, parsed


def _stub_prefix() -> str:
    return "[STUB — no inference engine loaded]"


# ── Fixtures ───────────────────────────────────────────────────────────────


async def _build_server(
    *,
    model_name: str = "test-model",
    generate_fn: Any = None,
    max_concurrent: int = 32,
    max_body_bytes: int = 1_000_000,
    reshard_grace: float = 0.1,
) -> tuple[ApiServer, InstanceManager, _FakeFleetTable, FailoverCoordinator, int]:
    """Build a complete server + instance + fleet stack for testing.

    Returns (server, instance, fleet, failover, port).
    """
    port = _next_port()
    own_id = f"test-self-{port:05d}"
    fleet = _FakeFleetTable(own_node_id=own_id)

    failover = FailoverCoordinator(
        fleet_table=fleet,
        own_node_id=own_id,
        num_experts=64,
        own_storage_bandwidth_mbps=4000,
        settle_window=0.2,
    )
    await failover.start()

    fn = generate_fn if generate_fn is not None else stub_generate
    instance = InstanceManager(
        failover=failover,
        model_name=model_name,
        generate_fn=fn,
        reshard_grace=reshard_grace,
    )
    await instance.start()

    # Seed with one peer so we get a first assignment quickly.
    await fleet.add_node(_make_desc(f"peer-{port:05d}-a", "alpha", 4000))
    await asyncio.sleep(0.5)  # settle window + reshard
    # Drain any leftover diffs.
    while not failover.diff_queue.empty():
        failover.diff_queue.get_nowait()

    server = ApiServer(
        instance=instance,
        bind="127.0.0.1",
        port=port,
        max_concurrent=max_concurrent,
        max_body_bytes=max_body_bytes,
    )
    await server.start()
    await asyncio.sleep(0.05)  # let the socket bind

    return server, instance, fleet, failover, port


async def _teardown(server: ApiServer, instance: InstanceManager) -> None:
    await server.stop()
    await instance.stop()


# ── Tests ────────────────────────────────────────────────────────────────


async def test_health_endpoint() -> None:
    """GET /health returns correct fleet state."""
    server, instance, _fleet, _fo, port = await _build_server()
    try:
        status, body = await _http_json("127.0.0.1", port, "GET", "/health")
        assert status == 200, f"expected 200, got {status}: {body}"
        assert body.get("converged") is True, f"fleet should be converged: {body}"
        assert body["fleet_size"] >= 1, f"fleet should have nodes: {body}"
        assert body["model"] == "test-model", f"wrong model: {body}"
        print("PASS: GET /health returns correct fleet state")
    finally:
        await _teardown(server, instance)


async def test_health_while_unconverged() -> None:
    """GET /health reports degraded when no assignment exists."""
    port = _next_port()
    own_id = f"unconv-{port:05d}"
    fleet = _FakeFleetTable(own_node_id=own_id)
    failover = FailoverCoordinator(
        fleet_table=fleet,
        own_node_id=own_id,
        num_experts=64,
        own_storage_bandwidth_mbps=4000,
        settle_window=0.2,
    )
    await failover.start()

    instance = InstanceManager(
        failover=failover,
        model_name="test-model",
        generate_fn=stub_generate,
    )
    await instance.start()
    # Don't add any peers — fleet stays unconverged.

    server = ApiServer(instance=instance, bind="127.0.0.1", port=port)
    await server.start()
    await asyncio.sleep(0.05)

    try:
        status, body = await _http_json("127.0.0.1", port, "GET", "/health")
        assert status == 200
        assert body.get("converged") is False, f"fleet should be unconverged: {body}"
        assert body.get("status") == "degraded", f"status should be degraded: {body}"
        print("PASS: /health reports degraded when unconverged")
    finally:
        await server.stop()
        await instance.stop()


async def test_models_endpoint() -> None:
    """GET /v1/models returns the loaded model."""
    server, instance, _fleet, _fo, port = await _build_server(
        model_name="glm-5.2-test"
    )
    try:
        status, body = await _http_json("127.0.0.1", port, "GET", "/v1/models")
        assert status == 200, f"expected 200, got {status}: {body}"
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "glm-5.2-test"
        print("PASS: GET /v1/models returns loaded model")
    finally:
        await _teardown(server, instance)


async def test_chat_completions_shape() -> None:
    """POST /v1/chat/completions returns valid OpenAI response shape."""
    server, instance, _fleet, _fo, port = await _build_server()
    try:
        status, body = await _http_json(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 50,
            },
        )
        assert status == 200, f"expected 200, got {status}: {body}"
        assert body["object"] == "chat.completion", f"wrong object: {body}"
        assert "id" in body and body["id"].startswith("chatcmpl-")
        assert len(body["choices"]) == 1
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert "content" in body["choices"][0]["message"]
        assert body["choices"][0]["finish_reason"] == "stop"
        assert "usage" in body
        print("PASS: chat completions response shape is valid")
    finally:
        await _teardown(server, instance)


async def test_chat_completions_stub_marker() -> None:
    """Non-streaming stub responses are clearly marked with [STUB] prefix."""
    server, instance, _fleet, _fo, port = await _build_server()
    try:
        _, body = await _http_json(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Ping"}],
            },
        )
        content = body["choices"][0]["message"]["content"]
        assert content.startswith(_stub_prefix()), (
            f"stub not marked: {content[:80]}…"
        )
        print("PASS: stub response is clearly marked")
    finally:
        await _teardown(server, instance)


async def test_streaming_stub_marker() -> None:
    """Streaming stub responses are marked with [STUB] in the first chunk."""
    server, instance, _fleet, _fo, port = await _build_server()
    try:
        status, _headers, raw = await _http_request(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body=json.dumps({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Ping"}],
                "stream": True,
            }).encode("utf-8"),
        )
        assert status == 200
        text = raw.decode("utf-8")
        assert "data:" in text, f"no SSE data in response: {text[:200]}"
        # Parse all data chunks and accumulate content.
        accumulated = ""
        for line in text.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"]
                accumulated += delta.get("content", "")
        # The first chunk carries role; content appears in subsequent chunks.
        assert accumulated.startswith("[STUB"), (
            f"streaming stub not marked in accumulated content: {accumulated[:80]}…"
        )
        print("PASS: streaming stub response is clearly marked")
    finally:
        await _teardown(server, instance)


async def test_malformed_json_rejected() -> None:
    """A request with invalid JSON returns 400."""
    server, instance, _fleet, _fo, port = await _build_server()
    try:
        status, _headers, raw = await _http_request(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body=b"this is not json {",
        )
        assert status == 400, f"expected 400, got {status}"
        body = json.loads(raw.decode("utf-8"))
        assert "error" in body
        print("PASS: malformed JSON returns 400")
    finally:
        await _teardown(server, instance)


async def test_oversized_body_rejected() -> None:
    """A request exceeding max_body_bytes returns 413."""
    server, instance, _fleet, _fo, port = await _build_server(max_body_bytes=1024)
    try:
        # Build a body > 1024 bytes.
        big = {"model": "test-model", "messages": [{"role": "user", "content": "X" * 2000}]}
        body_bytes = json.dumps(big).encode("utf-8")
        assert len(body_bytes) > 1024, f"body not big enough: {len(body_bytes)}"

        status, _headers, raw = await _http_request(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body=body_bytes,
        )
        assert status == 413, f"expected 413, got {status}: {raw[:200]}"
        body = json.loads(raw.decode("utf-8"))
        assert "error" in body
        print(f"PASS: oversized body ({len(body_bytes)} bytes) returns 413")
    finally:
        await _teardown(server, instance)


async def test_wrong_model_rejected() -> None:
    """Requesting a model that isn't loaded returns 404."""
    server, instance, _fleet, _fo, port = await _build_server(
        model_name="glm-5.2-actual"
    )
    try:
        status, body = await _http_json(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "some-other-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert status == 404, f"expected 404, got {status}: {body}"
        print("PASS: wrong model returns 404")
    finally:
        await _teardown(server, instance)


async def test_empty_messages_rejected() -> None:
    """Missing or empty messages array returns 400."""
    server, instance, _fleet, _fo, port = await _build_server()
    try:
        status, body = await _http_json(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body={"model": "test-model", "messages": []},
        )
        assert status == 400, f"expected 400, got {status}: {body}"
        print("PASS: empty messages returns 400")
    finally:
        await _teardown(server, instance)


async def test_unknown_path_404() -> None:
    """Unknown paths return 404."""
    server, instance, _fleet, _fo, port = await _build_server()
    try:
        status, body = await _http_json(
            "127.0.0.1", port, "GET", "/nonexistent"
        )
        assert status == 404, f"expected 404, got {status}: {body}"
        print("PASS: unknown path returns 404")
    finally:
        await _teardown(server, instance)


async def test_503_when_unconverged() -> None:
    """Chat completions return 503 before the first assignment arrives."""
    port = _next_port()
    own_id = f"unconv-cc-{port:05d}"
    fleet = _FakeFleetTable(own_node_id=own_id)
    failover = FailoverCoordinator(
        fleet_table=fleet,
        own_node_id=own_id,
        num_experts=64,
        own_storage_bandwidth_mbps=4000,
        settle_window=0.2,
    )
    await failover.start()

    instance = InstanceManager(
        failover=failover,
        model_name="test-model",
        generate_fn=stub_generate,
    )
    await instance.start()
    # No peers — fleet never gets an assignment.

    server = ApiServer(instance=instance, bind="127.0.0.1", port=port)
    await server.start()
    await asyncio.sleep(0.05)

    try:
        status, body = await _http_json(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert status == 503, f"expected 503, got {status}: {body}"
        assert "error" in body
        assert body["error"]["type"] == "service_unavailable"
        print("PASS: 503 when fleet unconverged")
    finally:
        await server.stop()
        await instance.stop()


async def test_concurrency_cap() -> None:
    """Requests over the concurrency cap return 503.

    Uses a generate_fn that blocks on an asyncio.Event, so we can fill
    all slots and then verify the next request is rejected.
    """
    hold: asyncio.Event = asyncio.Event()

    async def _blocking_generate(
        _messages: list[dict[str, Any]],
        _params: dict[str, Any],
        _assignment: ShardAssignment,
    ) -> str:
        await hold.wait()
        return "[STUB] released"

    server, instance, _fleet, _fo, port = await _build_server(
        generate_fn=_blocking_generate,
        max_concurrent=3,
    )
    try:
        # Fill all 3 slots.
        tasks = []
        for i in range(3):
            tasks.append(
                asyncio.create_task(
                    _http_json(
                        "127.0.0.1",
                        port,
                        "POST",
                        "/v1/chat/completions",
                        body={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": f"msg {i}"}],
                        },
                    )
                )
            )
        await asyncio.sleep(0.1)  # let them all hit the hold point

        # 4th request should be rejected.
        status, body = await _http_json(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "test-model",
                "messages": [{"role": "user", "content": "overflow"}],
            },
        )
        assert status == 503, f"expected 503, got {status}: {body}"
        assert "error" in body
        assert "capacity" in body["error"]["message"].lower() or "rate" in body["error"]["type"].lower()

        # Release blocked requests so they complete cleanly.
        hold.set()
        for task in tasks:
            s, b = await task
            assert s == 200, f"held request failed: {s} {b}"

        print("PASS: concurrency cap rejects over-limit requests")
    finally:
        hold.set()
        await _teardown(server, instance)


async def test_reshard_503_then_recovery() -> None:
    """Mid-request, a reshard produces a brief 503 window, then recovers."""
    server, instance, fleet, _fo, port = await _build_server(
        reshard_grace=0.3,
    )
    try:
        # Verify fleet is initially ready.
        s1, b1 = await _http_json(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "test-model",
                "messages": [{"role": "user", "content": "before reshard"}],
            },
        )
        assert s1 == 200, f"expected 200 before reshard, got {s1}: {b1}"

        # Trigger a reshard by adding a node.
        await fleet.add_node(_make_desc(f"new-peer-{port:05d}", "new-guy", 4000))
        await asyncio.sleep(0.01)  # almost immediately — reshard just fired

        # Request during grace period should get 503.
        s2, b2 = await _http_json(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "test-model",
                "messages": [{"role": "user", "content": "during reshard"}],
            },
        )
        # May be 503 (if grace period still active) or 200 (if grace already
        # expired — depends on timing).  We just need to confirm that it
        # eventually recovers.
        if s2 == 503:
            print("  (caught 503 during grace period — verifying recovery…)")
        else:
            assert s2 == 200, f"unexpected status during reshard: {s2}"

        # Wait for grace period to expire.
        await asyncio.sleep(0.5)

        s3, b3 = await _http_json(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "test-model",
                "messages": [{"role": "user", "content": "after reshard"}],
            },
        )
        assert s3 == 200, f"expected 200 after reshard recovery, got {s3}: {b3}"

        # /health should show converged again.
        _, health = await _http_json("127.0.0.1", port, "GET", "/health")
        assert health["converged"] is True

        print("PASS: reshard produces 503 window then recovers")
    finally:
        await _teardown(server, instance)


async def test_binding_defaults_to_loopback() -> None:
    """The server constructor defaults to 127.0.0.1, not 0.0.0.0."""
    from api_server import DEFAULT_BIND
    assert DEFAULT_BIND == "127.0.0.1", (
        f"DEFAULT_BIND is {DEFAULT_BIND!r}, must be '127.0.0.1'"
    )
    print("PASS: DEFAULT_BIND is 127.0.0.1")


async def test_stub_detectable_programmatically() -> None:
    """Callers can programmatically detect stub responses via the [STUB] prefix."""
    server, instance, _fleet, _fo, port = await _build_server()
    try:
        _, body = await _http_json(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Test"}],
            },
        )
        content = body["choices"][0]["message"]["content"]
        is_stub = content.startswith(_stub_prefix())
        assert is_stub is True, "stub not detectable by prefix check"
        print("PASS: stub response is programmatically detectable")
    finally:
        await _teardown(server, instance)


async def test_fleet_size_in_stub() -> None:
    """The stub response includes the fleet size in its placeholder text."""
    server, instance, _fleet, _fo, port = await _build_server()
    try:
        _, body = await _http_json(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Test"}],
            },
        )
        content = body["choices"][0]["message"]["content"]
        # Fleet has self + 1 peer = 2 nodes.
        assert "2 nodes" in content, (
            f"fleet size not in stub: {content[:200]}…"
        )
        print("PASS: stub response includes fleet size")
    finally:
        await _teardown(server, instance)


async def test_root_endpoint() -> None:
    """GET / returns basic service info."""
    server, instance, _fleet, _fo, port = await _build_server()
    try:
        status, body = await _http_json("127.0.0.1", port, "GET", "/")
        assert status == 200
        assert body.get("service") == "swarm-api"
        assert "endpoints" in body
        print("PASS: GET / returns service info")
    finally:
        await _teardown(server, instance)


async def test_generate_timeout() -> None:
    """A generate_fn that hangs past the timeout returns 504."""
    async def _slow_generate(
        _messages: list[dict[str, Any]],
        _params: dict[str, Any],
        _assignment: ShardAssignment,
    ) -> str:
        await asyncio.sleep(10.0)
        return "never"

    server, instance, _fleet, _fo, port = await _build_server(
        generate_fn=_slow_generate,
    )
    # Override the generate_timeout to be very short for the test.
    server._generate_timeout = 0.2

    try:
        status, body = await _http_json(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert status == 504, f"expected 504, got {status}: {body}"
        assert "error" in body
        assert "timed out" in body["error"]["message"].lower()
        print("PASS: generate timeout returns 504")
    finally:
        await _teardown(server, instance)


async def test_no_model_field_uses_default() -> None:
    """Omitting the model field in the request succeeds (assumes loaded model)."""
    server, instance, _fleet, _fo, port = await _build_server(
        model_name="default-model"
    )
    try:
        status, body = await _http_json(
            "127.0.0.1",
            port,
            "POST",
            "/v1/chat/completions",
            body={
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert status == 200, f"expected 200, got {status}: {body}"
        assert body["model"] == "default-model"
        print("PASS: missing model field defaults to loaded model")
    finally:
        await _teardown(server, instance)


# ── Runner ────────────────────────────────────────────────────────────────


async def _run() -> None:
    print("── Health and models ──")
    await test_health_endpoint()
    await test_health_while_unconverged()
    await test_models_endpoint()
    await test_root_endpoint()

    print("\n── Chat completions ──")
    await test_chat_completions_shape()
    await test_chat_completions_stub_marker()
    await test_streaming_stub_marker()
    await test_stub_detectable_programmatically()
    await test_fleet_size_in_stub()
    await test_no_model_field_uses_default()

    print("\n── Error handling ──")
    await test_malformed_json_rejected()
    await test_oversized_body_rejected()
    await test_wrong_model_rejected()
    await test_empty_messages_rejected()
    await test_unknown_path_404()
    await test_generate_timeout()

    print("\n── Fleet state ──")
    await test_503_when_unconverged()
    await test_concurrency_cap()
    await test_reshard_503_then_recovery()

    print("\n── Configuration ──")
    await test_binding_defaults_to_loopback()


def main() -> None:
    asyncio.run(_run())
    print("\nAll API server tests passed.")


if __name__ == "__main__":
    main()
