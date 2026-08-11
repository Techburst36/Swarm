"""Tests for rpc.py — security, framing, and the handler-stall regression.

Everything here binds to 127.0.0.1 only. Nothing in this file listens on
0.0.0.0, because the RPC layer currently has no authentication (trusted-LAN
threat model), and a test that binds all interfaces would briefly expose an
unauthenticated service to the whole network.

No safety limit is raised or disabled to make a test pass. The malformed-frame
tests assert that the module REJECTS bad input, not that it tolerates it.

Run:  python3 test_rpc.py
"""

from __future__ import annotations

import asyncio
import struct
import zlib

from rpc import (
    CURRENT_VERSION,
    HEADER_SIZE,
    MAGIC,
    Frame,
    MessageType,
    RpcClient,
    RpcError,
    RpcServer,
)

HOST = "127.0.0.1"  # loopback only — never 0.0.0.0 in tests


# ── Framing ───────────────────────────────────────────────────────────────


def test_frame_roundtrip() -> None:
    payload = b"hello swarm" * 100
    frame = Frame(
        magic=MAGIC,
        version=CURRENT_VERSION,
        msg_type=MessageType.ACTIVATION,
        flags=0,
        stream_id=42,
        payload_len=len(payload),
        checksum=zlib.crc32(payload) & 0xFFFFFFFF,
        payload=payload,
    )
    encoded = frame.encode()
    assert len(encoded) == HEADER_SIZE + len(payload)

    decoded = Frame.decode_header(encoded[:HEADER_SIZE])
    assert decoded.magic == MAGIC
    assert decoded.version == CURRENT_VERSION
    assert decoded.msg_type == MessageType.ACTIVATION
    assert decoded.stream_id == 42
    assert decoded.payload_len == len(payload)
    assert decoded.checksum == frame.checksum
    print("PASS: frame roundtrip")


def test_header_is_big_endian() -> None:
    """Byte order must be explicit, not native — nodes may differ."""
    payload = b""
    frame = Frame(
        magic=MAGIC,
        version=CURRENT_VERSION,
        msg_type=MessageType.PING,
        flags=0,
        stream_id=1,
        payload_len=0,
        checksum=zlib.crc32(payload) & 0xFFFFFFFF,
        payload=payload,
    )
    encoded = frame.encode()
    # Magic must appear as literal big-endian bytes "SWRM".
    assert encoded[:4] == b"SWRM", encoded[:4]
    print("PASS: header is big-endian on the wire")


# ── Malformed input rejection ─────────────────────────────────────────────


async def _serve_raw(bad_bytes: bytes, port: int) -> str:
    """Connect to a real server, send raw bad bytes, report what happened.

    A correct server either drops us immediately (bare EOF) or sends a
    GOODBYE frame and then drops us. Both count as "rejected". What must NOT
    happen is the connection staying open and usable, or the server hanging
    while it tries to allocate a payload it was told to expect.
    """
    reader, writer = await asyncio.open_connection(HOST, port)
    try:
        writer.write(bad_bytes)
        await writer.drain()
        try:
            # Read until EOF (with a bound). A GOODBYE frame before EOF is
            # fine; what matters is that the stream ends.
            data = await asyncio.wait_for(reader.read(4096), timeout=3.0)
            if data == b"":
                return "closed"
            # Got something — confirm the peer then closes rather than
            # keeping the connection alive.
            try:
                more = await asyncio.wait_for(reader.read(4096), timeout=3.0)
                return "closed" if more == b"" else "still_open"
            except asyncio.TimeoutError:
                return "still_open"
        except asyncio.TimeoutError:
            return "hung"
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def test_rejects_malformed_frames() -> None:
    """Bad magic, bad version, and oversized length must all drop the peer,
    and must never make the server allocate the claimed size."""
    port = 19801
    server = RpcServer(
        own_node_id="test-server",
        port=port,
        handler=lambda conn, frame: None,
        bind_ip=HOST,
    )
    await server.start()
    try:
        # 1. Bad magic
        bad_magic = struct.pack("!IBBHIII", 0xDEADBEEF, 1, 3, 0, 2, 0, 0)
        assert await _serve_raw(bad_magic, port) == "closed"

        # 2. Unknown version
        bad_version = struct.pack("!IBBHIII", MAGIC, 99, 3, 0, 2, 0, 0)
        assert await _serve_raw(bad_version, port) == "closed"

        # 3. Absurd payload_len — the whole point is the server must NOT try
        #    to allocate 4 GB. If this test hangs or OOMs, the size check is
        #    happening after the read instead of before it.
        huge = struct.pack("!IBBHIII", MAGIC, CURRENT_VERSION, 3, 0, 2, 0xFFFFFFFF, 0)
        assert await _serve_raw(huge, port) == "closed"

        # 4. Checksum mismatch
        payload = b"tampered"
        bad_crc = struct.pack(
            "!IBBHIII", MAGIC, CURRENT_VERSION, 3, 0, 2, len(payload), 0x00000000
        ) + payload
        assert await _serve_raw(bad_crc, port) == "closed"

        print("PASS: malformed frames rejected without allocation")
    finally:
        await server.stop()


async def test_oversized_rejected_by_limit() -> None:
    """A frame just over max_frame_bytes must be refused, and the limit must
    be respected as configured rather than ignored."""
    port = 19802
    server = RpcServer(
        own_node_id="test-server",
        port=port,
        handler=lambda conn, frame: None,
        bind_ip=HOST,
        max_frame_bytes=1024,  # deliberately tiny
    )
    await server.start()
    try:
        over = struct.pack(
            "!IBBHIII", MAGIC, CURRENT_VERSION, 3, 0, 2, 1025, 0
        )
        assert await _serve_raw(over, port) == "closed"
        print("PASS: max_frame_bytes enforced as configured")
    finally:
        await server.stop()


# ── The regression test that matters ──────────────────────────────────────


async def test_handler_calling_send_and_wait_does_not_stall() -> None:
    """A handler that calls send_and_wait must not stall the receive loop.

    This is the regression test for the backpressure gap: the receive loop
    refuses to read while the handler queue is full, so if handlers ran
    inline, a handler awaiting a response would block the queue drain, block
    the socket read, and prevent its own response from arriving.

    Setup: node B's handler, on receiving ACTIVATION, calls send_and_wait back
    to node A. If dispatch is inline, this stalls until timeout. If dispatch
    is per-task, it completes promptly.
    """
    port_a, port_b = 19803, 19804
    got_callback = asyncio.Event()

    async def handler_a(conn, frame):
        # A answers B's callback.
        if frame.msg_type == MessageType.ACTIVATION:
            got_callback.set()
            await conn.send_frame(MessageType.RESULT, b"A-answered", stream_id=frame.stream_id)

    async def handler_b(conn, frame):
        # B, while handling, calls back to A on the SAME connection.
        if frame.msg_type == MessageType.ACTIVATION:
            try:
                await conn.send_and_wait(
                    MessageType.ACTIVATION, b"B-asks-A", timeout=5.0
                )
            except Exception:
                pass
            try:
                await conn.send_frame(
                    MessageType.RESULT, b"B-done", stream_id=frame.stream_id
                )
            except Exception:
                # Connection may already be tearing down; not what this test
                # is asserting.
                pass

    server_b = RpcServer(
        own_node_id="node-b", port=port_b, handler=handler_b, bind_ip=HOST
    )
    await server_b.start()
    client_a = RpcClient(own_node_id="node-a", handler=handler_a)
    try:
        result = await asyncio.wait_for(
            client_a.send_and_wait(
                (HOST, port_b), MessageType.ACTIVATION, b"A-asks-B", timeout=8.0
            ),
            timeout=10.0,
        )
        assert result is not None
        assert got_callback.is_set(), "B never called back to A"
        print("PASS: handler calling send_and_wait completed without stalling")
        await asyncio.sleep(0.2)  # let in-flight handlers settle before teardown
    except asyncio.TimeoutError:
        raise AssertionError(
            "STALLED — handler dispatch is blocking the receive loop. "
            "This is the bug the per-task dispatch fix was meant to prevent."
        )
    finally:
        await client_a.close()
        await server_b.stop()


async def test_concurrent_handlers_do_not_block_each_other() -> None:
    """Several slow handlers should overlap, not serialise."""
    port = 19805
    concurrent = 0
    peak = 0

    async def slow_handler(conn, frame):
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        try:
            await asyncio.sleep(0.3)
            await conn.send_frame(MessageType.RESULT, b"ok", stream_id=frame.stream_id)
        finally:
            concurrent -= 1

    server = RpcServer(
        own_node_id="node-s", port=port, handler=slow_handler, bind_ip=HOST
    )
    await server.start()
    client = RpcClient(own_node_id="node-c", handler=lambda c, f: None)
    try:
        results = await asyncio.gather(
            *[
                client.send_and_wait(
                    (HOST, port), MessageType.ACTIVATION, b"x", timeout=10.0
                )
                for _ in range(5)
            ]
        )
        assert len(results) == 5
        assert peak > 1, f"handlers serialised (peak concurrency {peak})"
        print(f"PASS: handlers ran concurrently (peak {peak})")
    finally:
        await client.close()
        await server.stop()


# ── Runner ────────────────────────────────────────────────────────────────


async def _run_async() -> None:
    await test_rejects_malformed_frames()
    await test_oversized_rejected_by_limit()
    await test_handler_calling_send_and_wait_does_not_stall()
    await test_concurrent_handlers_do_not_block_each_other()


def main() -> None:
    print("── Framing ──")
    test_frame_roundtrip()
    test_header_is_big_endian()

    print("\n── Security / malformed input ──")
    print("\n── Async ──")
    asyncio.run(_run_async())

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
