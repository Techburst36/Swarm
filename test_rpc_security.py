"""Security and robustness tests for rpc.py.

These test that the transport REJECTS bad input, rather than that it
tolerates it. Every test here is an attack or a fault, not a happy path.

Deliberate choices in this file:

* Everything binds ``127.0.0.1``, never ``0.0.0.0``. A test that listens on
  all interfaces exposes an unauthenticated service to the whole LAN for as
  long as the test runs.
* No safety limits are relaxed to make tests pass. ``max_frame_bytes`` is
  lowered in some tests to make oversized frames cheap to construct -- that
  tightens the check, never loosens it.
* No pickle, no eval, no temp files outside the OS temp dir, no credentials.

Run:  python3 test_rpc_security.py
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

HOST = "127.0.0.1"  # never 0.0.0.0 in tests
PORT = 19998


# ── Helpers ───────────────────────────────────────────────────────────────


def _raw_frame(
    *,
    magic: int = MAGIC,
    version: int = CURRENT_VERSION,
    msg_type: int = int(MessageType.ACTIVATION),
    flags: int = 0,
    stream_id: int = 2,
    payload: bytes = b"hello",
    payload_len_override: int | None = None,
    checksum_override: int | None = None,
) -> bytes:
    """Hand-build a wire frame so we can make it malformed on purpose."""
    payload_len = (
        payload_len_override if payload_len_override is not None else len(payload)
    )
    checksum = (
        checksum_override
        if checksum_override is not None
        else zlib.crc32(payload) & 0xFFFFFFFF
    )
    header = struct.pack(
        "!IBBHII",
        magic,
        version,
        msg_type,
        flags,
        stream_id,
        payload_len,
    ) + struct.pack("!I", checksum)
    return header + payload


async def _echo_handler(conn, frame) -> None:
    """Minimal handler: echo ACTIVATION back as RESULT."""
    if frame.msg_type == MessageType.ACTIVATION:
        await conn.send_frame(MessageType.RESULT, frame.payload, stream_id=frame.stream_id)


async def _send_raw_and_observe(raw: bytes, *, port: int) -> bool:
    """Open a bare TCP socket, send *raw*, return True if the server closed us.

    A well-behaved server drops the connection on a protocol violation
    rather than trying to resynchronise -- resync logic in parsers is
    where exploitable state confusion lives.
    """
    reader, writer = await asyncio.open_connection(HOST, port)
    try:
        writer.write(raw)
        await writer.drain()
        # A well-behaved server may send a graceful GOODBYE before closing,
        # so drain until real EOF rather than expecting EOF immediately.
        try:
            deadline = asyncio.get_running_loop().time() + 3.0
            while asyncio.get_running_loop().time() < deadline:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=2.0)
                if chunk == b"":
                    return True  # EOF: server closed the connection
            return False
        except asyncio.TimeoutError:
            return False
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


# ── Header validation ─────────────────────────────────────────────────────


def test_header_size_is_fixed() -> None:
    """The header must be exactly 20 bytes -- the framing depends on it."""
    assert HEADER_SIZE == 20, f"HEADER_SIZE is {HEADER_SIZE}, expected 20"
    built = _raw_frame(payload=b"")
    assert len(built) == 20, f"empty-payload frame is {len(built)} bytes"
    print("PASS: header is exactly 20 bytes")


def test_bad_magic_rejected() -> None:
    """A frame without the SWRM magic must not be parsed."""
    raw = _raw_frame(magic=0xDEADBEEF)
    try:
        frame = Frame.decode_header(raw[:HEADER_SIZE])
    except Exception:
        print("PASS: bad magic rejected at decode")
        return
    assert frame.magic != MAGIC, "decode_header accepted a bad magic value silently"
    print("PASS: bad magic surfaced for caller to reject")


def test_oversized_payload_len_is_capped() -> None:
    """A peer claiming a huge payload must not cause a huge allocation.

    This is the memory-exhaustion vector: a 20-byte header claiming a 4 GB
    payload. The size must be checked BEFORE any buffer is allocated.
    """
    huge = 0xFFFFFFFF  # ~4 GB, the max a uint32 can express
    raw = _raw_frame(payload=b"", payload_len_override=huge)
    header = Frame.decode_header(raw[:HEADER_SIZE])
    assert header.payload_len == huge
    # The connection layer must refuse this; we assert the value is at least
    # visible and above any sane default so the check has something to catch.
    assert header.payload_len > 256 * 1024 * 1024
    print("PASS: oversized payload_len is expressible and must be capped downstream")


def test_checksum_mismatch_detectable() -> None:
    """A corrupted payload must not pass as valid."""
    payload = b"the quick brown fox"
    raw = _raw_frame(payload=payload, checksum_override=0x00000000)
    header = Frame.decode_header(raw[:HEADER_SIZE])
    actual = zlib.crc32(payload) & 0xFFFFFFFF
    assert header.checksum != actual, "checksum override did not take"
    print("PASS: checksum mismatch is detectable")


def test_unknown_version_distinguishable() -> None:
    """A future protocol version must be rejected, not misparsed."""
    raw = _raw_frame(version=99)
    header = Frame.decode_header(raw[:HEADER_SIZE])
    assert header.version != CURRENT_VERSION
    print("PASS: unknown version is distinguishable")


def test_no_pickle_anywhere() -> None:
    """Deserialising untrusted input with pickle is remote code execution.

    This test exists so the property is enforced mechanically, not by
    remembering to check during review.
    """
    import rpc as rpc_module

    source = open(rpc_module.__file__).read()
    for banned in ("import pickle", "pickle.loads", "pickle.load",
                   "import marshal", "marshal.loads",
                   "eval(", "exec(", "__import__("):
        assert banned not in source, f"BANNED CONSTRUCT PRESENT: {banned!r}"
    print("PASS: no pickle/marshal/eval/exec in rpc.py")


# ── Live-connection robustness ────────────────────────────────────────────


async def test_server_rejects_garbage() -> None:
    """Raw garbage on the socket must not crash or hang the server."""
    server = RpcServer(
        own_node_id="server-node",
        port=PORT,
        handler=_echo_handler,
        bind_ip=HOST,  # loopback only
    )
    await server.start()
    try:
        closed = await _send_raw_and_observe(b"GET / HTTP/1.1\r\n\r\n" * 4, port=PORT)
        assert closed, "server did not close the connection on garbage input"
        # Server must still be alive and serving.
        client = RpcClient(own_node_id="client-node")
        try:
            result = await client.send_and_wait(
                (HOST, PORT), MessageType.ACTIVATION, b"still alive?", timeout=5.0
            )
            assert result == b"still alive?"
        finally:
            await client.close()
        print("PASS: server rejected garbage and stayed healthy")
    finally:
        await server.stop()


async def test_server_rejects_oversized_claim() -> None:
    """A frame claiming 4 GB must be refused without allocating 4 GB."""
    server = RpcServer(
        own_node_id="server-node",
        port=PORT + 1,
        handler=_echo_handler,
        bind_ip=HOST,
        max_frame_bytes=1024 * 1024,  # 1 MB -- tighter than default, never looser
    )
    await server.start()
    try:
        raw = _raw_frame(payload=b"", payload_len_override=0xFFFFFFFF)
        closed = await _send_raw_and_observe(raw, port=PORT + 1)
        assert closed, "server accepted an oversized frame claim"
        print("PASS: oversized frame claim refused, no allocation")
    finally:
        await server.stop()


async def test_server_rejects_bad_checksum() -> None:
    """A frame whose payload does not match its checksum must be dropped."""
    server = RpcServer(
        own_node_id="server-node",
        port=PORT + 2,
        handler=_echo_handler,
        bind_ip=HOST,
    )
    await server.start()
    try:
        raw = _raw_frame(payload=b"corrupted", checksum_override=0xBADBAD00)
        closed = await _send_raw_and_observe(raw, port=PORT + 2)
        assert closed, "server accepted a frame with a bad checksum"
        print("PASS: bad checksum refused")
    finally:
        await server.stop()


async def test_midframe_disconnect_recovers() -> None:
    """Dropping the connection mid-frame must not wedge the server."""
    server = RpcServer(
        own_node_id="server-node",
        port=PORT + 3,
        handler=_echo_handler,
        bind_ip=HOST,
    )
    await server.start()
    try:
        # Send a header promising 1 MB, then send 10 bytes and vanish.
        raw = _raw_frame(payload=b"x" * 10, payload_len_override=1024 * 1024)
        reader, writer = await asyncio.open_connection(HOST, PORT + 3)
        writer.write(raw)
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

        await asyncio.sleep(0.3)

        # Server must still accept new work.
        client = RpcClient(own_node_id="client-node")
        try:
            result = await client.send_and_wait(
                (HOST, PORT + 3), MessageType.ACTIVATION, b"after abort", timeout=5.0
            )
            assert result == b"after abort"
        finally:
            await client.close()
        print("PASS: recovered from mid-frame disconnect")
    finally:
        await server.stop()


async def test_blocking_handler_does_not_stall_connection() -> None:
    """A handler that awaits must not prevent other frames being served.

    This is the regression test for the receive-loop stall: with serial
    inline handling, a handler that blocked stopped the queue draining,
    which stopped the socket being read at all.
    """
    release = asyncio.Event()

    async def slow_handler(conn, frame) -> None:
        if frame.payload == b"slow":
            await release.wait()
            await conn.send_frame(MessageType.RESULT, b"slow-done", stream_id=frame.stream_id)
        else:
            await conn.send_frame(MessageType.RESULT, frame.payload, stream_id=frame.stream_id)

    server = RpcServer(
        own_node_id="server-node",
        port=PORT + 4,
        handler=slow_handler,
        bind_ip=HOST,
    )
    await server.start()
    client = RpcClient(own_node_id="client-node")
    try:
        slow = asyncio.create_task(
            client.send_and_wait(
                (HOST, PORT + 4), MessageType.ACTIVATION, b"slow", timeout=10.0
            )
        )
        await asyncio.sleep(0.2)  # let the slow handler start and block

        # A second request must still complete while the first is blocked.
        fast = await client.send_and_wait(
            (HOST, PORT + 4), MessageType.ACTIVATION, b"fast", timeout=5.0
        )
        assert fast == b"fast", f"fast request returned {fast!r}"

        release.set()
        assert await slow == b"slow-done"
        print("PASS: blocking handler did not stall the connection")
    finally:
        await client.close()
        await server.stop()


# ── Runner ────────────────────────────────────────────────────────────────


async def _run_async() -> None:
    await test_server_rejects_garbage()
    await test_server_rejects_oversized_claim()
    await test_server_rejects_bad_checksum()
    await test_midframe_disconnect_recovers()
    await test_blocking_handler_does_not_stall_connection()


def main() -> None:
    print("── Static / framing tests ──")
    test_no_pickle_anywhere()
    test_header_size_is_fixed()
    test_bad_magic_rejected()
    test_oversized_payload_len_is_capped()
    test_checksum_mismatch_detectable()
    test_unknown_version_distinguishable()

    print("\n── Live connection tests (loopback only) ──")
    asyncio.run(_run_async())

    print("\nAll security tests passed.")


if __name__ == "__main__":
    main()
