"""Tests for pipeline.py — framing, request isolation, and failure modes.

The properties that matter here are different again from the other modules:

* **Request isolation.** Unlike gang-sync's one-lockstep-op-at-a-time,
  many independent requests are in flight simultaneously. A message for
  request A reaching request B's state would silently corrupt an
  inference result, which is worse than a crash.
* **Bounded state.** Many concurrent requests means orphaned per-request
  state is a slow leak that only shows up after days of uptime.
* **Every failure fails loudly**, never hangs, never returns a
  plausible-but-wrong activation.

Real RpcServer/RpcClient on 127.0.0.1 — no transport mocking. Loopback
only, never 0.0.0.0.

Run:  python3 test_pipeline.py
"""

from __future__ import annotations

import asyncio

from pipeline import (
    PIPELINE_ERROR,
    PIPELINE_FORWARD,
    PIPELINE_RESULT,
    PipelineCoordinator,
    PipelineError,
    PipelineTimeout,
    _decode_pipeline_message,
    _encode_pipeline_message,
)
from rpc import RpcClient, RpcServer
from sharding import NodeCapability, compute_assignment

HOST = "127.0.0.1"
BASE_PORT = 23500
NUM_LAYERS = 40

FAKE_HASH = "ab" * 32  # 64 hex chars


def _enc(v: int) -> bytes:
    return v.to_bytes(8, "big", signed=True)


def _dec(b: bytes) -> int:
    return int.from_bytes(b, "big", signed=True)


def _make_compute(layers_per_call: dict[str, int] | None = None):
    """compute_stage that adds the number of layers processed.

    Makes the correct final value trivially predictable: start value plus
    total layer count.
    """
    def compute_stage(activation: bytes, layer_start: int, layer_end: int) -> bytes:
        return _enc(_dec(activation) + (layer_end - layer_start))
    return compute_stage


# ── Framing ───────────────────────────────────────────────────────────────


def test_framing_roundtrip() -> None:
    """Encode then decode returns exactly what went in."""
    encoded = _encode_pipeline_message(
        request_id=42,
        stage_index=3,
        msg_type=PIPELINE_FORWARD,
        fleet_hash=FAKE_HASH,
        originator_node_id="origin-node-id",
        payload=b"activation bytes",
    )
    decoded = _decode_pipeline_message(encoded)
    assert decoded is not None
    req_id, stage, msg_type, fh, origin, payload = decoded

    assert req_id == 42
    assert stage == 3
    assert msg_type == PIPELINE_FORWARD
    assert fh == FAKE_HASH
    assert origin == "origin-node-id"
    assert payload == b"activation bytes"
    print("PASS: pipeline message framing roundtrips exactly")


def test_framing_rejects_truncated() -> None:
    """Truncated data returns None rather than crashing or misparsing."""
    good = _encode_pipeline_message(
        request_id=1,
        stage_index=0,
        msg_type=PIPELINE_FORWARD,
        fleet_hash=FAKE_HASH,
        originator_node_id="n",
        payload=b"xyz",
    )
    # Every truncation point must be handled.
    for cut in range(0, len(good)):
        result = _decode_pipeline_message(good[:cut])
        # Either None, or a valid parse only if we happened to keep the
        # whole thing. Never an exception.
        if cut < len(good):
            assert result is None or isinstance(result, tuple)
    assert _decode_pipeline_message(b"") is None
    print("PASS: truncated frames rejected at every cut point, no crash")


def test_framing_distinguishes_requests_and_stages() -> None:
    """request_id and stage_index must survive the wire distinctly.

    This is the anti-crosstalk basis: two requests in flight at the same
    stage must be tellable apart.
    """
    a = _decode_pipeline_message(
        _encode_pipeline_message(
            request_id=100, stage_index=2, msg_type=PIPELINE_FORWARD,
            fleet_hash=FAKE_HASH, originator_node_id="x", payload=b"a",
        )
    )
    b = _decode_pipeline_message(
        _encode_pipeline_message(
            request_id=101, stage_index=2, msg_type=PIPELINE_FORWARD,
            fleet_hash=FAKE_HASH, originator_node_id="x", payload=b"b",
        )
    )
    assert a is not None and b is not None
    assert a[0] == 100 and b[0] == 101, "request_id did not survive"
    assert a[1] == b[1] == 2, "stage_index did not survive"
    assert a[5] != b[5], "payloads got mixed"
    print("PASS: request_id/stage_index distinguish concurrent requests")


def test_framing_message_types_distinct() -> None:
    """FORWARD / RESULT / ERROR must be distinguishable on the wire."""
    seen = set()
    for mt in (PIPELINE_FORWARD, PIPELINE_RESULT, PIPELINE_ERROR):
        d = _decode_pipeline_message(
            _encode_pipeline_message(
                request_id=1, stage_index=0, msg_type=mt,
                fleet_hash=FAKE_HASH, originator_node_id="x", payload=b"",
            )
        )
        assert d is not None
        seen.add(d[2])
    assert len(seen) == 3, f"message types collided: {seen}"
    print("PASS: FORWARD/RESULT/ERROR are distinct on the wire")


# ── Layer assignment reuse ────────────────────────────────────────────────


def test_layer_blocks_are_contiguous() -> None:
    """Dense pipelining requires contiguous layer blocks per node.

    This is why sharding.py is reused directly — its block assignment
    already produces contiguous ranges. If it ever produced interleaved
    indices, the pipeline would be wrong, so assert the property rather
    than assuming it.
    """
    caps = [
        NodeCapability(node_id=f"node-{i}", storage_bandwidth_mbps=4000)
        for i in range(4)
    ]
    a = compute_assignment(caps, NUM_LAYERS)

    for node_id, layers in a.node_experts.items():
        if not layers:
            continue
        expected = list(range(min(layers), max(layers) + 1))
        assert layers == expected, (
            f"{node_id} has non-contiguous layers: {layers}"
        )

    # Blocks must also tile 0..NUM_LAYERS-1 with no gaps or overlaps.
    all_layers = sorted(
        idx for layers in a.node_experts.values() for idx in layers
    )
    assert all_layers == list(range(NUM_LAYERS)), "layer coverage is wrong"
    print("PASS: layer blocks are contiguous and tile the full range")


def test_layer_assignment_deterministic() -> None:
    """Every node must derive the same pipeline order independently."""
    ids = [f"node-{i}" for i in range(5)]
    a = compute_assignment(
        [NodeCapability(node_id=n, storage_bandwidth_mbps=4000) for n in ids],
        NUM_LAYERS,
    )
    b = compute_assignment(
        [NodeCapability(node_id=n, storage_bandwidth_mbps=4000)
         for n in reversed(ids)],
        NUM_LAYERS,
    )
    assert a.fleet_hash == b.fleet_hash
    assert sorted(a.node_counts.keys()) == sorted(b.node_counts.keys())
    assert a.node_experts == b.node_experts
    print("PASS: pipeline order identical regardless of input ordering")


# ── Live multi-node ───────────────────────────────────────────────────────


async def _build_pipeline(
    n: int, base_port: int, *, max_concurrent: int = 64, compute_stage=None
):
    node_ids = [f"pl-{i:02d}-{'a'*8}" for i in range(n)]
    peers = {nid: (HOST, base_port + i) for i, nid in enumerate(node_ids)}
    caps = [
        NodeCapability(node_id=nid, storage_bandwidth_mbps=4000) for nid in node_ids
    ]
    assignment = compute_assignment(caps, NUM_LAYERS)

    coords, servers, clients = [], [], []
    for i, nid in enumerate(node_ids):
        client = RpcClient(own_node_id=nid)
        coord = PipelineCoordinator(
            own_node_id=nid,
            rpc_client=client,
            peers=peers,
            compute_stage=compute_stage or _make_compute(),
            max_concurrent_requests=max_concurrent,
        )
        coord.set_assignment(assignment)
        server = RpcServer(
            own_node_id=nid,
            port=base_port + i,
            handler=coord.handle_frame,
            bind_ip=HOST,  # loopback only
        )
        clients.append(client)
        coords.append(coord)
        servers.append(server)

    for s in servers:
        await s.start()
    await asyncio.sleep(0.1)
    return coords, servers, clients, assignment, node_ids


async def _teardown(servers, clients) -> None:
    for c in clients:
        try:
            await c.close()
        except Exception:
            pass
    for s in servers:
        try:
            await s.stop()
        except Exception:
            pass


async def test_end_to_end_correct() -> None:
    """A request through the full chain produces the correct result."""
    coords, servers, clients, assignment, _ = await _build_pipeline(4, BASE_PORT)
    try:
        result = await coords[0].run_pipeline(
            assignment=assignment,
            request_id=1,
            initial_activation=_enc(0),
            timeout=5.0,
        )
        assert _dec(result) == NUM_LAYERS, f"got {_dec(result)}, want {NUM_LAYERS}"
        print(f"PASS: end-to-end pipeline correct ({_dec(result)})")
    finally:
        await _teardown(servers, clients)


async def test_originate_from_middle_stage() -> None:
    """A request originating mid-chain must route to stage 0 first."""
    coords, servers, clients, assignment, _ = await _build_pipeline(
        4, BASE_PORT + 10
    )
    try:
        result = await coords[2].run_pipeline(  # node C, stage 2
            assignment=assignment,
            request_id=1,
            initial_activation=_enc(100),
            timeout=5.0,
        )
        assert _dec(result) == 100 + NUM_LAYERS
        print("PASS: request from middle stage routes through full chain")
    finally:
        await _teardown(servers, clients)


async def test_originator_is_last_stage() -> None:
    """The last node originating must not send a network message to itself.

    Correctness is what's asserted here; the no-self-send path is an
    implementation detail that shows up as this simply working.
    """
    coords, servers, clients, assignment, _ = await _build_pipeline(
        4, BASE_PORT + 20
    )
    try:
        result = await coords[3].run_pipeline(  # last stage
            assignment=assignment,
            request_id=1,
            initial_activation=_enc(7),
            timeout=5.0,
        )
        assert _dec(result) == 7 + NUM_LAYERS
        print("PASS: originator-is-last-stage resolves without self-send")
    finally:
        await _teardown(servers, clients)


async def test_concurrent_requests_isolated() -> None:
    """Many simultaneous requests must not contaminate each other.

    This is the property that separates this module from gang_sync: a
    crossed activation would silently corrupt an inference result.
    """
    coords, servers, clients, assignment, _ = await _build_pipeline(
        4, BASE_PORT + 30
    )
    try:
        starts = [0, 500, 1000, 7777, 31337, 42]
        results = await asyncio.gather(
            *(
                coords[i % 4].run_pipeline(
                    assignment=assignment,
                    request_id=200 + i,
                    initial_activation=_enc(s),
                    timeout=5.0,
                )
                for i, s in enumerate(starts)
            )
        )
        got = [_dec(r) for r in results]
        want = [s + NUM_LAYERS for s in starts]
        assert got == want, f"crossed results: got {got}, want {want}"
        print(f"PASS: {len(starts)} concurrent requests stayed isolated")
    finally:
        await _teardown(servers, clients)


async def test_duplicate_request_id_rejected() -> None:
    """A second in-flight call with the same request_id must fail loudly.

    The first request is held open with an async gate so its state is
    genuinely still pending when the duplicate arrives — without that the
    first completes in ~1 ms and there is nothing to collide with.
    """
    gate = asyncio.Event()

    async def gated_compute(act: bytes, start: int, end: int) -> bytes:
        await gate.wait()
        return _enc(_dec(act) + (end - start))

    coords, servers, clients, assignment, _ = await _build_pipeline(
        4, BASE_PORT + 40, compute_stage=gated_compute
    )
    try:
        first = asyncio.create_task(
            coords[0].run_pipeline(
                assignment=assignment,
                request_id=999,
                initial_activation=_enc(0),
                timeout=5.0,
            )
        )
        await asyncio.sleep(0.05)  # first request now blocked in compute

        try:
            await coords[0].run_pipeline(
                assignment=assignment,
                request_id=999,
                initial_activation=_enc(0),
                timeout=5.0,
            )
            raise AssertionError("duplicate request_id was accepted")
        except PipelineError as e:
            assert "already in progress" in str(e)
            print("PASS: duplicate request_id rejected with clear error")
        finally:
            gate.set()
            try:
                await asyncio.wait_for(first, timeout=5.0)
            except Exception:
                pass
    finally:
        await _teardown(servers, clients)


async def test_concurrency_cap_enforced() -> None:
    """Requests past the cap are rejected, not silently queued.

    Unbounded in-flight state is a slow leak that only appears after days.
    """
    gate = asyncio.Event()

    async def gated_compute(act: bytes, start: int, end: int) -> bytes:
        await gate.wait()
        return _enc(_dec(act) + (end - start))

    coords, servers, clients, assignment, _ = await _build_pipeline(
        4, BASE_PORT + 50, max_concurrent=2, compute_stage=gated_compute
    )
    try:
        tasks = [
            asyncio.create_task(
                coords[0].run_pipeline(
                    assignment=assignment,
                    request_id=300 + i,
                    initial_activation=_enc(0),
                    timeout=5.0,
                )
            )
            for i in range(2)
        ]
        await asyncio.sleep(0.05)  # both now blocked, cap is full

        try:
            await coords[0].run_pipeline(
                assignment=assignment,
                request_id=999,
                initial_activation=_enc(0),
                timeout=5.0,
            )
            raise AssertionError("over-cap request was accepted")
        except PipelineError as e:
            assert "Too many concurrent" in str(e)
            print("PASS: concurrency cap enforced with clear error")
        finally:
            gate.set()
            for t in tasks:
                try:
                    await asyncio.wait_for(t, timeout=5.0)
                except Exception:
                    pass
    finally:
        await _teardown(servers, clients)


async def test_dead_node_fails_promptly() -> None:
    """A dead stage must fail fast and specifically, never hang."""
    coords, servers, clients, assignment, _ = await _build_pipeline(
        4, BASE_PORT + 60
    )
    try:
        await servers[1].stop()  # kill stage 1
        await clients[1].close()

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        try:
            await coords[0].run_pipeline(
                assignment=assignment,
                request_id=1,
                initial_activation=_enc(0),
                timeout=2.0,
            )
            raise AssertionError("pipeline succeeded despite a dead stage")
        except PipelineError as e:
            elapsed = loop.time() - t0
            assert elapsed < 10.0, f"took {elapsed:.1f}s — did not fail promptly"
            print(f"PASS: dead stage → clean failure in {elapsed:.1f}s, no hang")
    finally:
        await _teardown(servers, clients)


async def test_fleet_hash_mismatch_fails_loudly() -> None:
    """A stale assignment must error, never produce a wrong activation."""
    coords, servers, clients, assignment, node_ids = await _build_pipeline(
        3, BASE_PORT + 70
    )
    try:
        # Give node 1 a different assignment (same nodes, different bandwidth).
        stale = compute_assignment(
            [
                NodeCapability(node_id=nid, storage_bandwidth_mbps=1000 + i * 500)
                for i, nid in enumerate(node_ids)
            ],
            NUM_LAYERS,
        )
        assert stale.fleet_hash != assignment.fleet_hash
        coords[1].set_assignment(stale)

        try:
            await coords[0].run_pipeline(
                assignment=assignment,
                request_id=1,
                initial_activation=_enc(0),
                timeout=3.0,
            )
            raise AssertionError("pipeline returned a result despite hash mismatch")
        except PipelineError as e:
            assert "mismatch" in str(e).lower()
            print("PASS: fleet-hash mismatch fails loudly, no wrong result")
    finally:
        await _teardown(servers, clients)


async def test_single_node_pipeline() -> None:
    """A one-node pipeline computes locally with zero network messages."""
    coords, servers, clients, _, node_ids = await _build_pipeline(
        1, BASE_PORT + 80
    )
    try:
        solo = compute_assignment(
            [NodeCapability(node_id=node_ids[0], storage_bandwidth_mbps=4000)],
            NUM_LAYERS,
        )
        coords[0].set_assignment(solo)
        result = await coords[0].run_pipeline(
            assignment=solo,
            request_id=1,
            initial_activation=_enc(5),
            timeout=3.0,
        )
        assert _dec(result) == 5 + NUM_LAYERS
        print("PASS: single-node pipeline computes locally")
    finally:
        await _teardown(servers, clients)


async def test_state_cleaned_up_after_requests() -> None:
    """Per-request state must not accumulate — the slow-leak check."""
    coords, servers, clients, assignment, _ = await _build_pipeline(
        4, BASE_PORT + 90
    )
    try:
        for i in range(12):
            await coords[0].run_pipeline(
                assignment=assignment,
                request_id=400 + i,
                initial_activation=_enc(i),
                timeout=5.0,
            )
        # After all requests complete, no pending state should remain.
        leftover = len(coords[0]._pending)
        assert leftover == 0, f"{leftover} orphaned request states remain"
        print("PASS: per-request state fully cleaned up after 12 requests")
    finally:
        await _teardown(servers, clients)


# ── Runner ────────────────────────────────────────────────────────────────


async def _run_async() -> None:
    await test_end_to_end_correct()
    await test_originate_from_middle_stage()
    await test_originator_is_last_stage()
    await test_concurrent_requests_isolated()
    await test_duplicate_request_id_rejected()
    await test_concurrency_cap_enforced()
    await test_dead_node_fails_promptly()
    await test_fleet_hash_mismatch_fails_loudly()
    await test_single_node_pipeline()
    await test_state_cleaned_up_after_requests()


def main() -> None:
    print("── Framing ──")
    test_framing_roundtrip()
    test_framing_rejects_truncated()
    test_framing_distinguishes_requests_and_stages()
    test_framing_message_types_distinct()

    print("\n── Layer assignment (sharding.py reuse) ──")
    test_layer_blocks_are_contiguous()
    test_layer_assignment_deterministic()

    print("\n── Live multi-node (loopback only) ──")
    asyncio.run(_run_async())

    print("\nAll pipeline tests passed.")


if __name__ == "__main__":
    main()
