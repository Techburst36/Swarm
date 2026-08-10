"""Tests for gang_sync.py — framing, ring determinism, and failure modes.

The properties that matter here are different from the other modules:

* **Ring order must be identical on every node**, derived independently.
  If two nodes disagree, messages go to the wrong peer and the reduce
  either hangs or silently produces a wrong sum.
* **Every failure mode must fail loudly**, never hang and never return a
  plausible-but-wrong result. A distributed sum that is quietly incorrect
  is worse than one that crashes.

Uses real RpcServer/RpcClient on 127.0.0.1 — no transport mocking, same
approach as rpc.py's own tests. Loopback only, never 0.0.0.0.

Run:  python3 test_gang_sync.py
"""

from __future__ import annotations

import asyncio

from gang_sync import (
    FAN_OUT_RESULT,
    MAX_RING_PAYLOAD_BYTES,
    RING_HOP,
    GangSync,
    GangSyncError,
    GangSyncTimeout,
    _decode_ring_message,
    _encode_ring_message,
)
from rpc import RpcClient, RpcServer
from sharding import NodeCapability, compute_assignment

HOST = "127.0.0.1"
BASE_PORT = 22400


def _int_combine(acc: bytes, local: bytes) -> bytes:
    """Integer addition, so the correct sum is trivially checkable."""
    a = int.from_bytes(acc, "big", signed=True)
    b = int.from_bytes(local, "big", signed=True)
    return (a + b).to_bytes(8, "big", signed=True)


def _enc(v: int) -> bytes:
    return v.to_bytes(8, "big", signed=True)


def _dec(b: bytes) -> int:
    return int.from_bytes(b, "big", signed=True)


# ── Framing ───────────────────────────────────────────────────────────────


def test_framing_roundtrip() -> None:
    """Encode then decode must return exactly what went in."""
    fleet_hash = "ab" * 32  # 64 hex chars
    encoded = _encode_ring_message(
        layer_id=7,
        ring_msg_type=RING_HOP,
        fleet_hash=fleet_hash,
        payload=b"hello ring",
    )
    decoded = _decode_ring_message(encoded)
    assert decoded is not None
    layer_id, msg_type, fh, payload = decoded

    assert layer_id == 7
    assert msg_type == RING_HOP
    assert fh == fleet_hash
    assert payload == b"hello ring"
    print("PASS: ring message framing roundtrips exactly")


def test_framing_rejects_short_data() -> None:
    """Data too short to hold a header returns None, does not crash."""
    for bad in [b"", b"x", b"\x00" * 10, b"\x00" * 36]:
        assert _decode_ring_message(bad) is None, f"accepted {len(bad)}-byte input"
    print("PASS: undersized frames rejected without crashing")


def test_framing_rejects_bad_fleet_hash() -> None:
    """A fleet_hash that isn't 64 hex chars must raise, not silently truncate."""
    try:
        _encode_ring_message(
            layer_id=1, ring_msg_type=RING_HOP, fleet_hash="abcd", payload=b""
        )
    except (ValueError, GangSyncError):
        print("PASS: malformed fleet_hash rejected at encode")
        return
    raise AssertionError("short fleet_hash was silently accepted")


def test_payload_size_cap() -> None:
    """An oversized payload is rejected before it can be amplified round the ring.

    A runaway combine() result would otherwise be forwarded to every node.
    """
    oversized = b"\x00" * (MAX_RING_PAYLOAD_BYTES + 1)
    try:
        _encode_ring_message(
            layer_id=1,
            ring_msg_type=RING_HOP,
            fleet_hash="ab" * 32,
            payload=oversized,
        )
    except GangSyncError as e:
        assert "cap" in str(e).lower() or "bytes" in str(e).lower()
        print("PASS: oversized ring payload rejected at encode")
        return
    raise AssertionError("oversized payload was accepted")


def test_layer_id_distinguishes_messages() -> None:
    """Two layers' messages must be distinguishable — this is the anti-crosstalk
    mechanism, so verify the field actually survives the wire."""
    fh = "cd" * 32
    a = _decode_ring_message(
        _encode_ring_message(layer_id=1, ring_msg_type=RING_HOP, fleet_hash=fh, payload=b"x")
    )
    b = _decode_ring_message(
        _encode_ring_message(layer_id=2, ring_msg_type=RING_HOP, fleet_hash=fh, payload=b"x")
    )
    assert a is not None and b is not None
    assert a[0] == 1 and b[0] == 2, "layer_id did not survive the round trip"

    # Message type likewise.
    fan = _decode_ring_message(
        _encode_ring_message(
            layer_id=1, ring_msg_type=FAN_OUT_RESULT, fleet_hash=fh, payload=b"x"
        )
    )
    assert fan is not None and fan[1] == FAN_OUT_RESULT
    print("PASS: layer_id and msg_type survive the wire (anti-crosstalk basis)")


# ── Ring order determinism ────────────────────────────────────────────────


def test_ring_order_is_deterministic() -> None:
    """Every node derives the identical ring from the same assignment.

    This is the coordinator-free property. Ring order comes from
    sorted(assignment.node_counts.keys()) — verify that input ordering
    cannot change it, since no two nodes' FleetTables will agree on order.
    """
    ids = [f"node-{i:02d}" for i in range(6)]
    caps_a = [NodeCapability(node_id=n, storage_bandwidth_mbps=4000) for n in ids]
    caps_b = [NodeCapability(node_id=n, storage_bandwidth_mbps=4000) for n in reversed(ids)]

    a = compute_assignment(caps_a, 64)
    b = compute_assignment(caps_b, 64)

    ring_a = sorted(a.node_counts.keys())
    ring_b = sorted(b.node_counts.keys())

    assert ring_a == ring_b, "ring order differs with input ordering"
    assert a.fleet_hash == b.fleet_hash, "fleet hash differs with input ordering"
    print("PASS: ring order identical regardless of input ordering")


# ── Live multi-node behaviour ─────────────────────────────────────────────


async def _build_fleet(n: int, base_port: int):
    """Spin up n real nodes on loopback. Returns (gangs, servers, clients, assignment)."""
    node_ids = [f"node-{i:02d}-{'0'*8}" for i in range(n)]
    peers = {nid: (HOST, base_port + i) for i, nid in enumerate(node_ids)}
    caps = [
        NodeCapability(node_id=nid, storage_bandwidth_mbps=4000) for nid in node_ids
    ]
    assignment = compute_assignment(caps, 64)

    gangs, servers, clients = [], [], []
    for i, nid in enumerate(node_ids):
        client = RpcClient(own_node_id=nid)
        gang = GangSync(own_node_id=nid, rpc_client=client, peers=peers)
        server = RpcServer(
            own_node_id=nid,
            port=base_port + i,
            handler=gang.handle_frame,
            bind_ip=HOST,  # loopback only, never 0.0.0.0
        )
        clients.append(client)
        gangs.append(gang)
        servers.append(server)

    for s in servers:
        await s.start()
    await asyncio.sleep(0.1)
    return gangs, servers, clients, assignment, node_ids


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


async def test_all_nodes_agree() -> None:
    """The core contract: every node ends with the identical correct sum."""
    gangs, servers, clients, assignment, _ = await _build_fleet(4, BASE_PORT)
    try:
        contribs = [10, 20, 30, 40]
        results = await asyncio.gather(
            *(
                gangs[i].ring_reduce(
                    assignment=assignment,
                    layer_id=1,
                    local_contribution=_enc(contribs[i]),
                    combine=_int_combine,
                    timeout=5.0,
                )
                for i in range(4)
            )
        )
        sums = [_dec(r) for r in results]
        assert len(set(sums)) == 1, f"nodes disagree: {sums}"
        assert sums[0] == sum(contribs), f"got {sums[0]}, expected {sum(contribs)}"
        print(f"PASS: all 4 nodes agree on correct sum ({sums[0]})")
    finally:
        await _teardown(servers, clients)


async def test_sequential_layers_isolated() -> None:
    """Consecutive layers must not contaminate each other.

    Runs three layers back to back; each must produce its own correct sum,
    proving state is cleaned up and layer_id gating works end to end.
    """
    gangs, servers, clients, assignment, _ = await _build_fleet(4, BASE_PORT + 10)
    try:
        for layer_id, contribs in [
            (1, [1, 2, 3, 4]),
            (2, [100, 200, 300, 400]),
            (3, [-5, 5, -5, 5]),
        ]:
            results = await asyncio.gather(
                *(
                    gangs[i].ring_reduce(
                        assignment=assignment,
                        layer_id=layer_id,
                        local_contribution=_enc(contribs[i]),
                        combine=_int_combine,
                        timeout=5.0,
                    )
                    for i in range(4)
                )
            )
            sums = [_dec(r) for r in results]
            assert len(set(sums)) == 1, f"layer {layer_id}: nodes disagree {sums}"
            assert sums[0] == sum(contribs), (
                f"layer {layer_id}: got {sums[0]}, expected {sum(contribs)}"
            )
        print("PASS: three consecutive layers each correct, no cross-contamination")
    finally:
        await _teardown(servers, clients)


async def test_duplicate_layer_id_rejected() -> None:
    """A second concurrent call for the same layer_id must fail loudly.

    Without this guard the second call silently overwrites the first's
    state, orphaning the first caller's future so it hangs until timeout
    with no indication why.
    """
    gangs, servers, clients, assignment, _ = await _build_fleet(2, BASE_PORT + 20)
    try:
        first = asyncio.create_task(
            gangs[0].ring_reduce(
                assignment=assignment,
                layer_id=99,
                local_contribution=_enc(1),
                combine=_int_combine,
                timeout=2.0,
            )
        )
        await asyncio.sleep(0.05)  # let it register state

        try:
            await gangs[0].ring_reduce(
                assignment=assignment,
                layer_id=99,
                local_contribution=_enc(2),
                combine=_int_combine,
                timeout=2.0,
            )
        except GangSyncError as e:
            assert "already in progress" in str(e)
            print("PASS: duplicate concurrent layer_id rejected with clear error")
        else:
            raise AssertionError("duplicate layer_id was silently accepted")
        finally:
            first.cancel()
            try:
                await first
            except (asyncio.CancelledError, GangSyncError, GangSyncTimeout):
                pass
    finally:
        await _teardown(servers, clients)


async def test_node_not_in_assignment_rejected() -> None:
    """A node absent from the assignment must fail immediately, not hang."""
    gangs, servers, clients, _, node_ids = await _build_fleet(2, BASE_PORT + 30)
    try:
        # Assignment that excludes node 0 entirely.
        other_caps = [
            NodeCapability(node_id="somebody-else", storage_bandwidth_mbps=4000)
        ]
        foreign = compute_assignment(other_caps, 64)

        try:
            await gangs[0].ring_reduce(
                assignment=foreign,
                layer_id=1,
                local_contribution=_enc(1),
                combine=_int_combine,
                timeout=2.0,
            )
        except ValueError as e:
            assert "not in the" in str(e)
            print("PASS: node missing from assignment rejected immediately")
            return
        raise AssertionError("node absent from assignment was accepted")
    finally:
        await _teardown(servers, clients)


async def test_dead_node_times_out_not_hangs() -> None:
    """A dead ring member must produce a timeout, never an indefinite hang.

    The bound matters as much as the exception: a hang in gang mode stalls
    the whole fleet, so verify it actually returns near the timeout value.
    """
    gangs, servers, clients, assignment, _ = await _build_fleet(3, BASE_PORT + 40)
    try:
        await servers[2].stop()
        await clients[2].close()

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        results = await asyncio.gather(
            gangs[0].ring_reduce(
                assignment=assignment,
                layer_id=1,
                local_contribution=_enc(1),
                combine=_int_combine,
                timeout=1.5,
            ),
            gangs[1].ring_reduce(
                assignment=assignment,
                layer_id=1,
                local_contribution=_enc(2),
                combine=_int_combine,
                timeout=1.5,
            ),
            return_exceptions=True,
        )
        elapsed = loop.time() - t0

        assert all(isinstance(r, GangSyncError) for r in results), (
            f"expected all failures, got {[type(r).__name__ for r in results]}"
        )
        assert not any(isinstance(r, bytes) for r in results), (
            "a node returned a result from a broken ring"
        )
        assert elapsed < 8.0, f"took {elapsed:.1f}s — did not fail promptly"
        print(f"PASS: dead node → clean failure in {elapsed:.1f}s, no hang")
    finally:
        await _teardown(servers, clients)


async def test_fleet_hash_mismatch_fails_loudly() -> None:
    """A node with a stale assignment must error, never compute a wrong sum.

    This is the worst-case failure mode for a distributed reduce: a
    plausible-looking but incorrect number. Verify nobody gets one.
    """
    gangs, servers, clients, assignment, node_ids = await _build_fleet(
        3, BASE_PORT + 50
    )
    try:
        # Same nodes, different bandwidths → different fleet_hash.
        stale_caps = [
            NodeCapability(node_id=nid, storage_bandwidth_mbps=1000 + i * 500)
            for i, nid in enumerate(node_ids)
        ]
        stale = compute_assignment(stale_caps, 64)
        assert stale.fleet_hash != assignment.fleet_hash

        results = await asyncio.gather(
            gangs[0].ring_reduce(
                assignment=assignment, layer_id=1,
                local_contribution=_enc(1), combine=_int_combine, timeout=2.0,
            ),
            gangs[1].ring_reduce(
                assignment=assignment, layer_id=1,
                local_contribution=_enc(2), combine=_int_combine, timeout=2.0,
            ),
            gangs[2].ring_reduce(  # stale assignment
                assignment=stale, layer_id=1,
                local_contribution=_enc(3), combine=_int_combine, timeout=2.0,
            ),
            return_exceptions=True,
        )

        assert not any(isinstance(r, bytes) for r in results), (
            "a node produced a result despite a fleet-hash mismatch"
        )
        assert any(
            isinstance(r, GangSyncError) and "mismatch" in str(r) for r in results
        ), "no node reported the hash mismatch specifically"
        print("PASS: fleet-hash mismatch fails loudly, nobody gets a wrong sum")
    finally:
        await _teardown(servers, clients)


async def test_combine_exception_surfaces() -> None:
    """A raising combine() callback must surface as an error, not a hang."""
    gangs, servers, clients, assignment, _ = await _build_fleet(2, BASE_PORT + 60)
    try:
        def bad_combine(acc: bytes, local: bytes) -> bytes:
            raise RuntimeError("combine blew up")

        results = await asyncio.gather(
            gangs[0].ring_reduce(
                assignment=assignment, layer_id=1,
                local_contribution=_enc(1), combine=bad_combine, timeout=2.0,
            ),
            gangs[1].ring_reduce(
                assignment=assignment, layer_id=1,
                local_contribution=_enc(2), combine=bad_combine, timeout=2.0,
            ),
            return_exceptions=True,
        )
        assert not any(isinstance(r, bytes) for r in results), (
            "got a result despite combine() raising"
        )
        print("PASS: combine() exception surfaces rather than hanging")
    finally:
        await _teardown(servers, clients)


async def test_single_node_ring() -> None:
    """A one-node fleet returns its own contribution without any network I/O."""
    gangs, servers, clients, _, _ = await _build_fleet(1, BASE_PORT + 70)
    try:
        caps = [NodeCapability(node_id="node-00-00000000", storage_bandwidth_mbps=4000)]
        solo = compute_assignment(caps, 64)
        result = await gangs[0].ring_reduce(
            assignment=solo, layer_id=1,
            local_contribution=_enc(42), combine=_int_combine, timeout=2.0,
        )
        assert _dec(result) == 42
        print("PASS: single-node ring returns own contribution")
    finally:
        await _teardown(servers, clients)


# ── Runner ────────────────────────────────────────────────────────────────


async def _run_async() -> None:
    await test_all_nodes_agree()
    await test_sequential_layers_isolated()
    await test_duplicate_layer_id_rejected()
    await test_node_not_in_assignment_rejected()
    await test_dead_node_times_out_not_hangs()
    await test_fleet_hash_mismatch_fails_loudly()
    await test_combine_exception_surfaces()
    await test_single_node_ring()


def main() -> None:
    print("── Framing ──")
    test_framing_roundtrip()
    test_framing_rejects_short_data()
    test_framing_rejects_bad_fleet_hash()
    test_payload_size_cap()
    test_layer_id_distinguishes_messages()

    print("\n── Ring determinism ──")
    test_ring_order_is_deterministic()

    print("\n── Live multi-node (loopback only) ──")
    asyncio.run(_run_async())

    print("\nAll gang_sync tests passed.")


if __name__ == "__main__":
    main()
