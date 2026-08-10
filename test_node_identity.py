"""Unit tests for node_identity.py — no networking involved.

These test the parts of Layer 3 that are actually ours: descriptor
serialisation, FleetTable dedupe/self-filtering, and stale eviction.

The UDP discovery path is deliberately NOT tested here. It depends on the
host's multicast/broadcast behaviour, which varies wildly across
environments (notably WSL2, where local multicast does not loop back to
local listeners at all). Testing it here would test the OS, not this module.
Discovery gets validated on real hardware, or on a Linux host with a normal
network stack, per test-plan.md.

Run:  python3 -m pytest test_node_identity.py -v
  or: python3 test_node_identity.py        (no pytest needed)
"""

from __future__ import annotations

import asyncio
import json
import time

from node_identity import FleetTable, NodeDescriptor


def _make_descriptor(
    node_id: str,
    hostname: str = "testnode",
    *,
    timestamp: float | None = None,
    load: float = 0.25,
    hardware_gen: str = "test-gen",
    storage_bandwidth_mbps: int = 3200,
) -> NodeDescriptor:
    """Build a descriptor with sensible defaults for testing.

    NodeDescriptor is a frozen dataclass (immutable snapshot), so every
    field that a test needs to vary must be passed at construction time.
    """
    return NodeDescriptor(
        node_id=node_id,
        hostname=hostname,
        ip="192.168.0.99",
        port=9999,
        ram_total_mb=8192,
        ram_available_mb=4096,
        storage_bandwidth_mbps=storage_bandwidth_mbps,
        hardware_gen=hardware_gen,
        load=load,
        uptime_seconds=42.0,
        timestamp=timestamp if timestamp is not None else time.time(),
    )


# ── Descriptor serialisation ──────────────────────────────────────────────


def test_descriptor_roundtrip() -> None:
    """to_json/from_json should preserve every field exactly."""
    original = _make_descriptor("node-aaa", "alpha")
    restored = NodeDescriptor.from_json(original.to_json())

    assert restored is not None, "from_json returned None on valid input"
    assert restored.node_id == original.node_id
    assert restored.hostname == original.hostname
    assert restored.ip == original.ip
    assert restored.port == original.port
    assert restored.ram_total_mb == original.ram_total_mb
    assert restored.hardware_gen == original.hardware_gen
    assert restored.load == original.load
    print("PASS: descriptor roundtrip")


def test_descriptor_rejects_malformed() -> None:
    """Malformed input should return None, not raise."""
    for bad in [
        b"not json at all",
        b"{}",
        b'{"node_id": "x"}',            # missing required fields
        b'{"v": 999, "node_id": "x"}',  # unknown schema version
        b"",
    ]:
        result = NodeDescriptor.from_json(bad)
        assert result is None, f"expected None for {bad!r}, got {result!r}"
    print("PASS: malformed input rejected cleanly")


# ── FleetTable behaviour ──────────────────────────────────────────────────


async def test_fleet_ignores_self() -> None:
    """A node must not add itself to its own fleet table."""
    own_id = "node-self"
    fleet = FleetTable(own_id, stale_timeout=8.0)

    added = await fleet.update(_make_descriptor(own_id, "myself"))
    assert added is False, "fleet accepted its own descriptor"
    assert len(await fleet.get_live_nodes()) == 0
    print("PASS: self-descriptor filtered")


async def test_fleet_adds_peers() -> None:
    """Distinct peers should each appear exactly once."""
    fleet = FleetTable("node-self", stale_timeout=8.0)

    await fleet.update(_make_descriptor("node-aaa", "alpha"))
    await fleet.update(_make_descriptor("node-bbb", "bravo"))

    live = await fleet.get_live_nodes()
    assert len(live) == 2, f"expected 2 peers, got {len(live)}"
    names = sorted(d.hostname for d in live)
    assert names == ["alpha", "bravo"], names
    print("PASS: distinct peers added")


async def test_fleet_dedupes_by_node_id() -> None:
    """Repeat broadcasts from one peer update it, never duplicate it."""
    fleet = FleetTable("node-self", stale_timeout=8.0)

    await fleet.update(_make_descriptor("node-aaa", "alpha"))
    for _ in range(5):
        await fleet.update(_make_descriptor("node-aaa", "alpha"))

    live = await fleet.get_live_nodes()
    assert len(live) == 1, f"dedupe failed: {len(live)} entries for one node_id"
    print("PASS: repeat broadcasts deduped")


async def test_fleet_updates_existing_peer() -> None:
    """A newer descriptor should replace the stored one, not be ignored."""
    fleet = FleetTable("node-self", stale_timeout=8.0)

    await fleet.update(_make_descriptor("node-aaa", "alpha", load=0.1))
    await fleet.update(_make_descriptor("node-aaa", "alpha", load=0.9))

    live = await fleet.get_live_nodes()
    assert len(live) == 1
    assert live[0].load == 0.9, f"stale load retained: {live[0].load}"
    print("PASS: existing peer updated in place")


async def test_fleet_evicts_stale() -> None:
    """A peer not heard from within stale_timeout should be evicted."""
    fleet = FleetTable("node-self", stale_timeout=1.0)

    await fleet.update(_make_descriptor("node-aaa", "alpha"))
    assert len(await fleet.get_live_nodes()) == 1

    # Simulate 2 seconds passing without a broadcast.
    await asyncio.sleep(1.2)
    await fleet.evict_stale()

    live = await fleet.get_live_nodes()
    assert len(live) == 0, f"stale peer not evicted: {live}"
    print("PASS: stale peer evicted")


async def test_fleet_keeps_fresh_peer() -> None:
    """Eviction must not remove a peer that is still broadcasting."""
    fleet = FleetTable("node-self", stale_timeout=1.0)

    await fleet.update(_make_descriptor("node-aaa", "alpha"))
    await asyncio.sleep(0.5)
    await fleet.update(_make_descriptor("node-aaa", "alpha"))  # refresh
    await fleet.evict_stale()

    live = await fleet.get_live_nodes()
    assert len(live) == 1, "fresh peer was wrongly evicted"
    print("PASS: fresh peer retained through eviction sweep")


async def test_mixed_generation_fleet() -> None:
    """Peers with different hardware_gen coexist — the compatibility contract
    requires no fixed roles and no homogeneity assumption."""
    fleet = FleetTable("node-self", stale_timeout=8.0)

    await fleet.update(
        _make_descriptor(
            "node-old", "legacy",
            hardware_gen="osd32mp2", storage_bandwidth_mbps=1900,
        )
    )
    await fleet.update(
        _make_descriptor(
            "node-new", "current",
            hardware_gen="rk3588-8gb", storage_bandwidth_mbps=4000,
        )
    )

    live = await fleet.get_live_nodes()
    assert len(live) == 2
    gens = sorted(d.hardware_gen for d in live)
    assert gens == ["osd32mp2", "rk3588-8gb"], gens
    print("PASS: mixed-generation fleet accepted")


# ── Runner ────────────────────────────────────────────────────────────────


async def _run_async_tests() -> None:
    await test_fleet_ignores_self()
    await test_fleet_adds_peers()
    await test_fleet_dedupes_by_node_id()
    await test_fleet_updates_existing_peer()
    await test_fleet_evicts_stale()
    await test_fleet_keeps_fresh_peer()
    await test_mixed_generation_fleet()


def main() -> None:
    print("── Synchronous tests ──")
    test_descriptor_roundtrip()
    test_descriptor_rejects_malformed()

    print("\n── Async tests ──")
    asyncio.run(_run_async_tests())

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
