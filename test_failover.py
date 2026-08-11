"""Tests for failover.py — diff computation, settle-window coalescing,
and the fleet-returns-to-prior-state edge case.

Uses _FakeFleetTable (from failover.py) for fully deterministic tests
without real UDP sockets.  The fake exercises the same callback path the
real FleetTable would.

Run:  python3 test_failover.py
"""

from __future__ import annotations

import asyncio

from failover import (
    FailoverCoordinator,
    ReshardDiff,
    _FakeFleetTable,
    _compute_diff,
    _make_desc,
)
from sharding import NodeCapability, ShardAssignment, compute_assignment

# ── Diff computation (unit-tested without live FleetTable) ────────────────


def test_diff_first_assignment() -> None:
    """First-ever assignment: old_owner=None for every expert, all nodes added."""
    caps = [
        NodeCapability(node_id="aaa", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="bbb", storage_bandwidth_mbps=4000),
    ]
    new = compute_assignment(caps, 16)

    diff = _compute_diff(None, new)

    assert diff.old_fleet_hash is None
    assert diff.new_fleet_hash == new.fleet_hash
    assert diff.moved_count == 16
    assert diff.unchanged_count == 0
    assert len(diff.nodes_added) == 2
    assert len(diff.nodes_removed) == 0
    # Every expert should have old_owner=None.
    for expert_idx, (old, new_owner) in diff.moved.items():
        assert old is None, f"expert {expert_idx}: expected old_owner=None"
        assert new_owner is not None
    print("PASS: first assignment shows old_owner=None for all experts")


def test_diff_node_added() -> None:
    """Adding a node redistributes some experts to it."""
    caps_old = [
        NodeCapability(node_id="aaa", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="bbb", storage_bandwidth_mbps=4000),
    ]
    caps_new = caps_old + [
        NodeCapability(node_id="ccc", storage_bandwidth_mbps=4000),
    ]

    old = compute_assignment(caps_old, 30)
    new = compute_assignment(caps_new, 30)

    diff = _compute_diff(old, new)

    assert "ccc" in diff.nodes_added
    assert len(diff.nodes_removed) == 0
    # With 3 equal nodes over 30 experts, each gets 10. The new node
    # should own some experts it didn't before.
    assert diff.moved_count > 0, "no experts moved to the new node"
    # Verify the new node actually gained experts.
    gains_for_ccc = sum(
        1 for _, (_, new_owner) in diff.moved.items() if new_owner == "ccc"
    )
    assert gains_for_ccc > 0, "new node ccc got no experts"
    print(f"PASS: node added → {diff.moved_count} moved, ccc gained {gains_for_ccc}")


def test_diff_node_removed() -> None:
    """Removing a node redistributes all its experts."""
    caps_old = [
        NodeCapability(node_id="aaa", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="bbb", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="ccc", storage_bandwidth_mbps=4000),
    ]
    caps_new = [
        NodeCapability(node_id="aaa", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="bbb", storage_bandwidth_mbps=4000),
    ]

    old = compute_assignment(caps_old, 30)
    new = compute_assignment(caps_new, 30)

    diff = _compute_diff(old, new)

    assert "ccc" in diff.nodes_removed
    assert len(diff.nodes_added) == 0
    assert diff.moved_count > 0
    # Every expert that ccc owned should show old_owner=ccc.
    ccc_evicted = sum(
        1 for _, (old_owner, _) in diff.moved.items() if old_owner == "ccc"
    )
    old_ccc_count = old.node_counts.get("ccc", 0)
    assert ccc_evicted == old_ccc_count, (
        f"ccc had {old_ccc_count} experts but only {ccc_evicted} show as moved from ccc"
    )
    print(f"PASS: node removed → all {ccc_evicted} of ccc's experts moved")


def test_diff_zero_bandwidth_node() -> None:
    """A joining node with bandwidth=0 gets zero experts, no moves."""
    caps_old = [
        NodeCapability(node_id="aaa", storage_bandwidth_mbps=4000),
    ]
    caps_new = [
        NodeCapability(node_id="aaa", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="unmeasured", storage_bandwidth_mbps=0),
    ]

    old = compute_assignment(caps_old, 16)
    new = compute_assignment(caps_new, 16)

    diff = _compute_diff(old, new)

    assert "unmeasured" in diff.nodes_added
    assert diff.moved_count == 0, (
        "zero-bandwidth node should not cause any expert moves"
    )
    assert diff.unchanged_count == 16
    print("PASS: zero-bandwidth node added → no expert moves")


def test_diff_no_change() -> None:
    """Same fleet state → no moved experts, no added/removed nodes."""
    caps = [
        NodeCapability(node_id="aaa", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="bbb", storage_bandwidth_mbps=1900),
    ]
    old = compute_assignment(caps, 64)
    new = compute_assignment(caps, 64)  # same input

    diff = _compute_diff(old, new)

    assert diff.old_fleet_hash == diff.new_fleet_hash
    assert diff.moved_count == 0
    assert diff.unchanged_count == 64
    assert len(diff.nodes_added) == 0
    assert len(diff.nodes_removed) == 0
    print("PASS: identical fleet → zero moves, zero changes")


def test_diff_bandwidth_change() -> None:
    """Same nodes, different bandwidth → a real reshard."""
    caps_old = [
        NodeCapability(node_id="aaa", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="bbb", storage_bandwidth_mbps=4000),
    ]
    caps_new = [
        NodeCapability(node_id="aaa", storage_bandwidth_mbps=8000),
        NodeCapability(node_id="bbb", storage_bandwidth_mbps=4000),
    ]

    old = compute_assignment(caps_old, 30)
    new = compute_assignment(caps_new, 30)

    diff = _compute_diff(old, new)

    assert diff.moved_count > 0, "bandwidth change should cause expert moves"
    assert len(diff.nodes_added) == 0
    assert len(diff.nodes_removed) == 0
    # aaa should have gained experts (2x bandwidth of bbb now).
    aaa_old = old.node_counts["aaa"]
    aaa_new = new.node_counts["aaa"]
    assert aaa_new >= aaa_old, (
        f"aaa should not lose experts when its bandwidth doubles "
        f"(was {aaa_old}, now {aaa_new})"
    )
    print(f"PASS: bandwidth change → {diff.moved_count} moves, "
          f"aaa {aaa_old}→{aaa_new}")


# ── Live coordinator behaviour (with FakeFleetTable) ────────────────────


async def test_settle_window_coalescing() -> None:
    """Two rapid membership changes → ONE diff, not two."""
    OWN = "self-node"
    fleet = _FakeFleetTable(own_node_id=OWN)
    coord = FailoverCoordinator(
        fleet_table=fleet,
        own_node_id=OWN,
        num_experts=64,
        own_storage_bandwidth_mbps=4000,
        settle_window=0.3,
    )
    await coord.start()

    # Seed with one peer so we have a baseline.
    await fleet.add_node(_make_desc("peer-a", "alpha"))
    await asyncio.sleep(0.5)
    assert coord.diff_queue.qsize() == 1
    await coord.diff_queue.get()  # drain first assignment
    assert coord.diff_queue.empty()

    # Two rapid adds within settle window.
    await fleet.add_node(_make_desc("peer-b", "bravo"))
    await asyncio.sleep(0.05)
    await fleet.add_node(_make_desc("peer-c", "charlie"))
    await asyncio.sleep(0.5)

    # Exactly one diff should appear, covering both additions.
    assert coord.diff_queue.qsize() == 1, (
        f"expected 1 diff after coalescing, got {coord.diff_queue.qsize()}"
    )
    diff = coord.diff_queue.get_nowait()
    assert len(diff.nodes_added) == 2, (
        f"coalesced diff should add 2 nodes, got {len(diff.nodes_added)}"
    )
    assert coord.diff_queue.empty(), "extra diff in queue — coalescing failed"
    print("PASS: settle window coalesced two changes into one diff")


async def test_fleet_returns_to_prior_state_no_diff() -> None:
    """A blip where a node leaves and rejoins within the settle window
    must produce NO diff — the fleet ended up unchanged."""
    OWN = "self-node"
    fleet = _FakeFleetTable(own_node_id=OWN)
    coord = FailoverCoordinator(
        fleet_table=fleet,
        own_node_id=OWN,
        num_experts=64,
        own_storage_bandwidth_mbps=4000,
        settle_window=0.3,
    )
    await coord.start()

    await fleet.add_node(_make_desc("peer-x", "xray"))
    await fleet.add_node(_make_desc("peer-y", "yankee"))
    await asyncio.sleep(0.5)
    # Drain first assignment.
    assert coord.diff_queue.qsize() == 1
    await coord.diff_queue.get()

    hash_before = coord.current_assignment().fleet_hash

    # Blip: remove peer-y, then immediately re-add it.
    await fleet.remove_node("peer-y")
    await asyncio.sleep(0.05)
    await fleet.add_node(_make_desc("peer-y", "yankee"))
    await asyncio.sleep(0.5)

    # Fleet state should be back to exactly what it was.
    hash_after = coord.current_assignment().fleet_hash
    assert hash_before == hash_after, "fleet_hash changed after blip"

    # No diff should have been emitted.
    assert coord.diff_queue.empty(), (
        "diff was emitted for a fleet that returned to its prior state"
    )
    print("PASS: fleet-returns-to-prior-state → no diff emitted")


async def test_gang_sync_failure_alive_node_no_reshard() -> None:
    """report_gang_sync_failure for a live node logs a warning, emits no diff."""
    OWN = "self-node"
    fleet = _FakeFleetTable(own_node_id=OWN)
    coord = FailoverCoordinator(
        fleet_table=fleet,
        own_node_id=OWN,
        num_experts=64,
        own_storage_bandwidth_mbps=4000,
        settle_window=0.3,
    )
    await coord.start()

    await fleet.add_node(_make_desc("peer-live", "living"))
    await asyncio.sleep(0.5)
    await coord.diff_queue.get()  # drain first assignment
    assert coord.diff_queue.empty()

    coord.report_gang_sync_failure(layer_id=3, node_id="peer-live")
    await asyncio.sleep(0.2)

    assert coord.diff_queue.empty(), (
        "reshard triggered for a live node's transient gang-sync failure"
    )
    print("PASS: gang-sync failure for live node → warning, no reshard")


async def test_gang_sync_failure_dead_node_reshard_in_progress() -> None:
    """report_gang_sync_failure for a dead node should not double-trigger."""
    OWN = "self-node"
    fleet = _FakeFleetTable(own_node_id=OWN)
    coord = FailoverCoordinator(
        fleet_table=fleet,
        own_node_id=OWN,
        num_experts=64,
        own_storage_bandwidth_mbps=4000,
        settle_window=0.3,
    )
    await coord.start()

    await fleet.add_node(_make_desc("peer-doomed", "doomed"))
    await asyncio.sleep(0.5)
    await coord.diff_queue.get()  # drain first assignment

    # Remove the node — this fires the leave callback and starts a reshard.
    await fleet.remove_node("peer-doomed")
    await asyncio.sleep(0.05)  # let leave callback fire

    # Now report gang-sync failure for the same node.
    coord.report_gang_sync_failure(layer_id=3, node_id="peer-doomed")
    await asyncio.sleep(0.5)

    # Exactly one diff should arrive (from the leave callback), not two.
    assert coord.diff_queue.qsize() == 1, (
        f"expected 1 diff (from leave callback), got {coord.diff_queue.qsize()}"
    )
    diff = coord.diff_queue.get_nowait()
    assert "peer-doomed" in diff.nodes_removed
    print("PASS: gang-sync failure for dead node → info log, one diff (from leave)")


async def test_first_assignment_triggers_immediately() -> None:
    """The first-ever membership change should produce an assignment,
    not wait forever for a non-existent prior state."""
    OWN = "self-node"
    fleet = _FakeFleetTable(own_node_id=OWN)
    coord = FailoverCoordinator(
        fleet_table=fleet,
        own_node_id=OWN,
        num_experts=64,
        own_storage_bandwidth_mbps=4000,
        settle_window=0.3,
    )
    await coord.start()

    # No peers yet, no assignment.
    assert coord.current_assignment() is None

    await fleet.add_node(_make_desc("peer-first", "first"))
    await asyncio.sleep(0.5)

    assert coord.current_assignment() is not None
    assert coord.diff_queue.qsize() == 1
    diff = coord.diff_queue.get_nowait()
    assert diff.old_fleet_hash is None
    print("PASS: first membership change triggers assignment immediately")


async def test_consecutive_resets_do_not_double_fire() -> None:
    """Many rapid membership changes should produce minimal diffs.

    Specifically: 5 nodes join in quick succession within the settle
    window → exactly ONE diff, covering all 5.
    """
    OWN = "self-node"
    fleet = _FakeFleetTable(own_node_id=OWN)
    coord = FailoverCoordinator(
        fleet_table=fleet,
        own_node_id=OWN,
        num_experts=64,
        own_storage_bandwidth_mbps=4000,
        settle_window=0.5,
    )
    await coord.start()

    for i in range(5):
        await fleet.add_node(_make_desc(f"peer-{i:02d}", f"node-{i}"))
        await asyncio.sleep(0.02)  # well within settle window

    await asyncio.sleep(0.7)

    assert coord.diff_queue.qsize() == 1, (
        f"expected 1 diff after 5 rapid joins, got {coord.diff_queue.qsize()}"
    )
    diff = coord.diff_queue.get_nowait()
    assert len(diff.nodes_added) == 6, (  # 5 peers + self (first assignment)
        f"first assignment diff should add 6 nodes (self + 5 peers), "
        f"got {len(diff.nodes_added)}: {diff.nodes_added}"
    )
    assert coord.diff_queue.empty()
    print("PASS: 5 rapid joins → 1 diff")


async def test_no_diff_on_empty_fleet_start() -> None:
    """If no peers ever join, no assignment should be computed.

    The coordinator should not fire a reshard until a membership change
    actually happens — an empty FleetTable on start() should not trigger
    anything by itself.
    """
    OWN = "self-node"
    fleet = _FakeFleetTable(own_node_id=OWN)
    coord = FailoverCoordinator(
        fleet_table=fleet,
        own_node_id=OWN,
        num_experts=64,
        own_storage_bandwidth_mbps=4000,
        settle_window=0.3,
    )
    await coord.start()
    await asyncio.sleep(0.5)

    assert coord.current_assignment() is None
    assert coord.diff_queue.empty()
    print("PASS: empty fleet on start → no assignment computed")


# ── Runner ───────────────────────────────────────────────────────────────


async def _run_async() -> None:
    await test_settle_window_coalescing()
    await test_fleet_returns_to_prior_state_no_diff()
    await test_gang_sync_failure_alive_node_no_reshard()
    await test_gang_sync_failure_dead_node_reshard_in_progress()
    await test_first_assignment_triggers_immediately()
    await test_consecutive_resets_do_not_double_fire()
    await test_no_diff_on_empty_fleet_start()


def main() -> None:
    print("── Diff computation (no live FleetTable) ──")
    test_diff_first_assignment()
    test_diff_node_added()
    test_diff_node_removed()
    test_diff_zero_bandwidth_node()
    test_diff_no_change()
    test_diff_bandwidth_change()

    print("\n── Live coordinator (FakeFleetTable) ──")
    asyncio.run(_run_async())

    print("\nAll failover tests passed.")


if __name__ == "__main__":
    main()
