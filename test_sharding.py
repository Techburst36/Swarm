"""Tests for sharding.py — determinism, apportionment correctness, edge cases.

The property this module lives or dies on is **determinism**: there is no
coordinator, so every node computes its own assignment independently and they
must agree exactly. A test suite that only checked "the counts look about
right" would miss the failure mode that actually matters — two nodes quietly
disagreeing about who owns what.

Run:  python3 test_sharding.py
"""

from __future__ import annotations

import random

from sharding import (
    NodeCapability,
    ShardAssignment,
    compute_assignment,
    fleet_state_hash,
    _apportion_largest_remainder,
)


# ── Apportionment correctness ─────────────────────────────────────────────


def test_counts_sum_exactly() -> None:
    """The core promise: assigned counts sum to exactly num_experts.

    Naive round() would fail this — that's why largest remainder is used.
    """
    for num_experts in [1, 2, 7, 64, 128, 999]:
        for weights in [
            [1000, 1000, 1000],
            [1900, 4000, 4000, 4000],
            [1, 2, 3, 5, 8, 13],
            [7777],
            [100, 1],
        ]:
            nodes = [
                NodeCapability(node_id=f"node-{i:03d}", storage_bandwidth_mbps=w)
                for i, w in enumerate(weights)
            ]
            a = compute_assignment(nodes, num_experts)
            total = sum(a.node_counts.values())
            assert total == num_experts, (
                f"counts summed to {total}, expected {num_experts} "
                f"(weights={weights})"
            )
    print("PASS: counts always sum exactly to num_experts")


def test_every_expert_assigned_exactly_once() -> None:
    """No expert index may be dropped or double-assigned."""
    nodes = [
        NodeCapability(node_id="a", storage_bandwidth_mbps=1900),
        NodeCapability(node_id="b", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="c", storage_bandwidth_mbps=4000),
    ]
    a = compute_assignment(nodes, 64)

    seen: list[int] = []
    for experts in a.node_experts.values():
        seen.extend(experts)

    assert sorted(seen) == list(range(64)), "expert coverage is not exactly 0..63"
    assert len(seen) == len(set(seen)), "an expert index was assigned twice"
    print("PASS: every expert assigned exactly once, no gaps or overlaps")


def test_proportional_not_equal() -> None:
    """A node with ~2x the bandwidth should get ~2x the experts.

    This is the entire point of weighted sharding — verify it actually
    weights rather than splitting evenly.
    """
    nodes = [
        NodeCapability(node_id="slow", storage_bandwidth_mbps=1000),
        NodeCapability(node_id="fast", storage_bandwidth_mbps=2000),
    ]
    a = compute_assignment(nodes, 90)

    slow = a.node_counts["slow"]
    fast = a.node_counts["fast"]
    assert slow + fast == 90
    assert fast == 60 and slow == 30, f"expected 60/30 split, got {fast}/{slow}"
    print("PASS: allocation is proportional to bandwidth, not equal")


def test_apportion_matches_hand_calculation() -> None:
    """Verify largest remainder against a hand-worked example.

    Weights 1900/4000/4000/4000 (total 13900) over 64 experts:
      quotas   = 8.748, 18.417, 18.417, 18.417
      floors   = 8, 18, 18, 18  (sum 62, so 2 remain)
      the two largest remainders take one each
    Expected: the 1900 node gets 9, one 4000 node gets 19, other two get 18.
    """
    items = [("d-slow", 1900), ("a", 4000), ("b", 4000), ("c", 4000)]
    counts = _apportion_largest_remainder(items, 64)

    assert sum(counts.values()) == 64
    assert counts["d-slow"] == 9, f"slow node got {counts['d-slow']}, expected 9"
    # Among the three tied 4000-weight nodes, exactly one gets the extra.
    fast_counts = sorted([counts["a"], counts["b"], counts["c"]])
    assert fast_counts == [18, 18, 19], f"tied nodes got {fast_counts}"
    # Tie-break is lexicographic: 'a' wins over 'b' and 'c'.
    assert counts["a"] == 19, "lexicographic tie-break not applied"
    print("PASS: apportionment matches hand-worked example incl. tie-break")


# ── Determinism: the property that actually matters ───────────────────────


def test_deterministic_across_input_order() -> None:
    """Same fleet in a different order must produce an identical assignment.

    This is the real-world case: two nodes' FleetTables will almost never
    have peers in the same insertion order. If order leaked into the output,
    every node would compute a different assignment and the fleet would
    silently disagree about ownership.
    """
    nodes = [
        NodeCapability(node_id=f"node-{i:03d}", storage_bandwidth_mbps=bw)
        for i, bw in enumerate([1900, 4000, 4000, 4000, 2500, 3300])
    ]

    reference = compute_assignment(nodes, 64)

    for trial in range(50):
        shuffled = nodes[:]
        random.Random(trial).shuffle(shuffled)
        candidate = compute_assignment(shuffled, 64)

        assert candidate.fleet_hash == reference.fleet_hash, (
            f"hash differs on shuffle {trial}"
        )
        assert candidate.node_counts == reference.node_counts, (
            f"counts differ on shuffle {trial}"
        )
        assert candidate.node_experts == reference.node_experts, (
            f"expert assignment differs on shuffle {trial}"
        )
    print("PASS: assignment identical across 50 shuffled input orderings")


def test_repeated_calls_identical() -> None:
    """Pure function: calling twice with identical input gives identical output."""
    nodes = [
        NodeCapability(node_id="x", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="y", storage_bandwidth_mbps=1900),
    ]
    a = compute_assignment(nodes, 64)
    b = compute_assignment(nodes, 64)

    assert a.fleet_hash == b.fleet_hash
    assert a.node_counts == b.node_counts
    assert a.node_experts == b.node_experts
    print("PASS: repeated calls are byte-identical (no hidden state)")


def test_hash_detects_change() -> None:
    """Hash must change when the fleet changes, and match when it doesn't."""
    base = [
        NodeCapability(node_id="x", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="y", storage_bandwidth_mbps=1900),
    ]

    # Same fleet, different order → same hash.
    assert fleet_state_hash(base) == fleet_state_hash(list(reversed(base)))

    # Node added → different hash.
    added = base + [NodeCapability(node_id="z", storage_bandwidth_mbps=3000)]
    assert fleet_state_hash(base) != fleet_state_hash(added)

    # Node removed → different hash.
    assert fleet_state_hash(base) != fleet_state_hash(base[:1])

    # Bandwidth changed → different hash. This matters: a node that finishes
    # benchmarking itself goes from 0 to a real figure, and the fleet must
    # notice that its share should change.
    rebenched = [
        NodeCapability(node_id="x", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="y", storage_bandwidth_mbps=2500),
    ]
    assert fleet_state_hash(base) != fleet_state_hash(rebenched)
    print("PASS: hash detects add / remove / bandwidth change, ignores order")


# ── Edge cases ────────────────────────────────────────────────────────────


def test_zero_bandwidth_node() -> None:
    """A node that hasn't benchmarked yet (bandwidth 0) gets zero experts.

    This is a real, expected input — NodeDescriptor defaults to 0 until a
    storage benchmark runs — not an error condition.
    """
    nodes = [
        NodeCapability(node_id="measured", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="unmeasured", storage_bandwidth_mbps=0),
    ]
    a = compute_assignment(nodes, 64)

    assert a.node_counts["unmeasured"] == 0
    assert a.node_counts["measured"] == 64
    assert a.node_experts["unmeasured"] == []
    print("PASS: zero-bandwidth node gets zero experts, no crash")


def test_all_zero_fleet() -> None:
    """Whole fleet unbenchmarked: no division by zero, everyone gets zero."""
    nodes = [
        NodeCapability(node_id="a", storage_bandwidth_mbps=0),
        NodeCapability(node_id="b", storage_bandwidth_mbps=0),
    ]
    a = compute_assignment(nodes, 64)
    assert all(c == 0 for c in a.node_counts.values())
    print("PASS: all-zero fleet handled without division by zero")


def test_fewer_experts_than_nodes() -> None:
    """More nodes than experts: some nodes get zero, counts still sum right."""
    nodes = [
        NodeCapability(node_id=f"n{i}", storage_bandwidth_mbps=4000)
        for i in range(10)
    ]
    a = compute_assignment(nodes, 3)

    assert sum(a.node_counts.values()) == 3
    assert sum(1 for c in a.node_counts.values() if c == 0) == 7
    print("PASS: fewer experts than nodes handled gracefully")


def test_single_node() -> None:
    """One node takes everything."""
    nodes = [NodeCapability(node_id="only", storage_bandwidth_mbps=4000)]
    a = compute_assignment(nodes, 64)
    assert a.node_counts["only"] == 64
    assert a.node_experts["only"] == list(range(64))
    print("PASS: single-node fleet takes all experts")


def test_negative_bandwidth_rejected() -> None:
    """Negative bandwidth must raise, not silently corrupt the allocation.

    The correctness proof depends on non-negative weights. A negative value
    would break it quietly, so fail loudly instead.
    """
    nodes = [
        NodeCapability(node_id="ok", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="bad", storage_bandwidth_mbps=-100),
    ]
    try:
        compute_assignment(nodes, 64)
    except ValueError as e:
        assert "non-negative" in str(e)
        print("PASS: negative bandwidth rejected with a clear error")
        return
    raise AssertionError("negative bandwidth was silently accepted")


def test_empty_fleet_rejected() -> None:
    """An empty fleet is a caller bug, not something to paper over."""
    try:
        compute_assignment([], 64)
    except ValueError:
        print("PASS: empty fleet rejected")
        return
    raise AssertionError("empty fleet was silently accepted")


def test_zero_experts() -> None:
    """Zero experts is degenerate but valid — everyone gets nothing."""
    nodes = [NodeCapability(node_id="a", storage_bandwidth_mbps=4000)]
    a = compute_assignment(nodes, 0)
    assert sum(a.node_counts.values()) == 0
    print("PASS: zero experts handled")


# ── Realistic mixed-generation fleet ──────────────────────────────────────


def test_mixed_generation_fleet() -> None:
    """The compatibility contract's actual use case.

    An old OSD32MP2-class node (~1900 Mbps) alongside RK3588-class nodes
    (~4000 Mbps). The whole point of weighted sharding is that the old node
    contributes at its own pace rather than gating the fleet.
    """
    nodes = [
        NodeCapability(node_id="old-osd32mp2", storage_bandwidth_mbps=1900),
        NodeCapability(node_id="new-rk3588-a", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="new-rk3588-b", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="new-rk3588-c", storage_bandwidth_mbps=4000),
    ]
    a = compute_assignment(nodes, 64)

    old = a.node_counts["old-osd32mp2"]
    new_total = sum(
        a.node_counts[n] for n in a.node_counts if n.startswith("new-")
    )

    assert old + new_total == 64
    # Old node's share should track its ~13.7% bandwidth share, not 25%.
    old_share = old / 64
    assert 0.10 < old_share < 0.18, (
        f"old node got {old_share:.1%} of experts, expected ~13.7%"
    )
    print(f"PASS: mixed fleet — old node got {old} experts ({old_share:.1%}), "
          f"tracking its bandwidth share not an equal split")


# ── Runner ────────────────────────────────────────────────────────────────


def main() -> None:
    print("── Apportionment correctness ──")
    test_counts_sum_exactly()
    test_every_expert_assigned_exactly_once()
    test_proportional_not_equal()
    test_apportion_matches_hand_calculation()

    print("\n── Determinism ──")
    test_deterministic_across_input_order()
    test_repeated_calls_identical()
    test_hash_detects_change()

    print("\n── Edge cases ──")
    test_zero_bandwidth_node()
    test_all_zero_fleet()
    test_fewer_experts_than_nodes()
    test_single_node()
    test_negative_bandwidth_rejected()
    test_empty_fleet_rejected()
    test_zero_experts()

    print("\n── Realistic fleet ──")
    test_mixed_generation_fleet()

    print("\nAll sharding tests passed.")


if __name__ == "__main__":
    main()
