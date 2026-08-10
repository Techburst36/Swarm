#!/usr/bin/env python3
"""
sharding.py — Deterministic, coordinator-free expert-to-node sharding
for the Swarm distributed inference fleet.

Design constraints
------------------
**No coordinator.**  Every node independently computes the same sharding
assignment from the same fleet view.  No leader election, no voting, no
single node that "decides."  If two nodes see the same fleet state, they
produce byte-identical assignments without communicating.

**Deterministic given identical input.**  Same set of ``(node_id,
storage_bandwidth_mbps)`` pairs and same ``num_experts`` must always produce
the same assignment, on any node, any time.  This means explicit sorting by
``node_id`` whenever iteration order could matter, no randomness, and no
dependency on wall-clock time anywhere in the assignment logic.

**Pure function.**  ``compute_assignment()`` has no side effects, no caching,
no hidden mutable state.  Repeated calls with the same inputs always return
the same result.  This makes it safe to call speculatively ("would the
assignment change if this node joined?") without worrying about stale cache.
It also means the caller can debounce reshard events trivially: call once,
compare the hash, call again later — the function itself imposes no timing
constraint.

**Fleet-state hash.**  Every ``ShardAssignment`` carries a SHA-256 hash of
the (sorted) input fleet state.  Note this is a *correctness/agreement*
check, not a security boundary: it detects whether two nodes computed from
the same fleet snapshot.  It does not and cannot detect a node that reports
false capability figures — that is a trust problem belonging to a higher
layer, under the same trusted-LAN threat model the transport layer assumes.  Two nodes can compare hashes to detect
whether their assignments were computed from the same fleet snapshot without
recomputing the full assignment.  Matching hashes ⇒ guaranteed identical
assignments.  Mismatched hashes ⇒ the caller (a higher layer) knows to wait
for convergence rather than trusting a stale assignment.  This module does
*not* attempt to solve convergence — it only makes disagreement detectable.

**Largest remainder method (Hamilton's method).**  Expert counts are
apportioned proportionally to each node's measured storage bandwidth using
the largest remainder algorithm.  This guarantees the assigned counts always
sum exactly to ``num_experts``, with no systematic bias toward large or
small shares.  The method is explicitly named and isolated so a future
reader can look it up.

**Zero-bandwidth nodes are valid, not errors.**  A node that just joined
and hasn't benchmarked its storage yet will have ``storage_bandwidth_mbps=0``
(the default in ``NodeDescriptor``).  The allocation must not crash, divide
by zero, or produce nonsensical output.  Zero-bandwidth nodes get zero
experts.

**Uniform across layers.**  Expert *slot indices* (0 to E−1) are assigned
to nodes once for the whole model, not independently per layer.  Every
layer uses the same node←→expert mapping.  This is a deliberate
simplification that keeps gang-mode layer-sync tractable — per-layer
assignment is out of scope and would add complexity nothing here needs.

**Standard library only.**  Runs on the same minimal ARM Linux image as the
rest of the Swarm stack.  No numpy, no scipy, no third-party packages.
Python 3.11+.
"""

from __future__ import annotations

import dataclasses
import hashlib
from fractions import Fraction

# ═══════════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass(frozen=True, slots=True)
class NodeCapability:
    """Minimal node information needed for sharding.

    Intentionally a subset of ``NodeDescriptor`` so this module stays
    decoupled from the discovery layer.  The caller is responsible for
    extracting these fields.

    Attributes
    ----------
    node_id:
        Stable unique identifier (UUID string).  Used as the deterministic
        tie-breaker when two nodes have identical fractional remainders in
        the largest-remainder distribution, and as the sort key for
        contiguous expert-index assignment.
    storage_bandwidth_mbps:
        Measured storage bandwidth in megabits per second.  May be ``0``
        for a node that has not yet benchmarked itself — this is a real,
        expected input, not an error condition.
    """

    node_id: str
    storage_bandwidth_mbps: int


@dataclasses.dataclass(frozen=True)
class ShardAssignment:
    """Result of a single deterministic sharding computation.

    All fields are set at construction time and never change.  The
    assignment is a pure function of the input ``(nodes, num_experts)``
    and carries no hidden state.

    Attributes
    ----------
    node_experts:
        Mapping from ``node_id`` to the sorted list of expert slot indices
        (0-based) that node owns.  Every expert index from 0 to
        ``num_experts-1`` appears in exactly one node's list.  The list
        for each node is a contiguous range, assigned in ``node_id``-sorted
        order.
    node_counts:
        Mapping from ``node_id`` to the integer number of experts assigned.
        Equivalent to ``len(node_experts[node_id])``; stored separately for
        convenience.
    node_bandwidths:
        Mapping from ``node_id`` to its ``storage_bandwidth_mbps``, carried
        through from the input so the summary can show per-node bandwidth
        shares without the caller having to re-supply the original list.
    fleet_hash:
        SHA-256 hex digest of the deterministically-serialised input fleet
        state.  Two assignments with the same hash were computed from
        identical input and are guaranteed byte-identical.
    num_experts:
        Total number of expert slot indices (e.g. 64 for OLMoE).
    total_bandwidth_mbps:
        Sum of all nodes' ``storage_bandwidth_mbps``.  Used in the summary
        to show percentage shares.
    """

    node_experts: dict[str, list[int]]
    node_counts: dict[str, int]
    node_bandwidths: dict[str, int]
    fleet_hash: str
    num_experts: int
    total_bandwidth_mbps: int

    def summary(self) -> str:
        """Human-readable breakdown of the assignment.

        Returns a formatted table showing, for each node (sorted by
        ``node_id``): expert count, share of total experts, bandwidth, and
        share of total bandwidth.  Suitable for logging on every reshard
        event.  Always ends with a totals row and the fleet-state hash.
        """
        lines: list[str] = []
        header = (
            f"{'node_id':>38}  {'experts':>8}  {'% of E':>7}  "
            f"{'bw_mbps':>8}  {'% of BW':>7}"
        )
        sep = "─" * 90

        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        for nid in sorted(self.node_counts.keys()):
            count = self.node_counts[nid]
            bw = self.node_bandwidths.get(nid, 0)
            pct_e = (count / self.num_experts * 100) if self.num_experts > 0 else 0.0
            pct_bw = (bw / self.total_bandwidth_mbps * 100) if self.total_bandwidth_mbps > 0 else 0.0
            lines.append(
                f"{nid:>38}  {count:>8}  {pct_e:>6.1f}%  "
                f"{bw:>8}  {pct_bw:>6.1f}%"
            )

        lines.append(sep)
        total_pct_e = 100.0 if self.num_experts > 0 else 0.0
        total_pct_bw = 100.0 if self.total_bandwidth_mbps > 0 else 0.0
        lines.append(
            f"{'Total':>38}  {self.num_experts:>8}  {total_pct_e:>6.1f}%  "
            f"{self.total_bandwidth_mbps:>8}  {total_pct_bw:>6.1f}%"
        )
        lines.append(sep)
        lines.append(f"Fleet state hash: {self.fleet_hash}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Fleet-state hashing
# ═══════════════════════════════════════════════════════════════════════════════


def fleet_state_hash(nodes: list[NodeCapability]) -> str:
    """Compute a deterministic SHA-256 hash of the fleet state.

    The hash covers every ``(node_id, storage_bandwidth_mbps)`` pair in
    ``node_id``-sorted order, so two lists that contain the same nodes in
    a different order produce the same hash.

    Exposed as a separate function so a caller can cheaply ask "would my
    assignment change?" without running the full apportionment.  Two equal
    hashes guarantee the same assignment; two different hashes guarantee
    different input (though not necessarily different output — it is
    possible for different fleets to produce the same assignment by
    coincidence, but the hash is a reliable gating check).

    Parameters
    ----------
    nodes:
        The fleet snapshot to hash.  Order does not matter; the function
        sorts internally.

    Returns
    -------
    str
        A 64-character hex SHA-256 digest.
    """
    payload = ";".join(
        f"{n.node_id}:{n.storage_bandwidth_mbps}"
        for n in sorted(nodes, key=lambda n: n.node_id)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# Largest remainder apportionment (Hamilton's method)
# ═══════════════════════════════════════════════════════════════════════════════


def _apportion_largest_remainder(
    items: list[tuple[str, int]],
    total_items: int,
) -> dict[str, int]:
    """Apportion ``total_items`` among nodes proportionally to their weights.

    Implements the **largest remainder method** (Hamilton's method), the
    same algorithm used for allocating legislative seats proportionally to
    population.  It guarantees:

    * Every node gets at least ``floor(weight_i / total_weight * total_items)``.
    * The assigned counts always sum exactly to ``total_items``.
    * No systematic bias toward large or small shares (unlike naive rounding,
      which can leave unallocated items or over-allocate).

    Tie-breaking: when two nodes have identical fractional remainders, the
    node with the lexicographically smaller ``node_id`` wins.  This rule is
    the single source of determinism in the algorithm and is load-bearing —
    change it and identical fleets will no longer produce identical
    assignments.

    Parameters
    ----------
    items:
        List of ``(node_id, weight)`` pairs.  Weights must be non-negative.
        The list is sorted internally; input order does not matter.
    total_items:
        The total number of items to apportion (e.g. 64 experts).  Must be
        non-negative.

    Returns
    -------
    dict[str, int]
        Mapping from ``node_id`` to the integer number of items assigned.
        Sums to ``total_items`` exactly.

    Raises
    ------
    ValueError
        If ``total_items`` is negative.
    """
    if total_items < 0:
        raise ValueError(f"total_items must be non-negative, got {total_items}")

    if total_items == 0:
        return {nid: 0 for nid, _ in items}

    # Reject negative weights explicitly.  The correctness proof for this
    # method (counts are non-negative and sum exactly to total_items) rests
    # on weights being non-negative.  A negative weight would break that
    # silently rather than loudly, so fail loudly instead.
    negatives = [nid for nid, w in items if w < 0]
    if negatives:
        raise ValueError(
            f"storage_bandwidth_mbps must be non-negative; got negative "
            f"values for node(s): {sorted(negatives)}"
        )

    total_weight = sum(w for _, w in items)

    # All-zero weights → everyone gets zero.  We must not divide by zero
    # and there is no meaningful way to distribute items.
    if total_weight == 0:
        return {nid: 0 for nid, _ in items}

    # ── Phase 1: floor shares ──────────────────────────────────────────
    # Each node gets the integer part of its proportional quota.
    # Sum of floors ≤ total_items because sum(quota) = total_items and
    # floor(x) ≤ x for every term.
    #
    # Exact rational arithmetic, not float.  This module's core promise is
    # that counts sum to *exactly* total_items.  With floats, sum(quota)
    # can drift from total_items by ~1e-15, which in principle could flip
    # a floor() result or a remainder comparison.  Fraction makes the
    # guarantee exact rather than merely overwhelmingly likely, at
    # negligible cost for the small integers involved here, and keeps the
    # tie-break comparison exact too (two nodes with equal weight have
    # *identical* remainders, not almost-identical ones).
    quotas: list[tuple[str, Fraction, int]] = []  # (node_id, remainder, floor_share)
    for node_id, weight in items:
        quota = Fraction(weight, total_weight) * total_items
        floor_share = quota.numerator // quota.denominator  # exact floor, quota >= 0
        remainder = quota - floor_share  # exact Fraction
        quotas.append((node_id, remainder, floor_share))

    remaining = total_items - sum(q[2] for q in quotas)

    # ── Phase 2: distribute remainders ─────────────────────────────────
    # Sort by remainder descending, then by node_id ascending for ties.
    # The node_id tie-break is the load-bearing determinism source.
    quotas.sort(key=lambda q: (-q[1], q[0]))

    result: dict[str, int] = {}
    for i, (node_id, _remainder, floor_share) in enumerate(quotas):
        extra = 1 if i < remaining else 0
        result[node_id] = floor_share + extra

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def compute_assignment(
    nodes: list[NodeCapability],
    num_experts: int,
) -> ShardAssignment:
    """Compute a deterministic expert-to-node shard assignment.

    This is the main entry point.  It is a pure function: given the same
    ``nodes`` and ``num_experts``, it always returns the same
    ``ShardAssignment``, on any node, at any time, with no side effects.

    Parameters
    ----------
    nodes:
        The fleet snapshot.  Each ``NodeCapability`` describes one peer
        node's identity and measured storage bandwidth.  Order does not
        matter — the function sorts internally for determinism.  Nodes with
        ``storage_bandwidth_mbps == 0`` are valid and will receive zero
        experts.
    num_experts:
        Total number of expert slot indices to distribute (e.g. 64 for
        OLMoE).  Must be non-negative.  If smaller than the number of nodes
        with positive bandwidth, some nodes will receive zero experts — this
        is handled gracefully.

    Returns
    -------
    ShardAssignment
        The complete assignment, including the expert-to-node mapping, the
        fleet-state hash, and a ``summary()`` method for logging.

    Raises
    ------
    ValueError
        If ``num_experts`` is negative, or ``nodes`` is empty.
    """
    if not nodes:
        raise ValueError("nodes must not be empty")
    if num_experts < 0:
        raise ValueError(f"num_experts must be non-negative, got {num_experts}")

    # ── Apportion expert counts ────────────────────────────────────────
    items: list[tuple[str, int]] = [
        (n.node_id, n.storage_bandwidth_mbps) for n in nodes
    ]
    counts = _apportion_largest_remainder(items, num_experts)

    # ── Assign expert slot indices ─────────────────────────────────────
    # Contiguous blocks in node_id-sorted order.  This is the simplest
    # deterministic scheme and has no known downside for gang-mode
    # layer-sync (all layers share the same expert←→node mapping).
    node_experts: dict[str, list[int]] = {}
    offset = 0
    for nid in sorted(counts.keys()):
        c = counts[nid]
        if c > 0:
            node_experts[nid] = list(range(offset, offset + c))
        else:
            node_experts[nid] = []
        offset += c

    # ── Carry bandwidths for the summary ───────────────────────────────
    node_bandwidths: dict[str, int] = {
        n.node_id: n.storage_bandwidth_mbps for n in nodes
    }

    # ── Fleet-state hash ───────────────────────────────────────────────
    fhash = fleet_state_hash(nodes)

    total_bw = sum(n.storage_bandwidth_mbps for n in nodes)

    return ShardAssignment(
        node_experts=node_experts,
        node_counts=counts,
        node_bandwidths=node_bandwidths,
        fleet_hash=fhash,
        num_experts=num_experts,
        total_bandwidth_mbps=total_bw,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════


def _demo() -> None:
    """Run a self-contained demo with a realistic mixed-generation fleet.

    Models the actual hardware values from this project's docs/dials.md:
    one old-generation OSD32MP2-class node at ~1900 Mbps and three
    RK3588-class nodes at ~4000 Mbps each, apportioning 64 experts
    (OLMoE's real count).
    """
    print("=" * 90)
    print("  Swarm sharding demo — mixed-generation fleet, 64 experts (OLMoE)")
    print("=" * 90)

    # ── Build the fleet ────────────────────────────────────────────────
    fleet = [
        NodeCapability(node_id="44444444-0000-0000-0000-000000000001", storage_bandwidth_mbps=1900),
        NodeCapability(node_id="22222222-0000-0000-0000-000000000001", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="33333333-0000-0000-0000-000000000001", storage_bandwidth_mbps=4000),
        NodeCapability(node_id="11111111-0000-0000-0000-000000000001", storage_bandwidth_mbps=4000),
    ]

    print(f"\nFleet: {len(fleet)} nodes, "
          f"{sum(n.storage_bandwidth_mbps for n in fleet)} Mbps total bandwidth\n")

    # ── Compute ────────────────────────────────────────────────────────
    assignment = compute_assignment(fleet, num_experts=64)

    print(assignment.summary())

    # ── Show which experts each node owns (first 8 and last 2) ─────────
    print("\nExpert slot index ownership (samples):")
    for nid in sorted(assignment.node_experts.keys()):
        experts = assignment.node_experts[nid]
        if len(experts) <= 10:
            preview = str(experts)
        else:
            preview = f"[{experts[0]}, {experts[1]}, {experts[2]}, ..., {experts[-2]}, {experts[-1]}]  (total {len(experts)})"
        print(f"  {nid[:8]}…  →  {preview}")

    # ── Verify properties ──────────────────────────────────────────────
    print("\n── Verification ──")
    all_experts: list[int] = []
    for experts in assignment.node_experts.values():
        all_experts.extend(experts)
    all_experts.sort()
    expected = list(range(64))
    coverage_ok = all_experts == expected
    print(f"  Every expert index 0–63 assigned exactly once: {'✓' if coverage_ok else '✗ FAIL'}")

    count_sum = sum(assignment.node_counts.values())
    count_ok = count_sum == 64
    print(f"  Expert counts sum to 64: {'✓' if count_ok else f'✗ FAIL (got {count_sum})'}")

    # Determinism check: second call must be byte-identical.
    assignment2 = compute_assignment(fleet, num_experts=64)
    det_ok = assignment.fleet_hash == assignment2.fleet_hash
    print(f"  Determinism (same input → same hash): {'✓' if det_ok else '✗ FAIL'}")

    # Different fleet → different hash.
    fleet_changed = fleet + [NodeCapability(node_id="new-node-0000-0000-0000-00000000", storage_bandwidth_mbps=2000)]
    assignment3 = compute_assignment(fleet_changed, num_experts=64)
    diff_ok = assignment.fleet_hash != assignment3.fleet_hash
    print(f"  Fleet change → hash change: {'✓' if diff_ok else '✗ FAIL'}")

    # Zero-bandwidth node doesn't crash.
    fleet_with_zero = fleet + [NodeCapability(node_id="zero-node-0000-0000-0000-00000000", storage_bandwidth_mbps=0)]
    try:
        assignment_zero = compute_assignment(fleet_with_zero, num_experts=64)
        zero_count = assignment_zero.node_counts.get("zero-node-0000-0000-0000-00000000", -1)
        zero_ok = zero_count == 0
        print(f"  Zero-bandwidth node gets 0 experts: {'✓' if zero_ok else f'✗ FAIL (got {zero_count})'}")
    except Exception as e:
        print(f"  Zero-bandwidth node CRASHED: {e}")

    # num_experts < num_nodes: graceful zero-assignment, no crash.
    tiny = compute_assignment(fleet, num_experts=2)
    tiny_sum = sum(tiny.node_counts.values())
    tiny_ok = tiny_sum == 2
    print(f"  num_experts=2 < num_nodes=4, sum=2: {'✓' if tiny_ok else f'✗ FAIL (got {tiny_sum})'}")

    # All-zero fleet: no crash, everyone gets 0.
    fleet_zero = [
        NodeCapability(node_id="aaa", storage_bandwidth_mbps=0),
        NodeCapability(node_id="bbb", storage_bandwidth_mbps=0),
    ]
    try:
        zero_fleet_assignment = compute_assignment(fleet_zero, num_experts=10)
        zf_ok = all(c == 0 for c in zero_fleet_assignment.node_counts.values())
        print(f"  All-zero fleet assigned 0 experts: {'✓' if zf_ok else '✗ FAIL'}")
    except Exception as e:
        print(f"  All-zero fleet CRASHED: {e}")

    if all([coverage_ok, count_ok, det_ok, diff_ok, zero_ok, tiny_ok, zf_ok]):
        print("\n  All checks passed ✓")
    else:
        print("\n  Some checks FAILED — see above")


if __name__ == "__main__":
    _demo()
