#!/usr/bin/env python3
"""
failover.py — Fleet-membership-change → reshard-diff coordinator for the
Swarm distributed inference scheduler.

Layer 4 of the software stack.  This module's job: when FleetTable (Layer 3)
reports a membership change, wait a settle window, compute the new
expert-to-node assignment via ``sharding.py``, and produce a precise diff
describing which expert slot indices moved from which node to which node.

What this module does NOT do
----------------------------
- **Does not move bytes.**  It does not fetch expert weights, does not touch
  NVMe, does not know about model files.  Its output is a ``ReshardDiff``
  describing what changed; a not-yet-built higher layer (Layer 1/2) is
  responsible for actually fetching weights for newly-assigned experts.
- **Does not independently decide a node is dead.**  FleetTable (Layer 3) is
  the single source of truth for fleet membership, and it already debounces
  (8-second stale timeout, ~4 missed broadcasts).  This module must NOT
  implement a second, separately-tuned aliveness timer.
- **Does not do coordinator/leader election.**  Every node computes its own
  reshard diff independently from its own FleetTable view, same as
  ``sharding.py`` and ``gang_sync.py``.

Why Layer 3 stays the sole membership authority
-----------------------------------------------
FleetTable already owns the hard parts of membership tracking: receiving
UDP broadcasts, maintaining per-node ``(descriptor, last_seen)`` state,
an eviction sweep with a tuned stale timeout, and join/leave callbacks
that fire *after* state has changed.  Building a second aliveness layer on
top of that means two independently-tuned timeouts that can disagree — the
reshard would fire at a different time than the fleet view changes, and
the two can oscillate against each other.  Instead, this module registers
for FleetTable's callbacks and adds a short *settle* window on top: not to
determine aliveness (Layer 3 already did that), but to coalesce bursts of
membership changes that arrive close together in time.

Why the settle window exists (and why it's on top of Layer 3, not instead of)
-----------------------------------------------------------------------------
Layer 3's 8-second stale timeout answers "is this node still broadcasting?"
Layer 4's 2-second settle window answers a different question: "have
membership changes stopped arriving, or is the fleet still in flux?"

During bring-up, several nodes may join within a second of each other.
Without a settle window, that produces one reshard per join — three or
more recomputations in rapid succession, each invalidating the gang-sync
ring order before it can be used.  With a 2-second settle window, those
three joins produce ONE reshard once they've all landed.

The settle window is *on top of* Layer 3's eviction debounce, not instead
of it, because they answer different questions at different timescales.

Why report_gang_sync_failure does NOT directly trigger a reshard
----------------------------------------------------------------
A ``GangSyncTimeout`` naming a specific peer means that peer didn't respond
during a ring-reduce pass.  But that could mean:

1. The peer is genuinely dead → FleetTable's eviction sweep will detect it
   soon (next sweep, within 2 seconds) and fire a leave callback, which
   triggers a reshard through the normal path.
2. The peer had a transient network blip → it's still broadcasting, still
   in FleetTable, and the next layer pass will likely succeed.

If ``report_gang_sync_failure`` directly triggered a reshard, case 2 would
cause the fleet to reshard on every transient blip — thrashing the expert
assignment, forcing weight re-fetches, and disrupting ongoing work.  By
instead only checking FleetTable's current state, we let Layer 3's existing
aliveness machinery be the sole trigger for reshard events, and use
gang-sync failures only as a diagnostic signal that can prompt an
out-of-cycle FleetTable check.

Design rules (do not break)
----------------------------
- Python 3.11+, standard library only, plus node_identity.py and sharding.py.
- All background tasks strongly referenced (asyncio.create_task pitfall —
  this makes four times across this project).
- FleetTable is the sole membership authority; no second timer.
- Coordinator-free: every node independently computes the same diff.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from node_identity import FleetTable, NodeDescriptor
from sharding import NodeCapability, ShardAssignment, compute_assignment

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger("swarm.failover")

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_SETTLE_WINDOW: float = 2.0  # seconds


# ── ReshardDiff ─────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ReshardDiff:
    """Precise description of what changed between two ShardAssignments.

    Built by ``FailoverCoordinator`` after a membership change settles.
    The caller uses this to decide which experts to fetch from cold storage
    and which to discard, without recomputing the assignment itself.

    Attributes
    ----------
    old_fleet_hash:
        SHA-256 of the previous fleet state, or ``None`` for the first-ever
        assignment (no prior state existed).
    new_fleet_hash:
        SHA-256 of the new fleet state.
    new_assignment:
        The full newly-computed ``ShardAssignment``.  This is what
        ``gang_sync.ring_reduce`` needs for its next layer pass.
    moved:
        Mapping from expert slot index to ``(old_owner, new_owner)``.
        Only experts whose ownership changed appear here.  ``old_owner``
        is ``None`` for experts that had no prior owner (first assignment,
        or fleet grew).  ``new_owner`` is ``None`` for experts that have
        no owner after the change (fleet shrank below the expert count
        with all-zero-bandwidth peers — a real, valid state, not an error).
    nodes_added:
        Node IDs present in the new assignment but not the old one.
    nodes_removed:
        Node IDs present in the old assignment but not the new one.
    unchanged_count:
        Number of expert slot indices whose owner did not change.
    moved_count:
        Number of expert slot indices whose owner changed.  Equal to
        ``len(moved)``.
    """

    old_fleet_hash: str | None
    new_fleet_hash: str
    new_assignment: ShardAssignment
    moved: dict[int, tuple[str | None, str | None]]
    nodes_added: list[str]
    nodes_removed: list[str]
    unchanged_count: int
    moved_count: int

    def summary(self) -> str:
        """Human-readable description of the reshard."""
        lines: list[str] = []
        sep = "─" * 72

        lines.append(sep)
        if self.old_fleet_hash is None:
            lines.append("  Reshard: FIRST ASSIGNMENT (no prior state)")
        else:
            lines.append("  Reshard: fleet state changed")
            lines.append(
                f"    Old hash: {self.old_fleet_hash[:16]}…"
            )
        lines.append(f"    New hash: {self.new_fleet_hash[:16]}…")

        if self.nodes_added:
            lines.append(f"    Nodes added:   {_fmt_ids(self.nodes_added)}")
        if self.nodes_removed:
            lines.append(f"    Nodes removed: {_fmt_ids(self.nodes_removed)}")

        lines.append(
            f"    Experts: {self.moved_count} moved, "
            f"{self.unchanged_count} unchanged"
        )

        if self.moved_count > 0 and self.moved_count <= 20:
            # Print every moved expert when the count is small enough to
            # be readable.
            lines.append("    Expert moves:")
            for expert_idx in sorted(self.moved):
                old, new = self.moved[expert_idx]
                old_str = old[:8] + "…" if old else "(none)"
                new_str = new[:8] + "…" if new else "(none)"
                lines.append(f"      expert {expert_idx:>4}: {old_str} → {new_str}")
        elif self.moved_count > 20:
            # Too many to print individually — show a sample.
            lines.append("    Expert moves (sample of first 10):")
            for expert_idx in sorted(self.moved)[:10]:
                old, new = self.moved[expert_idx]
                old_str = old[:8] + "…" if old else "(none)"
                new_str = new[:8] + "…" if new else "(none)"
                lines.append(f"      expert {expert_idx:>4}: {old_str} → {new_str}")
            lines.append(f"      … and {self.moved_count - 10} more")

        lines.append(sep)
        return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _fmt_ids(node_ids: list[str]) -> str:
    """Format a list of node_ids for human display (first 8 chars each)."""
    return ", ".join(nid[:8] + "…" for nid in node_ids)


def _build_expert_owner_map(
    assignment: ShardAssignment,
) -> dict[int, str]:
    """Build an ``expert_idx → node_id`` reverse map from an assignment.

    Every expert index appears in exactly one node's list by construction
    (``sharding.py`` guarantees this), so there are no collisions.
    """
    owner: dict[int, str] = {}
    for node_id, experts in assignment.node_experts.items():
        for idx in experts:
            owner[idx] = node_id
    return owner


def _compute_diff(
    old: ShardAssignment | None,
    new: ShardAssignment,
) -> ReshardDiff:
    """Build a ``ReshardDiff`` comparing two assignments.

    Parameters
    ----------
    old:
        The previous assignment, or ``None`` for the first-ever assignment.
    new:
        The newly-computed assignment.
    """
    if old is None:
        # First-ever assignment: every expert is newly assigned.
        old_owners: dict[int, str] = {}
        old_node_ids: set[str] = set()
    else:
        old_owners = _build_expert_owner_map(old)
        old_node_ids = set(old.node_counts.keys())

    new_owners = _build_expert_owner_map(new)
    new_node_ids = set(new.node_counts.keys())

    num_experts = new.num_experts
    moved: dict[int, tuple[str | None, str | None]] = {}
    unchanged_count = 0

    for expert_idx in range(num_experts):
        old_owner = old_owners.get(expert_idx)  # None if not assigned before
        new_owner = new_owners.get(expert_idx)  # None if not assigned now
        if old_owner != new_owner:
            moved[expert_idx] = (old_owner, new_owner)
        else:
            unchanged_count += 1

    nodes_added = sorted(new_node_ids - old_node_ids)
    nodes_removed = sorted(old_node_ids - new_node_ids)

    return ReshardDiff(
        old_fleet_hash=old.fleet_hash if old else None,
        new_fleet_hash=new.fleet_hash,
        new_assignment=new,
        moved=moved,
        nodes_added=nodes_added,
        nodes_removed=nodes_removed,
        unchanged_count=unchanged_count,
        moved_count=len(moved),
    )


# ── FailoverCoordinator ────────────────────────────────────────────────────────


class FailoverCoordinator:
    """Watches FleetTable for membership changes and produces ReshardDiffs.

    Registers for FleetTable's join/leave callbacks in ``start()``.  On
    every membership change, starts (or resets) a settle-window timer.
    When the settle window expires with no further changes, computes a
    new ``ShardAssignment`` via ``sharding.py``, compares it to the
    previous one, and — if the fleet state genuinely changed — pushes a
    ``ReshardDiff`` onto ``diff_queue``.

    Parameters
    ----------
    fleet_table:
        The Layer 3 FleetTable whose membership events drive resharding.
        Must have been created with the same ``own_node_id``.
    own_node_id:
        Stable UUID of *this* node.  Must match the FleetTable's
        ``own_node_id``.  Used to include self in the
        ``compute_assignment`` call (FleetTable filters self out).
    num_experts:
        Total number of expert slot indices (e.g. 64 for OLMoE).
    own_storage_bandwidth_mbps:
        This node's measured storage bandwidth, advertised in its own
        ``NodeCapability`` when computing the assignment.  Defaults to
        ``0`` ("unmeasured") — a separate benchmarking module should
        update this via ``update_own_bandwidth()``.
    settle_window:
        Seconds to wait after the last membership change before
        recomputing (default 2.0).  See module docstring for why this
        exists and why it's layered on top of Layer 3's own debounce.
    """

    def __init__(
        self,
        *,
        fleet_table: FleetTable,
        own_node_id: str,
        num_experts: int,
        own_storage_bandwidth_mbps: int = 0,
        settle_window: float = DEFAULT_SETTLE_WINDOW,
    ) -> None:
        self._fleet = fleet_table
        self._own_node_id = own_node_id
        self._num_experts = num_experts
        self._own_bandwidth_mbps = own_storage_bandwidth_mbps
        self._settle_window = settle_window

        # The most recently computed assignment.  None before the first one.
        self._current: ShardAssignment | None = None

        # Notification queue.  Every ReshardDiff is pushed here after a
        # membership change settles and the fleet state genuinely changed.
        # The caller (Layer 1/2) pulls from this and decides what weights
        # to fetch/discard.
        self.diff_queue: asyncio.Queue[ReshardDiff] = asyncio.Queue()

        # Settle-window task state.  Each membership change cancels the
        # previous settle task (if still waiting) and creates a new one.
        # The task that survives to the end of the settle window fires
        # the reshard computation.
        self._settle_task: asyncio.Task[None] | None = None

        # Strong references to background tasks (see module docstring).
        self._background_tasks: set[asyncio.Task] = set()

        # Whether start() has been called.
        self._started: bool = False

    # ── Public API ─────────────────────────────────────────────────────────

    def current_assignment(self) -> ShardAssignment | None:
        """The most recently computed ``ShardAssignment``.

        Returns ``None`` before the first assignment is computed (i.e.
        before ``start()`` is called and a membership change triggers
        the first reshard).
        """
        return self._current

    def update_own_bandwidth(self, mbps: int) -> None:
        """Update this node's storage bandwidth for future assignments.

        Does NOT trigger a reshard on its own — a bandwidth change is
        not a membership change.  The new value takes effect on the
        next membership-driven recomputation.
        """
        self._own_bandwidth_mbps = mbps
        logger.info("Own storage bandwidth updated to %d Mbps", mbps)

    def report_gang_sync_failure(self, layer_id: int, node_id: str) -> None:
        """Diagnostic hook called when a ``GangSyncTimeout`` names *node_id*.

        **This does NOT directly trigger a reshard.**  See the module
        docstring for the full reasoning.  In summary:

        1. If FleetTable still considers *node_id* alive → log a warning
           and do nothing further.  This was a transient network blip;
           the next layer pass will likely succeed.
        2. If FleetTable has already removed *node_id* → the leave
           callback already fired (or will fire momentarily), and a
           reshard is already in progress through the normal path.  Log
           an informational message and do nothing further.

        Parameters
        ----------
        layer_id:
            The layer whose ring pass timed out.
        node_id:
            The peer that failed to respond.
        """
        # This is a sync method so it can be called from inside an
        # exception handler without restructuring the caller.  It requires
        # a *running* event loop -- asyncio.create_task raises RuntimeError
        # otherwise.  That is the intended contract: this is only ever
        # called from async code that just caught a GangSyncTimeout.
        task = asyncio.create_task(
            self._check_gang_sync_failure(layer_id, node_id)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def start(self) -> None:
        """Register for FleetTable callbacks and begin watching.

        Idempotent — calling ``start()`` on an already-started coordinator
        is a no-op.
        """
        if self._started:
            return
        self._started = True

        self._fleet.on_join(self._on_peer_join)
        self._fleet.on_leave(self._on_peer_leave)
        logger.info(
            "FailoverCoordinator started (node %s…, %d experts, "
            "settle window %.1fs)",
            self._own_node_id[:8],
            self._num_experts,
            self._settle_window,
        )

    # ── FleetTable callbacks ───────────────────────────────────────────────

    async def _on_peer_join(self, descriptor: NodeDescriptor) -> None:
        """FleetTable join callback — a new peer appeared."""
        logger.info(
            "Peer joined: %s… (%s, %d Mbps)",
            descriptor.node_id[:8],
            descriptor.hostname,
            descriptor.storage_bandwidth_mbps,
        )
        self._schedule_reshard()

    async def _on_peer_leave(self, node_id: str, descriptor: NodeDescriptor) -> None:
        """FleetTable leave callback — a peer was evicted as stale."""
        logger.info(
            "Peer left: %s… (%s)",
            node_id[:8],
            descriptor.hostname,
        )
        self._schedule_reshard()

    # ── Settle-window machinery ────────────────────────────────────────────

    def _schedule_reshard(self) -> None:
        """(Re)start the settle-window timer.

        Called on every membership change.  Cancels any in-progress settle
        task and starts a fresh one, so a burst of changes produces exactly
        one reshard after the last change plus ``settle_window`` seconds.
        """
        if self._settle_task is not None and not self._settle_task.done():
            self._settle_task.cancel()

        self._settle_task = asyncio.create_task(self._settle_and_reshard())
        self._background_tasks.add(self._settle_task)
        self._settle_task.add_done_callback(self._background_tasks.discard)

    async def _settle_and_reshard(self) -> None:
        """Wait for the settle window, then recompute and emit the diff.

        If cancelled (a new membership change arrived before the window
        expired), the CancelledError is caught silently — the newer task
        will handle it.
        """
        try:
            await asyncio.sleep(self._settle_window)
        except asyncio.CancelledError:
            # A newer membership change cancelled us.  Let it handle the
            # reshard — we should not fire a stale one.
            return

        # Once we get here, we're committed — the settle window expired
        # with no further changes.
        await self._do_reshard()

    async def _do_reshard(self) -> None:
        """Snapshot FleetTable, compute the new assignment, and emit a diff.

        If the fleet state is unchanged from the previous assignment
        (same ``fleet_hash``), no diff is emitted — the caller sees
        nothing, and no work is wasted on a fleet that returned to its
        prior state during the settle window.
        """
        # ── Snapshot: self + all live peers ────────────────────────────
        peers = await self._fleet.get_live_nodes()

        caps: list[NodeCapability] = []
        caps.append(
            NodeCapability(
                node_id=self._own_node_id,
                storage_bandwidth_mbps=self._own_bandwidth_mbps,
            )
        )
        for peer in peers:
            caps.append(
                NodeCapability(
                    node_id=peer.node_id,
                    storage_bandwidth_mbps=peer.storage_bandwidth_mbps,
                )
            )

        # ── Compute new assignment ─────────────────────────────────────
        new_assignment = compute_assignment(caps, self._num_experts)

        # ── Skip if fleet state is unchanged ───────────────────────────
        # This handles the "fleet returned to prior state during settle
        # window" case: a node's broadcast was briefly delayed, Layer 3
        # almost evicted it (leave callback → settle timer started), then
        # it reappeared (join callback → settle timer reset).  When the
        # timer finally fires, the fleet is back to exactly what it was
        # before — same fleet_hash — so we skip the reshard entirely.
        if self._current is not None and new_assignment.fleet_hash == self._current.fleet_hash:
            logger.debug(
                "Fleet state unchanged (hash %s…) — skipping reshard",
                new_assignment.fleet_hash[:16],
            )
            return

        # ── Build and emit the diff ────────────────────────────────────
        #
        # Order matters, and so does using put_nowait rather than await put.
        # A settle task can be cancelled at any await point.  If _current
        # were updated and then an *awaiting* put() were cancelled before
        # delivering, the coordinator would believe it had published a
        # reshard that no caller ever received -- and the next diff would
        # be computed against the new baseline, silently skipping the
        # transition entirely.  diff_queue is unbounded, so put_nowait
        # cannot fail here, and it introduces no cancellation point between
        # the state update and the publish.
        diff = _compute_diff(self._current, new_assignment)

        logger.info(
            "Reshard: %s → %s (%d experts moved, %d unchanged, "
            "%d nodes added, %d removed)",
            diff.old_fleet_hash[:16] + "…" if diff.old_fleet_hash else "(none)",
            diff.new_fleet_hash[:16] + "…",
            diff.moved_count,
            diff.unchanged_count,
            len(diff.nodes_added),
            len(diff.nodes_removed),
        )

        self._current = new_assignment
        self.diff_queue.put_nowait(diff)

    # ── Gang-sync failure check ────────────────────────────────────────────

    async def _check_gang_sync_failure(
        self, layer_id: int, node_id: str
    ) -> None:
        """Async body of ``report_gang_sync_failure``.

        Checks FleetTable for *node_id*'s current status and logs
        accordingly.  Never triggers a reshard directly.
        """
        peers = await self._fleet.get_live_nodes()
        peer_ids = {p.node_id for p in peers}

        if node_id in peer_ids:
            # Node is still in FleetTable — transient network blip.
            logger.warning(
                "GangSyncTimeout for %s… at layer %d, but FleetTable "
                "reports it as alive — transient network blip, ignoring. "
                "No reshard triggered.",
                node_id[:8],
                layer_id,
            )
        else:
            # Node is already gone from FleetTable.  Its leave callback
            # already fired (or will fire momentarily), and a reshard is
            # in progress (or already completed) through the normal path.
            logger.info(
                "GangSyncTimeout for %s… at layer %d — node already "
                "removed from fleet. Reshard is in progress via normal "
                "FleetTable leave path.",
                node_id[:8],
                layer_id,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Minimal fake FleetTable for deterministic demo
# ═══════════════════════════════════════════════════════════════════════════════


class _FakeFleetTable:
    """Minimal fake of ``FleetTable`` for the demo.

    Supports ``on_join`` / ``on_leave`` callback registration and manual
    node injection.  No UDP, no timers, fully deterministic — the demo
    controls exactly when membership events fire.
    """

    def __init__(self, own_node_id: str) -> None:
        self._own_node_id = own_node_id
        self._nodes: dict[str, NodeDescriptor] = {}
        self._join_callbacks: list[
            Callable[[NodeDescriptor], Awaitable[None] | None]
        ] = []
        self._leave_callbacks: list[
            Callable[[str, NodeDescriptor], Awaitable[None] | None]
        ] = []

    def on_join(
        self, callback: Callable[[NodeDescriptor], Awaitable[None] | None]
    ) -> None:
        self._join_callbacks.append(callback)

    def on_leave(
        self, callback: Callable[[str, NodeDescriptor], Awaitable[None] | None]
    ) -> None:
        self._leave_callbacks.append(callback)

    async def get_live_nodes(self) -> list[NodeDescriptor]:
        return list(self._nodes.values())

    async def add_node(self, desc: NodeDescriptor) -> None:
        """Inject a node and fire join callbacks.

        Raises ``ValueError`` if *desc.node_id* matches ``own_node_id``
        (same as the real FleetTable, which silently drops self).
        """
        if desc.node_id == self._own_node_id:
            raise ValueError("Cannot add own_node_id to fake FleetTable")
        self._nodes[desc.node_id] = desc
        for cb in self._join_callbacks:
            result = cb(desc)
            if asyncio.iscoroutine(result):
                await result

    async def remove_node(self, node_id: str) -> None:
        """Remove a node and fire leave callbacks.  No-op if not present."""
        desc = self._nodes.pop(node_id, None)
        if desc is None:
            return
        for cb in self._leave_callbacks:
            result = cb(node_id, desc)
            if asyncio.iscoroutine(result):
                await result


# ── Demo helpers ───────────────────────────────────────────────────────────────


def _make_desc(
    node_id: str,
    hostname: str,
    storage_bandwidth_mbps: int = 4000,
) -> NodeDescriptor:
    """Build a ``NodeDescriptor`` with sensible demo defaults.

    Only ``node_id``, ``hostname``, and ``storage_bandwidth_mbps`` vary
    across demo nodes; the rest are fixed filler.
    """
    return NodeDescriptor(
        node_id=node_id,
        hostname=hostname,
        ip="192.168.0.1",
        port=9999,
        ram_total_mb=8192,
        ram_available_mb=4096,
        storage_bandwidth_mbps=storage_bandwidth_mbps,
        hardware_gen="rk3588-8gb",
        load=0.25,
        uptime_seconds=42.0,
        timestamp=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════


async def _demo() -> None:
    """Run a self-contained demo of failover coordination.

    Scenario:
      1. Start with 3 peers + self → first-ever assignment
      2. Add a 4th peer → reshard (some experts move to new node)
      3. Remove one of the original 3 → reshard (its experts redistribute)
      4. Settle-window coalescing: two rapid membership changes produce
         ONE diff, not two
      5. Fleet returns to prior state during settle window → no-op, no diff
      6. report_gang_sync_failure: node alive (warning) vs node dead (info)
    """
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("swarm.failover").setLevel(logging.INFO)

    OWN_ID = "self-0000-0000-0000-000000000000"
    NODE_A = "aaaa0000-0000-0000-0000-000000000000"
    NODE_B = "bbbb0000-0000-0000-0000-000000000000"
    NODE_C = "cccc0000-0000-0000-0000-000000000000"
    NODE_D = "dddd0000-0000-0000-0000-000000000000"

    fleet = _FakeFleetTable(own_node_id=OWN_ID)
    coordinator = FailoverCoordinator(
        fleet_table=fleet,
        own_node_id=OWN_ID,
        num_experts=64,
        own_storage_bandwidth_mbps=4000,
        settle_window=0.3,  # short for demo speed
    )
    await coordinator.start()

    # ── Helper: pull and print the next diff ───────────────────────────
    # The queue has no timeout here since the demo controls timing exactly.
    # In production a caller would use asyncio.wait_for or a get-nowait
    # pattern; for the demo, straight get() is fine because we know each
    # step produces exactly one diff.

    async def await_diff(expected_scenario: str) -> ReshardDiff:
        diff = await coordinator.diff_queue.get()
        print(f"\n── {expected_scenario} ──")
        print(diff.summary())
        print()
        return diff

    # ═══════════════════════════════════════════════════════════════════
    # Step 1: First-ever assignment (3 peers + self)
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 72)
    print("  FailoverCoordinator Demo")
    print("=" * 72)
    print(f"  Self:  {OWN_ID[:8]}…  @ 4000 Mbps")
    print(f"  Peers: 3x RK3588-class, each @ 4000 Mbps")
    print(f"  Experts: 64 (OLMoE), settle window: 0.3s")
    print()

    await fleet.add_node(_make_desc(NODE_A, "alpha"))
    await fleet.add_node(_make_desc(NODE_B, "bravo"))
    await fleet.add_node(_make_desc(NODE_C, "charlie"))
    await asyncio.sleep(0.5)  # let settle window expire

    diff1 = await await_diff("Step 1: First-ever assignment (3 peers + self)")
    assert diff1.old_fleet_hash is None, "first assignment should have no old hash"
    assert diff1.moved_count == 64, "all 64 experts should be newly assigned"
    assert diff1.unchanged_count == 0
    assert len(diff1.nodes_added) == 4  # self + 3 peers
    assert len(diff1.nodes_removed) == 0

    # Show the full assignment.
    print("  Full assignment (via current_assignment()):")
    print(coordinator.current_assignment().summary())
    print()

    # ═══════════════════════════════════════════════════════════════════
    # Step 2: Add a 4th peer
    # ═══════════════════════════════════════════════════════════════════
    await fleet.add_node(_make_desc(NODE_D, "delta"))
    await asyncio.sleep(0.5)

    diff2 = await await_diff("Step 2: 4th peer (delta) joins")
    assert NODE_D in diff2.nodes_added
    assert len(diff2.nodes_removed) == 0
    assert diff2.moved_count > 0, "some experts should move to the new node"

    # Show the new assignment so we can see the bandwidth-weighted split.
    print("  Full assignment (5 nodes now):")
    print(coordinator.current_assignment().summary())
    print()

    # ═══════════════════════════════════════════════════════════════════
    # Step 3: Remove one of the original peers
    # ═══════════════════════════════════════════════════════════════════
    await fleet.remove_node(NODE_A)
    await asyncio.sleep(0.5)

    diff3 = await await_diff("Step 3: alpha leaves")
    assert NODE_A in diff3.nodes_removed
    assert len(diff3.nodes_added) == 0
    assert diff3.moved_count > 0, "alpha's experts should redistribute"

    print("  Full assignment (back to 4 nodes):")
    print(coordinator.current_assignment().summary())
    print()

    # ═══════════════════════════════════════════════════════════════════
    # Step 4: Settle-window coalescing
    # ═══════════════════════════════════════════════════════════════════
    print("── Step 4: Settle-window coalescing ──")
    print("  Two changes within 0.3s → ONE diff, not two.")
    print()

    # Add one node, then quickly add another — both within settle window.
    await fleet.add_node(_make_desc(NODE_A, "alpha-returns"))
    await asyncio.sleep(0.05)  # well within the 0.3s settle window
    await fleet.add_node(_make_desc("eeee0000-0000-0000-0000-000000000000", "echo"))
    await asyncio.sleep(0.5)

    diff4 = await await_diff("Step 4: Two joins coalesced into one diff")
    assert len(diff4.nodes_added) == 2, (
        f"expected 2 nodes added, got {len(diff4.nodes_added)} "
        f"({diff4.nodes_added})"
    )
    assert len(diff4.nodes_removed) == 0

    # Verify no second diff arrived (the coalescing worked).
    assert coordinator.diff_queue.empty(), (
        "a second diff was emitted — coalescing failed"
    )
    print("  Queue empty after coalesced diff ✓")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # Step 5: Fleet returns to prior state during settle window
    # ═══════════════════════════════════════════════════════════════════
    print("── Step 5: Fleet returns to prior state during settle window ──")
    print("  Remove echo, add echo back within settle window → "
          "fleet_state_hash unchanged → no diff emitted.")
    print()

    # Capture current hash before the blip.
    hash_before = coordinator.current_assignment().fleet_hash

    await fleet.remove_node("eeee0000-0000-0000-0000-000000000000")
    await asyncio.sleep(0.05)
    await fleet.add_node(_make_desc("eeee0000-0000-0000-0000-000000000000", "echo"))
    await asyncio.sleep(0.5)

    # Fleet should be back to the exact same state.
    hash_after = coordinator.current_assignment().fleet_hash
    assert hash_before == hash_after, "fleet_hash should be unchanged"

    # Queue must still be empty — no diff was emitted.
    assert coordinator.diff_queue.empty(), (
        "a diff was emitted for a fleet that returned to its prior state"
    )
    print("  No diff emitted — fleet returned to prior state ✓")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # Step 6: report_gang_sync_failure
    # ═══════════════════════════════════════════════════════════════════
    print("── Step 6: report_gang_sync_failure ──")

    # Case A: Node is still alive (transient blip).
    print("  Case A: gang-sync timeout for a node still in FleetTable…")
    coordinator.report_gang_sync_failure(layer_id=5, node_id=NODE_B)
    await asyncio.sleep(0.2)
    # No diff should appear — B is still alive.
    assert coordinator.diff_queue.empty(), (
        "a diff was emitted for a transient gang-sync blip"
    )
    print("  → Warning logged, no reshard triggered ✓")

    # Case B: Node is genuinely gone.
    print("  Case B: gang-sync timeout for a node already removed…")
    await fleet.remove_node(NODE_B)
    await asyncio.sleep(0.05)  # let leave callback fire
    coordinator.report_gang_sync_failure(layer_id=5, node_id=NODE_B)
    await asyncio.sleep(0.5)
    # A diff SHOULD appear — because FleetTable's leave callback already
    # started the reshard. The gang-sync failure just logged info.
    diff_dead = await await_diff("Step 6b: Reshard triggered by FleetTable leave (not by gang-sync failure)")
    assert NODE_B in diff_dead.nodes_removed
    print("  → Info logged, reshard already in progress via FleetTable ✓")

    # ═══════════════════════════════════════════════════════════════════
    print("=" * 72)
    print("  Demo complete — all steps passed.")
    print("=" * 72)


def main() -> None:
    """Entry point: run the demo."""
    asyncio.run(_demo())


if __name__ == "__main__":
    main()
