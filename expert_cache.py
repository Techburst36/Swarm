#!/usr/bin/env python3
"""
expert_cache.py — Expert cache manager and prefetch predictor for Swarm layer 2.

Single-node expert streaming: decides what to keep in memory, what to evict,
and what to speculatively fetch before it is needed.

Design rules (do not break)
----------------------------
- Python 3.11+, standard library only, plus storage_io.py.
- Hard memory budget in **bytes**, not entry count.  Experts differ in size
  across quantization formats, and a count-based cache silently overshoots.
- Pinned experts are never evicted.  The shared expert (dials.md dial 14) is
  pinned explicitly; pinning is a first-class API, not a hack.
- LRU eviction over the unpinned remainder.
- Prefetch is speculative: a wrong prediction wastes bandwidth, never
  correctness.  A prefetched expert that isn't needed is simply evicted.
- Concurrent ``get()`` for the same uncached expert coalesces into **one**
  read.  Two callers requesting the same expert must not double-read.
- Instrumentation is first-class: every counter is exposed and dumpable.
- All background tasks are strongly referenced (asyncio.create_task pitfall
  — this has been a real bug four times in this codebase already).

Concurrent-read coalescing guarantee
-------------------------------------
When two ``get()`` calls request the same uncached expert simultaneously,
exactly one read is issued to the ``ExpertStore``.  Both callers receive
the same bytes.  This is implemented via a per-key ``asyncio.Future``
dictionary (``_in_flight``):

1. Caller A calls ``get(layer=0, expert=5)`` — cache miss, no in-flight
   future → creates a future, stores it, starts a read task.
2. Caller B calls ``get(layer=0, expert=5)`` — cache miss, finds the
   in-flight future → awaits it without starting a second read.
3. Read completes → future resolved → both A and B receive the data.
   The entry is cached once.  Both callers' ``bytes_served`` counters
   are incremented.

Prefetch uses the same coalescing dictionary, so a ``prefetch()`` and a
concurrent ``get()`` for the same expert also share one read.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from storage_io import ExpertStore

logger = logging.getLogger("swarm.expert_cache")

# ── Constants ──────────────────────────────────────────────────────────────────

# Provisional defaults — all overridable.
DEFAULT_MAX_CONCURRENT_PREFETCH: int = 8  # max in-flight prefetch reads
DEFAULT_SHARED_EXPERT_INDEX: int = 0  # expert index 0 is conventionally shared


# ── CacheEntry ─────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class CacheEntry:
    """One cached expert with LRU and prefetch-tracking metadata.

    Attributes
    ----------
    data:
        The expert weight bytes.  Opaque to this module.
    size:
        Length of *data* in bytes.  Cached at insertion time so the eviction
        loop does not call ``len()`` repeatedly.
    last_access:
        ``time.monotonic()`` at the moment of the most recent ``get()`` hit
        or insertion.  Guards the entry against eviction: entries with higher
        ``last_access`` are evicted after ones with lower values.
    from_prefetch:
        ``True`` if this entry was loaded via ``prefetch()`` rather than a
        direct ``get()`` miss.
    accessed:
        ``True`` if this entry was ever returned by a ``get()`` hit after
        being cached.  Distinguishes "prefetched and used" from "prefetched
        and wasted."
    pinned:
        ``True`` if this entry must never be evicted (e.g. the shared expert).
    """

    data: bytes
    size: int
    last_access: float
    from_prefetch: bool
    accessed: bool
    pinned: bool = False


# ── CacheStats ─────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class CacheStats:
    """Instrumentation counters for an ExpertCache.

    All fields are monotonic counters except ``current_bytes`` and
    ``current_entries``, which reflect instantaneous state.

    Use ``snapshot()`` on a live cache to get a copy of the current values.
    """

    hits: int = 0
    misses: int = 0
    coalesced: int = 0  # get() that found an in-flight read
    bytes_read: int = 0  # bytes actually read from ExpertStore
    bytes_served: int = 0  # bytes returned to callers
    prefetch_issued: int = 0  # prefetch() calls (one per expert)
    prefetch_hits: int = 0  # prefetched entries that were later accessed
    prefetch_waste: int = 0  # prefetched entries evicted without being accessed
    evictions: int = 0
    current_bytes: int = 0
    current_entries: int = 0
    peak_bytes: int = 0
    peak_entries: int = 0

    def snapshot(self) -> CacheStats:
        """Return a copy of the current counters."""
        return dataclasses.replace(self)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def prefetch_accuracy(self) -> float:
        """Fraction of prefetched entries that were actually used."""
        total = self.prefetch_hits + self.prefetch_waste
        if total == 0:
            return 0.0
        return self.prefetch_hits / total

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"Cache: {self.current_entries} entries, "
            f"{self.current_bytes / (1024*1024):.1f} MB, "
            f"hit_rate={self.hit_rate:.1%}, "
            f"prefetch_acc={self.prefetch_accuracy:.1%}, "
            f"bytes_read={self.bytes_read / (1024*1024):.1f} MB, "
            f"bytes_served={self.bytes_served / (1024*1024):.1f} MB"
        )

    def to_dict(self) -> dict[str, Any]:
        """All fields as a flat dict for JSON serialisation."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "coalesced": self.coalesced,
            "bytes_read": self.bytes_read,
            "bytes_served": self.bytes_served,
            "prefetch_issued": self.prefetch_issued,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_waste": self.prefetch_waste,
            "evictions": self.evictions,
            "current_bytes": self.current_bytes,
            "current_entries": self.current_entries,
            "peak_bytes": self.peak_bytes,
            "peak_entries": self.peak_entries,
            "hit_rate": self.hit_rate,
            "prefetch_accuracy": self.prefetch_accuracy,
        }


# ── ExpertCache ────────────────────────────────────────────────────────────────


class ExpertCache:
    """LRU cache for expert weights with pinned entries and prefetch.

    Parameters
    ----------
    store:
        The ``ExpertStore`` to read through on cache miss.
    memory_budget_bytes:
        Hard cap on total cached bytes.  Evictions fire on insertion to
        keep usage at or below this limit.  Pinned entries count against
        the budget (they occupy real memory) but are never chosen for
        eviction.
    max_concurrent_prefetch:
        Maximum number of in-flight prefetch reads.  ``prefetch()`` calls
        beyond this limit block briefly at a semaphore.
    """

    def __init__(
        self,
        *,
        store: ExpertStore,
        memory_budget_bytes: int,
        max_concurrent_prefetch: int = DEFAULT_MAX_CONCURRENT_PREFETCH,
    ) -> None:
        self._store = store
        self._budget = memory_budget_bytes
        self._prefetch_sem = asyncio.Semaphore(max_concurrent_prefetch)

        # ── Cache storage ──────────────────────────────────────────────
        # (layer, expert) → CacheEntry
        self._cache: dict[tuple[int, int], CacheEntry] = {}

        # LRU order: OrderedDict mapping key → True, with most-recently-used
        # at the right end.  Eviction picks from the left (least-recent).
        # Only non-pinned entries appear here.
        self._lru: OrderedDict[tuple[int, int], bool] = OrderedDict()

        # In-flight reads: key → Future[bytes].  This is the coalescing
        # dictionary that guarantees one read per concurrent request.
        self._in_flight: dict[tuple[int, int], asyncio.Future[bytes]] = {}

        # Track which in-flight reads were initiated by prefetch so we can
        # credit prefetch_hits when a get() coalesces onto them.
        self._in_flight_from_prefetch: set[tuple[int, int]] = set()

        # Track which keys were accessed via coalescing (rather than cache
        # hit) so _insert() can mark the resulting entry as accessed.
        self._coalesced_from_prefetch: set[tuple[int, int]] = set()

        # ── Instrumentation ────────────────────────────────────────────
        self.stats = CacheStats()

        # ── Background task tracking ───────────────────────────────────
        self._background_tasks: set[asyncio.Task[Any]] = set()

    # ── Public API ─────────────────────────────────────────────────────────

    async def get(self, layer: int, expert: int) -> bytes:
        """Return the expert's weight bytes, reading through if necessary.

        Cache hit → return immediately, update LRU, increment hit counters.
        Cache miss → check in-flight coalescing → if coalesced, await the
        existing read; otherwise read from store, cache, and return.

        Concurrent callers for the same uncached expert share one read.
        """
        key = (layer, expert)

        # ── 1. Cache hit ───────────────────────────────────────────────
        entry = self._cache.get(key)
        if entry is not None:
            entry.last_access = time.monotonic()
            if not entry.accessed and entry.from_prefetch:
                # This was a prefetched entry being accessed for the first
                # time — mark it as used.
                entry.accessed = True
                self.stats.prefetch_hits += 1
            self._touch_lru(key)
            self.stats.hits += 1
            self.stats.bytes_served += entry.size
            return entry.data

        # ── 2. In-flight coalescing ────────────────────────────────────
        inflight = self._in_flight.get(key)
        if inflight is not None:
            self.stats.coalesced += 1
            # If the in-flight read was initiated by prefetch, credit it
            # and mark for the upcoming _insert() to set accessed=True.
            if key in self._in_flight_from_prefetch:
                self.stats.prefetch_hits += 1
                self._coalesced_from_prefetch.add(key)
            data = await inflight
            # The first caller already cached this and updated stats;
            # we just need to count our bytes_served.
            self.stats.bytes_served += len(data)
            return data

        # ── 3. Miss — issue a read ─────────────────────────────────────
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bytes] = loop.create_future()
        self._in_flight[key] = future

        try:
            data = await self._store.read_expert(layer, expert)
            self.stats.misses += 1
            self.stats.bytes_read += len(data)

            # Cache the result.
            self._insert(key, data, from_prefetch=False)

            future.set_result(data)
            self.stats.bytes_served += len(data)
            return data
        except Exception:
            future.cancel()
            raise
        finally:
            self._in_flight.pop(key, None)
            self._in_flight_from_prefetch.discard(key)

    async def prefetch(self, layer: int, experts: list[int]) -> None:
        """Issue speculative reads for *experts* in *layer*.

        Non-blocking: spawns background tasks bounded by
        ``max_concurrent_prefetch``.  A wrong prediction wastes bandwidth
        but never correctness — unneeded prefetched experts are evicted
        normally.

        Prefetched entries are marked ``from_prefetch=True`` so the
        instrumentation can track accuracy.
        """
        async def _fetch_one(expert: int) -> None:
            key = (layer, expert)

            # Already cached — nothing to do.
            if key in self._cache:
                return

            # Already in-flight (from a concurrent prefetch or get).
            if key in self._in_flight:
                # Await the existing read so we know it completed, but
                # don't return the data — the caller didn't ask for it.
                try:
                    await self._in_flight[key]
                except Exception:
                    pass
                return

            async with self._prefetch_sem:
                # Re-check after acquiring the semaphore — a concurrent
                # prefetch or get may have beaten us.
                if key in self._cache or key in self._in_flight:
                    return

                loop = asyncio.get_running_loop()
                future: asyncio.Future[bytes] = loop.create_future()
                self._in_flight[key] = future
                self._in_flight_from_prefetch.add(key)
                self.stats.prefetch_issued += 1

                try:
                    data = await self._store.read_expert(layer, expert)
                    self.stats.bytes_read += len(data)

                    # Insert with from_prefetch=True.
                    self._insert(key, data, from_prefetch=True)

                    future.set_result(data)
                except Exception:
                    future.cancel()
                    raise
                finally:
                    self._in_flight.pop(key, None)
                    self._in_flight_from_prefetch.discard(key)

        for expert in experts:
            task = asyncio.create_task(_fetch_one(expert))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    def pin(self, layer: int, expert: int) -> None:
        """Mark an expert as pinned — never evicted.

        Must already be in the cache (call ``get()`` first).  Pinning an
        uncached expert raises ``KeyError``.
        """
        key = (layer, expert)
        entry = self._cache.get(key)
        if entry is None:
            raise KeyError(
                f"Cannot pin uncached expert (layer={layer}, expert={expert})"
            )
        if entry.pinned:
            return
        entry.pinned = True
        # Remove from LRU tracking — pinned entries are never eviction candidates.
        self._lru.pop(key, None)

    def unpin(self, layer: int, expert: int) -> None:
        """Remove the pinned flag from an expert, making it evictable.

        Does nothing if the expert was not pinned.
        """
        key = (layer, expert)
        entry = self._cache.get(key)
        if entry is None or not entry.pinned:
            return
        entry.pinned = False
        self._lru[key] = True  # re-enter LRU at the MRU end

    async def close(self) -> None:
        """Cancel in-flight background tasks and release the store."""
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._background_tasks.clear()
        self._in_flight.clear()
        self._in_flight_from_prefetch.clear()
        self._coalesced_from_prefetch.clear()
        self._cache.clear()
        self._lru.clear()
        await self._store.close()

    # ── Internal ──────────────────────────────────────────────────────────

    def _insert(
        self, key: tuple[int, int], data: bytes, *, from_prefetch: bool
    ) -> None:
        """Insert *data* for *key* into the cache, evicting if necessary.

        If the entry is already present (should not happen — callers check),
        it is updated in place.

        Pinned entries are never evicted.  If the budget is too small to
        accommodate all pinned entries plus the new entry, the new entry is
        still inserted (pinned entries are a hard requirement and the budget
        is a soft target in that case), but a warning is logged.
        """
        size = len(data)

        # ── Evict to make room ─────────────────────────────────────────
        needed = self.stats.current_bytes + size
        while needed > self._budget and self._lru:
            # Pick the LRU (leftmost non-pinned entry).
            victim_key, _ = self._lru.popitem(last=False)
            victim = self._cache.pop(victim_key, None)
            if victim is None:
                continue

            self.stats.current_bytes -= victim.size
            self.stats.current_entries -= 1
            self.stats.evictions += 1

            # Track prefetch waste: entries that were prefetched but never
            # accessed before eviction.
            if victim.from_prefetch and not victim.accessed:
                self.stats.prefetch_waste += 1

            needed -= victim.size

        if needed > self._budget:
            logger.warning(
                "Cache budget exceeded by pinned entries: %d bytes needed, "
                "budget is %d bytes.  Inserting anyway — pinned entries are "
                "a hard requirement.",
                needed,
                self._budget,
            )

        # ── Insert ─────────────────────────────────────────────────────
        now = time.monotonic()
        # If this prefetched entry was already accessed via coalescing,
        # mark it as accessed so the eviction path doesn't count it as waste.
        was_coalesced = from_prefetch and key in self._coalesced_from_prefetch
        if was_coalesced:
            self._coalesced_from_prefetch.discard(key)
        entry = CacheEntry(
            data=data,
            size=size,
            last_access=now,
            from_prefetch=from_prefetch,
            accessed=was_coalesced,
            pinned=False,
        )
        self._cache[key] = entry
        self._lru[key] = True  # MRU end
        self.stats.current_bytes += size
        self.stats.current_entries += 1

        if self.stats.current_bytes > self.stats.peak_bytes:
            self.stats.peak_bytes = self.stats.current_bytes
        if self.stats.current_entries > self.stats.peak_entries:
            self.stats.peak_entries = self.stats.current_entries

    def _touch_lru(self, key: tuple[int, int]) -> None:
        """Move *key* to the MRU end of the LRU order.

        No-op for pinned entries (they are not in the LRU at all).
        """
        if key in self._lru:
            self._lru.move_to_end(key, last=True)


# ── Prefetch predictor ─────────────────────────────────────────────────────────


class DummyPrefetchPredictor:
    """Stub prefetch predictor — predicts nothing.

    This is the default when no real predictor is wired in.  It exists so
    the caller can always call ``predictor.predict()`` without a None check,
    and so the prefetch infrastructure is exercised even without a working
    model.

    Replace with a real predictor (hidden-state → expert projection) once
    Layer 1 is built.
    """

    async def predict(self, layer: int, cache: ExpertCache) -> None:
        """No-op — predict nothing."""
        pass


class OraclePrefetchPredictor:
    """Prefetch predictor that knows the future — for testing.

    Given a complete routing trace (layer → list of expert sets), issues
    prefetches for layer N+1 while layer N is "computing."  This produces
    ideal prefetch accuracy and lets the cache layer's instrumentation be
    tested against a known ceiling.

    Parameters
    ----------
    trace:
        Routing trace as a dict mapping layer index (int) to a list of
        expert-index lists, one per token position.
        The same format as ``seed_*/expert_indices.json`` → ``expert_trace``.
    advance_callback:
        Optional async callback called before issuing prefetch for each
        token position.  Receives ``(position, layer)``.  Use this to
        simulate layer compute time by sleeping.
    """

    def __init__(
        self,
        trace: dict[int, list[list[int]]],
        advance_callback: Callable[[int, int], Any] | None = None,
    ) -> None:
        self._trace = trace
        self._advance_callback = advance_callback
        self._position: int = 0
        self._max_position: int = min(
            len(v) for v in trace.values()
        ) if trace else 0

    def reset(self) -> None:
        """Reset position to 0 for a new replay."""
        self._position = 0

    async def predict(self, current_layer: int, cache: ExpertCache) -> None:
        """Prefetch experts for the next layer at the current position.

        The predictor tracks position independently — callers should call
        this once per layer during token generation.  Position advances
        only when *current_layer* is the last layer (NUM_LAYERS-1).

        If the predictor has run past the end of the trace, this is a no-op.
        """
        if self._position >= self._max_position:
            return

        if self._advance_callback is not None:
            result = self._advance_callback(self._position, current_layer)
            if asyncio.iscoroutine(result):
                await result

        next_layer = current_layer + 1
        if next_layer not in self._trace:
            # Past the last layer — advance to next token position.
            self._position += 1
            return

        next_experts = self._trace[next_layer][self._position]
        await cache.prefetch(next_layer, next_experts)


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════


async def _demo_synthetic() -> None:
    """Demo: synthetic 64-expert, 16-layer model streamed through the cache.

    Uses a SimulatedExpertStore to model fast NVMe (~3.2 GB/s), a 1 GB
    cache budget, and a synthetic routing trace that repeats a small set
    of "hot" experts interspersed with cold ones.

    Shows:
      - Hit rate climbing as the hot set warms up
      - Pinned shared expert surviving eviction
      - Prefetch accuracy with an oracle predictor
      - Instrumentation dump
    """
    import random as _random

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from storage_io import SimulatedExpertStore

    NUM_LAYERS = 16
    NUM_EXPERTS = 64
    TOP_K = 8
    EXPERT_SIZE_KB = 64  # 64 KB per expert for demo speed (real: 48 MB)
    EXPERT_SIZE = EXPERT_SIZE_KB * 1024
    TOKENS_TO_GENERATE = 20
    CACHE_BUDGET_KB = 2048  # 2 MB cache (fits 16 pinned + ~16 others)
    CACHE_BUDGET = CACHE_BUDGET_KB * 1024

    # Use a fast simulated bandwidth so the demo runs quickly.
    # The point is demonstrating cache behavior, not measuring throughput.
    BW_MBPS = 500.0

    # ── Build a synthetic routing trace ────────────────────────────────
    # Hot set: 12 experts that appear 80% of the time (models locality).
    # Cold set: the remaining 52 experts.
    # Expert 0 is always included (simulates the shared expert).
    _random.seed(42)
    hot_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    cold_set = list(range(12, NUM_EXPERTS))

    trace: dict[int, list[list[int]]] = {}
    for layer in range(NUM_LAYERS):
        layer_trace: list[list[int]] = []
        for _ in range(TOKENS_TO_GENERATE):
            experts: list[int] = [0]  # shared expert always present
            # 80% chance: fill remaining 7 from hot set
            if _random.random() < 0.80:
                pool = hot_set[1:]  # exclude 0, already added
            else:
                pool = cold_set
            picks = _random.sample(pool, min(TOP_K - 1, len(pool)))
            experts.extend(picks)
            # Pad to TOP_K if needed.
            while len(experts) < TOP_K:
                extra = _random.choice(cold_set)
                if extra not in experts:
                    experts.append(extra)
            layer_trace.append(experts[:TOP_K])
        trace[layer] = layer_trace

    # ── Create store and cache ─────────────────────────────────────────
    store = SimulatedExpertStore(
        expert_size_bytes=EXPERT_SIZE,
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        bandwidth_mbps=BW_MBPS,
        latency_ms=0.1,
        seed=42,
    )
    cache = ExpertCache(
        store=store,
        memory_budget_bytes=CACHE_BUDGET,
    )

    # ── Pin the shared expert (dials.md dial 14) ───────────────────────
    # Must get() it first to load it into the cache.
    print("=" * 72)
    print("  Swarm Expert Cache — synthetic demo")
    print("=" * 72)
    print(f"  Model:     {NUM_LAYERS} layers × {NUM_EXPERTS} experts")
    print(f"  Expert:    {EXPERT_SIZE_KB} KB each")
    print(f"  Budget:    {CACHE_BUDGET_KB} KB (~{CACHE_BUDGET // EXPERT_SIZE} experts)")
    print(f"  Tokens:    {TOKENS_TO_GENERATE}")
    print(f"  Hot set:   {len(hot_set)} experts, 80% probability")
    print()

    # Pin the shared expert (expert 0) across all layers.
    print("── Pinning shared expert (expert 0) across all layers ──")
    for layer in range(NUM_LAYERS):
        await cache.get(layer, 0)
        cache.pin(layer, 0)
    print(f"  Pinned {NUM_LAYERS} shared experts "
          f"({NUM_LAYERS * EXPERT_SIZE_KB} KB total)")
    print(f"  Remaining budget: ~{(CACHE_BUDGET - NUM_LAYERS * EXPERT_SIZE) // EXPERT_SIZE} experts")
    print()

    # ── Stream tokens through the cache ────────────────────────────────
    print("── Streaming tokens (no prefetch) ──")
    t0 = time.monotonic()

    for position in range(TOKENS_TO_GENERATE):
        for layer in range(NUM_LAYERS):
            layer_experts = trace[layer][position]
            for expert in layer_experts:
                await cache.get(layer, expert)

        if position % 10 == 0 or position == TOKENS_TO_GENERATE - 1:
            s = cache.stats
            print(
                f"  pos {position:>4}: {s.summary()}"
            )

    elapsed = time.monotonic() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Final: {cache.stats.summary()}")
    print()

    # ── Reset and run with oracle prefetch ─────────────────────────────
    print("── Streaming tokens (WITH oracle prefetch) ──")
    await cache.close()

    store2 = SimulatedExpertStore(
        expert_size_bytes=EXPERT_SIZE,
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        bandwidth_mbps=BW_MBPS,
        latency_ms=0.1,
        seed=42,
    )
    cache2 = ExpertCache(
        store=store2,
        memory_budget_bytes=CACHE_BUDGET,
    )

    # Pin shared experts again.
    for layer in range(NUM_LAYERS):
        await cache2.get(layer, 0)
        cache2.pin(layer, 0)

    predictor = OraclePrefetchPredictor(trace=trace)

    t0 = time.monotonic()

    for position in range(TOKENS_TO_GENERATE):
        for layer in range(NUM_LAYERS):
            layer_experts = trace[layer][position]
            for expert in layer_experts:
                await cache2.get(layer, expert)

            # Prefetch next layer while current layer is "computing".
            await predictor.predict(layer, cache2)

        if position % 10 == 0 or position == TOKENS_TO_GENERATE - 1:
            s = cache2.stats
            print(
                f"  pos {position:>4}: {s.summary()}"
            )

    elapsed2 = time.monotonic() - t0
    print(f"\n  Total time: {elapsed2:.1f}s")
    print(f"  Final: {cache2.stats.summary()}")
    print()

    # ── Comparison ─────────────────────────────────────────────────────
    print("── Comparison ──")
    print(f"  Without prefetch: hit_rate={cache.stats.hit_rate:.1%}, "
          f"bytes_read={cache.stats.bytes_read/(1024*1024):.0f} MB")
    print(f"  With prefetch:    hit_rate={cache2.stats.hit_rate:.1%}, "
          f"bytes_read={cache2.stats.bytes_read/(1024*1024):.0f} MB, "
          f"prefetch_acc={cache2.stats.prefetch_accuracy:.1%}")
    print()

    await cache.close()
    await cache2.close()

    print("── Demo complete ──")


async def _demo_real_trace() -> None:
    """Demo: replay a real OLMoE routing trace through the cache.

    Uses seed_1/expert_indices.json if available.  Falls back to the
    synthetic demo if the file is missing.
    """
    import os as _os

    trace_path = "seed_1/expert_indices.json"
    if not _os.path.exists(trace_path):
        print(
            f"Real trace file '{trace_path}' not found — "
            f"falling back to synthetic demo."
        )
        await _demo_synthetic()
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from storage_io import SimulatedExpertStore

    with open(trace_path) as fh:
        raw = json.load(fh)

    num_layers = raw["num_layers"]
    num_experts = raw["num_experts"]
    top_k = raw["top_k"]
    num_tokens = min(raw.get("num_generated_tokens", len(raw["generated_ids"])), 100)

    # expert_trace: dict[str, list[list[int]]] — string keys.
    raw_trace: dict[str, list[list[int]]] = raw["expert_trace"]
    trace: dict[int, list[list[int]]] = {
        int(k): v for k, v in raw_trace.items()
    }

    # Use a small expert size for the demo so it runs quickly.
    EXPERT_SIZE = 64 * 1024  # 64 KB per expert for demo speed
    CACHE_BUDGET = 10 * 1024 * 1024  # 10 MB cache (~160 experts of 64 KB)

    store = SimulatedExpertStore(
        expert_size_bytes=EXPERT_SIZE,
        num_layers=num_layers,
        num_experts=num_experts,
        bandwidth_mbps=2000.0,  # fast for demo
        latency_ms=0.05,
        seed=42,
    )
    cache = ExpertCache(
        store=store,
        memory_budget_bytes=CACHE_BUDGET,
    )

    print("=" * 72)
    print("  Swarm Expert Cache — real OLMoE routing trace")
    print("=" * 72)
    print(f"  Trace:     {trace_path}")
    print(f"  Model:     {raw['model']}")
    print(f"  Layers:    {num_layers}")
    print(f"  Experts:   {num_experts} (top-{top_k})")
    print(f"  Tokens:    {num_tokens}")
    print(f"  Budget:    {CACHE_BUDGET / (1024*1024):.0f} MB")
    print()

    # Pin expert 0 (shared expert proxy) across layers.
    for layer in range(num_layers):
        await cache.get(layer, 0)
        cache.pin(layer, 0)

    predictor = OraclePrefetchPredictor(trace=trace)

    print("── Replaying trace ──")
    t0 = time.monotonic()

    for position in range(num_tokens):
        for layer in range(num_layers):
            if position < len(trace.get(layer, [])):
                experts = trace[layer][position]
                for expert in experts:
                    await cache.get(layer, expert)

            await predictor.predict(layer, cache)

        if position % 100 == 0 or position == num_tokens - 1:
            s = cache.stats
            print(f"  token {position:>4}: {s.summary()}")

    elapsed = time.monotonic() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Final: {cache.stats.summary()}")
    print(f"  Peak: {cache.stats.peak_entries} entries, "
          f"{cache.stats.peak_bytes / (1024*1024):.1f} MB")
    print()

    # ── Stats as JSON ──────────────────────────────────────────────────
    print("── Instrumentation (JSON) ──")
    print(json.dumps(cache.stats.to_dict(), indent=2))
    print()

    await cache.close()
    print("── Demo complete ──")


async def _main() -> None:
    """Run both demos."""
    import sys as _sys

    if "--real" in _sys.argv:
        await _demo_real_trace()
    else:
        await _demo_synthetic()


if __name__ == "__main__":
    asyncio.run(_main())
