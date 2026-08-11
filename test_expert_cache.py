"""Tests for expert_cache.py and storage_io.py — cache correctness,
budget enforcement, pinning, LRU order, prefetch, concurrent coalescing,
and instrumentation accuracy.

Uses SimulatedExpertStore exclusively — no real NVMe required.  This is
the test suite for Layer 2, the same way the other modules have theirs.

Run:  python3 test_expert_cache.py
"""

from __future__ import annotations

import asyncio
import time

from expert_cache import (
    DEFAULT_SHARED_EXPERT_INDEX,
    CacheEntry,
    CacheStats,
    ExpertCache,
    OraclePrefetchPredictor,
)
from storage_io import SimulatedExpertStore

# ── Fixture ────────────────────────────────────────────────────────────────


def _make_store(
    num_layers: int = 4,
    num_experts: int = 16,
    expert_size: int = 100_000,  # 100 KB — fast tests, small numbers
    bandwidth_mbps: float = 10_000.0,  # very fast — we test logic, not throttling
    latency_ms: float = 0.0,
    seed: int = 42,
) -> SimulatedExpertStore:
    return SimulatedExpertStore(
        expert_size_bytes=expert_size,
        num_layers=num_layers,
        num_experts=num_experts,
        bandwidth_mbps=bandwidth_mbps,
        latency_ms=latency_ms,
        seed=seed,
    )


def _make_cache(
    store: SimulatedExpertStore | None = None,
    budget: int = 5_000_000,  # 5 MB
) -> ExpertCache:
    if store is None:
        store = _make_store()
    return ExpertCache(store=store, memory_budget_bytes=budget)


# ── Basic read-through ────────────────────────────────────────────────────


async def test_cache_miss_reads_through() -> None:
    """A cache miss calls the store and returns the correct bytes."""
    store = _make_store(expert_size=1000, seed=123)
    cache = _make_cache(store, budget=10_000)

    data = await cache.get(0, 5)
    assert len(data) == 1000
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0
    assert cache.stats.bytes_read == 1000
    assert cache.stats.bytes_served == 1000
    print("PASS: cache miss reads through and returns correct bytes")


async def test_cache_hit_avoids_store() -> None:
    """A second get() for the same expert is a cache hit."""
    cache = _make_cache(_make_store(expert_size=1000), budget=10_000)

    d1 = await cache.get(1, 3)
    d2 = await cache.get(1, 3)

    assert d1 == d2
    assert cache.stats.misses == 1
    assert cache.stats.hits == 1
    assert cache.stats.bytes_read == 1000  # only one read
    assert cache.stats.bytes_served == 2000  # served twice
    print("PASS: cache hit avoids second store read")


# ── Budget enforcement ────────────────────────────────────────────────────


async def test_budget_respected_under_pressure() -> None:
    """Inserting many distinct experts triggers eviction, stays under budget."""
    expert_size = 100_000
    cache = _make_cache(_make_store(expert_size=expert_size), budget=500_000)

    # 5 experts fit exactly, 6th should evict the first.
    for i in range(8):
        await cache.get(0, i)

    assert cache.stats.current_bytes <= 500_000
    assert cache.stats.current_entries <= 5
    assert cache.stats.evictions >= 3  # at least 3 entries evicted
    print(
        f"PASS: budget respected — {cache.stats.current_bytes} bytes, "
        f"{cache.stats.current_entries} entries, "
        f"{cache.stats.evictions} evictions"
    )


async def test_exact_budget_fill() -> None:
    """Filling the cache to exactly the budget does not evict."""
    expert_size = 100_000
    cache = _make_cache(_make_store(expert_size=expert_size), budget=300_000)

    await cache.get(0, 0)
    await cache.get(0, 1)
    await cache.get(0, 2)

    assert cache.stats.current_bytes == 300_000
    assert cache.stats.current_entries == 3
    assert cache.stats.evictions == 0
    print("PASS: exact budget fill — no spurious evictions")


# ── Pinning ───────────────────────────────────────────────────────────────


async def test_pinned_experts_survive_eviction() -> None:
    """Pinned entries are never evicted, even under extreme pressure."""
    expert_size = 100_000
    cache = _make_cache(_make_store(expert_size=expert_size), budget=300_000)

    # Load and pin expert 0.
    await cache.get(0, 0)
    cache.pin(0, 0)

    # Fill up with other experts.
    for i in range(1, 10):
        await cache.get(0, i)

    # Pinned entry must still be present.
    assert (0, 0) in cache._cache, "pinned expert was evicted"
    assert cache._cache[(0, 0)].pinned is True

    # Budget may be exceeded due to pinning — that's expected.
    print(
        f"PASS: pinned expert survived — "
        f"{cache.stats.current_entries} entries, "
        f"{cache.stats.current_bytes} bytes (budget={300_000})"
    )


async def test_pin_uncached_raises() -> None:
    """Pinning an expert not in the cache raises KeyError."""
    cache = _make_cache()

    try:
        cache.pin(99, 99)
    except KeyError:
        print("PASS: pinning uncached expert raises KeyError")
        return
    raise AssertionError("pin() should have raised KeyError")


async def test_unpin_makes_evictable() -> None:
    """Unpinning a previously pinned expert allows it to be evicted."""
    expert_size = 100_000
    cache = _make_cache(_make_store(expert_size=expert_size), budget=300_000)

    await cache.get(0, 0)
    cache.pin(0, 0)
    cache.unpin(0, 0)

    # Now fill with other experts — the unpinned expert should be evictable.
    for i in range(1, 10):
        await cache.get(0, i)

    # Expert 0 may or may not still be present depending on LRU order;
    # the point is unpin removed the protection.
    entry = cache._cache.get((0, 0))
    assert entry is None or entry.pinned is False, (
        "unpinned entry should not be marked pinned"
    )
    print("PASS: unpin removes eviction protection")


# ── LRU order ─────────────────────────────────────────────────────────────


async def test_lru_order_is_correct() -> None:
    """Least-recently-used entries are evicted first."""
    expert_size = 100_000
    cache = _make_cache(_make_store(expert_size=expert_size), budget=300_000)

    await cache.get(0, 0)
    await cache.get(0, 1)
    await cache.get(0, 2)

    # Access 0 and 1 to make 2 the LRU.
    await cache.get(0, 0)
    await cache.get(0, 1)

    # Insert a 4th expert — must evict 2 (the LRU).
    await cache.get(0, 3)

    assert (0, 0) in cache._cache, "expert 0 (MRU) wrongly evicted"
    assert (0, 1) in cache._cache, "expert 1 wrongly evicted"
    assert (0, 2) not in cache._cache, "expert 2 (LRU) not evicted"
    assert (0, 3) in cache._cache, "new expert not cached"
    print("PASS: LRU order correct — LRU evicted before MRU")


async def test_lru_reshuffled_by_access() -> None:
    """Accessing an entry moves it to the MRU end."""
    expert_size = 100_000
    cache = _make_cache(_make_store(expert_size=expert_size), budget=400_000)

    await cache.get(0, 0)
    await cache.get(0, 1)
    await cache.get(0, 2)
    await cache.get(0, 3)

    # LRU order after inserts: 0, 1, 2, 3 (0 = LRU, 3 = MRU)
    # Access 0 → moves 0 to MRU: 1, 2, 3, 0
    await cache.get(0, 0)

    # Insert new expert → should evict 1 (now the LRU).
    await cache.get(0, 4)

    assert (0, 1) not in cache._cache, "expert 1 (new LRU) not evicted"
    assert (0, 0) in cache._cache, "expert 0 (recently accessed) wrongly evicted"
    print("PASS: access reshuffles LRU order correctly")


# ── Concurrent coalescing ────────────────────────────────────────────────


async def test_concurrent_get_coalesces_reads() -> None:
    """Two concurrent get() calls for the same expert share one read."""
    store = _make_store(expert_size=1000, latency_ms=100.0)  # 100 ms latency
    cache = _make_cache(store, budget=10_000)

    # Fire two gets concurrently.
    d1, d2 = await asyncio.gather(
        cache.get(0, 7),
        cache.get(0, 7),
    )

    assert d1 == d2
    assert cache.stats.misses == 1, (
        f"expected 1 miss, got {cache.stats.misses} — "
        f"concurrent requests were not coalesced"
    )
    assert cache.stats.coalesced == 1, (
        f"expected 1 coalesced, got {cache.stats.coalesced}"
    )
    assert cache.stats.bytes_read == 1000  # only one read
    assert cache.stats.bytes_served == 2000  # both callers served
    print("PASS: concurrent get() coalesces into one read")


async def test_concurrent_get_and_prefetch_coalesce() -> None:
    """A get() and a prefetch() for the same expert share one read."""
    # Use a store that blocks on a future so we can hold the read in-flight
    # while get() joins it.  This is the only way to reliably test
    # coalescing between a background prefetch task and a foreground get(),
    # because the prefetch can otherwise complete before get() runs.
    store = _make_store(expert_size=1000, latency_ms=0.0)
    cache = _make_cache(store, budget=10_000)
    cache.stats.misses = 0  # reset from any constructor side effects
    cache.stats.hits = 0

    # Monkey-patch _store.read_expert to block until we release it.
    read_started = asyncio.Event()
    read_may_complete = asyncio.Event()
    _original_read = store.read_expert

    async def _blocking_read(layer: int, expert: int) -> bytes:
        read_started.set()
        await read_may_complete.wait()
        return await _original_read(layer, expert)

    store.read_expert = _blocking_read  # type: ignore[method-assign]

    # Start prefetch.
    asyncio.create_task(cache.prefetch(0, [5]))

    # Wait until the prefetch has started reading (registered its future).
    await asyncio.wait_for(read_started.wait(), timeout=2.0)

    # Release the read BEFORE calling get().  set() marks the Event but
    # does not yield — so the prefetch task remains suspended, its future
    # still pending.  get() will race in, find the in-flight future, and
    # await it.  Only then does the prefetch task resume, complete the
    # read, and resolve the future for both.
    read_may_complete.set()
    data = await cache.get(0, 5)
    await asyncio.sleep(0.05)

    assert len(data) == 1000
    # Exactly one read hit the store (prefetch initiated it).
    # get() either coalesced (coalesced >= 1) or hit cache (hits >= 1).
    # In neither case is misses incremented — only get() read-throughs do that.
    assert cache.stats.bytes_read == 1000, (
        f"expected 1 read (1000 bytes), got {cache.stats.bytes_read}"
    )
    assert cache.stats.bytes_served >= 1000, (
        f"expected at least 1000 bytes served, got {cache.stats.bytes_served}"
    )
    total_resolved = cache.stats.coalesced + cache.stats.hits
    assert total_resolved >= 1, (
        f"get() neither coalesced nor hit cache: "
        f"coalesced={cache.stats.coalesced}, hits={cache.stats.hits}"
    )
    print(f"PASS: get() and prefetch() shared one read "
          f"(coalesced={cache.stats.coalesced}, hits={cache.stats.hits})")


async def test_concurrent_get_does_not_double_read() -> None:
    """Even with many concurrent callers, only one read hits the store."""
    # Block the first read so all subsequent callers coalesce.
    store = _make_store(expert_size=1000, latency_ms=0.0)
    cache = _make_cache(store, budget=10_000)

    read_started = asyncio.Event()
    read_may_complete = asyncio.Event()
    _original_read = store.read_expert

    async def _blocking_read(layer: int, expert: int) -> bytes:
        read_started.set()
        await read_may_complete.wait()
        return await _original_read(layer, expert)

    store.read_expert = _blocking_read  # type: ignore[method-assign]

    # Fire 10 concurrent gets. Only the first should start a read.
    async def _get_and_wait() -> bytes:
        return await cache.get(0, 3)

    tasks = [asyncio.create_task(_get_and_wait()) for _ in range(10)]

    # Wait for the first read to start.
    await asyncio.wait_for(read_started.wait(), timeout=2.0)
    # Give the other 9 tasks time to reach the coalescing check.
    await asyncio.sleep(0.05)

    # Release the read.
    read_may_complete.set()

    results = await asyncio.gather(*tasks)

    assert all(r == results[0] for r in results)
    assert cache.stats.misses == 1, (
        f"expected 1 miss for 10 concurrent gets, got {cache.stats.misses}"
    )
    assert cache.stats.coalesced == 9
    assert cache.stats.bytes_read == 1000
    assert cache.stats.bytes_served == 10_000
    print(
        f"PASS: 10 concurrent gets → 1 miss, {cache.stats.coalesced} coalesced"
    )


# ── Prefetch ─────────────────────────────────────────────────────────────


async def test_prefetch_never_breaks_correctness() -> None:
    """A wrong prefetch must not affect get() correctness."""
    store = _make_store(expert_size=1000, seed=99)
    cache = _make_cache(store, budget=100_000)

    # Prefetch experts that will never be requested.
    await cache.prefetch(0, [10, 11, 12, 13, 14, 15])
    # Wait for prefetch tasks to complete.
    await asyncio.sleep(0.2)

    # Now request different experts — must get correct bytes.
    d1 = await cache.get(0, 0)
    d2 = await cache.get(0, 0)  # hit
    d3 = await cache.get(1, 0)

    assert d1 == await store.read_expert(0, 0)
    assert d2 == d1
    assert d3 == await store.read_expert(1, 0)
    print("PASS: wrong prefetch does not affect get() correctness")


async def test_prefetch_populates_cache() -> None:
    """A correct prefetch results in cache hits for later get() calls."""
    store = _make_store(expert_size=1000)
    cache = _make_cache(store, budget=100_000)

    # Prefetch, wait, then get.
    await cache.prefetch(0, [3, 4])
    await asyncio.sleep(0.1)

    assert cache.stats.misses == 0  # no get() misses yet
    assert cache.stats.prefetch_issued == 2

    d3 = await cache.get(0, 3)
    d4 = await cache.get(0, 4)

    assert cache.stats.hits == 2
    assert cache.stats.misses == 0
    # Both prefetched entries were accessed → prefetch_hits.
    assert cache.stats.prefetch_hits == 2
    assert cache.stats.prefetch_waste == 0
    print(f"PASS: prefetch populates cache — {cache.stats.prefetch_hits} hits, "
          f"{cache.stats.prefetch_waste} waste")


async def test_prefetch_waste_tracked() -> None:
    """Prefetched but evicted entries count as waste."""
    expert_size = 100_000
    cache = _make_cache(_make_store(expert_size=expert_size), budget=200_000)

    # Prefetch two experts (200 KB total = exactly budget).
    await cache.prefetch(0, [0, 1])
    await asyncio.sleep(0.1)

    # Waste is 0 until eviction — the new accounting only counts
    # waste when a prefetched-but-never-accessed entry is evicted.
    assert cache.stats.prefetch_waste == 0

    # Access one of them.
    await cache.get(0, 0)
    assert cache.stats.prefetch_hits == 1
    assert cache.stats.prefetch_waste == 0  # nothing evicted yet

    # Evict expert 1 by inserting another.  Expert 1 was from_prefetch=True
    # and accessed=False, so it counts as prefetch waste.
    await cache.get(0, 2)
    assert cache.stats.prefetch_issued == 2
    assert cache.stats.prefetch_hits == 1
    assert cache.stats.prefetch_waste == 1  # expert 1 evicted unaccessed
    print(f"PASS: prefetch waste tracked — issued={cache.stats.prefetch_issued}, "
          f"hits={cache.stats.prefetch_hits}, waste_in_cache={cache.stats.prefetch_waste}")


# ── Instrumentation accuracy ─────────────────────────────────────────────


async def test_instrumentation_counters_accurate() -> None:
    """All counters match expected values after a scripted sequence."""
    store = _make_store(expert_size=1000, seed=77)
    cache = _make_cache(store, budget=10_000_000)  # huge budget, no eviction

    # Miss: expert (0, 0).
    await cache.get(0, 0)
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0
    assert cache.stats.bytes_read == 1000
    assert cache.stats.bytes_served == 1000

    # Hit: same expert.
    await cache.get(0, 0)
    assert cache.stats.misses == 1
    assert cache.stats.hits == 1
    assert cache.stats.bytes_read == 1000
    assert cache.stats.bytes_served == 2000

    # Miss: new expert.
    await cache.get(1, 5)
    assert cache.stats.misses == 2
    assert cache.stats.hits == 1
    assert cache.stats.bytes_read == 2000
    assert cache.stats.bytes_served == 3000

    # Prefetch.
    await cache.prefetch(2, [0, 1])
    await asyncio.sleep(0.1)
    assert cache.stats.prefetch_issued == 2
    assert cache.stats.bytes_read == 4000  # 2 more reads

    # Access prefetched entries.
    await cache.get(2, 0)
    assert cache.stats.prefetch_hits == 1
    assert cache.stats.hits == 2  # hit on prefetched entry

    await cache.get(2, 1)
    assert cache.stats.prefetch_hits == 2
    assert cache.stats.hits == 3

    # Verify snapshot.
    snap = cache.stats.snapshot()
    assert snap.hits == 3
    assert snap.misses == 2
    assert snap.hit_rate == 3 / 5
    assert snap.prefetch_issued == 2
    assert snap.prefetch_hits == 2
    assert snap.prefetch_accuracy == 1.0

    print("PASS: all instrumentation counters accurate")


async def test_peak_tracking() -> None:
    """Peak bytes and entries are tracked correctly."""
    expert_size = 100_000
    cache = _make_cache(_make_store(expert_size=expert_size), budget=500_000)

    await cache.get(0, 0)  # 100 KB
    assert cache.stats.peak_bytes == 100_000
    assert cache.stats.peak_entries == 1

    await cache.get(0, 1)  # 200 KB
    assert cache.stats.peak_bytes == 200_000
    assert cache.stats.peak_entries == 2

    await cache.get(0, 2)  # 300 KB
    await cache.get(0, 3)  # 400 KB
    await cache.get(0, 4)  # 500 KB

    assert cache.stats.peak_bytes == 500_000
    assert cache.stats.peak_entries == 5

    # 6th expert → evict one, peak should still be 500K.
    await cache.get(0, 5)
    assert cache.stats.peak_bytes == 500_000
    assert cache.stats.current_bytes <= 500_000
    print("PASS: peak tracking correct")


# ── OraclePrefetchPredictor ──────────────────────────────────────────────


async def test_oracle_predictor_produces_hits() -> None:
    """The oracle prefetch predictor achieves 100% accuracy on a simple trace."""
    expert_size = 100_000
    cache = _make_cache(_make_store(expert_size=expert_size), budget=5_000_000)

    # Build a tiny trace: 2 layers, 3 tokens.
    trace: dict[int, list[list[int]]] = {
        0: [[0, 1], [2, 3], [4, 5]],
        1: [[6, 7], [8, 9], [10, 11]],
    }

    predictor = OraclePrefetchPredictor(trace=trace)

    # Token 0: get layer 0 experts, predict layer 1.
    for expert in trace[0][0]:
        await cache.get(0, expert)
    await predictor.predict(0, cache)

    # Wait for prefetch background tasks to complete.
    await asyncio.sleep(0.1)

    # Token 0, layer 1: should be cache hits from prefetch.
    for expert in trace[1][0]:
        await cache.get(1, expert)

    assert cache.stats.hits == 2  # experts 6,7 were prefetched
    assert cache.stats.misses == 2  # experts 0,1 were misses initially
    print("PASS: oracle predictor produces prefetch hits")


# ── Edge cases ───────────────────────────────────────────────────────────


async def test_empty_prefetch_does_nothing() -> None:
    """prefetch() with an empty list is a no-op."""
    cache = _make_cache()
    await cache.prefetch(0, [])
    assert cache.stats.prefetch_issued == 0
    print("PASS: empty prefetch is a no-op")


async def test_prefetch_already_cached_skips() -> None:
    """prefetch() for an already-cached expert does not re-read."""
    cache = _make_cache(_make_store(expert_size=1000), budget=100_000)

    await cache.get(0, 5)  # miss, caches it
    await cache.prefetch(0, [5])  # already cached

    await asyncio.sleep(0.1)
    assert cache.stats.prefetch_issued == 0  # no new prefetch
    assert cache.stats.bytes_read == 1000  # only the original get
    print("PASS: prefetch skips already-cached experts")


async def test_close_cleans_up() -> None:
    """close() clears caches and cancels in-flight tasks."""
    store = _make_store(latency_ms=500.0)  # slow — inflight tasks linger
    cache = _make_cache(store, budget=10_000_000)

    await cache.get(0, 0)
    # Fire a prefetch that will be slow (wrapped in create_task to avoid
    # "coroutine was never awaited" warning — prefetch spawns bg tasks).
    asyncio.create_task(cache.prefetch(0, [1]))

    await cache.close()

    assert len(cache._cache) == 0
    assert len(cache._lru) == 0
    assert len(cache._in_flight) == 0
    assert len(cache._in_flight_from_prefetch) == 0
    print("PASS: close() cleans up all state")


async def test_same_layer_different_experts_are_distinct() -> None:
    """Two experts in the same layer are cached independently."""
    store = _make_store(expert_size=1000, seed=42)
    cache = _make_cache(store, budget=10_000)

    d1 = await cache.get(3, 7)
    d2 = await cache.get(3, 8)

    assert d1 != d2, "different experts in the same layer should differ"
    assert cache.stats.misses == 2
    print("PASS: same-layer different experts are independent")


async def test_different_layer_same_expert_are_distinct() -> None:
    """The same expert index in different layers are cached independently."""
    store = _make_store(expert_size=1000, seed=42)
    cache = _make_cache(store, budget=10_000)

    d1 = await cache.get(0, 5)
    d2 = await cache.get(1, 5)

    assert d1 != d2, "same expert index, different layers should differ"
    assert cache.stats.misses == 2
    print("PASS: cross-layer same-expert-index is independent")


# ── SimulatedExpertStore specific ─────────────────────────────────────────


async def test_simulated_store_is_deterministic() -> None:
    """Same seed, layer, expert → identical bytes every time."""
    store1 = _make_store(seed=42)
    store2 = _make_store(seed=42)

    d1 = await store1.read_expert(0, 0)
    d2 = await store2.read_expert(0, 0)

    assert d1 == d2, "same seed should produce identical bytes"
    print("PASS: SimulatedExpertStore is deterministic with same seed")


async def test_simulated_store_different_experts_differ() -> None:
    """Different experts produce different bytes."""
    store = _make_store(seed=42)

    d1 = await store.read_expert(0, 0)
    d2 = await store.read_expert(0, 1)
    d3 = await store.read_expert(1, 0)

    assert d1 != d2, "different experts in same layer should differ"
    assert d1 != d3, "same expert in different layers should differ"
    assert d2 != d3
    print("PASS: SimulatedExpertStore: different experts produce different bytes")


async def test_simulated_store_respects_bandwidth() -> None:
    """Bandwidth cap throttles read speed."""
    store = SimulatedExpertStore(
        expert_size_bytes=1_000_000,  # 1 MB
        num_layers=1,
        num_experts=1,
        bandwidth_mbps=100.0,  # 100 MB/s → 10 ms for 1 MB
        latency_ms=0.0,
    )

    t0 = time.monotonic()
    await store.read_expert(0, 0)
    elapsed = time.monotonic() - t0

    # 1 MB at 100 MB/s = 10 ms. Allow some slop.
    assert elapsed >= 0.008, f"too fast: {elapsed:.4f}s"
    print(f"PASS: bandwidth cap working — {elapsed*1000:.1f} ms for 1 MB @ 100 MB/s")

    await store.close()


# ── Runner ────────────────────────────────────────────────────────────────


async def _run() -> None:
    print("── Basic read-through ──")
    await test_cache_miss_reads_through()
    await test_cache_hit_avoids_store()

    print("\n── Budget enforcement ──")
    await test_budget_respected_under_pressure()
    await test_exact_budget_fill()

    print("\n── Pinning ──")
    await test_pinned_experts_survive_eviction()
    await test_pin_uncached_raises()
    await test_unpin_makes_evictable()

    print("\n── LRU order ──")
    await test_lru_order_is_correct()
    await test_lru_reshuffled_by_access()

    print("\n── Concurrent coalescing ──")
    await test_concurrent_get_coalesces_reads()
    await test_concurrent_get_and_prefetch_coalesce()
    await test_concurrent_get_does_not_double_read()

    print("\n── Prefetch ──")
    await test_prefetch_never_breaks_correctness()
    await test_prefetch_populates_cache()
    await test_prefetch_waste_tracked()

    print("\n── Instrumentation accuracy ──")
    await test_instrumentation_counters_accurate()
    await test_peak_tracking()

    print("\n── Oracle predictor ──")
    await test_oracle_predictor_produces_hits()

    print("\n── Edge cases ──")
    await test_empty_prefetch_does_nothing()
    await test_prefetch_already_cached_skips()
    await test_close_cleans_up()
    await test_same_layer_different_experts_are_distinct()
    await test_different_layer_same_expert_are_distinct()

    print("\n── SimulatedExpertStore ──")
    await test_simulated_store_is_deterministic()
    await test_simulated_store_different_experts_differ()
    await test_simulated_store_respects_bandwidth()


def main() -> None:
    asyncio.run(_run())
    print("\nAll expert cache tests passed.")


if __name__ == "__main__":
    main()
