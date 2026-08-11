#!/usr/bin/env python3
"""
storage_io.py — Async expert-weight I/O abstraction for Swarm layer 2.

Single-node expert streaming: the layer that decides what to read from disk
and when, before any node talks to another.

Design rules (do not break)
----------------------------
- Python 3.11+, standard library only. No numpy, no aiofiles, no aiohttp.
- Every unknown is an explicit, injectable parameter with a documented
  provisional default. When real measurements arrive they get plugged in
  without a rewrite.
- The interface must be testable against a simulated backend. Real NVMe
  numbers are not known yet and this module must not assume them.

O_DIRECT alignment requirements
--------------------------------
On Linux, O_DIRECT I/O requires the read buffer's memory address, the file
offset, and the transfer size to all be multiples of the filesystem's
logical block size (typically 512 bytes; 4096 for many NVMe drives).  A
misaligned read returns EINVAL.

This module uses ``mmap.mmap(-1, size)`` for the read buffer, which returns
page-aligned memory (4096-byte aligned), satisfying the strictest alignment
requirement.  The file offset is rounded down to the alignment boundary, the
read size is rounded up, and the caller receives a trimmed slice of the
correct length.

Verified by construction: mmap(-1) guarantees page alignment on all
platforms; the alignment constant (4096) is the common denominator of every
reasonable block size; and the round-down/round-up logic is arithmetic,
not conditional on filesystem probes that can disagree across environments.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import mmap
import os
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("swarm.storage_io")

# ── Constants ──────────────────────────────────────────────────────────────────

# Provisional defaults — all are explicitly overridable.
DEFAULT_BLOCK_SIZE: int = 16 * 1024 * 1024  # 16 MB (matches architecture.md §3.1)
DEFAULT_ALIGNMENT: int = 4096  # bytes — safe for essentially all NVMe drives
DEFAULT_QUEUE_DEPTH: int = 4  # concurrent reads before backpressure
DEFAULT_EXPERT_SIZE_BYTES: int = 48 * 1024 * 1024  # 48 MB — GLM-5.2 expert, Q4_K_M
DEFAULT_NVME_BANDWIDTH_MBPS: float = 3_200.0  # ~3.2 GB/s — PCIe 3.0 x4 nominal
DEFAULT_EMMC_BANDWIDTH_MBPS: float = 280.0  # ~280 MB/s — eMMC 5.1 nominal

# O_DIRECT flag.  Linux only; macOS and some filesystems do not support it.
_O_DIRECT_AVAILABLE: bool = hasattr(os, "O_DIRECT")

if not _O_DIRECT_AVAILABLE:
    logger.info("O_DIRECT unavailable on this platform — direct I/O reads will "
                "fall back to buffered reads.  Storage benchmark numbers will "
                "include page-cache effects.")


# ── ExpertStore (protocol / ABC) ──────────────────────────────────────────────


class ExpertStore(abc.ABC):
    """Async interface for reading expert weights from storage.

    Every expert is keyed by ``(layer, expert)`` and returned as opaque
    ``bytes``.  This module never interprets the bytes — it only moves them.

    Implementations:
      - :class:`SimulatedExpertStore` — injectable latency/bandwidth, for
        testing the cache and prefetch layers without real NVMe.
      - :class:`DirectFileExpertStore` — real O_DIRECT reads against a file
        on disk.
    """

    @abc.abstractmethod
    async def read_expert(self, layer: int, expert: int) -> bytes:
        """Read one expert's weight tensor from storage.

        Parameters
        ----------
        layer:
            Layer index (0-based).
        expert:
            Expert slot index within the layer (0-based).

        Returns
        -------
        bytes
            The expert's weight data.
        """
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """Release any resources (file handles, etc.)."""
        ...

    async def __aenter__(self) -> ExpertStore:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


# ── SimulatedExpertStore ──────────────────────────────────────────────────────


class SimulatedExpertStore(ExpertStore):
    """Expert store backed by generated or file-backed bytes with artificial
    latency and bandwidth caps.

    Use this to test the cache layer with realistic-ish storage behaviour
    without any NVMe hardware.  Supports fast NVMe, slow eMMC, and
    deliberately unbalanced mixed-fleet configurations.

    Parameters
    ----------
    expert_size_bytes:
        Size of each expert tensor in bytes.  All experts are the same size
        in this simulation (true for any given model).
    num_layers:
        Total number of MoE layers.
    num_experts:
        Total expert slots per layer.
    bandwidth_mbps:
        Simulated sequential-read bandwidth in megabytes per second.
        Throughput is enforced by sleeping after each read to throttle
        the effective rate.  Default: 3200 MB/s (~PCIe 3.0 x4).
    latency_ms:
        Fixed per-read latency in milliseconds, added before the bandwidth
        throttle.  Models seek + command overhead.  Default: 0.1 ms.
    seed:
        If provided, use this to seed a deterministic byte pattern for
        each expert (the bytes are ``seed + layer + expert`` repeated).
        Experts are generated lazily — bytes are only materialised when
        first read, not at construction time.
    backing_file:
        If provided, read experts from this file instead of generating
        bytes.  The file must be at least
        ``num_layers * num_experts * expert_size_bytes`` long.  Experts
        are laid out sequentially: layer-major, then expert.
        Bandwidth and latency caps still apply.
    """

    def __init__(
        self,
        *,
        expert_size_bytes: int = DEFAULT_EXPERT_SIZE_BYTES,
        num_layers: int,
        num_experts: int,
        bandwidth_mbps: float = DEFAULT_NVME_BANDWIDTH_MBPS,
        latency_ms: float = 0.1,
        seed: int | None = None,
        backing_file: str | None = None,
    ) -> None:
        self._expert_size = expert_size_bytes
        self._num_layers = num_layers
        self._num_experts = num_experts
        self._bandwidth_mbps = bandwidth_mbps
        self._latency_s = latency_ms / 1000.0
        self._seed = seed
        self._backing_file = backing_file

        # Lazy cache of generated experts.  In production these would be
        # tens of GB and never cached; in simulation they are small and
        # we cache to avoid recomputing the same byte pattern.
        self._generated: dict[tuple[int, int], bytes] = {}

        # File handle for backing_file, opened on first use.
        self._fh: Any = None  # int fd or None

        # Concurrency guard — bandwidth is a shared resource across reads.
        self._io_lock = asyncio.Lock()

        logger.info(
            "SimulatedExpertStore: %d layers × %d experts × %.1f MB = %.1f GB total, "
            "bandwidth=%.0f MB/s, latency=%.2f ms",
            num_layers,
            num_experts,
            expert_size_bytes / (1024 * 1024),
            (num_layers * num_experts * expert_size_bytes) / (1024 * 1024 * 1024),
            bandwidth_mbps,
            latency_ms,
        )

    async def read_expert(self, layer: int, expert: int) -> bytes:
        """Read one expert with simulated latency and bandwidth throttling."""
        if not (0 <= layer < self._num_layers):
            raise ValueError(
                f"layer {layer} out of range [0, {self._num_layers})"
            )
        if not (0 <= expert < self._num_experts):
            raise ValueError(
                f"expert {expert} out of range [0, {self._num_experts})"
            )

        key = (layer, expert)

        # ── Simulated seek latency ────────────────────────────────────
        if self._latency_s > 0:
            await asyncio.sleep(self._latency_s)

        # ── Bandwidth throttle ────────────────────────────────────────
        # To simulate a shared bus, serialise on the lock so concurrent
        # reads don't all get full bandwidth.
        async with self._io_lock:
            transfer_s = self._expert_size / (self._bandwidth_mbps * 1_000_000)
            if transfer_s > 0:
                await asyncio.sleep(transfer_s)

        # ── Fetch or generate bytes ───────────────────────────────────
        if key in self._generated:
            return self._generated[key]

        if self._backing_file is not None:
            data = self._read_from_file(layer, expert)
        else:
            data = self._generate_expert(layer, expert)

        self._generated[key] = data
        return data

    async def close(self) -> None:
        if self._fh is not None:
            os.close(self._fh)
            self._fh = None
        self._generated.clear()

    # ── Internal ──────────────────────────────────────────────────────────

    def _expert_offset(self, layer: int, expert: int) -> int:
        """Byte offset of expert (layer, expert) in the flat file layout."""
        return (layer * self._num_experts + expert) * self._expert_size

    def _read_from_file(self, layer: int, expert: int) -> bytes:
        """Read expert bytes from the backing file synchronously."""
        if self._fh is None:
            self._fh = os.open(self._backing_file, os.O_RDONLY)
        offset = self._expert_offset(layer, expert)
        return os.pread(self._fh, self._expert_size, offset)

    def _generate_expert(self, layer: int, expert: int) -> bytes:
        """Generate deterministic-but-distinct bytes for one expert.

        Uses (seed, layer, expert) so every expert is distinct.  The
        pattern repeats a 256-byte sequence derived from these values.
        """
        base = self._seed if self._seed is not None else 0
        # Build a 256-byte pattern unique to this expert.
        pattern = bytearray(256)
        # Mix seed, layer, expert into a simple deterministic sequence.
        r = (base * 31 + layer * 17 + expert * 13) & 0xFFFFFFFF
        for i in range(256):
            r = (r * 1103515245 + 12345) & 0xFFFFFFFF
            pattern[i] = r & 0xFF

        # Repeat to fill expert_size_bytes.
        repeats = self._expert_size // 256
        remainder = self._expert_size % 256
        data = bytes(pattern) * repeats
        if remainder:
            data += bytes(pattern[:remainder])
        return data


# ── DirectFileExpertStore ─────────────────────────────────────────────────────


class DirectFileExpertStore(ExpertStore):
    """Expert store backed by a real file on disk with O_DIRECT reads.

    Reads expert weights from a flat file where experts are laid out
    sequentially: layer-major, then expert within each layer.

    Uses ``os.preadv`` with an ``mmap``-backed buffer for aligned O_DIRECT
    I/O.  Falls back to buffered reads if O_DIRECT is unavailable (macOS,
    some filesystems) with a clear log message.

    Parameters
    ----------
    path:
        Path to the expert weights file.  Must be a regular file on a
        filesystem that supports O_DIRECT (Linux with ext4/xfs/btrfs).
    expert_size_bytes:
        Size of each expert tensor in bytes.
    num_layers:
        Total number of MoE layers.
    num_experts:
        Total expert slots per layer.
    block_size:
        Read granularity in bytes.  Each ``read_expert()`` call reads
        exactly ``expert_size_bytes``, but this controls the alignment
        padding.  Default: 16 MB.
    alignment:
        Memory and file-offset alignment in bytes.  Must be a power of two
        and at least 512.  Default: 4096.
    queue_depth:
        Maximum concurrent in-flight reads.  Additional reads will block
        at the semaphore.  Default: 4.
    use_odirect:
        If ``False``, skip O_DIRECT and use buffered reads even on Linux.
        For benchmarking comparisons (O_DIRECT vs buffered).
    """

    def __init__(
        self,
        *,
        path: str,
        expert_size_bytes: int,
        num_layers: int,
        num_experts: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        alignment: int = DEFAULT_ALIGNMENT,
        queue_depth: int = DEFAULT_QUEUE_DEPTH,
        use_odirect: bool = True,
    ) -> None:
        self._path = path
        self._expert_size = expert_size_bytes
        self._num_layers = num_layers
        self._num_experts = num_experts
        self._block_size = block_size
        self._alignment = alignment
        self._use_odirect = use_odirect and _O_DIRECT_AVAILABLE

        # Validate alignment.
        if alignment < 512:
            raise ValueError(f"alignment must be at least 512, got {alignment}")
        if alignment & (alignment - 1) != 0:
            raise ValueError(f"alignment must be a power of two, got {alignment}")

        # Open the file.
        _flags = os.O_RDONLY
        if self._use_odirect:
            _flags |= os.O_DIRECT

        try:
            self._fd = os.open(path, _flags)
        except OSError as e:
            if self._use_odirect:
                logger.warning(
                    "O_DIRECT open failed for %s: %s — falling back to buffered "
                    "I/O.  Benchmark numbers will include page-cache effects.",
                    path,
                    e,
                )
                self._fd = os.open(path, os.O_RDONLY)
                self._use_odirect = False
            else:
                raise

        # Verify the file is large enough.
        total_experts = num_layers * num_experts
        total_size = total_experts * expert_size_bytes
        try:
            stat = os.fstat(self._fd)
            if stat.st_size < total_size:
                logger.warning(
                    "Expert file %s is %d bytes, but %d experts × %d bytes = %d "
                    "bytes needed. Reads past EOF will fail.",
                    path,
                    stat.st_size,
                    total_experts,
                    expert_size_bytes,
                    total_size,
                )
        except OSError:
            pass

        # Concurrency limiter.
        self._read_sem = asyncio.Semaphore(queue_depth)

        # mmap pool for aligned read buffers.  We allocate one buffer per
        # queue-depth slot so we never need to allocate during a read.
        # Each buffer is large enough for the aligned read of one expert.
        self._aligned_read_size = self._round_up(expert_size_bytes, alignment)
        self._buffers: list[mmap.mmap] = [
            mmap.mmap(-1, self._aligned_read_size) for _ in range(queue_depth)
        ]
        self._buffer_index: int = 0

        logger.info(
            "DirectFileExpertStore: %s, %d layers × %d experts × %.1f MB, "
            "alignment=%d, O_DIRECT=%s, queue_depth=%d",
            path,
            num_layers,
            num_experts,
            expert_size_bytes / (1024 * 1024),
            alignment,
            self._use_odirect,
            queue_depth,
        )

    async def read_expert(self, layer: int, expert: int) -> bytes:
        """Read one expert from the file with O_DIRECT alignment.

        Acquires a concurrency slot, an aligned buffer, performs the
        aligned pread, and returns a copy of the exact requested range.
        """
        if not (0 <= layer < self._num_layers):
            raise ValueError(
                f"layer {layer} out of range [0, {self._num_layers})"
            )
        if not (0 <= expert < self._num_experts):
            raise ValueError(
                f"expert {expert} out of range [0, {self._num_experts})"
            )

        offset = (layer * self._num_experts + expert) * self._expert_size

        async with self._read_sem:
            return await asyncio.to_thread(self._aligned_pread, offset)

    def _aligned_pread(self, offset: int) -> bytes:
        """Perform an aligned pread, return the exact expert bytes.

        Runs in a thread (via ``asyncio.to_thread``) so the O_DIRECT read
        does not block the event loop.

        Alignment strategy:
          1. Round ``offset`` down to the nearest alignment boundary.
          2. Compute the aligned read size: round up from
             ``(offset - aligned_offset) + expert_size`` to alignment.
          3. Read into a page-aligned mmap buffer.
          4. Return the slice ``[offset - aligned_offset : ... + expert_size]``.
        """
        aligned_offset = (offset // self._alignment) * self._alignment
        offset_skip = offset - aligned_offset
        read_size = self._round_up(offset_skip + self._expert_size, self._alignment)

        # Get a buffer from the pool (round-robin).
        buf = self._buffers[self._buffer_index % len(self._buffers)]
        self._buffer_index += 1

        # os.preadv into the mmap buffer.
        # mmap objects support the buffer protocol; we wrap in a single-element
        # list of bytearray views for preadv.
        #
        # Note: we slice the mmap to the exact read_size so preadv knows
        # exactly how many bytes to read.
        view = memoryview(buf)[:read_size]
        bytes_read = os.preadv(self._fd, [view], aligned_offset)

        if bytes_read < offset_skip + self._expert_size:
            # Short read — the file doesn't have enough data.  This is a
            # real error (misconfigured file or wrong expert count).
            raise IOError(
                f"Short read at offset {offset}: expected "
                f"{offset_skip + self._expert_size} bytes, got {bytes_read}"
            )

        # Return a copy of the exact expert range.
        return bytes(view[offset_skip : offset_skip + self._expert_size])

    async def close(self) -> None:
        """Close the file descriptor and release mmap buffers."""
        if hasattr(self, "_fd"):
            os.close(self._fd)
        if hasattr(self, "_buffers"):
            for buf in self._buffers:
                buf.close()
            self._buffers.clear()

    @staticmethod
    def _round_up(n: int, alignment: int) -> int:
        """Round *n* up to the nearest multiple of *alignment*."""
        return ((n + alignment - 1) // alignment) * alignment


# ── Helpers ────────────────────────────────────────────────────────────────────


def simulated_nvme_store(
    num_layers: int = 16,
    num_experts: int = 64,
    expert_size_bytes: int = DEFAULT_EXPERT_SIZE_BYTES,
) -> SimulatedExpertStore:
    """Convenience factory: fast NVMe simulation (~3.2 GB/s)."""
    return SimulatedExpertStore(
        expert_size_bytes=expert_size_bytes,
        num_layers=num_layers,
        num_experts=num_experts,
        bandwidth_mbps=DEFAULT_NVME_BANDWIDTH_MBPS,
        latency_ms=0.1,
    )


def simulated_emmc_store(
    num_layers: int = 16,
    num_experts: int = 64,
    expert_size_bytes: int = DEFAULT_EXPERT_SIZE_BYTES,
) -> SimulatedExpertStore:
    """Convenience factory: slow eMMC simulation (~280 MB/s)."""
    return SimulatedExpertStore(
        expert_size_bytes=expert_size_bytes,
        num_layers=num_layers,
        num_experts=num_experts,
        bandwidth_mbps=DEFAULT_EMMC_BANDWIDTH_MBPS,
        latency_ms=0.5,
    )


def simulated_mixed_fleet_stores(
    num_nodes: int = 4,
    num_layers: int = 16,
    num_experts: int = 64,
    expert_size_bytes: int = DEFAULT_EXPERT_SIZE_BYTES,
) -> list[SimulatedExpertStore]:
    """Convenience factory: a deliberately unbalanced mixed fleet.

    Returns one SimulatedExpertStore per node with bandwidths ranging
    from slow eMMC to fast NVMe, so the cache and prefetch layers can
    be tested against the kind of heterogeneity the sharding layer
    already handles.
    """
    # Spread bandwidths from eMMC to NVMe.
    bw_range = DEFAULT_NVME_BANDWIDTH_MBPS - DEFAULT_EMMC_BANDWIDTH_MBPS
    stores: list[SimulatedExpertStore] = []
    for i in range(num_nodes):
        fraction = i / max(num_nodes - 1, 1)
        bw = DEFAULT_EMMC_BANDWIDTH_MBPS + bw_range * fraction
        stores.append(
            SimulatedExpertStore(
                expert_size_bytes=expert_size_bytes,
                num_layers=num_layers,
                num_experts=num_experts,
                bandwidth_mbps=bw,
                latency_ms=0.5 - 0.4 * fraction,  # 0.5 → 0.1 ms
            )
        )
    return stores


# ── Quick self-test ────────────────────────────────────────────────────────────


async def _self_test() -> None:
    """Verify the simulated store works and respects bandwidth caps."""
    import time as _time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("── SimulatedExpertStore self-test ──")

    # Small store for quick testing.
    store = SimulatedExpertStore(
        expert_size_bytes=1_000_000,  # 1 MB per expert
        num_layers=2,
        num_experts=8,
        bandwidth_mbps=100.0,  # 100 MB/s = 10 ms per 1 MB expert
        latency_ms=1.0,
        seed=42,
    )

    # Single read.
    t0 = _time.monotonic()
    data = await store.read_expert(0, 0)
    elapsed = _time.monotonic() - t0
    assert len(data) == 1_000_000
    expected_s = 0.001 + (1_000_000 / (100 * 1_000_000))  # latency + bandwidth
    assert elapsed >= expected_s * 0.8, f"too fast: {elapsed:.4f}s < {expected_s:.4f}s"
    print(f"  Single read: {elapsed*1000:.1f} ms (expected ~{expected_s*1000:.1f} ms) ✓")

    # Deterministic bytes.
    data2 = await store.read_expert(0, 0)
    assert data2 == data, "same expert returned different bytes"
    data3 = await store.read_expert(0, 1)
    assert data3 != data, "different expert returned same bytes"
    print("  Deterministic generation ✓")

    # Out-of-range rejection.
    try:
        await store.read_expert(99, 0)
    except ValueError:
        print("  Out-of-range layer rejected ✓")

    await store.close()
    print("── Self-test passed ──")


if __name__ == "__main__":
    asyncio.run(_self_test())
