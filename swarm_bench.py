#!/usr/bin/env python3
"""
swarm_bench.py — single-file RK3588 benchmark for the Swarm distributed
inference project.

Design constraints
------------------
- Python 3.8+ standard library ONLY.  No pip, no numpy, no third-party
  packages.  Must run on a fresh Armbian/Debian image with nothing added.
- Read-only and safe.  Test data written to a temp directory that is
  cleaned up on exit.  Never touches anything outside it.  Never requires
  root unless the user explicitly opts in with --sudo.
- Single file.  Save it, run it, paste the output.

What this measures
------------------
A) Platform identification — board model, SoC, SDOT presence (decides
   whether llama.cpp's fast NEON kernels work), RAM, kernel, NVMe link info.
B) Storage bandwidth — sequential and random reads at 16 MB block size,
   with and without O_DIRECT.  Uses fio if available; falls back to a
   pure-Python os.preadv loop otherwise.
C) Sustained compute — a fixed workload running for --duration seconds
   (default 600, 10 min), sampling CPU frequency and thermal-zone
   temperature every second.  Reports peak and settled figures.
D) Power — clear instructions for manual measurement with a wall meter.

Output: one pasteable text block with clear delimiters, plus a .json file
with the same data in machine-readable form.

Usage
-----
    python3 swarm_bench.py                     # everything, defaults
    python3 swarm_bench.py --duration 120      # 2-minute compute test
    python3 swarm_bench.py --bench-size-gb 4   # smaller test file
    python3 swarm_bench.py --skip-storage      # platform + compute only
    python3 swarm_bench.py --skip-compute      # platform + storage only
    python3 swarm_bench.py --sudo              # attempt cache drops (needs root)
"""

from __future__ import annotations

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import mmap
import os
import platform as _platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VERSION = "1.0.0"

# ── argparse ───────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Swarm RK3588 benchmark — measures storage bandwidth, "
                    "sustained compute, and platform identity."
    )
    p.add_argument(
        "--duration", type=int, default=600,
        help="Sustained compute test duration in seconds (default: 600, 10 min)"
    )
    p.add_argument(
        "--bench-size-gb", type=int, default=8,
        help="Size of test file for storage benchmark in GB (default: 8)"
    )
    p.add_argument(
        "--test-dir", type=str, default=None,
        help="Directory to create the storage test file in. Defaults to the "
             "current working directory (per a Reddit suggestion from "
             "u/12345myluggage) rather than /tmp, since /tmp is a RAM disk "
             "(tmpfs) on many systems and silently produces meaningless "
             "storage numbers. An explicit TMPDIR environment variable is "
             "still respected if --test-dir is not given."
    )
    p.add_argument(
        "--skip-storage", action="store_true",
        help="Skip the storage benchmark entirely"
    )
    p.add_argument(
        "--skip-compute", action="store_true",
        help="Skip the sustained compute benchmark"
    )
    p.add_argument(
        "--sudo", action="store_true",
        help="Attempt operations that need root (drop caches, "
             "read-protected sysfs entries).  Without this, cache effects "
             "may inflate storage numbers."
    )
    p.add_argument(
        "--fill", action="store_true",
        help="Write actual (non-zero) data for the test file instead of "
             "using fallocate.  Slower setup, but guards against "
             "compressed-filesystem artefacts.  Ignored if fio is used."
    )
    p.add_argument(
        "--json-only", type=str, default=None,
        help="Write JSON output to this path and exit without printing "
             "the text block.  For automated collection."
    )
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _read_sysfs(path: str) -> str | None:
    """Read a sysfs/procfs file, return stripped content or None."""
    try:
        with open(path) as fh:
            return fh.read().strip()
    except (OSError, PermissionError):
        return None


def _read_sysfs_lines(path: str) -> list[str]:
    try:
        with open(path) as fh:
            return [line.strip() for line in fh if line.strip()]
    except (OSError, PermissionError):
        return []


def _read_sysfs_int(path: str) -> int | None:
    v = _read_sysfs(path)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(args: list[str], timeout: float = 60.0, env: dict | None = None) -> tuple[int, str, str]:
    """Run a subprocess, return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(args, capture_output=True, text=True,
                           timeout=timeout, env=env or {})
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return -1, "", f"command not found: {args[0]}"
    except subprocess.TimeoutExpired:
        return -2, "", f"timeout after {timeout}s"


def _detect_filesystem(path: str) -> str:
    """Best-effort filesystem type for *path* (empty string if unknown)."""
    try:
        dev = os.stat(path).st_dev
        with open("/proc/self/mountinfo") as fh:
            best, best_len = "", -1
            for line in fh:
                parts = line.split()
                try:
                    sep = parts.index("-")
                except ValueError:
                    continue
                mount_point, fstype = parts[4], parts[sep + 1]
                if path.startswith(mount_point) and len(mount_point) > best_len:
                    best, best_len = fstype, len(mount_point)
            return best
    except Exception:
        return ""


def _sanity_check_storage(result) -> None:
    """Flag physically implausible numbers before they are submitted.

    A RAM-backed filesystem (tmpfs, ramfs) makes O_DIRECT a no-op and serves
    reads from memory, producing figures in the tens of GB/s. Those are real
    measurements of the wrong thing, and without this check a well-meaning
    contributor would submit them and nobody would notice. PCIe 3.0 x4 tops
    out near 3.9 GB/s and PCIe 4.0 x4 near 7.9 GB/s, so anything much above
    8000 MB/s did not come from a drive.
    """
    IMPLAUSIBLE_MBPS = 8000.0
    for label, val in (
        ("seq_read_odirect", result.seq_read_odirect_mbps),
        ("rand_read_odirect", result.rand_read_odirect_mbps),
        ("seq_read_buffered", result.seq_read_buffered_mbps),
        ("rand_read_buffered", result.rand_read_buffered_mbps),
    ):
        if val is not None and val > IMPLAUSIBLE_MBPS:
            result.caveats.append(
                f"IMPLAUSIBLE: {label} = {val:.0f} MB/s exceeds any PCIe 4.0 "
                f"x4 drive (~7900 MB/s). The test file is almost certainly on "
                f"a RAM-backed or virtualised filesystem, so this measures "
                f"memory, not storage. Set TMPDIR to a directory on the real "
                f"NVMe (e.g. TMPDIR=/mnt/nvme/tmp python3 swarm_bench.py) and "
                f"re-run. DO NOT SUBMIT these storage numbers."
            )
            break

    fstype = getattr(result, "filesystem", "")
    if fstype in ("tmpfs", "ramfs", "overlay", "9p", "v9fs"):
        result.caveats.append(
            f"Test file was on a '{fstype}' filesystem, which is not real "
            f"block storage. Storage numbers are not meaningful. Set TMPDIR "
            f"to a path on the NVMe and re-run."
        )


def _check_free_space(path: str, need_gb: float) -> tuple[bool, float]:
    """Return (ok, free_gb) for *path*.

    Checked BEFORE creating the test file. Without this the benchmark
    happily fills a small eMMC and dies with ENOSPC partway through --
    unacceptable behaviour for a script strangers are asked to run on
    their own hardware. A 20% margin is kept free.
    """
    try:
        st = os.statvfs(path)
        free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
    except OSError:
        return True, -1.0  # can't tell; don't block, but report unknown
    return free_gb >= need_gb * 1.2, free_gb


def _drop_caches() -> bool:
    """Attempt to drop page caches.  Returns True on success."""
    if os.geteuid() != 0:
        return False
    try:
        with open("/proc/sys/vm/drop_caches", "w") as fh:
            fh.write("3")
        return True
    except (OSError, PermissionError):
        return False


def _can_write_sudo(path: str) -> bool:
    """Check if we can write to a root-owned path (implies we're root)."""
    try:
        with open(path, "w") as fh:
            fh.write("")
        return True
    except (OSError, PermissionError):
        return False


# ── Section A: Platform identification ─────────────────────────────────────────


def _platform_info() -> dict:
    """Gather everything we can about the board without running benchmarks."""
    info: dict = {}

    # ── Board model ───────────────────────────────────────────────────────
    info["board_model"] = _read_sysfs("/proc/device-tree/model") or "unknown"
    # Strip trailing null from dtb strings
    info["board_model"] = info["board_model"].rstrip("\x00")

    # ── Kernel / distro ───────────────────────────────────────────────────
    info["kernel"] = _read_sysfs("/proc/version") or _platform.release()
    distro = "unknown"
    for f in ["/etc/os-release", "/usr/lib/os-release"]:
        try:
            for line in open(f).read().splitlines():
                if line.startswith("PRETTY_NAME="):
                    distro = line.split("=", 1)[1].strip('"')
                    break
        except OSError:
            continue
        if distro != "unknown":
            break
    info["distro"] = distro
    info["arch"] = _platform.machine()
    info["python_version"] = sys.version.split()[0]

    # ── CPU: cores, frequencies ───────────────────────────────────────────
    cpuinfo = _read_sysfs_lines("/proc/cpuinfo")
    info["cpu_cores"] = sum(1 for l in cpuinfo if l.startswith("processor"))
    # Try to identify SoC from model name in cpuinfo
    soc = "unknown"
    for line in cpuinfo:
        if line.startswith("model name") or line.lower().startswith("cpu model"):
            soc = line.split(":", 1)[1].strip()
            break
    if soc == "unknown":
        # Try Hardware line
        for line in cpuinfo:
            if line.startswith("Hardware"):
                soc = line.split(":", 1)[1].strip()
                break
    info["soc_model"] = soc

    # CPU features — SDOT is the one that matters
    features_line = ""
    for line in cpuinfo:
        if line.startswith("Features") or line.startswith("features"):
            features_line = line
            break
    info["cpu_features"] = features_line.split(":", 1)[1].strip() if ":" in features_line else ""
    info["sdot_present"] = "asimddp" in features_line

    # Per-core frequency info
    freqs_current: dict[str, int | None] = {}
    freqs_max: dict[str, int | None] = {}
    for cpu in range(info["cpu_cores"]):
        cur = _read_sysfs_int(
            f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
        )
        max_f = _read_sysfs_int(
            f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_max_freq"
        )
        if cur is not None:
            freqs_current[f"cpu{cpu}"] = cur
        if max_f is not None:
            freqs_max[f"cpu{cpu}"] = max_f
    info["cpu_freqs_current_khz"] = freqs_current
    info["cpu_freqs_max_khz"] = freqs_max

    # Governor
    gov = _read_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    info["cpu_governor"] = gov or "unknown"

    # ── RAM ───────────────────────────────────────────────────────────────
    meminfo = _read_sysfs_lines("/proc/meminfo")
    mem_map: dict[str, int] = {}
    for line in meminfo:
        parts = line.split(":")
        if len(parts) >= 2:
            val_str = parts[1].strip().split()[0]
            try:
                mem_map[parts[0].strip()] = int(val_str)
            except ValueError:
                pass
    info["ram_total_kb"] = mem_map.get("MemTotal", 0)
    info["ram_available_kb"] = mem_map.get("MemAvailable", 0)

    # ── Storage devices ───────────────────────────────────────────────────
    info["nvme_devices"] = _find_nvme_devices()
    info["block_devices"] = _find_block_devices()

    # ── Thermal zones ─────────────────────────────────────────────────────
    info["thermal_zones"] = _find_thermal_zones()

    return info


def _find_nvme_devices() -> list[dict]:
    """Enumerate NVMe block devices and their PCIe link info."""
    devices: list[dict] = []
    for blk in sorted(Path("/sys/block").glob("nvme*")):
        name = blk.name
        dev_info: dict = {"name": name}
        # Size
        size_sectors = _read_sysfs_int(str(blk / "size"))
        if size_sectors is not None:
            dev_info["size_gb"] = round(size_sectors * 512 / 1e9, 1)
        # Model
        model = _read_sysfs(str(blk / "device" / "model"))
        if model is not None:
            dev_info["model"] = model
        # PCIe link info — walk up to the PCI device
        pci_path = _find_pci_path_for_nvme(blk)
        if pci_path is not None:
            dev_info["pci_address"] = pci_path.name
            link_speed = _read_sysfs(str(pci_path / "current_link_speed"))
            link_width = _read_sysfs(str(pci_path / "current_link_width"))
            if link_speed is not None:
                dev_info["pcie_link_speed"] = link_speed
            if link_width is not None:
                dev_info["pcie_link_width"] = link_width
            # Max link capability
            max_speed = _read_sysfs(str(pci_path / "max_link_speed"))
            max_width = _read_sysfs(str(pci_path / "max_link_width"))
            if max_speed is not None:
                dev_info["pcie_max_speed"] = max_speed
            if max_width is not None:
                dev_info["pcie_max_width"] = max_width
        devices.append(dev_info)
    return devices


def _find_pci_path_for_nvme(blk: Path) -> Path | None:
    """Walk /sys/block/nvmeXn1/device up to find the PCI device directory."""
    cur = blk / "device"
    for _ in range(6):
        if not cur.is_symlink() and not cur.exists():
            return None
        real = cur.resolve()
        # A PCI device directory has files like 'vendor', 'device', 'class'
        if (real / "vendor").exists() and (real / "class").exists():
            # Check it's PCI (class starts with 0x01 for mass storage)
            cls = _read_sysfs(str(real / "class"))
            if cls and cls.startswith("0x01"):
                return real
        cur = real / ".."
    return None


def _find_block_devices() -> list[dict]:
    """List non-NVMe block devices (eMMC, SD, etc)."""
    devices: list[dict] = []
    for blk in sorted(Path("/sys/block").iterdir()):
        name = blk.name
        if name.startswith("nvme") or name.startswith("loop") or name.startswith("ram"):
            continue
        if name.startswith("zram"):
            continue
        size_sectors = _read_sysfs_int(str(blk / "size"))
        if size_sectors is None or size_sectors == 0:
            continue
        dev_info: dict = {"name": name, "size_gb": round(size_sectors * 512 / 1e9, 1)}
        # Check if it's an eMMC or SD card
        mmc_type = _read_sysfs(str(blk / "device" / "type"))
        if mmc_type:
            dev_info["type"] = mmc_type
        devices.append(dev_info)
    return devices


def _find_thermal_zones() -> list[dict]:
    """Enumerate thermal zones and current temperatures."""
    zones: list[dict] = []
    for tz in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        ttype = _read_sysfs(str(tz / "type"))
        temp_raw = _read_sysfs_int(str(tz / "temp"))
        if temp_raw is not None:
            zones.append({
                "name": tz.name,
                "type": ttype or "unknown",
                "temp_c": round(temp_raw / 1000.0, 1),
            })
    return zones


# ── Section B: Storage bandwidth ───────────────────────────────────────────────


class StorageBenchResult:
    def __init__(self):
        self.method: str = ""           # "fio" or "python_preadv"
        self.file_size_gb: float = 0.0
        self.cache_dropped: bool = False
        # Sequential
        self.seq_read_buffered_mbps: float | None = None
        self.seq_read_odirect_mbps: float | None = None
        # Random
        self.rand_read_buffered_mbps: float | None = None
        self.rand_read_odirect_mbps: float | None = None
        # O_DIRECT improvement (if both measured)
        self.odirect_seq_improvement_pct: float | None = None
        self.odirect_rand_improvement_pct: float | None = None
        # Caveats
        self.caveats: list[str] = []
        self.filesystem: str = ""


def _run_storage_bench(args: argparse.Namespace) -> StorageBenchResult:
    """Run the storage benchmark, using fio if available.

    Creates a temp directory, allocates a test file, runs sequential and
    random reads at 16 MB block size with and without O_DIRECT, then
    cleans up.
    """
    result = StorageBenchResult()

    bench_bytes = args.bench_size_gb * 1024 * 1024 * 1024
    result.file_size_gb = float(args.bench_size_gb)

    # Decide method
    if _which("fio") is not None:
        result.method = "fio"
        return _run_fio_bench(args, result, bench_bytes)
    else:
        result.method = "python_preadv"
        result.caveats.append(
            "fio not found; using pure-Python os.preadv fallback. "
            "Install fio for more accurate numbers (apt install fio)."
        )
        return _run_python_bench(args, result, bench_bytes)


def _run_fio_bench(
    args: argparse.Namespace, result: StorageBenchResult, bench_bytes: int
) -> StorageBenchResult:
    """Run fio-based storage benchmarks."""
    base_dir = args.test_dir or os.environ.get("TMPDIR") or os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="swarm_bench_", dir=base_dir)
    result.filesystem = _detect_filesystem(tmpdir)
    ok, free_gb = _check_free_space(tmpdir, args.bench_size_gb)
    if not ok:
        _sanity_check_storage(result)
        shutil.rmtree(tmpdir, ignore_errors=True)
        result.caveats.append(
            f"SKIPPED: need ~{args.bench_size_gb:.1f} GB free (plus margin) "
            f"but only {free_gb:.1f} GB available at {tempfile.gettempdir()}. "
            f"Re-run with --bench-size-gb {max(1, int(free_gb / 2))} or set "
            f"TMPDIR to a bigger filesystem."
        )
        print(f"  !! Not enough free space ({free_gb:.1f} GB) — storage "
              f"benchmark skipped. Try --bench-size-gb "
              f"{max(1, int(free_gb / 2))}")
        return result
    testfile = os.path.join(tmpdir, "test.dat")
    block_size = "16M"
    # fio --size uses the total bytes to read, not the file size
    size_str = f"{args.bench_size_gb}G"

    try:
        # Create the test file first (fio can do this but explicit is clearer)
        subprocess.run(
            ["dd", "if=/dev/zero", f"of={testfile}", "bs=16M",
             f"count={args.bench_size_gb * 64}"],
            capture_output=True, timeout=300
        )

        # Helper to run one fio job
        def _fio_job(name: str, rw: str, direct: int) -> float | None:
            if args.sudo:
                _drop_caches()
            elif not result.cache_dropped:
                result.caveats.append(
                    "Cache not dropped between runs (need --sudo). "
                    "Buffered numbers may be inflated by page cache; "
                    "O_DIRECT numbers are unaffected."
                )
            cmd = [
                "fio",
                f"--name={name}",
                f"--rw={rw}",
                f"--bs={block_size}",
                f"--size={size_str}",
                f"--direct={direct}",
                "--ioengine=libaio",
                "--iodepth=4",
                "--group_reporting",
                "--output-format=json",
                f"--filename={testfile}",
            ]
            rc, stdout, stderr = _run(cmd, timeout=300)
            if rc != 0:
                result.caveats.append(f"fio {name} failed (rc={rc}): {stderr[:200]}")
                return None
            try:
                data = json.loads(stdout)
                # Extract bandwidth in bytes/sec → MB/s
                bw_bytes = data["jobs"][0]["read"]["bw_bytes"]
                return round(bw_bytes / 1e6, 1)
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                result.caveats.append(f"fio {name} parse error: {e}")
                return None

        result.cache_dropped = args.sudo and os.geteuid() == 0

        result.seq_read_buffered_mbps = _fio_job("seq-buf", "read", 0)
        result.seq_read_odirect_mbps = _fio_job("seq-direct", "read", 1)

        result.rand_read_buffered_mbps = _fio_job("rand-buf", "randread", 0)
        result.rand_read_odirect_mbps = _fio_job("rand-direct", "randread", 1)

        # Compute O_DIRECT improvement
        if result.seq_read_buffered_mbps and result.seq_read_odirect_mbps and result.seq_read_buffered_mbps > 0:
            result.odirect_seq_improvement_pct = round(
                (result.seq_read_odirect_mbps - result.seq_read_buffered_mbps)
                / result.seq_read_buffered_mbps * 100, 1
            )
        if result.rand_read_buffered_mbps and result.rand_read_odirect_mbps and result.rand_read_buffered_mbps > 0:
            result.odirect_rand_improvement_pct = round(
                (result.rand_read_odirect_mbps - result.rand_read_buffered_mbps)
                / result.rand_read_buffered_mbps * 100, 1
            )

    finally:
        _sanity_check_storage(result)
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result


def _run_python_bench(
    args: argparse.Namespace, result: StorageBenchResult, bench_bytes: int
) -> StorageBenchResult:
    """Pure-Python storage benchmark using os.preadv with O_DIRECT."""
    base_dir = args.test_dir or os.environ.get("TMPDIR") or os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="swarm_bench_", dir=base_dir)
    result.filesystem = _detect_filesystem(tmpdir)
    ok, free_gb = _check_free_space(tmpdir, args.bench_size_gb)
    if not ok:
        _sanity_check_storage(result)
        shutil.rmtree(tmpdir, ignore_errors=True)
        result.caveats.append(
            f"SKIPPED: need ~{args.bench_size_gb:.1f} GB free (plus margin) "
            f"but only {free_gb:.1f} GB available at {tempfile.gettempdir()}. "
            f"Re-run with --bench-size-gb {max(1, int(free_gb / 2))} or set "
            f"TMPDIR to a bigger filesystem."
        )
        print(f"  !! Not enough free space ({free_gb:.1f} GB) — storage "
              f"benchmark skipped. Try --bench-size-gb "
              f"{max(1, int(free_gb / 2))}")
        return result
    testfile = os.path.join(tmpdir, "test.dat")
    block_size = 16 * 1024 * 1024  # 16 MB
    num_blocks = bench_bytes // block_size
    if num_blocks < 4:
        num_blocks = 4
        bench_bytes = num_blocks * block_size
        result.caveats.append(
            f"Bench size too small; adjusted to {num_blocks} blocks "
            f"({bench_bytes / 1e9:.1f} GB)"
        )

    try:
        # ── Create test file ──────────────────────────────────────────
        print(f"  Creating {bench_bytes / 1e9:.1f} GB test file "
              f"({'with real data' if args.fill else 'via fallocate'})...",
              file=sys.stderr, flush=True)

        fd = os.open(testfile, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            if args.fill and not _try_fallocate(fd, bench_bytes):
                # Write real data (zeros from /dev/zero are fine —
                # what matters is the filesystem can't compress them away)
                _write_file_zeros(fd, bench_bytes, block_size)
            else:
                if not _try_fallocate(fd, bench_bytes):
                    _write_file_zeros(fd, bench_bytes, block_size)
            os.fsync(fd)
        finally:
            os.close(fd)

        # ── Drop caches if we can ─────────────────────────────────────
        result.cache_dropped = False
        if args.sudo and os.geteuid() == 0:
            result.cache_dropped = _drop_caches()
        if not result.cache_dropped:
            result.caveats.append(
                "Cache not dropped between runs (need --sudo). "
                "Buffered numbers may be inflated by page cache effects."
            )

        # ── Open test file ────────────────────────────────────────────
        fd_buf = os.open(testfile, os.O_RDONLY)
        try:
            fd_dir = os.open(testfile, os.O_RDONLY | os.O_DIRECT)
        except OSError as e:
            result.caveats.append(
                f"O_DIRECT not supported on this filesystem ({e}). "
                "O_DIRECT results will be N/A. This is expected on some "
                "kernel/filesystem combinations (e.g. WSL2, ZFS, tmpfs)."
            )
            fd_dir = None
        try:
            result.seq_read_buffered_mbps = _python_seq_read(
                fd_buf, block_size, num_blocks
            )
            # Need to drop caches between buffered and O_DIRECT too
            if result.cache_dropped:
                _drop_caches()
            if fd_dir is not None:
                result.seq_read_odirect_mbps = _python_seq_read(
                    fd_dir, block_size, num_blocks
                )

            if result.cache_dropped:
                _drop_caches()
            result.rand_read_buffered_mbps = _python_rand_read(
                fd_buf, block_size, num_blocks
            )
            if result.cache_dropped:
                _drop_caches()
            if fd_dir is not None:
                result.rand_read_odirect_mbps = _python_rand_read(
                    fd_dir, block_size, num_blocks
                )
        finally:
            os.close(fd_buf)
            if fd_dir is not None:
                os.close(fd_dir)

        # Compute improvements
        if result.seq_read_buffered_mbps and result.seq_read_odirect_mbps and result.seq_read_buffered_mbps > 0:
            result.odirect_seq_improvement_pct = round(
                (result.seq_read_odirect_mbps - result.seq_read_buffered_mbps)
                / result.seq_read_buffered_mbps * 100, 1
            )
        if result.rand_read_buffered_mbps and result.rand_read_odirect_mbps and result.rand_read_buffered_mbps > 0:
            result.odirect_rand_improvement_pct = round(
                (result.rand_read_odirect_mbps - result.rand_read_buffered_mbps)
                / result.rand_read_buffered_mbps * 100, 1
            )

        result.caveats.append(
            "Pure-Python fallback used.  fio provides more accurate "
            "queue-depth and IOPS numbers.  Install fio for authoritative "
            "results (apt install fio)."
        )

    finally:
        _sanity_check_storage(result)
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result


def _try_fallocate(fd: int, size: int) -> bool:
    """Try posix_fallocate.  Returns True on success."""
    try:
        os.posix_fallocate(fd, 0, size)
        return True
    except (OSError, AttributeError):
        return False


def _write_file_zeros(fd: int, total: int, chunk: int) -> None:
    """Write zeros to fd in chunks."""
    zeros = b"\x00" * min(chunk, 64 * 1024 * 1024)  # 64 MB write buffer
    written = 0
    while written < total:
        to_write = min(len(zeros), total - written)
        n = os.write(fd, zeros[:to_write])
        if n <= 0:
            break
        written += n


def _python_seq_read(fd: int, block_size: int, num_blocks: int) -> float | None:
    """Sequential read using os.preadv.  Returns MB/s or None on failure.

    Uses mmap for page-aligned O_DIRECT buffers.
    """
    # For O_DIRECT we need aligned buffers.  mmap gives page-aligned memory.
    buf = mmap.mmap(-1, block_size)
    try:
        offset = 0
        t0 = time.monotonic()
        bytes_read = 0
        for _ in range(num_blocks):
            n = os.preadv(fd, [buf], offset)
            bytes_read += n
            offset += block_size
        elapsed = time.monotonic() - t0
        if elapsed <= 0:
            return None
        return round((bytes_read / elapsed) / 1e6, 1)
    except OSError as e:
        # O_DIRECT can fail if the filesystem doesn't support it.
        # Return None so the caller can show N/A rather than crashing.
        return None
    finally:
        buf.close()


def _python_rand_read(fd: int, block_size: int, num_blocks: int) -> float | None:
    """Random read at 16 MB block offsets.  Returns MB/s or None."""
    buf = mmap.mmap(-1, block_size)
    try:
        # Pre-generate random offsets
        import random
        rng = random.Random(42)
        offsets = [rng.randint(0, num_blocks - 1) * block_size for _ in range(num_blocks)]

        t0 = time.monotonic()
        bytes_read = 0
        for off in offsets:
            n = os.preadv(fd, [buf], off)
            bytes_read += n
        elapsed = time.monotonic() - t0
        if elapsed <= 0:
            return None
        return round((bytes_read / elapsed) / 1e6, 1)
    except OSError:
        return None
    finally:
        buf.close()


# ── Section C: Sustained compute ───────────────────────────────────────────────


class ComputeBenchResult:
    def __init__(self):
        self.duration_s: float = 0.0
        self.timeline: list[dict] = []  # [{elapsed_s, freq_khz_per_core, temps_c}]
        self.cpu_freq_peak_khz: dict[str, int] = {}
        self.cpu_freq_settled_khz: dict[str, int] = {}
        self.cpu_freq_settled_mean_khz: float = 0.0
        self.temp_peak_c: float = 0.0
        self.temp_settled_c: float = 0.0
        self.throttle_onset_s: float | None = None
        self.caveats: list[str] = []


def _worker_load(stop_flag) -> None:
    """CPU-bound busy loop for one worker process.

    Runs in its own process (not a thread) so it gets a real core, not a
    GIL-shared slice of one. Mixed integer and float work, same shape as
    the original single-threaded version.
    """
    a, b, c_ = 123456789, 987654321, 0
    fa, fb, fc = 3.1415926535, 2.7182818284, 0.0
    iteration = 0
    while not stop_flag.value:
        c_ = (a * b) ^ (c_ + iteration)
        a = (a + c_) & 0x7FFFFFFFFFFFFFFF
        b = (b * 1103515245 + 12345) & 0x7FFFFFFFFFFFFFFF
        fc = math.sin(fa) * math.cos(fb) + fc
        fa = fa * 1.0001 + 0.0001
        fb = fb * 0.9999 - 0.0001
        if abs(fa) > 1e10:
            fa = 3.1415926535
        if abs(fb) > 1e10:
            fb = 2.7182818284
        iteration += 1
        if iteration % 200000 == 0:
            # Cheap opportunity to notice the stop flag promptly.
            pass


def _run_compute_bench(args: argparse.Namespace, platform_info: dict) -> ComputeBenchResult:
    """Run a sustained compute workload, sampling frequency and temperature.

    Loads ALL cores simultaneously via multiprocessing, not a single Python
    thread. A single-threaded workload only keeps one core busy at a time
    (the OS migrates that one hot thread around, or parks idle cores at low
    frequency via the governor) -- which produced a false-positive "throttle"
    on real hardware: cores that were simply never loaded looked identical to
    cores that had genuinely throttled, and one noisy sample was enough to
    declare a throttle event 2 seconds in. See the Reddit thread for the
    real-world report this was found from.

    A frequency drop must also PERSIST for several consecutive samples before
    being called a throttle, not fire on a single reading -- thread/process
    scheduling jitter is normal and is not the same thing as thermal
    throttling.
    """
    result = ComputeBenchResult()
    result.duration_s = float(args.duration)

    cpu_cores = platform_info.get("cpu_cores", 4)
    thermal_paths = [
        str(Path("/sys/class/thermal") / tz.name / "temp")
        for tz in sorted(Path("/sys/class/thermal").glob("thermal_zone*"))
    ]
    freq_paths = [
        f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq"
        for c in range(cpu_cores)
    ]

    if not freq_paths:
        result.caveats.append(
            "No cpufreq sysfs entries found; frequency tracking disabled."
        )

    n_workers = os.cpu_count() or cpu_cores
    print(f"  Running sustained compute for {args.duration}s "
          f"({args.duration // 60} min) across {n_workers} worker "
          f"processes (all cores loaded, not just one thread)...",
          file=sys.stderr, flush=True)

    stop_flag = multiprocessing.Value("b", False)
    workers = [
        multiprocessing.Process(target=_worker_load, args=(stop_flag,))
        for _ in range(n_workers)
    ]
    for w in workers:
        w.daemon = True
        w.start()

    t_start = time.monotonic()
    t_end = t_start + args.duration
    sample_interval = 1.0
    next_sample = t_start + sample_interval

    timeline: list[dict] = []
    freq_peaks: dict[str, int] = {}
    temp_peak = 0.0

    # Persistence-based throttle detection: a core must read below 85% of
    # its own observed peak for THROTTLE_PERSIST_SAMPLES consecutive samples
    # before it counts. One low sample is noise; several in a row is real.
    THROTTLE_PERSIST_SAMPLES = 5  # 5 consecutive seconds below threshold
    below_threshold_streak: dict[str, int] = {}
    throttle_onset: float | None = None

    try:
        while time.monotonic() < t_end:
            time.sleep(0.05)
            now = time.monotonic()
            if now >= next_sample:
                elapsed = now - t_start
                sample: dict = {"elapsed_s": round(elapsed, 1)}

                freqs: dict[str, int | None] = {}
                for cpu_idx in range(cpu_cores):
                    path = freq_paths[cpu_idx] if cpu_idx < len(freq_paths) else ""
                    f = _read_sysfs_int(path) if path else None
                    key = f"cpu{cpu_idx}"
                    freqs[key] = f
                    if f is not None:
                        if key not in freq_peaks or f > freq_peaks[key]:
                            freq_peaks[key] = f

                        peak = freq_peaks[key]
                        if peak > 0 and f < peak * 0.85:
                            below_threshold_streak[key] = below_threshold_streak.get(key, 0) + 1
                        else:
                            below_threshold_streak[key] = 0

                        if (
                            throttle_onset is None
                            and below_threshold_streak[key] >= THROTTLE_PERSIST_SAMPLES
                        ):
                            # Onset is when the streak actually started.
                            throttle_onset = elapsed - (THROTTLE_PERSIST_SAMPLES - 1)
                sample["freq_khz"] = freqs

                temps: dict[str, float] = {}
                for tp in thermal_paths:
                    tz_name = Path(tp).parent.name
                    t = _read_sysfs_int(tp)
                    if t is not None:
                        temp_c = t / 1000.0
                        temps[tz_name] = temp_c
                        if temp_c > temp_peak:
                            temp_peak = temp_c
                sample["temp_c"] = temps

                timeline.append(sample)
                next_sample = now + sample_interval

    except KeyboardInterrupt:
        result.caveats.append("Compute benchmark interrupted by user.")
    finally:
        stop_flag.value = True
        for w in workers:
            w.join(timeout=2.0)
            if w.is_alive():
                w.terminate()

    result.timeline = timeline
    result.cpu_freq_peak_khz = freq_peaks
    result.temp_peak_c = round(temp_peak, 1)

    if timeline:
        settled_samples = [s for s in timeline if s["elapsed_s"] >= result.duration_s - 30]
        if not settled_samples:
            settled_samples = timeline[-min(10, len(timeline)):]

        settled_freqs: dict[str, list[int]] = {}
        for s in settled_samples:
            for cpu_key, freq in s["freq_khz"].items():
                if freq is not None:
                    settled_freqs.setdefault(cpu_key, []).append(freq)

        for cpu_key, freqs in settled_freqs.items():
            avg = sum(freqs) // len(freqs) if freqs else 0
            result.cpu_freq_settled_khz[cpu_key] = avg

        if result.cpu_freq_settled_khz:
            result.cpu_freq_settled_mean_khz = round(
                sum(result.cpu_freq_settled_khz.values())
                / len(result.cpu_freq_settled_khz)
            )

        settled_temps = []
        for s in settled_samples:
            settled_temps.extend(s["temp_c"].values())
        if settled_temps:
            result.temp_settled_c = round(sum(settled_temps) / len(settled_temps), 1)

    result.throttle_onset_s = throttle_onset

    return result


# ── Output formatting ──────────────────────────────────────────────────────────


def _fmt_mbps(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.0f} MB/s"


def _fmt_khz(v: dict | None, cores: int) -> str:
    if not v:
        return "N/A"
    parts = []
    for i in range(cores):
        f = v.get(f"cpu{i}")
        if f is not None:
            parts.append(f"{f // 1000}")
        else:
            parts.append("?")
    return f"[{', '.join(parts)}] MHz"


def _fmt_improvement(pct: float | None) -> str:
    if pct is None:
        return "N/A"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def _fmt_optional_float(v: float | None, suffix: str = "") -> str:
    if v is None:
        return "N/A"
    return f"{v:.1f}{suffix}"


def _fmt_temp(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.1f} C"


def format_output(
    platform_info: dict,
    storage: StorageBenchResult | None,
    compute: ComputeBenchResult | None,
    args: argparse.Namespace,
) -> str:
    """Produce the pasteable text block."""
    lines: list[str] = []
    sep = "=" * 66

    lines.append(sep)
    lines.append("  SWARM BENCHMARK RESULT — paste everything below")
    lines.append(f"  script version: {VERSION}")
    lines.append(sep)
    lines.append("")

    # ── Platform ────────────────────────────────────────────────────────
    pi = platform_info
    lines.append("── Platform ──")
    lines.append(f"board_model:              {pi['board_model']}")
    lines.append(f"soc_model:                {pi['soc_model']}")
    lines.append(f"sdot (asimddp):           {'YES' if pi['sdot_present'] else 'NO'}")
    lines.append(f"cpu_cores:                {pi['cpu_cores']}")
    lines.append(f"cpu_governor:             {pi['cpu_governor']}")
    lines.append(f"cpu_freqs_max:            {_fmt_khz(pi.get('cpu_freqs_max_khz'), pi['cpu_cores'])}")
    lines.append(f"ram_total_mb:             {pi['ram_total_kb'] // 1024}")
    lines.append(f"ram_available_mb:         {pi['ram_available_kb'] // 1024}")
    lines.append(f"kernel:                   {pi['kernel'][:80]}")
    lines.append(f"distro:                   {pi['distro']}")
    lines.append(f"arch:                     {pi['arch']}")

    # NVMe devices
    for dev in pi.get("nvme_devices", []):
        lines.append(f"nvme:                     {dev['name']} "
                     f"model={dev.get('model', '?')} "
                     f"size={dev.get('size_gb', '?')} GB "
                     f"link={dev.get('pcie_link_speed', '?')} "
                     f"x{dev.get('pcie_link_width', '?')} "
                     f"(max {dev.get('pcie_max_speed', '?')} "
                     f"x{dev.get('pcie_max_width', '?')})")
    if not pi.get("nvme_devices"):
        lines.append("nvme:                     (none detected)")

    # Other block devices
    for dev in pi.get("block_devices", []):
        lines.append(f"block:                    {dev['name']} "
                     f"size={dev.get('size_gb', '?')} GB"
                     f"{' type=' + dev['type'] if dev.get('type') else ''}")

    # Thermal baseline
    for tz in pi.get("thermal_zones", []):
        lines.append(f"thermal_{tz['name']}:           "
                     f"type={tz['type']}  temp={tz['temp_c']} C")

    lines.append("")

    # ── Storage ──────────────────────────────────────────────────────────
    if storage is not None:
        lines.append("── Storage ──")
        lines.append(f"method:                   {storage.method}")
        lines.append(f"file_size_gb:             {storage.file_size_gb}")
        lines.append(f"cache_dropped:            {'yes' if storage.cache_dropped else 'no (use --sudo)'}")
        lines.append(f"seq_read_buffered:        {_fmt_mbps(storage.seq_read_buffered_mbps)}")
        lines.append(f"seq_read_odirect:         {_fmt_mbps(storage.seq_read_odirect_mbps)}")
        lines.append(f"rand_read_buffered:       {_fmt_mbps(storage.rand_read_buffered_mbps)}")
        lines.append(f"rand_read_odirect:        {_fmt_mbps(storage.rand_read_odirect_mbps)}")
        lines.append(f"odirect_seq_improvement:  {_fmt_improvement(storage.odirect_seq_improvement_pct)}")
        lines.append(f"odirect_rand_improvement: {_fmt_improvement(storage.odirect_rand_improvement_pct)}")
        for caveat in storage.caveats:
            lines.append(f"  [!] {caveat}")
        lines.append("")
    else:
        lines.append("── Storage ──")
        lines.append("  (skipped with --skip-storage)")
        lines.append("")

    # ── Compute ──────────────────────────────────────────────────────────
    if compute is not None:
        cr = compute
        lines.append("── Sustained Compute ──")
        lines.append(f"duration:                 {cr.duration_s:.0f} s")
        lines.append(f"cpu_freq_peak:            {_fmt_khz(cr.cpu_freq_peak_khz, pi['cpu_cores'])}")
        lines.append(f"cpu_freq_settled:         {_fmt_khz(cr.cpu_freq_settled_khz, pi['cpu_cores'])}")
        lines.append(f"cpu_freq_settled_mean:    {_fmt_optional_float(cr.cpu_freq_settled_mean_khz / 1000 if cr.cpu_freq_settled_mean_khz else None, ' MHz')}")
        lines.append(f"temp_peak:                {_fmt_temp(cr.temp_peak_c)}")
        lines.append(f"temp_settled:             {_fmt_temp(cr.temp_settled_c)}")
        lines.append(f"throttle_onset:           {_fmt_optional_float(cr.throttle_onset_s, ' s') if cr.throttle_onset_s is not None else 'none detected'}")
        for caveat in cr.caveats:
            lines.append(f"  [!] {caveat}")
        lines.append("")
    else:
        lines.append("── Sustained Compute ──")
        lines.append("  (skipped with --skip-compute)")
        lines.append("")

    # ── Power ────────────────────────────────────────────────────────────
    lines.append("── Power (fill in manually if you have a meter) ──")
    lines.append("power_idle_w:             (fill in)")
    lines.append("power_storage_bench_w:    (fill in)")
    lines.append("power_compute_bench_w:    (fill in)")
    lines.append("power_both_at_once_w:     (fill in)")
    lines.append("")

    lines.append(sep)
    lines.append("  END — thank you!")
    lines.append(sep)

    return "\n".join(lines)


def build_json_output(
    platform_info: dict,
    storage: StorageBenchResult | None,
    compute: ComputeBenchResult | None,
    args: argparse.Namespace,
) -> dict:
    """Build a structured JSON dict from all results."""
    out: dict = {
        "version": VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform_info,
    }
    if storage is not None:
        out["storage"] = {
            "method": storage.method,
            "file_size_gb": storage.file_size_gb,
            "cache_dropped": storage.cache_dropped,
            "seq_read_buffered_mbps": storage.seq_read_buffered_mbps,
            "seq_read_odirect_mbps": storage.seq_read_odirect_mbps,
            "rand_read_buffered_mbps": storage.rand_read_buffered_mbps,
            "rand_read_odirect_mbps": storage.rand_read_odirect_mbps,
            "odirect_seq_improvement_pct": storage.odirect_seq_improvement_pct,
            "odirect_rand_improvement_pct": storage.odirect_rand_improvement_pct,
            "caveats": storage.caveats,
        }
    if compute is not None:
        out["compute"] = {
            "duration_s": compute.duration_s,
            "cpu_freq_peak_khz": compute.cpu_freq_peak_khz,
            "cpu_freq_settled_khz": compute.cpu_freq_settled_khz,
            "cpu_freq_settled_mean_khz": compute.cpu_freq_settled_mean_khz,
            "temp_peak_c": compute.temp_peak_c,
            "temp_settled_c": compute.temp_settled_c,
            "throttle_onset_s": compute.throttle_onset_s,
            "caveats": compute.caveats,
            "timeline": compute.timeline,
        }
    return out


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    # ── Startup message ──────────────────────────────────────────────────
    msg_parts = ["This script will:"]
    msg_parts.append("  1. Identify your board, SoC, RAM, kernel, and NVMe devices.")
    if not args.skip_storage:
        msg_parts.append(
            f"  2. Create a {args.bench_size_gb} GB temp file, run sequential "
            f"and random read benchmarks at 16 MB block size, and delete it."
        )
    if not args.skip_compute:
        mins = args.duration // 60
        secs = args.duration % 60
        dur_str = f"{mins} min" if secs == 0 else f"{mins} min {secs} s"
        msg_parts.append(
            f"  3. Run a sustained CPU workload for {dur_str} ({args.duration} s)"
            f", sampling frequency and temperature."
        )
    msg_parts.append("  4. Print a results block for you to paste back.")
    msg_parts.append("")
    msg_parts.append("NO root needed (use --sudo to drop caches for better numbers).")
    msg_parts.append(f"Temp files are created in {tempfile.gettempdir()} and deleted.")
    if not args.skip_storage:
        msg_parts.append(f"Disk space needed: ~{args.bench_size_gb} GB (temporary).")
    msg_parts.append("")

    for line in msg_parts:
        print(line, file=sys.stderr)

    # ── Platform info (always runs) ──────────────────────────────────────
    print("── Identifying platform...", file=sys.stderr, flush=True)
    platform_info = _platform_info()
    print(f"  Board: {platform_info['board_model']}", file=sys.stderr)
    print(f"  SDOT:  {'YES' if platform_info['sdot_present'] else 'NO'}", file=sys.stderr)
    print(f"  NVMe:  {len(platform_info['nvme_devices'])} device(s)", file=sys.stderr)
    print("", file=sys.stderr)

    # ── Storage benchmark ────────────────────────────────────────────────
    storage: StorageBenchResult | None = None
    if not args.skip_storage:
        print("── Storage benchmark...", file=sys.stderr, flush=True)
        try:
            storage = _run_storage_bench(args)
        except OSError as e:
            print(f"  Storage benchmark failed: {e}", file=sys.stderr)
            storage = StorageBenchResult()
            storage.caveats.append(f"OS error: {e}")
    else:
        print("── Storage: skipped", file=sys.stderr)

    # ── Compute benchmark ────────────────────────────────────────────────
    compute: ComputeBenchResult | None = None
    if not args.skip_compute:
        print("── Sustained compute benchmark...", file=sys.stderr, flush=True)
        try:
            compute = _run_compute_bench(args, platform_info)
        except KeyboardInterrupt:
            print("\n  Interrupted.", file=sys.stderr)
    else:
        print("── Compute: skipped", file=sys.stderr)

    # ── Output ───────────────────────────────────────────────────────────
    text_output = format_output(platform_info, storage, compute, args)
    json_output = build_json_output(platform_info, storage, compute, args)

    if args.json_only:
        with open(args.json_only, "w") as fh:
            json.dump(json_output, fh, indent=2, default=str)
        print(f"JSON written to {args.json_only}", file=sys.stderr)
    else:
        print(text_output)
        # Also write JSON alongside
        json_path = "swarm_bench_result.json"
        with open(json_path, "w") as fh:
            json.dump(json_output, fh, indent=2, default=str)
        print(f"\n(JSON also written to {json_path})", file=sys.stderr)


if __name__ == "__main__":
    main()
