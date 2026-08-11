# Swarm Benchmark — run this on your RK3588 board

## What's this?

I'm designing an open-hardware board that runs large AI models (like GLM-5.2 at 744 billion parameters) on cheap ARM processors and NVMe drives instead of GPUs. It's design-stage — nothing has been built yet, and every number in the project is derived from datasheets and arithmetic, not from measurement.

**Your board can answer the two biggest unknowns without me spending a dollar.** Specifically: how fast does NVMe storage *actually* read on RK3588 silicon, and what happens to CPU speed after 10 minutes of sustained load?

## What the script does

A single Python file, no dependencies. It:

1. **Identifies your board** — model, SoC, RAM, kernel, NVMe link speed
2. **Measures storage speed** — sequential and random reads at 16 MB block size (the chunk size the project reads expert tensors in), with and without O_DIRECT. Uses `fio` if installed; falls back to a pure-Python method if not.
3. **Runs a sustained CPU workload** for 10 minutes (configurable), sampling frequency and temperature every second. Reports both peak and settled figures.
4. **Prints a results block** for you to paste back.

## Safety

- **Read-only for your data.** The script creates one big temp file, reads from it, and deletes it. It never touches anything outside its temp directory.
- **No root needed.** Without root, cache effects may inflate the buffered-read numbers (the O_DIRECT numbers are unaffected). Pass `--sudo` if you want it to drop caches between runs — it won't do anything else.
- **Standard library only.** No `pip install`, no numpy, no dependencies. Runs on Python 3.8+.

## Two things that will waste your time if you miss them

**1. The test file must be on the NVMe you want measured.** By default it
goes in `/tmp`, which on some systems is a RAM disk — you'll get numbers like
20,000 MB/s, which is your memory, not your drive. The script now detects
this and tells you, but you can avoid it up front:

```bash
TMPDIR=/path/on/your/nvme python3 swarm_bench.py
```

**2. Install `fio` if you can** — `sudo apt install fio`. The pure-Python
fallback works, but it reads one block at a time (queue depth 1), which
significantly under-reports random-read performance on NVMe, since these
drives depend on command queuing. The sequential number is closer to right.
If you only do one thing, do this one.

## How to run

```bash
# Download
wget https://raw.githubusercontent.com/Techburst36/swarm/main/swarm_bench.py

# Run everything (takes ~15 minutes — mostly the 10-min compute test)
python3 swarm_bench.py

# Quick version: storage only, 2-minute compute
python3 swarm_bench.py --duration 120

# With root for better cache-drop accuracy
sudo python3 swarm_bench.py --sudo

# Skip the storage benchmark (platform + compute only)
python3 swarm_bench.py --skip-storage

# Smaller test file if you're low on space
python3 swarm_bench.py --bench-size-gb 4
```

The script tells you upfront what it will do, how long it takes, and how much disk space it needs.

## What to paste back

At the end, you'll get a block like this:

```
==================================================================
  SWARM BENCHMARK RESULT — paste everything below
==================================================================
...
==================================================================
  END — thank you!
==================================================================
```

Paste the whole thing. If you have a wall power meter or USB-C meter, the power fields at the bottom are gold — fill them in if you can.

The script also writes `swarm_bench_result.json` with the same data in machine-readable form.

## Which boards work

Any RK3588 board with an NVMe slot: Orange Pi 5 Plus, Radxa Rock 5B, FriendlyElec NanoPC-T6, NanoPi R6C, and similar. An eMMC-only board still produces useful platform and compute data.

Other ARM boards (RK3568, RK3399, Allwinner H618) mostly work too — the platform identification and compute benchmark are chip-agnostic. The storage numbers are the main thing I'm after on RK3588 specifically.

## Why your number matters

The entire board design currently assumes ~3.2 GB/s sequential read from a PCIe 3.0 x4 NVMe drive. If real hardware gets 2.1 GB/s, every timing figure in the project is wrong by 50%. That's a much better thing to learn from a stranger's $120 SBC than from a $2,400 board order.

The sustained-compute numbers tell me whether the board needs active cooling (fan + heatsink) or whether a passive heatsink is enough — that changes the enclosure design and per-blade cost.

## Thank you

Seriously. This project is open hardware (CERN-OHL-P) and open documentation (CC-BY-4.0). Your measurement is more useful than anything I could produce by reasoning from datasheets.
