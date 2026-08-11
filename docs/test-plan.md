# Test Plan

### Validating the architecture on an RK3588 SBC before committing to a PCB

The board design rests on nine unknowns listed in [architecture.md](architecture.md) section 9. Twelve of the fifteen questions worth answering can be resolved with one or two development kits for roughly \$300, on the same silicon the production node uses.

**This document exists so the work is ready the day hardware arrives**, rather than being planned then.

---

## 1. The hardware

**Any RK3588 single-board computer**, roughly \$100 to \$150. Orange Pi 5 Plus, Radxa Rock 5B, FriendlyElec NanoPC-T6 and similar all use the same silicon as the target module, and people already publish llama.cpp benchmarks on them.

What matters for these tests: an **M.2 slot on PCIe 3.0 x4**, at least 8 GB of RAM, and a mainline-ish Linux image (Armbian, Debian, Ubuntu).

| Item | Cost | Purpose |
|---|---|---|
| RK3588 SBC, 8 or 16 GB | ~\$120 | the platform |
| NVMe 512 GB Gen3, DRAM cache | ~\$112 | model storage and the bandwidth measurement |
| Power meter | ~\$15 | unknown 5 |
| Ethernet cable | ~\$5 | two-board tests |
| **Second SBC** | ~\$120 | everything in section 4 |

**First purchase: ~\$250.**

---

## 2. Step 0: the experiment that needs no hardware — done

**Does speculative decoding pay on a streaming MoE? Resolved: no.** Measured across 12 seeds on OLMoE-1B-7B, net loss at every tested batch size. See [architecture.md](architecture.md) section 6.5 for the full result. The method below is kept as a record of how it was answered.

Section 6.5 of [architecture.md](architecture.md) argues it may be a net loss, because verifying B draft tokens requires loading the union of the experts those tokens route to. Under independent routing on OLMoE (64 experts, top-8), 4 draft tokens touch **3.31x** the experts of 1 token, against an acceptance of perhaps 2.2. (An earlier revision of this document said 3.81x; that was wrong — see the note under the results table.)

**The open question is routing correlation**, and it is measurable on any machine with any MoE model, today, for nothing.

**Method.** Instrument a small MoE (OLMoE at 7B, or Qwen3.6-35B-A3B) to log the routed expert indices per layer per token across a few thousand tokens of realistic text. Then, for consecutive token windows of size 2, 4 and 8, compute the size of the expert union against the independent-sampling prediction.

| Result | Consequence |
|---|---|
| Union near independent prediction | **Disable speculative decoding.** Remove the multiplier from every figure. |
| Union 30 to 50% below prediction | Marginal, roughly 1.2x. Worth having, not worth optimising for. |
| Union far below prediction | Speculation pays. Restore the 1.5x to 2x figures. |

**Also worth extracting from the same trace:** the expert popularity distribution, which determines whether LRU caching is worth 5% or 15% (dial 14). GLM-5.2 uses Quantile Balancing during training, which may deliberately flatten it. **Still open.**

### 2.1 Step 0b: does request batching pay? — done

The same machinery answers a second, opposite question. Speculative decoding batches *consecutive tokens from one stream*; request batching groups *one token each from independent requests*. Same union cost, different token yield — B tokens instead of an acceptance rate — so the economics could well come out the other way.

**Method.** `cross_request_routing_experiment.py`: generate from 16 deliberately unrelated prompts, then compute the expert union across all C(16,B) combinations of *different* prompts at the same generation position. The consecutive condition is recomputed in the same run from the same traces, so the two are directly comparable.

**Result (seed 42, 8-bit, results in `cross_request_seed42_v2/`):**

| Condition | B=2 | B=4 | B=8 |
|---|---|---|---|
| Consecutive, same stream | 0.858 | 0.768 | 0.729 |
| Cross-request, independent | **0.975** | **0.947** | **0.911** |

Cross-request routing is close to fully independent — the correlation that killed speculative decoding is a context effect, and it disappears between unrelated prompts. On OLMoE this still yields a 1.28x net gain at B=4, but **that does not transfer to GLM-5.2**, whose 256-expert pool at the same top-8 makes collisions rare: the projected gain there is ~1.11x. See [dials.md](dials.md) dial 3 sections 3.1–3.3 for the full reasoning.

**Verdict: batch 1, no scheduler, for the first runtime.**

Three experiments, one trace, zero dollars, and they set the performance figures for the entire project.

---

## 3. Single-board sequence

### Step 1: Boot and baseline

Flash Armbian or the vendor image, boot, get a shell. Record kernel version, `/proc/cpuinfo`, `free -h`, `lsblk`.

**Confirm SDOT is present:**

```
grep asimddp /proc/cpuinfo
```

On the A76 this should be **non-empty**, unlike the Armv8.0 part this design previously targeted. That single line confirms llama.cpp will use the fast quantized kernels.

### Step 2: Thermal discipline, before any benchmark

Four A76 cores under sustained matmul will throttle on a passively cooled SBC. **Every compute measurement must be sustained, not burst.** Run at least ten minutes and log frequency alongside throughput:

```
watch -n1 'cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; \
           cat /sys/class/thermal/thermal_zone*/temp'
```

Record both the peak and the settled figure. **The settled figure is what the board design rests on.** A heatsink and fan are worth having before starting.

### Step 3: Compute throughput

```
llama-bench -m model.gguf -p 512 -n 128
```

Run on CPU with NEON, then on the GPU via OpenCL or Vulkan if llama.cpp builds against Panfrost, then the NPU if an RKNN path exists.

Convert to effective GFLOPS and compare against **~14 GFLOPS**, which is what consuming 4.0 GB/s of Q4 weights requires. Expect the A76 cluster to clear this comfortably.

**The comparison that matters is compute against storage**, not backend against backend. Whichever is smaller is the real bottleneck, and the architecture assumes it is storage.

### Step 4: Storage characterisation

The most load-bearing measurement in the project.

```
fio --name=seq --rw=read --bs=16M --size=8G --direct=1 --ioengine=libaio --iodepth=4
fio --name=rand --rw=randread --bs=16M --size=8G --direct=1 --ioengine=libaio --iodepth=4
```

Run against the NVMe on the x4 link, and against the onboard eMMC for comparison.

**Test with and without `--direct=1`.** Colibri's benchmarks report O\_DIRECT giving +34% decode and 4.25 to 9.69 GB/s in their own storage benchmark, because bypassing the page cache avoids double-buffering weights that are read once.

Expected: ~3.2 GB/s sequential on the x4 link. If it comes in materially lower, every timing figure needs revision.

### Step 5: Power

With a meter inline, record wattage at idle, CPU-only inference, GPU inference, storage benchmark, and everything at once.

**This closes an unknown that has been an estimate for the entire project.**

### Step 6: Expert streaming on a real MoE

Build [Colibri](https://github.com/JustVugg/colibri) and port from x86 AVX2 to ARM NEON as needed.

**Start with OLMoE** (7B total, 1B active, ~4 GB at Q4). Colibri ships a dedicated engine for it and it fits comfortably. Then **Qwen3.6-35B-A3B** (~20 GB) from the NVMe.

Record: tokens/second, measured bytes moved per token, and how close the ratio comes to `bytes_per_token / storage_bandwidth`.

**That ratio is the whole thesis.** If measured throughput tracks storage bandwidth, the architecture is validated. If not, something else is binding.

---

## 4. Two-board sequence

This is where the actual project starts. Everything above is characterisation; this is the runtime.

### Step 7: Transport

Connect the two boards directly by Ethernet cable, using the integrated switch ports. Measure with `iperf3`: throughput, latency, jitter.

Confirm the negotiated rate and whether the integrated switch introduces measurable overhead versus a direct MAC-to-MAC path.

**Expected result, for comparison against measurement.** At 6144 hidden dimensions and FP16 activations, a layer boundary moves ~12 KB per token.

| Workload | Per layer | Total | At 125 MB/s |
|---|---|---|---|
| Decode, 1 token, 75 layers | 12 KB | ~900 KB | **~7.2 ms**, about 1% of a 0.78 s token |
| Prefill, 100k tokens | 1.2 GB | ~90 GB | **~12 minutes** |

**Decode is unaffected by the 1 GbE link. Prefill is bounded by it.** That asymmetry is why long-prompt processing is the design's worst workload and why the ideal-node specification calls for 10 GbE.

### Step 8: Layer sharding

Split a dense model's layers across two nodes as a pipeline. Measure the cost of the activation handoff (~12 KB per layer boundary in theory) and confirm it is negligible against per-layer compute.

### Step 9: Expert sharding

Split OLMoE's experts across two nodes in gang mode: both nodes work the same layer, each holding half the experts, in lockstep.

Measure the load imbalance factor. Theory says the slowest node gates the layer and expected worst-case load is 1.5 to 2x the mean. **Measure the real number** and record whether it changes with batch size.

### Step 10: Weighted sharding

Deliberately unbalance the nodes: run one from the NVMe on the x4 link and one from the onboard eMMC.

Confirm the runtime assigns experts proportionally to measured bandwidth rather than equally, as required by [compatibility.md](compatibility.md) section 3.1.

**This is the test that validates the cross-generation compatibility contract**, and it is the reason weighted sharding has to be in the runtime from the first line rather than added later.

---

## 5. What this produces

**If step 2 passes and step 8 tracks theory:** the architecture is validated on real silicon, and the remaining work is a PCB plus a runtime that already has its hard parts proven.

**If step 5 or 6 comes in far below assumption:** every timing figure in the project needs revision, and it is far better to learn that for \$235 than after a board order.

**If step 2 fails:** the project needs a different chip or a from-scratch inference engine, and \$148 has saved a year.

Steps 7 through 10 are the beginning of the distributed runtime, which is **hardware-agnostic and the actual long pole.** Whatever silicon a future tier uses, that software carries over unchanged. Two dev kits and an Ethernet cable is the entire development environment for it.

---

## 6. Results

*(To be filled in as tests are run. Record measured values, not impressions.)*

| Step | Measurement | Assumed | Measured | Date |
|---|---|---|---|---|
| 0 | **Expert union at 4 draft tokens** | 3.31x independent* | 2.20x–2.59x measured, 12 seeds | done |
| 0 | **Speculative decoding net gain** | unknown, may be < 1 | **net loss confirmed, all seeds, all B** | done |
| 0 | **Cross-request union ratio, B=4** | unknown | **0.947 of independent** (16 prompts, seed 42) | done |
| 0 | **Batching net gain, B=4, OLMoE** | unknown | **1.28x measured** | done |
| 0 | **Batching net gain, B=4, GLM-5.2** | unknown | **~1.11x projected** (k/E differs, see dials §3.2) | projected |
| 0 | Expert popularity skew, top 5% share | unknown | | |
| 1 | SDOT present | yes | | |
| 2 | Sustained CPU frequency under load | 2.4 GHz nominal | | |
| 3 | Tokens/s, small dense model, best backend | | | |
| 3 | Effective GFLOPS vs the ~14 needed | comfortable | | |
| 4 | **NVMe sequential read, 16 MB blocks** | **~3.2 GB/s** | | |
| 4 | NVMe random read, 16 MB blocks | unknown | | |
| 4 | O\_DIRECT improvement | ~34% | | |
| 5 | Power, loaded | ~22 W/node | | |
| 6 | OLMoE tokens/s | | | |
| 6 | **Measured bytes/token vs theory** | | | |
| 7 | Inter-node throughput | 125 MB/s | | |
| 9 | Load imbalance factor | 1.5x to 2x | | |

*The 3.81x figure previously in this row was wrong. For OLMoE (64 experts, top-8) the independent-sampling union at B=4 is 26.5 experts, i.e. **3.31x** the 8 read by a single token. Corrected against the formula `E × (1 − (1 − k/E)^B)` and confirmed empirically: synthetic fully-independent traces reproduce the prediction to within 0.1%.

---

*Companion to [architecture.md](architecture.md), [dials.md](dials.md), [compatibility.md](compatibility.md) and [ideal-node.md](ideal-node.md). Hardware pricing reflects Mouser Canada, August 2026.*
