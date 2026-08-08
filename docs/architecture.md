# Architecture

A distributed inference board for frontier-scale Mixture-of-Experts models.

**Design stage.** No number in this document is anchored to a measurement on real silicon. Figures derive from vendor datasheets, published model configs, and arithmetic. Several have been revised more than once.

---

## 1. The premise

Mixture-of-Experts decode at batch 1 is memory-bandwidth-bound.

GLM-5.2 has 744B total parameters and activates roughly 40B per token: 8 routed experts out of 256, per layer, across 75 MoE layers. Every one of those weights is read once, used once, and discarded. Arithmetic intensity is around 2 FLOPs per byte.

A GPU is thousands of multipliers sharing one memory bus. It wins whenever weights can be fed to those multipliers repeatedly from cache. At 2 FLOPs/byte there is nothing to reuse, so the multipliers idle and the bus is the only thing that matters.

This design inverts the ratio: **many modest processors, each with private storage channels.** Aggregate bandwidth scales with node count instead of being fixed at purchase. A GPU's 1.8 TB/s does not grow when you attach more drives; this does.

### 1.1 The general law

> **Good wherever weights are used once and discarded. Bad wherever weights are reused heavily.**

| Workload | Weight reuse | Fit |
|---|---|---|
| MoE decode, batch 1 | none | best case |
| Dense LLM decode | none | good |
| Heterogeneous multi-model fleet | none | excellent |
| LLM prefill (long prompts) | heavy across prompt | poor |
| Diffusion, convolutions | massive spatial reuse | poor |
| Video DiT | reuse + quadratic attention | hopeless |
| Training | n/a (INT8 NPU, no backward pass) | impossible |

Every fit and every disqualification below follows from that line.

### 1.2 Where this loses

Stated up front so the document isn't self-flattering:

- **Anything that fits in VRAM.** A single RTX 5090 running a 32B model at Q4 does perhaps 1,000–2,000 tok/s aggregate. Twelve boards do far less, for more money, in a system with vastly more parts.
- **Edge inference.** A Jetson Orin Nano is $249, draws 15 W, and works out of the box.
- **Latency.** The architecture trades latency for throughput at every level.
- **Software maturity.** Everything here needs a runtime that does not exist yet.

The niche is narrow and real: **models whose weights are enormous but whose active compute per token is small.** That is precisely what MoE is, and precisely where the open-weight frontier went.

---

## 2. The node: Octavo OSD32MP2

Selected after evaluating and rejecting roughly nineteen alternatives; see [chip-selection.md](chip-selection.md) for the trail and the reasoning behind each rejection.

A System-in-Package integrating STMicroelectronics' STM32MP257, DDR4, an STPMIC2 power management IC, EEPROM, oscillators and passives into a single 21×21 mm BGA.

| Spec | Value |
|---|---|
| Package | 21×21 mm, 437 ball, **1.0 mm pitch** |
| CPU | 2× Cortex-A35 @ 1.5 GHz with NEON, 512 KB L2 |
| Coprocessor | Cortex-M33 @ 400 MHz |
| NPU | 1.35 TOPS |
| GPU | **Vulkan 1.1, OpenCL 1.2, OpenGL ES 3.1** |
| Memory | DDR4, integrated (**capacity unconfirmed; see §9**) |
| Storage | **3× SDMMC** (SD/eMMC/SDIO, 8-bit), 2× Octal SPI, 8× SPI |
| Networking | **3× Gigabit Ethernet with 2+1 integrated switch**, TSN, IEEE 1588v2 |
| PCIe | 1× Gen2, embedded 5 Gbit/s PHY |
| USB | 1× 2.0 HS host, 1× 2.0/3.0 DRD (5 Gbit/s PHY **shared with PCIe**) |
| Video | 1080p60 H.264 encode/decode |
| Temperature | −40 to +85 °C case |

**Why this rather than the bare STM32MP257 at a third the price:** the 1.0 mm pitch makes a 4-layer board routable and hand-assemblable, the integrated DDR4 removes nine fly-by memory routes, and the integrated PMIC removes a 24-rail power distribution problem. Those three risks dominate a first BGA project. The bare chip is a version-two option and the software carries over unchanged.

**Why the GPU matters more than the NPU:** Vulkan 1.1 means llama.cpp's Vulkan backend has a real chance of running. That reduces the software task from *write a distributed MoE runtime from scratch against an undocumented accelerator* to *port an existing engine and add multi-node RPC*. This was the single largest risk reduction in the entire chip evaluation.

---

## 3. Interconnect

Each node has three Gigabit Ethernet interfaces with an integrated 2+1 switch. Nodes chain **MAC-to-MAC over RGMII**, direct copper between adjacent nodes, no PHY chips, no magnetics, no external switch IC.

| Property | Value |
|---|---|
| Per-link throughput | ~125 MB/s |
| Latency | microseconds |
| Topology | linear chain via integrated switches; third port free for uplink |
| Signalling | RGMII, 12 signals per direction at 125 MHz |

One PHY plus magnetics per board handles the outside world. Board-to-board runs as more RGMII over the stacking header, or over ordinary Ethernet cabling if the stack is spread out for thermal reasons.

**Consequence:** a switched fabric gives any-to-any connectivity, so tensor-parallel groups are not constrained by physical adjacency and a single dead node does not partition the network.

---

## 4. Board

### 4.1 Form factor

100×100 mm, 4 layers, double-sided assembly. 150×150 mm is under consideration, the PCB is roughly 0.2% of BOM cost, so the larger size buys routing headroom and easier bring-up for about $25.

### 4.2 Population

Nine nodes in a 3×3 grid. Storage on the reverse face.

| Interface | Attached per node | Bandwidth | Capacity |
|---|---|---|---|
| SDMMC ×3 | 3× eMMC 32 GB | 840 MB/s | 96 GB |
| PCIe Gen2 ×1 | 1× BGA NVMe | ~500 MB/s | 256 GB–1 TB |
| Octal SPI ×2 | 2× Octal NAND 4 Gb | ~400 MB/s | 1 GB |
| **Per node** | **6 devices** | **~1.74 GB/s** | **~350 GB–1.1 TB** |

Per board: **54 storage devices, ~15.7 GB/s aggregate, ~3.2 TB.**

The USB 3.0 port is unavailable in this configuration, its 5 Gbit/s PHY is shared with PCIe, and NVMe is the better use. USB 2.0 HS remains for console and debug.

### 4.3 eMMC density is a bandwidth decision, not a capacity one

Sequential read by density, eMMC 5.1, 8-bit bus, HS400 mode:

| Density | Sequential read |
|---|---|
| ≤16 GB | 160 MB/s |
| ≥32 GB | 280 MB/s |

Below 32 GB there are not enough NAND dies to interleave and the interface idles. **HS400 support does not deliver HS400 speed on small parts.** 32 GB modules are therefore mandatory, and the resulting capacity over-provisioning (about 2 TB against a 419 GB model) is not waste, it funds parked-session KV, a local corpus, a LoRA library, and a resident draft model.

### 4.4 Power distribution

Per-board DC input, **12 V distributed, regulated to 5 V on board.** At 12 V a board's ~65 W is 5.4 A rather than 13 A, which halves conductor cross-section and IR drop.

Daisy-chaining power through a stack does not work: twelve boards at 5 V would put 156 A through the bottom board's connector, and that board would dissipate everyone else's conduction loss. Per-board input also gives hot-swap, independent fault isolation, and per-board current measurement.

Metal standoffs through the mounting holes tie board grounds together, which RGMII signal integrity wants anyway.

### 4.5 Design-for-bring-up

Roughly $45 of parts that either save debugging time or unlock later capability:

- **8 M3 mounting holes**, corners plus side midpoints, for stacking geometry and a ground path
- **PCIe routed to the header** even if unpopulated, free now, impossible later
- **INA219 per node on I2C**, power consumption is currently unmeasured, and this is how that stops being true
- **Test points on every rail and RGMII pair**
- **Boot-select jumper per node**, recover a node without reflowing it
- **Status LED per node**
- **Board ID EEPROM**, stack addressing without DIP switches
- **Fan header and standoff airgap**, 65 W in a stack needs real airflow, not flush stacking

---

## 5. Dense models

With 2 GB per node, a transformer layer of essentially any dense model fits on a single node. Tensor parallelism therefore becomes a **speed choice rather than a capacity requirement**, the opposite of the situation on smaller-memory parts.

| Model | Layers | Layer @Q4_K_M | Nodes at TP=1 | Boards |
|---|---|---|---|---|
| Qwen2.5-32B | 64 | ~274 MB | 64 | 8 |
| Llama 3.1-70B | 80 | ~481 MB | 80 | 9 |
| Llama 3.1-405B | 126 | ~1.47 GB | 126 | 14 |

**The partitioning rule:** TP degree is bounded by all-reduce cost. On RGMII a 3-node group costs roughly 0.2 ms per all-reduce, four times per layer. Past about 4–6 nodes per group, sync overhead outweighs the reduction in per-node weight read. Everything beyond that goes to pipeline depth.

**Single-query** latency is one token walking every stage. **Aggregate** throughput is every stage busy on a different query, a pipeline emitting one token per stage-time, spread across as many conversations as there are stages. The second number is much larger and is not a latency figure.

### 5.1 The embedding imbalance

Large-vocabulary models have an embedding table bigger than any transformer layer. Qwen2.5-32B's 152k × 5120 embedding is 778M parameters, and untied means the LM head is another 778M, roughly 875 MB at Q4, against 274 MB for a layer.

In a pipeline the slowest stage sets the rate for all stages. Either give embed and unembed dedicated nodes, or shard the vocabulary dimension. Decide before finalising node count.

---

## 6. MoE models

### 6.1 The structural flip

Dense: nodes hold *different layers*, weights stay resident, activations move. A pipeline.

MoE: nodes hold *the same layer*, split by expert, in lockstep. A gang. Weights stream from local storage, activations stay. The same nodes are reused for every layer, per token.

Consequence: dense wants maximum nodes; MoE wants maximum storage bandwidth. In MoE mode this is fundamentally a storage device, not a compute device.

### 6.2 Sizing

| Model | Total | Layers | Experts per layer | Layer @Q4 | Total storage |
|---|---|---|---|---|---|
| Qwen3.5 | 397B |, |, | ~2.8 GB | ~223 GB |
| GLM-5.2 | 744B | 78 (75 MoE) | 256 + 1 shared, 8 routed | **5.25 GB** | ~419 GB |
| DeepSeek V4-Pro | 1.6T |, |, | ~9 GB | ~900 GB |
| Kimi K3 | 2.78T | 93 | 896 + 2 shared, 16 routed | **14.83 GB** | ~1.56 TB |

**All four layer sizes fit in one board's 18 GB.** Board count above one buys speed, linearly, not capability.

*GLM-5.2's per-expert size is derived from total-minus-overhead rather than read from config.json; the layer figure could move ±20%. Kimi K3's per-expert dimensions are inferred from reported architecture, not from a model card.*

### 6.3 Top-K streaming

Only the routed experts move. GLM-5.2 at batch 1 streams 8 of 256 per layer, roughly 12.3 GB per token, against 419 GB of total weights.

| Model | Bytes per token | 1 board | 2 boards | 4 boards |
|---|---|---|---|---|
| GLM-5.2 | 12.3 GB | 0.78 s | 0.39 s | 0.20 s |
| Kimi K3 | 24.6 GB | 1.57 s | 0.78 s | 0.39 s |

Load imbalance across nodes (the slowest node gates the layer) realistically adds 1.5–2×. Expert co-location by measured co-activation frequency reduces it.

### 6.4 Batching inverts

All batch elements must be processed before a layer's weights are discarded, so a step serves every query at once and **step time is the per-user latency.**

Expected distinct experts for batch B across E experts with k routed: `E × (1 − (1 − k/E)^B)`.

GLM-5.2 on one board:

| Batch | Experts touched | Step time | Throughput per token | **Per-user latency** |
|---|---|---|---|---|
| 1 | 8 | 0.78 s | 0.78 s | **0.78 s** |
| 8 | 57 | 5.6 s | 0.70 s | **5.6 s** |
| 32 | 163 | 15.9 s | 0.50 s | **15.9 s** |
| 128 | 252 | 24.6 s | 0.19 s | **24.6 s** |

Throughput improves about 4×. Latency degrades about 31×.

**Batch 1 for conversation. Large batch only for asynchronous, verifier-checked work.**

### 6.5 Replication beats batching

When a model is small enough that boards divide into groups, separate instances each run at batch-1 latency:

| Mode | Throughput | Per-user latency |
|---|---|---|
| 1 instance, batch 4 | ~1.6× | ~4× worse |
| 4 instances, batch 1 | ~4× | **unchanged** |

Groups need not run the same model. A heterogeneous fleet (one frontier model for hard reasoning, a mid-size model for bulk work, a small model for routing and classification) is something a GPU structurally cannot do, because a GPU shares weights across *copies* but not across *models*. LoRA multiplexing solves diversity only when everything shares a base.

Each group needs its own full weight copy on its own boards.

### 6.6 The prefetch problem

Layer N+1's routing is unknown until layer N's router runs, so streaming and compute serialise. Mitigations in order of value:

1. **Pin shared experts.** They fire on every token. GLM-5.2 has 1 per layer, Kimi K3 has 2. Never stream them.
2. **LRU-cache hot experts.** Expert popularity is skewed; holding the top ~15% resident kills a large fraction of loads. GLM-5.2's Quantile Balancing may flatten this distribution, measure rather than assume.
3. **Speculative prefetch.** Predict layer N+1 routing from layer N's hidden state and eat occasional misses.

### 6.7 Fault behaviour

Dense degrades gracefully, lose a node, re-shard across survivors, run slower. MoE **fails hard**: below the node count needed to hold a layer, the model does not run at all.

Graceful degradation is a software feature, not a free property. It requires failover logic in the runtime.

---

## 7. Context

KV cache scales with `attention layers × bytes per token per layer × context × batch`. **Total model size does not appear.**

KV is layer-local, only the current layer's cache needs to be resident, and it streams alongside the experts. Context is therefore a **bandwidth tax, not a capacity cliff.**

GLM-5.2 on one board (5.25 GB layer, 12.75 GB free, MLA with sparse attention):

| Context | Concurrent sessions |
|---|---|
| 32k | ~360 |
| 128k | ~90 |
| 256k | ~45 |
| 1M (full) | ~11 |

FP8 KV cache roughly doubles these. Kimi K3 is tighter: a 14.83 GB layer leaves 3.17 GB, giving roughly 22 sessions at 128k.

**Sparse attention reduces attention compute, not KV storage.** Keys must still be stored, because any of them may later be selected.

### 7.1 Speculative decoding

The only mechanism that improves *single-user* latency rather than serving more users. A small resident draft model proposes several tokens; the large model verifies them in one batched pass, with batch staying 1 from the user's perspective.

GLM-5.2 ships an MTP layer for EAGLE-style speculation, sharing the indexer and KV cache. Expect 1.5–2×.

---

## 8. What else the silicon does

Each node carries hardware the LLM path never touches. Nine of each per board:

- **H.264/H.265 encoders and decoders** at 1080p60, a serious transcoder or NVR
- **Mali GPU with OpenCL**, general compute beyond the NPU
- **Cortex-M33 at 400 MHz**, idle during inference. Useful for housekeeping, prefetch scheduling, thermal management, watchdog duty, and hard-real-time supervision that a busy Linux scheduler cannot starve
- **3× CAN FD**, a multi-drop differential bus usable as an out-of-band control plane independent of RGMII, for health checks and resets when the data network is saturated or a node has hung
- Camera interfaces, ADCs, crypto acceleration, 27 Ethernet ports, 36 CPU cores

### 8.1 Image generation, honest assessment

Compute-bound, not bandwidth-bound. The `FLOPs ≈ 2 × params` heuristic underestimates convolutions by roughly three orders of magnitude, because a 3×3 kernel on a 128×128 feature map applies the same weights 16,384 times.

| Configuration | Per image |
|---|---|
| SDXL 1024², 25 steps + refiner | ~40 s |
| SDXL + LCM, 4 steps | ~7 s |
| SD1.5 + LCM, 512², 4 steps | ~1 s |

A single RTX 3080 Ti beats a full board by 15–40× on quality per second. Usable for bulk generation with small distilled models; not a reason to build the machine.

### 8.2 Video generation

Modern video DiTs use full 3D spatiotemporal attention, roughly 100k latent tokens attending to each other, quadratically. Days per clip. Out.

SVD-class UNet models (~1.5B, temporal convolution plus frame-axis attention) involve four orders of magnitude less attention and are structurally the same problem as SDXL. Plausible at minutes per clip.

*Wan 2.2's MoE is timestep-switched rather than per-token routed (two weight swaps per video instead of ninety-three per token) which the streaming architecture handles well. It is purely the attention that kills it.*

---

## 9. Open unknowns

Ordered by how much they gate everything else.

1. **OSD32MP2 DDR4 capacity and unit price.** Every figure here assumes 2 GB per node at roughly $150. **Blocking.**
2. **Does llama.cpp's Vulkan backend run on Mali-G52?** Decides whether the software task is a port or a rewrite. Resolvable for about $148 with an STM32MP257F-DK. **Blocking.**
3. **Are the 2× Octal SPI real xSPI flash controllers**, or the 50 Mbit peripheral kind? Decides whether 18 NAND parts belong on the board.
4. **Do the 3 SDMMC controllers do independent concurrent DMA**, or share an arbiter? If they share, per-node storage bandwidth is 280 MB/s rather than 840.
5. **Real eMMC read profile at ~16 MB granularity.** The 280 MB/s figure is sequential; expert reads sit between sequential and random.
6. **Power consumption.** Estimated from product category. Unmeasured.
7. **Real matmul throughput on transformer shapes.** The 1.35 TOPS figure is a vendor number, not a transformer benchmark.
8. **RGMII chain stability** across nine nodes under sustained load.
9. **GLM-5.2 per-expert dimensions**, currently derived rather than read from config.json.

---

## 10. Build path

**Step 1, about $148.** One STM32MP257F-DK. Resolve unknowns 2, 6, and 7 on target silicon. If Vulkan does not work, this saves the rest of the budget and a year of effort.

**Step 2, about $150 more.** A second dev board. Two nodes over Ethernet is where the entire distributed runtime gets written and debugged (expert sharding, RGMII transport, streaming scheduler, failover) before spending anything on a PCB. **This is where the year of work lives, and it is hardware-agnostic.**

**Step 3, free.** Schematic and layout in KiCad, with §4.5 designed in from the start.

**Step 4, about $300.** Fab five boards, assemble one partially (3 nodes, 9 eMMC). Validate BGA fanout, RGMII chain, storage array, thermals, power distribution, toolchain.

**Step 5, about $2,775.** Fully populate one board. This is a complete device on its own: a frontier MoE model plus a local corpus with retrieval, at roughly 65 W, with no network dependency.

**Step 6, incremental.** Everything past step 4 is repetition of a validated design.

---

## 11. Prior art

**[Colibrì](https://github.com/JustVugg/colibri)** (JustVugg, July 2026) is a pure-C inference engine that runs GLM-5.2 on a 25 GB consumer machine by streaming experts from NVMe. It is independent confirmation of this document's premise and the software this hardware would build on.

Its published benchmarks are the best available evidence that storage bandwidth is the binding constraint: **0.05–0.1 tok/s** on a 12-core laptop with one NVMe channel, against **6.84 tok/s** on a 6× RTX 5090 host with full expert residency. Same engine, same model, 57× apart, with the only difference being whether storage sits in the decode path.

Also relevant: expert offloading in the ML-systems literature (Mixtral-offloading and successors), and Petals for a different approach to distributing large-model inference.

---

## 12. A note on the numbers

Nearly every figure in this project that mattered was wrong at least once before being corrected against a primary source. An entire interconnect design was invalidated when a datasheet revealed the intended fabric was a master-only flash controller. A per-token figure was wrong by 65×. A storage requirement was wrong by 93×. An image-generation estimate was optimistic by three orders of magnitude.

Read everything here as *best current estimate, several times revised.* Something else will invalidate something else.
