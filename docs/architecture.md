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

## 2. The node: RK3588 LGA module

**Superseded the Octavo OSD32MP2** (32-bit LPDDR4 SiP, six storage interfaces per node) after per-interface throughput caps and the actual GPU vendor were checked against primary sources rather than assumed. Full trail, and why the OSD32MP2 was the target until it wasn't, in [chip-selection.md](chip-selection.md).

**Banana Pi BPI-LM7** or **ArmSoM LM7** — two vendors shipping the same 506-pin LGA pinout, solder-down, no board-to-board connector.

| Spec | Value |
|---|---|
| Package | 45×50 mm, **LGA 506-pin, solder-down** |
| CPU | 4× Cortex-A76 @ 2.4 GHz + 4× Cortex-A55, **SDOT-capable** |
| NPU | 6 TOPS |
| GPU | Mali-G610 MP4, **OpenCL 2.2, Vulkan 1.1, mature Panfrost/Panthor open driver** |
| Memory | **8 GB 64-bit LPDDR4x, ~34 GB/s** (32 GB option available) |
| Storage/link | **PCIe 3.0 x4** plus 2× PCIe 2.0 x1 |
| Networking | GMAC Ethernet |
| Temperature | commercial range, unconfirmed for this module — see §9 |

**Why this rather than the OSD32MP2:** 64-bit LPDDR4x roughly doubles node memory bandwidth, the A76 cores carry `SDOT` where the OSD32MP2's A35 cores did not, and a real PCIe 3.0 x4 link (confirmed on the published LGA pin function list — PCIE30_PORT0 carries lanes 0–1, PCIE30_PORT1 carries lanes 2–3, all bifurcation pins broken out) replaces six slower, harder-to-route storage interfaces with one fast one.

**Why the GPU matters less than it used to:** the OSD32MP2's Vulkan 1.1 was the deciding factor against a GC8000/Etnaviv part with no working Vulkan path. The RK3588's Mali-G610 also runs Vulkan 1.1 on a mature open driver, so this axis is now a wash rather than a differentiator — the real gap moved to memory bandwidth and PCIe lane count.

**Confirmed the hard way, twice.** CM4-form-factor RK3588 modules (Radxa CM5, and similar) only expose PCIe x1 through their connector — ~400 MB/s, which would gut the entire advantage. A module must break out the full x4 on its own pins. The BPI-LM7/LM7 pinout does; the CM4-compatible modules do not.

---

## 3. Interconnect

**Superseded the RGMII MAC-to-MAC chain** that the OSD32MP2's integrated 2+1 switch made possible. The RK3588 has one GMAC Ethernet interface per node, not three, so board-to-board and node-to-node traffic both run over ordinary switched Ethernet rather than a bespoke chain.

**Consequence, unchanged from the prior design:** a switched fabric gives any-to-any connectivity, so tensor-parallel groups are not constrained by physical adjacency and a single dead node does not partition the network. This is also what `docs/compatibility.md`'s Ethernet-only inter-board contract already assumed, so no cross-generation compatibility work is lost in the pivot.

**Open:** real achievable throughput and latency per node over GMAC has not been measured; treat it as an unknown rather than assume it matches the old RGMII figures (~125 MB/s, microsecond latency).

---

## 4. Board

### 4.1 Form factor

150×150 mm, 4–6 nodes per carrier. Four is the baseline figure used throughout this document; six is possible on the same board size if layout allows.

### 4.2 Population

Per node, one M.2 2280 NVMe on the PCIe 3.0 x4 link plus two M.2 2242 drives on the two PCIe 2.0 x1 links.

| Interface | Attached per node | Bandwidth | Capacity |
|---|---|---|---|
| PCIe 3.0 x4 | 1× M.2 2280 NVMe | ~3.2 GB/s | up to 4+ TB |
| PCIe 2.0 x1 ×2 | 2× M.2 2242 NVMe | ~400 MB/s each | up to 1 TB each |
| **Per node** | **3 devices** | **~4.0 GB/s** | **multi-TB** |

Per board at 4 nodes: **~16 GB/s aggregate** (up to ~18.8 GB/s if all PCIe 2.0 x1 links are populated), **multi-TB total.**

### 4.3 Power distribution

**Not yet re-derived for this node.** The OSD32MP2 design's 12 V-distributed, 5 V-on-board scheme and its reasoning (avoid daisy-chaining current through a stack, keep per-board hot-swap and fault isolation) likely still apply, but RK3588 per-node draw is estimated at 10–15 W under load versus the OSD32MP2's ~65 W/board figure — real numbers pending §9's power-under-load unknown.

### 4.4 Design-for-bring-up

Carries over in spirit from the OSD32MP2 design; specifics (mounting hole count, connector choices) need to be re-derived against the LGA land pattern and 150×150 mm layout once that's in hand. Still wanted regardless of node: current-sense per node, test points, boot-select jumper, status LED, board ID EEPROM, and real airflow provisioning.

---

## 5. Dense models

**Not yet re-derived for the RK3588 pivot.** The tables below assume the OSD32MP2's 2 GB/node and 9-node board; the RK3588 board carries 8 GB/node (32 GB option) across 4 nodes, which changes every node-count and board-count figure in this section. The structural argument (TP is a speed choice, not a capacity requirement) still holds — the specific numbers need recomputing before they're trusted.

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

**Board capacity figures below still assume the OSD32MP2's 18 GB/board.** The RK3588 board carries 32 GB across 4 nodes — every layer size below still fits comfortably, so the qualitative conclusion is unchanged, but the "board count above one buys speed, not capability" framing should be re-checked against the new per-board GB once the numbers are redone.

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

### 7.1 Speculative decoding — measured, and it does not pay here

**This section previously stated an expected 1.5–2× gain, borrowed from dense-model figures. That expectation has since been tested against this architecture's actual mechanism and did not hold. Full method, table, and caveats live in `docs/dials.md` dial 13; this is the short version.**

The mechanism that makes speculative decoding close to free on a dense model — verifying B draft tokens costs the same weight reads as verifying 1 — does not transfer to a *streaming* MoE. Verifying B draft tokens here requires loading the union of whatever experts those tokens route to, and that union grows with B.

Measured on OLMoE-1B-7B (16 layers, 64 experts, top-8) by hooking the router and comparing the real expert union against a break-even threshold derived from this document's own bandwidth math: **net loss at every tested batch size (B=2, 4, 8), across twelve independent seeds.** An initial 4-seed pass showed one outlier (seed 123, B=4 multiplier 2.20× against a 1.82× break-even) close enough to the line to be worth checking properly; an 8-seed follow-up sweep landed at 2.33–2.59× on B=4, never approaching seed 123's low mark — confirming it was the natural tail of a real distribution, not the start of a different regime. Full 12-seed range: 1.54–1.86× at B=2, 2.20–2.59× at B=4, 3.06–3.91× at B=8, against break-even of 1.25×/1.82×/2.35× respectively. Routing correlation is real — every layer sits below the independent-sampling prediction — but not strong enough to clear break-even under the acceptance-rate assumptions borrowed from dense-model EAGLE/MTP literature (no draft model was built to measure this MoE's real acceptance rate). **This question is closed** under those assumptions; only a real draft model's measured acceptance rate could reopen it.

**Conclusion: do not budget engineering effort on speculative decoding for the first working runtime.** GLM-5.2's MTP layer costs nothing to leave available for later, but it is not the free win it was assumed to be here, and it should not gate or shape the initial software plan.

---

## 8. What else the silicon does

**Not yet re-derived for the RK3588 pivot.** The bullets below describe the OSD32MP2's idle-hardware inventory (Cortex-M33 housekeeping core, CAN FD out-of-band control plane) which does not carry over — the RK3588 has no M33 coprocessor and no CAN FD. What it does carry: a full 8K video codec block, the Mali-G610 for general OpenCL compute beyond the NPU, and per-node crypto acceleration. An updated inventory belongs here once the carrier board design is further along.

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

Ordered by how much they gate everything else. Rewritten for the RK3588 pivot — the OSD32MP2-era list (DDR4 capacity, Mali-G52 Vulkan, Octal SPI type, SDMMC DMA contention) is retired along with that node.

1. **Whether speculative decoding pays here.** Resolved — see §7.1 and `dials.md` dial 13. **No longer blocking.**
2. **Real NVMe throughput on RK3588 silicon**, PCIe 3.0 x4 and the two x1 links, under real transformer-shaped access patterns rather than sequential benchmarks. Resolvable for ~$120 on an RK3588 SBC. **Blocking.**
3. **Module dimensions confirmed at 45×50 mm**, but the carrier layout still needs the LGA mechanical drawing and land pattern before routing can start.
4. **Real power under sustained load.** The ~10–15 W/node estimate is from product category, not measurement. **Blocking.**
5. **Load imbalance factor** across nodes in gang (MoE) mode. Estimated at 1.5–2×, consistent with §6.3, but not measured on real hardware.
6. **Thermal behaviour** of four A76 clusters on one 150×150 mm board under continuous matmul.
7. **Power rail regulation** to within ±5% across a 4–6-node board under transient load.
8. **Real GMAC Ethernet throughput and latency per node**, since the RGMII chain figures no longer apply (see §3).
9. **GLM-5.2 per-expert dimensions**, still derived rather than read from config.json — carried over unresolved from the OSD32MP2-era list.

---

## 10. Build path

**Step 1, ~$120.** One RK3588 single-board computer — Orange Pi 5, Rock 5C, NanoPi, or similar. All run $100–150 with published llama.cpp benchmarks already available. Measure real NVMe throughput, NEON quantized GEMM, power, and thermals on the target silicon. Resolves unknowns 2, 4, and 6.

**Step 2, $0.** Run the speculative-decoding experiment against a routing trace. **Done** — see §7.1. Resolved unknown 1 before this rewrite.

**Step 3, ~$120 more.** A second SBC. Two nodes over Ethernet is where the distributed runtime gets written and debugged: expert sharding, transport, streaming scheduler, weighted distribution, failover. **This is the year of work, and it is hardware-agnostic** — see `software-architecture.md` layers 3–4 and `test-plan.md` for the localhost-simulated-nodes harness.

**Step 4, free.** Carrier schematic and layout in KiCad, adapting the vendor's published reference design.

**Step 5, ~$400.** Fab five carriers, populate one with a single module and one drive. Validate LGA reflow, power rail regulation, PCIe routing, Ethernet, thermals.

**Step 6, ~$2,400.** Populate four nodes. This is a complete device: a frontier MoE model plus a local corpus with retrieval, in a box, with no network dependency.

---

## 11. Prior art

**[Colibrì](https://github.com/JustVugg/colibri)** (JustVugg, July 2026) is a pure-C inference engine that runs GLM-5.2 on a 25 GB consumer machine by streaming experts from NVMe. It is independent confirmation of this document's premise and the software this hardware would build on.

Its published benchmarks are the best available evidence that storage bandwidth is the binding constraint: **0.05–0.1 tok/s** on a 12-core laptop with one NVMe channel, against **6.84 tok/s** on a 6× RTX 5090 host with full expert residency. Same engine, same model, 57× apart, with the only difference being whether storage sits in the decode path.

Also relevant: expert offloading in the ML-systems literature (Mixtral-offloading and successors), and Petals for a different approach to distributing large-model inference.

---

## 12. A note on the numbers

Nearly every figure in this project that mattered was wrong at least once before being corrected against a primary source. An entire interconnect design was invalidated when a datasheet revealed the intended fabric was a master-only flash controller. A per-token figure was wrong by 65×. A storage requirement was wrong by 93×. An image-generation estimate was optimistic by three orders of magnitude. A GPU was assumed to be Arm Mali for two days and turned out to be VeriSilicon. Speculative decoding was credited with a 1.5–2× gain that measurement showed to be a net loss on this architecture.

The node itself changed late, from a 32-bit LPDDR4 SiP with six storage interfaces per node to a 64-bit LPDDR4x module with one PCIe 3.0 x4 link, after per-interface throughput caps made the first design's aggregate unreachable. That pivot then itself got lost in a git merge conflict and had to be reconstructed from conversation history rather than the repository — worth noting as its own instance of the pattern this section describes.

Read everything here as *best current estimate, several times revised.* Several tables in sections 5, 6.2, and 8 above are flagged as not yet re-derived for the RK3588 node and should be treated as provisional until they are. Something else will invalidate something else.

Read everything here as *best current estimate, several times revised.* Something else will invalidate something else.
