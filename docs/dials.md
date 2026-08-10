# Design Dials

Every tunable parameter in the design, what it trades, and when it stops being changeable.

Reference document, read the entry you need, not the whole thing. Companion to [architecture.md](architecture.md).

---

## Index

| # | Dial | Trades | Changeable |
|---|---|---|---|
| 1 | Tensor-parallel degree | latency ↔ pipeline depth | software |
| 2 | Board count above minimum | money ↔ speed | incremental purchase |
| 3 | Batch size (MoE) | throughput ↔ per-user latency | software, per request |
| 4 | Instance count vs instance size | parallelism ↔ context | software |
| 5 | Context length | sessions ↔ conversation memory | software |
| 6 | Quantization format | accuracy ↔ bytes moved | software |
| 7 | Resident vs streamed weights | hardware ↔ speed | software |
| 8 | Storage population per node | money ↔ bandwidth | assembly |
| 9 | Drive selection | money ↔ **sustained read** | purchase |
| 10 | Node population per board | money ↔ capability | assembly |
| 11 | **Board dimensions** | money ↔ routability | **fab** |
| 12 | Node memory configuration | money ↔ context | purchase |
| 13 | Speculative decoding | **may be a net loss here** | software |
| 14 | Expert caching and prefetch | complexity ↔ latency | software |
| 15 | Model choice | everything | software |
| 16 | **Board specialization** | workload envelope | **fab** |

**Fab-time dials are permanent.** Numbers 11 and 16 cannot be changed after the PCB is ordered. Dial 13 is the one most likely to be got wrong, because speculative decoding is free on dense models and may be a net loss here.

---

## 1. Tensor-parallel degree

With 2 GB per node, a dense transformer layer of essentially any model fits on one node. **Tensor parallelism is therefore a speed choice, not a capacity requirement.**

Splitting a layer across N nodes divides the per-node weight read by N, but costs a ring all-reduce of N−1 hops, four times per layer.

| Group size | Hops | All-reduce | Per layer |
|---|---|---|---|
| 1 | 0 | none | 0 |
| 3 | 2 | ~0.05 ms | ~0.2 ms |
| 6 | 5 | ~0.13 ms | ~0.5 ms |
| 9 | 8 | ~0.20 ms | ~0.8 ms |

RGMII latency is measured in microseconds, so groups can be larger than a slow interconnect would allow, but the return still diminishes past roughly 4–6 nodes, because per-node weight read shrinks linearly while sync cost grows linearly.

**Everything not spent on TP goes to pipeline depth**, which buys aggregate throughput across concurrent conversations rather than single-query latency.

Because the interconnect is a switched fabric rather than a ring, TP groups are **not constrained by physical adjacency**. Any set of nodes can form a group.

---

## 2. Board count above the minimum

The minimum is one board, 18 GB holds a layer of every current frontier model. Above that, each node adds ~4.0 GB/s and each 4-node board adds ~16 GB/s to a fixed workload.

**GLM-5.2** (12.3 GB streamed per token):

| Boards | Cost | Bandwidth | Per token | With spec decoding |
|---|---|---|---|---|
| 1 | ~\$2,775 | 11.2 GB/s | 1.13 s | ~0.51 s |
| 2 | ~\$5,550 | 22.4 | 0.57 s | ~0.26 s |
| 3 | ~\$8,325 | 33.6 | 0.38 s | ~0.17 s |
| 4 | ~\$11,100 | 44.8 | 0.28 s | ~0.13 s |

**Kimi K3** (24.6 GB per token): double every time above.

Roughly linear. Extra boards also free SiP for KV cache, so context improves as a side effect.

**The trap:** adding boards to carry a *larger format* is not the same as adding boards to a *fixed* format. See #6.

---

## 3. Batch size (MoE)

The dial that behaves opposite to intuition.

All batch elements must be processed before a layer's weights are discarded, so a step serves every query at once, **step time is the per-user latency**.

GLM-5.2 on one board, 256 experts, 8 routed:

| Batch | Experts touched | Step time | Throughput/token | **Per-user latency** |
|---|---|---|---|---|
| 1 | 8 | 0.78 s | 0.78 s | **0.78 s** |
| 8 | 57 | 5.6 s | 0.70 s | **5.6 s** |
| 32 | 163 | 15.9 s | 0.50 s | **15.9 s** |
| 128 | 252 | 24.6 s | 0.19 s | **24.6 s** |

Expected distinct experts for batch B: `E × (1 − (1 − k/E)^B)`.

Throughput improves ~4×. Latency degrades ~31×.

**Batch 1 for conversation. Large batch only for asynchronous, verifier-checked work.**

---

## 4. Instance count vs instance size

When boards divide into groups, **replication beats batching**, separate instances each run at batch-1 latency.

| Mode | Throughput | Per-user latency |
|---|---|---|
| 1 instance, batch 4 | ~1.6× | ~4× worse |
| 4 instances, batch 1 | ~4× | **unchanged** |

On 4 boards:

| Model | Layer | Boards per instance | Instances |
|---|---|---|---|
| Kimi K3 | 14.83 GB | 1 | 4 |
| DeepSeek V4-Pro | ~9 GB | 1 | 4 |
| GLM-5.2 | 5.25 GB | 1 | 4 |
| Qwen3.5 | ~2.8 GB | 1 | 4 |

**Sub-dial:** four instances on one board each, or two instances on two boards each? Fewer, larger instances run faster per token and get more KV headroom. More, smaller instances serve more concurrent work. Directly trades parallelism for both speed and context.

**Heterogeneous fleets are allowed and are the strongest version of this dial.** Nothing requires groups to run the same model, a frontier model for hard reasoning, a mid-size model for bulk work, a small model for routing, all resident simultaneously and physically isolated. A GPU shares weights across *copies* but not across *models*, and LoRA multiplexing only solves diversity when everything shares a base.

**Cost:** each group needs its own full weight copy on its own boards.

---

## 5. Context length

KV cache scales with `attention layers × bytes per token per layer × context × batch`. **Total model size does not appear.**

KV is layer-local, only the current layer's cache is resident, streaming alongside the experts. Context is a **bandwidth tax (~20% on attention layers), not a capacity cliff.**

**GLM-5.2 on one board**, 12.75 GB free after the layer:

| Context | Concurrent sessions |
|---|---|
| 32k | ~360 |
| 128k | ~90 |
| 256k | ~45 |
| 1M (full) | ~11 |

**Kimi K3 on one board**, 3.17 GB free: roughly 22 sessions at 128k, 2 at full context.

FP8 KV cache roughly doubles every figure.

**Sparse attention reduces attention compute, not KV storage**, keys must still be stored because any may later be selected.

---

## 6. Quantization format

| Format | Bits/weight | GLM-5.2 size | Layer | Boards min |
|---|---|---|---|---|
| Q4_K_M | ~4.5 | 419 GB | 5.25 GB | 1 |
| **NVFP4-style mixed** | mixed | **~465 GB** | ~5.8 GB | 1 |
| FP8 | ~8 | ~744 GB | 9.3 GB | 1 |
| Q8 | ~8.5 | ~790 GB | 9.9 GB | 1 |
| BF16 | 16 | ~1.5 TB | 18.6 GB | 2 |

With 18 GB per board, everything up to Q8 fits on one board, so unlike smaller-memory designs, format choice does **not** drive board count until BF16.

**But it drives bytes moved per token, which is what sets speed.** Q8 doubles the streaming volume and therefore roughly doubles per-token time on the same hardware. Adding a board to compensate costs ~\$2,775 and gets you back to where Q4 already was.

**The best format for this machine is NVFP4-style mixed quantization**: 4-bit on the MoE expert linears only, with shared experts, attention, embeddings and early dense layers at FP8 or BF16. Reported accuracy is within about a point of the FP8 baseline.

Why it fits: the compression lands exactly on the *streamed* tensors, where bytes moved is the bottleneck, while keeping precision on the *resident* tensors, where extra bytes cost almost nothing.

**Exception:** keep a high-precision reference if doing quantization research, you need a baseline to measure degradation against.

---

## 7. Resident vs streamed weights

Any model can trade hardware for speed by streaming rather than holding.

Most relevant for diffusion, where the entire UNet is re-read every denoising step:

| Mode | Boards | Per image |
|---|---|---|
| SDXL resident, staged pipeline | 2 | ~20 s |
| SDXL streamed | 1 | ~40 s |

The real value is models that would not otherwise fit. Flux.1-dev at 12B and SD3.5 Large at 8B both run streamed on one board.

**Mixed residency is allowed:** stream the large refiner UNet, keep the VAE and text encoders resident.

---

## 8. Storage population per node

Three usable PCIe links per node, from the confirmed LGA pinout:

| Link | Attached | Throughput | Cost |
|---|---|---|---|
| PCIe 3.0 x4 | 1x M.2 2280 NVMe | ~3.2 GB/s | ~\$112 (512 GB) |
| PCIe 2.0 x1 | 1x M.2 2242 NVMe | ~400 MB/s | ~\$80 |
| PCIe 2.0 x1 | 1x M.2 2242 NVMe | ~400 MB/s | ~\$80 |
| **Total** | **3 drives** | **~4.0 GB/s** | ~\$272 |

A fourth PCIe 2.0 lane exists but is shared with USB 3.0 SuperSpeed and is better reserved for a host link.

Cumulative per 4-node board:

| Populated | Bandwidth | GLM-5.2/token |
|---|---|---|
| x4 link only | 12.8 GB/s | 0.81 s |
| **All three links** | **16 GB/s** | **0.65 s** |

**Headroom.** Node DRAM gives roughly 10.5 GB/s of effective streaming, so 4.0 GB/s of storage leaves 2.6x of margin. Unlike the previous design, memory is not close to binding.

**Route all three links at fab time regardless.** Populating later is cheap; adding traces is impossible.

---

## 9. Drive selection

The workload is **read-only in practice.** A model loads once and streams forever; weights are never modified. Writes occur only at initial load, occasional KV paging, and logging.

| Spec | Requirement | Why |
|---|---|---|
| **DRAM cache** | **mandatory** | DRAM-less drives collapse on sustained reads, every access hits the mapping table |
| Generation | **Gen3** | The link is PCIe 3.0. A Gen4 drive negotiates down and costs more for nothing |
| Capacity | **512 GB minimum** | Below that, too few NAND dies to saturate the link |
| NAND type | TLC preferred | QLC is acceptable given near-zero writes, but check sustained read behaviour |
| Write speed | irrelevant | Do not pay for it |
| Endurance | non-issue | ~600 TBW per TB at near-zero writes outlives the project |

**Sequential read on the box is not the number that matters.** Expert reads are roughly 16 MB chunks, sitting between sequential and random. A drive advertising 3,400 MB/s sequential may deliver 1,200 on this pattern. Measure at 16 MB block size.

Capacity sizing, model split across nodes:

| Model | Per node, 4 nodes | Per node, 6 nodes |
|---|---|---|
| GLM-5.2 | 105 GB | 70 GB |
| Kimi K3 | 390 GB | 260 GB |
| K3 plus a 445 GB corpus | 500 GB | 334 GB |

512 GB per node covers everything including a local corpus. 1 TB gives room for several models resident.

---

## 10. Node population per board

JLCPCB assembles a subset of designed pads, so boards can be populated incrementally.

| Nodes | RAM | Runs |
|---|---|---|
| 3 | 6 GB | GLM-5.2 (5.25 GB layer), Qwen3.5, dense models |
| 6 | 12 GB | DeepSeek V4-Pro, more KV headroom |
| 9 | 18 GB | Kimi K3, full board |

**A three-node board already runs a frontier model.** That is the cheapest genuinely useful configuration and the right first assembly.

**No design-for-holes needed.** Because the interconnect is a switched Ethernet fabric rather than a ring, an unpopulated node site does not break connectivity for anything else, a real simplification over ring or bus topologies, which require bypass links or segmentable partitioning designed in from the start.

---

## 11. Board dimensions, fab time, permanent

**100 × 100 mm** is the current design. **150 × 150 mm** is under consideration.

The component tally at 100 × 100 mm is tight: roughly 90% of the usable top face and, with all three storage tiers populated, over 100% of the bottom. Dropping the NVMe and NAND tiers brings it to about 72%.

The PCB is roughly 0.2% of BOM cost, \$5 against ~\$2,775. Going to 150 × 150 mm costs perhaps \$25 and buys:

- Wide routing channels
- Proper decoupling placement near each BGA
- Thermal spacing between nine packages
- Room to bodge a fix during bring-up

**First boards die during bring-up, not during design.** The \$25 buys margin where it matters most.

Counterarguments, all small: slightly longer RGMII traces (fine at 125 MHz), more board flex during reflow (manageable at 4 layers on 1.6 mm FR4), less dense stacking.

**Also fixed at fab time:** PCIe routed to the header, all three Ethernet ports broken out per node, eight M3 mounting holes, and footprints for storage tiers you may not populate immediately.

---

## 12. Node memory configuration

The module ships in 4, 8, 16 and 32 GB variants. 8 GB is default and the basis of every figure here.

| Per node | 4-node board | Holds a layer of | Free for KV |
|---|---|---|---|
| 4 GB | 16 GB | everything up to Kimi K3 | ~1 GB |
| **8 GB** | **32 GB** | everything | **~17 to 26 GB** |
| 16 GB | 64 GB | everything | ~49 GB |

**8 GB is the sweet spot.** Kimi K3's 14.83 GB layer fits in 32 GB with 17 GB spare, which is ample KV headroom. Going to 16 GB per node buys context and concurrent sessions, not speed.

**4 GB is a false economy.** A 4-node board at 16 GB holds K3's layer with almost nothing left, so context collapses.

---

## 13. Speculative decoding, and why it may not apply

**This entry was previously titled "the only free latency win." That was wrong on this architecture, and the correction matters.**

For a **dense** model, speculative decoding is close to free. A small draft model proposes several tokens and the large model verifies them in one pass, and because verification reads the same weights regardless of how many tokens are checked, any acceptance above 1 is a direct multiplier. Reported gains of 1.5x to 2x come from this regime.

For a **streaming MoE**, verification is not free. Checking B draft tokens requires loading the *union* of the experts those tokens route to, which expands exactly as batching does:

| Draft tokens | Distinct experts (GLM-5.2) | Bytes vs 1 token |
|---|---|---|
| 1 | 8 | 1.0x |
| 2 | 15.8 | 1.97x |
| 4 | 30.5 | **3.81x** |

At 4 draft tokens with a typical acceptance around 2.2, you pay 3.8x the bytes for 2.2x the tokens. **Net loss.**

**The open question is routing correlation.** Consecutive tokens in similar context may route to substantially overlapping experts, in which case the real union is far below the independent-sampling estimate. High correlation could still yield 1.2x to 1.4x. Low correlation means speculation should be disabled here.

**Measure before enabling.** This is testable today against a routing trace from any MoE model, with no hardware. It is the highest-value software experiment currently available.

GLM-5.2 ships an MTP head, so the mechanism is available at no training cost. The question is purely whether it pays on a streaming architecture.

---

## 14. Expert caching and prefetch

The prefetch problem: layer N+1's routing cannot be known until layer N's router runs, so streaming and compute serialise.

Three mitigations, in order of value:

1. **Pin shared experts.** They fire on every token. GLM-5.2 has 1 per layer, Kimi K3 has 2. Never stream them. The NAND tier from #8 is the natural home.
2. **LRU-cache hot experts.** Popularity is heavily skewed; holding the top ~15% resident kills a large fraction of loads. GLM-5.2's Quantile Balancing may flatten this distribution, measure rather than assume.
3. **Speculative prefetch.** Predict layer N+1 routing from layer N's hidden state and eat occasional misses.

**Related (expert co-location.** Routed experts scatter across nodes, and the slowest node gates the layer. Expected worst-case load is roughly 1.5–2× the mean, which is where the imbalance factor in every timing estimate comes from. Placing frequently co-activating experts on the same node reduces it) and requires measurement on a real model, making it a research output in its own right.

---

## 15. Model choice

| Model | Total | Layer @Q4 | Boards min | License |
|---|---|---|---|---|
| Qwen3.5 | 397B | ~2.8 GB | 1 |, |
| **GLM-5.2** | **744B** | **5.25 GB** | **1** | **MIT** |
| Kimi K2.6 | 1T | ~9.5 GB | 1 | MIT |
| DeepSeek V4-Pro | 1.6T | ~9 GB | 1 |, |
| Kimi K3 | 2.78T | 14.83 GB | 1 | custom |

**Every frontier open-weight model's layer fits on one board.** Model choice therefore drives *streaming volume per token* (and thus speed) rather than board count.

**GLM-5.2 is the recommended target.** MIT licensed, half the streaming volume of Kimi K3, coding-focused, ships an MTP layer for free speculative decoding, and is roughly a quarter of K3's total storage.

**Kimi K3 is the capability ceiling.** Keep it as proof the architecture scales; it is not the model to build around.

---

## Worked example: one firm, day and night

Dial 4 in practice, at building scale, using the properties established in [architecture.md](architecture.md) and [compatibility.md](compatibility.md): Ethernet-only interconnect, per-board power, no chaining. Twenty employees, one 4-node board each.

**Day, per desk, isolated.** One board, undivided, ~1.5 tok/s: an interactive autocomplete-style agent, no contention with anyone else's work. Or split 4 nodes into 2 instances of 2: two small parallel reviewers. Or 4 instances of 1 node each: a batch of tiny single-purpose agents. Same 16 GB/s, divided differently depending on what the desk needs that hour. Context stays whatever fits, since lowering it barely moves speed (see dial 5) and mainly frees room for more concurrent sessions rather than faster ones.

**Night, building-wide, pooled.** All twenty boards join over Ethernet, which is possible only because inter-board data is Ethernet-only per the compatibility contract. Eighty nodes, ~320 GB/s aggregate. Either gang everything into one Kimi K3 instance for maximum single-task throughput, or partition into ten instances of eight nodes running ten different overnight jobs in parallel, a heterogeneous pipeline at building scale rather than one machine's scale. Membership changes at both boundaries, boards leaving the pool at login time and rejoining at night, are the same graceful-membership logic dial 4 and the compatibility contract already require.

**Nothing about the hardware differs between these modes.** It is the same boards, repartitioned by the runtime on a schedule. The scheduling work (detect idle desks, form the pool, hand boards back before login, all without a visible slowdown) is the same weighted-sharding and membership-change machinery required elsewhere, applied at a different scale.

**The one lever worth using unconditionally.** Pinning the shared expert (dial 14) costs the same fixed 1.6 GB and yields the same roughly 11% reduction regardless of instance count, pool size, or time of day. It has no tradeoff, so there is no scenario where it should be off.

---

## Which dials matter for which goal

**Chat with a large model**
Batch 1 (#3). Speculative decoding (#13). Minimum TP (#1). Long context, few sessions (#5). Boards above minimum for speed (#2).

**Many parallel coding agents**
Multiple instances at batch 1 (#4), not one instance at high batch. Large batch acceptable *within* an instance if verifier-checked (#3). Maximum pipeline depth (#1). GLM-5.2 (#15).

**Image generation**
Streamed on one board (#7). Distilled models only. Ignore most other dials.

**Spend the least money**
Consumer eMMC (#12), partial node population (#10), eMMC tier only (#8), GLM-5.2 or smaller (#15). **Do not economise on #9, 32 GB modules, or the machine runs at half speed forever.**

**Research**
Full population, all storage tiers, high-precision reference format (#6). The unique capability is *observing* frontier MoE internals (routing distributions, co-activation patterns, expert popularity skew) which requires the whole model accessible and is not possible on any consumer GPU or unified-memory box.

---

## 16. Board specialization, fab time

Currently compute and storage are bought in a locked 9:54 ratio because they share one PCB. Splitting into two board types on the same connector decouples them:

| Type | Nodes | Storage | Character |
|---|---|---|---|
| Storage board | 3 | 27 devices | bandwidth-dense |
| Compute board | 9 | 9 devices | FLOPs-dense |

Mix per workload. MoE decode wants storage boards. Long-context prefill and diffusion (both reuse-heavy, both on the wrong side of the general law) want compute boards.

This is the **only dial that moves the ridge point**, the FLOPs-to-bandwidth ratio that determines which workloads the machine is good at. Everything else optimizes within a fixed ratio.

Costs nothing extra: same header, same firmware, two populate lists.

---

## The permanent decisions, restated

Everything else is software or a later purchase. These are locked when PCBs are ordered:

1. **Board dimensions**, 150 x 150 mm
2. **All three PCIe links routed per node**, even if not populated
3. **4.0 V regulation per node**, to +/- 5%
4. **Ethernet-only inter-board data**, per the [compatibility contract](compatibility.md)
5. **Board type**, general purpose or specialized per dial 16

---

*Companion to [architecture.md](architecture.md) and [chip-selection.md](chip-selection.md). Figures derive from vendor datasheets, a published LGA pinout, and arithmetic. Nothing here is anchored to a measurement on real silicon.*
