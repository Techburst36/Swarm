# The Ideal Node

### The node specification, and the announced part that meets it

The current design is limited by the best available silicon, not by the idea. Sections 1 through 4 specify the node that would make a swarm board competitive with a datacenter server at a third the cost.

That specification was written expecting no such part to exist. **Rockchip's announced RK3688 meets or exceeds every line of it**, and Rockchip is the one vendor in this space that sells to individuals rather than gatekeeping behind NDAs and licensed module vendors. Section 6 covers it.

The part is not shipping. Public timelines put it somewhere between 2026 mass production and Q1 2027, with no BSP, no boards announced, and specifications marked subject to change. Treat sections 1 through 5 as the durable analysis and section 6 as a candidate worth tracking, not a plan.

---

## 1. The target

Measured baseline: GLM-5.2 served by a production API returned 1,780 output tokens in 31.81 seconds, or **~56 tokens/second**. That is the number a user notices, and it is what any local machine must reach before speed stops being an objection.

GLM-5.2 streams roughly 12.7 GB of expert weights per token. With speculative decoding at ~1.8x effective, matching 56 tok/s requires about **395 GB/s of aggregate storage bandwidth.**

At nine nodes per board, that is **~44 GB/s of storage per node.**

The OSD32MP2 delivers ~1.9 GB/s. **The gap is 23x, and it is entirely in one spec.**

---

## 2. Specification

| Requirement | Target | Current (OSD32MP2) | Gap |
|---|---|---|---|
| **Storage bandwidth per node** | **~44 GB/s** | ~1.9 GB/s | **23x** |
| Memory bandwidth | ~136 GB/s | ~10 GB/s | 13x |
| On-package or attached memory | 4–8 GB | ~2 GB | 3x |
| Compute (INT8/INT4) | ~5 TOPS | 1.35 TOPS | 3.7x |
| Inter-node link | 10 GbE or better | 1 GbE | 10x |
| Package pitch | ≥0.8 mm | 1.0 mm | met |
| Power per node | ≤15 W | ~5 W (est.) | headroom |
| Price per node | ≤\$300 | ~\$150 | 2x acceptable |

**Storage bandwidth is the only one that matters.** Everything else is either already close or follows from it. A part that hits 44 GB/s of storage almost certainly has the memory bandwidth to absorb it and the compute to use it.

---

## 3. How each requirement could be met

### 3.1 Storage, the binding constraint

Three routes, in increasing order of plausibility:

| Interface | Per device | Devices for 44 GB/s | Notes |
|---|---|---|---|
| eMMC 5.1 HS400 | 0.4 GB/s ceiling | 110 | **Impossible.** Protocol caps at 400 MB/s |
| UFS 4.0 | ~4 GB/s | 11 | Plausible. Needs a UFS controller |
| PCIe Gen5 x16 | ~56 GB/s | 1 link, 4 drives | Cleanest, needs server-class I/O |
| PCIe Gen4 x8 | ~14 GB/s | 3 links | Realistic middle ground |

**The eMMC ceiling is the hard wall in the current design.** eMMC 5.1 tops out at 400 MB/s by protocol, so no eMMC part can ever get past ~10.8 GB/s per board with three controllers per node. That is a specification limit, not a component limit.

**UFS 4.0 is the realistic path.** It is already in flagship phones at ~4 GB/s sequential read. It uses a command queue with out-of-order execution (much better for the ~16 MB random-ish reads expert streaming produces), and comes in BGA-153 11.5x13 mm, **the same footprint as the eMMC array in the current design.** A future board could reuse the mechanical layout.

**PCIe Gen5 is the clean answer** but implies server-class I/O on a node meant to cost under \$300.

### 3.2 Memory

LPDDR5X-8533 at 128-bit gives ~136 GB/s, which is enough to absorb 44 GB/s of inbound streaming with headroom for KV cache and activations.

This is standard in current flagship phone SoCs. It is not standard in anything you can buy as a component.

**Consequence: on-package memory has to go.** The SiP approach that made version one buildable caps memory bandwidth at whatever the packager chose to integrate, Octavo integrates a modest DDR4 configuration because they package for general-purpose designs, not for streaming. Reaching 136 GB/s means routing LPDDR5X yourself, which is a genuinely hard PCB problem.

**That trade is the central one in this document.** SiP bought buildability at the cost of bandwidth. Version one needed the former. A competitive version needs the latter.

### 3.3 Compute

~5 TOPS INT8, or ~2.5 TOPS with native INT4 support.

This is the easiest requirement and the least interesting. MoE decode at batch 1 has arithmetic intensity around 2 FLOPs/byte, so compute is nowhere near binding. Any part with the storage bandwidth will have more than enough.

**Native INT4 would matter more than raw TOPS.** Current parts unpack 4-bit weights to INT8 before the accelerator touches them, so you save bandwidth but not compute. A datapath that executes packed 4-bit directly removes that step.

**A Vulkan or OpenCL path remains non-negotiable.** It is the difference between porting an inference engine and writing one, and it was the single largest risk reduction in the version-one chip selection.

### 3.4 Interconnect

At 44 GB/s per node, activations become worth thinking about again. Decode moves only ~12 KB per layer boundary, so 1 GbE is still adequate there. But prefill moves ~1.2 GB per layer boundary at 100k context, which at 125 MB/s is minutes of transfer.

**10 GbE or 25 GbE would make prefill viable across nodes**, which is currently the design's worst-performing workload and the reason long-context prompt processing is bad.

---

## 4. What such a board would look like

Nine ideal nodes, same 100x100 mm form factor, same architecture:

| | Value |
|---|---|
| Nodes | 9 |
| Memory | 36–72 GB |
| Storage bandwidth | ~400 GB/s |
| Storage capacity | ~4 TB (4x PCIe Gen5 NVMe per node) |
| **GLM-5.2** | **~56 tok/s** |
| Kimi K3 | ~28 tok/s |
| Power | ~400 W |
| **Cost** | **~\$9,500** |

### 4.1 Against the commodity alternative

The honest comparison at this speed is an EPYC whitebox, a single EPYC 9004 has 128 native PCIe Gen5 lanes, enough for 24 NVMe drives at x4 with 32 lanes left over. That is genuinely what EPYC is built for.

| | Ideal board | EPYC whitebox |
|---|---|---|
| Cost | **~\$9,500** | ~\$26,500 |
| Speed | ~56 tok/s | ~56 tok/s |
| Power | **~400 W** | ~900 W |
| Form | one PCB | 2U chassis |
| Software | write it | vLLM works today |

**2.8x cheaper and 2.3x less power at the same speed.**

The reason is that **EPYC pays for generality this workload does not use.** A 32-core CPU, twelve memory channels, a socket, a chassis, redundant supplies, expert streaming needs almost none of it. It needs storage channels and just enough compute to do the FFN. The ideal node is that and nothing else.

*(EPYC pricing reflects August 2026 conditions: ~\$6,000 for 384 GB of DDR5 RDIMM and ~\$14,000 for 24x 3.84 TB Gen5 NVMe, both shortage-inflated. That inflation is itself part of the argument for a design that uses the cheapest memory tier.)*

### 4.2 The economics finally work

For a 20-engineer firm paying roughly \$30,000/year for max-tier coding seats:

| | 2 ideal boards | 20 API seats |
|---|---|---|
| Year 1 | \$19,000 + \$700 power | \$30,000 |
| Year 2 | \$700 | \$30,000 |
| Year 3 | \$700 | \$30,000 |
| **3-year total** | **\$21,100** | **\$90,000** |

**Payback in 7.6 months**, then electricity only.

This is the configuration where the pitch stops being "you have no choice" and becomes "run the arithmetic." At 0.73 tok/s the market was firms with a contractual prohibition on cloud AI, which is narrow. At 56 tok/s it is any firm that would rather own than rent.

---

## 5. What exists today, and how close

| Class | Storage BW/node | Memory BW | Buyable as a part? |
|---|---|---|---|
| **Ideal** | 44 GB/s | 136 GB/s |, |
| **Rockchip RK3688** (announced) | **~56 GB/s** (PCIe Gen5 x16) | **~200 GB/s** | **Expected yes** |
| Qualcomm QCS8550 (SOM) | ~10 GB/s | ~67 GB/s | SOM only, ~\$400 |
| Flagship phone SoC | ~4 GB/s (UFS 4.0) | ~120 GB/s | **No** |
| Mid-tier ARM SoC | ~1–2 GB/s | ~25 GB/s | Yes |
| **OSD32MP2 (current)** | **1.9 GB/s** | **~10 GB/s** | **Yes** |
| Server CPU | 300+ GB/s | 460 GB/s | Yes, at 10x the price |

**Phone SoCs are the closest and are drifting the right way**. UFS 4.0 and wide LPDDR5X are becoming standard in the mid-tier, and mid-tier parts do eventually reach the open component market through Rockchip, Amlogic, MediaTek and others.

The memory side is nearly solved in that class. **The storage side is a tenth of what is needed**, and that is the number to watch.

---

## 6. RK3688: the part that meets the spec

Rockchip announced the RK3688 at their 2025 Investor Conference as the successor to the RK3588. The published specification meets or exceeds every line of section 2.

| Requirement | Target | RK3688 | |
|---|---|---|---|
| Storage bandwidth per node | 44 GB/s | **~56 GB/s** (PCIe Gen5 x16) | exceeds |
| Memory bandwidth | 136 GB/s | **~200 GB/s** (8-channel LPDDR5/5x/6 @ 8400 Mbps) | exceeds |
| Memory per node | 4–8 GB | 16–32 GB feasible | exceeds |
| Compute | 5 TOPS | 32 TOPS (RKNN-P3) | exceeds |
| Vulkan path | required | Mali-class Magni GPU, open Panfrost/Panthor driver | met |
| Purchasable |, | **Rockchip sells on the open market** | met |

**Accessibility is why this matters more than the raw numbers.** Qualcomm and MediaTek gatekeep flagship silicon behind NDAs and \$400 modules from a handful of licensed vendors. Rockchip publishes datasheets and its parts routinely land on \$100–180 boards from Radxa, Orange Pi and Firefly. Radxa has already committed to using the RK3688 in the successor to its ROCK 5 family.

That makes the version-three plan concrete: **wait for the ROCK 6, then design a carrier for it.** Same architecture, same runtime, roughly 30x the storage bandwidth per node.

### Three tiers

The same architecture and the same runtime across three node classes:

| Tier | Node | GLM-5.2 | Board cost | Buyer |
|---|---|---|---|---|
| Entry | OSD32MP2 | ~0.4 tok/s | ~\$2,775 | hobbyist, research lab |
| Mid | QCS8550 SOM | ~10 tok/s | ~\$3,400 | software firm, biotech |
| **Full** | **RK3688** | **~18 tok/s** | **~\$3,500** | law firm, finance, any seat replacement |

Only the node changes. Storage topology, interconnect pattern, expert-streaming runtime, and every dial in [dials.md](dials.md) apply unchanged at all three.

### Tier three at scale

Eight boards, the budget of 20 max-tier coding seats for one year:

| | Value |
|---|---|
| Cost | ~\$30,000 |
| Memory | 512 GB |
| Storage bandwidth | ~896 GB/s |
| Storage capacity | ~16 TB |
| Power | ~2,400 W (one 240 V circuit) |
| **Kimi K3** | **~66 tok/s** with speculative decoding |
| **GLM-5.2** | **~127 tok/s** |
| Or | 8 independent K3 agents at ~8 tok/s each |

Against running Kimi K3 in VRAM: roughly 44 RTX 5090s at ~\$185,000 and ~26 kW, which requires a three-phase service upgrade before it can be switched on.

**6x cheaper, 11x less power, and faster than the API those seats were paying for.**

Three-year total cost of ownership against 20 API seats at ~\$30,000/year:

| | 8 boards | 20 seats |
|---|---|---|
| Year 1 | \$30,000 + \$2,000 power | \$30,000 |
| Year 2 | \$2,000 | \$30,000 |
| Year 3 | \$2,000 | \$30,000 |
| **Total** | **\$36,000** | **\$90,000** |

Payback in roughly 13 months, and the code never leaves the building.

### The caveats, stated plainly

**Timing.** SBCwiki lists Q1 2027 with no public BSP, no mainline support, and no boards announced. Other sources report 2026 mass production. Rockchip timelines slip.

**Pre-production specs.** Rockchip's own materials note that final parameters are subject to the mass-produced version. PCIe Gen5 x16 may be reduced.

**Board difficulty rises sharply.** PCIe Gen5 at 32 GT/s needs controlled impedance, low-loss laminate and probably ten or more layers. Eight-channel LPDDR6 is beyond what a hobbyist should route. This tier realistically requires a SOM, which reintroduces a price floor.

**And the memory shortage may end in the same window.** If DRAM and NAND prices fall, every component here gets cheaper, but the 512 GB unified-memory machines that were withdrawn in 2026 also return. The floor argument survives that; the ceiling argument weakens.

**Nothing above should be treated as a product roadmap.** It is a trajectory. The entry tier is the only one buildable today, it is the one that proves the runtime, and the runtime is what carries across all three unchanged.

---

## 7. Three routes to a version three

**Route A, bare chip plus your own memory.**
Drop the SiP, route LPDDR4X or LPDDR5 yourself. Cheaper per node and you control the memory configuration. But the chip's memory controller is still the ceiling, so this buys perhaps 2x, not 13x. A meaningful improvement and not a transformative one.

**Route B, wait for the right SoC. Now the leading route.**
This was written as the speculative option. The RK3688 announcement makes it concrete: an announced part, from an open-market vendor, that meets the full specification. Radxa has committed to a ROCK 5 successor built on it, which would be the accessible on-ramp.

Costs nothing to wait, the runtime work carries over unchanged, and the waiting period is exactly the time the runtime needs anyway. **The correct action while waiting is to write the software on two dev boards**, so it is ready when the hardware lands.

The risk is that Rockchip slips, cuts PCIe Gen5 x16 from the production part, or prices the first boards above the historical range. None of those are recoverable by waiting harder, which is why route A and route C stay on the table.

**Route C, invert the memory hierarchy.**
The current design streams from flash because DRAM is expensive per gigabyte. If the shortage ends and DRAM gets cheap, a node with 32 GB of LPDDR5X holds a large fraction of the model *resident* and the streaming problem changes shape entirely. Less a redesign than a different point on the same curve.

Worth noting that the projected end of the memory shortage and the projected arrival of the RK3688 fall in roughly the same window. That is favourable on components and unfavourable on competition: the 512 GB unified-memory machines withdrawn in 2026 would likely return at the same time. The floor argument survives that. The ceiling argument weakens.

---

## 8. What carries over regardless

Everything above is a hardware conversation. The parts that survive any of these routes:

- **The distributed runtime**, expert sharding, streaming scheduler, prefetch prediction, failover. This is the year of work and it is hardware-agnostic.
- **The topology**, many nodes, private storage channels, aggregate bandwidth scaling with node count.
- **The general law**, good where weights are used once and discarded, bad where they are reused.
- **The storage footprint**, BGA-153 11.5x13 mm is the package for both eMMC and UFS. A version-three board could reuse the mechanical layout.
- **The dials**, batching, instance count, quantization format, speculative decoding, expert caching. All of it applies at any bandwidth.

---

## 9. The honest summary

**The architecture is right at every scale. The current silicon does not reach the interesting part of it, but announced silicon does.**

Version one is limited by one number (storage bandwidth per node) and that number is fixed by chip choice, which is the one axis a board designer cannot control. Everything slow about the current design traces back to it.

That is a better position than a design flaw, because it means the project is waiting on the industry rather than on a mistake. And the industry is moving toward the spec above for its own reasons.

This document was written as a shopping list for a part that did not exist. The RK3688 announcement means it may arrive sooner than expected, from the one vendor that sells to individuals.

---

*Companion to [architecture.md](architecture.md), [dials.md](dials.md) and [chip-selection.md](chip-selection.md). The 56 tok/s target derives from a single timed measurement against a production API and should be treated as one data point, not a benchmark. All hardware figures are estimates from vendor specifications and arithmetic; nothing here is anchored to a measurement on real silicon.*
