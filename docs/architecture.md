# Architecture

A distributed inference board for frontier-scale Mixture-of-Experts models.

**Design stage.** No number in this document is anchored to a measurement on real silicon. Figures derive from vendor datasheets, a published LGA pinout, model configuration files, and arithmetic. Several have been revised more than once.

---

## 1. The premise

Mixture-of-Experts decode at batch 1 is memory-bandwidth-bound.

GLM-5.2 has 744B total parameters and activates roughly 40B per token: 8 routed experts out of 256, per layer, across 75 MoE layers. Every one of those weights is read once, used once, and discarded. Arithmetic intensity is around 2 FLOPs per byte.

A GPU is thousands of multipliers sharing one memory bus. It wins whenever weights can be fed to those multipliers repeatedly from cache. At 2 FLOPs/byte there is nothing to reuse, so the multipliers idle and the bus is the only thing that matters.

This design inverts the ratio: **several modest compute nodes, each with private storage channels.** Aggregate bandwidth scales with node count instead of being fixed at purchase. A GPU's 1.8 TB/s does not grow when you attach more drives; this does.

### 1.1 The general law

> **Good wherever weights are used once and discarded. Bad wherever weights are reused heavily.**

| Workload | Weight reuse | Fit |
|---|---|---|
| MoE decode, batch 1 | none | best case |
| Dense LLM decode | none | good |
| Heterogeneous multi-model fleet | none | excellent |
| LLM prefill (long prompts) | heavy across prompt | poor |
| Diffusion, convolutions | massive spatial reuse | poor |
| Video DiT | reuse plus quadratic attention | hopeless |
| Training | not viable, see section 8 | impossible |

Every fit and every disqualification below follows from that line.

### 1.2 Where this loses

Stated up front so the document is not self-flattering:

- **Anything that fits in VRAM.** A single RTX 5090 running a 32B model at Q4 does perhaps 1,000 to 2,000 tok/s aggregate. Four nodes do a fraction of that, for more money.
- **Edge inference.** A Jetson Orin Nano is \$249, draws 15 W, and works out of the box.
- **Latency.** The architecture trades latency for throughput at every level.
- **Software maturity.** The distributed runtime does not exist yet.

The niche is narrow and real: **models whose weights are enormous but whose active compute per token is small.** That is precisely what MoE is, and precisely where the open-weight frontier went.

---

## 2. The node: RK3588 LGA module

Selected after evaluating and rejecting roughly twenty alternatives. See [chip-selection.md](chip-selection.md) for the trail, including the STM32MP257 SiP design this replaced and why.

**Banana Pi BPI-LM7** or **ArmSoM LM7**, two vendors shipping the same 506-pin LGA pinout around Rockchip's RK3588.

| Spec | Value |
|---|---|
| Package | 45 x 50 x 4.5 mm, **LGA 506-pin, solder-down** |
| CPU | 4x Cortex-A76 @ 2.4 GHz + 4x Cortex-A55 @ 1.8 GHz, **Armv8.2 with SDOT** |
| GPU | Mali-G610 MP4, OpenGL ES 3.2 / OpenCL 2.2 / Vulkan 1.1 |
| NPU | 6 TOPS INT8, INT4/INT8/INT16 mixed |
| Memory | **8 GB LPDDR4x default, 64-bit** (4/8/16/32 GB options) |
| Onboard storage | 32 GB eMMC (OS lives here) |
| **PCIe** | **3.0 x4, plus 2x PCIe 2.0 x1 usable** (confirmed from pinout) |
| Power | **VCC4V0\_SYS, 4.0 V +/- 5%**, 5 pins |
| Temperature | 0 to 70 C commercial |
| Price | ~\$268 USD (8 GB + 32 GB), 20 in stock at time of writing |

### 2.1 The PCIe breakout, confirmed

The published pin function list resolves the question that decides the whole design. High-speed differential pairs on the LGA:

| Port | Lanes | Notes |
|---|---|---|
| PCIE30\_PORT0 | 2 (TX0/1, RX0/1) | lanes 0 and 1 |
| PCIE30\_PORT1 | 2 (TX0/1, RX0/1) | mapped as lanes 2 and 3 on the reference mainboard |
| PCIE20\_0 | 1 | also usable as SATA 3.0 |
| PCIE20\_1 | 1 | also usable as SATA 3.0 |
| PCIE20\_2 | 1 | **shared with USB 3.0 SuperSpeed** |

PORT0 and PORT1 combine to give **the full PCIe 3.0 x4**. All bifurcation control pins are broken out (`PCIE30X4_PERSTN`, `CLKREQN`, `WAKEN`, plus X2 and X1 variants), so 1x4, 2x2 and 4x1 modes are all reachable.

This matters because CM4-form-factor RK3588 modules, such as the Radxa CM5, only expose PCIe x1 through the connector. That would cap a node at roughly 400 MB/s and eliminate the reason to use this chip.

### 2.2 Why a module rather than a bare chip

The general rule from [chip-selection.md](chip-selection.md) is that **a system-on-module is a category error when its price buys packaging rather than capability.** A \$250 module carrying 2 GB and one eMMC channel fails that test. This one carries 8 to 32 GB of 64-bit LPDDR4x, a PCIe 3.0 x4 link, and SDOT-capable cores.

What the module price buys here:

- **A 12-layer PCB doing the LPDDR4x routing.** Getting a 64-bit memory interface wrong on a custom board means the PHY training sequence falls back to a lower rate or narrower bus, halving bandwidth, and you discover it after fabrication.
- **LGA rather than BGA.** Solder-down, no balls to collapse, visually inspectable, and far more forgiving than a fine-pitch BGA on a first design.
- **Published reference designs.** Schematics, PCB pads, 2D drawings and SMD files for the vendor's own carrier board are public, so the carrier is an adaptation rather than a clean sheet.
- **Two vendors, one pinout.** Second-sourcing is possible, which matters for a design intended to outlive one supplier.

### 2.3 The compute constraint has gone away

The previous design was built on dual Cortex-A35 cores, which are Armv8.0-A and lack the `SDOT` dot-product instruction. llama.cpp's fast quantized NEON kernels gate on `FEAT_DotProd` and fall back to widening multiply-accumulate without it, roughly halving throughput.

The A76 is Armv8.2-A with native `SDOT`. Four of them at 2.4 GHz deliver well over 50 GOPS on quantized GEMM, against roughly **14 GFLOPS needed** to consume 4 GB/s of streamed Q4 weights.

**Compute is no longer near the constraint.** The GPU and NPU become optional optimisation rather than blocking questions, which removes the largest software risk in the previous design.

---

## 3. Interconnect

Nodes connect over **standard Ethernet**. The RK3588 provides a GMAC interface; each blade carries its own PHY and magnetics (see section 4.0), and every blade's conditioned Ethernet signal lands on a switch IC hosted on the backplane. No direct node-to-node RGMII — all traffic routes through the backplane switch.

| Property | Value |
|---|---|
| Per-link throughput | ~125 MB/s at 1 GbE |
| Latency | microseconds |
| Topology | switched, any-to-any, via the backplane switch |

### 3.1 Decode is unaffected, prefill is bounded

At 6144 hidden dimensions and FP16 activations, a layer boundary moves about 12 KB per token.

| Workload | Per layer | Total | At 125 MB/s |
|---|---|---|---|
| Decode, 1 token, 75 layers | 12 KB | ~900 KB | **~7.2 ms**, about 1% of a token |
| Prefill, 100k tokens | 1.2 GB | ~90 GB | **~12 minutes** |

**Decode does not care about the link. Prefill is bounded by it.** That asymmetry is why long-prompt processing is this design's worst workload, and why the ideal-node specification calls for 10 GbE.

Ethernet-only inter-board data is a hard requirement of the [compatibility contract](compatibility.md), not a convenience.

---

## 4. Board

### 4.0 Blade and backplane, superseding a single shared board

**The original plan — four to six LGA-506 modules on one shared 150x150mm, 4-layer board — does not route.** Each module needs full escape routing for its LGA-506 breakout (PCIe 3.0 x4 diff pairs, PCIe 2.0 x1 pairs, GMAC, control and power) from a dense footprint out to via fields where it can reach other components. Doing that for one module on 4 layers is already tight. Doing it for four, sharing board area and needing routing channels between them, pushes into 8 or more layers — a materially different, and materially riskier, board.

**Split into two boards instead: one small, high-layer-count blade per node, plugged into one larger, low-layer-count backplane.**

- **Blade**: one LGA-506 module, its NVMe drives, its own Ethernet PHY and magnetics, on an 8 to 10 layer board sized to just the module footprint plus routing margin. High layer count, but the board is small, so cost per blade stays bounded even at 10 layers.
- **Backplane**: 150x150mm-class, back down to **4 layers**, because the hard problem — LGA escape routing — never touches it. The backplane carries only power distribution, an already-PHY'd Ethernet signal per blade, and a couple of low-speed control lines (I2C for board-ID, a boot-select line).

**This directly derisks the single largest unknown in the project.** `test-plan.md`'s unknown 3 (the LGA mechanical drawing and land pattern) can now be validated on one cheap blade before committing to four or six. A bad blade design costs one small board, not a $2,400 populated carrier. A blade that fails in the field gets swapped, not the whole board.

#### Why the Ethernet PHY belongs on the blade, not the backplane

Two very different things can cross a board-to-board connector labeled "Ethernet":

- **Raw GMAC/RGMII** (PHY chip lives on the backplane): a parallel bus with tight timing and length-matching requirements — the same class of routing problem that made the shared board fail, just relocated to the backplane connector.
- **Post-PHY, post-magnetics MDI signal** (PHY and magnetics live on each blade, next to the connector): electrically identical to what comes out of an RJ45 jack — isolated, common-mode-rejected, and forgiving. Real industrial backplane systems (CompactPCI, VPX, MicroTCA) route Ethernet across backplanes this way routinely, without exotic layer counts.

**PHY and magnetics go on the blade.** The backplane connector only ever carries the conditioned signal.

#### Why the backplane hosts the switch, not per-blade cables

`compatibility.md` requires Ethernet-only inter-board data, and `dials.md`'s day/night pooling and multi-building federation scenarios assume a Swarm board can join or leave a fleet as a single unit. If each blade had its own cabled Ethernet jack, connecting two Swarm boards together would mean one cable per blade — 4 to 6 cables for a 2-board link, and it gets worse as more boards join a pool.

**The backplane hosts a small Ethernet switch IC** that every blade's conditioned Ethernet signal lands on, with **one uplink port** leaving the board. A candidate part: **Realtek RTL8367S/RTL8367RB**, a 5-port Gigabit switch with integrated PHYs and an RGMII CPU port — enough ports for up to 4 blades plus the uplink, with the 6-node configuration needing a second switch or a larger part. Connecting a Swarm board to another Swarm board, or to the rest of a network, becomes one cable, exactly the abstraction the rest of the docs already assume.

#### Connector

**Perpendicular mount, not stacked/parallel.** The blade stands up from the backplane like a card in a slot — this is what makes "one small blade first, swap on failure" physically real, and keeps the backplane footprint compact regardless of blade count.

**Settled on a real PCIe edge connector, repurposed rather than used electrically as PCIe.** Superseded an earlier consideration of a generic right-angle board-to-board connector (Samtec Tiger Eye or similar) once the actual PCIe pinout was checked against a real datasheet rather than assumed.

The case for reusing a PCIe slot connector directly:

- **It's a real, stocked catalog part**, not something to spec from scratch. Amphenol and Sullins both sell standalone PCIe card-edge connectors through Digi-Key and Mouser, in x1/x4/x8/x16 widths, various pin counts (36, 64, 98, 164, 280), through-hole or SMT — the exact part motherboard manufacturers buy in bulk.
- **It inherits the entire PC industry's mechanical coordinate system, not just the bracket shape.** The PCI Express CEM spec defines bracket position relative to the slot connector, connector position relative to the case rail, and rail position relative to standard case geometry, all as one worked-out system. Mounting a real PCIe connector on the backplane at the spec-correct position means a blade automatically lands correctly relative to a standard PC case's mounting hardware — solving the backplane/standoff alignment question raised earlier by inheriting a solved layout rather than re-deriving one.
- **Real mechanical retention.** Most PCIe slots include a locking latch that clips over a notch in the card edge — genuine retention, not friction alone.
- **Direct precedent for exactly this repurposing.** Cryptocurrency-mining PCIe risers already run only power and a couple of signal lines through a PCIe x1 connector, ignoring the rest of the pins. Known, low-risk pattern.

**Checked against a real pin table (not assumed) whether it actually carries what a blade needs — power, Ethernet, and control:**

| Requirement | PCIe pin(s) | Native or repurposed |
|---|---|---|
| Power | Multiple dedicated +12V, +3.3V, GND pins | **Native** — this is what those pins are for |
| Board-ID / EEPROM | SMCLK / SMDAT | **Native** — PCIe's SMBus sideband is I2C by another name |
| Boot-select / reset | PERST#, WAKE# | Repurposed spare control pins |
| Ethernet MDI | 4 of the connector's differential pairs | Repurposed — see below |

**The connector must be x4-class, not x1.** Each PCIe lane provides exactly two dedicated differential pairs (one TX-only, one RX-only). A x1 connector has only 2 pairs total — not enough, since full-speed Gigabit Ethernet (1000BASE-T) needs 4 differential pairs simultaneously to reach the PHY's magnetics. A x4 connector provides 8 pairs: 4 carry the Ethernet MDI signal, 4 sit spare (real headroom, plausibly enough for a later 2.5GbE upgrade without a connector change). This is what fixes the connector size class, not a power or pin-count argument — it's the Ethernet pair budget that forces x4.

Per-blade power draw (10-15 W estimated) sits comfortably inside a stock PCIe slot's native power budget — no supplemental power connector needed.

**Recommendation: x4-class PCIe card-edge connector** (64-pin family), Ethernet PHY and magnetics on the blade side of the connector as established above, backplane side wired to the switch IC.

**Still open:** exact connector part number and insertion-cycle rating (worth a decent-quality part — Amphenol over the cheapest source — given the repeated-swap/disaster-zone use case), whether 4 or 6 ports on the backplane switch is the right target, and blade mechanical dimensions once the LGA land pattern is in hand.

---

### 4.1 Blade form factor and layout

**Blade sized to the module footprint plus routing margin, 8 to 10 layers.** Backplane at 150x150mm-class, **4 layers**, hosting 4 to 6 blade connectors plus the switch IC and power input.

Each module is 2,250 mm2. A blade sized around it, rather than four to six modules sharing one large board, is what makes the high layer count affordable — the expensive layers are only ever as large as they need to be.

**Mechanical basis: standard full-height PCIe bracket geometry**, per the PCI-SIG spec — 120.02 mm bracket height, ~18.4 mm width, 20.32 mm case slot pitch. Real, stocked hardware, not a custom design: any ATX/mATX case already has the rail, the screw, and the slot opening. Early bring-up (`test-plan.md` steps 5-6, one blade first) can happen in an off-the-shelf open-frame test bench case with zero custom mechanical work. A custom enclosure only becomes necessary once past single-blade validation — and it's a cheaper quote from a fab shop because the blade's retention geometry already matches something they tool for routinely.

**M.2 drives mounted rotated 90 degrees from the usual orientation** — length running along the blade's height instead of its length. This is not novel: ASUS's Hyper M.2 x16 cards already ship four M.2 drives arranged exactly this way, in a row under one shared heatsink and blower fan, as a real product. The geometry:

| | Dimension | Fit |
|---|---|---|
| M.2 2280 | 22 mm wide x 80 mm long | length runs along blade height |
| Available card height (full-height bracket) | ~100-110 mm usable | 80 mm module fits with ~20-30 mm to spare for standoff and PCB margin |
| Per-drive footprint along blade length | ~26 mm (22 mm width + connector/routing clearance) | |
| 3 drives (RK3588's native count) | ~78 mm of blade length | |
| Standard GPU card length, for comparison | 240-300+ mm | ~150-200 mm of length left over for the module, power regulation, and connector |

**Consequence: the blade does not need to be GPU-length.** The M.2 row leaves most of a standard card's length unused — the blade can be considerably shorter, which is cheaper to fab and lighter to swap. Exact length is a layout-stage decision once the module footprint and connector are placed.

**Blade thickness is not set by the M.2 mounting.** A standard M.2 socket holds the module roughly parallel to the PCB — connector clearance plus module plus PCB is on the order of 6-10 mm, comfortably under a single 20.32 mm slot pitch. Worth re-checking against real component datasheets, but the earlier assumption of needing two slot-widths per blade may not hold.

**All three drives are mechanically identical** in this layout — the M.2 connector doesn't change shape based on lane count (x1 vs x4 is wired behind the same physical socket), so the row is uniform regardless of which drive gets the x4 link.

**Still open:** whether the backplane standoff positions can be made to land on standard ATX standoff locations. The ATX standoff grid is a published, fixed pattern; the backplane's own component placement is not derived from it. Most cases carry more pre-threaded standoff posts than a motherboard actually uses, so hitting a usable subset directly is plausible, but this is a layout-stage question, not yet resolved — either some backplane mounting holes land on standard positions and the rest use repositioned or added standoffs, or the test-bench case needs a few minutes of setup instead of none. Not a blocker either way.

### 4.2 Population, per blade

| Interface | Attached per node | Throughput |
|---|---|---|
| PCIe 3.0 x4 | 1x M.2 2280 NVMe | ~3.2 GB/s |
| PCIe 2.0 x1 | 1x M.2 2242 NVMe | ~400 MB/s |
| PCIe 2.0 x1 | 1x M.2 2242 NVMe | ~400 MB/s |
| **Per node** | **3 drives** | **~4.0 GB/s** |

The third PCIe 2.0 lane is shared with USB 3.0 SuperSpeed and is reserved for a host or debug link.

| | 4 blades | 6 blades |
|---|---|---|
| Memory | 32 GB | 48 GB |
| Storage bandwidth | **~16 GB/s** | **~24 GB/s** |
| Storage capacity (512 GB drives) | 6 TB | 9 TB |
| Power (estimated) | ~90 W | ~135 W |

### 4.3 Drives are read-only in practice

The workload loads a model once and then streams weights forever. Weights are never modified. Writes occur only at initial model load, occasional KV paging, and logging.

Consequences for procurement:

- **Buy for read performance.** Sequential and 16 MB-block random read. Write speed is nearly irrelevant.
- **DRAM cache is not optional.** DRAM-less drives collapse on sustained reads because every access hits the mapping table. This matters more than the sequential figure on the box.
- **Gen3, not Gen4.** The link is PCIe 3.0. A Gen4 drive negotiates down and you have paid for nothing.
- **512 GB is the practical minimum.** Below that, fewer NAND dies means less internal parallelism and the drive will not saturate the link. Above it, capacity is cheap headroom.
- **Endurance is a non-issue.** Consumer TLC is rated around 600 TBW per terabyte, which at near-zero writes outlives the project.

Capacity sizing, model split across nodes:

| Model | Total | Per node, 4 nodes | Per node, 6 nodes |
|---|---|---|---|
| GLM-5.2 | 419 GB | 105 GB | 70 GB |
| DeepSeek V4-Pro | 900 GB | 225 GB | 150 GB |
| Kimi K3 | 1.56 TB | 390 GB | 260 GB |
| K3 plus a 445 GB local corpus | 2 TB | 500 GB | 334 GB |

### 4.4 The memory ceiling, and why it is no longer binding

The node's DRAM is 64-bit LPDDR4x, roughly **34 GB/s theoretical**.

Streamed weights cross that bus twice: once when storage DMAs them into DRAM, and again when the compute core reads them out. Constant alternation between DMA writes and CPU reads also incurs turnaround penalties, so real efficiency under this pattern is roughly 62% rather than the 80% a one-directional benchmark shows.

`34 GB/s x 0.62 / 2 = ~10.5 GB/s of effective streaming per node`

Against 4.0 GB/s of attached storage, that is **2.6x of headroom.** Storage is cleanly the bottleneck, which is the regime the architecture is designed for and the one where adding hardware helps linearly.

*(The previous design was built on a 32-bit LPDDR4 node giving roughly 3.0 GB/s effective, which sat uncomfortably close to its storage. That constraint is gone.)*

### 4.5 Power distribution

The module requires **4.0 V +/- 5% on VCC4V0\_SYS**. All other rails are generated on-module.

**Distribute 12 V across the backplane, step down to 4.0 V on each blade.** At 12 V a 6-blade board's ~135 W is 11 A rather than 34 A at 4 V, which keeps backplane conductors and the connector's power pins reasonable. Per-blade regulation also means a blade can be pulled and reseated without touching any other blade's supply.

Per-board DC input, never chained between boards. Metal standoffs tie board grounds together.

**The tight tolerance is the design point to get right, on each blade's own regulator.** The vendor's reference carrier schematic shows their regulation approach and should be the starting point.

### 4.6 Design-for-bring-up

- **Backplane switch IC candidate: Realtek RTL8367S/RTL8367RB** (5-port Gigabit, integrated PHYs, RGMII CPU port) or equivalent — enough ports for 4 blades plus uplink; 6 blades needs a second switch or a larger part
- **Blade connector: x4-class PCIe card-edge connector** (64-pin family, Amphenol or similar quality), power/SMBus/control on native pins, 4 of 8 differential pairs carrying the Ethernet MDI signal — see section 4.0
- **PCIe bracket geometry on the blade**, full-height (120.02 mm), so early bring-up can use a stock ATX/mATX test bench case with no custom mechanical work
- **All three PCIe 2.0 lanes routed on the blade** even if not populated. Free now, impossible later
- **Per-blade current sense on I2C**, because power draw is currently an estimate
- **Test points on every rail, on both the blade and the backplane**
- **Status LED and boot-select jumper per blade**
- **Board ID EEPROM per blade** for backplane addressing
- **Fan header and standoff airgap.** Four A76 clusters at load need real airflow, and thermal throttling will otherwise corrupt every measurement

### 4.7 Thermal and enclosure

**Closed, server-rack-style enclosure with one end-mounted fan forcing air across the blade row, not an open frame.** A sealed enclosure is what makes forced airflow actually work — an open frame lets air take the path of least resistance around the blades rather than through the gaps between them, which defeats the point of a fan.

**Rough sizing coincidence, worth naming so it doesn't get mistaken for a real constraint:** a 4-blade stack, at an estimated ~1 to 1.5 slot-pitches (20.32 mm each) per blade once real heatsinks are on, lands somewhere around 80-160 mm. A standard 120 mm fan happens to sit in the middle of that range. This is a convenient starting point, not a derived spec — bracket height (120.02 mm) and slot-stacking height are unrelated dimensions on different axes of the case, and the two both landing near 120 mm is coincidence, not physics.

**Fan spec: static-pressure-optimized, not high-airflow-optimized.** Forcing air through narrow gaps between blades populated with M.2 drives and a module is a higher-resistance path than open-air cooling. A radiator-style static-pressure fan is built for exactly this; a generic case fan will underperform at the same size and speed.

**Two things carried over from real rack-cooling practice, easy to miss until airflow silently fails:**

- **Blanking panels for empty blade slots.** The blade architecture's whole point is swap-on-failure — a pulled blade leaves a gap that becomes the path of least resistance, starving airflow to the blades still installed. Server chassis solve this with filler panels that keep the airflow channel sealed even with a slot empty. This matters more here than in a normal build, not less, given how central hot-swap is to the design.
- **A defined intake path, not just an exhaust fan.** One fan forcing air through is half the circuit — air needs a deliberate way in, whether that's a push design (fan at intake, vented exhaust elsewhere) or a pull design (fan at exhaust, open intake elsewhere). Either works; leaving it undefined does not.

**One geometric check worth recording:** the PCIe bracket opening at the back of each blade does not double as part of this airflow path. The bracket faces out the rear of a normal case; forced air moving along a fan-driven blade row travels sideways across the stack, a different direction entirely. The enclosure needs its own dedicated intake/exhaust venting, independent of the bracket openings.

**Still open, and genuinely needs real hardware to resolve, not more reasoning:** push vs. pull orientation, real per-blade heatsink selection, and actual thermal performance once RK3588 power-under-load is measured (`test-plan.md` step 5). Everything above is a sound starting topology, not a validated design.

---

## 5. Dense models

With 8 GB per node, a transformer layer of any dense model fits on a single node. Tensor parallelism is therefore a **speed choice, not a capacity requirement.**

Splitting a layer across N nodes divides the per-node weight read by N but costs a ring all-reduce of N-1 hops, four times per layer. On Ethernet at microsecond latency, groups of 4 to 6 remain reasonable, after which sync cost grows faster than the read shrinks.

**Everything not spent on tensor parallelism goes to pipeline depth**, which buys aggregate throughput across concurrent conversations rather than single-query latency.

### 5.1 The embedding imbalance

Large-vocabulary models have an embedding table bigger than any transformer layer. Qwen2.5-32B's 152k x 5120 embedding is 778M parameters, and untied means the LM head is another 778M, roughly 875 MB at Q4 against 274 MB for a layer.

In a pipeline the slowest stage sets the rate for all stages. Either give embed and unembed dedicated nodes, or shard the vocabulary dimension.

---

## 6. MoE models

### 6.1 The structural flip

Dense: nodes hold *different layers*, weights stay resident, activations move. A pipeline.

MoE: nodes hold *the same layer*, split by expert, in lockstep. A gang. Weights stream from local storage, activations stay. The same nodes are reused for every layer, per token.

Consequence: dense wants maximum nodes; MoE wants maximum storage bandwidth. In MoE mode this is fundamentally a storage device.

### 6.2 Sizing

| Model | Total | Layers | Experts per layer | Layer @Q4 | Total storage |
|---|---|---|---|---|---|
| Qwen3.5 | 397B | | | ~2.8 GB | ~223 GB |
| **GLM-5.2** | **744B** | 78 (75 MoE) | 256 + 1 shared, 8 routed | **5.46 GB** | ~419 GB |
| DeepSeek V4-Pro | 1.6T | | | ~9 GB | ~900 GB |
| Kimi K3 | 2.78T | 93 | 896 + 2 shared, 16 routed | **14.83 GB** | ~1.56 TB |

**All four layer sizes fit in one 4-node board's 32 GB**, with 17 GB spare even for Kimi K3. Node count above the minimum buys speed, linearly, not capability.

*GLM-5.2's figures are derived from the published config: hidden\_size 6144, moe\_intermediate\_size 2048, 78 layers with first\_k\_dense\_replace of 3. A SwiGLU expert is 3 x 6144 x 2048 = 37.75M parameters. Cross-check: 75 layers x 257 experts x 37.75M = 727.5B against a 743B model total, leaving a plausible ~16B for attention, embeddings and the MTP head.*

### 6.3 Streaming volume

Only routed experts move. GLM-5.2 at batch 1 streams 8 of 256 per layer, roughly **12.7 GB per token** against 419 GB of total weights.

With the shared expert pinned (section 6.5), that falls to about **11.3 GB**, and hot-expert caching brings it to roughly **10.4 GB**.

| Model | Effective bytes/token | 4 nodes (16 GB/s) | 6 nodes (24 GB/s) |
|---|---|---|---|
| GLM-5.2 | ~10.4 GB | **1.54 tok/s** | **2.31 tok/s** |
| GLM-5.2 at IQ3\_S | ~7.0 GB | ~2.3 tok/s | ~3.4 tok/s |
| Kimi K3 | ~21 GB | 0.76 tok/s | 1.14 tok/s |

Load imbalance across nodes, where the slowest node gates the layer, realistically adds 1.5x to 2x. Expert co-location by measured co-activation frequency reduces it.

### 6.4 Batching inverts

All batch elements must be processed before a layer's weights are discarded, so a step serves every query at once and **step time is the per-user latency.**

Expected distinct experts for batch B across E experts with k routed: `E x (1 - (1 - k/E)^B)`.

GLM-5.2 on a 4-node board:

| Batch | Experts touched | Step time | Throughput/token | **Per-user latency** |
|---|---|---|---|---|
| 1 | 8 | 0.65 s | 0.65 s | **0.65 s** |
| 8 | 57 | 4.6 s | 0.58 s | **4.6 s** |
| 32 | 163 | 13.2 s | 0.41 s | **13.2 s** |
| 128 | 252 | 20.4 s | 0.16 s | **20.4 s** |

Throughput improves about 4x. Latency degrades about 31x.

**Batch 1 for conversation. Large batch only for asynchronous, verifier-checked work.**

### 6.5 Speculative decoding does not pay here — measured, closed

**This is the most important correction in the current document set, and it inverts a claim carried through several earlier revisions. It was an open question; it no longer is.**

For a **dense** model, speculative decoding is close to free: verifying 4 draft tokens costs the same weight reads as verifying 1, so acceptance of 2 or more tokens is a direct multiplier. Reported figures of 1.5x to 2x come from that regime.

For a **streaming MoE**, it is not. Verifying B draft tokens requires loading the *union* of the experts those tokens route to, which follows the same expansion as batching:

| Draft tokens | Distinct experts | Bytes vs 1 token |
|---|---|---|
| 1 | 8 | 1.0x |
| 2 | 15.8 | 1.97x |
| 4 | 30.5 | **3.81x** |

At 4 draft tokens with a typical acceptance of around 2.2, you would pay 3.8x the bytes for 2.2x the tokens. **That is a net loss.**

The escape hatch was **routing correlation.** Consecutive tokens in similar context might route to substantially overlapping experts, in which case the real union would fall far below the independent-sampling estimate above.

**Measured on OLMoE-1B-7B (16 layers, 64 experts, top-8), 500 tokens, twelve seeds** (4 initial: 42, 42/2500-token, 7, 123; 8-seed follow-up sweep: 1, 2, 3, 99, 256, 777, 1000, 2024, run specifically to check whether seed 123's near-miss was a real tail or noise):

| Draft tokens (B) | Measured bytes multiplier, full 12-seed range | Break-even multiplier* | Verdict |
|---|---|---|---|
| 2 | 1.54x–1.86x | 1.25x | **NET LOSS, all 12 seeds** |
| 4 | 2.20x–2.59x | 1.82x | **NET LOSS, all 12 seeds** |
| 8 | 3.06x–3.91x | 2.35x | **NET LOSS, all 12 seeds** |

*Assumes dense-model-typical acceptance rates (1.6, 2.2, 3.4 tokens at B=2/4/8) — no draft model was built to measure this MoE's real acceptance rate.

Routing correlation is real — every layer measured below the independent-sampling prediction — but across twelve independent seeds it never once cleared break-even at any tested window size. Seed 123's B=4 multiplier (2.20x, the closest approach to the 1.82x line) was not exceeded on the low side by any of the 8 follow-up seeds: it was the natural tail of a real distribution, not the leading edge of a different regime.

**Disable speculative decoding in the first working runtime.** GLM-5.2 ships an MTP head, so the mechanism costs nothing to leave available for later, but this result closes the question under the stated assumptions. Reproduction script and all 12 seeds' raw output: `speculative_routing_experiment.py` and `seed_*/` in this repository. The only thing that could reopen this is a real draft model's measured acceptance rate substantially above the assumed figures.

### 6.6 Expert caching and the prefetch problem

Layer N+1's routing cannot be known until layer N's router runs, so streaming and compute serialise. Mitigations in order of certainty:

1. **Pin the shared expert.** GLM-5.2 has 1 shared expert per layer firing on every token, so it is 1 of 9 active. Pinning all 75 costs about 1.6 GB of RAM and removes **11% of streamed bytes, with certainty.** Kimi K3 has 2 per layer.
2. **LRU-cache hot routed experts.** Popularity is skewed, but the arithmetic bounds this: 15% of GLM-5.2's routed experts is 63 GB, against roughly 26 GB free on a 4-node board. Realistically you cache 5 to 6%, worth perhaps another 5 to 10% of loads. **GLM-5.2 uses Quantile Balancing during training, which may deliberately flatten the popularity distribution.** Measure before assuming.
3. **Speculative prefetch.** Predict layer N+1 routing from layer N's hidden state and eat occasional misses. Standard in the expert-offloading literature.

Combined, pinning plus caching gives roughly **0.80x to 0.85x** on streamed bytes. That is the basis of the 10.4 GB/token figure in 6.3.

### 6.7 Fault behaviour

Dense degrades gracefully: lose a node, re-shard across survivors, run slower. MoE **fails hard** below the node count needed to hold a layer.

Graceful degradation is a software feature, not a free property.

---

## 7. Context

KV cache scales with `attention layers x bytes per token per layer x context x batch`. **Total model size does not appear.**

KV is layer-local, so only the current layer's cache needs to be resident and it streams alongside the experts. Context is a **bandwidth tax, not a capacity cliff.**

GLM-5.2 on a 4-node board (5.46 GB layer, ~26 GB free, MLA with sparse attention):

| Context | Concurrent sessions |
|---|---|
| 32k | ~740 |
| 128k | ~185 |
| 256k | ~92 |
| 1M (full) | ~23 |

Kimi K3 leaves ~17 GB after its layer, giving roughly 120 sessions at 128k. FP8 KV cache roughly doubles both.

**Sparse attention reduces attention compute, not KV storage.** Keys must still be stored because any of them may later be selected.

---

## 8. What else the silicon does

Each node carries hardware the LLM path never touches:

- **8K H.265/H.264 encode and decode**, plus AV1 decode. A serious transcoder or NVR.
- **Mali-G610 MP4 with OpenCL 2.2 and Vulkan 1.1.** Unlike the previous design's VeriSilicon GPU, this has a mature open driver in Panfrost/Panthor and a large community running compute workloads on it.
- **6 TOPS NPU** with RKNN tooling.
- **4x Cortex-A55** idle during inference, available for housekeeping and prefetch scheduling.
- 48 MP ISP, camera interfaces, SATA-capable lanes.

### 8.1 Image generation

Compute-bound, not bandwidth-bound. The `FLOPs = 2 x params` heuristic underestimates convolutions by roughly three orders of magnitude, because a 3x3 kernel on a 128x128 feature map applies the same weights 16,384 times.

The RK3588 is far better at this than the previous node, with a real GPU and a working OpenCL path, but a consumer GPU still beats a full board by a wide margin. Usable for bulk generation with small distilled models; not a reason to build the machine.

### 8.2 Training

The NPU is INT8/INT16 and there is no floating-point backward path worth using. Training memory is 4x model size. Gradient all-reduce over Ethernet is brutal. Not viable, and not a gap that can be engineered around.

---

## 9. Open unknowns

Ordered by how much they gate everything else.

1. ~~Does speculative decoding pay on a streaming MoE?~~ **Resolved, section 6.5: no, measured net loss across 12 seeds. Disabled.**
2. **Real NVMe throughput on RK3588** at 16 MB block size, with and without O\_DIRECT. Every timing figure derives from ~3.2 GB/s on the x4 link.
3. **Expert popularity distribution** for GLM-5.2, given Quantile Balancing. Determines whether caching is worth 5% or 15%.
4. **Module dimensions confirmed** at 45 x 50 mm, but blade layout needs the mechanical drawing and LGA land pattern. The blade/backplane split (section 4.0) means this now blocks only one small blade, not the whole board.
5. **Real power under sustained load.** The ~22 W per node estimate is from product category, not measurement.
6. **Load imbalance factor** across nodes in gang mode. Estimated at 1.5x to 2x.
7. **Thermal behaviour** of four A76 clusters on one board under continuous matmul.
8. **4.0 V regulation** to +/- 5% across a 6-node board under transient load.

---

## 10. Build path

**Step 1, ~\$120.** One RK3588 single-board computer. Orange Pi 5, Rock 5C, NanoPi and similar run \$100 to \$150 and people already publish llama.cpp benchmarks on them. Measure real NVMe throughput, NEON quantized GEMM, power and thermals on the target silicon.

**Step 2, \$0.** ~~Run the speculative-decoding experiment against a routing trace.~~ **Done — see section 6.5. Net loss, disabled.**

**Step 3, ~\$120 more.** A second SBC. Two nodes over Ethernet is where the distributed runtime gets written and debugged: expert sharding, transport, streaming scheduler, weighted distribution, failover. **This is the year of work, and it is hardware-agnostic.**

**Step 4, free.** Blade schematic and layout in KiCad, adapting the vendor's published reference design. Backplane schematic and layout separately — much simpler, 4 layers, no LGA breakout.

**Step 5, ~\$400.** Fab **one blade** and the backplane. Populate the single blade with one module and one drive. Validate LGA reflow on the smallest, cheapest possible board before committing to more. Validate 4.0 V regulation, PCIe routing, the blade connector, Ethernet through the backplane switch, thermals.

**Step 6, ~\$400 more.** Fab three to five more blades once step 5 validates the design — each blade is small and cheap to iterate on independently. A bad blade revision costs one blade, not the whole board.

**Step 7, ~\$2,000.** Populate the remaining blades. This is a complete device: a frontier MoE model plus a local corpus with retrieval, in a box, with no network dependency — and any single blade can be swapped without touching the rest.

---

## 11. Prior art

**[Colibri](https://github.com/JustVugg/colibri)** (JustVugg, July 2026) is a pure-C inference engine that runs GLM-5.2 on a 25 GB consumer machine by streaming experts from NVMe. It is independent confirmation of this document's premise and the software this hardware would build on.

Its published benchmarks are the best available evidence that storage bandwidth is the binding constraint: **0.05 to 0.1 tok/s** on a 12-core laptop with one NVMe channel, against **6.84 tok/s** on a 6x RTX 5090 host with full expert residency. Same engine, same model, 57x apart, with the only difference being whether storage sits in the decode path.

Also relevant: expert offloading in the ML-systems literature (Mixtral-offloading and successors), and Petals for a different approach to distributing large-model inference.

---

## 12. A note on the numbers

Nearly every figure in this project that mattered was wrong at least once before being corrected against a primary source.

An entire interconnect design was invalidated when a datasheet revealed the intended fabric was a master-only flash controller. A per-token figure was wrong by 65x. A storage requirement was wrong by 93x. An image-generation estimate was optimistic by three orders of magnitude. A GPU was assumed to be Arm Mali for two days and turned out to be VeriSilicon. Speculative decoding was credited with a 1.5x to 2x gain that measurement across 12 seeds confirmed to be a net loss on this architecture.

The node itself changed late, from a 32-bit LPDDR4 SiP with six storage interfaces per node to a 64-bit LPDDR4x module with one PCIe 3.0 x4 link, after per-interface throughput caps made the first design's aggregate unreachable.

The software has the same character. Four latent asyncio bugs survived their own passing unit tests and only surfaced under a newer Python or under integration: an orphaned `drain()` coroutine that silently skipped backpressure, a swallowed `CancelledError` that left tasks stuck cancelling forever, a `close()` that awaited itself, and a server shutdown that deadlocked because one flag meant two things. A fifth was subtler: `failover.py` passed thirteen tests against a fake `FleetTable` that fires callbacks the real one does not, so a caller wiring it to the real class would have got a coordinator that silently never resharded. Unit tests did not catch any of these; a newer interpreter and an integration harness did.

Read everything here as *best current estimate, several times revised.* Something else will invalidate something else.
