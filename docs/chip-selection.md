# Chip Selection

### The evaluation trail, including the wrong turns

This document exists because the *decision process* is the actual work. Roughly twenty parts were evaluated and nineteen rejected. The reasons are more useful than the conclusion, and several rejections were reversals of earlier positions.

---

## The screen

Five hard criteria, applied in order. A part failing any of the first three is out regardless of anything else.

| # | Criterion | Why |
|---|---|---|
| 1 | **SIMD or NPU** | No vector unit means MoE decode flips from bandwidth-bound to compute-bound, which destroys the entire premise |
| 2 | **≥512 MB on-package memory** | Sets how many nodes hold one model layer, which cascades into board count, cost, power, and KV headroom |
| 3 | **≥4 GB/s memory bandwidth** | DDR2/DDR3-class is disqualifying |
| 4 | **≥2 storage controllers, ≥1 Gbps link** | Storage channels are the entire moat; link speed caps tensor-parallel group size |
| 5 | **Package pitch ≥0.8 mm** | Below this, layer count and assembly cost escalate past hobbyist reach |

A sixth criterion emerged late and turned out to matter as much as any of the five:

| 6 | **A GPU with a Vulkan or OpenCL path** | Decides whether you port an existing inference engine or write one from scratch |

**Criterion 2 is the master variable.** It's also the one a board designer cannot influence, it's fixed by chip choice. Every performance figure in this project descends from it.

---

## First pass, ~15 parts, screened out fast

The initial survey covered edge-AI SoCs and NPU accelerators broadly. Rejected, grouped by failure mode:

**No on-package memory** (fails #2, you'd route DRAM yourself on a first BGA project, defeating the purpose):
- Rockchip RV1103, RV1126B
- Axera AX630C
- MediaTek MT7981B (also no NPU, router silicon)

**Accelerator, not a system** (fails #4, no storage controllers, needs a host):
- Hailo-8, Hailo-10H
- Google Coral / Edge TPU

**Memory far too small** (fails #2 badly. MCU-class, tens of MB):
- Alif Ensemble

**Availability** (fails on procurement rather than specs):
- Sophgo SG2000, SG2002, stock could not be confirmed

**Not swarm candidates, kept as comparison points:**
- NVIDIA Jetson Orin Nano ($249), the honest baseline any custom board must beat. Its measured 0.095 img/s on SDXL (RidgeRun) is the reference number for image generation.
- Ambarella CV-series, Qualcomm QCS, automotive/industrial pricing, no accessible supply

**Selected from this pass: Canaan Kendryte K230D.**

---

## K230D (Canaan Kendryte), the original target

$13.70 at LCSC, 11×11 mm, 256-ball BGA.

| Criterion | Result |
|---|---|
| 1. SIMD/NPU | ✅ RVV 1.0 on CPU1, KPU ~0.35 TOPS (derived from ResNet50 FPS, not official) |
| 2. ≥512 MB | ❌ **128 MB** |
| 3. ≥4 GB/s | ✅ 6.4 GB/s (16-bit LPDDR4 @ 3200, half the non-SiP K230's 12.8) |
| 4. Storage + link | ⚠️ 2× SD/eMMC, but **no Ethernet MAC and no PCIe** |
| 5. Pitch ≥0.8 mm | ❌ **0.65 mm**, full 16×16 grid |
| 6. GPU/Vulkan | ❌ 2.5D compositor only, write the runtime from scratch |

**Kept for a long time on price alone.** At $107/GB of on-package memory it buys the cheapest bandwidth per dollar of anything evaluated, because sixteen cheap chips give you 32 eMMC controllers per board.

**What eventually disqualified it:**

- **128 MB forces high board counts.** Kimi K3's 14.83 GB layer needs 8 boards. GLM-5.2 needs 3.
- **The interconnect had to be invented.** No MAC, no PCIe, and the OSPI "DRT200" that v1 assumed was a peer-to-peer fabric turned out to be a **master-only flash controller**, there is no SPI slave mode. The replacement was a USB 2.0 OTG ring at ~45 MB/s with ~125 µs per hop, which caps tensor-parallel groups at 3–6 chips and means a single dead node breaks the loop.
- **The ball map was never obtained.** At 0.65 mm pitch you escape ~2 ball rings per signal layer, so 4 layers gets ~192 of 256 balls, the inner 8×8 must be all power/ground for 4 layers to work. Plausible given ~24 supply rails, never confirmed. That risk sat unresolved over the entire design.
- **No GPU means no existing software path.** A distributed MoE runtime, expert-streaming scheduler, quantized kernels and KV manager, all from scratch, against an undocumented NPU. Easily a year with a real chance of failure.

**Still the cheapest-bandwidth option on paper.** Rejected because buildability and software risk dominate a first project.

*Fallback noted during this pass: Allwinner T113-S3 ($18.39) as a head node, it has the Gigabit MAC the K230D lacks. Its HiFi4 DSP was evaluated as a softmax fallback unit and rejected: ~4–8 GFLOPS against CPU1's ~12.8 with RVV, so it's worse at the job than the chip it would assist.*

---

## Rejected: Microchip SAM9X60D1G

SiP with 128 MB DDR2, 4 Gb NAND, Ethernet PHY, PMU, genuinely nice integration, single-sided, hand-solderable.

| Criterion | Result |
|---|---|
| 1. SIMD/NPU | ❌ **ARM926EJ-S, ARMv5, no SIMD, likely no FPU** |
| 2. ≥512 MB | ❌ 128 MB |
| 3. ≥4 GB/s | ❌ **DDR2, ~0.7 GB/s** |

~0.6 GOPS per node against the K230D's ~0.35 TOPS, roughly **580,000× worse**. GLM-5.2 would need ~15 s of pure compute per token per board before any weight streaming.

**The lesson:** integration quality is worthless if the core can't keep up. This is a 2001 architecture in a 2026 package, aimed at PLCs and HMI panels.

---

## Rejected: Octavo OSDZU3 (Zynq UltraScale+ SiP)

Passes the entire screen, 2 GB LPDDR4, ~9.6 GB/s, 1.0 mm pitch, quad A53 with NEON, 360 DSP slices (~0.7 TOPS), 2× GbE, USB 3.0, PCIe.

**Rejected on cost.** Bare XCZU3EG is $250–350; with 2 GB LPDDR4 and two IRPS5401 PMICs the SiP is plausibly $500–700/node. Nine nodes is $4,500–6,300 in chips alone, roughly **16× worse per unit of memory** than the STM32MP2 route, plus 5–15 W/node and 820 mm² each. No Vulkan (Mali-400 MP2, OpenGL ES 2.0 only), so no llama.cpp path either.

**Kept on file for a different project.** The FPGA fabric can build a hardware softmax unit, native MXFP4 datapaths, and arbitrary custom number formats, which is the right apparatus for testing whether quantization path-dependence findings measured in simulation survive on silicon that genuinely computes in those formats. That's a separate project with its own year-long learning curve.

---

## Rejected: Octavo OSD32MP15x

Same family as the eventual target, one generation back. 1.0 mm pitch, 302-ball, up to 1 GB DDR3L, integrated STPMIC1.

| Criterion | Result |
|---|---|
| 1. SIMD/NPU | ⚠️ NEON only, **no NPU, no GPU**. ~6.4 GFLOPS/node |
| 3. ≥4 GB/s | ❌ DDR3L, ~2.1 GB/s |
| 6. GPU/Vulkan | ❌ none |

~200× less compute than the MP2 generation, and priced the same: **$137.98** for the OSD32MP157F-1G against a bare STM32MP157F at $25.

**The lesson that came out of this:** Octavo's premium is for the *integration* (SiP assembly, PMIC, DDR routing, test) and costs roughly the same regardless of what silicon is inside. So if you're paying it at all, always take the newest part in the family. There is no saving in going older, only capability loss.

---

## Rejected: SOMs, a category error at any scale

Two vendors, two chip families, same conclusion.

| Part | Contents | Price |
|---|---|---|
| MYIR MYC-LR3568 | RK3568, 2 GB LPDDR4, 16 GB eMMC | $250.29 |
| MYIR MYC-LR3568 (4 GB) | RK3568, 4 GB LPDDR4, 32 GB eMMC | $221.11 |
| MYIR MYC-LD257 | STM32MP257D, 2 GB LPDDR4, 8 GB eMMC | $270.75 |

The RK3568 modules pass the screen comfortably, quad A55 at 2 GHz, 1 TOPS NPU, Mali-G52 with OpenCL 2.0 and Vulkan, LGA package that's hand-solderable and visually inspectable. On the merits it's an excellent part.

**But: $221–307 per node, and each module exposes only one eMMC.** The RK3568's other two SDMMC controllers aren't reachable through the 381-pin LGA. At 45×43 mm you fit four per 100×100 mm carrier:

| | 4× MYC-LR3568 | 9× bare-chip board |
|---|---|---|
| Storage channels | 4 | 27 |
| Bandwidth | ~1.1 GB/s | ~7.6 GB/s |
| Cost | ~$1,100 | ~$1,450 |

**Seven times less bandwidth for similar money.** GLM-5.2 would run at ~11 s/token.

**The general rule:** SOM vendors price by module, and modules cost $200+ regardless of contents, the price is assembly, test, warranty, and low volume. SOMs are built for products that use one or two. This design needs dozens. The model breaks at that scale.

---

## Rejected as first target: Octavo OSD32MP2-PM

The dense variant of the selected part: same STM32MP25 and DDR4 in **9×14 mm** instead of 21×21 mm. 3.5× the area density, which would allow 12–16 nodes per board instead of 9.

**Rejected on package pitch: 500 balls at 0.5 mm.**

That's ~4 balls/mm², nearly 5× the density of the full SiP, and *worse than the K230D's 0.65 mm*, the exact problem the full package solves. It implies 8+ layers, via-in-pad (filled and capped, a fab upcharge), professional assembly with X-ray verification, and designing 12–16 power supplies since the PMIC isn't included.

The trade is ~30% more nodes per board for roughly 10× the build difficulty. And the density isn't needed: at 2 GB/node, **nine nodes already holds any frontier model's layer.**

**Kept as a version-three shrink option** once the design is proven.

---

## Considered: bare STM32MP257F

~$26–37 CAD at Mouser, genuinely stocked (1,720 units of FAI3, 1,099 of FAK3), $25.82 at volume. Same silicon as the selected part without the SiP wrapper.

| Route | Per node | You route DDR4? |
|---|---|---|
| OSD32MP2 SiP | ~$150 | No |
| Bare MP257 + DDR4 + STPMIC2 | ~$60 | **Yes, ×9** |

**A third the cost per node**, and MYIR's modules prove the chip runs 2 GB LPDDR4. Genuinely the better economics.

**Not chosen for board one** because nine LPDDR4 fly-by routes at 3200 Mbps on a first BGA project is how projects die on the bench, and the package pitch on the bare part is unverified, so the layer-count risk that killed the K230D reappears.

**Kept as the version-two option.** The software carries over unchanged.

---

## Runner-up: Octavo OSD32MP2

21×21 mm, 437 ball, **1.0 mm pitch.** STM32MP257 + DDR4 + STPMIC2 + EEPROM + oscillators + passives in one package.

| Criterion | Result |
|---|---|
| 1. SIMD/NPU | ✅ 1.35 TOPS NPU, dual Cortex-A35 @1.5 GHz with NEON, Cortex-M33 @400 MHz |
| 2. ≥512 MB | ✅ DDR4 integrated (**capacity unconfirmed, the one blocking number**) |
| 3. ≥4 GB/s | ✅ DDR4 |
| 4. Storage + link | ✅ **3× SDMMC**, 2× Octal SPI, PCIe Gen2, **3× GbE with 2+1 switch** |
| 5. Pitch ≥0.8 mm | ✅ **1.0 mm, 4-layer routable, hand-solderable** |
| 6. GPU/Vulkan | ✅ **Vulkan 1.1, OpenCL 1.2, OpenGL ES 3.1** |

**What it solves, in order of importance:**

1. **The software problem.** Vulkan 1.1 means llama.cpp's Vulkan backend has a real chance of running. The work shrinks from *invent a distributed MoE runtime* to *port ~2,400 lines of C from x86 AVX2 to ARM NEON and add multi-node RPC*. This is the single largest risk reduction in the whole evaluation.
2. **The layout problem.** 1.0 mm pitch removes the ball-map risk that hung over the entire K230D design. 4 layers, and you could hand-place it on a hot plate.
3. **The interconnect problem.** 3× GbE with an integrated 2+1 switch means nodes chain MAC-to-MAC over RGMII, no PHYs, no hub ICs, no USB ring to invent, no bypass links, no ring-segmentation constraint. 125 MB/s per link at microsecond latency, against the USB ring's 45 MB/s at 125 µs.
4. **The power problem.** Integrated STPMIC2 eliminates the 24-rail, 20 A-on-0.8 V distribution design.
5. **The DDR problem.** Octavo did the fly-by routing.

**What it costs:** roughly 4–5× the bare chip per node, and 21×21 mm limits you to 9 per 100×100 mm board.

**Still open (at the time this was the pick):** unit price and DDR4 capacity options. Also unverified: whether the 2× Octal SPI are real xSPI flash controllers or the 50 Mbit peripheral kind, which decides whether 18 NAND parts belong on the board.

### Why it was superseded

The OSD32MP2 was the target until per-interface throughput figures were checked against controller limits rather than device datasheets. Four findings, in the order they landed:

1. **The GPU is VeriSilicon Vivante GC8000, not Arm Mali.** The open driver is Etnaviv, whose Vulkan support does not cover this generation, and community reports indicate ggml-vulkan fails at instance creation on the proprietary stack.
2. **Memory is 32-bit LPDDR4 at ~9.6 GB/s, shared.** Streamed weights cross it twice, capping effective streaming at roughly 3.0 GB/s per node.
3. **The Cortex-A35 is Armv8.0-A and lacks `SDOT`**, giving roughly 3.5–5 GFLOPS on quantized GEMM where ~4.4 was needed. Compute sat uncomfortably close to storage.
4. **Per-interface caps make the aggregate unreachable.** eMMC via the SDMMC controller near 200 MB/s, PCIe Gen2 x1 at ~410 MB/s, Octal SPI near 115 MB/s. Six devices per node reach ~1.24 GB/s, not the 1.74 assumed.

Nine nodes with 54 storage devices therefore gave ~11.2 GB/s, at ~$2,775 and 63 BGA placements.

**It remains a reasonable part** and the SiP integration argument still holds. It is simply beaten by a module that costs similar money and carries 64-bit memory, an SDOT-capable CPU, and a PCIe 3.0 x4 link.

---

## Selected: RK3588 LGA module

**Banana Pi BPI-LM7** or **ArmSoM LM7** — two vendors shipping the same 506-pin LGA pinout. ~$268 USD (~$368 CAD) at 8 GB RAM plus 32 GB eMMC; a 32 GB RAM option also exists.

| Criterion | Result |
|---|---|
| 1. SIMD/NPU | 4× Cortex-A76 **with SDOT**, 6 TOPS NPU, Mali-G610 MP4 |
| 2. Memory capacity | **8 GB default, up to 32 GB** |
| 3. Memory bandwidth | **64-bit LPDDR4x, ~34 GB/s** |
| 4. Storage and link | **PCIe 3.0 x4 plus 2× PCIe 2.0 x1**, GMAC Ethernet |
| 5. Package | **LGA 506-pin, solder-down**, 45×50 mm |
| 6. Compute API | OpenCL 2.2, Vulkan 1.1, **mature Panfrost/Panthor open driver** |

**The confirmation that decided it.** The published pin function list shows PCIE30_PORT0 carrying lanes 0 and 1 and PCIE30_PORT1 carrying lanes 2 and 3, which combine to the full x4. All bifurcation control pins are broken out. This matters because CM4-form-factor RK3588 modules such as the Radxa CM5 expose only PCIe x1 through the connector, capping a node near 400 MB/s.

**Why a module here and not for MYIR.** The rule established above is that a SOM is a category error when its price buys packaging rather than capability. MYIR's RK3568 module at $250 carried 2 GB and one eMMC channel. This one carries 8–32 GB of 64-bit LPDDR4x, a PCIe 3.0 x4 link, and SDOT cores, at similar money. The rule holds; this passes it.

**Against the OSD32MP2 board (both priced with post-shortage NVMe figures):**

| | OSD32MP2 ×9 | RK3588 ×4 |
|---|---|---|
| Cost | ~$2,775 | **~$2,470** |
| Memory | 18 GB | **32 GB** |
| Storage bandwidth | ~11.2 GB/s | **~16 GB/s** |
| Storage capacity | 864 GB | **6 TB** |
| Parts to place | 63 BGA | **4 LGA + 12 sockets** |
| Compute headroom | ~1× | **~4×** |
| Software path | unproven (Etnaviv, no Vulkan) | **llama.cpp runs on RK3588 today** |

**Also rejected in this round:** Radxa CM5, disqualified by its CM4-compatible pinout exposing only PCIe x1. FriendlyElec CM3588, which breaks out four PCIe Gen3 lanes but uses a board-to-board connector rather than solder-down. Firefly Core-3588L, functionally similar to the LM7 at $397 CAD for 4 GB against $368 for 8 GB — dominated on price and memory.

**Still open:** the LGA land pattern and mechanical drawing for layout, real NVMe throughput on RK3588 silicon, sustained power draw, and 4.0 V regulation to ±5% under transient load. The dev-board question got easier alongside this pivot too — RK3588 SBCs (Orange Pi 5, Rock 5C, NanoPi) run $100–150 with published llama.cpp benchmarks, a materially better first purchase than a bare-silicon dev kit at similar cost.

---

## What the trail actually shows

**The cheapest part is not the cheapest board.** The K230D at $13.70 buys cheaper bandwidth per dollar than anything else evaluated, and it lost anyway. A board you can't route, with an interconnect you have to invent, running software you have to write from scratch, costs more than the parts do.

**Price per node is the wrong metric.** Price per gigabyte of on-package memory is closer: K230D $107/GB, OSD32MP2 ~$75/GB at 2 GB, bare MP257 ~$30/GB, MYC-LR3568 $55/GB, OSDZU3 $250–350/GB. But even that misses it, because storage channel count comes from node count, so cheap nodes buy more channels.

**The real ranking is by total risk**: layout risk, interconnect risk, software risk, procurement risk. The OSD32MP2 wins by reducing three of those four to near zero, and it's worth a 4× price premium per node to do it.

**And the master variable was never under the designer's control.** On-package memory capacity sets board count, which sets bandwidth, cost, power, and KV headroom. Every figure descends from one number in a package spec. That's the argument for keeping the board layout as chip-agnostic as a BGA footprint allows.
