# Swarm

**A distributed inference board for frontier-scale MoE models, built from cheap SiP microprocessors.**

Design stage. Nothing has been built. No number in this repository is anchored to a measurement on real silicon.

---

## The idea in one paragraph

Mixture-of-Experts decode at batch 1 is **memory-bandwidth-bound**, not compute-bound. Every weight is read once and discarded. A 744B model activates ~40B parameters per token, so ~95% of the weights sit idle at any step. That means the binding constraint isn't FLOPs, it's how fast you can move expert weights from storage into memory.

A GPU has enormous compute and one memory bus. This design inverts that: **many modest processors, each with its own private storage channels.** Aggregate bandwidth scales with hardware count instead of being fixed at purchase. Nine nodes with 36 storage devices give ~15.7 GB/s on one 100×100 mm board, enough to hold a layer of any current frontier open-weight model.

## The general law

> **Good wherever weights are used once and discarded. Bad wherever weights are reused heavily.**

| Workload | Weight reuse | Fit |
|---|---|---|
| MoE decode, batch 1 | none | best case |
| Dense LLM decode | none | good |
| Heterogeneous multi-model fleet | none | excellent |
| LLM prefill (long prompts) | heavy | poor |
| Diffusion / convolutions | massive | poor |
| Video DiT | reuse + quadratic attention | hopeless |
| Training | n/a (INT8 NPU, no backward pass) | impossible |

This is not a slow GPU. It's the opposite of a GPU, and MoE decode happens to sit at the far end from where GPUs are strong.

## Current target board

| | |
|---|---|
| Nodes | 9 × Octavo OSD32MP2 (STM32MP257 + DDR4 + PMIC in a 21×21 mm SiP) |
| Package pitch | 1.0 mm, 437 ball, 4-layer routable, hand-solderable |
| RAM | 18 GB (assuming 2 GB/node, **unconfirmed**) |
| Storage | 27 × eMMC 32 GB + 9 × BGA NVMe + 18 × Octal NAND |
| Aggregate bandwidth | ~15.7 GB/s |
| Interconnect | RGMII MAC-to-MAC chain via each node's 2+1 integrated switch, no PHYs |
| Board | 100×100 mm (150×150 under consideration for routing headroom) |
| Power | ~65 W estimated |
| Cost | ~$2,775 CAD estimated |

Every frontier open-weight model's layer fits on one board:

| Model | Total | Layer @ Q4 | Boards |
|---|---|---|---|
| Qwen3.5 | 397B | ~2.8 GB | 1 |
| GLM-5.2 | 744B | 5.25 GB | 1 |
| DeepSeek V4-Pro | 1.6T | ~9 GB | 1 |
| Kimi K3 | 2.78T | 14.83 GB | 1 |

Board count above the minimum buys **speed**, linearly. It does not buy capability.

## Status

- [x] Architecture designed and documented
- [x] Chip selection (see `docs/chip-selection.md`)
- [x] Board floorplan and component tally
- [ ] Octavo pricing and DDR4 capacity options, **blocking**
- [ ] Verify llama.cpp Vulkan runs on Mali-G52 via STM32MP257F-DK (~$148), **blocking**
- [ ] Schematic
- [ ] PCB layout
- [ ] Distributed runtime

**The two blocking items are the whole project right now.** Nothing downstream is worth doing before they're answered.

## Prior art and credit

**[Colibrì](https://github.com/JustVugg/colibri)** by JustVugg (published July 2026) is a pure-C inference engine that runs GLM-5.2 on a 25 GB consumer machine by streaming experts from NVMe. It is independent confirmation of this project's core thesis, and it is the software this hardware would build on rather than replace.

Colibrì's own measured floor is **0.05–0.1 tok/s** on a 12-core laptop, because a laptop has one NVMe channel. Its measured ceiling on a 6× RTX 5090 host with full expert residency is **6.84 tok/s**. That 57× gap between streaming and resident, on identical software and model, is the clearest available evidence that storage bandwidth is the binding constraint, which is exactly what this board is built to widen.

This repository is hardware. Colibrì is software. They are complementary, and any working version of this design would port and credit it.

Other relevant prior work: expert offloading in the ML-systems literature (Mixtral-offloading and successors), and Petals for a different take on distributing large-model inference.

## Documents

- `docs/architecture.md`, full architecture, performance math, open unknowns
- `docs/dials.md`, 15 tunable parameters, what each trades, and which are permanent at fab time
- `docs/chip-selection.md`, the evaluation trail across ~20 candidate parts

## Honest limitations

- **Slow.** GLM-5.2 at roughly 0.4–0.8 s/token on one board. Kimi K3 at ~1.6 s. This is asynchronous-work hardware, not a fast interactive assistant.
- **Nothing measured.** All figures derive from datasheets, vendor specs, and published model configs.
- **Prefill is bad.** Long-prompt processing is reuse-heavy and lands on the wrong side of the general law.
- **A GPU wins for anything that fits in VRAM.** A single RTX 5090 beats a 12-board tower by 20–30× on a 32B model.
- **The software is the hard part**, and it's the part that doesn't exist yet.

## Why build it anyway

Because a \$2,000 unified-memory box tops out at 128 GB, and every frontier open model is 400 GB to 1.6 TB. The alternative for running these locally is roughly \$185,000 of GPUs and a three-phase electrical service. This is ~\$2,775 and a wall outlet.

Not faster. Possible where it currently isn't.

## License

Hardware: [CERN-OHL-P v2](LICENSE). Documentation: CC-BY-4.0.
