# Software Architecture

### The runtime this project actually has to build

The hardware documents describe what the boards are. This describes what has to run on them. Most of the difficulty in this project is here, not in the PCB.

**Status: layers 3 and 4 are built and tested. Layers 1, 2 and 5 are not.** This document remains the map of the work, organized so pieces can be built, tested and reasoned about independently, and so it is clear which parts are genuinely novel versus standard plumbing.

| Layer | Status |
|---|---|
| 1, inference core | not started — needs real silicon to validate NEON kernels |
| 2, single-node expert streaming | not started — blocked on the storage measurements in `test-plan.md` step 4 |
| 3, node identity and discovery | **built, 9 tests** (`node_identity.py`) |
| 4, distributed scheduler | **built, 69 tests** (`rpc.py`, `sharding.py`, `gang_sync.py`, `pipeline.py`, `failover.py`) |
| Integration Suite | **built, 9 tests** (`test_integration.py`) |
| 5, interfaces | not started |

Plus `test_integration.py`, which wires layers 3 and 4 together against real sockets and the real `FleetTable`. **87 tests total.**

---

## 0. Why this is organized in layers

Each layer depends only on the one below it, and each can be tested in isolation before the next is built on top:

- Layer 1 is testable on any x86 machine with a GGUF file.
- Layer 2 is testable on one node, real or simulated, with a slow disk standing in for the storage tiers.
- Layer 3 is testable with fake nodes on localhost.
- Layer 4 is testable with the simulated multi-node harness described in [test-plan.md](test-plan.md), before any board exists.
- Layer 5 is testable once layer 4 works, real or simulated.

**This ordering is also the build order.** Nothing above a layer should be started before that layer has a passing test, because the layers below define the interface everything above depends on.

---

## 1. Inference core

The engine that turns a GGUF file into logits. Mostly borrowed, not built.

- **[Colibri](https://github.com/JustVugg/colibri)**, ported from x86 AVX2 to ARM NEON. This is the one place hand-written low-level code is unavoidable, and it is a bounded porting task rather than new design: swapping intrinsics, not inventing an inference engine. See [chip-selection.md](chip-selection.md) and [architecture.md](architecture.md) section 11 for why this is the software this hardware is built to run.
- **Quantization support.** Q4_K_M as the floor. IQ3_S per [dials.md](dials.md) dial 6 if the accuracy cost proves acceptable.
- **GGUF loading adapted to stream**, reading experts from local NVMe on demand rather than holding the full model in RAM. This is the fork point from stock llama.cpp/Colibri: standard engines assume the model fits in memory, and this one cannot assume that by design.

**Language: C.** Performance-critical, and the only layer where that is true for a good reason rather than by default.

---

## 2. Single-node expert streaming

Everything a single node needs to decide what to read and when, before any node talks to another.

- **Expert cache manager.** Pins the shared expert per [dials.md](dials.md) dial 14 (certain, ~11% reduction), LRU-caches routed experts within whatever memory remains after the shared ones and the current layer's KV cache.
- **Prefetch predictor.** Predicts layer N+1's likely experts from layer N's hidden state, issues speculative reads while layer N still computes. Standard technique in the expert-offloading literature.
- **Storage I/O layer.** O_DIRECT reads at the block size [test-plan.md](test-plan.md) step 4 determines empirically, bypassing the page cache since weights are read once and never reread.

**Language: C, calling into layer 1, or Python with a thin C extension for the I/O path.** Decide after step 4's measurement shows whether Python's overhead is negligible next to storage latency, which it very likely is.

---

## 3. Node identity and discovery

What makes a node a citizen of a fleet rather than an isolated machine. This is where the [compatibility contract](compatibility.md) becomes code.

- **Capability descriptor.** What a node reports at boot: memory, measured storage bandwidth (not rated, per the contract), compute, negotiated link speed, supported precisions. Contract section 2.3, restated as a schema.
- **Node discovery.** How nodes find each other. **Built** — and not with mDNS, which this document originally proposed. `node_identity.py` uses UDP broadcast/multicast carrying a JSON capability descriptor, because the stdlib-only constraint rules out Avahi on a minimal ARM image. That makes discovery a genuinely custom component rather than borrowed plumbing; see section 7. **Known limitation:** local multicast does not loop back on WSL2 in either NAT or mirrored networking mode, so discovery cannot be exercised there. The logic is unit-tested directly against `FleetTable`; the transport gets validated on real hardware at `test-plan.md` step 7.
- **Health and heartbeat.** Is a node alive, is it lagging behind the fleet, should it be dropped from the current job.

**Language: Python.** No part of this is performance-sensitive. It is bookkeeping and network calls, and the interpreter's overhead is irrelevant next to network latency.

---

## 4. The distributed scheduler

**This is the actual project.** Everything below exists to make this layer possible; everything above exists to make it useful.

- **RPC layer.** How nodes exchange activations and results. A hand-rolled protocol over raw sockets is enough at the traffic volumes in [architecture.md](architecture.md) section 3.1 (roughly 12 KB per layer boundary at decode), and avoids a gRPC dependency on constrained boards.
- **Weighted sharding.** Assigns experts to nodes in proportion to measured capability, per [dials.md](dials.md) dial 4 and compatibility contract section 3.1. This is the piece that makes a mixed fleet of old and new boards actually work rather than being gated at the speed of the slowest member.
- **Layer-sync coordinator, MoE mode.** Every node must finish layer N before any node starts N+1, since MoE mode is a gang: all nodes work the same layer in lockstep. See [architecture.md](architecture.md) section 6.1.
- **Pipeline coordinator, dense mode.** Hands activations down a chain of nodes, each holding different layers. See section 5.
- **Failover.** A node drops mid-generation: re-shard its experts across survivors or fetch from cold storage, resume at the next layer boundary. Required by compatibility contract section 3.3, and it is the same mechanism whether the node dropped from hardware failure or because it is a desk board that just logged in for the morning (see the day/night worked example in [dials.md](dials.md)).
- **Batch scheduler.** Routes interactive requests to batch 1, routes asynchronous or verifier-checked work to larger batches, per dial 3. This is the component that decides, per request, which regime it is in.

**Language: Python.** This layer is coordination logic, not computation. Nothing here touches a matmul.

**This is where the simulated multi-node harness in [test-plan.md](test-plan.md) section 2 does its work.** Every component in this layer can be built and tested against fake nodes on localhost with `tc netem` injecting realistic latency and artificial bandwidth caps, before a single board exists.

---

## 5. Interfaces

What a user, another program, or an idle desk actually talks to.

- **API server.** OpenAI-compatible endpoint, so existing tools and libraries work against it unmodified.
- **Instance manager.** Which model is loaded on which node group, how many instances are running, per dials.md dial 4.
- **Job dispatcher.** For the desk, building, and federation scenarios: decides which pool of nodes gets which task, and manages nodes joining or leaving a pool (the day-to-night handoff described in the dials.md worked example, or a WAN-relayed job between a company's buildings). This is closer to a build-farm job queue than to the tight coordination in layer 4, since jobs dispatched this way do not need step-by-step synchronization with each other.

**Language: Python.**

---

## 6. Cross-cutting, supporting

- **Config and state store.** What is currently deployed where. SQLite per board is enough; this does not need to be distributed.
- **Logging and metrics.** Per-node power, temperature, throughput. Feeds the results table in [test-plan.md](test-plan.md) section 6 directly, and is the ongoing version of the same measurement discipline.
- **CLI and bring-up tooling.** Flash a node, run the benchmark suite from the test plan, join a node to a fleet.

---

## 7. What is genuinely novel versus what is plumbing

Worth stating plainly, since it shapes where effort should go.

**Borrowed or standard:** layer 1's inference core (Colibri does the hard part), an OpenAI-compatible API shape, SQLite for local state. None of this needs inventing.

**Novel, and specific to this project:** the expert cache and prefetch predictor tuned for a storage-bound rather than memory-bound machine, weighted sharding across a genuinely heterogeneous fleet, the MoE gang-mode layer-sync coordinator, and the day/night pool membership logic. These do not have off-the-shelf implementations to adapt, because almost nobody else has built an inference system where the bottleneck is deliberately moved to cheap storage.

**The one-month target — met.** It was: layers 3 and 4 built and validated against the simulated harness, plus the OLMoE speculative-decoding trace resolving whether layer 2's prefetch predictor should include speculative decoding at all. Both are done, and the speculative-decoding answer was *no* — see [dials.md](dials.md) dial 13. A second experiment settled request batching the same way (dial 3): also not worth building.

Layer 1's ARM port and any real-hardware validation of layers 2 through 5 still wait on owning actual silicon.

---

## 8. Suggested language split, restated

| Layer | Language | Why |
|---|---|---|
| 1, inference core | C | performance-critical, mostly porting not inventing |
| 2, single-node streaming | C, or Python with a thin C I/O extension | decide after measuring whether Python overhead matters next to storage latency |
| 3, node identity | Python | pure bookkeeping and network calls |
| 4, distributed scheduler | Python | coordination logic, not computation |
| 5, interfaces | Python | standard web service work |

**No layer requires Rust.** The C surface is small, bounded, and mostly a porting task rather than new systems design. Everything else is Python, permanently, not as a prototype awaiting a rewrite: the bottleneck throughout this system is storage bandwidth and network latency, and Python's interpreter overhead does not compete with either.

---

*Companion to [architecture.md](architecture.md), [dials.md](dials.md), [compatibility.md](compatibility.md) and [test-plan.md](test-plan.md). Layers 3 and 4 are built; 1, 2 and 5 are not.*