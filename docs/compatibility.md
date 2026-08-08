# Compatibility Contract

### What every board in this family guarantees, across generations

The point of this document is a single promise:

> **Any board in this family connects to any other board in this family, regardless of generation, node type, or when it was built.**

Boards can be added, replaced, upgraded or removed one at a time. A five-year-old entry-tier board and a brand-new full-tier board share the same fleet and contribute to the same model. A dead board is unplugged and a different one takes its place. Nothing about a working board ever becomes obsolete by connector, protocol, or vendor decision.

That promise costs almost nothing to make now and is impossible to retrofit later. This page states exactly what it requires.

---

## 1. Why this is achievable

Boards interconnect over **standard Ethernet**, not a proprietary bus.

Ethernet auto-negotiates. A 1 GbE entry board and a 10 GbE full-tier board plug into the same switch and communicate at 1 Gbps without configuration, special cabling, or a bridge.

And the traffic between boards is small. In MoE mode, weights stream from storage local to each node; only activations cross the board boundary, roughly 12 KB per layer transition. A slow link between fast boards costs almost nothing.

That combination is what makes cross-generation fleets viable here in a way they would not be on a custom fabric. **The interconnect was chosen for correctness, and compatibility fell out of it.**

---

## 2. The four requirements

Everything above depends on four things being true of every board. These are the contract.

### 2.1 Ethernet is the only inter-board data interface

No proprietary bus, no custom connector carrying data, no vendor-specific fabric between boards.

**This is the one that will be tempting to break.** A parallel bus or a PCIe link between boards would be faster, and at some point a future tier will make that look worth doing. It is not. The moment inter-board data leaves Ethernet, every board built before that decision stops being able to join a fleet.

Stacking headers may carry power, control, reset and board identity. They must not carry the data plane.

### 2.2 Power is per board, never chained

Each board takes its own DC input. Power never passes through one board to reach another.

A 20 W entry board and a 400 W full-tier board coexist without the small one carrying the large one's current. It also means hot-swap, independent fault isolation, and per-board power measurement.

**Distribute at 12 V**, regulate on board. Higher voltage means lower current for the same power, which keeps connectors and conductors sane across a wide range of board draws.

### 2.3 Every node publishes a capability descriptor

At startup, each node reports over the control plane:

| Field | Purpose |
|---|---|
| `node_id` | stable unique identifier |
| `generation` | board revision and node class |
| `memory_bytes` | usable memory for layer residency |
| `storage_bandwidth_bps` | measured, not rated |
| `storage_bytes` | total local capacity |
| `compute_tops` | INT8 throughput |
| `link_bps` | negotiated Ethernet rate |
| `precisions` | supported formats, e.g. INT4, INT8, FP16 |

**Measured, not rated, for bandwidth.** Vendor figures vary and real throughput depends on the parts actually populated. A node should benchmark its own storage once at first boot and cache the result.

Ten lines of code today. Impossible to introduce into a deployed fleet later, because there is no way to ask a node that does not know how to answer.

### 2.4 No role is fixed in hardware

Any node can coordinate. There is no head node, no master board, no position in the stack that carries special meaning.

Roles are assigned by the runtime at startup based on capability descriptors. A fleet whose coordinator dies elects another and continues.

---

## 3. What the runtime must do

The contract above is necessary but not sufficient. A mixed fleet creates a scheduling problem that a runtime written for identical nodes will not handle.

### 3.1 Weighted sharding is mandatory

In MoE gang mode, all nodes work the same layer in lockstep, so **the slowest node gates every layer**. Eight entry boards plus one full-tier board runs at entry speed unless work is distributed proportionally to capability.

Assign experts in proportion to each node's `storage_bandwidth_bps`. A node with 30x the bandwidth takes roughly 30x the experts.

| Fleet | Naive equal sharding | Weighted sharding |
|---|---|---|
| 8 entry boards | baseline | baseline |
| 8 entry + 1 full | ~baseline | ~4x baseline |

This must be in the runtime from the beginning. Retrofitting proportional assignment into a scheduler built on the assumption of identical nodes is a rewrite, not a patch.

### 3.2 Tiering across generations

Mixed fleets suggest a natural division that pure fleets do not:

- **New boards** hold the hot expert cache and do the compute
- **Old boards** become the cold-expert store and the corpus tier

This is the same LRU logic as expert caching within a board (see [dials.md](dials.md) #14), applied across hardware generations rather than across storage tiers. Old hardware gets demoted rather than discarded, and its storage capacity remains fully useful.

### 3.3 Graceful membership changes

Nodes join and leave without stopping the fleet. A board that disappears mid-generation causes its experts to be re-fetched from a peer or from cold storage, and the fleet re-shards on the next layer boundary.

This is the same failover logic a homogeneous fleet needs for reliability. Cross-generation compatibility does not add a requirement here; it just makes it visible sooner, because people will actually swap boards.

---

## 4. The honest limit

**Compatibility cannot cross a memory-capacity boundary.**

A board can only participate in layer residency if its nodes can collectively hold a layer of the model being run. Entry-tier boards hold 18 GB, which covers every current frontier model. If a future model has a 40 GB layer, entry boards cannot hold it no matter how good the scheduler is.

They remain useful as a storage and bandwidth tier, feeding experts to boards that can hold the layer. That is a demotion, not equal participation, and the contract should not be read as promising otherwise.

**What is guaranteed:** an old board always connects, is always addressable, and always contributes its storage bandwidth and capacity.

**What is not guaranteed:** that an old board can run any future model's layer.

---

## 5. What this buys

**For one person.** Buy one board now. Add a second next year, of whatever generation is current. The first board does not become e-waste, and the fleet gets faster than either board alone.

**For a firm.** Budget accumulates instead of resetting. Twelve months of API spend buys hardware that is still running in year five, joined by four more purchases, none of which required replacing what came before.

**For repair.** A failed board is unplugged and replaced with any board in the family. No matched pairs, no firmware version lock, no need to source the same generation.

**For the project.** A stated compatibility contract is the difference between a platform and a one-off. It is also what makes an open-hardware design worth building on: anyone fabricating a board to this contract knows it will interoperate with every other board built to it, including ones designed after theirs.

---

## 6. Summary of the contract

Any board claiming membership in this family must:

1. Use **Ethernet only** for inter-board data
2. Take **its own DC power input**, never chained
3. Publish a **capability descriptor** at startup, with measured storage bandwidth
4. Assign **no role in hardware** and accept coordination from any peer

Any runtime claiming to serve this family must:

5. **Shard work proportionally** to node capability, not equally
6. Handle **nodes joining and leaving** without restarting the fleet

Six items. All of them cost nothing today. All of them are unrecoverable if omitted.

---

*Companion to [architecture.md](architecture.md), [dials.md](dials.md) and [ideal-node.md](ideal-node.md). This contract describes intended behaviour of a system that has not yet been built.*
