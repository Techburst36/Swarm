import asyncio
from api_server import ApiServer, InstanceManager
from generation_engine import GenerationEngine
from tokenizer import SwarmTokenizer
from failover import FailoverCoordinator, _FakeFleetTable
from sharding import NodeCapability, compute_assignment

async def main():
    gguf_path = "olmoe-1b-7b-0924-instruct-f16.gguf"
    
    print("Loading BPE tokenizer and dense backbone parameters into RAM...")
    tok = SwarmTokenizer(gguf_path)
    engine = GenerationEngine(gguf_path, tok)

    fleet = _FakeFleetTable(own_node_id="desktop-5800x")
    failover = FailoverCoordinator(
        fleet_table=fleet, own_node_id="desktop-5800x", num_experts=64
    )
    await failover.start()

    # Seed a single-node ShardAssignment so InstanceManager finds the
    # fleet ready without waiting for a reshard that will never come
    # (FakeFleetTable has no peers, so no join/leave callbacks fire).
    failover._current = compute_assignment(
        [NodeCapability(node_id="desktop-5800x", storage_bandwidth_mbps=4000)],
        64,
    )

    instance = InstanceManager(
        failover=failover,
        model_name="OLMoE-1B-7B",
        generate_fn=engine.generate_stream,
    )
    await instance.start()

    server = ApiServer(instance=instance, bind="127.0.0.1", port=8000)
    await server.start()
    
    print("\n==================================================")
    print(" Swarm engine live on http://127.0.0.1:8000")
    print(" Press Ctrl+C to stop.")
    print("==================================================\n")
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
