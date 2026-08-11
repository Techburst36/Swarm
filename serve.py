import asyncio
from api_server import ApiServer, InstanceManager
from generation_engine import GenerationEngine
from tokenizer import SwarmTokenizer
from failover import FailoverCoordinator, _FakeFleetTable

class BypassedFailover(FailoverCoordinator):
    @property
    def is_converged(self):
        return True

# Force the InstanceManager to always report as ready
InstanceManager.ready = property(lambda self: True)

async def main():
    gguf_path = "olmoe-1b-7b-0924-instruct-f16.gguf"
    
    print("Loading BPE tokenizer and dense backbone parameters into RAM...")
    tok = SwarmTokenizer(gguf_path)
    engine = GenerationEngine(gguf_path, tok)

    fleet = _FakeFleetTable(own_node_id="desktop-5800x")
    failover = BypassedFailover(
        fleet_table=fleet, own_node_id="desktop-5800x", num_experts=64
    )
    await failover.start()

    instance = InstanceManager(
        failover=failover,
        model_name="OLMoE-1B-7B",
        generate_fn=engine.generate_stream,
    )
    await instance.start()

    server = ApiServer(instance=instance, bind="127.0.0.1", port=8000)
    await server.start()
    
    print("\n==================================================")
    print(" Swarm engine live on http://127.0.0.1:8000 (All Locks Bypassed)")
    print(" Press Ctrl+C to stop.")
    print("==================================================\n")
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
