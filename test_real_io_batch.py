import asyncio
import time
from storage_io import DirectFileExpertStore

async def main():
    store = DirectFileExpertStore(
        path="/tmp/dummy_experts.bin",
        expert_size_bytes=4 * 1024 * 1024,
        num_layers=16,
        num_experts=16,
        queue_depth=4,
        use_odirect=True
    )
    
    t0 = time.monotonic()
    # Read 16 experts concurrently across 4 layers
    tasks = [store.read_expert(layer=l, expert=0) for l in range(16)]
    results = await asyncio.gather(*tasks)
    elapsed = time.monotonic() - t0
    
    total_mb = (len(results) * 4)
    mbps = total_mb / elapsed
    
    print(f"Read {total_mb} MB (16 experts) in {elapsed*1000:.2f} ms ({mbps:.1f} MB/s) with O_DIRECT.")
    await store.close()

if __name__ == "__main__":
    asyncio.run(main())
