import asyncio
import time
from storage_io import DirectFileExpertStore

async def main():
    # 16 layers x 16 experts x 4 MB = 1 GB file layout
    store = DirectFileExpertStore(
        path="/tmp/dummy_experts.bin",
        expert_size_bytes=4 * 1024 * 1024,
        num_layers=16,
        num_experts=16,
        queue_depth=4,
        use_odirect=True
    )
    
    t0 = time.monotonic()
    data = await store.read_expert(layer=0, expert=0)
    elapsed = time.monotonic() - t0
    
    mb = len(data) / (1024 * 1024)
    mbps = mb / elapsed if elapsed > 0 else 0
    
    print(f"Read {mb:.1f} MB in {elapsed*1000:.2f} ms ({mbps:.1f} MB/s) directly from disk.")
    await store.close()

if __name__ == "__main__":
    asyncio.run(main())
