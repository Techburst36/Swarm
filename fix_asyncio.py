import re

with open("api_server.py", "r") as f:
    code = f.read()

# Replace 3.11 context manager with 3.10 wait_for
code = re.sub(
    r"async with asyncio\.timeout\((.*?)\):\s*await self\._concurrency_sem\.acquire\(\)",
    r"await asyncio.wait_for(self._concurrency_sem.acquire(), timeout=\1)",
    code
)

# Python 3.10 throws asyncio.TimeoutError instead of the built-in TimeoutError
code = code.replace("except TimeoutError:", "except (TimeoutError, asyncio.TimeoutError):")

with open("api_server.py", "w") as f:
    f.write(code)

print("Patched api_server.py for Python 3.10 compatibility!")
