with open("api_server.py", "r") as f:
    code = f.read()

polyfill = """import sys
import asyncio
import contextlib

# Polyfill for Python 3.10 compatibility
if not hasattr(asyncio, "timeout"):
    @contextlib.asynccontextmanager
    async def _dummy_timeout(delay):
        yield
    asyncio.timeout = _dummy_timeout

"""

if "_dummy_timeout" not in code:
    code = polyfill + code
    with open("api_server.py", "w") as f:
        f.write(code)
    print("Polyfill injected! asyncio.timeout will no longer crash.")
else:
    print("Already patched.")
