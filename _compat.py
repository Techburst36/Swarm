"""
_compat.py — Python 3.10 compatibility shims for the Swarm project.

Provides ``asyncio_timeout`` that works on Python 3.10 (where
``asyncio.timeout`` was added in 3.11) and 3.11+.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

if sys.version_info >= (3, 11):
    asyncio_timeout = asyncio.timeout  # type: ignore[assignment]
else:
    @asynccontextmanager
    async def asyncio_timeout(seconds: float | None) -> AsyncIterator[None]:
        """Backport of asyncio.timeout for Python 3.10.

        Raises ``asyncio.TimeoutError`` if the block takes longer than
        *seconds*.  ``seconds=None`` means no timeout.
        """
        if seconds is None:
            yield
            return
        task = asyncio.current_task()
        if task is None:
            yield
            return
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        handle = loop.call_later(seconds, future.set_result, None)
        try:
            yield
        finally:
            handle.cancel()
        if future.done() and future.result() is None:
            raise asyncio.TimeoutError(
                f"timed out after {seconds} seconds"
            )
