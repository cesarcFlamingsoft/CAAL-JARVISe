"""Single-worker bridge from synchronous synthesis to backpressured PCM delivery."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

_DONE = object()


class BusyError(Exception):
    pass


def _next(iterator):
    return next(iterator, _DONE)


class Engine:
    def __init__(self, generate, chunk_timeout=15.0, total_timeout=60.0):
        self.generate = generate
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-trial")
        self.busy = False
        self.last_used = time.monotonic()
        self.cleanups = set()
        self.chunk_timeout = chunk_timeout
        self.total_timeout = total_timeout

    async def stream(self, text, language=None):
        """Synthesize one request. The language travels with the call: the engine
        holds none of its own, so a Spanish turn cannot leak into the next
        session's English one, and an older caller that passes nothing keeps
        exactly the English behaviour it has today."""
        if self.busy:
            raise BusyError("Synthesis worker is busy")
        self.busy = True
        self.last_used = time.monotonic()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.total_timeout
        iterator = self.generate(text) if language is None else self.generate(text, language)
        pending = None
        emitted = False
        try:
            while True:
                pending = loop.run_in_executor(self.executor, _next, iterator)
                value = await asyncio.wait_for(
                    asyncio.shield(pending),
                    timeout=min(self.chunk_timeout, max(0, deadline - loop.time())),
                )
                if value is _DONE:
                    if not emitted:
                        raise ValueError("No audio generated")
                    return
                if (
                    not isinstance(value, bytes)
                    or not value
                    or len(value) % 2
                    or len(value) > 480000
                ):
                    raise ValueError("Invalid PCM audio")
                emitted = True
                yield value
        finally:

            async def cleanup():
                try:
                    if pending is not None:
                        try:
                            await pending
                        except Exception:
                            pass
                    close = getattr(iterator, "close", None)
                    if close:
                        await loop.run_in_executor(self.executor, close)
                finally:
                    self.busy = False
                    self.last_used = time.monotonic()

            task = asyncio.create_task(cleanup())
            self.cleanups.add(task)
            task.add_done_callback(self.cleanups.discard)
            if pending is None or pending.done():
                await asyncio.shield(task)

    async def close(self):
        if self.cleanups:
            await asyncio.gather(*self.cleanups)
        await asyncio.to_thread(self.executor.shutdown, True)
