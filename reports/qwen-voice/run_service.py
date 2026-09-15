import asyncio
import os
import time
from pathlib import Path

import uvicorn
from trial_model import QwenModel
from trial_service import create_app
from trial_stream import Engine


async def run_bounded(server, engine, idle_seconds=300, poll_interval=1):
    async def idle_monitor():
        while True:
            await asyncio.sleep(poll_interval)
            if not engine.busy and time.monotonic() - engine.last_used >= idle_seconds:
                server.should_exit = True
                return

    monitor = asyncio.create_task(idle_monitor())
    try:
        await server.serve()
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)


if __name__ == "__main__":
    token = Path(__file__).with_name(".token").read_text().strip()
    backend = QwenModel(interval=float(os.environ.get("QWEN_TRIAL_INTERVAL", ".24")))
    engine = Engine(backend.generate)
    app = create_app(engine, token, prewarm=True)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=18003,
            workers=1,
            access_log=False,
            timeout_keep_alive=120,
            timeout_graceful_shutdown=20,
        )
    )
    asyncio.run(server.serve())
