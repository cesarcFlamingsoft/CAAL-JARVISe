import asyncio, time
from types import SimpleNamespace
import pytest
import run_service


@pytest.mark.asyncio
async def test_idle_service_exits_but_never_interrupts_active_generation():
    engine = SimpleNamespace(busy=True, last_used=time.monotonic() - 10)

    class Server:
        should_exit = False

        async def serve(self):
            while not self.should_exit:
                await asyncio.sleep(0.001)

    server = Server()
    task = asyncio.create_task(
        run_service.run_bounded(server, engine, idle_seconds=0.01, poll_interval=0.001)
    )
    await asyncio.sleep(0.02)
    assert not server.should_exit
    engine.busy = False
    await asyncio.wait_for(task, 0.1)
    assert server.should_exit


@pytest.mark.asyncio
async def test_engine_records_actual_generation_activity_for_idle_bound():
    from trial_stream import Engine

    engine = Engine(lambda text: iter([b"\x01\x00"]))
    try:
        before = engine.last_used
        await asyncio.sleep(0.002)
        assert [x async for x in engine.stream("Test.")] == [b"\x01\x00"]
        assert engine.last_used > before
    finally:
        await engine.close()
