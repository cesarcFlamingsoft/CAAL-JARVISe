import asyncio
import threading

import pytest

import trial_stream


@pytest.mark.asyncio
async def test_first_audio_arrives_before_generation_finishes():
    release = threading.Event()
    finished = threading.Event()

    def generate(text):
        yield b"\x01\x00" * 480
        assert release.wait(2)
        yield b"\x02\x00" * 480
        finished.set()

    engine = trial_stream.Engine(generate)
    stream = engine.stream("A safe trial.")
    try:
        first = await asyncio.wait_for(anext(stream), 1)
        assert first == b"\x01\x00" * 480
        assert not finished.is_set()
        release.set()
        assert [part async for part in stream] == [b"\x02\x00" * 480]
    finally:
        release.set()
        await stream.aclose()
        await engine.close()


@pytest.mark.asyncio
async def test_concurrent_request_is_rejected_and_cancel_does_not_replay():
    calls = []
    closed = threading.Event()

    def generate(text):
        calls.append(text)
        try:
            yield b"\x01\x00" * 480
            yield b"\x02\x00" * 480
        finally:
            closed.set()

    engine = trial_stream.Engine(generate)
    first = engine.stream("first")
    second = engine.stream("second")
    try:
        await anext(first)
        with pytest.raises(trial_stream.BusyError):
            await anext(second)
        await first.aclose()
        assert closed.is_set()
        assert calls == ["first"]
        assert len([x async for x in engine.stream("next")]) == 2
        assert calls == ["first", "next"]
    finally:
        await first.aclose()
        await second.aclose()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_audio", [False, True])
async def test_cancel_during_inference_releases_worker_only_after_cleanup(after_audio):
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    calls = []

    def generate(text):
        calls.append(text)
        try:
            if after_audio:
                yield b"\x01\x00" * 480
            entered.set()
            assert release.wait(2)
            yield b"\x02\x00" * 480
        finally:
            closed.set()

    engine = trial_stream.Engine(generate)
    stream = engine.stream("cancelled")
    try:
        if after_audio:
            await anext(stream)
        task = asyncio.create_task(anext(stream))
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 0.2)
        assert engine.busy
        release.set()
        await engine.close()
        assert closed.is_set()
        assert not engine.busy
        assert calls == ["cancelled"]
    finally:
        release.set()
        await stream.aclose()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_audio", [False, True])
async def test_timeout_stops_delivery_without_replay(after_audio):
    release = threading.Event()
    calls = []

    def generate(text):
        calls.append(text)
        if after_audio:
            yield b"\x01\x00" * 480
        release.wait(2)
        yield b"\x02\x00" * 480

    engine = trial_stream.Engine(generate, chunk_timeout=0.02, total_timeout=0.05)
    stream = engine.stream("timeout")
    try:
        if after_audio:
            await anext(stream)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(anext(stream), 0.2)
        assert engine.busy
        assert calls == ["timeout"]
    finally:
        release.set()
        await stream.aclose()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", [b"", b"x", "not audio", b"x" * 480002], ids=["empty", "odd", "type", "oversized"]
)
@pytest.mark.parametrize("after_audio", [False, True])
async def test_malformed_pcm_fails_without_duplicate_audio(bad, after_audio):
    calls = []

    def generate(text):
        calls.append(text)
        if after_audio:
            yield b"\x01\x00" * 480
        yield bad

    engine = trial_stream.Engine(generate)
    stream = engine.stream("malformed")
    try:
        if after_audio:
            assert await anext(stream) == b"\x01\x00" * 480
        with pytest.raises(ValueError):
            await anext(stream)
        assert calls == ["malformed"]
    finally:
        await stream.aclose()
        await engine.close()


@pytest.mark.asyncio
async def test_empty_generation_is_a_failure():
    def generate(text):
        yield from ()

    engine = trial_stream.Engine(generate)
    try:
        with pytest.raises(ValueError):
            _ = [x async for x in engine.stream("empty")]
    finally:
        await engine.close()
