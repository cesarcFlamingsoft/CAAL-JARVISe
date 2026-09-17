import hashlib

import httpx
import pytest
import trial_service
from trial_stream import Engine


@pytest.mark.asyncio
async def test_api_requires_auth_and_streams_only_validated_requests():
    calls = []

    def generate(text, language=None):
        calls.append((text, language))
        yield b"\x01\x00" * 480

    engine = Engine(generate)
    app = trial_service.create_app(engine, "a" * 32)
    payload = {
        "input": "Understood.",
        "model": "qwen-trial",
        "voice": "jarvis-designed",
        "response_format": "pcm",
    }
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            assert (await c.post("/v1/audio/speech", json=payload)).status_code == 401
            auth = {"Authorization": "Bearer " + "a" * 32}
            for bad in ["", "x" * 601]:
                assert (
                    await c.post("/v1/audio/speech", json={**payload, "input": bad}, headers=auth)
                ).status_code == 422
            assert calls == []
            r = await c.post("/v1/audio/speech", json=payload, headers=auth)
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("audio/pcm")
            assert r.content == b"\x01\x00" * 480
            assert calls == [("Understood.", "en")]
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,status", [(ValueError("bad"), 502), (TimeoutError(), 504)])
async def test_failure_before_audio_returns_failure_status(failure, status):
    def generate(text, language=None):
        raise failure
        yield

    engine = Engine(generate)
    app = trial_service.create_app(engine, "a" * 32)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            r = await c.post(
                "/v1/audio/speech",
                json={"input": "Test."},
                headers={"Authorization": "Bearer " + "a" * 32},
            )
            assert r.status_code == status
            assert "bad" not in r.text
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_health_requires_auth_and_reports_worker_state():
    engine = Engine(lambda text: iter(()))
    try:
        app = trial_service.create_app(engine, "a" * 32)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            assert (await c.get("/health")).status_code == 401
            r = await c.get("/health", headers={"Authorization": "Bearer " + "a" * 32})
            assert r.status_code == 200
            assert r.json()["busy"] is False
    finally:
        await engine.close()


def test_service_refuses_weak_or_missing_token():
    with pytest.raises(ValueError):
        trial_service.create_app(None, "")


@pytest.mark.asyncio
async def test_disconnect_before_first_audio_cancels_generation():
    import asyncio

    closed = asyncio.Event()

    class Request:
        async def receive(self):
            await asyncio.sleep(0.01)
            return {"type": "http.disconnect"}

    async def source():
        try:
            await asyncio.sleep(10)
            yield b"\x01\x00"
        finally:
            closed.set()

    stream = source()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(trial_service.prefetch(stream, Request()), 0.2)
    assert closed.is_set()


@pytest.mark.asyncio
async def test_voice_readback_is_authenticated_and_matches_synthesis_design():
    import trial_model

    engine = Engine(lambda text: iter(()))
    try:
        app = trial_service.create_app(engine, "a" * 32)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            assert (await c.get("/voice")).status_code == 401
            r = await c.get("/voice", headers={"Authorization": "Bearer " + "a" * 32})
            assert r.status_code == 200
            body = r.json()
            assert body["model"] == trial_model.MODEL
            assert body["revision"] == trial_model.REVISION
            assert body["voice"] == "jarvis-designed"
            assert body["seed"] == 42
            assert body["style"] == trial_model.STYLE
            assert body["style_sha256"] == hashlib.sha256(trial_model.STYLE.encode()).hexdigest()
            assert body["default_language"] == "en"
            assert body["designs"] == {
                code: {
                    "lang_code": trial_model.LANG_CODES[code],
                    "style": trial_model.STYLES[code],
                    "style_sha256": hashlib.sha256(trial_model.STYLES[code].encode()).hexdigest(),
                }
                for code in trial_model.SUPPORTED
            }
    finally:
        await engine.close()
