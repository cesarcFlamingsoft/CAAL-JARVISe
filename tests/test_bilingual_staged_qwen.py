"""Stage 2: the staged Qwen service speaks the language of the request.

Nothing here touches the running :18003 service. These tests import the staged
modules in ``reports/bilingual/staged-qwen`` and drive them with a fake backend,
then drive the **real** ``caal.qwen_tts`` client against the **real** staged
FastAPI app over ASGI, so the client/server protocol is exercised end to end
rather than asserted about.

The audio is synthetic: a deterministic PCM tone per language, so "the Spanish
request produced Spanish-path audio" is a checkable fact without loading a 1.7B
model or sending anything anywhere.
"""

import asyncio
import json
import math
import struct
import sys
from pathlib import Path

import httpx
import pytest

STAGED = Path(__file__).resolve().parents[1] / "reports" / "bilingual" / "staged-qwen"
LIVE = Path(__file__).resolve().parents[1] / "reports" / "qwen-voice"
for path in (str(STAGED), str(LIVE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import staged_model  # noqa: E402
import staged_service  # noqa: E402
import staged_stream  # noqa: E402

TOKEN = "x" * 40


def tone(language, frames=480):
    """Deterministic PCM whose waveform identifies the language path taken."""
    hz = {"en": 220, "es": 330}[language]
    return struct.pack(
        f"<{frames}h",
        *(int(12000 * math.sin(2 * math.pi * hz * n / 24000)) for n in range(frames)),
    )


class FakeBackend:
    """Stands in for QwenModel. Records the language of every call."""

    def __init__(self, chunks=2, fail=None):
        self.calls = []
        self.chunks = chunks
        self.fail = fail

    def generate(self, text, language=staged_model.DEFAULT_LANGUAGE):
        # Same contract as the staged model: resolve (and reject) before work.
        lang_code, style = staged_model.design_for(language)
        self.calls.append(
            {"text": text, "language": language, "lang_code": lang_code, "style": style}
        )
        if self.fail:
            raise self.fail
        for _ in range(self.chunks):
            yield tone(language)


def client_for(engine):
    app = staged_service.create_app(engine, TOKEN)
    return app


async def speak(app, body, headers=None):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://staged") as client:
        async with client.stream(
            "POST",
            "/v1/audio/speech",
            json=body,
            headers=headers if headers is not None else {"Authorization": "Bearer " + TOKEN},
        ) as response:
            audio = b"".join([chunk async for chunk in response.aiter_bytes()])
            return response.status_code, dict(response.headers), audio


# --- the approved designs, exactly ------------------------------------------


def test_the_english_design_is_the_live_one_character_for_character():
    import trial_model as live

    assert staged_model.STYLES["en"] == live.STYLE
    assert staged_model.LANG_CODES["en"] == "English"
    assert staged_model.MODEL == live.MODEL
    assert staged_model.REVISION == live.REVISION
    assert staged_model.SEED == 42
    assert staged_model.MAX_TOKENS == 700


def test_the_retired_male_spanish_manifest_is_historical_not_live():
    """Keep the prior audition as evidence, but never route live FRIDAY through it."""
    import trial_model as live

    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "reports/bilingual/audition/neutral-spanish-v2/audition-manifest.json"
        ).read_text()
    )
    approved = manifest["items"][0]
    assert approved["file"] == "es-candidate-longer.wav"
    assert approved["lang_code"] == "Spanish"
    assert "Un hombre adulto" in approved["style"]
    assert manifest["style_en_live_unchanged"] != live.STYLE
    assert live.STYLES["es"] == "Approved female FRIDAY synthetic-reference clone."
    assert live.LANG_CODES["es"] == "Spanish"


def test_an_unsupported_language_raises_instead_of_falling_back_to_english():
    for bad in ("fr", "pt", "EN", "", "english"):
        with pytest.raises(staged_model.UnsupportedLanguage):
            staged_model.design_for(bad)
    # `None` means "unspecified", which is the English default, not an error.
    assert staged_model.design_for(None) == staged_model.design_for("en")


def test_the_live_service_matches_the_approved_bilingual_contract():
    """The activated service must retain both approved per-request designs.

    This replaces the old Stage-1-only source snapshot assertion.  The live
    service intentionally became bilingual during the approved activation; a
    guard which bans its ``language`` field would turn a security regression
    (Spanish silently spoken in English) into the passing state.
    """
    import trial_model as live
    import trial_service as live_service

    assert set(live_service.Speech.model_fields["language"].annotation.__args__) == {"en", "es"}
    assert live.SUPPORTED == ("en", "es")
    assert live.DEFAULT_LANGUAGE == "en"
    approval = json.loads(
        (LIVE / "friday-female-audition" / "approved-live.json").read_text()
    )
    assert live.STYLE == approval["english"]["style"]
    assert live.MODEL == approval["english"]["model"]
    assert live.REVISION == approval["english"]["revision"]
    assert live.SPANISH_MODEL == approval["spanish"]["model"]
    assert live.SPANISH_REVISION == approval["spanish"]["revision"]
    assert live.SPANISH_REFERENCE_TEXT == approval["spanish"]["reference_text"]
    assert live.SPANISH_REFERENCE_AUDIO.name == approval["english"]["reference_audio"]
    assert live.STYLES["es"] == "Approved female FRIDAY synthetic-reference clone."
    assert live.LANG_CODES == staged_model.LANG_CODES
    for bad in ("fr", "pt", "EN", "", "english"):
        with pytest.raises(live.UnsupportedLanguage):
            live.design_for(bad)


# --- per-request language, no shared state ----------------------------------


def test_each_request_carries_its_own_language():
    backend = FakeBackend()
    engine = staged_stream.Engine(backend.generate)
    app = client_for(engine)

    async def run():
        for language in ("en", "es", "en", "es"):
            status, headers, audio = await speak(
                app, {"input": "Hola.", "language": language}
            )
            assert status == 200, (language, audio[:200])
            assert headers["x-audio-language"] == language
            assert audio == tone(language) * 2
        await engine.close()

    asyncio.run(run())
    assert [call["language"] for call in backend.calls] == ["en", "es", "en", "es"]
    assert [call["style"] for call in backend.calls] == [
        staged_model.STYLES[code] for code in ("en", "es", "en", "es")
    ]


def test_a_body_without_a_language_is_the_unchanged_english_request():
    backend = FakeBackend()
    engine = staged_stream.Engine(backend.generate)
    app = client_for(engine)

    async def run():
        # Exactly the body the live service is sent today.
        status, headers, audio = await speak(
            app,
            {
                "input": "Understood.",
                "model": "qwen-trial",
                "voice": "jarvis-designed",
                "response_format": "pcm",
            },
        )
        assert status == 200
        assert audio == tone("en") * 2
        await engine.close()

    asyncio.run(run())
    assert backend.calls[0]["language"] == "en"
    assert backend.calls[0]["lang_code"] == "English"


def test_a_spanish_request_leaves_no_language_behind_on_the_engine():
    backend = FakeBackend()
    engine = staged_stream.Engine(backend.generate)
    app = client_for(engine)

    async def run():
        await speak(app, {"input": "Hola.", "language": "es"})
        assert not any("language" in name for name in vars(engine))
        # A fresh session with no language field must be English again.
        _, headers, audio = await speak(app, {"input": "Hello."})
        assert headers["x-audio-language"] == "en"
        assert audio == tone("en") * 2
        await engine.close()

    asyncio.run(run())


def test_the_service_rejects_a_language_it_has_no_approved_design_for():
    backend = FakeBackend()
    engine = staged_stream.Engine(backend.generate)
    app = client_for(engine)

    async def run():
        for bad in ("fr", "pt-BR", "EN", ""):
            status, _, _ = await speak(app, {"input": "Bonjour.", "language": bad})
            assert status == 422, bad
        await engine.close()

    asyncio.run(run())
    # Nothing was synthesized, in any language.
    assert backend.calls == []


def test_an_unsupported_language_reaching_the_backend_is_a_400_not_english_audio():
    # The pydantic literal stops "fr" at the door; this covers the layer below it,
    # so a future widening of the schema still cannot produce a silent English reply.
    backend = FakeBackend()
    engine = staged_stream.Engine(backend.generate)
    app = staged_service.create_app(engine, TOKEN)

    async def run():
        original = engine.stream

        def stream(text, language=None):
            return original(text, "fr")

        engine.stream = stream
        status, _, audio = await speak(app, {"input": "Hola.", "language": "es"})
        assert status == 400
        assert tone("en") not in audio
        engine.stream = original
        await engine.close()

    asyncio.run(run())


# --- lock, cancellation and concurrency -------------------------------------


def test_the_single_worker_lock_still_rejects_a_concurrent_request():
    backend = FakeBackend(chunks=3)
    engine = staged_stream.Engine(backend.generate)
    app = staged_service.create_app(engine, TOKEN)

    async def run():
        stream = engine.stream("first", "es")
        assert await anext(stream) == tone("es")
        assert engine.busy is True
        status, _, _ = await speak(app, {"input": "second", "language": "en"})
        assert status == 409
        await stream.aclose()
        await asyncio.sleep(0)
        await engine.close()
        assert engine.busy is False

    asyncio.run(run())
    assert [call["language"] for call in backend.calls] == ["es"]


def test_cancelling_a_spanish_stream_releases_the_worker_for_english():
    backend = FakeBackend(chunks=5)
    engine = staged_stream.Engine(backend.generate)

    async def run():
        stream = engine.stream("larga", "es")
        assert await anext(stream) == tone("es")
        await stream.aclose()
        for _ in range(20):
            if not engine.busy:
                break
            await asyncio.sleep(0.01)
        assert engine.busy is False
        second = engine.stream("short", "en")
        assert await anext(second) == tone("en")
        await second.aclose()
        await engine.close()

    asyncio.run(run())
    assert [call["language"] for call in backend.calls] == ["es", "en"]


def test_authentication_is_unchanged():
    backend = FakeBackend()
    engine = staged_stream.Engine(backend.generate)
    app = staged_service.create_app(engine, TOKEN)

    async def run():
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}):
            status, _, _ = await speak(app, {"input": "Hola.", "language": "es"}, headers)
            assert status == 401
        await engine.close()

    asyncio.run(run())
    assert backend.calls == []


def test_the_voice_endpoint_publishes_both_approved_designs():
    backend = FakeBackend()
    engine = staged_stream.Engine(backend.generate)
    app = staged_service.create_app(engine, TOKEN)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://staged") as client:
            response = await client.get("/voice", headers={"Authorization": "Bearer " + TOKEN})
        await engine.close()
        return response

    response = asyncio.run(run())
    assert response.status_code == 200
    body = response.json()
    assert body["default_language"] == "en"
    assert set(body["designs"]) == {"en", "es"}
    assert body["designs"]["en"]["style"] == staged_model.STYLES["en"]
    assert body["designs"]["es"]["lang_code"] == "Spanish"
    import hashlib

    assert (
        body["designs"]["es"]["style_sha256"]
        == hashlib.sha256(staged_model.SPANISH_STYLE.encode()).hexdigest()
    )


# --- the real client against the real staged server -------------------------


def _emitted(app, language):
    """Run the real QwenTTS client against the staged app and return its PCM."""
    from caal.qwen_tts import QwenTTS

    async def run():
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18003")
        provider = QwenTTS(endpoint="http://127.0.0.1:18003", token=TOKEN, client=client)
        provider.language = language
        chunks = []
        stream = provider.synthesize("Hola, ¿cómo estás?")
        try:
            async for event in stream:
                chunks.append(bytes(event.frame.data))
        finally:
            await stream.aclose()
            await client.aclose()
        return b"".join(chunks)

    return asyncio.run(run())


@pytest.mark.parametrize("language", ["en", "es"])
def test_the_real_client_and_the_staged_server_agree_on_the_language(language):
    backend = FakeBackend()
    engine = staged_stream.Engine(backend.generate)
    app = staged_service.create_app(engine, TOKEN)
    audio = _emitted(app, language)
    assert backend.calls[0]["language"] == language
    assert backend.calls[0]["style"] == staged_model.STYLES[language]
    # The audio that came back is the one the language path generated.
    assert tone(language) in audio
    if language == "es":
        assert tone("en") not in audio
    asyncio.run(engine.close())


def test_a_client_with_no_language_set_gets_the_english_design():
    backend = FakeBackend()
    engine = staged_stream.Engine(backend.generate)
    app = staged_service.create_app(engine, TOKEN)
    audio = _emitted(app, None)
    assert backend.calls[0]["language"] == "en"
    assert tone("en") in audio
    asyncio.run(engine.close())
