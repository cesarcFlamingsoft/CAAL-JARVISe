"""Bounded, synthetic speech regressions; never contact a speech service."""

import asyncio
import json

import httpx
import pytest

from caal.qwen_tts import QwenTTS, sentence_adapter
from caal.tts_selection import set_language

HEADERS = {"content-type": "audio/pcm", "x-audio-sample-rate": "24000", "x-audio-channels": "1"}


async def drain(stream):
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_created_request_keeps_language_after_switch():
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=b"\x00\x01" * 480, headers=HEADERS)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = QwenTTS(endpoint="http://127.0.0.1:18003", token="t" * 40, client=client)
        set_language(provider, "es")
        stream = provider.synthesize("Todo está listo.")
        set_language(provider, "en")
        await drain(stream)
    assert seen[0].get("language") == "es"


@pytest.mark.asyncio
async def test_concurrent_sentence_streams_keep_language_across_cancellation():
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=b"\x00\x01" * 480, headers=HEADERS)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = QwenTTS(endpoint="http://127.0.0.1:18003", token="t" * 40, client=client)
        adapted = sentence_adapter(provider)
        ready = asyncio.Event()
        release = asyncio.Event()

        async def spanish():
            set_language(adapted, "es")
            stream = adapted.stream()
            ready.set()
            await release.wait()
            stream.push_text("Todo está listo. Hasta luego.")
            stream.end_input()
            await drain(stream)

        task = asyncio.create_task(spanish())
        await ready.wait()
        set_language(adapted, "en")
        cancelled = adapted.synthesize("Cancelled canned sentence.")
        await cancelled.aclose()
        release.set()
        await task
        await drain(adapted.synthesize("Everything is ready."))
        await adapted.aclose()
    spanish_requests = [b for b in seen if b["input"] != "Everything is ready."]
    assert spanish_requests
    assert all(b.get("language") == "es" for b in spanish_requests)
    assert "language" not in seen[-1]


@pytest.mark.asyncio
async def test_failed_spanish_request_cannot_flip_to_english_fallback():
    from livekit.agents import APIConnectionError
    from test_bilingual_fallback import RecordingFallback

    fallback = RecordingFallback()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503))
    ) as client:
        provider = QwenTTS(
            endpoint="http://127.0.0.1:18003", token="t" * 40, client=client, fallback=fallback
        )
        provider.language = "es"
        stream = provider.synthesize("Hasta luego.")
        provider.language = "en"
        with pytest.raises(APIConnectionError):
            await drain(stream)
    assert not fallback.used


@pytest.mark.asyncio
async def test_session_tts_reads_producer_profile_and_resets_after_cancel(monkeypatch):
    from types import SimpleNamespace

    from livekit.agents import Agent

    from caal.language_policy import LanguageSession
    from caal.speech_request import SpeechText, speech_profile
    from voice_agent import VoiceAssistant

    seen = []

    closed = []

    async def default(agent, text, settings):
        try:
            async for part in text:
                seen.append((str(part), speech_profile.get()))
                yield object()
        finally:
            closed.append(True)

    monkeypatch.setattr(Agent.default, "tts_node", default)
    agent = SimpleNamespace(_language_session=LanguageSession("en"), _waiting_cues_enabled=False)
    agent._speak_answer = lambda *args: VoiceAssistant._speak_answer(agent, *args)

    async def source():
        yield SpeechText("Hasta luego.", "es")
        agent._language_session.current = "en"
        yield SpeechText("Todo está listo.", "es")

    stream = VoiceAssistant.tts_node(agent, source(), {})
    await anext(stream)
    await stream.aclose()
    assert seen[0][1].language == "es"
    assert speech_profile.get() is None
    assert closed == [True]


@pytest.mark.asyncio
async def test_end_call_reply_carries_language_after_session_switch():
    from types import SimpleNamespace

    from caal import reply_localization
    from caal.call_termination import acknowledge_and_end_call
    from caal.speech_request import SpeechText

    said = []

    async def say(text):
        said.append(text)

    async def delete_room(request):
        pass

    token = reply_localization.reply_language.set("es")
    try:
        await acknowledge_and_end_call(
            SimpleNamespace(say=say), SimpleNamespace(delete_room=delete_room), "synthetic-room"
        )
        reply_localization.reply_language.set("en")
        assert isinstance(said[0], SpeechText)
        assert said[0].profile.language == "es"
        assert "Finalizando" in said[0]
    finally:
        reply_localization.reply_language.reset(token)


@pytest.mark.asyncio
async def test_queued_llm_turn_uses_message_language_not_newer_session(monkeypatch):
    from types import SimpleNamespace

    from livekit.agents.llm import ChatContext

    from caal.language_policy import LanguageSession
    from caal.llm.providers.base import LLMResponse
    from caal.llm.providers.ollama_provider import OllamaProvider
    from voice_agent import VoiceAssistant

    seen = []

    class Provider(OllamaProvider):
        async def chat(self, messages, **kwargs):
            seen.extend(messages)
            return LLMResponse("Todo está listo.", [])

    ctx = ChatContext()
    ctx.add_message(role="user", content="Dime si está listo.", extra={"caal_reply_language": "es"})
    agent = SimpleNamespace(
        _language_session=LanguageSession("en"),
        _waiting_cues_enabled=False,
        _provider=Provider(),
        _tool_data_cache=None,
        _max_turns=20,
    )
    chunks = [part async for part in VoiceAssistant.llm_node(agent, ctx, [], {})]
    assert chunks[0].profile.language == "es"
    assert "Responde en español" in seen[0]["content"]


@pytest.mark.asyncio
async def test_cancelling_inflight_spanish_does_not_fallback_or_poison_next_request():
    from test_bilingual_fallback import RecordingFallback

    entered = asyncio.Event()
    released = asyncio.Event()
    seen = []

    async def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        if body.get("language") == "es":
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()
        return httpx.Response(200, content=b"\x00\x01" * 480, headers=HEADERS)

    fallback = RecordingFallback()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = QwenTTS(
            endpoint="http://127.0.0.1:18003", token="t" * 40, client=client, fallback=fallback
        )
        provider.language = "es"
        stream = provider.synthesize("Una frase de prueba.")
        task = asyncio.create_task(drain(stream))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(released.wait(), 2)
        provider.language = "en"
        await drain(provider.synthesize("A canned test sentence."))
    assert not fallback.used
    assert seen[0]["language"] == "es"
    assert "language" not in seen[1]


@pytest.mark.asyncio
async def test_production_text_filter_preserves_language_metadata():
    from caal.speech_request import SpeechText
    from caal.waiting_audio import waiting_text_transform

    async def source():
        yield SpeechText("**Todo está listo.**", "es")

    chunks = [part async for part in waiting_text_transform(source())]
    assert chunks
    assert all(part.profile.language == "es" for part in chunks)
    assert "**" not in "".join(chunks)
