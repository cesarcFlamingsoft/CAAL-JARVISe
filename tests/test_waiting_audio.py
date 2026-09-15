"""Waiting cues use the same audio stream as the answer, never queued say()."""

import asyncio

import pytest
from test_runtime_imports import _load_voice_agent_module


@pytest.mark.asyncio
async def test_slow_answer_gets_one_delayed_cue(monkeypatch):
    voice = _load_voice_agent_module()
    from caal.llm import CAALLLM
    from caal.llm.providers import OllamaProvider

    spoken = []

    async def synth(agent, text, model_settings):
        words = "".join([part async for part in text])
        spoken.append(words)
        if words == "Answer.":
            await asyncio.sleep(0.06)
        yield words

    monkeypatch.setattr(voice.Agent.default, "tts_node", synth)
    agent = voice.VoiceAssistant(CAALLLM(provider=OllamaProvider()))
    agent._waiting_cues_enabled = True
    agent._waiting_cue_delay = 0.01

    async def source():
        yield " "
        await asyncio.sleep(0.02)
        yield "Answer."

    result = [frame async for frame in agent.tts_node(source(), None)]
    assert result == ["One moment.", "Answer."]
    assert spoken.count("One moment.") == 1


@pytest.mark.asyncio
async def test_answer_audio_cancels_cue_even_during_cue_synthesis():
    from caal.waiting_audio import with_waiting_cue

    cue_closed = asyncio.Event()

    async def answer():
        await asyncio.sleep(0.03)
        yield "answer"

    async def cue():
        try:
            await asyncio.sleep(0.08)
            yield "too late"
        finally:
            cue_closed.set()

    result = [x async for x in with_waiting_cue(answer(), cue, delay=0.005)]
    assert result == ["answer"]
    assert cue_closed.is_set()


@pytest.mark.asyncio
async def test_generated_reply_activates_waiting_before_model_text(monkeypatch):
    voice = _load_voice_agent_module()
    from livekit.agents.llm import ChatContext

    from caal.llm import CAALLLM
    from caal.llm.providers import OllamaProvider

    async def slow(*args, **kwargs):
        await asyncio.sleep(0.08)
        yield "Answer."

    monkeypatch.setattr(voice, "llm_node", slow)
    agent = voice.VoiceAssistant(CAALLLM(provider=OllamaProvider()))
    agent._waiting_cues_enabled = True
    stream = agent.llm_node(ChatContext(), [], None)
    try:
        assert await asyncio.wait_for(anext(stream), 0.02) == " "
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("delay,empty", [(0.1, False), (0.001, True)])
async def test_fast_or_empty_answer_never_starts_cue(delay, empty):
    from caal.waiting_audio import with_waiting_cue

    async def answer():
        if not empty:
            yield "answer"

    def cue():
        pytest.fail("cue must not start")

    assert [x async for x in with_waiting_cue(answer(), cue, delay=delay)] == (
        [] if empty else ["answer"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("delay", [0.001, 1])
async def test_interruption_or_session_cancellation_closes_pending_audio(delay):
    from caal.waiting_audio import with_waiting_cue

    closed = asyncio.Event()
    cue_closed = asyncio.Event()

    async def answer():
        try:
            await asyncio.sleep(10)
            yield "late answer"
        finally:
            closed.set()

    async def cue():
        try:
            await asyncio.sleep(10)
            yield "late cue"
        finally:
            cue_closed.set()

    stream = with_waiting_cue(answer(), cue, delay=delay)
    task = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    if delay < 0.02:
        assert cue_closed.is_set()


@pytest.mark.asyncio
async def test_fixed_speech_does_not_get_waiting_cue(monkeypatch):
    voice = _load_voice_agent_module()
    from caal.llm import CAALLLM
    from caal.llm.providers import OllamaProvider

    async def synth(agent, text, model_settings):
        words = "".join([part async for part in text])
        await asyncio.sleep(0.02)
        yield words

    monkeypatch.setattr(voice.Agent.default, "tts_node", synth)
    agent = voice.VoiceAssistant(CAALLLM(provider=OllamaProvider()))
    agent._waiting_cue_delay = 0.001

    async def source():
        yield "Please confirm."

    assert [x async for x in agent.tts_node(source(), None)] == ["Please confirm."]


@pytest.mark.asyncio
async def test_sdk_can_close_generation_from_another_task():
    voice = _load_voice_agent_module()
    from livekit.agents.llm import ChatContext

    from caal.llm import CAALLLM
    from caal.llm.providers import OllamaProvider

    agent = voice.VoiceAssistant(CAALLLM(provider=OllamaProvider()))
    agent._waiting_cues_enabled = True
    stream = agent.llm_node(ChatContext(), [], None)
    assert await asyncio.create_task(anext(stream)) == " "
    await stream.aclose()


@pytest.mark.asyncio
async def test_sdk_filters_preserve_early_marker_and_still_clean_answer():
    from caal.waiting_audio import waiting_text_transform

    async def source():
        yield " "
        await asyncio.sleep(0.08)
        yield "**Hello** 🌞."

    stream = waiting_text_transform(source())
    assert await asyncio.wait_for(anext(stream), 0.02) == " "
    rest = "".join([x async for x in stream])
    assert "Hello" in rest
    assert "**" not in rest
    assert "🌞" not in rest


@pytest.mark.asyncio
async def test_delayed_ordinary_reply_has_no_default_waiting_cue(monkeypatch):
    voice = _load_voice_agent_module()
    from livekit.agents.llm import ChatContext

    from caal.llm import CAALLLM
    from caal.llm.providers import OllamaProvider

    async def slow(*args, **kwargs):
        await asyncio.sleep(0.03)
        yield "The file is ready."

    async def synth(agent, text, model_settings):
        words = "".join([part async for part in text])
        await asyncio.sleep(0.03)
        yield words

    monkeypatch.setattr(voice, "llm_node", slow)
    monkeypatch.setattr(voice.Agent.default, "tts_node", synth)
    agent = voice.VoiceAssistant(CAALLLM(provider=OllamaProvider()))
    agent._waiting_cue_delay = 0.001
    for _ in range(2):
        text = agent.llm_node(ChatContext(), [], None)
        assert [x async for x in agent.tts_node(text, None)] == ["The file is ready."]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["model", "synthesis"])
async def test_default_cancelled_reply_never_speaks_cue(monkeypatch, stage):
    voice = _load_voice_agent_module()
    from livekit.agents.llm import ChatContext

    from caal.llm import CAALLLM
    from caal.llm.providers import OllamaProvider

    entered = asyncio.Event()
    closed = asyncio.Event()
    spoken = []

    async def model(*args, **kwargs):
        if stage == "model":
            entered.set()
            try:
                await asyncio.sleep(10)
            finally:
                closed.set()
        yield "Answer."

    async def synth(agent, text, model_settings):
        words = "".join([x async for x in text])
        if stage == "synthesis":
            entered.set()
            try:
                await asyncio.sleep(10)
            finally:
                closed.set()
        spoken.append(words)
        yield words

    monkeypatch.setattr(voice, "llm_node", model)
    monkeypatch.setattr(voice.Agent.default, "tts_node", synth)
    agent = voice.VoiceAssistant(CAALLLM(provider=OllamaProvider()))
    agent._waiting_cue_delay = 0.001
    text = agent.llm_node(ChatContext(), [], None)
    stream = agent.tts_node(text, None)
    task = asyncio.create_task(anext(stream))
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert spoken == []
    await stream.aclose()
    await text.aclose()
