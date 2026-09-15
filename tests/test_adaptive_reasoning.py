"""Request-level reasoning contracts against the actual provider wrapper."""

import asyncio
from types import SimpleNamespace

import pytest

from caal.llm.providers import OllamaProvider, RoutedProvider


class Client:
    def __init__(self, capabilities=()):
        self.capabilities = capabilities
        self.calls = []

    def show(self, model):
        return SimpleNamespace(capabilities=self.capabilities)

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        reply = SimpleNamespace(
            message=SimpleNamespace(content="A fox found a light.", tool_calls=[])
        )
        return iter([reply]) if kwargs["stream"] else reply


def local(monkeypatch, capabilities=(), **kwargs):
    client = Client(capabilities)
    monkeypatch.setattr("caal.llm.providers.ollama_provider.ollama.Client", lambda **kw: client)
    return OllamaProvider(model="test-model", **kwargs), client


@pytest.mark.asyncio
async def test_unsupported_model_omits_think_even_with_override(monkeypatch):
    provider, client = local(monkeypatch)
    await provider.chat([{"role": "user", "content": "Hello"}], think=True)
    assert "think" not in client.calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_routed_effort_follows_existing_route_without_changing_defaults(monkeypatch, stream):
    primary, client = local(monkeypatch, ["thinking"], think=True)
    provider = RoutedProvider(primary=primary)
    story = [
        {"role": "system", "content": "Custom owner prompt stays intact."},
        {"role": "user", "content": "Tell me a gentle little tale about a lantern."},
    ]
    hard = [{"role": "user", "content": "Research the options and build a detailed plan."}]

    async def request(messages):
        from caal.work_router import request_reasoning

        request_reasoning.set(messages is hard)
        if stream:
            return "".join([x async for x in provider.chat_stream(messages)])
        return (await provider.chat(messages)).content

    await asyncio.gather(request(story), request(hard))
    by_text = {call["messages"][-1]["content"]: call for call in client.calls}
    assert by_text[story[-1]["content"]]["think"] is False
    assert by_text[hard[-1]["content"]]["think"] is True
    assert by_text[story[-1]["content"]]["messages"] == story
    assert primary.think is True


@pytest.mark.asyncio
async def test_stream_read_does_not_block_waiting_cue_or_interruption(monkeypatch):
    import time

    provider, client = local(monkeypatch, ["thinking"])
    progressed = asyncio.Event()

    def blocking_chat(**kwargs):
        def chunks():
            time.sleep(0.08)
            yield SimpleNamespace(message=SimpleNamespace(content="Done"))

        return chunks()

    client.chat = blocking_chat

    async def heartbeat():
        await asyncio.sleep(0.02)
        progressed.set()

    task = asyncio.create_task(heartbeat())
    try:
        async for _ in provider.chat_stream([{"role": "user", "content": "Hi"}]):
            assert progressed.is_set(), "network iteration blocked the voice event loop"
    finally:
        await task


@pytest.mark.asyncio
async def test_existing_semantic_read_selects_effort_without_an_extra_call():
    from caal.work_router import SemanticWorkRouter, request_reasoning

    calls = []

    async def classify(messages):
        calls.append(messages)
        return '{"route":"conversation","reasoning":false}'

    token = request_reasoning.set(None)
    try:
        await SemanticWorkRouter(classify=classify).route("Tell me a little tale about a lantern.")
        assert request_reasoning.get() is False
        assert len(calls) == 1
    finally:
        request_reasoning.reset(token)


@pytest.mark.asyncio
async def test_unrecognized_logic_preserves_reasoning(monkeypatch):
    primary, client = local(monkeypatch, ["thinking"], think=False)
    await RoutedProvider(primary=primary).chat(
        [{"role": "user", "content": "Every dax is a mip; no mip is a zub. Can a dax be a zub?"}]
    )
    assert client.calls[0]["think"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply,effort",
    [
        ('{"route":"conversation","reasoning":true}', True),
        ('{"route":"conversation"}', None),
        ('{"route":"conversation","reasoning":"false"}', None),
        ("unusable", None),
    ],
)
async def test_semantic_effort_is_strict_and_never_inherits_previous_turn(reply, effort):
    from caal.work_router import SemanticWorkRouter, request_reasoning

    async def classify(messages):
        return reply

    token = request_reasoning.set(False)
    try:
        await SemanticWorkRouter(classify=classify).route("Could a dax be a zub?")
        assert request_reasoning.get() is effort
    finally:
        request_reasoning.reset(token)


@pytest.mark.asyncio
async def test_voice_hook_carries_effort_across_tasks_without_provider_mutation(monkeypatch):
    from livekit.agents.llm import ChatContext, ChatMessage
    from test_runtime_imports import _load_voice_agent_module

    from caal.llm import CAALLLM
    from caal.work_router import request_reasoning

    voice = _load_voice_agent_module()
    primary, client = local(monkeypatch, ["thinking"], think=True)

    async def consume(text):
        request_reasoning.set(False)
        return False

    agent = voice.VoiceAssistant(
        CAALLLM(provider=RoutedProvider(primary=primary)), turn_consumed=consume
    )
    message = ChatMessage(role="user", content=["A little story, please."])
    ctx = ChatContext(items=[message])
    await asyncio.create_task(agent.on_user_turn_completed(ctx, message))

    async def node(*args, **kwargs):
        yield (
            await kwargs["provider"].chat(
                [{"role": "user", "content": message.text_content}], think=kwargs["reasoning"]
            )
        ).content

    monkeypatch.setattr(voice, "llm_node", node)

    async def run():
        return [x async for x in agent.llm_node(ctx.copy(), [], None)]

    await asyncio.create_task(run())
    assert client.calls[0]["think"] is False
    assert primary.think is True
    assert request_reasoning.get() is None


@pytest.mark.asyncio
async def test_capability_discovery_is_reused_and_does_not_add_per_turn_latency(monkeypatch):
    primary, client = local(monkeypatch, ["thinking"])
    calls = []

    def show(model):
        calls.append(model)
        return SimpleNamespace(capabilities=["thinking"])

    client.show = show
    await primary.chat([{"role": "user", "content": "Hello"}], think=False)
    await primary.chat([{"role": "user", "content": "A puzzle"}], think=True)
    assert len(calls) == 1
    assert [call["think"] for call in client.calls] == [False, True]


@pytest.mark.asyncio
async def test_real_stt_task_returns_effort_to_completed_voice_turn(monkeypatch):
    from livekit.agents.llm import ChatContext, ChatMessage
    from test_runtime_imports import _load_voice_agent_module

    from caal.llm import CAALLLM
    from caal.work_router import request_reasoning

    voice = _load_voice_agent_module()

    async def end_call():
        pytest.fail("must not end a session")

    handler = voice.LocalTurnHandler(phone_handoff=None, session=object(), end_call=end_call)

    async def local_commands(text):
        request_reasoning.set(False)
        return False

    monkeypatch.setattr(handler, "_handle_local_commands", local_commands)
    primary, _ = local(monkeypatch, ["thinking"])
    agent = voice.VoiceAssistant(CAALLLM(provider=primary), turn_consumed=handler.turn_consumed)
    message = ChatMessage(role="user", content=["Tell me a little story."])
    handler.on_final_transcript(message.text_content)
    await agent.on_user_turn_completed(ChatContext(), message)
    assert message.extra["caal_reasoning"] is False
    assert request_reasoning.get() is None
