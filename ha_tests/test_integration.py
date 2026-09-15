"""Run with HA 2026.9.1 in the isolated HA test environment."""

import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import web
from homeassistant.components.conversation import ConversationInput
from homeassistant.components.tts import TTSAudioRequest
from homeassistant.core import Context

from custom_components.jarvis_satellite.client import BridgeClient, validate_endpoint
from custom_components.jarvis_satellite.conversation import JarvisConversation
from custom_components.jarvis_satellite.tts import JarvisTTS

PILOT = "assist_satellite.home_assistant_voice_0a3d6b_assist_satellite"
DEVICE = "0bf018dfe200b28d8f7cff95e8d2aa75"


def register_satellite(hass, satellite_id=PILOT, device_id=DEVICE):
    from homeassistant.helpers import device_registry as dr, entity_registry as er
    from pytest_homeassistant_custom_component.common import MockConfigEntry, mock_device_registry

    source = MockConfigEntry(domain="esphome")
    source.add_to_hass(hass)
    mock_device_registry(
        hass,
        {
            device_id: dr.DeviceEntry(
                id=device_id,
                config_entry_id=source.entry_id,
                identifiers={("esphome", satellite_id)},
            )
        },
    )
    return er.async_get(hass).async_get_or_create(
        "assist_satellite",
        "esphome",
        satellite_id,
        config_entry=source,
        device_id=device_id,
        suggested_object_id=satellite_id.split(".")[1],
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://8.8.8.8:8889",
        "http://169.254.169.254:80",
        "http://example.com:8889",
        "http://user:pass@10.0.0.1:8889",
        "http://10.0.0.1:8889/path",
        "http://10.0.0.1:8889?x",
        "https://10.0.0.1:8889",
        "http://10.0.0.1",
    ],
)
def test_private_target_rejects_unsafe(endpoint):
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


@pytest.mark.asyncio
async def test_authenticated_client_stream_and_redirect_rejection(socket_enabled):
    seen = []

    async def identity(req):
        seen.append(req.headers.get("Authorization"))
        return web.json_response(
            {
                "id": "sat_test",
                "satellite_id": PILOT,
                "device_id": DEVICE,
                "protocol": 1,
                "personal_data": False,
                "device_actions": False,
            }
        )

    async def turn(req):
        seen.append(await req.json())
        return web.Response(text='{"text":"Hello"}\n{"done":true}\n')

    app = web.Application()
    app.router.add_get("/satellite/v1/identity", identity)
    app.router.add_post("/satellite/v1/turn", turn)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = BridgeClient(f"http://127.0.0.1:{port}", "s" * 43)
    try:
        assert (await client.identity())["device_id"] == DEVICE
        assert "".join([s async for s in client.turn({"text": "test"})]) == "Hello"
        assert seen[0] == "Bearer " + "s" * 43
        assert seen[1] == {"text": "test"}
    finally:
        await client.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_tts_native_stream_emits_audio_before_input_end_and_bounds():
    completed = False

    async def messages():
        yield "First sentence. "
        assert completed
        yield "Second sentence."

    class Client:
        async def audio(self, text):
            nonlocal completed
            assert len(text) < 300
            yield b"\x01\x02" * 50
            completed = True

    entry = SimpleNamespace(entry_id="entry", runtime_data=Client())
    entity = JarvisTTS(entry)
    assert entity.async_supports_streaming_input()
    response = await entity.async_stream_tts_audio(TTSAudioRequest("en", {}, messages()))
    assert response.extension == "wav"
    chunks = [x async for x in response.data_gen]
    assert chunks[0].startswith(b"RIFF") and sum(map(len, chunks)) == 244

    async def huge():
        yield "x" * 4097

    response = await entity.async_stream_tts_audio(TTSAudioRequest("en", {}, huge()))
    with pytest.raises(Exception):
        _ = [x async for x in response.data_gen]


@pytest.mark.asyncio
async def test_conversation_stream_binding_and_no_person_context():
    requests = []

    class Client:
        identity_data = {"id": "sat_test", "satellite_id": PILOT, "device_id": DEVICE}

        async def turn(self, body):
            requests.append(body)
            yield "Hello."

        async def cancel(self, rid):
            pass

    class Log:
        conversation_id = "ha-chat-id"
        content = []

        async def async_add_delta_content_stream(self, agent_id, stream):
            async for delta in stream:
                self.content.append(delta)
            yield SimpleNamespace(content="Hello.")

    entity = JarvisConversation(SimpleNamespace(entry_id="entry", runtime_data=Client()))
    entity.entity_id = "conversation.jarvis_satellite"
    user = ConversationInput(
        text="Hello",
        context=Context(),
        conversation_id="ha-chat-id",
        device_id=DEVICE,
        satellite_id=PILOT,
        language="en",
        agent_id=entity.entity_id,
    )
    log = Log()
    result = await entity._async_handle_message(user, log)
    assert result.response.speech["plain"]["speech"] == "Hello."
    assert requests[0]["device_id"] == DEVICE
    assert set(requests[0]) == {
        "text",
        "satellite_id",
        "device_id",
        "conversation_id",
        "request_id",
    }
    user.device_id = "living-room"
    with pytest.raises(Exception):
        await entity._async_handle_message(user, log)
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_config_flow_validates_backend_and_creates_entry(
    hass, enable_custom_integrations, monkeypatch
):
    from unittest.mock import AsyncMock

    from homeassistant.config_entries import SOURCE_USER

    from custom_components.jarvis_satellite import config_flow

    monkeypatch.setattr(
        config_flow.BridgeClient,
        "identity",
        AsyncMock(return_value={"id": "sat_test", "satellite_id": PILOT, "device_id": DEVICE}),
    )
    from homeassistant.setup import async_setup_component

    assert await async_setup_component(hass, "homeassistant", {})
    register_satellite(hass)
    flow = await hass.config_entries.flow.async_init(
        "jarvis_satellite", context={"source": SOURCE_USER}
    )
    assert flow["type"] == "form"
    bad = await hass.config_entries.flow.async_configure(
        flow["flow_id"], {"backend": "http://8.8.8.8:8889", "credential": "s" * 43}
    )
    assert bad["errors"]["base"] == "cannot_connect"
    good = await hass.config_entries.flow.async_configure(
        flow["flow_id"], {"backend": "http://10.0.0.10:8889", "credential": "s" * 43}
    )
    assert good["type"] == "create_entry"
    assert good["result"].unique_id == PILOT
    await hass.async_block_till_done()


@pytest.mark.asyncio
async def test_tts_cancellation_closes_audio_and_input():
    from contextlib import aclosing

    closed = []

    class Client:
        async def audio(self, text):
            try:
                yield b"\1\2"
                await asyncio.Event().wait()
            finally:
                closed.append("audio")

    async def text():
        try:
            yield "Sentence."
            await asyncio.Event().wait()
        finally:
            closed.append("text")

    entity = JarvisTTS(SimpleNamespace(entry_id="test", runtime_data=Client()))
    response = await entity.async_stream_tts_audio(TTSAudioRequest("en", {}, text()))
    async with aclosing(response.data_gen) as stream:
        assert (await anext(stream)).startswith(b"RIFF")
        assert await anext(stream) == b"\1\2"
    assert sorted(closed) == ["audio", "text"]
    assert entity._busy is False


@pytest.mark.asyncio
async def test_config_flow_transport_error_is_safe(hass, monkeypatch):
    from unittest.mock import AsyncMock

    import aiohttp

    from custom_components.jarvis_satellite.client import BridgeClient
    from custom_components.jarvis_satellite.config_flow import JarvisSatelliteConfigFlow

    monkeypatch.setattr(
        BridgeClient,
        "identity",
        AsyncMock(side_effect=aiohttp.ClientConnectionError("private detail")),
    )
    flow = JarvisSatelliteConfigFlow()
    flow.hass = hass
    result = await flow.async_step_user(
        {"backend": "http://10.0.0.10:8889", "credential": "s" * 43}
    )
    assert result["errors"] == {"base": "cannot_connect"}


@pytest.mark.asyncio
async def test_tts_long_sentence_is_split_before_bounded_synthesis():
    texts = []

    class Client:
        async def audio(self, text):
            texts.append(text)
            yield b"\1\2"

    async def text():
        yield "a" * 700 + "."

    entity = JarvisTTS(SimpleNamespace(entry_id="test", runtime_data=Client()))
    response = await entity.async_stream_tts_audio(TTSAudioRequest("en", {}, text()))
    _ = [x async for x in response.data_gen]
    assert max(map(len, texts)) <= 240
    assert "".join(texts) == "a" * 700 + "."


@pytest.mark.asyncio
async def test_native_setup_chatlog_and_core_tts_conversion(
    hass, enable_custom_integrations, monkeypatch, hass_client_no_auth
):
    from homeassistant.components import conversation
    from homeassistant.components.tts.const import DATA_TTS_MANAGER
    from homeassistant.helpers import device_registry as dr
    from homeassistant.helpers import entity_registry as er
    from homeassistant.setup import async_setup_component
    from pytest_homeassistant_custom_component.common import MockConfigEntry, mock_device_registry

    from custom_components.jarvis_satellite.client import BridgeClient

    assert await async_setup_component(hass, "homeassistant", {})
    source = MockConfigEntry(domain="esphome")
    source.add_to_hass(hass)
    mock_device_registry(
        hass,
        {
            DEVICE: dr.DeviceEntry(
                id=DEVICE, config_entry_id=source.entry_id, identifiers={("esphome", "pilot")}
            )
        },
    )
    registry = er.async_get(hass)
    registered = registry.async_get_or_create(
        "assist_satellite",
        "esphome",
        "pilot",
        config_entry=source,
        device_id=DEVICE,
        suggested_object_id=PILOT.split(".")[1],
    )
    assert registered.entity_id == PILOT

    async def identity(self):
        self.identity_data = {"id": "sat_test", "satellite_id": PILOT, "device_id": DEVICE}
        return self.identity_data

    async def turn(self, body):
        yield "A synthetic answer."

    async def audio(self, text):
        yield b"\x10\x20" * 4800

    monkeypatch.setattr(BridgeClient, "identity", identity)
    monkeypatch.setattr(BridgeClient, "turn", turn)
    monkeypatch.setattr(BridgeClient, "audio", audio)
    entry = MockConfigEntry(
        domain="jarvis_satellite",
        title="Jarvis Bedroom Pilot",
        unique_id=PILOT,
        data={"backend": "http://10.0.0.10:8889", "credential": "s" * 43},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    states = hass.states.async_all()
    conv = next(
        x.entity_id for x in states if x.entity_id.startswith("conversation.jarvis_bedroom")
    )
    voice = next(x.entity_id for x in states if x.entity_id.startswith("tts.jarvis_bedroom"))
    result = await conversation.async_converse(
        hass,
        "Hello",
        None,
        Context(),
        language="en",
        agent_id=conv,
        device_id=DEVICE,
        satellite_id=PILOT,
    )
    assert result.response.speech["plain"]["speech"] == "A synthetic answer."
    manager = hass.data[DATA_TTS_MANAGER]
    global_cache = manager.use_file_cache

    async def messages():
        yield "Synthetic sentence."

    # Use the native manager and its actual ffmpeg converter, consuming only memory.
    cache = manager.async_cache_message_stream_in_memory(
        voice, messages(), "en", {"preferred_format": "wav", "preferred_sample_rate": 16000}
    )
    await hass.async_block_till_done(wait_background_tasks=True)
    audio_data = b"".join([chunk async for chunk in cache.async_stream_data()])
    assert audio_data.startswith(b"RIFF") and len(audio_data) > 100
    assert manager.use_file_cache is global_cache and manager.file_cache == {}
    result_stream = manager.async_create_result_stream(
        voice, use_file_cache=False, language="en", options={"preferred_format": "wav"}
    )
    result_stream.async_set_message_stream(messages())
    http_client = await hass_client_no_auth()
    response = await http_client.get(result_stream.url)
    assert response.status == 200
    assert (await response.read()).startswith(b"RIFF")
    unknown = await http_client.get("/api/tts_proxy/unknown.wav")
    assert unknown.status == 404
    assert manager.use_file_cache is global_cache and manager.file_cache == {}
    client = entry.runtime_data
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert client.session.closed


@pytest.mark.asyncio
async def test_reconfigure_accepts_rotated_credential(
    hass, enable_custom_integrations, monkeypatch
):
    from unittest.mock import AsyncMock

    from homeassistant.config_entries import SOURCE_RECONFIGURE
    from homeassistant.setup import async_setup_component
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    from custom_components.jarvis_satellite.client import BridgeClient

    assert await async_setup_component(hass, "homeassistant", {})
    monkeypatch.setattr(
        BridgeClient,
        "identity",
        AsyncMock(return_value={"id": "sat_rotated", "satellite_id": PILOT, "device_id": DEVICE}),
    )
    register_satellite(hass)
    entry = MockConfigEntry(
        domain="jarvis_satellite",
        unique_id=PILOT,
        data={"backend": "http://10.0.0.2:8889", "credential": "s" * 43},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(
        "jarvis_satellite", context={"source": SOURCE_RECONFIGURE, "entry_id": entry.entry_id}
    )
    assert result["type"] == "form"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"backend": "http://10.0.0.2:8889", "credential": "r" * 43}
    )
    assert result["type"] == "abort"
    assert result["reason"] == "reconfigure_successful"
    assert entry.data["credential"] == "r" * 43
    await hass.async_block_till_done()


@pytest.mark.asyncio
async def test_multidevice_conversation_uses_verified_identity_and_cannot_cross():
    living = "assist_satellite.living"
    device = "b" * 32
    seen = []

    class Client:
        identity_data = {"id": "sat_living", "satellite_id": living, "device_id": device}

        async def turn(self, body):
            seen.append(body)
            yield "Living answer."

        async def cancel(self, rid):
            pass

    class Log:
        conversation_id = "same-ha-conversation"

        async def async_add_delta_content_stream(self, agent_id, stream):
            async for _ in stream:
                pass
            yield SimpleNamespace(content="Living answer.")

    entity = JarvisConversation(
        SimpleNamespace(entry_id="living", title="Jarvis Living", runtime_data=Client())
    )
    entity.entity_id = "conversation.jarvis_living"
    user = ConversationInput(
        text="Which lights are on?",
        context=Context(),
        conversation_id="same-ha-conversation",
        device_id=device,
        satellite_id=living,
        language="en",
        agent_id=entity.entity_id,
    )
    result = await entity._async_handle_message(user, Log())
    assert result.response.speech["plain"]["speech"] == "Living answer."
    assert seen[0]["satellite_id"] == living and seen[0]["device_id"] == device
    user.satellite_id = PILOT
    user.device_id = DEVICE
    with pytest.raises(Exception):
        await entity._async_handle_message(user, Log())
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_client_accepts_server_bound_multi_device_contract(socket_enabled):
    async def identity(req):
        return web.json_response(
            {
                "id": "sat_living",
                "satellite_id": "assist_satellite.living",
                "device_id": "b" * 32,
                "protocol": 1,
                "personal_data": False,
                "device_actions": True,
                "scope": "states_and_lights",
            }
        )

    app = web.Application()
    app.router.add_get("/satellite/v1/identity", identity)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    client = BridgeClient(f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}", "s" * 43)
    try:
        assert (await client.identity())["satellite_id"] == "assist_satellite.living"
    finally:
        await client.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_config_entry_migration_preserves_backend_credential(hass):
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    from custom_components.jarvis_satellite import async_migrate_entry

    entry = MockConfigEntry(
        domain="jarvis_satellite",
        version=1,
        unique_id=PILOT,
        title="Jarvis Bedroom Pilot",
        data={"backend": "http://10.0.0.2:8889", "credential": "s" * 43},
    )
    entry.add_to_hass(hass)
    before = dict(entry.data)
    assert await async_migrate_entry(hass, entry)
    assert entry.version == 2 and entry.data == before and entry.unique_id == PILOT
