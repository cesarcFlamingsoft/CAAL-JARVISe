"""Run via caal-agent's installed LiveKit SDK; no rooms, tools, LLMs or calls."""
import asyncio
import json
import os
import time
import wave
import sys

# Match voice_agent.py: /app/src precedes the installed package.
sys.path.insert(0, "/app/src")
from pathlib import Path

import httpx
from caal import settings
from caal.tts_selection import create_tts, trial_config

TEXT = 'Understood. I am ready to help.'


async def measure(obj, name, streaming):
    events = []
    pcm = []
    started = time.perf_counter()
    stream = obj.stream() if streaming else obj.synthesize(TEXT)
    async with stream:
        if streaming:
            stream.push_text(TEXT)
            stream.end_input()
        async for event in stream:
            events.append({'at_s': time.perf_counter() - started, 'samples': event.frame.samples_per_channel})
            pcm.append(bytes(event.frame.data))
    elapsed = time.perf_counter() - started
    data = b''.join(pcm)
    assert events and data and len(data) % 2 == 0
    with wave.open('/tmp/' + name + '.wav', 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(data)
    return {'name': name, 'first_sdk_frame_s': events[0]['at_s'], 'generation_s': elapsed,
            'audio_s': len(data) / 48000, 'events': events}


async def main():
    assert settings.SETTINGS_PATH == Path('/app/settings.json')
    selected = settings.load_settings()
    explicit = settings.load_user_settings()
    runtime = {
        'tts_provider': explicit.get('tts_provider') or os.getenv('TTS_PROVIDER', 'kokoro'),
        'tts_voice_kokoro': selected.get('tts_voice_kokoro') or os.getenv('TTS_VOICE', 'am_puck'),
        'tts_voice_piper': selected.get('tts_voice_piper') or 'speaches-ai/piper-en_US-ryan-high',
    }
    kwargs = dict(kokoro_url=os.environ['KOKORO_URL'], speaches_url=os.environ['SPEACHES_URL'],
                  kokoro_model=os.environ['TTS_MODEL'])
    config = trial_config()
    assert config is not None
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        url = config['endpoint'] + '/health'
        anon = await client.get(url)
        auth = await client.get(url, headers={'Authorization': 'Bearer ' + config['token']})
        assert anon.status_code == 401 and auth.status_code == 200
    result = {'selected_provider': runtime['tts_provider'], 'kokoro_voice': runtime['tts_voice_kokoro'],
              'docker_anonymous_qwen_health': anon.status_code, 'docker_authenticated_qwen_health': auth.status_code,
              'qwen_health': auth.json(), 'measurements': []}
    obj = create_tts(runtime, **kwargs)
    try:
        result['measurements'].append(await measure(obj, 'live-selected-provider', obj.capabilities.streaming))
    finally:
        await obj.aclose()
    # Explicit test-only candidate; this never writes settings or selects it for user sessions.
    candidate = create_tts({**runtime, 'tts_provider': 'qwen-trial'}, **kwargs)
    qwen = candidate._wrapped_tts
    result['qwen_model'] = qwen.model
    result['qwen_voice'] = 'jarvis-designed'
    result['fallback_model'] = qwen.fallback.model
    result['fallback_voice'] = runtime['tts_voice_kokoro']
    fallback_calls = []
    original = qwen.fallback.synthesize

    def observed_fallback(*args, **kw):
        fallback_calls.append(True)
        return original(*args, **kw)

    qwen.fallback.synthesize = observed_fallback
    try:
        for n in range(3):
            result['measurements'].append(await measure(candidate, f'live-approved-qwen-{n}', True))
    finally:
        await candidate.aclose()
    result['qwen_fallback_calls'] = len(fallback_calls)
    assert not fallback_calls, 'Qwen test fell back'
    print(json.dumps(result))

asyncio.run(main())
