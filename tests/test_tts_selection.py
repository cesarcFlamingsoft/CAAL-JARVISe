import pytest

from caal import tts_selection


@pytest.mark.asyncio
async def test_default_kokoro_remains_available_but_explicit_qwen_never_changes_voice(monkeypatch):
    monkeypatch.delenv("CAAL_QWEN_TRIAL_TOKEN", raising=False)
    provider = tts_selection.create_tts(
        {"tts_provider": "kokoro", "tts_voice_kokoro": "am_adam"},
        kokoro_url="http://localhost:8001",
        speaches_url="http://localhost:8001",
        kokoro_model="prince-canuma/Kokoro-82M",
    )
    try:
        assert provider.model == "prince-canuma/Kokoro-82M"
        assert provider._opts.voice == "am_adam"
    finally:
        await provider.aclose()
    with pytest.raises(RuntimeError, match="qwen_trial_unavailable"):
        tts_selection.create_tts(
            {"tts_provider": "qwen-trial", "tts_voice_kokoro": "am_adam"},
            kokoro_url="http://localhost:8001",
            speaches_url="http://localhost:8001",
            kokoro_model="prince-canuma/Kokoro-82M",
        )


@pytest.mark.asyncio
async def test_piper_selection_is_preserved():
    provider = tts_selection.create_tts(
        {"tts_provider": "piper", "tts_voice_piper": "speaches-ai/piper-en_US-ryan-high"},
        kokoro_url="http://kokoro:8880",
        speaches_url="http://localhost:8001",
        kokoro_model="kokoro",
    )
    try:
        assert provider.model == "speaches-ai/piper-en_US-ryan-high"
        assert provider._opts.voice == "default"
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_explicit_qwen_uses_private_service_without_a_second_voice_fallback(monkeypatch):
    from caal.qwen_tts import QwenTTS

    monkeypatch.setenv("CAAL_QWEN_TRIAL_TOKEN", "a" * 32)
    provider = tts_selection.create_tts(
        {"tts_provider": "qwen-trial", "tts_voice_kokoro": "am_adam"},
        kokoro_url="http://localhost:8001",
        speaches_url="http://localhost:8001",
        kokoro_model="prince-canuma/Kokoro-82M",
    )
    try:
        assert provider.capabilities.streaming
        assert isinstance(provider._wrapped_tts, QwenTTS)
        assert provider._wrapped_tts.endpoint == "http://host.docker.internal:18003"
        assert provider._wrapped_tts.fallback is None
    finally:
        await provider.aclose()


def test_voice_entrypoint_builds_tts_through_trial_aware_factory():
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).parents[1] / "voice_agent.py").read_text())
    entry = next(
        n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "entrypoint"
    )
    calls = [
        n
        for n in ast.walk(entry)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "create_tts"
    ]
    assert len(calls) == 1
