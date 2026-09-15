"""TTS-only provider selection; trial infrastructure stays outside user settings."""

import os

from livekit.plugins import openai

from .qwen_tts import QwenTTS, sentence_adapter


def trial_config():
    token = os.environ.get("CAAL_QWEN_TRIAL_TOKEN", "")
    if len(token) < 32 or not token.isascii():
        return None
    return {"endpoint": "http://host.docker.internal:18003", "token": token}


def create_tts(runtime, *, kokoro_url, speaches_url, kokoro_model):
    if runtime["tts_provider"] == "piper":
        return openai.TTS(
            base_url=f"{speaches_url}/v1",
            api_key="not-needed",
            model=runtime["tts_voice_piper"],
            voice="default",
        )
    kokoro = openai.TTS(
        base_url=f"{kokoro_url}/v1",
        api_key="not-needed",
        model=kokoro_model,
        voice=runtime["tts_voice_kokoro"],
    )
    config = trial_config()
    if runtime["tts_provider"] == "qwen-trial" and config:
        return sentence_adapter(QwenTTS(**config, fallback=kokoro))
    return kokoro


async def select_user_tts(existing, runtime, *, user_id, identity, **kwargs):
    """Called only after signed room/SIP identity resolution; retain the gate voice."""
    if not user_id or identity is None:
        return existing
    from .tts_store import TTSStore
    from .voicebox import VoiceboxTTS

    store = TTSStore(identity)
    preference = store.preference(user_id)
    if not preference:
        return existing
    selected = {**runtime, "tts_provider": preference["provider"]}
    if preference["provider"] == "voicebox":
        config = store.config()
        fallback = create_tts({**runtime, "tts_provider": "kokoro"}, **kwargs)
        if config and preference.get("config_revision") == config["revision"]:
            try:
                provider = sentence_adapter(
                    VoiceboxTTS(
                        endpoint=config["endpoint"],
                        credential=config["credential"],
                        profile_id=preference["profile_id"],
                        engine=preference["engine"],
                        model_size=preference["model_size"],
                        fallback=fallback,
                    )
                )
            except ValueError:
                provider = fallback
        else:
            provider = fallback
    else:
        provider = create_tts(selected, **kwargs)
    await existing.aclose()
    return provider
