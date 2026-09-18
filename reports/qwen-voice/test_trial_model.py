from types import SimpleNamespace

import numpy as np
import pytest
import trial_model


def test_english_uses_the_fixed_friday_anchor_clone_and_resets_decoder():
    loaded = []
    resets = []
    options = []

    class Fake:
        speech_tokenizer = SimpleNamespace(
            decoder=SimpleNamespace(reset_streaming_state=lambda: resets.append(1))
        )

        def generate(self, text, **kwargs):
            options.append(kwargs)
            yield SimpleNamespace(audio=np.array([0.5, -0.5], dtype=np.float32), sample_rate=24000)
            yield SimpleNamespace(audio=np.array([0.1], dtype=np.float32), sample_rate=24000)

    def load(path):
        loaded.append(path)
        return Fake()

    model = trial_model.QwenModel(load=load, seed=lambda value: None)
    for _ in range(2):
        gen = model.generate("Understood.")
        assert next(gen) == np.array([16383, -16383], dtype="<i2").tobytes()
        gen.close()
    assert loaded == [trial_model.SPANISH_MODEL_PATH]
    assert len(resets) == 2
    assert options[0]["stream"] is True
    assert options[0]["streaming_interval"] == 0.24
    assert options[0]["lang_code"] == "English"
    assert options[0]["ref_audio"] == str(trial_model.SPANISH_REFERENCE_AUDIO)
    assert options[0]["ref_text"] == trial_model.SPANISH_REFERENCE_TEXT
    assert "instruct" not in options[0]


def test_spanish_is_routed_to_the_fixed_female_friday_clone_prompt():
    loaded = []
    options = []

    class Fake:
        speech_tokenizer = SimpleNamespace(
            decoder=SimpleNamespace(reset_streaming_state=lambda: None)
        )

        def generate(self, text, **kwargs):
            options.append(kwargs)
            yield SimpleNamespace(audio=np.array([0.5], dtype=np.float32), sample_rate=24000)

    model = trial_model.QwenModel(
        load=lambda path: loaded.append(path) or Fake(), seed=lambda value: None
    )
    gen = model.generate("Buenas noches.", language="es")
    assert next(gen) == np.array([16383], dtype="<i2").tobytes()
    gen.close()

    assert loaded == [trial_model.SPANISH_MODEL_PATH]
    assert options[0]["lang_code"] == "Spanish"
    assert options[0]["ref_audio"] == str(trial_model.SPANISH_REFERENCE_AUDIO)
    assert options[0]["ref_text"] == trial_model.SPANISH_REFERENCE_TEXT
    assert "instruct" not in options[0]


@pytest.mark.parametrize(
    "audio,rate", [(np.array([np.nan]), 24000), (np.zeros((2, 2)), 24000), (np.zeros(4), 22050)]
)
def test_invalid_model_audio_is_rejected(audio, rate):
    fake = SimpleNamespace(
        generate=lambda *a, **k: iter([SimpleNamespace(audio=audio, sample_rate=rate)]),
        speech_tokenizer=SimpleNamespace(
            decoder=SimpleNamespace(reset_streaming_state=lambda: None)
        ),
    )
    model = trial_model.QwenModel(load=lambda path: fake, seed=lambda value: None)
    with pytest.raises(ValueError):
        list(model.generate("Test."))
