import numpy as np
import pytest
from types import SimpleNamespace
import trial_model


def test_warm_model_is_reused_and_decoder_reset_on_close():
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
    assert len(loaded) == 1
    assert len(resets) == 2
    assert options[0]["stream"] is True
    assert options[0]["streaming_interval"] == 0.24
    assert options[0]["instruct"] == trial_model.STYLE
    assert "ref_audio" not in options[0]


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
