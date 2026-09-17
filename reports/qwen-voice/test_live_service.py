import json
import plistlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import trial_model
import trial_service

P = Path(__file__).parent


def test_live_voice_exactly_matches_professional_direction():
    approved = json.loads((P / 'balanced-male-audition/results.json').read_text())
    assert trial_model.STYLE == (P / 'professional-tuning/style.txt').read_text().strip()
    assert trial_model.MODEL == approved['model']
    assert trial_model.REVISION == '5c390979e4b93af5f2932f90742ca99c7dd04687'


def test_live_model_sets_bounded_mlx_memory(monkeypatch):
    import sys
    calls = []
    mx = SimpleNamespace(
        random=SimpleNamespace(seed=lambda _: None),
        set_memory_limit=lambda n: calls.append(('memory', n)),
        set_cache_limit=lambda n: calls.append(('cache', n)),
    )
    monkeypatch.setitem(sys.modules, 'mlx.core', mx)
    monkeypatch.setitem(sys.modules, 'mlx', SimpleNamespace(core=mx))
    trial_model.QwenModel(load=lambda _: None)
    assert calls == [('memory', 4 * 1024**3), ('cache', 256 * 1024**2)]


@pytest.mark.asyncio
async def test_service_prewarms_before_accepting_requests_and_closes():
    events = []

    class Engine:
        async def stream(self, text):
            events.append('warm')
            yield b'\x01\x00'
            events.append('ready')

        async def close(self):
            events.append('close')

    app = trial_service.create_app(Engine(), 'a' * 32, prewarm=True)
    async with app.router.lifespan_context(app):
        assert events == ['warm', 'ready']
    assert events == ['warm', 'ready', 'close']


@pytest.mark.asyncio
async def test_failed_prewarm_fails_startup_and_closes():
    closed = []

    class Engine:
        async def stream(self, text):
            raise RuntimeError('warm failed')
            yield

        async def close(self):
            closed.append(True)

    app = trial_service.create_app(Engine(), 'a' * 32, prewarm=True)
    with pytest.raises(RuntimeError, match='warm failed'):
        async with app.router.lifespan_context(app):
            pytest.fail('Must not serve a cold or failed model')
    assert closed == [True]


def test_supervision_is_durable_private_and_restart_throttled():
    config = plistlib.loads((P / 'com.caal.qwen-trial.plist').read_bytes())
    assert config['KeepAlive'] is True
    assert config['RunAtLoad'] is True
    assert config['ThrottleInterval'] >= 15
    source = (P / 'run_service.py').read_text()
    assert 'prewarm=True' in source
    assert 'asyncio.run(server.serve())' in source
    assert 'host="127.0.0.1"' in source
