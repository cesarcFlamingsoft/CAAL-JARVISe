from pathlib import Path
import yaml


def test_overlay_only_adds_trial_credential_to_agent():
    path=Path(__file__).with_name('docker-compose.qwen-trial.yaml')
    config=yaml.safe_load(path.read_text())
    assert set(config)=={'services'}
    assert set(config['services'])=={'agent'}
    agent=config['services']['agent']
    assert set(agent)=={'environment'}
    assert set(agent['environment'])=={'CAAL_QWEN_TRIAL_TOKEN'}
    assert ':?' in agent['environment']['CAAL_QWEN_TRIAL_TOKEN']
