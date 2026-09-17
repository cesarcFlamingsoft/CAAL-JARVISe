"""Targeted rollout, preserving the running environment; never prints credentials."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'reports/qwen-voice'
os.chdir(ROOT)


def run(args, **kw):
    result = subprocess.run(args, capture_output=True, text=True, **kw)
    if result.returncode:
        raise RuntimeError('Command failed; output suppressed: ' + args[0])
    return result.stdout


def inspect(name):
    return json.loads(run(['docker', 'inspect', 'caal-' + name]))[0]


def environment(container):
    return dict(value.split('=', 1) for value in container['Config']['Env'])


def gate():
    with urllib.request.urlopen('http://127.0.0.1:8889/health', timeout=5) as r:
        health = json.load(r)
    assert health.get('status') == 'ok' and health.get('active_sessions') == [], 'Active sessions or unknown health: STOP'
    code = '''import asyncio,os,json
from livekit import api
async def main():
 c=api.LiveKitAPI(os.environ['LIVEKIT_URL'],os.environ['LIVEKIT_API_KEY'],os.environ['LIVEKIT_API_SECRET'])
 try:
  r=await c.room.list_rooms(api.ListRoomsRequest())
  print(json.dumps({'rooms':len(r.rooms),'participants':sum(x.num_participants for x in r.rooms)}))
 finally: await c.aclose()
asyncio.run(main())
'''
    rooms = json.loads(run(['docker', 'exec', 'caal-agent', '/app/.venv/bin/python', '-c', code]))
    assert rooms == {'rooms': 0, 'participants': 0}, 'LiveKit rooms present: STOP'
    return {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'active_sessions': 0, **rooms}


before = {name: inspect(name) for name in ['agent', 'frontend']}
envs = {name: environment(c) for name, c in before.items()}
private = Path.home() / '.config/caal/qwen-live'
private.mkdir(parents=True, exist_ok=True, mode=0o700)
private.chmod(0o700)
preserve = private / 'compose-preserve.json'
# Compose treats $$ as a literal dollar; do not apply the known .env hash drift.
content = {'services': {'agent': {'environment': {
    'CAAL_BOOTSTRAP_ADMIN_PASSWORD_HASH': envs['agent']['CAAL_BOOTSTRAP_ADMIN_PASSWORD_HASH'].replace('$', '$$')
}}}}
fd = os.open(preserve, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, 'w') as f:
    json.dump(content, f)
preserve.chmod(0o600)
env = os.environ.copy()
env['CAAL_QWEN_TRIAL_TOKEN'] = (P / '.token').read_text().strip()
compose = ['docker', 'compose', '-f', 'docker-compose.apple.yaml', '-f', 'docker-compose.telephony.yaml',
           '-f', str(P / 'docker-compose.qwen-trial.yaml'), '-f', str(preserve)]
rendered = json.loads(run(compose + ['config', '--format', 'json'], env=env))
for name in before:
    ce = rendered['services'][name]['environment']
    expected = dict(envs[name])
    if name == 'agent':
        expected['CAAL_QWEN_TRIAL_TOKEN'] = env['CAAL_QWEN_TRIAL_TOKEN']
    assert all((str(v or '').replace('$$', '$') if k == 'CAAL_BOOTSTRAP_ADMIN_PASSWORD_HASH' else str(v or '')) == expected.get(k) for k, v in ce.items()), 'Unexpected environment drift: STOP'
    image = json.loads(run(['docker', 'image', 'inspect', rendered['services'][name].get('image', before[name]['Config']['Image'])]))[0]
    assert image['Id'] == before[name]['Image'], 'Image drift: STOP'
receipt = {'config_verified': True, 'overlays': compose[2:], 'environment_changed_keys': ['CAAL_QWEN_TRIAL_TOKEN']}
print('Rendered config verified: only agent Qwen credential added; existing environment and images preserved.', flush=True)
# Publish only a coherent, already tested production build.
print(run(['./publish-frontend-build.sh']).strip(), flush=True)
# Independent inventories immediately before agent recreation; neither joins a room.
receipt['pre_recreate_gate'] = gate()
print(json.dumps(receipt['pre_recreate_gate']), flush=True)
run(compose + ['up', '-d', '--no-deps', '--no-build', '--pull', 'never', '--force-recreate', 'agent', 'frontend'], env=env)
receipt['deployed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt['containers'] = {}
for name in before:
    live = inspect(name)
    expected = dict(envs[name])
    if name == 'agent':
        expected['CAAL_QWEN_TRIAL_TOKEN'] = env['CAAL_QWEN_TRIAL_TOKEN']
    assert environment(live) == expected, 'Post-deployment environment differs: STOP'
    assert live['Image'] == before[name]['Image']
    receipt['containers'][name] = {'id': live['Id'], 'image': live['Image'], 'started_at': live['State']['StartedAt']}
receipt['environment_preserved'] = True
(P / 'live-deployment.json').write_text(json.dumps(receipt, indent=2))
print('Agent and frontend recreated; exact runtime environments verified.', flush=True)
