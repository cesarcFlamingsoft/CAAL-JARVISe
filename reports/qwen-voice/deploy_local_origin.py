"""Publish and recreate only frontend, preserving runtime env without secret output."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'reports/qwen-voice'
os.chdir(ROOT)


def run(args, **kwargs):
    result = subprocess.run(args, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError('Command failed; private output suppressed: ' + args[0])
    return result.stdout


def inspect(name):
    return json.loads(run(['docker', 'inspect', name]))[0]


def environment(container):
    return dict(item.split('=', 1) for item in container['Config']['Env'])


def inventory():
    names = run(['docker', 'ps', '--format', '{{.Names}}']).splitlines()
    return {name: {'id': (c := inspect(name))['Id'], 'started_at': c['State']['StartedAt']} for name in names}


def gate():
    with urllib.request.urlopen('http://127.0.0.1:8889/health', timeout=5) as response:
        health = json.load(response)
    assert health.get('status') == 'ok' and health.get('active_sessions') == [], 'Sessions present or health unknown: STOP'
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
    assert rooms == {'rooms': 0, 'participants': 0}, 'Rooms present: STOP'
    return {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'active_sessions': 0, **rooms}


def main():
    before_all = inventory()
    before = inspect('caal-frontend')
    env_before = environment(before)
    agent_env = environment(inspect('caal-agent'))
    expected = {**env_before, 'CAAL_TRUSTED_LOCAL_ORIGINS': 'http://localhost:3000'}
    assert expected['CAAL_PUBLIC_ORIGIN'] == 'https://jarvis.mexcantech.io'
    assert expected['CAAL_ALLOW_INSECURE_COOKIES'] == 'false'
    private = Path.home() / '.config/caal/qwen-live'
    assert (private / 'compose-preserve.json').is_file(), 'Existing hash preservation overlay missing: STOP'
    preserve = private / 'compose-frontend-preserve.json'
    # Preserve the complete actual frontend environment, including literal dollars.
    content = {'services': {'frontend': {'environment': {k: v.replace('$', '$$') for k, v in env_before.items()}}}}
    fd = os.open(preserve, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as output:
        json.dump(content, output)
    preserve.chmod(0o600)
    env = os.environ.copy()
    env['CAAL_QWEN_TRIAL_TOKEN'] = agent_env['CAAL_QWEN_TRIAL_TOKEN']
    compose = ['docker', 'compose', '-f', 'docker-compose.apple.yaml', '-f', 'docker-compose.telephony.yaml',
               '-f', str(P / 'docker-compose.qwen-trial.yaml'), '-f', str(private / 'compose-preserve.json'),
               '-f', str(preserve), '-f', str(P / 'docker-compose.local-origin.yaml')]
    rendered = json.loads(run(compose + ['config', '--format', 'json'], env=env))
    for name, actual in [('frontend', expected), ('agent', agent_env)]:
        values = rendered['services'][name]['environment']
        assert all(str(v or '').replace('$$', '$') == actual.get(k) for k, v in values.items()), 'Rendered environment drift: STOP'
    image = json.loads(run(['docker', 'image', 'inspect', rendered['services']['frontend'].get('image', before['Config']['Image'])]))[0]
    assert image['Id'] == before['Image'], 'Frontend image drift: STOP'
    assert run(['docker', 'exec', 'caal-frontend', 'cat', '/app/.next/BUILD_ID']).strip() == (ROOT / 'frontend/.next-deploy/BUILD_ID').read_text().strip()
    baseline = json.loads((P / 'local-origin-preservation.json').read_text())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in baseline.items()), 'Unrelated source drift: STOP'
    receipt = {'previous_build': (ROOT / 'frontend/.next-deploy/BUILD_ID').read_text().strip(), 'before': before_all,
               'environment_changed_keys': ['CAAL_TRUSTED_LOCAL_ORIGINS'], 'pre_publish_gate': gate()}
    print('Config and image verified; no active sessions or rooms.', flush=True)
    print(run(['./publish-frontend-build.sh']).strip(), flush=True)
    receipt['pre_recreate_gate'] = gate()
    run(compose + ['up', '-d', '--no-deps', '--no-build', '--pull', 'never', '--force-recreate', 'frontend'], env=env)
    after = inspect('caal-frontend')
    assert environment(after) == expected, 'Runtime environment mismatch: STOP'
    assert after['Image'] == before['Image']
    after_all = inventory()
    for name, record in before_all.items():
        if name != 'caal-frontend':
            assert after_all.get(name) == record, 'Unrelated container changed: STOP'
    receipt.update({'after': after_all, 'build': (ROOT / 'frontend/.next-deploy/BUILD_ID').read_text().strip(),
                    'environment_preserved': True, 'unrelated_containers_unchanged': True,
                    'at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    (P / 'local-origin-deployment.json').write_text(json.dumps(receipt, indent=2))
    print('Only frontend recreated; runtime environment and unrelated containers verified.', flush=True)


if __name__ == '__main__':
    main()
