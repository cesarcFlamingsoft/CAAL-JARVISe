"""Explicitly authorized, frontend-only deployment through CAAL's full manifest."""
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path('/Users/cesar/caal')
MANIFEST = Path.home() / '.config/caal/startup/manifest.json'
RUNTIME_OVERLAY = Path.home() / '.config/caal/runtime/compose-runtime-effective.json'
ROLLBACK = MANIFEST.parent / 'frontend-rollback-20260917'
sys.path.insert(0, str(ROOT))
import startup_runtime as ops


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env(container: dict) -> dict[str, str]:
    return dict(item.split('=', 1) for item in container['Config']['Env'])


def reconcile_current_baseline() -> None:
    """Repair stale pins from earlier direct deployment without changing a service."""
    data = json.loads(MANIFEST.read_text())
    artifact = Path(data['artifact'])
    current = artifact.joinpath('BUILD_ID').read_text().strip()
    if not current or artifact.joinpath('.next/BUILD_ID').read_text().strip() != current:
        raise RuntimeError('Cannot reconcile an incomplete published artifact')
    overlay = json.loads(RUNTIME_OVERLAY.read_text())
    overlay['services']['frontend']['environment']['CAAL_FRONTEND_EXPECTED_BUILD_ID'] = current
    staged_overlay = RUNTIME_OVERLAY.with_suffix('.baseline-tmp')
    staged_overlay.write_text(json.dumps(overlay, indent=2) + '\n')
    os.chmod(staged_overlay, 0o600)
    os.replace(staged_overlay, RUNTIME_OVERLAY)
    data['build_id'] = current
    data['artifact_hashes'] = {
        str(path.relative_to(artifact)): sha(path) for path in artifact.rglob('*') if path.is_file()
    }
    # The currently running local Qwen service is the approved live voice; pin
    # its authenticated style receipt so a later restart cannot accept drift.
    import urllib.request
    token = Path(data['token_file']).read_text().strip()
    request = urllib.request.Request('http://127.0.0.1:18003/voice', headers={'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(request, timeout=4) as response:
        voice = json.load(response)
    if not voice.get('style_sha256'):
        raise RuntimeError('Current Qwen voice receipt is incomplete')
    data['style_hash'] = voice['style_sha256']
    data['file_hashes'] = {name: sha(Path(name)) for name in data.get('file_hashes', {})}
    data['file_hashes'][str(RUNTIME_OVERLAY)] = sha(RUNTIME_OVERLAY)
    for name in data['containers']:
        container = json.loads(__import__('subprocess').check_output(['docker', 'inspect', name]))[0]
        data['containers'][name] = {
            'image': container['Image'],
            'env_hash': hashlib.sha256(json.dumps(env(container), sort_keys=True).encode()).hexdigest(),
        }
    staged_manifest = MANIFEST.with_suffix('.baseline-tmp')
    staged_manifest.write_text(json.dumps(data, indent=2) + '\n')
    os.chmod(staged_manifest, 0o600)
    os.replace(staged_manifest, MANIFEST)
    ops.load_manifest(MANIFEST)


def main() -> None:
    try:
        data = ops.load_manifest(MANIFEST)
    except ops.StartupError:
        reconcile_current_baseline()
        data = ops.load_manifest(MANIFEST)
    runtime = ops.Runtime(data)
    build = (ROOT / 'frontend/.next/BUILD_ID').read_text().strip()
    served = __import__('subprocess').check_output(['docker', 'exec', 'caal-frontend', 'cat', '/app/.next/BUILD_ID'], text=True).strip()
    if not build or (build == data['build_id'] and served == build):
        raise RuntimeError('Expected a new complete frontend build')
    if RUNTIME_OVERLAY not in [Path(item) for item in data['compose_files']]:
        raise RuntimeError('Effective runtime overlay is not part of the authoritative stack')

    with ops.startup_lock(data['lock_file']):
        runtime.validate()
        runtime.native_health(attempts=1)
        # The published artifact deliberately differs from the currently
        # served frontend until the targeted replacement below; validate voice,
        # worker and room idleness without applying the served-build check yet.
        runtime.idle_gate()
        before = {name: runtime.inspect(name) for name in data['containers']}

        if not ROLLBACK.exists():
            ROLLBACK.mkdir(mode=0o700)
            shutil.copy2(MANIFEST, ROLLBACK / 'manifest.json')
            shutil.copy2(RUNTIME_OVERLAY, ROLLBACK / 'compose-runtime-effective.json')
            os.chmod(ROLLBACK / 'manifest.json', 0o600)
            os.chmod(ROLLBACK / 'compose-runtime-effective.json', 0o600)
        elif not (ROLLBACK / 'manifest.json').is_file() or not (ROLLBACK / 'compose-runtime-effective.json').is_file():
            raise RuntimeError('Existing rollback directory is incomplete')

        artifact = Path(data['artifact'])
        already_published = artifact.joinpath('BUILD_ID').is_file() and artifact.joinpath('BUILD_ID').read_text().strip() == build
        if not already_published:
            runtime.run([str(ROOT / 'publish-frontend-build.sh')], timeout=120)
            previous = artifact.with_name(artifact.name + '.previous')
            if previous.joinpath('BUILD_ID').read_text().strip() != data['build_id']:
                raise RuntimeError('Published artifact rollback marker mismatch')
            for name, digest in data['artifact_hashes'].items():
                if sha(previous / name) != digest:
                    raise RuntimeError('Published artifact rollback integrity mismatch')

        overlay = json.loads(RUNTIME_OVERLAY.read_text())
        overlay['services']['frontend']['environment']['CAAL_FRONTEND_EXPECTED_BUILD_ID'] = build
        staged = RUNTIME_OVERLAY.with_suffix('.deploy-tmp')
        staged.write_text(json.dumps(overlay, indent=2) + '\n')
        os.chmod(staged, 0o600)
        os.replace(staged, RUNTIME_OVERLAY)

        # Validate the full ordered Compose stack but never print expanded private config.
        rendered = json.loads(runtime.run(runtime.compose + ['config', '--format', 'json']))
        declared = rendered['services']['frontend']['environment']
        expected = {key: '' if value is None else str(value) for key, value in declared.items()}
        if expected.get('CAAL_FRONTEND_EXPECTED_BUILD_ID') != build:
            raise RuntimeError('Rendered frontend build pin is not the requested artifact')
        # The old container predates a newly declared key only when that key is
        # part of this render. Every existing value must remain identical apart
        # from the deliberate build-pin replacement.
        for key, value in env(before['caal-frontend']).items():
            if key != 'CAAL_FRONTEND_EXPECTED_BUILD_ID' and expected.get(key) != value:
                raise RuntimeError('Rendered frontend environment changed an existing runtime value')

        runtime.idle_gate()
        runtime.run(
            runtime.compose + ['up', '-d', '--force-recreate', '--no-deps', '--no-build', '--pull', 'never', 'frontend'],
            timeout=180,
        )
        after = {name: runtime.inspect(name) for name in data['containers']}
        for name in data['containers']:
            if name != 'caal-frontend' and before[name]['Id'] != after[name]['Id']:
                raise RuntimeError('A non-frontend service changed during frontend deployment')
        frontend = after['caal-frontend']
        if env(frontend) != expected:
            raise RuntimeError('Frontend runtime environment does not match rendered contract')
        if runtime.run(runtime.docker + ['exec', 'caal-frontend', 'cat', '/app/.next/BUILD_ID']).strip() != build:
            raise RuntimeError('Frontend served build mismatch after recreation')

        updated = copy.deepcopy(data)
        updated['build_id'] = build
        updated['artifact_hashes'] = {
            str(path.relative_to(artifact)): sha(path) for path in artifact.rglob('*') if path.is_file()
        }
        updated.setdefault('file_hashes', {})[str(RUNTIME_OVERLAY)] = sha(RUNTIME_OVERLAY)
        updated['containers']['caal-frontend'] = {
            'image': frontend['Image'],
            'env_hash': hashlib.sha256(json.dumps(env(frontend), sort_keys=True).encode()).hexdigest(),
        }
        verified = ops.Runtime(updated)
        verified.validate()
        verified.native_health(attempts=1)
        verified.verify(attempts=1)
        verified.idle_gate()
        staged_manifest = MANIFEST.with_suffix('.deploy-tmp')
        staged_manifest.write_text(json.dumps(updated, indent=2) + '\n')
        os.chmod(staged_manifest, 0o600)
        os.replace(staged_manifest, MANIFEST)
        ops.load_manifest(MANIFEST)
        print(json.dumps({
            'build_id': build,
            'frontend_recreated': before['caal-frontend']['Id'] != frontend['Id'],
            'other_services_unchanged': True,
            'rollback': str(ROLLBACK),
            'artifact_files': len(updated['artifact_hashes']),
        }))


if __name__ == '__main__':
    main()
