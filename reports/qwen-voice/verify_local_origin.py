"""Anonymous deployment smoke only: no passwords, vault, or user session tokens."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request
from http.cookies import SimpleCookie

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
LOCAL = 'http://localhost:3000'
PUBLIC = 'https://jarvis.mexcantech.io'


def request(path, method='GET', body=None, headers=None):
    req = urllib.request.Request(LOCAL + path, data=body, method=method, headers=headers or {})
    try:
        response = urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, response.headers, response.read()


def main():
    results = {}
    status, headers, data = request('/api/auth/me')
    identity = json.loads(data)
    assert status == 200 and identity['configured'] and not identity['authenticated'] and identity['passwordLogin']
    assert 'user' not in identity
    csrf = identity['csrfToken']  # newly issued anonymous CSRF; never printed or saved
    cookies = SimpleCookie()
    for value in headers.get_all('set-cookie', []):
        cookies.load(value)
    assert 'caal_session' not in cookies
    assert cookies['caal_csrf']['httponly'] and cookies['caal_csrf']['samesite'] == 'strict'
    assert not cookies['caal_csrf']['secure']
    base = {'Origin': LOCAL, 'Content-Type': 'application/json', 'Cookie': f'caal_csrf={csrf}', 'X-CAAL-CSRF': csrf}
    cases = [
        ('local_empty_login', {}, b'{}', 401, 'invalid_credentials'),
        ('local_invalid_body', {}, b'[]', 401, 'invalid_credentials'),
        ('local_missing_csrf', {'X-CAAL-CSRF': ''}, b'{}', 403, 'csrf'),
        ('public_empty_login', {'Origin': PUBLIC, 'Host': 'jarvis.mexcantech.io', 'X-Forwarded-Proto': 'https'}, b'{}', 401, 'invalid_credentials'),
        ('hostile_origin', {'Origin': 'https://evil.example'}, b'{}', 403, 'bad_origin'),
        ('null_origin', {'Origin': 'null'}, b'{}', 403, 'bad_origin'),
        ('malformed_origin', {'Origin': LOCAL + '/path'}, b'{}', 403, 'bad_origin'),
        ('other_local_port', {'Origin': 'http://localhost:3001'}, b'{}', 403, 'bad_origin'),
        ('host_forwarded_spoof', {'Origin': 'http://evil.example', 'Host': 'evil.example', 'X-Forwarded-Host': 'evil.example'}, b'{}', 403, 'bad_origin'),
    ]
    for label, extra, body, expected_status, expected_error in cases:
        status, headers, data = request('/api/auth/login', 'POST', body, {**base, **extra})
        error = json.loads(data).get('error')
        assert (status, error) == (expected_status, expected_error), (label, status, error)
        assert headers.get('cache-control') == 'no-store'
        assert headers.get('access-control-allow-origin') is None
        assert not any('caal_session=' in value for value in headers.get_all('set-cookie', []))
        results[label] = {'status': status, 'error': error}
    for label, extra in [('local_logout', {}), ('public_logout', {'Origin': PUBLIC, 'Host': 'jarvis.mexcantech.io', 'X-Forwarded-Proto': 'https'})]:
        status, headers, data = request('/api/auth/logout', 'POST', b'{}', {**base, **extra})
        assert status == 200 and json.loads(data)['ok']
        cleared = SimpleCookie()
        for value in headers.get_all('set-cookie', []):
            cleared.load(value)
        assert cleared['caal_session'].value == '' and cleared['caal_session']['max-age'] == '0'
        assert bool(cleared['caal_session']['secure']) == (label == 'public_logout')
        results[label] = {'status': status, 'secure_cookie': bool(cleared['caal_session']['secure'])}
    for method in ['GET', 'PUT']:
        status, headers, data = request('/api/tts', method, b'{}' if method == 'PUT' else None, base)
        assert status == 401 and headers.get('cache-control') == 'no-store'
        results['anonymous_tts_' + method] = {'status': status}
    for path in ['/', '/login']:
        status, headers, data = request(path)
        assert status == 200
        assets = re.findall(rb'src="(/_next/static/[^"?]+\.js)', data)
        assert assets
        for asset in set(assets):
            assert request(asset.decode())[0] == 200
        results[path] = {'status': status, 'served_js_assets': len(set(assets))}
    served = subprocess.check_output(['docker', 'exec', 'caal-frontend', 'cat', '/app/.next/BUILD_ID'], text=True).strip()
    expected = (ROOT / 'frontend/.next-deploy/BUILD_ID').read_text().strip()
    assert served == expected
    results['build'] = served
    info = json.loads(subprocess.check_output(['docker', 'inspect', 'caal-frontend']))[0]
    assert info['State']['Health']['Status'] == 'healthy'
    results['frontend_health'] = 'healthy'
    with urllib.request.urlopen('http://127.0.0.1:8889/health', timeout=5) as response:
        health = json.load(response)
    assert health['status'] == 'ok'
    results['backend_health'] = 'ok'
    results['active_sessions'] = len(health['active_sessions'])
    baseline = json.loads((P / 'local-origin-preservation.json').read_text())
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest for path, digest in baseline.items())
    results['unrelated_source_files_unchanged'] = len(baseline)
    (P / 'local-origin-verification.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
