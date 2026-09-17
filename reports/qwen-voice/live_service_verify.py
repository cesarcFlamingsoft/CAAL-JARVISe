import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import wave

import httpx
import numpy as np

P = Path(__file__).parent
URL = 'http://127.0.0.1:18003'
AUTH = {'Authorization': 'Bearer ' + (P / '.token').read_text().strip()}


def pid():
    return subprocess.check_output(['lsof', '-t', '-iTCP:18003', '-sTCP:LISTEN'], text=True).strip()


result = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pid_before': pid()}
with httpx.Client(timeout=20, trust_env=False) as c:
    assert c.get(URL + '/health', headers=AUTH).json() == {'status': 'ok', 'busy': False}
    # Only the isolated Qwen service; the current selected provider is still Kokoro.
    subprocess.run(['launchctl', 'kill', 'SIGTERM', f'gui/{os.getuid()}/com.caal.qwen-trial'], check=True)
    start = time.perf_counter()
    deadline = start + 60
    while time.perf_counter() < deadline:
        time.sleep(.25)
        try:
            new_pid = pid()
            r = c.get(URL + '/health', headers=AUTH)
            if new_pid != result['pid_before'] and r.status_code == 200:
                result['pid_after'] = new_pid
                result['restart_to_prewarmed_health_s'] = time.perf_counter() - start
                break
        except (httpx.HTTPError, subprocess.CalledProcessError):
            pass
    else:
        raise RuntimeError('Supervised restart failed')
    result['anonymous_health'] = c.get(URL + '/health').status_code
    result['authenticated_health'] = c.get(URL + '/health', headers=AUTH).status_code
    result['measurements'] = []
    for n in range(3):
        start = time.perf_counter()
        chunks = []
        with c.stream('POST', URL + '/v1/audio/speech', headers=AUTH,
                      json={'input': 'Understood. I am ready to help.'}) as r:
            r.raise_for_status()
            headers_s = time.perf_counter() - start
            for chunk in r.iter_bytes():
                if chunk:
                    if not chunks:
                        first_s = time.perf_counter() - start
                    chunks.append(chunk)
        complete_s = time.perf_counter() - start
        data = b''.join(chunks)
        assert data and len(data) % 2 == 0
        samples = np.frombuffer(data, dtype='<i2').astype(float) / 32768
        windows = [float(np.sqrt(np.mean(samples[i:i+480]**2))) for i in range(0, len(samples), 480)]
        leading = next((i * .02 for i, value in enumerate(windows) if value > .001), None)
        file = P / f'live-host-qwen-{n}.wav'
        with wave.open(str(file), 'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(data)
        result['measurements'].append({'repeat': n, 'headers_s': headers_s, 'first_pcm_s': first_s,
                                       'generation_s': complete_s, 'audio_s': len(data)/48000,
                                       'leading_silence_s': leading})
    with c.stream('POST', URL + '/v1/audio/speech', headers=AUTH,
                  json={'input': 'Understood. I am ready to help with a longer explanation.'}) as r:
        r.raise_for_status()
        result['cancel_after_audio_bytes'] = len(next(r.iter_bytes()))
    start = time.perf_counter()
    while time.perf_counter() - start < 15:
        if not c.get(URL + '/health', headers=AUTH).json()['busy']:
            break
        time.sleep(.02)
    else:
        raise RuntimeError('Cancellation cleanup failed')
    result['cancel_cleanup_s'] = time.perf_counter() - start
    r = c.post(URL + '/v1/audio/speech', headers=AUTH, json={'input': 'I am ready.'})
    r.raise_for_status();assert r.content
    result['synthesis_after_cancellation_bytes'] = len(r.content)
    result['final_health'] = c.get(URL + '/health', headers=AUTH).json()
result['process'] = subprocess.check_output(['ps','-p',result['pid_after'],'-o','pid=,lstart=,rss='],text=True).strip()
result['launchd_installed_plist'] = str(Path.home() / 'Library/LaunchAgents/com.caal.qwen-trial.plist')
(P / 'live-service-verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
