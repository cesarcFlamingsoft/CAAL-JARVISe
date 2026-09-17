import json,socket,time,subprocess,wave
from pathlib import Path
import httpx
P=Path(__file__).parent
TOKEN=(P/'.token').read_text().strip()
AUTH={'Authorization':'Bearer '+TOKEN}
URL='http://127.0.0.1:18003'
result={}

def wait_idle(client):
    start=time.perf_counter()
    while time.perf_counter()-start<15:
        r=client.get(URL+'/health',headers=AUTH);r.raise_for_status()
        if not r.json()['busy']:return time.perf_counter()-start
        time.sleep(.05)
    raise RuntimeError('Worker did not finish cleanup')

with httpx.Client(timeout=20,trust_env=False) as client:
    result['anonymous_health_status']=client.get(URL+'/health').status_code
    result['authenticated_health_status']=client.get(URL+'/health',headers=AUTH).status_code
    payload=json.dumps({'input':'Understood. I will prepare a concise summary.'}).encode()
    sock=socket.create_connection(('127.0.0.1',18003),timeout=5)
    req=(f'POST /v1/audio/speech HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {TOKEN}\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n').encode()+payload
    sock.sendall(req);time.sleep(.05);sock.close()
    # Allow disconnect to be observed before polling the cleanup state.
    time.sleep(.05)
    result['cancel_before_first_audio_cleanup_s']=wait_idle(client)
    with client.stream('POST',URL+'/v1/audio/speech',json={'input':'The old lighthouse had one rule: never lend the keeper your umbrella.'},headers=AUTH) as r:
        r.raise_for_status()
        result['cancel_after_first_audio_bytes']=len(next(r.iter_bytes()))
    time.sleep(.05)
    result['cancel_after_first_audio_cleanup_s']=wait_idle(client)
    r=client.post(URL+'/v1/audio/speech',json={'input':'Understood. I will prepare a concise summary.'},headers=AUTH)
    r.raise_for_status();assert r.content and len(r.content)%2==0
    with wave.open(str(P/'qwen-after-cancellation.wav'),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(r.content)
    result['synthesis_after_cancellation_bytes']=len(r.content)
    result['mlx_models_after']=client.get('http://localhost:8001/v1/models').json()
    result['agent_health_after']=client.get('http://localhost:8889/health').json()
code="import sys,urllib.request; token=sys.stdin.read().strip(); r=urllib.request.urlopen(urllib.request.Request('http://host.docker.internal:18003/health',headers={'Authorization':'Bearer '+token}),timeout=5);print(r.status)"
p=subprocess.run(['docker','exec','-i','caal-agent','/app/.venv/bin/python','-c',code],input=TOKEN,text=True,capture_output=True,check=True)
result['docker_authenticated_health_status']=int(p.stdout.strip())
code="""import asyncio,os,json
from livekit import api
async def main():
 c=api.LiveKitAPI(os.environ['LIVEKIT_URL'],os.environ['LIVEKIT_API_KEY'],os.environ['LIVEKIT_API_SECRET'])
 try:
  rooms=await c.room.list_rooms(api.ListRoomsRequest())
  print(json.dumps({'independent_livekit_rooms':len(rooms.rooms),'participants':sum(r.num_participants for r in rooms.rooms)}))
 finally:await c.aclose()
asyncio.run(main())
"""
p=subprocess.run(['docker','exec','caal-agent','/app/.venv/bin/python','-c',code],text=True,capture_output=True,check=True)
result.update(json.loads(p.stdout))
p=subprocess.run(['docker','logs','caal-agent'],text=True,capture_output=True,check=True)
result['agent_worker_registration_in_logs']='registered worker' in p.stdout+p.stderr
(P/'runtime-verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
