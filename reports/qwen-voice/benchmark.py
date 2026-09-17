"""Synthetic HTTP audition benchmark. Never contacts tools, rooms, or private data."""
import io
import json
import time
import wave
from pathlib import Path
import httpx
import numpy as np

ROOT=Path(__file__).parent
TEXTS={
 'ack':'Understood. I will prepare a concise summary.',
 'story':'The old lighthouse had one rule: never lend the keeper your umbrella. One rainy evening, a fox ignored it. By morning, the umbrella was dry, the keeper was missing, and the fox had a very respectable new job.',
}

def decode_wavs(raw):
    parts=[]; offset=0
    while offset+12<=len(raw):
        if raw[offset:offset+4]!=b'RIFF':raise ValueError('Invalid WAV')
        size=int.from_bytes(raw[offset+4:offset+8],'little')+8
        segment=raw[offset:offset+size]
        try:
            with wave.open(io.BytesIO(segment),'rb') as w:
                assert w.getsampwidth()==2 and w.getnchannels()==1 and w.getframerate()==24000
                parts.append(w.readframes(w.getnframes()))
        except (EOFError,wave.Error):break
        if len(segment)<size:break
        offset+=size
    return b''.join(parts)


def run(provider, name, repeat, client):
    text=TEXTS[name]
    if provider=='qwen':
        url='http://127.0.0.1:18003/v1/audio/speech'
        payload={'input':text,'model':'qwen-trial','voice':'jarvis-designed','response_format':'pcm'}
        headers={'Authorization':'Bearer '+(ROOT/'.token').read_text().strip()}
    else:
        url='http://127.0.0.1:8001/v1/audio/speech'
        payload={'input':text,'model':'prince-canuma/Kokoro-82M','voice':'am_adam','response_format':'wav'}
        headers={}
    raw=b'';events=[];first_byte=first_decoded=first_playable=first_signal=None
    started=time.perf_counter()
    with client.stream('POST',url,json=payload,headers=headers) as r:
        header_time=time.perf_counter()-started
        r.raise_for_status()
        for chunk in r.iter_bytes():
            now=time.perf_counter()-started
            if not chunk:continue
            if first_byte is None:first_byte=now
            raw+=chunk
            pcm=raw[:len(raw)//2*2] if provider=='qwen' else decode_wavs(raw)
            pcm=pcm[:len(pcm)//2*2]
            if pcm and first_decoded is None:first_decoded=time.perf_counter()-started
            if len(pcm)>=960 and first_playable is None:first_playable=time.perf_counter()-started
            values=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
            full=len(values)//480*480
            if full:
                rms=np.sqrt(np.mean(values[:full].reshape(-1,480)**2,axis=1))
                if (rms>.001).any() and first_signal is None:first_signal=time.perf_counter()-started
            events.append({'at_s':now,'bytes':len(chunk),'decoded_samples':len(pcm)//2})
    complete=time.perf_counter()-started
    pcm=raw if provider=='qwen' else decode_wavs(raw)
    if not pcm or len(pcm)%2:raise ValueError('Incomplete PCM')
    name_out=f'{provider}-{name}-{repeat}'
    with wave.open(str(ROOT/(name_out+'.wav')),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(pcm)
    v=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
    frames=len(v)//480*480
    rms=np.sqrt(np.mean(v[:frames].reshape(-1,480)**2,axis=1))
    leading=float(np.argmax(rms>.001)*.02) if (rms>.001).any() else None
    row={'provider':provider,'text_name':name,'text':text,'repeat':repeat,'headers_s':header_time,
      'first_byte_s':first_byte,'first_decoded_audio_s':first_decoded,'first_playable_20ms_s':first_playable,
      'first_signal_received_s':first_signal,'leading_silence_s_at_minus60dB':leading,
      'complete_s':complete,'audio_s':len(v)/24000,'generation_x_realtime':len(v)/24000/complete,
      'peak':float(np.max(np.abs(v))),'events':events,'artifact':name_out+'.wav'}
    with (ROOT/'http-measurements.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps({k:row[k] for k in ['provider','text_name','repeat','first_byte_s','first_decoded_audio_s','first_signal_received_s','complete_s','audio_s']}),flush=True)

if __name__=='__main__':
    with httpx.Client(timeout=httpx.Timeout(75,connect=5),trust_env=False) as client:
        for repeat in range(5):
            for name in TEXTS:
                for provider in ['qwen','kokoro']:
                    run(provider,name,repeat,client)
