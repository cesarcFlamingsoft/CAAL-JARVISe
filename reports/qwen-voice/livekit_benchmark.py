"""Offline LiveKit TTS SDK test: no microphone, room, LLM, tool, or call."""
import asyncio,json,time,wave
from pathlib import Path
from livekit.agents import tts
from livekit.plugins import openai
from caal.qwen_tts import QwenTTS,sentence_adapter
from benchmark import TEXTS

ROOT=Path(__file__).parent

async def main():
    q=sentence_adapter(QwenTTS(endpoint='http://127.0.0.1:18003',token=(ROOT/'.token').read_text().strip()))
    kbase=openai.TTS(base_url='http://localhost:8001/v1',api_key='not-needed',model='prince-canuma/Kokoro-82M',voice='am_adam')
    k=tts.StreamAdapter(tts=kbase)
    try:
        for repeat in range(3):
            for name,text in TEXTS.items():
                for provider,obj in [('qwen',q),('kokoro',k)]:
                    start=time.perf_counter();frames=[];events=[]
                    async with obj.stream() as stream:
                        stream.push_text(text);stream.end_input()
                        async for event in stream:
                            events.append({'at_s':time.perf_counter()-start,'samples':event.frame.samples_per_channel})
                            frames.append(bytes(event.frame.data))
                    complete=time.perf_counter()-start
                    pcm=b''.join(frames)
                    file=f'livekit-{provider}-{name}-{repeat}.wav'
                    with wave.open(str(ROOT/file),'wb') as w:
                        w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(pcm)
                    row={'provider':provider,'text':name,'repeat':repeat,'first_sdk_frame_s':events[0]['at_s'],
                         'complete_s':complete,'audio_s':len(pcm)/48000,'events':events,'file':file}
                    with (ROOT/'livekit-measurements.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print({k:row[k] for k in ['provider','text','repeat','first_sdk_frame_s','complete_s','audio_s']},flush=True)
    finally:
        await q.aclose();await k.aclose();await kbase.aclose()

asyncio.run(main())
