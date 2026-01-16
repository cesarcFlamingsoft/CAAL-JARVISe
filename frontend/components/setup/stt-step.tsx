'use client';

import { useEffect, useState } from 'react';
import type { SetupData } from './setup-wizard';

interface SttStepProps {
  data: SetupData;
  updateData: (updates: Partial<SetupData>) => void;
}

export function SttStep({ data, updateData }: SttStepProps) {
  const [voices, setVoices] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  // Fetch voices when provider changes
  useEffect(() => {
    const fetchVoices = async () => {
      setLoading(true);
      try {
        const res = await fetch(`/api/voices?provider=${data.tts_provider}`);
        if (res.ok) {
          const json = await res.json();
          setVoices(json.voices || []);
        }
      } catch (err) {
        console.error('Failed to fetch voices:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchVoices();
  }, [data.tts_provider]);

  const currentVoice = data.tts_provider === 'piper' ? data.tts_voice_piper : data.tts_voice_kokoro;

  const handleVoiceChange = (voice: string) => {
    if (data.tts_provider === 'piper') {
      updateData({ tts_voice_piper: voice });
    } else {
      updateData({ tts_voice_kokoro: voice });
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <label className="text-sm font-medium">Text-to-Speech Engine</label>
        <div className="grid grid-cols-1 gap-2">
          <button
            onClick={() => updateData({ tts_provider: 'kokoro' })}
            className={`rounded-lg border p-4 text-left transition-colors ${
              data.tts_provider === 'kokoro'
                ? 'border-primary bg-primary/5'
                : 'border-input hover:border-muted-foreground'
            }`}
          >
            <div className="font-medium">Kokoro</div>
            <div className="text-muted-foreground text-xs">
              GPU-accelerated, high quality neural TTS
            </div>
          </button>
          <button
            onClick={() => updateData({ tts_provider: 'piper' })}
            className={`rounded-lg border p-4 text-left transition-colors ${
              data.tts_provider === 'piper'
                ? 'border-primary bg-primary/5'
                : 'border-input hover:border-muted-foreground'
            }`}
          >
            <div className="font-medium">Piper</div>
            <div className="text-muted-foreground text-xs">
              CPU-friendly, lightweight with 35+ languages
            </div>
          </button>
        </div>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Voice</label>
        <select
          value={currentVoice}
          onChange={(e) => handleVoiceChange(e.target.value)}
          disabled={loading}
          className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm disabled:opacity-50"
        >
          {voices.length > 0 ? (
            voices.map((voice) => (
              <option key={voice} value={voice}>
                {voice}
              </option>
            ))
          ) : (
            <option value={currentVoice}>{currentVoice}</option>
          )}
        </select>
      </div>

      <p className="text-muted-foreground text-xs">
        {data.tts_provider === 'kokoro'
          ? 'Kokoro requires a GPU for optimal performance.'
          : 'Piper runs on CPU and supports many languages.'}
      </p>
    </div>
  );
}
