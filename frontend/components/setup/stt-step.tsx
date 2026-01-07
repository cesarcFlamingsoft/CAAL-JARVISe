'use client';

import type { SetupData } from './setup-wizard';

interface SttStepProps {
  data: SetupData;
  updateData: (updates: Partial<SetupData>) => void;
}

export function SttStep({ data, updateData }: SttStepProps) {
  return (
    <div className="space-y-4">
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
            disabled
            className={`rounded-lg border p-4 text-left transition-colors ${
              data.tts_provider === 'piper'
                ? 'border-primary bg-primary/5'
                : 'border-input hover:border-muted-foreground'
            } disabled:cursor-not-allowed disabled:opacity-50`}
          >
            <div className="flex items-center gap-2 font-medium">
              Piper
              <span className="bg-muted text-muted-foreground rounded px-1.5 py-0.5 text-xs">
                Coming Soon
              </span>
            </div>
            <div className="text-muted-foreground text-xs">CPU-based, lightweight and fast</div>
          </button>
        </div>
      </div>

      <p className="text-muted-foreground text-xs">
        Kokoro requires a GPU for optimal performance. Piper will be available for CPU-only setups.
      </p>
    </div>
  );
}
