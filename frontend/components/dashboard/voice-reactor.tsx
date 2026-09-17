'use client';

import type { VoiceStatus } from '@/lib/dashboard/activity';

/** Decorative instruments encode state only; ring lengths are never telemetry. */
export function VoiceReactor({ status, transport }: { status: VoiceStatus; transport: string }) {
  return (
    <div className="voice-reactor" data-tone={status.tone}>
      <div className="reactor-coordinate reactor-coordinate-top" aria-hidden>
        VOICE INTERFACE / F.R.I.D.A.Y.
      </div>
      <svg className="reactor-instruments" viewBox="0 0 400 400" fill="none" aria-hidden="true">
        <path
          className="reactor-trace"
          d="M0 85H48L85 122H115M400 315H352L315 278H285M0 315H50L92 273M400 85H350L308 127"
        />
        <circle cx="200" cy="200" r="183" className="reactor-hairline" />
        <circle
          cx="200"
          cy="200"
          r="174"
          className="reactor-ticks"
          pathLength="120"
          strokeDasharray=".25 .75"
        />
        <g className="reactor-orbit">
          <circle
            cx="200"
            cy="200"
            r="159"
            className="reactor-arc"
            pathLength="100"
            strokeDasharray="23 5 12 10"
          />
          <circle
            cx="200"
            cy="200"
            r="149"
            className="reactor-hairline"
            pathLength="100"
            strokeDasharray="42 8"
          />
        </g>
        <circle cx="200" cy="200" r="137" className="reactor-hairline" />
        <g className="reactor-orbit reactor-orbit-reverse">
          <circle
            cx="200"
            cy="200"
            r="125"
            className="reactor-inner"
            pathLength="100"
            strokeDasharray="29 4"
          />
          <circle
            cx="200"
            cy="200"
            r="113"
            className="reactor-hairline"
            pathLength="100"
            strokeDasharray="16 9"
          />
        </g>
        <path className="reactor-trace" d="M194 8H206M194 392H206M8 194V206M392 194V206" />
      </svg>
      <div className="reactor-nucleus">
        <span className="reactor-identity">F.R.I.D.A.Y.</span>
        <span className="reactor-mode">
          {status.tone === 'idle'
            ? 'STANDBY'
            : status.tone === 'error'
              ? 'ATTENTION'
              : status.tone === 'busy'
                ? 'PROCESSING'
                : 'VOICE ACTIVE'}
        </span>
      </div>
      <div className="reactor-coordinate reactor-coordinate-bottom">
        Transport <span>{transport}</span>
      </div>
    </div>
  );
}
