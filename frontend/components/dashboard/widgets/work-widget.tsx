'use client';

/**
 * What JARVIS is doing right now. Everything live here comes from the LiveKit
 * room: the agent's state and the tool calls it reports. Background tasks
 * exist on the backend but have no endpoint yet, and the widget says so.
 */
import { useEffect, useState } from 'react';
import { ConnectionState } from 'livekit-client';
import { useSessionContext, useVoiceAssistant } from '@livekit/components-react';
import type { CapabilitiesState } from '@/hooks/useDashboardCapabilities';
import { useToolActivity } from '@/hooks/useToolActivity';
import { type VoiceTone, voiceStatus } from '@/lib/dashboard/activity';
import { cn } from '@/lib/utils';
import { WidgetBlocked, WidgetLoading, WidgetSignIn } from '../widget-notice';

const TONE_DOT: Record<VoiceTone, string> = {
  idle: 'bg-muted-foreground/50',
  busy: 'bg-amber-500 animate-pulse',
  live: 'bg-green-500',
  error: 'bg-destructive',
};

function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function formatClock(at: number): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(at);
}

/** When the current call started, for as long as it lasts. */
function useCallStartedAt(isConnected: boolean): number | null {
  const [startedAt, setStartedAt] = useState<number | null>(null);
  useEffect(() => {
    setStartedAt(isConnected ? Date.now() : null);
  }, [isConnected]);
  return startedAt;
}

interface WorkWidgetProps {
  capabilities: CapabilitiesState & { reload: () => void };
  passwordLogin: boolean;
  now: Date | null;
}

export function WorkWidget({ capabilities, passwordLogin, now }: WorkWidgetProps) {
  const session = useSessionContext();
  const { state: agentState } = useVoiceAssistant();
  const activity = useToolActivity();
  const startedAt = useCallStartedAt(session.isConnected);

  const status = voiceStatus({
    isConnected: session.isConnected,
    connecting: session.connectionState === ConnectionState.Connecting,
    agentState,
  });

  return (
    <div className="space-y-4">
      <section aria-label="Voice session" className="space-y-1">
        <p className="flex items-center gap-2 text-sm font-medium">
          <span aria-hidden className={cn('size-2 shrink-0 rounded-full', TONE_DOT[status.tone])} />
          {status.label}
        </p>
        {session.isConnected && startedAt !== null && now && (
          <p className="text-muted-foreground text-xs tabular-nums">
            On the call for {formatElapsed(now.getTime() - startedAt)}
          </p>
        )}
      </section>

      <section aria-label="Tool calls this session">
        <p className="text-muted-foreground mb-1.5 text-xs font-medium tracking-wider uppercase">
          Tool calls
        </p>
        {activity.length === 0 ? (
          <p className="text-muted-foreground text-sm">
            No tools have run yet. They appear here as JARVIS uses them.
          </p>
        ) : (
          <ol className="space-y-1.5">
            {activity.map((entry) => (
              <li key={entry.id} className="flex items-baseline justify-between gap-3 text-sm">
                <span className="min-w-0 truncate font-mono text-xs">
                  {entry.tools.map((tool) => tool.replace(/_/g, ' ')).join(', ')}
                </span>
                <time
                  dateTime={new Date(entry.at).toISOString()}
                  className="text-muted-foreground shrink-0 text-xs tabular-nums"
                >
                  {formatClock(entry.at)}
                </time>
              </li>
            ))}
          </ol>
        )}
      </section>

      <section aria-label="Background tasks">
        {capabilities.status === 'loading' ? (
          <WidgetLoading label="Checking background work…" />
        ) : capabilities.status === 'unauthorized' ? (
          <WidgetSignIn passwordLogin={passwordLogin} what="your background work" />
        ) : (
          <WidgetBlocked
            title="Background tasks"
            detail="Long-running work JARVIS queues for you is tracked on the backend but not exposed to the dashboard yet."
            endpoint={
              capabilities.status === 'ready' ? capabilities.data.work.tasksEndpoint : 'GET /tasks'
            }
          />
        )}
      </section>
    </div>
  );
}
