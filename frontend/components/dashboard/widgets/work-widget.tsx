'use client';

/** Durable work owned by the signed-in user, alongside current voice activity. */
import { useEffect, useState } from 'react';
import { ConnectionState } from 'livekit-client';
import { useSessionContext, useVoiceAssistant } from '@livekit/components-react';
import type { FeedController } from '@/hooks/useDashboardFeed';
import { useToolActivity } from '@/hooks/useToolActivity';
import { type VoiceTone, voiceStatus } from '@/lib/dashboard/activity';
import { formatDayOrTime } from '@/lib/dashboard/provider-data';
import { WORK_STATUS, type WorkFeed } from '@/lib/dashboard/work';
import { cn } from '@/lib/utils';
import { FeedFreshness } from '../feed-freshness';
import { WidgetEmpty, WidgetError, WidgetLoading, WidgetSignIn } from '../widget-notice';

const TONE_DOT: Record<VoiceTone, string> = {
  idle: 'bg-muted-foreground/50',
  busy: 'bg-amber-500 animate-pulse',
  live: 'bg-cyan-500',
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
  feed: FeedController<WorkFeed>;
  passwordLogin: boolean;
  now: Date | null;
}

export function WorkWidget({ feed, passwordLogin, now }: WorkWidgetProps) {
  const activeCount =
    feed.status === 'ready'
      ? feed.data.items.filter((item) => item.status === 'queued' || item.status === 'running')
          .length
      : 0;
  const session = useSessionContext();
  const { state: agentState } = useVoiceAssistant();
  const activity = useToolActivity();
  const startedAt = useCallStartedAt(session.isConnected);

  const status = voiceStatus({
    isConnected: session.isConnected,
    connectionState: session.connectionState,
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
            No tools have run yet. They appear here as FRIDAY uses them.
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

      <section aria-label="Background tasks" className="space-y-3">
        <FeedFreshness feed={feed} now={now} />
        {feed.status === 'loading' ? (
          <WidgetLoading label="Loading background work…" />
        ) : feed.status === 'unauthorized' ? (
          <WidgetSignIn passwordLogin={passwordLogin} what="your background work" />
        ) : feed.status === 'unconfigured' ? (
          <WidgetEmpty
            title="Personal access is not configured"
            detail="Background work is unavailable until personal access is configured."
          />
        ) : feed.status === 'error' ? (
          <WidgetError title="Could not load background work" onRetry={feed.reload} />
        ) : (
          <>
            {activeCount > 0 && (
              <p role="status" className="text-sm font-medium">
                FRIDAY is working in the background · {activeCount}{' '}
                {activeCount === 1 ? 'task' : 'tasks'}
              </p>
            )}
            {feed.data.items.length === 0 ? (
              <WidgetEmpty title="No background work yet" />
            ) : (
              <ol className="space-y-3">
                {feed.data.items.map((item, index) => {
                  const status = WORK_STATUS[item.status];
                  const updated = new Date(item.updated_at * 1000).toISOString();
                  return (
                    <li key={index} className="space-y-1 text-sm">
                      <p className="font-medium break-words">{item.title}</p>
                      <p>{status.label}</p>
                      {status.detail && (
                        <p className="text-muted-foreground text-xs">{status.detail}</p>
                      )}
                      <p className="text-muted-foreground text-xs">
                        Updated{' '}
                        <time
                          dateTime={updated}
                          title={new Date(updated).toLocaleString()}
                          aria-label={new Date(updated).toLocaleString()}
                        >
                          {now ? formatDayOrTime(updated, now) : new Date(updated).toLocaleString()}
                        </time>
                      </p>
                    </li>
                  );
                })}
              </ol>
            )}
          </>
        )}
      </section>
    </div>
  );
}
