'use client';

import { ArrowCounterClockwise, Gear } from '@phosphor-icons/react/dist/ssr';
import { Button } from '@/components/livekit/button';

function greetingFor(hour: number): string {
  if (hour < 5) return 'Good evening';
  if (hour < 12) return 'Good morning';
  if (hour < 18) return 'Good afternoon';
  return 'Good evening';
}

interface WorkspaceHeaderProps {
  now: Date | null;
  displayName?: string;
  onResetLayout: () => void;
  onOpenSettings: () => void;
}

export function WorkspaceHeader({
  now,
  displayName,
  onResetLayout,
  onOpenSettings,
}: WorkspaceHeaderProps) {
  const greeting = now ? greetingFor(now.getHours()) : 'Welcome';
  const dateLabel = now
    ? new Intl.DateTimeFormat(undefined, {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      }).format(now)
    : '';
  const timeLabel = now
    ? new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' }).format(now)
    : '--:--';

  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        <p className="text-muted-foreground font-mono text-xs font-medium tracking-wider uppercase">
          Monitor
        </p>
        <h1 className="truncate text-2xl font-semibold tracking-tight md:text-3xl">
          {greeting}
          {displayName ? `, ${displayName}` : ''}
        </h1>
        <p className="text-muted-foreground text-sm" aria-live="off">
          {dateLabel}
        </p>
      </div>

      <div className="flex items-center gap-2">
        <time
          dateTime={now?.toISOString()}
          className="font-mono text-2xl font-medium tabular-nums md:text-3xl"
          aria-label={now ? `Current time ${timeLabel}` : undefined}
        >
          {timeLabel}
        </time>
        <Button variant="ghost" size="sm" onClick={onResetLayout} className="hidden md:inline-flex">
          <ArrowCounterClockwise aria-hidden weight="bold" />
          Reset layout
        </Button>
        <Button variant="ghost" size="icon" aria-label="Settings" onClick={onOpenSettings}>
          <Gear className="size-5" weight="bold" />
        </Button>
      </div>
    </header>
  );
}
