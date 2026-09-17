'use client';

import { ArrowCounterClockwise, Eye, Gear, Hand } from '@phosphor-icons/react/dist/ssr';
import { Button } from '@/components/livekit/button';
import { Toggle } from '@/components/livekit/toggle';

function greetingFor(hour: number): string {
  if (hour < 5) return 'Good evening';
  if (hour < 12) return 'Good morning';
  if (hour < 18) return 'Good afternoon';
  return 'Good evening';
}

interface WorkspaceHeaderProps {
  visionOpen?: boolean;
  visionAllowed?: boolean;
  onVisionChange?: (open: boolean) => void;
  now: Date | null;
  displayName?: string;
  /** Access and loading surfaces retain the identity header without personal controls. */
  showWorkspaceControls?: boolean;
  /** Hand control is opt-in: off on every load until this is pressed. */
  handsEnabled: boolean;
  onHandsChange: (enabled: boolean) => void;
  onResetLayout: () => void;
  onOpenSettings: () => void;
}

export function WorkspaceHeader({
  visionOpen = false,
  visionAllowed = false,
  onVisionChange,
  now,
  displayName,
  showWorkspaceControls = true,
  handsEnabled,
  onHandsChange,
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
    <header className="workspace-heading mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        <p className="text-muted-foreground font-mono text-xs font-medium tracking-wider uppercase">
          FRIDAY / COMMAND CENTER
        </p>
        <h1 className="truncate text-2xl font-semibold tracking-tight md:text-3xl">
          {greeting}
          {displayName ? `, ${displayName}` : ''}
        </h1>
        <p className="text-muted-foreground text-sm" aria-live="off">
          {dateLabel}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <time
          dateTime={now?.toISOString()}
          className="font-mono text-2xl font-medium tabular-nums md:text-3xl"
          aria-label={now ? `Current time ${timeLabel}` : undefined}
        >
          {timeLabel}
        </time>
        {showWorkspaceControls && (
          <>
            <Toggle
              variant="outline"
              size="sm"
              pressed={handsEnabled}
              onPressedChange={onHandsChange}
              aria-label="Hand control"
              title="Hand control uses your camera, locally in this browser"
              className="inline-flex font-mono text-xs tracking-wider uppercase"
            >
              <Hand aria-hidden weight="bold" />
              Hands
            </Toggle>
            <Button
              variant="outline"
              size="sm"
              aria-label="Vision"
              aria-pressed={visionOpen}
              disabled={!visionAllowed}
              title={
                visionAllowed
                  ? 'Open local camera preview'
                  : 'Vision requires sign-in and Personal Mode'
              }
              onClick={(event) => {
                if (event.isTrusted) onVisionChange?.(!visionOpen);
              }}
              className="inline-flex font-mono text-xs tracking-wider uppercase"
            >
              <Eye aria-hidden weight="bold" />
              Vision
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={onResetLayout}
              className="hidden md:inline-flex"
            >
              <ArrowCounterClockwise aria-hidden weight="bold" />
              Reset layout
            </Button>
            <Button variant="ghost" size="icon" aria-label="Settings" onClick={onOpenSettings}>
              <Gear className="size-5" weight="bold" />
            </Button>
          </>
        )}
      </div>
    </header>
  );
}
