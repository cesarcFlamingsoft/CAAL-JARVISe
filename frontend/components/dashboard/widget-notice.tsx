'use client';

/**
 * The honest states a widget can be in when it has nothing live to show.
 * Nothing here ever renders a placeholder number.
 */
import type { ReactNode } from 'react';
import { PlugsConnected, SignIn, Warning } from '@phosphor-icons/react/dist/ssr';
import { Button } from '@/components/livekit/button';
import { cn } from '@/lib/utils';

interface NoticeProps {
  title: string;
  detail?: string;
  action?: ReactNode;
  className?: string;
}

export function WidgetLoading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div role="status" aria-live="polite" className="space-y-2 py-1">
      <span className="sr-only">{label}</span>
      <div aria-hidden className="bg-muted h-3 w-2/3 animate-pulse rounded" />
      <div aria-hidden className="bg-muted h-3 w-1/2 animate-pulse rounded" />
      <div aria-hidden className="bg-muted h-3 w-3/5 animate-pulse rounded" />
    </div>
  );
}

/** Nothing is configured yet; explains what to do, never what the data would be. */
export function WidgetEmpty({ title, detail, action, className }: NoticeProps) {
  return (
    <div className={cn('flex flex-col items-start gap-2 py-1', className)}>
      <p className="text-sm font-medium">{title}</p>
      {detail && <p className="text-muted-foreground text-sm">{detail}</p>}
      {action}
    </div>
  );
}

interface BlockedProps extends NoticeProps {
  /** The backend endpoint this data is waiting on, e.g. `GET /calendar/events`. */
  endpoint: string;
}

/** Configured on the backend, but no HTTP endpoint exists yet to read it. */
export function WidgetBlocked({ title, detail, endpoint, className }: BlockedProps) {
  return (
    <div
      className={cn(
        'border-border/70 text-muted-foreground rounded-lg border border-dashed px-3 py-2.5',
        className
      )}
    >
      <p className="text-foreground flex items-center gap-1.5 text-sm font-medium">
        <PlugsConnected aria-hidden className="size-4 shrink-0" weight="bold" />
        {title}
      </p>
      {detail && <p className="mt-1 text-sm">{detail}</p>}
      <p className="mt-1.5 text-xs">
        Needs backend <code className="text-foreground/80 font-mono">{endpoint}</code>
      </p>
    </div>
  );
}

export function WidgetError({ title, detail, onRetry }: NoticeProps & { onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-start gap-2 py-1">
      <p className="text-destructive flex items-center gap-1.5 text-sm font-medium">
        <Warning aria-hidden className="size-4 shrink-0" weight="bold" />
        {title}
      </p>
      {detail && <p className="text-muted-foreground text-sm">{detail}</p>}
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

export function WidgetSignIn({ passwordLogin, what }: { passwordLogin: boolean; what: string }) {
  return (
    <div className="flex flex-col items-start gap-2 py-1">
      <p className="flex items-center gap-1.5 text-sm font-medium">
        <SignIn aria-hidden className="size-4 shrink-0" weight="bold" />
        Sign in to see {what}
      </p>
      <p className="text-muted-foreground text-sm">
        This workspace only shows what belongs to you.
      </p>
      {passwordLogin && (
        <Button asChild variant="outline" size="sm">
          <a href="/login">Sign in</a>
        </Button>
      )}
    </div>
  );
}
