'use client';

/**
 * The persistent, signed-in workspace: the Monitor dashboard with voice docked
 * beneath it. Nothing here is gated on a LiveKit call; starting or ending one
 * only changes the dock.
 */
import { useRef, useState } from 'react';
import {
  BellSimple,
  CalendarBlank,
  CloudSun,
  EnvelopeSimple,
  Pulse,
} from '@phosphor-icons/react/dist/ssr';
import type { AppConfig } from '@/app-config';
import { SettingsPanel } from '@/components/settings/settings-panel';
import { useDashboardCapabilities } from '@/hooks/useDashboardCapabilities';
import { useDashboardFeed } from '@/hooks/useDashboardFeed';
import { useDashboardLayout } from '@/hooks/useDashboardLayout';
import { useMe } from '@/hooks/useMe';
import { useNow } from '@/hooks/useNow';
import { useWeather } from '@/hooks/useWeather';
import { type WidgetId, layoutScopeFor } from '@/lib/dashboard/layout';
import { browserCalendarFeed, browserInboxFeed } from '@/lib/dashboard/provider-data';
import { browserReminders } from '@/lib/dashboard/reminders';
import type { HandSurfaceController } from '@/lib/hands/surface';
import { HandsDock } from './hands-dock';
import { MonitorDashboard } from './monitor-dashboard';
import { VoiceDock } from './voice-dock';
import { CalendarWidget } from './widgets/calendar-widget';
import { InboxWidget } from './widgets/inbox-widget';
import { RemindersWidget } from './widgets/reminders-widget';
import { WeatherWidget } from './widgets/weather-widget';
import { WorkWidget } from './widgets/work-widget';
import { WorkspaceHeader } from './workspace-header';

interface WorkspaceProps {
  appConfig: AppConfig;
}

/** First tab stop of the page: jump straight to the docked voice controls. */
export function SkipToVoiceLink() {
  return (
    <a
      href="#voice-dock"
      className="bg-background text-foreground sr-only rounded-full border px-4 py-2 text-sm focus:not-sr-only focus:fixed focus:top-4 focus:left-1/2 focus:z-50 focus:-translate-x-1/2"
    >
      Skip to voice controls
    </a>
  );
}

/** One account, two accounts; three unread. */
function countLabel(count: number, noun: string, plural = noun + 's'): string {
  return count + ' ' + (count === 1 ? noun : plural);
}

export function Workspace({ appConfig }: WorkspaceProps) {
  const me = useMe();
  const now = useNow();
  const capabilities = useDashboardCapabilities();
  const calendar = useDashboardFeed('/api/dashboard/calendar', browserCalendarFeed);
  const inbox = useDashboardFeed('/api/dashboard/inbox', browserInboxFeed);
  const reminders = useDashboardFeed('/api/dashboard/reminders', browserReminders);
  const weather = useWeather();
  const [settingsOpen, setSettingsOpen] = useState(false);
  // Hand control is opt-in for this visit only; the camera is never opened unasked.
  const [handsEnabled, setHandsEnabled] = useState(false);
  const handController = useRef<HandSurfaceController | null>(null);

  // Undefined while identity resolves: the layout waits rather than flashing
  // a default and then someone's saved arrangement.
  const scope =
    me.status === 'loading' ? undefined : me.status === 'ready' ? layoutScopeFor(me.me) : null;
  const layout = useDashboardLayout(scope);

  const signedIn = me.status === 'ready' && me.me.authenticated ? me.me : null;
  const displayName = signedIn?.user?.displayName || undefined;
  const passwordLogin = me.status === 'ready' && me.me.passwordLogin;
  const needsSignIn = me.status === 'ready' && me.me.configured && !me.me.authenticated;

  const openSettings = () => setSettingsOpen(true);

  const renderWidget = (id: WidgetId) => {
    switch (id) {
      case 'weather':
        return {
          icon: <CloudSun className="size-4" weight="bold" />,
          meta:
            weather.status === 'ready' ? (weather.data.location?.label ?? undefined) : undefined,
          body: (
            <WeatherWidget
              weather={weather}
              passwordLogin={passwordLogin}
              now={now}
              onOpenSettings={openSettings}
            />
          ),
        };
      case 'calendar':
        return {
          icon: <CalendarBlank className="size-4" weight="bold" />,
          meta:
            calendar.status === 'ready'
              ? countLabel(calendar.data.accounts.length, 'account')
              : undefined,
          body: (
            <CalendarWidget
              capabilities={capabilities}
              feed={calendar}
              passwordLogin={passwordLogin}
              today={now}
              onOpenSettings={openSettings}
            />
          ),
        };
      case 'inbox':
        return {
          icon: <EnvelopeSimple className="size-4" weight="bold" />,
          meta:
            inbox.status === 'ready'
              ? countLabel(inbox.data.unreadCount, 'unread', 'unread')
              : undefined,
          body: (
            <InboxWidget
              feed={inbox}
              passwordLogin={passwordLogin}
              now={now}
              onOpenSettings={openSettings}
            />
          ),
        };
      case 'reminders':
        return {
          icon: <BellSimple className="size-4" weight="bold" />,
          meta:
            reminders.status === 'ready'
              ? countLabel(reminders.data.reminders.length, 'reminder')
              : undefined,
          body: <RemindersWidget feed={reminders} passwordLogin={passwordLogin} now={now} />,
        };
      case 'work':
        return {
          icon: <Pulse className="size-4" weight="bold" />,
          meta: 'Live',
          body: <WorkWidget capabilities={capabilities} passwordLogin={passwordLogin} now={now} />,
        };
    }
  };

  return (
    <>
      <main className="mx-auto w-full max-w-7xl px-4 pt-16 pb-48 md:px-8 md:pt-24 md:pb-56">
        <WorkspaceHeader
          now={now}
          displayName={displayName}
          handsEnabled={handsEnabled}
          onHandsChange={setHandsEnabled}
          onResetLayout={layout.reset}
          onOpenSettings={openSettings}
        />

        {needsSignIn && (
          <p className="bg-muted/60 text-muted-foreground mb-4 rounded-lg px-3 py-2 text-sm">
            You are not signed in, so this workspace shows nothing personal.{' '}
            {passwordLogin && (
              <a href="/login" className="text-foreground underline underline-offset-4">
                Sign in
              </a>
            )}
          </p>
        )}
        {me.status === 'error' && (
          <p className="bg-muted/60 text-muted-foreground mb-4 rounded-lg px-3 py-2 text-sm">
            Could not tell who is signed in; layout changes will not be saved for this visit.
          </p>
        )}

        {handsEnabled && (
          <HandsDock controller={handController} onClose={() => setHandsEnabled(false)} />
        )}

        {layout.ready ? (
          <MonitorDashboard
            layout={layout.layout}
            onPreview={layout.preview}
            onCommit={layout.commit}
            renderWidget={renderWidget}
            handController={handController}
          />
        ) : (
          <p role="status" className="text-muted-foreground text-sm">
            Loading your workspace…
          </p>
        )}
      </main>

      <VoiceDock appConfig={appConfig} />
      <SettingsPanel isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </>
  );
}
