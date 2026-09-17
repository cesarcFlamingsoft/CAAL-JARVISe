'use client';

/**
 * The persistent, signed-in workspace: the cinematic dashboard with a voice core
 * beside it. Nothing here is gated on a LiveKit call; starting or ending one
 * only changes the dock.
 */
import { useEffect, useRef, useState } from 'react';
import {
  ArrowRight,
  BellSimple,
  CalendarBlank,
  CloudSun,
  EnvelopeSimple,
  LockKey,
  Pulse,
} from '@phosphor-icons/react/dist/ssr';
import type { AppConfig } from '@/app-config';
import { SettingsPanel } from '@/components/settings/settings-panel';
import { useCamera } from '@/hooks/useCamera';
import { useDashboardCapabilities } from '@/hooks/useDashboardCapabilities';
import { useDashboardFeed } from '@/hooks/useDashboardFeed';
import { useDashboardLayout } from '@/hooks/useDashboardLayout';
import { type MeState, useMe } from '@/hooks/useMe';
import { useNow } from '@/hooks/useNow';
import { useScheduledUpdates } from '@/hooks/useScheduledUpdates';
import { useWeather } from '@/hooks/useWeather';
import { companySessionRequested } from '@/lib/company/session-mode';
import { type WidgetId, layoutScopeFor } from '@/lib/dashboard/layout';
import { browserCalendarFeed, browserInboxFeed } from '@/lib/dashboard/provider-data';
import { browserReminders } from '@/lib/dashboard/reminders';
import { browserWork } from '@/lib/dashboard/work';
import { workspaceMode } from '@/lib/dashboard/workspace-access';
import type { HandSurfaceController } from '@/lib/hands/surface';
import type { VisionAnalysis } from '@/lib/visual/bridge';
import './cinematic.css';
import { FeedFreshness } from './feed-freshness';
import { HandsDock } from './hands-dock';
import { MonitorDashboard } from './monitor-dashboard';
import { VisualAnalyzer } from './visual-analyzer';
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
  const mode = workspaceMode(me);

  if (me.status === 'loading') {
    return (
      <main className="cinematic-workspace">
        <WorkspaceHeader
          now={now}
          showWorkspaceControls={false}
          handsEnabled={false}
          onHandsChange={() => {}}
          onResetLayout={() => {}}
          onOpenSettings={() => {}}
        />
        <p role="status" className="text-muted-foreground text-sm">
          Loading your workspace…
        </p>
      </main>
    );
  }

  if (mode === 'signed-out' && me.status === 'ready') {
    return <SignedOutAccess now={now} passwordLogin={me.me.passwordLogin} />;
  }

  return <PersonalWorkspace appConfig={appConfig} me={me} now={now} />;
}

function SignedOutAccess({ now, passwordLogin }: { now: Date | null; passwordLogin: boolean }) {
  return (
    <main className="cinematic-workspace">
      <WorkspaceHeader
        now={now}
        showWorkspaceControls={false}
        handsEnabled={false}
        onHandsChange={() => {}}
        onResetLayout={() => {}}
        onOpenSettings={() => {}}
      />

      <section className="signed-out-access" aria-labelledby="signed-out-title">
        <div className="signed-out-access-mark" aria-hidden="true">
          <span />
          <LockKey weight="light" />
          <span />
        </div>
        <p className="signed-out-access-kicker">FRIDAY / ACCESS REQUIRED</p>
        <h2 id="signed-out-title">Your workspace is private</h2>
        <p className="signed-out-access-copy">
          Sign in to open your personal dashboard, connected accounts, and voice controls.
        </p>
        {passwordLogin ? (
          <a href="/login" className="signed-out-access-action">
            Sign in
            <ArrowRight aria-hidden weight="bold" />
          </a>
        ) : (
          <p className="signed-out-access-note">
            Use your configured local access method to continue.
          </p>
        )}
      </section>

      <div className="workspace-footnote">
        <span>FRIDAY / PERSONAL OPERATING SURFACE</span>
        <span>Personal feeds remain unavailable while signed out</span>
      </div>
    </main>
  );
}

function PersonalWorkspace({
  appConfig,
  me,
  now,
}: WorkspaceProps & { me: Exclude<MeState, { status: 'loading' }>; now: Date | null }) {
  const capabilities = useDashboardCapabilities();
  const calendar = useDashboardFeed('/api/dashboard/calendar', browserCalendarFeed);
  const inbox = useDashboardFeed('/api/dashboard/inbox', browserInboxFeed);
  const reminders = useDashboardFeed('/api/dashboard/reminders', browserReminders);
  const work = useDashboardFeed('/api/dashboard/work', browserWork);
  // A reminder or alarm FRIDAY just set reaches this page at once, instead of
  // waiting out the polling interval. Only the scheduled feed reloads.
  useScheduledUpdates();
  const weather = useWeather();
  const [settingsOpen, setSettingsOpen] = useState(false);
  // Hand control is opt-in for this visit only; the camera is never opened unasked.
  const [handsEnabled, setHandsEnabled] = useState(false);
  const [visionOpen, setVisionOpen] = useState(false);
  const visionAnalysis = useRef<VisionAnalysis | null>(null);
  const camera = useCamera();
  const companyPrivate =
    typeof window !== 'undefined' && companySessionRequested(window.location.search);
  const handController = useRef<HandSurfaceController | null>(null);

  const scope = me.status === 'ready' ? layoutScopeFor(me.me) : null;
  const layout = useDashboardLayout(scope);

  const signedIn = me.status === 'ready' && me.me.authenticated ? me.me : null;
  const visionAllowed = Boolean(signedIn) && !companyPrivate;
  const changeHands = (enabled: boolean) => {
    setHandsEnabled(enabled);
    if (enabled) void camera.acquire('hands');
    else camera.release('hands');
  };
  const changeVision = (open: boolean) => {
    if (open && !visionAllowed) return;
    if (!open) visionAnalysis.current = null;
    setVisionOpen(open);
    if (open) void camera.acquire('vision');
    else camera.release('vision');
  };
  const { release } = camera;
  useEffect(() => {
    if (!visionAllowed) {
      setVisionOpen(false);
      release('vision');
    }
  }, [visionAllowed, release]);
  const displayName = signedIn?.user?.displayName || undefined;
  const passwordLogin = me.status === 'ready' && me.me.passwordLogin;
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
            <>
              <FeedFreshness feed={calendar} now={now} />
              <CalendarWidget
                capabilities={capabilities}
                feed={calendar}
                passwordLogin={passwordLogin}
                today={now}
                onOpenSettings={openSettings}
              />
            </>
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
            <>
              <FeedFreshness feed={inbox} now={now} />
              <InboxWidget
                feed={inbox}
                passwordLogin={passwordLogin}
                now={now}
                onOpenSettings={openSettings}
              />
            </>
          ),
        };
      case 'reminders':
        return {
          icon: <BellSimple className="size-4" weight="bold" />,
          meta:
            reminders.status === 'ready'
              ? countLabel(reminders.data.reminders.length + reminders.data.alarms.length, 'item')
              : undefined,
          body: (
            <>
              <FeedFreshness feed={reminders} now={now} />
              <RemindersWidget feed={reminders} passwordLogin={passwordLogin} now={now} />
            </>
          ),
        };
      case 'work':
        return {
          icon: <Pulse className="size-4" weight="bold" />,
          meta: 'Background work and session activity',
          body: companyPrivate ? (
            <p className="text-muted-foreground text-sm">
              Background work is available in your personal workspace.
            </p>
          ) : (
            <WorkWidget feed={work} passwordLogin={passwordLogin} now={now} />
          ),
        };
    }
  };

  return (
    <>
      <SkipToVoiceLink />
      <main className="cinematic-workspace">
        <WorkspaceHeader
          now={now}
          displayName={displayName}
          handsEnabled={handsEnabled}
          onHandsChange={changeHands}
          visionOpen={visionOpen && visionAllowed}
          visionAllowed={visionAllowed}
          onVisionChange={changeVision}
          onResetLayout={layout.reset}
          onOpenSettings={openSettings}
        />

        {me.status === 'error' && (
          <p className="bg-muted/60 text-muted-foreground mb-4 rounded-lg px-3 py-2 text-sm">
            Could not tell who is signed in; layout changes will not be saved for this visit.
          </p>
        )}

        {handsEnabled && (
          <HandsDock
            controller={handController}
            camera={camera}
            visionOpen={visionOpen}
            onClose={() => changeHands(false)}
          />
        )}

        {visionOpen && visionAllowed && (
          <VisualAnalyzer
            analysisRef={visionAnalysis}
            userId={signedIn?.user?.userId ?? ''}
            stream={camera.stream}
            cameraStatus={camera.status}
            cameraDetail={camera.detail}
            companyPrivate={companyPrivate}
            onRetry={() => void camera.acquire('vision')}
            onClose={() => changeVision(false)}
          />
        )}

        <div className="operating-surface">
          <VoiceDock
            appConfig={appConfig}
            visionAnalysis={visionAnalysis}
            userId={signedIn?.user?.userId ?? null}
            companyPrivate={companyPrivate}
            visionOpen={visionOpen}
            cameraLive={camera.status === 'live'}
            stream={camera.stream}
          />
          <div className="workspace-data">
            <div className="surface-heading">
              <span>01 / PERSONAL WORKSPACE</span>
              <span>ACCOUNT-SCOPED FEEDS</span>
            </div>
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
          </div>
        </div>
        <div className="workspace-footnote">
          <span>FRIDAY / PERSONAL OPERATING SURFACE</span>
          <span>Workspace available independently of voice</span>
        </div>
      </main>
      <SettingsPanel isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </>
  );
}
