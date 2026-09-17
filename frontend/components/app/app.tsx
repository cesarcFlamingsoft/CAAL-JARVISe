'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { TokenSource } from 'livekit-client';
import { SessionProvider, StartAudio, useSession } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { AgentAudioRenderer } from '@/components/app/agent-audio-renderer';
import { DevicePresence } from '@/components/app/device-presence';
import { WakeWordProvider } from '@/components/app/wake-word-provider';
import { SkipToVoiceLink, Workspace } from '@/components/dashboard/workspace';
import { Toaster } from '@/components/livekit/toaster';
import { SetupWizard } from '@/components/setup';
// import { useAgentErrors } from '@/hooks/useAgentErrors';
import { useConnectionErrors } from '@/hooks/useConnectionErrors';
import { useDebugMode } from '@/hooks/useDebug';
import { ORDINARY_SESSION_HREF, companySessionRequested } from '@/lib/company/session-mode';
import { getSandboxTokenSource } from '@/lib/utils';

// Porcupine access key from environment
const PORCUPINE_ACCESS_KEY = process.env.NEXT_PUBLIC_PORCUPINE_ACCESS_KEY ?? '';

const IN_DEVELOPMENT = process.env.NODE_ENV !== 'production';

// Generate unique session ID for each conversation
// This ensures each device/tab gets its own isolated conversation
function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function AppSetup() {
  useDebugMode({ enabled: IN_DEVELOPMENT });
  // useAgentErrors(); // Disabled for multi-device support - timeout breaks second device
  useConnectionErrors(); // Show MCP connection errors from agent

  return null;
}

/**
 * What a company session looks like, so that it is never a surprise.
 *
 * A company session is more private and much less capable -- it can answer
 * questions about the library and do nothing else -- and both halves are
 * stated here rather than discovered. It is shown for the whole session,
 * because the session cannot stop being one.
 *
 * The wording must keep matching `caal.company_privacy.PRIVATE_SESSION_TOOLS`
 * and `PRIVATE_SESSION_BLOCKED_ROUTES`, which are what actually enforce it.
 */
function CompanySessionBanner() {
  return (
    <div
      role="status"
      className="border-input bg-muted/60 text-foreground sticky top-0 z-50 flex flex-wrap items-center justify-between gap-3 border-b px-4 py-2 text-sm"
    >
      <p className="max-w-3xl">
        <strong className="font-semibold">Company session — questions only.</strong> This
        conversation stays on this machine: the local model answers it, and the only thing I can do
        in it is look things up in the company library. No reminders, no alarms, no notes to memory,
        no mail or calendar, no Home Assistant, no web, no background work — nothing that sends
        anything, and nothing that outlives this session. If the local model cannot answer, I will
        say so rather than ask anything else. Everything else is back to normal in an ordinary
        session.
      </p>
      <a className="border-input hover:bg-accent shrink-0 rounded-md border px-3 py-1.5" href={ORDINARY_SESSION_HREF}>
        Leave company session
      </a>
    </div>
  );
}

interface AppProps {
  appConfig: AppConfig;
}

export function App({ appConfig }: AppProps) {
  const [setupCompleted, setSetupCompleted] = useState<boolean | null>(null);

  // Check setup status on mount
  useEffect(() => {
    const checkSetup = async () => {
      try {
        const res = await fetch('/api/setup/status');
        const data = await res.json();
        setSetupCompleted(data.completed ?? false);
      } catch {
        // If we can't reach the backend, assume setup not completed
        setSetupCompleted(false);
      }
    };
    checkSetup();
  }, []);

  const handleSetupComplete = () => {
    setSetupCompleted(true);
    // Reload the page to pick up new settings
    window.location.reload();
  };

  // Generate unique session ID once when component mounts
  const sessionId = useMemo(() => generateSessionId(), []);

  // Whether this page load is asking for a company-private session. Read once,
  // at mount, from the URL the user navigated to -- the kind of a session is
  // fixed when its room is created, so entering and leaving are both fresh
  // loads. The request is only granted if the BFF signs it for a verified user
  // and the voice worker matches that user to the company library's owner.
  const companyPrivate = useMemo(
    () =>
      typeof window === 'undefined' ? false : companySessionRequested(window.location.search),
    []
  );

  const tokenSource = useMemo(() => {
    if (typeof process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT === 'string') {
      return getSandboxTokenSource(appConfig);
    }

    // Create custom token source that includes client_id in the request
    return TokenSource.custom(async (options) => {
      const response = await fetch('/api/connection-details', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...options,
          client_id: sessionId,
          agentName: appConfig.agentName,
          company_private: companyPrivate,
        }),
      });
      return await response.json();
    });
  }, [appConfig, sessionId, companyPrivate]);

  const session = useSession(tokenSource);

  // Clean up session on page unload to prevent orphaned agent jobs
  useEffect(() => {
    const handleUnload = () => {
      session.end();
    };
    window.addEventListener('beforeunload', handleUnload);
    window.addEventListener('pagehide', handleUnload);
    return () => {
      window.removeEventListener('beforeunload', handleUnload);
      window.removeEventListener('pagehide', handleUnload);
    };
  }, [session]);

  // Handle wake word detection - unmute mic and call backend to trigger greeting
  const handleWakeWordDetected = useCallback(async () => {
    console.log('[App] Wake word detected');

    // Unmute microphone
    const micTrack = Array.from(
      session.room?.localParticipant?.audioTrackPublications.values() || []
    ).find((pub) => pub.source === 'microphone')?.track;

    if (micTrack && micTrack.isMuted) {
      console.log('[App] Unmuting microphone');
      await micTrack.unmute();
    }

    // Call backend to trigger greeting
    try {
      const roomName = session.room?.name || 'voice_assistant_room';
      const response = await fetch('/api/wake', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ room_name: roomName }),
      });
      if (!response.ok) {
        console.error('[App] Wake endpoint failed:', response.status);
      }
    } catch (error) {
      console.error('[App] Wake endpoint error:', error);
    }
  }, [session]);

  // Show loading state while checking setup status
  if (setupCompleted === null) {
    return (
      <main className="grid h-svh grid-cols-1 place-content-center">
        <div className="text-muted-foreground text-center">Loading...</div>
      </main>
    );
  }

  // Show setup wizard if not completed
  if (!setupCompleted) {
    return <SetupWizard onComplete={handleSetupComplete} />;
  }

  return (
    <SessionProvider session={session}>
      <WakeWordProvider
        accessKey={PORCUPINE_ACCESS_KEY}
        keywordPath="/hey_cal.ppn"
        onWakeWordDetected={handleWakeWordDetected}
        defaultEnabled={false}
      >
        <AppSetup />
        {companyPrivate && <CompanySessionBanner />}
        <SkipToVoiceLink />
        <DevicePresence />
        {/* The workspace is the application session; a voice call is docked inside it. */}
        <Workspace appConfig={appConfig} />
        <StartAudio label="Start Audio" />
        <AgentAudioRenderer />
        <Toaster />
      </WakeWordProvider>
    </SessionProvider>
  );
}
