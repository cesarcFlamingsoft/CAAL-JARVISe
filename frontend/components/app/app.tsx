'use client';

import { useCallback, useEffect, useMemo } from 'react';
import { TokenSource } from 'livekit-client';
import { SessionProvider, StartAudio, useSession } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { AgentAudioRenderer } from '@/components/app/agent-audio-renderer';
import { ViewController } from '@/components/app/view-controller';
import { WakeWordProvider } from '@/components/app/wake-word-provider';
import { Toaster } from '@/components/livekit/toaster';
// import { useAgentErrors } from '@/hooks/useAgentErrors';
import { useDebugMode } from '@/hooks/useDebug';
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

  return null;
}

interface AppProps {
  appConfig: AppConfig;
}

export function App({ appConfig }: AppProps) {
  // Generate unique session ID once when component mounts
  const sessionId = useMemo(() => generateSessionId(), []);
  
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
        }),
      });
      return await response.json();
    });
  }, [appConfig, sessionId]);

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

  return (
    <SessionProvider session={session}>
      <WakeWordProvider
        accessKey={PORCUPINE_ACCESS_KEY}
        keywordPath="/hey_cal.ppn"
        onWakeWordDetected={handleWakeWordDetected}
        defaultEnabled={false}
      >
        <AppSetup />
        <main className="grid h-svh grid-cols-1 place-content-center">
          <ViewController appConfig={appConfig} />
        </main>
        <StartAudio label="Start Audio" />
        <AgentAudioRenderer />
        <Toaster />
      </WakeWordProvider>
    </SessionProvider>
  );
}
