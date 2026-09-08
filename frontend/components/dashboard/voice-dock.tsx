'use client';

/**
 * Voice as a docked capability. The dock is always present at the bottom of
 * the workspace: idle it offers to start a call, live it carries the existing
 * agent control bar, and the dashboard behind it never goes away when the
 * call ends.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { ConnectionState } from 'livekit-client';
import {
  useSessionContext,
  useSessionMessages,
  useVoiceAssistant,
} from '@livekit/components-react';
import { Waveform } from '@phosphor-icons/react/dist/ssr';
import type { AppConfig } from '@/app-config';
import { ChatTranscript } from '@/components/app/chat-transcript';
import { PreConnectMessage } from '@/components/app/preconnect-message';
import {
  AgentControlBar,
  type ControlBarControls,
} from '@/components/livekit/agent-control-bar/agent-control-bar';
import { Button } from '@/components/livekit/button';
import { ScrollArea } from '@/components/livekit/scroll-area/scroll-area';
import { type VoiceTone, voiceStatus } from '@/lib/dashboard/activity';
import { cn } from '@/lib/utils';
import { AgentTile } from './agent-tile';

const TONE_DOT: Record<VoiceTone, string> = {
  idle: 'bg-muted-foreground/50',
  busy: 'bg-amber-500 animate-pulse',
  live: 'bg-green-500',
  error: 'bg-destructive',
};

interface VoiceDockProps {
  appConfig: AppConfig;
}

export function VoiceDock({ appConfig }: VoiceDockProps) {
  const session = useSessionContext();
  const { state: agentState } = useVoiceAssistant();
  const { messages } = useSessionMessages(session);
  const [transcriptOpen, setTranscriptOpen] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const connecting = session.connectionState === ConnectionState.Connecting;
  const status = voiceStatus({ isConnected: session.isConnected, connecting, agentState });

  const controls: ControlBarControls = {
    leave: true,
    microphone: true,
    chat: appConfig.supportsChatInput,
    camera: appConfig.supportsVideoInput,
    screenShare: appConfig.supportsVideoInput,
  };

  const startCall = useCallback(async () => {
    setConnectionError(null);
    try {
      await session.start();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown connection error';
      console.error('[JARVIS voice connection]', error);
      setConnectionError(`Unable to start JARVIS: ${message}`);
    }
  }, [session]);

  // The transcript belongs to a call; when the call ends it folds away.
  useEffect(() => {
    if (!session.isConnected) setTranscriptOpen(false);
  }, [session.isConnected]);

  useEffect(() => {
    const last = messages.at(-1);
    if (scrollAreaRef.current && last?.from?.isLocal) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <>
      {session.isConnected && transcriptOpen && (
        <section
          aria-label="Transcript"
          className={cn(
            'bg-background/95 border-border fixed inset-x-3 bottom-44 z-30 flex max-h-[45svh] flex-col rounded-2xl border shadow-xl backdrop-blur',
            'md:inset-x-auto md:right-6 md:bottom-40 md:w-[min(28rem,calc(100vw-3rem))]'
          )}
        >
          <p className="text-muted-foreground border-b px-4 py-2 text-xs font-medium tracking-wider uppercase">
            Transcript
          </p>
          <ScrollArea ref={scrollAreaRef} className="min-h-0 flex-1 px-4 py-3">
            {messages.length === 0 ? (
              <p className="text-muted-foreground text-sm">Nothing has been said yet.</p>
            ) : (
              <ChatTranscript messages={messages} className="space-y-3" />
            )}
          </ScrollArea>
        </section>
      )}

      <aside
        id="voice-dock"
        aria-label="Voice assistant"
        className="fixed inset-x-3 bottom-3 z-40 md:inset-x-0 md:bottom-6 md:mx-auto md:max-w-2xl"
      >
        {session.isConnected && appConfig.isPreConnectBufferEnabled && (
          <PreConnectMessage messages={messages} className="pb-3" />
        )}

        <div className="bg-background/95 border-border rounded-[31px] border p-3 shadow-lg backdrop-blur">
          <div className="flex items-center gap-3 px-1 pb-2">
            {session.isConnected ? (
              <AgentTile />
            ) : (
              <span
                aria-hidden
                className="bg-muted text-muted-foreground inline-flex size-14 shrink-0 items-center justify-center rounded-xl"
              >
                <Waveform className="size-6" weight="bold" />
              </span>
            )}
            <div className="min-w-0 flex-1">
              <p className="flex items-center gap-2 text-sm font-medium">
                <span
                  aria-hidden
                  className={cn('size-2 shrink-0 rounded-full', TONE_DOT[status.tone])}
                />
                <span role="status" className="truncate">
                  {status.label}
                </span>
              </p>
              <p className="text-muted-foreground truncate text-xs">
                {session.isConnected
                  ? 'The dashboard stays put when the call ends.'
                  : 'Voice is docked here. Start a call whenever you need JARVIS.'}
              </p>
            </div>
            {!session.isConnected && (
              <Button
                variant="primary"
                size="lg"
                onClick={startCall}
                disabled={connecting}
                className="shrink-0 font-mono"
              >
                {connecting ? 'Connecting…' : appConfig.startButtonText}
              </Button>
            )}
          </div>

          {session.isConnected && (
            <AgentControlBar
              controls={controls}
              isConnected={session.isConnected}
              onDisconnect={session.end}
              onChatOpenChange={setTranscriptOpen}
              className="rounded-none border-0 bg-transparent p-0 drop-shadow-none"
            />
          )}

          {connectionError && (
            <p
              role="alert"
              className="text-destructive bg-destructive/10 mt-2 rounded-lg px-3 py-2 text-sm"
            >
              {connectionError}
            </p>
          )}
        </div>
      </aside>
    </>
  );
}
