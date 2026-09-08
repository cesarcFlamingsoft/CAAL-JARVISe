'use client';

/**
 * A compact view of the agent for the voice dock: the avatar video when the
 * agent publishes one, otherwise the configured audio visualizer.
 */
import { BarVisualizer, VideoTrack, useVoiceAssistant } from '@livekit/components-react';
import { JarvisVisualizer } from '@/components/app/jarvis-visualizer';
import { useVisualizationType } from '@/hooks/useVisualizationType';
import { cn } from '@/lib/utils';

export function AgentTile({ className }: { className?: string }) {
  const { audioTrack, videoTrack } = useVoiceAssistant();
  const visualizationType = useVisualizationType();

  if (videoTrack) {
    return (
      <VideoTrack
        trackRef={videoTrack}
        className={cn('size-14 shrink-0 rounded-xl bg-black object-cover', className)}
      />
    );
  }

  if (visualizationType === 'jarvis') {
    return (
      <div className={cn('size-14 shrink-0', className)} aria-hidden>
        <JarvisVisualizer trackRef={audioTrack} className="h-full w-full" />
      </div>
    );
  }

  return (
    <div className={cn('size-14 shrink-0', className)} aria-hidden>
      <BarVisualizer
        barCount={5}
        options={{ minHeight: 8 }}
        trackRef={audioTrack}
        className="flex h-full w-full items-center justify-center gap-0.5 px-2"
      >
        <span className="bg-primary h-full w-1 origin-center rounded-full" />
      </BarVisualizer>
    </div>
  );
}
