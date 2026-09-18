'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Eye } from '@phosphor-icons/react/dist/ssr';
import { Button } from '@/components/livekit/button';
import type { VisionAnalysis } from '@/lib/visual/bridge';
import { analyzeCameraView } from '@/lib/visual/client';

interface VisualAnalyzerProps {
  analysisRef: React.RefObject<VisionAnalysis | null>;
  userId: string;
  stream: import('@/hooks/useCamera').CameraController['stream'];
  cameraDetail: string | null;
  onRetry: () => void;
  onClose: () => void;
  cameraStatus: string;
  companyPrivate: boolean;
}

const ERROR_TEXT: Record<string, string> = {
  camera_not_ready: 'The camera view is not ready yet.',
  frame_too_large: 'The camera frame could not be safely reduced.',
  company_mode_blocked: 'Visual analysis is unavailable in Company Mode.',
  not_signed_in: 'Sign in to analyze the camera view.',
  vision_unavailable: 'Local visual analysis is unavailable.',
  vision_no_description: 'The local visual model returned no description for this frame.',
  rate_limited: 'Please wait before analyzing another view.',
};

export function VisualAnalyzer({
  stream,
  analysisRef,
  userId,
  cameraStatus,
  cameraDetail,
  companyPrivate,
  onRetry,
  onClose,
}: VisualAnalyzerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const request = useRef<AbortController | null>(null);
  const active = useRef(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(
    async (externalSignal?: AbortSignal): Promise<string> => {
      if (
        active.current ||
        !videoRef.current ||
        cameraStatus !== 'live' ||
        companyPrivate ||
        !(stream as MediaStream | null)
          ?.getVideoTracks()
          .some((track) => track.readyState === 'live')
      )
        throw new Error('camera_not_ready');
      active.current = true;
      const controller = new AbortController();
      request.current = controller;
      const signal = externalSignal
        ? AbortSignal.any([controller.signal, externalSignal])
        : controller.signal;
      setBusy(true);
      setResult(null);
      setError(null);
      try {
        const description = await analyzeCameraView(videoRef.current, companyPrivate, {
          signal,
          expectedUser: userId,
        });
        signal.throwIfAborted();
        if (!externalSignal) setResult(description);
        return description;
      } catch (cause) {
        if (signal.aborted) throw new Error('cancelled');
        const code = cause instanceof Error ? cause.message : 'analysis_unavailable';
        setError(ERROR_TEXT[code] ?? 'Local visual analysis could not complete.');
        throw cause;
      } finally {
        active.current = false;
        if (request.current === controller) request.current = null;
        if (!controller.signal.aborted) setBusy(false);
      }
    },
    [cameraStatus, companyPrivate, stream, userId]
  );

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.srcObject = (stream as MediaStream | null) ?? null;
    if (stream) void video.play().catch(() => undefined);
    return () => {
      video.srcObject = null;
    };
  }, [stream]);
  useEffect(
    () => () => {
      request.current?.abort();
    },
    []
  );

  useEffect(() => {
    analysisRef.current = analyze;
    return () => {
      analysisRef.current = null;
      request.current?.abort();
      setResult(null);
    };
  }, [analysisRef, analyze]);

  return (
    <section aria-label="Vision camera view" className="bg-card mb-4 rounded-2xl border p-3">
      <div className="flex flex-wrap items-center gap-3">
        <video
          ref={videoRef}
          muted
          playsInline
          autoPlay
          aria-label="Local Vision preview"
          className="h-36 w-48 max-w-full rounded-lg object-contain"
        />
        <p role="status" className="text-muted-foreground text-sm">
          {cameraDetail ??
            (cameraStatus === 'live'
              ? 'Local preview — press Analyze camera view when ready.'
              : 'Waiting for camera…')}
        </p>
        {cameraStatus !== 'live' && cameraStatus !== 'requesting' && (
          <Button
            variant="outline"
            size="sm"
            onClick={(event) => {
              if (event.isTrusted) onRetry();
            }}
          >
            Retry camera
          </Button>
        )}
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            request.current?.abort();
            onClose();
          }}
        >
          Close Vision
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={(event) => {
            if (event.isTrusted) void analyze().catch(() => undefined);
          }}
          disabled={busy || cameraStatus !== 'live' || companyPrivate}
          aria-describedby="visual-analysis-privacy"
        >
          <Eye aria-hidden className="size-4" weight="bold" />
          {busy ? 'Analyzing…' : 'Analyze camera view'}
        </Button>
        <p id="visual-analysis-privacy" className="text-muted-foreground text-xs">
          Visual analysis is on-demand. One reduced frame goes only to FRIDAY’s local Ollama service
          and is not retained; gesture tracking remains browser-local.
        </p>
      </div>
      {result && (
        <p role="status" className="mt-2 text-sm">
          {result}
        </p>
      )}
      {error && (
        <p role="alert" className="text-destructive mt-2 text-sm">
          {error}
        </p>
      )}
    </section>
  );
}
