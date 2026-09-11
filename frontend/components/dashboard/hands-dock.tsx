'use client';

/**
 * Hand control, docked into the workspace: the camera state in one line, the
 * camera to use, a small local preview, the recognition runtime and the
 * pipeline itself. Mounting the dock asks for the camera and loads the hand
 * model into this browser; the status line says which of the two it is still
 * waiting on. The preview is the stream drawn straight onto a video element;
 * nothing is read back from it, recorded, or sent anywhere. Unmounting stops
 * every track and closes the model.
 */
import { type RefObject, useEffect, useRef, useState } from 'react';
import { LockSimple, VideoCameraSlash, X } from '@phosphor-icons/react/dist/ssr';
import { Button } from '@/components/livekit/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/livekit/select';
import { useCamera } from '@/hooks/useCamera';
import { useHandRuntime } from '@/hooks/useHandRuntime';
import type { CameraStatus } from '@/lib/hands/camera-session';
import { type HandTone, handStatus } from '@/lib/hands/status';
import type { HandSurfaceController } from '@/lib/hands/surface';
import { cn } from '@/lib/utils';
import { type HandActivity, HandLayer } from './hand-layer';

const TONE_DOT: Record<HandTone, string> = {
  idle: 'bg-muted-foreground/50',
  busy: 'bg-amber-500 animate-pulse motion-reduce:animate-none',
  live: 'bg-green-500',
  error: 'bg-destructive',
};

const RETRYABLE: CameraStatus[] = ['denied', 'no-camera', 'busy', 'unavailable', 'ended', 'error'];

const HOW_TO =
  'Open hand to point. Touch thumb to index finger to click; hold a fist over a widget to grab, then open to drop.';

interface HandsDockProps {
  controller: RefObject<HandSurfaceController | null>;
  onClose: () => void;
}

export function HandsDock({ controller, onClose }: HandsDockProps) {
  const camera = useCamera();
  const videoRef = useRef<HTMLVideoElement>(null);
  const runtime = useHandRuntime();
  const [activity, setActivity] = useState<HandActivity>({ phase: 'idle', target: null });

  // Switching hand control on is the request: ask for the camera at once.
  const { start } = camera;
  useEffect(() => {
    void start();
  }, [start]);

  // The preview is the live stream itself, mirrored like a mirror.
  const { stream } = camera;
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.srcObject = (stream as MediaStream | null) ?? null;
    if (stream) void video.play().catch(() => undefined);
    return () => {
      video.srcObject = null;
    };
  }, [stream]);

  const live = camera.status === 'live';
  const cameraRetryable = RETRYABLE.includes(camera.status);
  const runtimeFailed = runtime.status === 'failed';
  const status = handStatus({
    camera: camera.status,
    cameraDetail:
      camera.status === 'denied'
        ? (camera.detail ?? '') + ' Allow the camera for this site in your browser, then try again.'
        : camera.detail,
    runtime: runtime.status,
    runtimeDetail: runtime.detail,
    phase: activity.phase,
    target: activity.target,
  });

  return (
    <section
      aria-label="Hand control"
      className="bg-card text-card-foreground border-border mb-4 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-2xl border px-3 py-2 shadow-sm"
    >
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <div className="bg-muted relative h-12 w-16 shrink-0 overflow-hidden rounded-lg">
          <video
            ref={videoRef}
            muted
            playsInline
            autoPlay
            aria-label="Local camera preview"
            className={cn(
              'size-full -scale-x-100 object-cover transition-opacity duration-300 motion-reduce:transition-none',
              live ? 'opacity-100' : 'opacity-0'
            )}
          />
          {!live && (
            <span
              aria-hidden
              className="text-muted-foreground absolute inset-0 grid place-items-center"
            >
              <VideoCameraSlash className="size-5" weight="bold" />
            </span>
          )}
        </div>
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
          <p className="text-muted-foreground truncate text-xs">{status.detail ?? HOW_TO}</p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span
          className="text-muted-foreground inline-flex items-center gap-1 font-mono text-[10px] tracking-wider uppercase"
          title="Camera frames stay in this browser. Nothing is uploaded, stored or logged."
        >
          <LockSimple aria-hidden className="size-3" weight="bold" />
          Local only
        </span>
        <Select
          value={camera.selectedDeviceId ?? ''}
          onValueChange={(deviceId) => void camera.selectDevice(deviceId)}
          disabled={camera.devices.length === 0 || camera.status === 'requesting'}
        >
          <SelectTrigger size="sm" aria-label="Camera" className="max-w-56">
            <SelectValue placeholder="Camera" />
          </SelectTrigger>
          <SelectContent>
            {camera.devices.map((device) => (
              <SelectItem key={device.deviceId} value={device.deviceId}>
                {device.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {(cameraRetryable || runtimeFailed) && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              if (runtimeFailed) runtime.retry();
              if (cameraRetryable) void start();
            }}
          >
            Try again
          </Button>
        )}
        <Button variant="ghost" size="icon" aria-label="Turn off hand control" onClick={onClose}>
          <X className="size-4" weight="bold" />
        </Button>
      </div>

      <HandLayer
        video={videoRef}
        cameraStatus={camera.status}
        provider={runtime.provider}
        controller={controller}
        onActivity={setActivity}
      />
    </section>
  );
}
