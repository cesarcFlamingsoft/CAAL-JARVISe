'use client';

/**
 * Runs the hand pipeline while the camera is live: reads the current video
 * frame through the landmark provider, feeds the gesture reducer, applies the
 * resulting effects to the Monitor surface and draws the cursor. The loop
 * only runs once the camera is live and the runtime has produced a provider.
 * Frames and landmarks exist only inside this loop and are dropped on the
 * next one.
 */
import { type RefObject, useEffect, useRef, useState } from 'react';
import type { CameraStatus } from '@/lib/hands/camera-session';
import {
  DEFAULT_GESTURE_CONFIG,
  type GestureEffect,
  type GesturePhase,
  type GestureState,
  type Point,
  initialGestureState,
  reduceHandGesture,
} from '@/lib/hands/gesture';
import { type HandLandmarkProvider, mapToViewport } from '@/lib/hands/landmarks';
import type { HandSurfaceController, HandTarget } from '@/lib/hands/surface';
import { HandCursor, type HandCursorState } from './hand-cursor';

export interface HandActivity {
  phase: GesturePhase;
  target: HandTarget | null;
}

interface HandLayerProps {
  video: RefObject<HTMLVideoElement | null>;
  cameraStatus: CameraStatus;
  /** Null until the recognition runtime is ready. */
  provider: HandLandmarkProvider | null;
  controller: RefObject<HandSurfaceController | null>;
  /** Called when the phase or target changes, never per frame. */
  onActivity: (activity: HandActivity) => void;
}

/** Hands rarely reach the frame edges; the middle 70% maps onto the whole screen. */
const MAPPING = { mirror: true, margin: 0.15 };
const IDLE: HandCursorState = { phase: 'idle', point: null, dwellProgress: 0, target: null };

function toViewport(point: Point): Point {
  return mapToViewport(point, { width: window.innerWidth, height: window.innerHeight }, MAPPING);
}

export function HandLayer({
  video,
  cameraStatus,
  provider,
  controller,
  onActivity,
}: HandLayerProps) {
  const [cursor, setCursor] = useState<HandCursorState>(IDLE);
  const onActivityRef = useRef(onActivity);
  onActivityRef.current = onActivity;

  useEffect(() => {
    const element = video.current;
    if (cameraStatus !== 'live' || !provider || !element) {
      setCursor(IDLE);
      onActivityRef.current({ phase: 'idle', target: null });
      return;
    }

    let state: GestureState = initialGestureState();
    let target: HandTarget | null = null;
    let published = '';
    let stopped = false;
    let handle = 0;
    const videoFrames = typeof element.requestVideoFrameCallback === 'function';

    const apply = (effects: GestureEffect[], next: GestureState) => {
      const surface = controller.current;
      for (const effect of effects) {
        switch (effect.type) {
          case 'hover': {
            const hovered = surface?.hover(toViewport(effect.point)) ?? null;
            // While a fist is held the selection stands; hovering only paints.
            if (next.phase === 'tracking') target = hovered;
            break;
          }
          case 'hover-end':
            target = null;
            surface?.hoverEnd();
            break;
          case 'select':
            target = surface?.select(toViewport(effect.point)) ?? null;
            break;
          case 'grab':
            target = surface?.grab(toViewport(effect.point)) ?? null;
            break;
          case 'move':
            surface?.move(toViewport(effect.point));
            break;
          case 'release':
            surface?.release();
            break;
          case 'cancel':
            surface?.cancel();
            break;
        }
      }
      setCursor({
        phase: next.phase,
        point: next.point && next.phase !== 'idle' ? toViewport(next.point) : null,
        dwellProgress: next.dwellProgress,
        target: target ? { title: target.title, movable: target.movable } : null,
      });
      const key = next.phase + '|' + (target ? target.id + '|' + target.movable : '');
      if (key !== published) {
        published = key;
        onActivityRef.current({ phase: next.phase, target });
      }
    };

    const tick = (now: number) => {
      if (stopped) return;
      if (element.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !element.paused) {
        const hand = provider.detect(element, now);
        const result = reduceHandGesture(
          state,
          { type: 'observe', t: now, hand },
          DEFAULT_GESTURE_CONFIG
        );
        state = result.state;
        if (result.effects.length > 0 || result.state.phase !== 'idle') {
          apply(result.effects, result.state);
        }
      }
      schedule();
    };
    const schedule = () => {
      if (stopped) return;
      handle = videoFrames
        ? element.requestVideoFrameCallback((now) => tick(now))
        : window.requestAnimationFrame(tick);
    };
    schedule();

    return () => {
      stopped = true;
      if (videoFrames) element.cancelVideoFrameCallback(handle);
      else window.cancelAnimationFrame(handle);
      // Put the surface back: a grab in progress is cancelled, not committed.
      const result = reduceHandGesture(state, { type: 'disable' }, DEFAULT_GESTURE_CONFIG);
      apply(result.effects, result.state);
    };
  }, [cameraStatus, controller, provider, video]);

  return <HandCursor {...cursor} />;
}
