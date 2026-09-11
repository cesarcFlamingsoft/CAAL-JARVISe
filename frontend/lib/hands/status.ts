/**
 * One truthful line about hand control, for the dock. Mirrors the shape the
 * voice dock uses so the two capabilities read the same way. The camera comes
 * first, then the recognition runtime, and only once both are live does the
 * line follow the gesture.
 */
import type { CameraStatus } from './camera-session';
import type { GesturePhase } from './gesture';
import type { HandRuntimeStatus } from './runtime';

export type HandTone = 'idle' | 'busy' | 'live' | 'error';

export interface HandStatus {
  label: string;
  tone: HandTone;
  detail: string | null;
}

export interface HandStatusInput {
  camera: CameraStatus;
  cameraDetail: string | null;
  runtime: HandRuntimeStatus;
  runtimeDetail: string | null;
  phase: GesturePhase;
  target: { title: string; movable: boolean } | null;
}

const CAMERA_PROBLEMS: Partial<Record<CameraStatus, string>> = {
  unsupported: 'Camera unavailable in this browser',
  denied: 'Camera access declined',
  'no-camera': 'No camera found',
  busy: 'Camera busy',
  unavailable: 'Camera unavailable',
  ended: 'Camera stopped',
  error: 'Camera error',
};

export function handStatus({
  camera,
  cameraDetail,
  runtime,
  runtimeDetail,
  phase,
  target,
}: HandStatusInput): HandStatus {
  const problem = CAMERA_PROBLEMS[camera];
  if (problem) return { label: problem, tone: 'error', detail: cameraDetail };
  if (camera === 'idle') return { label: 'Camera off', tone: 'idle', detail: null };
  if (camera === 'requesting') {
    return { label: 'Asking for the camera', tone: 'busy', detail: null };
  }
  if (runtime === 'failed') {
    return {
      label: 'Camera on, hand recognition failed to load',
      tone: 'error',
      detail: runtimeDetail,
    };
  }
  if (runtime === 'loading') {
    return {
      label: 'Camera on, loading hand recognition',
      tone: 'busy',
      detail: 'The hand model is loading into this browser. Nothing is sent anywhere.',
    };
  }
  switch (phase) {
    case 'tracking':
      return {
        label: 'Tracking your hand',
        tone: 'live',
        detail: target ? 'Over ' + target.title + '. Touch thumb to index finger to click it.' : null,
      };
    case 'holding':
      return {
        label: target ? 'Hold to grab ' + target.title : 'Hold to grab',
        tone: 'live',
        detail: target && !target.movable ? target.title + ' cannot be moved here.' : null,
      };
    case 'grabbing':
      if (target && !target.movable) {
        return {
          label: target.title + ' cannot be moved here',
          tone: 'live',
          detail: 'Widgets only rearrange on the wide grid.',
        };
      }
      return {
        label: target ? 'Moving ' + target.title : 'Moving',
        tone: 'live',
        detail: 'Open your hand to drop it.',
      };
    default:
      return { label: 'Watching for a hand', tone: 'live', detail: null };
  }
}
