/**
 * The hand gesture state machine.
 *
 * A pure reducer: each camera frame becomes an observation (a pose and a
 * point, or nothing when no hand is seen) and the reducer answers with the
 * next state plus the effects the surface should apply. It never touches the
 * DOM or the layout, so the whole interaction, dwell timing included, is
 * tested with synthetic frames.
 *
 *   open hand   -> hover: the cursor follows the palm
 *   pinch       -> click the target under the cursor
 *   fist held   -> after a deliberate dwell, grab it
 *   fist moves  -> move the grabbed target
 *   hand opens  -> release
 *   hand lost   -> cancel a grab; the layout goes back to where it was
 */
export interface Point {
  x: number;
  y: number;
}

export type HandPose = 'open' | 'pinch' | 'fist' | 'unknown';

export interface HandObservation {
  pose: HandPose;
  /** Normalised frame coordinates, 0..1. */
  point: Point;
}

export type GesturePhase = 'idle' | 'tracking' | 'holding' | 'grabbing';

export interface GestureConfig {
  /** Consecutive frames a new pose must be seen before it counts. */
  confirmFrames: number;
  /** How long a fist is held before a grab engages. */
  dwellMs: number;
  /** How long the hand may vanish before the gesture is abandoned. */
  lostMs: number;
  /** Exponential smoothing weight for the cursor, 0..1; higher follows faster. */
  smoothing: number;
}

export const DEFAULT_GESTURE_CONFIG: GestureConfig = {
  confirmFrames: 3,
  dwellMs: 1600,
  lostMs: 400,
  smoothing: 0.35,
};

export interface GestureState {
  phase: GesturePhase;
  /** Smoothed cursor point, normalised. */
  point: Point | null;
  /** The pose currently believed, after confirmation. */
  confirmed: HandPose;
  candidate: HandPose | null;
  candidateFrames: number;
  /** When the fist was confirmed; drives the dwell. */
  fistSince: number | null;
  /** 0..1 progress toward a grab while holding; 1 while grabbing. */
  dwellProgress: number;
  lastSeen: number | null;
}

export type GestureEvent =
  | { type: 'observe'; t: number; hand: HandObservation | null }
  | { type: 'disable' };

export type GestureEffect =
  | { type: 'hover'; point: Point }
  | { type: 'hover-end' }
  | { type: 'select'; point: Point }
  | { type: 'grab'; point: Point }
  | { type: 'move'; point: Point }
  | { type: 'release'; point: Point }
  | { type: 'cancel' };

export interface GestureResult {
  state: GestureState;
  effects: GestureEffect[];
}

export function initialGestureState(): GestureState {
  return {
    phase: 'idle',
    point: null,
    confirmed: 'unknown',
    candidate: null,
    candidateFrames: 0,
    fistSince: null,
    dwellProgress: 0,
    lastSeen: null,
  };
}

function smooth(previous: Point | null, next: Point, weight: number): Point {
  if (!previous) return { x: next.x, y: next.y };
  const w = Math.min(1, Math.max(0, weight));
  return { x: previous.x + (next.x - previous.x) * w, y: previous.y + (next.y - previous.y) * w };
}

/** Effects that put the surface back to rest from a given phase. */
function abandon(state: GestureState): GestureEffect[] {
  const effects: GestureEffect[] = [];
  if (state.phase === 'grabbing') effects.push({ type: 'cancel' });
  if (state.phase !== 'idle') effects.push({ type: 'hover-end' });
  return effects;
}

/** Feed one pose reading through the confirmation window. Returns the newly confirmed pose, if any. */
function confirmPose(
  state: GestureState,
  pose: HandPose,
  confirmFrames: number
): { state: GestureState; changed: HandPose | null } {
  if (pose === 'unknown' || pose === state.confirmed) {
    return { state: { ...state, candidate: null, candidateFrames: 0 }, changed: null };
  }
  const candidateFrames = pose === state.candidate ? state.candidateFrames + 1 : 1;
  if (candidateFrames >= Math.max(1, confirmFrames)) {
    return {
      state: { ...state, confirmed: pose, candidate: null, candidateFrames: 0 },
      changed: pose,
    };
  }
  return { state: { ...state, candidate: pose, candidateFrames }, changed: null };
}

export function reduceHandGesture(
  state: GestureState,
  event: GestureEvent,
  config: GestureConfig = DEFAULT_GESTURE_CONFIG
): GestureResult {
  if (event.type === 'disable') {
    return { state: initialGestureState(), effects: abandon(state) };
  }

  const { t, hand } = event;

  if (!hand) {
    if (state.phase === 'idle') return { state, effects: [] };
    const lastSeen = state.lastSeen ?? t;
    if (t - lastSeen > config.lostMs) {
      return { state: initialGestureState(), effects: abandon(state) };
    }
    return { state: { ...state, lastSeen }, effects: [] };
  }

  const point = smooth(state.point, hand.point, config.smoothing);
  const seen: GestureState = {
    ...state,
    point,
    lastSeen: t,
    phase: state.phase === 'idle' ? 'tracking' : state.phase,
  };
  const { state: next, changed } = confirmPose(seen, hand.pose, config.confirmFrames);

  switch (next.phase) {
    case 'tracking': {
      if (changed === 'pinch') {
        // A pinch is deliberately click-only: holding it can never become a drag.
        return { state: next, effects: [{ type: 'select', point }] };
      }
      if (changed === 'fist') {
        return {
          state: { ...next, phase: 'holding', fistSince: t, dwellProgress: 0 },
          effects: [],
        };
      }
      return { state: next, effects: [{ type: 'hover', point }] };
    }
    case 'holding': {
      if (changed !== null && changed !== 'fist') {
        return {
          state: { ...next, phase: 'tracking', fistSince: null, dwellProgress: 0 },
          effects: [{ type: 'hover', point }],
        };
      }
      const since = next.fistSince ?? t;
      const dwellProgress = Math.min(1, Math.max(0, (t - since) / Math.max(1, config.dwellMs)));
      if (dwellProgress >= 1) {
        return {
          state: { ...next, phase: 'grabbing', dwellProgress: 1 },
          effects: [{ type: 'grab', point }],
        };
      }
      return { state: { ...next, dwellProgress }, effects: [{ type: 'hover', point }] };
    }
    case 'grabbing': {
      if (changed !== null && changed !== 'fist') {
        return {
          state: { ...next, phase: 'tracking', fistSince: null, dwellProgress: 0 },
          effects: [{ type: 'release', point }],
        };
      }
      return { state: next, effects: [{ type: 'move', point }] };
    }
    default:
      return { state: next, effects: [] };
  }
}
