/**
 * Hand landmarks: what they are, what they mean, and where they come from.
 *
 * Every common hand-landmark model (MediaPipe Hands and its descendants)
 * reports the same 21 points: the wrist, then four joints per finger from
 * thumb to pinky, each normalised to the video frame. The geometry here is
 * pure and model-agnostic, so it is tested without a camera and a swapped
 * model keeps working.
 *
 * A provider turns the current video frame into at most one observation
 * (pose plus intent point) entirely inside the browser. The provider this
 * build runs is the MediaPipe HandLandmarker, wrapped by
 * createLandmarkerProvider below and loaded by mediapipe.ts from the runtime
 * and model files this origin serves; runtime.ts owns its loading, ready and
 * failed states, so a provider only ever exists once its model is running.
 */
import type { HandObservation, HandPose, Point } from './gesture';

export type { HandPose, Point } from './gesture';

export const HAND_LANDMARK_COUNT = 21;

export interface HandLandmark {
  /** Normalised to the frame width, 0..1. */
  x: number;
  /** Normalised to the frame height, 0..1. */
  y: number;
  /** Depth relative to the wrist, when the model provides it. */
  z?: number;
}

const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_TIP = 8;
const PALM_WIDTH_START = 5;
const PALM_WIDTH_END = 17;

/** Index, middle, ring, pinky: knuckle (mcp) and tip. The thumb is left out. */
const FINGERS = [
  { mcp: 5, tip: 8 },
  { mcp: 9, tip: 12 },
  { mcp: 13, tip: 16 },
  { mcp: 17, tip: 20 },
];

/** Palm landmarks: the wrist and the four finger knuckles. */
const PALM = [WRIST, 5, 9, 13, 17];

/** A fingertip at least this far past its knuckle (relative to the wrist) is extended. */
const EXTENDED_RATIO = 1.6;
/** A fingertip no farther from the wrist than this (relative to its knuckle) is curled. */
const CURLED_RATIO = 1.15;
/** How many of the four fingers must agree. */
const AGREEING_FINGERS = 3;
/** Thumb and index must be this close relative to palm width to count as a pinch. */
const PINCH_RATIO = 0.45;

function isLandmark(value: unknown): value is HandLandmark {
  if (!value || typeof value !== 'object') return false;
  const point = value as Record<string, unknown>;
  return (
    typeof point.x === 'number' &&
    Number.isFinite(point.x) &&
    typeof point.y === 'number' &&
    Number.isFinite(point.y) &&
    (point.z === undefined || (typeof point.z === 'number' && Number.isFinite(point.z)))
  );
}

/** True for a complete, well-formed 21-point hand. */
export function isHandLandmarks(value: unknown): value is HandLandmark[] {
  return Array.isArray(value) && value.length === HAND_LANDMARK_COUNT && value.every(isLandmark);
}

function distance(a: HandLandmark, b: HandLandmark): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Pinch, open, fist, or unknown. A pinch is detected first from the thumb and
 * index tips relative to palm width, making it stable across camera distance.
 * The other poses each use finger votes by comparing how far a tip is from the
 * wrist with how far its knuckle is. Anything in between, or a split vote, is
 * unknown rather than a guess.
 */
export function classifyHandPose(landmarks: readonly HandLandmark[]): HandPose {
  if (!isHandLandmarks(landmarks)) return 'unknown';
  const palmWidth = distance(landmarks[PALM_WIDTH_START], landmarks[PALM_WIDTH_END]);
  if (palmWidth > 0 && distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) <= palmWidth * PINCH_RATIO) {
    return 'pinch';
  }
  const wrist = landmarks[WRIST];
  let extended = 0;
  let curled = 0;
  for (const finger of FINGERS) {
    const knuckle = distance(landmarks[finger.mcp], wrist);
    if (knuckle <= 0) return 'unknown';
    const ratio = distance(landmarks[finger.tip], wrist) / knuckle;
    if (ratio >= EXTENDED_RATIO) extended += 1;
    else if (ratio <= CURLED_RATIO) curled += 1;
  }
  if (extended >= AGREEING_FINGERS) return 'open';
  if (curled >= AGREEING_FINGERS) return 'fist';
  return 'unknown';
}

/**
 * Where the hand is pointing, as the palm centre. Fingertips move a lot when
 * a hand closes, so tracking one would make the cursor jump at the exact
 * moment a selection happens; the palm barely moves.
 */
export function handIntentPoint(landmarks: readonly HandLandmark[]): Point | null {
  if (!isHandLandmarks(landmarks)) return null;
  let x = 0;
  let y = 0;
  for (const index of PALM) {
    x += landmarks[index].x;
    y += landmarks[index].y;
  }
  return { x: x / PALM.length, y: y / PALM.length };
}

export interface ViewportSize {
  width: number;
  height: number;
}

export interface ViewportMapping {
  /** Flip horizontally so moving a hand right moves the cursor right. */
  mirror: boolean;
  /** Fraction of the frame on each edge a hand rarely reaches; mapped away. */
  margin: number;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

/** Normalised frame coordinates to viewport pixels. */
export function mapToViewport(
  point: Point,
  viewport: ViewportSize,
  mapping: ViewportMapping
): Point {
  const margin = mapping.margin > 0 && mapping.margin < 0.5 ? mapping.margin : 0;
  const span = 1 - 2 * margin;
  let nx = clamp01((point.x - margin) / span);
  const ny = clamp01((point.y - margin) / span);
  if (mapping.mirror) nx = 1 - nx;
  return { x: nx * viewport.width, y: ny * viewport.height };
}

export interface HandLandmarkProvider {
  readonly id: string;
  /** Read the current frame. Must be cheap, synchronous and never throw. */
  detect(video: HTMLVideoElement, timestampMs: number): HandObservation | null;
  dispose(): void;
}

/** The slice of a MediaPipe-style HandLandmarker this adapter relies on. */
export interface LandmarkerLike {
  detectForVideo(
    video: HTMLVideoElement,
    timestampMs: number
  ): { landmarks?: unknown[]; handedness?: unknown } | null | undefined;
  close?(): void;
}

/**
 * Wrap a landmarker so the frame loop only ever sees a clean observation or
 * null. Landmarkers insist on timestamps that advance, so a frame whose
 * timestamp has not moved on is answered from the last result instead of
 * being sent to the model again.
 */
export function createLandmarkerProvider(
  landmarker: LandmarkerLike,
  id = 'landmarker'
): HandLandmarkProvider {
  let lastTimestamp = Number.NEGATIVE_INFINITY;
  let last: HandObservation | null = null;
  return {
    id,
    detect(video, timestampMs) {
      if (timestampMs <= lastTimestamp) return last;
      lastTimestamp = timestampMs;
      last = null;
      let result: ReturnType<LandmarkerLike['detectForVideo']>;
      try {
        result = landmarker.detectForVideo(video, timestampMs);
      } catch {
        return null;
      }
      const first = result?.landmarks?.[0];
      if (!isHandLandmarks(first)) return null;
      const point = handIntentPoint(first);
      if (!point) return null;
      last = { pose: classifyHandPose(first), point };
      return last;
    },
    dispose() {
      try {
        landmarker.close?.();
      } catch {
        // Already closed.
      }
    },
  };
}
