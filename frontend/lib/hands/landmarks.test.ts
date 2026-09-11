import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  HAND_LANDMARK_COUNT,
  type HandLandmark,
  type LandmarkerLike,
  classifyHandPose,
  createLandmarkerProvider,
  handIntentPoint,
  mapToViewport,
} from './landmarks.ts';

/**
 * Synthetic hands in the 21-point topology every common landmark model shares
 * (wrist, then four joints per finger from thumb to pinky). Fingers radiate
 * from the wrist; the tip of an extended finger is far from the wrist, the tip
 * of a curled finger folds back toward the palm.
 */
function syntheticHand(tipReach: number, options: { thumbReach?: number } = {}): HandLandmark[] {
  const wrist = { x: 0.5, y: 0.8, z: 0 };
  const points: HandLandmark[] = [wrist];
  // Finger base angles fan out above the wrist.
  const angles = [-1.2, -0.45, -0.2, 0.05, 0.3];
  const reaches = [options.thumbReach ?? tipReach, tipReach, tipReach, tipReach, tipReach];
  angles.forEach((angle, finger) => {
    const dx = Math.sin(angle);
    const dy = -Math.cos(angle);
    // Joints at 1 (mcp), 1.6 (pip), 2 (dip) times the palm unit, then the tip.
    const scale = 0.1;
    for (const radius of [1, 1.6, 2, reaches[finger]]) {
      points.push({ x: wrist.x + dx * radius * scale, y: wrist.y + dy * radius * scale, z: 0 });
    }
  });
  return points;
}

describe('hand pose classification', () => {
  it('reads an open hand when the fingertips reach well past the knuckles', () => {
    assert.equal(classifyHandPose(syntheticHand(2.6)), 'open');
  });

  it('recognises a deliberate thumb-to-index pinch before the open-hand fallback', () => {
    const pinched = syntheticHand(2.6);
    pinched[4] = { ...pinched[8] };
    assert.equal(classifyHandPose(pinched), 'pinch');
  });

  it('reads a fist when the fingertips fold back toward the palm', () => {
    assert.equal(classifyHandPose(syntheticHand(0.7)), 'fist');
  });

  it('refuses to guess for a half-curled hand', () => {
    assert.equal(classifyHandPose(syntheticHand(1.45)), 'unknown');
  });

  it('ignores the thumb, which curls unreliably', () => {
    assert.equal(classifyHandPose(syntheticHand(2.6, { thumbReach: 0.7 })), 'open');
    assert.equal(classifyHandPose(syntheticHand(0.7, { thumbReach: 2.6 })), 'fist');
  });

  it('treats a malformed landmark list as unknown', () => {
    assert.equal(classifyHandPose([]), 'unknown');
    assert.equal(classifyHandPose(syntheticHand(2.6).slice(0, HAND_LANDMARK_COUNT - 1)), 'unknown');
  });
});

describe('the intent point', () => {
  it('is the palm centre, which barely moves as the fingers close', () => {
    const open = handIntentPoint(syntheticHand(2.6));
    const fist = handIntentPoint(syntheticHand(0.7));
    assert.ok(open && fist);
    assert.ok(Math.abs(open.x - fist.x) < 0.005, 'x stays put when the hand closes');
    assert.ok(Math.abs(open.y - fist.y) < 0.005, 'y stays put when the hand closes');
    assert.ok(open.y < 0.8 && open.y > 0.6, 'the palm centre sits above the wrist');
  });

  it('is null without a full hand', () => {
    assert.equal(handIntentPoint([]), null);
  });
});

describe('mapping the camera frame onto the viewport', () => {
  const viewport = { width: 1000, height: 500 };
  const near = (a: number, b: number) => Math.abs(a - b) < 1e-6;

  it('mirrors horizontally so a hand moving right moves the cursor right', () => {
    const left = mapToViewport({ x: 0.2, y: 0.5 }, viewport, { mirror: true, margin: 0 });
    const right = mapToViewport({ x: 0.8, y: 0.5 }, viewport, { mirror: true, margin: 0 });
    assert.ok(left.x > right.x);
    assert.ok(near(left.x, 800));
    assert.ok(near(right.x, 200));
  });

  it('stretches the reachable middle of the frame over the whole screen', () => {
    const margin = 0.15;
    const edge = mapToViewport({ x: margin, y: margin }, viewport, { mirror: false, margin });
    assert.equal(edge.x, 0);
    assert.equal(edge.y, 0);
    const far = mapToViewport({ x: 1 - margin, y: 1 - margin }, viewport, {
      mirror: false,
      margin,
    });
    assert.equal(far.x, viewport.width);
    assert.equal(far.y, viewport.height);
  });

  it('clamps to the viewport rather than pointing off screen', () => {
    const out = mapToViewport({ x: -0.5, y: 1.7 }, viewport, { mirror: false, margin: 0.1 });
    assert.equal(out.x, 0);
    assert.equal(out.y, viewport.height);
  });
});

describe('a landmarker adapter', () => {
  it('turns a landmarker result into one observation and nothing else', () => {
    const calls: number[] = [];
    const fake: LandmarkerLike = {
      detectForVideo(_video, timestamp) {
        calls.push(timestamp);
        return {
          landmarks: [syntheticHand(2.6)],
          handedness: [[{ categoryName: 'Right', score: 0.9 }]],
        };
      },
    };
    const provider = createLandmarkerProvider(fake);
    const observation = provider.detect({} as never, 42);
    assert.deepEqual(calls, [42]);
    assert.ok(observation);
    assert.equal(observation.pose, 'open');
    assert.ok(observation.point.x > 0.4 && observation.point.x < 0.6);
  });

  it('reports no hand for an empty or malformed result', () => {
    const empty = createLandmarkerProvider({ detectForVideo: () => ({ landmarks: [] }) });
    assert.equal(empty.detect({} as never, 1), null);
    const broken = createLandmarkerProvider({
      detectForVideo: () => ({ landmarks: [[{ x: 0.1, y: 0.1 }]] }),
    });
    assert.equal(broken.detect({} as never, 1), null);
  });

  it('asks the model once per advancing timestamp and reuses the answer otherwise', () => {
    const calls: number[] = [];
    const provider = createLandmarkerProvider({
      detectForVideo(_video, timestamp) {
        calls.push(timestamp);
        return { landmarks: [syntheticHand(2.6)] };
      },
    });
    const first = provider.detect({} as never, 10);
    const again = provider.detect({} as never, 10);
    const backwards = provider.detect({} as never, 9);
    assert.deepEqual(calls, [10]);
    assert.equal(again, first);
    assert.equal(backwards, first);
    provider.detect({} as never, 11);
    assert.deepEqual(calls, [10, 11]);
  });

  it('never rethrows a model failure into the frame loop', () => {
    const failing = createLandmarkerProvider({
      detectForVideo: () => {
        throw new Error('wasm not ready');
      },
    });
    assert.equal(failing.detect({} as never, 1), null);
  });

  it('closes the model when disposed', () => {
    let closed = 0;
    const provider = createLandmarkerProvider({
      detectForVideo: () => ({ landmarks: [] }),
      close: () => {
        closed += 1;
      },
    });
    provider.dispose();
    assert.equal(closed, 1);
  });
});
