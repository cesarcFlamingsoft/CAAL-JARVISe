import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  DEFAULT_GESTURE_CONFIG,
  type GestureEffect,
  type GestureState,
  type HandObservation,
  type HandPose,
  initialGestureState,
  reduceHandGesture,
} from './gesture.ts';

const FRAME_MS = 33;
const config = { ...DEFAULT_GESTURE_CONFIG, confirmFrames: 2, dwellMs: 1600, lostMs: 300 };

interface Run {
  state: GestureState;
  effects: GestureEffect[];
  t: number;
}

function start(): Run {
  return { state: initialGestureState(), effects: [], t: 0 };
}

/** Feed count frames of one pose at one point; collect the effects. */
function frames(run: Run, pose: HandPose | null, point = { x: 0.5, y: 0.5 }, count = 1): Run {
  let { state, t } = run;
  const effects: GestureEffect[] = [];
  for (let i = 0; i < count; i += 1) {
    t += FRAME_MS;
    const hand: HandObservation | null = pose ? { pose, point } : null;
    const result = reduceHandGesture(state, { type: 'observe', t, hand }, config);
    state = result.state;
    effects.push(...result.effects);
  }
  return { state, effects, t };
}

function kinds(run: Run): string[] {
  return run.effects.map((effect) => effect.type);
}

const DWELL_FRAMES = Math.ceil(config.dwellMs / FRAME_MS) + 1;
const LOST_FRAMES = Math.ceil(config.lostMs / FRAME_MS) + 1;

function grabbedRun(): Run {
  const held = frames(frames(start(), 'open', undefined, 3), 'fist', undefined, 2);
  return frames(held, 'fist', undefined, DWELL_FRAMES);
}

describe('an open hand', () => {
  it('tracks a hover point and nothing else', () => {
    const run = frames(start(), 'open', { x: 0.3, y: 0.4 }, 3);
    assert.equal(run.state.phase, 'tracking');
    assert.ok(kinds(run).every((kind) => kind === 'hover'));
    assert.ok(run.effects.length > 0);
  });

  it('smooths the point instead of snapping to every frame', () => {
    const settled = frames(start(), 'open', { x: 0.2, y: 0.2 }, 10);
    const jumped = frames(settled, 'open', { x: 0.8, y: 0.8 }, 1);
    const hover = jumped.effects.at(-1);
    assert.ok(hover && hover.type === 'hover');
    assert.ok(hover.point.x > 0.2 && hover.point.x < 0.8, 'moves toward, not onto, the new point');
  });
});

describe('closing a fist', () => {
  it('arms a grab only after the fist is confirmed for several frames', () => {
    const open = frames(start(), 'open', undefined, 3);
    const flicker = frames(open, 'fist', undefined, 1);
    assert.equal(flicker.state.phase, 'tracking', 'one fist frame is noise');
    const held = frames(flicker, 'fist', undefined, 1);
    assert.deepEqual(kinds(held), []);
    assert.equal(held.state.phase, 'holding');
    const more = frames(held, 'fist', undefined, 5);
    assert.ok(!kinds(more).includes('select'), 'fist movement never clicks a control');
  });

  it('does not select when the pose is unknown', () => {
    const run = frames(frames(start(), 'open', undefined, 3), 'unknown', undefined, 6);
    assert.ok(!kinds(run).includes('select'));
    assert.equal(run.state.phase, 'tracking');
  });

  it('reports dwell progress while the fist is held', () => {
    const held = frames(frames(start(), 'open', undefined, 3), 'fist', undefined, 2);
    assert.equal(held.state.dwellProgress, 0);
    const later = frames(held, 'fist', undefined, Math.round(800 / FRAME_MS));
    assert.ok(later.state.dwellProgress > 0.4 && later.state.dwellProgress < 0.6);
  });
});

describe('a pinch click', () => {
  it('activates once without ever becoming a widget grab', () => {
    const open = frames(start(), 'open', undefined, 3);
    const pinched = frames(open, 'pinch', undefined, 2);
    assert.equal(kinds(pinched).filter((kind) => kind === 'select').length, 1);
    assert.equal(pinched.state.phase, 'tracking');
    const held = frames(pinched, 'pinch', undefined, DWELL_FRAMES);
    assert.ok(!kinds(held).includes('grab'));
    assert.equal(held.state.phase, 'tracking');
  });
});


describe('holding the fist', () => {
  it('grabs after the dwell, then moves with the hand, then releases when the hand opens', () => {
    const grabbed = grabbedRun();
    assert.equal(kinds(grabbed).filter((kind) => kind === 'grab').length, 1);
    assert.equal(grabbed.state.phase, 'grabbing');
    assert.equal(grabbed.state.dwellProgress, 1);

    const moved = frames(grabbed, 'fist', { x: 0.7, y: 0.6 }, 4);
    assert.ok(kinds(moved).every((kind) => kind === 'move'));
    assert.equal(moved.state.phase, 'grabbing');

    const released = frames(moved, 'open', { x: 0.7, y: 0.6 }, 2);
    assert.ok(kinds(released).includes('release'));
    assert.equal(released.state.phase, 'tracking');
    assert.ok(!kinds(frames(released, 'open', undefined, 3)).includes('release'));
  });

  it('opening before the dwell completes is only a selection, never a grab', () => {
    const held = frames(frames(start(), 'open', undefined, 3), 'fist', undefined, 2);
    const brief = frames(held, 'fist', undefined, 5);
    const opened = frames(brief, 'open', undefined, 2);
    assert.ok(!kinds(opened).includes('grab'));
    assert.ok(!kinds(opened).includes('release'));
    assert.equal(opened.state.phase, 'tracking');
    assert.equal(opened.state.dwellProgress, 0);
  });
});

describe('losing the hand', () => {
  it('cancels a grab once the hand has been gone long enough', () => {
    const blink = frames(grabbedRun(), null, undefined, 2);
    assert.equal(blink.state.phase, 'grabbing', 'a couple of missed frames are tolerated');
    assert.ok(!kinds(blink).includes('cancel'));
    const gone = frames(blink, null, undefined, LOST_FRAMES);
    assert.deepEqual(kinds(gone), ['cancel', 'hover-end']);
    assert.equal(gone.state.phase, 'idle');
  });

  it('goes idle quietly when nothing was grabbed', () => {
    const tracking = frames(start(), 'open', undefined, 3);
    const gone = frames(tracking, null, undefined, LOST_FRAMES + 1);
    assert.equal(gone.state.phase, 'idle');
    assert.deepEqual(kinds(gone), ['hover-end']);
  });

  it('disabling mid-grab cancels and resets', () => {
    const result = reduceHandGesture(grabbedRun().state, { type: 'disable' }, config);
    assert.deepEqual(
      result.effects.map((effect) => effect.type),
      ['cancel', 'hover-end']
    );
    assert.deepEqual(result.state, initialGestureState());
  });
});

describe('the defaults', () => {
  it('ask for a deliberate dwell before anything moves', () => {
    assert.ok(DEFAULT_GESTURE_CONFIG.dwellMs >= 1500 && DEFAULT_GESTURE_CONFIG.dwellMs <= 2000);
    assert.ok(DEFAULT_GESTURE_CONFIG.confirmFrames >= 2);
  });
});
