import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { type HandStatusInput, handStatus } from './status.ts';

function input(overrides: Partial<HandStatusInput> = {}): HandStatusInput {
  return {
    camera: 'live',
    cameraDetail: null,
    runtime: 'ready',
    runtimeDetail: null,
    phase: 'idle',
    target: null,
    ...overrides,
  };
}

describe('hand control status', () => {
  it('names every camera problem and carries the browser detail', () => {
    for (const camera of [
      'denied',
      'no-camera',
      'busy',
      'unavailable',
      'ended',
      'error',
    ] as const) {
      const status = handStatus(input({ camera, cameraDetail: 'why' }));
      assert.equal(status.tone, 'error', camera);
      assert.equal(status.detail, 'why', camera);
      assert.ok(status.label.length > 0, camera);
    }
    assert.equal(handStatus(input({ camera: 'unsupported' })).tone, 'error');
    assert.equal(handStatus(input({ camera: 'requesting' })).tone, 'busy');
    assert.equal(handStatus(input({ camera: 'idle' })).tone, 'idle');
  });

  it('says the model is still loading instead of claiming to watch for a hand', () => {
    const loading = handStatus(input({ runtime: 'loading' }));
    assert.equal(loading.tone, 'busy');
    assert.match(loading.label, /loading/i);
    assert.match(loading.detail ?? '', /browser/i);
    const tracking = handStatus(
      input({ runtime: 'loading', phase: 'tracking', target: { title: 'Weather', movable: true } })
    );
    assert.equal(tracking.tone, 'busy');
    assert.doesNotMatch(tracking.label, /tracking/i);
  });

  it('puts the camera first while it is still being asked for', () => {
    const asking = handStatus(input({ camera: 'requesting', runtime: 'loading' }));
    assert.equal(asking.tone, 'busy');
    assert.match(asking.label, /camera/i);
    assert.doesNotMatch(asking.label, /loading/i);
  });

  it('reports a failed model load as an error with its reason', () => {
    const failed = handStatus(input({ runtime: 'failed', runtimeDetail: 'why' }));
    assert.equal(failed.tone, 'error');
    assert.match(failed.label, /recogni/i);
    assert.equal(failed.detail, 'why');
    assert.equal(handStatus(input({ runtime: 'failed' })).tone, 'error');
  });

  it('follows the gesture once a hand is being tracked', () => {
    assert.equal(handStatus(input()).tone, 'live');
    assert.match(handStatus(input()).label, /hand/i);

    const over = handStatus(
      input({ phase: 'tracking', target: { title: 'Weather', movable: true } })
    );
    assert.equal(over.tone, 'live');
    assert.match(over.detail ?? '', /Weather/);

    const holding = handStatus(
      input({ phase: 'holding', target: { title: 'Weather', movable: true } })
    );
    assert.match(holding.label, /hold/i);

    const grabbing = handStatus(
      input({ phase: 'grabbing', target: { title: 'Weather', movable: true } })
    );
    assert.match(grabbing.label, /Moving Weather/);

    const stuck = handStatus(
      input({ phase: 'grabbing', target: { title: 'Weather', movable: false } })
    );
    assert.match(stuck.label, /cannot be moved/i);
  });
});
