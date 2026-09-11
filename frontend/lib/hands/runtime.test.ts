import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import type { HandLandmarkProvider } from './landmarks.ts';
import { HandRuntime, type HandRuntimeSnapshot, describeRuntimeError } from './runtime.ts';

/**
 * The runtime owns the async life of the landmark model: created, loading,
 * then ready with a provider or failed with one plain sentence. These tests
 * drive it with a loader the test controls, so cancellation, late arrivals,
 * retries and timeouts are exercised without a browser or a model.
 */

interface FakeProvider extends HandLandmarkProvider {
  disposed: number;
}

function fakeProvider(id = 'fake'): FakeProvider {
  const provider: FakeProvider = {
    id,
    disposed: 0,
    detect: () => null,
    dispose() {
      provider.disposed += 1;
    },
  };
  return provider;
}

interface Attempt {
  signal: AbortSignal;
  resolve: (provider: HandLandmarkProvider) => void;
  reject: (error: unknown) => void;
}

/** Every call to load is recorded and settles only when the test says so. */
function controlledLoader() {
  const attempts: Attempt[] = [];
  const load = (signal: AbortSignal) =>
    new Promise<HandLandmarkProvider>((resolve, reject) => {
      attempts.push({ signal, resolve, reject });
    });
  return { load, attempts };
}

const wait = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

describe('the hand runtime', () => {
  it('loads from the moment it is created and becomes ready with the loaded provider', async () => {
    const seen: HandRuntimeSnapshot[] = [];
    const loader = controlledLoader();
    const runtime = new HandRuntime({ load: loader.load, onChange: (s) => seen.push(s) });
    assert.equal(runtime.snapshot().status, 'loading');
    assert.equal(runtime.snapshot().provider, null);
    assert.equal(loader.attempts.length, 1);

    const provider = fakeProvider();
    loader.attempts[0].resolve(provider);
    await runtime.settled();
    assert.equal(runtime.snapshot().status, 'ready');
    assert.equal(runtime.snapshot().provider, provider);
    assert.equal(runtime.snapshot().detail, null);
    assert.equal(seen.at(-1)?.status, 'ready');
    runtime.dispose();
  });

  it('turns a load failure into a failed state with a plain reason and no provider', async () => {
    const loader = controlledLoader();
    const runtime = new HandRuntime({ load: loader.load });
    loader.attempts[0].reject(new Error('Failed to fetch model: /hands/models/x.task (404)'));
    await runtime.settled();
    const snapshot = runtime.snapshot();
    assert.equal(snapshot.status, 'failed');
    assert.equal(snapshot.provider, null);
    assert.match(snapshot.detail ?? '', /could not be loaded/i);
    assert.ok(!(snapshot.detail ?? '').includes('/hands'), 'no paths in the sentence shown');
    runtime.dispose();
  });

  it('retries only after a failure, and only one load at a time', async () => {
    const loader = controlledLoader();
    const runtime = new HandRuntime({ load: loader.load });
    runtime.retry();
    assert.equal(loader.attempts.length, 1, 'retry while loading starts nothing');

    loader.attempts[0].reject(new Error('boom'));
    await runtime.settled();
    assert.equal(runtime.snapshot().status, 'failed');

    runtime.retry();
    assert.equal(loader.attempts.length, 2);
    assert.equal(runtime.snapshot().status, 'loading');
    runtime.retry();
    assert.equal(loader.attempts.length, 2, 'a second retry while loading starts nothing');

    const provider = fakeProvider();
    loader.attempts[1].resolve(provider);
    await runtime.settled();
    assert.equal(runtime.snapshot().status, 'ready');
    runtime.retry();
    assert.equal(loader.attempts.length, 2, 'retry while ready starts nothing');
    runtime.dispose();
  });

  it('aborts a load when disposed, and closes a provider that arrives anyway', async () => {
    const seen: HandRuntimeSnapshot[] = [];
    const loader = controlledLoader();
    const runtime = new HandRuntime({ load: loader.load, onChange: (s) => seen.push(s) });
    runtime.dispose();
    assert.equal(loader.attempts[0].signal.aborted, true);
    const before = seen.length;

    const late = fakeProvider('late');
    loader.attempts[0].resolve(late);
    await wait(0);
    assert.equal(late.disposed, 1, 'a provider nobody can use is closed at once');
    assert.equal(runtime.snapshot().provider, null);
    assert.equal(seen.length, before, 'nothing is reported after dispose');
  });

  it('ignores a failure that arrives after it was disposed', async () => {
    const seen: HandRuntimeSnapshot[] = [];
    const loader = controlledLoader();
    const runtime = new HandRuntime({ load: loader.load, onChange: (s) => seen.push(s) });
    runtime.dispose();
    const before = seen.length;
    loader.attempts[0].reject(new Error('too late'));
    await wait(0);
    assert.equal(seen.length, before);
    assert.notEqual(runtime.snapshot().status, 'ready');
  });

  it('closes the model when disposed while ready', async () => {
    const loader = controlledLoader();
    const runtime = new HandRuntime({ load: loader.load });
    const provider = fakeProvider();
    loader.attempts[0].resolve(provider);
    await runtime.settled();
    runtime.dispose();
    assert.equal(provider.disposed, 1);
    assert.equal(runtime.snapshot().provider, null);
  });

  it('gives up on a load that takes too long, then can retry', async () => {
    const loader = controlledLoader();
    const runtime = new HandRuntime({ load: loader.load, timeoutMs: 10 });
    await wait(40);
    assert.equal(runtime.snapshot().status, 'failed');
    assert.match(runtime.snapshot().detail ?? '', /too long/i);
    assert.equal(loader.attempts[0].signal.aborted, true);

    const late = fakeProvider('late');
    loader.attempts[0].resolve(late);
    await wait(0);
    assert.equal(late.disposed, 1);
    assert.equal(runtime.snapshot().status, 'failed');

    runtime.retry();
    assert.equal(loader.attempts.length, 2);
    const provider = fakeProvider();
    loader.attempts[1].resolve(provider);
    await runtime.settled();
    assert.equal(runtime.snapshot().status, 'ready');
    assert.equal(runtime.snapshot().provider, provider);
    runtime.dispose();
  });
});

describe('naming a runtime failure', () => {
  const plain = (detail: string) => {
    assert.ok(detail.length > 0);
    assert.ok(!/https?:|\/hands|\(\d+\)/.test(detail), 'no urls, paths or codes: ' + detail);
  };

  it('blames this server when a model or runtime file did not arrive', () => {
    const fromFetch = describeRuntimeError(new Error('Failed to fetch model: /x.task (404)'));
    assert.match(fromFetch, /loaded from this server/i);
    plain(fromFetch);
    const fromScript = describeRuntimeError({ type: 'error' });
    assert.match(fromScript, /loaded from this server/i);
    const fromFactory = describeRuntimeError(new Error('ModuleFactory not set.'));
    assert.match(fromFactory, /loaded from this server/i);
  });

  it('blames the browser when WebAssembly cannot run the model', () => {
    const compile = Object.assign(new Error('bad import'), { name: 'CompileError' });
    assert.match(describeRuntimeError(compile), /browser cannot run/i);
    const instantiate = describeRuntimeError(new Error('WebAssembly.instantiate failed'));
    assert.match(instantiate, /browser cannot run/i);
  });

  it('says the model could not start when the graphics stack refused it', () => {
    const detail = describeRuntimeError(new Error('Unable to obtain required WebGL resource: x'));
    assert.match(detail, /could not start/i);
    plain(detail);
  });

  it('has a plain sentence for anything else', () => {
    plain(describeRuntimeError('nonsense'));
    plain(describeRuntimeError(undefined));
  });
});
