import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRefreshGate, reduceFeed } from './refresh.ts';

test('refresh coalesces concurrent triggers, bounds focus storms, and pauses hidden tabs', async () => {
  let now = 10000,
    visible = true,
    calls = 0;
  let finish: () => void = () => {};
  const gate = createRefreshGate(
    () => {
      calls++;
      return new Promise<void>((r) => {
        finish = r;
      });
    },
    () => visible,
    () => now
  );
  const first = gate.refresh();
  void gate.refresh();
  void gate.refresh();
  assert.equal(calls, 1);
  finish();
  await first;
  await gate.refresh();
  assert.equal(calls, 1);
  now += 5000;
  visible = false;
  await gate.refresh();
  assert.equal(calls, 1);
  visible = true;
  const next = gate.refresh();
  assert.equal(calls, 2);
  finish();
  await next;
});
test('failed refresh retains explicitly stale data; authorization failure erases it', () => {
  const previous = { status: 'ready' as const, data: { items: ['fixture'] }, updatedAt: 100 };
  assert.deepEqual(reduceFeed(previous, { status: 'error', code: 'network' }), {
    ...previous,
    stale: true,
    refreshing: false,
  });
  assert.deepEqual(reduceFeed(previous, { status: 'unauthorized', code: 'http_401' }), {
    status: 'unauthorized',
    code: 'http_401',
  });
  assert.deepEqual(reduceFeed(previous, { status: 'ready', data: { items: [] }, updatedAt: 200 }), {
    status: 'ready',
    data: { items: [] },
    updatedAt: 200,
  });
});
