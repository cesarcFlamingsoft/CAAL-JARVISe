import assert from 'node:assert/strict';
import { test } from 'node:test';
import { VisionCommandHandler } from './bridge.ts';

const binding = { user: 'u', room: 'r', participant: 'p', agent: 'a', epoch: 'e'.repeat(32) };
const command = {
  action: 'vision.analyze',
  user: 'u',
  room: 'r',
  epoch: binding.epoch,
  seq: 1,
  expires: Date.now() + 30000,
};
function harness() {
  let captures = 0;
  const sent: unknown[] = [];
  const handler = new VisionCommandHandler(
    binding,
    async () => {
      captures++;
      return 'A mug.';
    },
    async (p) => {
      sent.push(p);
    }
  );
  return { handler, sent, captures: () => captures };
}
test('validated command captures once; duplicate, wrong sender/user/session and expired do nothing', async () => {
  const h = harness();
  for (const patch of [
    { user: 'other' },
    { room: 'other' },
    { epoch: 'old' },
    { expires: 0 },
    { action: 'other' },
    { prompt: 'injected' },
  ]) {
    await h.handler.receive({ ...command, ...patch }, 'a');
  }
  await h.handler.receive(command, 'wrong');
  assert.equal(h.captures(), 0);
  await h.handler.receive(command, 'a');
  await h.handler.receive(command, 'a');
  assert.equal(h.captures(), 1);
  assert.equal(h.sent.length, 1);
});
test('closed preview returns honest unavailable; closed session does nothing', async () => {
  const sent: unknown[] = [];
  const h = new VisionCommandHandler(binding, null, async (p) => {
    sent.push(p);
  });
  await h.receive(command, 'a');
  assert.equal((sent[0] as { description: string }).description, '');
  h.close();
  await h.receive({ ...command, seq: 2 }, 'a');
  assert.equal(sent.length, 1);
});
test('cancel/close aborts in-flight analysis and discards its result', async () => {
  let finish!: (v: string) => void;
  let signal!: AbortSignal;
  const sent: unknown[] = [];
  const h = new VisionCommandHandler(
    binding,
    (s) => {
      signal = s;
      return new Promise((r) => {
        finish = r;
      });
    },
    async (p) => {
      sent.push(p);
    }
  );
  const pending = h.receive(command, 'a');
  h.close();
  assert.equal(signal.aborted, true);
  finish('private');
  await pending;
  assert.deepEqual(sent, []);
});
test('a new turn cancels the old command without another capture', async () => {
  let finish!: (v: string) => void;
  let signal!: AbortSignal;
  const sent: unknown[] = [];
  const h = new VisionCommandHandler(
    binding,
    (s) => {
      signal = s;
      return new Promise((r) => {
        finish = r;
      });
    },
    async (p) => {
      sent.push(p);
    }
  );
  const pending = h.receive(command, 'a');
  await h.receive({ ...command, action: 'vision.cancel' }, 'a');
  assert.equal(signal.aborted, true);
  finish('private');
  await pending;
  assert.deepEqual(sent, []);
});

test('a cancellation arriving before its command prevents later capture', async () => {
  const h = harness();
  await h.handler.receive({ ...command, action: 'vision.cancel' }, 'a');
  await h.handler.receive(command, 'a');
  assert.equal(h.captures(), 0);
});
