import assert from 'node:assert/strict';
import { test } from 'node:test';
import { voiceStatus } from './activity.ts';

test('transport reconnecting takes precedence over old agent speaking state', () => {
  assert.deepEqual(
    voiceStatus({
      isConnected: false,
      connecting: false,
      connectionState: 'reconnecting',
      agentState: 'speaking',
    }),
    { label: 'Reconnecting voice…', tone: 'busy' }
  );
});
test('disconnected transport never reports a stale listening agent as live', () => {
  assert.equal(
    voiceStatus({
      isConnected: true,
      connecting: false,
      connectionState: 'disconnected',
      agentState: 'listening',
    }).tone,
    'idle'
  );
});
