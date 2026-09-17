import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const source = readFileSync(
  new URL('../../components/dashboard/voice-dock.tsx', import.meta.url),
  'utf8'
);

test('disconnected voice UI describes opt-in local browser standby truthfully', () => {
  assert.match(source, /Browser standby is off by default\./);
  assert.match(source, /matching local keyword model/i);
  assert.match(source, /wake control/i);
  assert.doesNotMatch(source, /Friday wake up/i);
  assert.doesNotMatch(source, /cloud speech recognition/i);
});
