import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const route = () => readFileSync(new URL('../../app/api/tts/route.ts', import.meta.url), 'utf8');

test('TTS selection uses authenticated reads and user CSRF-protected writes', () => {
  const source = route();
  assert.match(source, /await requireUser\(req\)/);
  assert.doesNotMatch(source, /await requireAdmin\(req\)/);
  assert.match(source, /guardMutation\(req, auth.config, auth.user.userId\)/);
  assert.match(source, /callAsUser\(auth.config, auth.user.userId, '\/users\/me\/tts'/);
  assert.match(source, /noStoreJson/);
  assert.doesNotMatch(source, /process\.env|console\./);
});
