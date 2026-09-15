import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import test from 'node:test';

test('BFF and browser share a bounded TTS contract with no credentials or invented status', async () => {
  assert.ok(existsSync(new URL('./contract.ts', import.meta.url)));
  const { parseTtsView } = await import('./contract.ts');
  const native = {provider:'voicebox', source:'personal', qwen_configured:true, qwen_voice:'jarvis-designed',
    applies_to:'new_sessions', voicebox_status:'api_verified', can_configure:false,
    profile_id:'chosen', engine:'kokoro', model_size:'1.7B', credential:'private', endpoint:'http://private:8000'};
  const view = parseTtsView(native);
  assert.ok(view);
  assert.equal(view.provider, 'voicebox');
  assert.deepEqual(parseTtsView(JSON.parse(JSON.stringify(view))), view);
  assert.ok(!JSON.stringify(view).includes('private'));
  for (const key of ['source', 'voicebox_status', 'can_configure', 'applies_to']) {
    assert.equal(parseTtsView({...native, [key]: undefined}), null);
  }
  assert.equal(parseTtsView({...native, voicebox_status:'ready'}), null);
  assert.equal(parseTtsView({...native, profile_id:null}), null);
});
