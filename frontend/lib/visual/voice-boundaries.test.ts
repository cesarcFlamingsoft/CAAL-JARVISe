import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const read = (path: string) => readFileSync(new URL('../../' + path, import.meta.url), 'utf8');

test('voice commands are gated by current user, Personal mode, room, participant and live preview', () => {
  const dock = read('components/dashboard/voice-dock.tsx');
  assert.match(
    dock,
    /!session\.isConnected[\s\S]*!room[\s\S]*!agent[\s\S]*!userId[\s\S]*companyPrivate[\s\S]*!visionOpen[\s\S]*!cameraLive/
  );
  assert.match(dock, /visionOpen && cameraLive/);
  assert.match(dock, /participant\?\.identity !== agent.identity/);
  assert.match(dock, /destinationIdentities: \[agent.identity\]/);
  assert.match(dock, /handler.close\(\)/);
  assert.match(dock, /RoomEvent.Disconnected/);
  assert.match(dock, /RoomEvent.Reconnecting/);
  assert.ok(!/analyzeCameraView|getUserMedia|\.acquire\(/.test(dock));
});

test('visual bridge never acquires cameras, stores results, sends chat, or exposes model tools', () => {
  for (const path of [
    'lib/visual/bridge.ts',
    'lib/visual/client.ts',
    'components/dashboard/visual-analyzer.tsx',
  ]) {
    const source = read(path);
    assert.ok(
      !/getUserMedia|\.acquire\(|localStorage|sessionStorage|indexedDB|console\.|sendText|sendChatMessage|generateReply/.test(
        source
      ),
      path
    );
  }
  const analyzer = read('components/dashboard/visual-analyzer.tsx');
  assert.match(analyzer, /expectedUser: userId/);
  assert.match(analyzer, /track.readyState === 'live'/);
  assert.match(analyzer, /analysisRef.current = null/);
  assert.match(analyzer, /request.current\?\.abort\(\)/);
  const bridge = read('lib/visual/bridge.ts');
  assert.match(bridge, /this.binding = null/);
  assert.match(bridge, /this.analyze = null/);
  assert.match(bridge, /this.send = null/);
});

test('voice result is transient rather than retained in the preview after the voice turn', () => {
  const analyzer = read('components/dashboard/visual-analyzer.tsx');
  assert.match(analyzer, /if \(!externalSignal\) setResult\(description\)/);
  assert.match(analyzer, /if \(request.current === controller\) request.current = null/);
});
