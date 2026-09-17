import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = (path: string) => readFileSync(join(ROOT, path), 'utf8');

describe('visual analysis source boundaries', () => {
  it('owns one camera at the workspace boundary with independent explicit mobile controls', () => {
    const workspace = read('components/dashboard/workspace.tsx');
    const dock = read('components/dashboard/hands-dock.tsx');
    const analyzer = read('components/dashboard/visual-analyzer.tsx');
    const header = read('components/dashboard/workspace-header.tsx');
    assert.equal((workspace.match(/useCamera\(\)/g) ?? []).length, 1);
    assert.match(workspace, /visionOpen && visionAllowed && \(\s*<VisualAnalyzer/);
    assert.match(workspace, /Boolean\(signedIn\) && !companyPrivate/);
    assert.match(workspace, /if \(open && !visionAllowed\) return/);
    assert.match(workspace, /camera\.acquire\('hands'\)/);
    assert.match(workspace, /camera\.acquire\('vision'\)/);
    assert.match(workspace, /camera\.release\('hands'\)/);
    assert.match(workspace, /camera\.release\('vision'\)/);
    assert.match(workspace, /camera=\{camera\}/);
    assert.match(workspace, /stream=\{camera\.stream\}/);
    assert.ok(!/VisualAnalyzer|useCamera\(|getUserMedia|\.clone\(/.test(dock));
    assert.match(dock, /video=\{videoRef\}/);
    assert.match(dock, /video\.srcObject = \(stream as MediaStream/);
    assert.match(dock, /disabled=\{visionOpen/);
    assert.match(analyzer, /Analyze camera view/);
    assert.match(analyzer, /Close Vision/);
    assert.match(analyzer, /video\.srcObject =/);
    assert.match(analyzer, /event\.isTrusted/);
    assert.match(analyzer, /abort\(\)/);
    assert.equal((analyzer.match(/await analyzeCameraView\(/g) ?? []).length, 1);
    assert.ok(!/useEffect\([\s\S]*?analyzeCameraView[\s\S]*?\}, \[/.test(analyzer));
    const visionButton = header.slice(
      header.indexOf('aria-label="Vision"'),
      header.indexOf('Reset layout')
    );
    assert.match(visionButton, /event\.isTrusted/);
    assert.match(visionButton, /disabled=\{!visionAllowed\}/);
    assert.ok(!/className="[^"]*\bhidden\b/.test(visionButton.split('</Button>')[0]));
    assert.match(analyzer, /gesture tracking remains browser-local/i);
  });

  it('does not own a stream or depend on the gesture pipeline', () => {
    const paths = [
      'components/dashboard/visual-analyzer.tsx',
      'lib/visual/capture.ts',
      'lib/visual/client.ts',
    ];
    for (const path of paths) {
      assert.ok(existsSync(join(ROOT, path)), path);
      const source = read(path);
      assert.ok(!source.includes('getUserMedia'), path);
      assert.ok(!/from ['"][^'"]*(?:hands|gesture|mediapipe|landmark)/i.test(source), path);
      assert.ok(!/localStorage|sessionStorage|indexedDB|caches\.|console\./i.test(source), path);
      assert.ok(!/MediaRecorder|captureStream|setInterval/.test(source), path);
    }
  });

  it('uses only the server-owned BFF endpoint and has no frame preview/history', () => {
    const client = read('lib/visual/client.ts');
    const component = read('components/dashboard/visual-analyzer.tsx');
    assert.match(client, /['"]\/api\/visual\/analyze['"]/);
    assert.match(client, /['"]\/api\/auth\/me['"]/);
    assert.ok(!/ollama|hermes|cloudflare/i.test(client));
    assert.ok(!/<img|createObjectURL|data:image/i.test(component));
  });
});
