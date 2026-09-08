import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

/**
 * Structural guard for the one rule the workspace must never break: a voice
 * call is docked *inside* the dashboard, so no LiveKit session state may gate
 * whether the dashboard renders. The shell components therefore must not read
 * the session at all; only the dock and the "running work" widget do, and they
 * render inside frames that stay mounted either way.
 */

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = (path: string) => readFileSync(join(ROOT, path), 'utf8');

const SESSION_READS = ['useSessionContext', 'useVoiceAssistant', 'useRoomContext', 'isConnected'];

describe('persistent workspace', () => {
  it('never gates the dashboard on LiveKit session state', () => {
    for (const path of [
      'components/dashboard/workspace.tsx',
      'components/dashboard/monitor-dashboard.tsx',
      'components/dashboard/widget-frame.tsx',
      'hooks/useDashboardLayout.ts',
    ]) {
      const source = read(path);
      for (const read of SESSION_READS) {
        assert.ok(!source.includes(read), `${path} must not read session state via ${read}`);
      }
    }
  });

  it('mounts the dashboard and the voice dock as siblings, unconditionally', () => {
    const workspace = read('components/dashboard/workspace.tsx');
    assert.match(workspace, /<MonitorDashboard\b/);
    assert.match(workspace, /<VoiceDock\b/);
    // The only thing allowed between "layout is ready" and the grid is the layout itself.
    assert.match(workspace, /layout\.ready \? \(\s*<MonitorDashboard/);
  });

  it('is what the app renders instead of a call-gated view switch', () => {
    const app = read('components/app/app.tsx');
    assert.match(app, /<Workspace\b/);
    assert.ok(!app.includes('isConnected'), 'app shell must not branch on the call');
    for (const removed of ['view-controller', 'welcome-view', 'session-view']) {
      assert.ok(!app.includes(removed), `${removed} must no longer be wired in`);
    }
  });

  it('builds the movable grid from the existing dependency set only', () => {
    const pkg = JSON.parse(read('package.json')) as {
      dependencies: Record<string, string>;
      devDependencies: Record<string, string>;
    };
    const declared = new Set([
      ...Object.keys(pkg.dependencies),
      ...Object.keys(pkg.devDependencies),
    ]);
    const sources = [
      'components/dashboard/monitor-dashboard.tsx',
      'components/dashboard/widget-frame.tsx',
      'components/dashboard/workspace.tsx',
      'components/dashboard/voice-dock.tsx',
      'lib/dashboard/layout.ts',
    ];
    for (const path of sources) {
      const imports = [...read(path).matchAll(/from '([^'.@][^']*|@[^/']+\/[^/']+)'/g)].map(
        (m) => m[1]
      );
      for (const spec of imports) {
        const pkgName = spec.startsWith('@')
          ? spec.split('/').slice(0, 2).join('/')
          : spec.split('/')[0];
        if (pkgName === 'react' || pkgName === 'next') continue;
        assert.ok(
          declared.has(pkgName),
          `${path} imports ${pkgName}, which is not a declared dependency`
        );
      }
    }
    assert.ok(!declared.has('react-grid-layout'));
    assert.ok(![...declared].some((name) => name.startsWith('@dnd-kit')));
  });
});
