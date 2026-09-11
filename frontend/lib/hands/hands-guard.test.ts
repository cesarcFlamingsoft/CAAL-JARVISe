import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import {
  HAND_MODEL_BYTES,
  HAND_MODEL_PATH,
  HAND_MODEL_SHA256,
  HAND_RUNTIME_FILES,
  HAND_RUNTIME_PACKAGE,
  HAND_RUNTIME_PATH,
  HAND_RUNTIME_VERSION,
} from './assets.ts';

/**
 * Structural guard for hand control. The React files cannot run under
 * node --test, so what is checked here is the posture they must keep: camera
 * frames and landmarks never leave the browser, the camera is only ever
 * opened through the one tested session, recognition runs on one pinned
 * runtime and one pinned model that this origin serves itself, gestures are
 * opt-in, keyboard and pointer remain complete controls, and nothing claims
 * to recognise a hand before the model is actually running.
 *
 * Model provenance: MediaPipe hand_landmarker, float16, version 1, from
 * storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/
 * (Apache-2.0), pinned here by size and SHA-256.
 */

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = (path: string) => readFileSync(join(ROOT, path), 'utf8');
const readBytes = (path: string) => readFileSync(join(ROOT, path));

const LIB = [
  'lib/hands/assets.ts',
  'lib/hands/landmarks.ts',
  'lib/hands/gesture.ts',
  'lib/hands/camera-session.ts',
  'lib/hands/runtime.ts',
  'lib/hands/mediapipe.ts',
  'lib/hands/status.ts',
  'lib/hands/surface.ts',
];
const CAMERA_HOOK = 'hooks/useCamera.ts';
const RUNTIME_HOOK = 'hooks/useHandRuntime.ts';
const MEDIAPIPE = 'lib/hands/mediapipe.ts';
const DOCK = 'components/dashboard/hands-dock.tsx';
const LAYER = 'components/dashboard/hand-layer.tsx';
const CURSOR = 'components/dashboard/hand-cursor.tsx';
const GESTURE = 'lib/hands/gesture.ts';
const DASHBOARD = 'components/dashboard/monitor-dashboard.tsx';
const FRAME = 'components/dashboard/widget-frame.tsx';
const WORKSPACE = 'components/dashboard/workspace.tsx';
const HEADER = 'components/dashboard/workspace-header.tsx';
const ALL = [...LIB, CAMERA_HOOK, RUNTIME_HOOK, DOCK, LAYER, CURSOR];

const RUNTIME_DIR = 'node_modules/' + HAND_RUNTIME_PACKAGE;
const SERVED_RUNTIME = 'public' + HAND_RUNTIME_PATH;
const SERVED_MODEL = 'public' + HAND_MODEL_PATH;

const LEAKS = [
  'fetch(',
  'XMLHttpRequest',
  'WebSocket',
  'sendBeacon',
  'postMessage',
  'localStorage',
  'sessionStorage',
  'indexedDB',
  'console.',
  'toDataURL',
  'toBlob',
  'captureStream',
  'MediaRecorder',
  'ImageCapture',
  'Math.random',
];

const literal = (text: string) => text.split('.').join('\\.');

describe('hand control keeps the camera in the browser', () => {
  it('exists', () => {
    for (const path of ALL) assert.ok(existsSync(join(ROOT, path)), path + ' is missing');
  });

  it('never uploads, stores or logs frames or landmarks', () => {
    for (const path of ALL) {
      const source = read(path);
      for (const leak of LEAKS) {
        assert.ok(!source.includes(leak), path + ' must not use ' + leak);
      }
      assert.ok(!/https?:\/\//i.test(source), path + ' must not name a URL');
    }
  });

  it('has no server endpoint for anything camera related', () => {
    for (const dir of ['app/api/hands', 'app/api/camera', 'app/api/dashboard/hands']) {
      assert.ok(!existsSync(join(ROOT, dir)), dir + ' must not exist');
    }
  });

  it('opens the camera only through the tested session, with no audio', () => {
    const session = read('lib/hands/camera-session.ts');
    assert.match(session, /getUserMedia\(/);
    assert.match(session, /audio: false/);
    assert.match(session, /track\.stop\(\)/);
    assert.match(session, /'devicechange'/);
    for (const path of ALL.filter((p) => p !== 'lib/hands/camera-session.ts')) {
      assert.ok(!read(path).includes('getUserMedia('), path + ' must go through CameraSession');
    }
    assert.match(read(CAMERA_HOOK), /\.dispose\(\)/, 'the hook must dispose the session');
  });

  it('declares every package it imports', () => {
    const pkg = JSON.parse(read('package.json')) as {
      dependencies: Record<string, string>;
      devDependencies: Record<string, string>;
    };
    const declared = new Set([
      ...Object.keys(pkg.dependencies),
      ...Object.keys(pkg.devDependencies),
    ]);
    for (const path of ALL) {
      const source = read(path);
      const specs = [
        ...[...source.matchAll(/from '([^'.@][^']*|@[^/']+\/[^/']+)'/g)].map((m) => m[1]),
        ...[...source.matchAll(/import\('([^'.][^']*)'\)/g)].map((m) => m[1]),
      ];
      for (const spec of specs) {
        const pkgName = spec.startsWith('@')
          ? spec.split('/').slice(0, 2).join('/')
          : spec.split('/')[0];
        if (pkgName === 'react' || pkgName === 'next') continue;
        assert.ok(declared.has(pkgName), path + ' imports undeclared ' + pkgName);
      }
    }
  });
});

describe('hand recognition runs on one pinned runtime served by this origin', () => {
  it('pins exactly one recognition dependency, in package.json and both lockfiles', () => {
    const pkg = JSON.parse(read('package.json')) as {
      dependencies: Record<string, string>;
      devDependencies: Record<string, string>;
    };
    assert.match(HAND_RUNTIME_VERSION, /^\d+\.\d+\.\d+$/, 'an exact version, not a range');
    assert.equal(pkg.dependencies[HAND_RUNTIME_PACKAGE], HAND_RUNTIME_VERSION);
    for (const name of [...Object.keys(pkg.dependencies), ...Object.keys(pkg.devDependencies)]) {
      if (name === HAND_RUNTIME_PACKAGE) continue;
      assert.ok(!/mediapipe|tensorflow|onnx/i.test(name), name + ' was added');
    }
    const installed = JSON.parse(read(RUNTIME_DIR + '/package.json')) as { version: string };
    assert.equal(installed.version, HAND_RUNTIME_VERSION, 'node_modules holds the pinned runtime');
    const pnpmEntry = new RegExp(
      "'" +
        literal(HAND_RUNTIME_PACKAGE) +
        "':\\s+specifier: " +
        literal(HAND_RUNTIME_VERSION) +
        '\\s+version: ' +
        literal(HAND_RUNTIME_VERSION)
    );
    assert.match(read('pnpm-lock.yaml'), pnpmEntry, 'pnpm-lock.yaml pins the runtime');
    const npmLock = JSON.parse(read('package-lock.json')) as {
      packages: Record<string, { version?: string }>;
    };
    assert.equal(
      npmLock.packages['node_modules/' + HAND_RUNTIME_PACKAGE]?.version,
      HAND_RUNTIME_VERSION,
      'package-lock.json pins the runtime'
    );
  });

  it('serves the pinned runtime from this origin, byte for byte', () => {
    assert.ok(HAND_RUNTIME_PATH.startsWith('/') && !HAND_RUNTIME_PATH.includes(':'));
    for (const variant of ['vision_wasm_internal', 'vision_wasm_nosimd_internal']) {
      for (const ext of ['.js', '.wasm']) {
        assert.ok(HAND_RUNTIME_FILES.includes(variant + ext), variant + ext + ' must be served');
      }
    }
    for (const file of HAND_RUNTIME_FILES) {
      assert.ok(existsSync(join(ROOT, SERVED_RUNTIME, file)), file + ' is not served');
      const served = readBytes(SERVED_RUNTIME + '/' + file);
      assert.ok(served.length > 0, file + ' is empty');
      assert.ok(
        served.equals(readBytes(RUNTIME_DIR + '/wasm/' + file)),
        file + ' differs from the pinned package'
      );
    }
  });

  it('serves the pinned model from this origin, unchanged', () => {
    assert.ok(HAND_MODEL_PATH.startsWith('/') && !HAND_MODEL_PATH.includes(':'));
    assert.ok(existsSync(join(ROOT, SERVED_MODEL)), SERVED_MODEL + ' is missing');
    const model = readBytes(SERVED_MODEL);
    assert.equal(model.length, HAND_MODEL_BYTES);
    assert.equal(createHash('sha256').update(model).digest('hex'), HAND_MODEL_SHA256);
  });

  it('ships a runtime that talks to no server', () => {
    const bundle = read(RUNTIME_DIR + '/vision_bundle.mjs');
    assert.ok(!/https?:\/\//i.test(bundle), 'the runtime bundle must name no URL');
    for (const leak of ['sendBeacon', 'XMLHttpRequest', 'WebSocket', 'localStorage', 'indexedDB']) {
      assert.ok(!bundle.includes(leak), 'the runtime bundle must not use ' + leak);
    }
    assert.ok(!bundle.includes('setInterval'), 'the runtime bundle must not report on a timer');
    const fetches = bundle.match(/\bfetch\(/g) ?? [];
    assert.ok(fetches.length <= 2, 'the runtime may only fetch the assets it is pointed at');
  });

  it('loads the runtime and model only from the pinned same-origin paths', () => {
    const source = read(MEDIAPIPE);
    assert.match(source, /import\('@mediapipe\/tasks-vision'\)/, 'loaded on demand, not at boot');
    assert.match(source, /forVisionTasks\(HAND_RUNTIME_PATH\)/);
    assert.match(source, /modelAssetPath: HAND_MODEL_PATH/);
    assert.match(source, /runningMode: 'VIDEO'/);
    assert.match(source, /numHands: 1/);
    assert.match(source, /createLandmarkerProvider\(/);
    assert.ok(!source.includes('modelAssetBuffer'));
    assert.ok(!/jsdelivr|unpkg|cdn|googleapis/i.test(source), 'no remote runtime');
    assert.ok(!read('lib/hands/runtime.ts').includes('@mediapipe'), 'the runtime state is generic');
    assert.ok(!/^import /m.test(read('lib/hands/assets.ts')), 'the asset pins import nothing');
    assert.ok(
      read('.prettierignore').split('\n').includes('public/hands/'),
      'assets stay verbatim'
    );
    for (const ignore of ['.gitignore', '.dockerignore']) {
      assert.ok(!/wasm|\.task|public/.test(read(ignore)), ignore + ' must not drop the runtime');
    }
  });
});

describe('hand control is honest about readiness', () => {
  it('recognises nothing until the local runtime is ready, and says so', () => {
    const dock = read(DOCK);
    assert.match(dock, /useHandRuntime\(\)/, 'the dock owns the runtime through the hook');
    assert.match(dock, /handStatus\(/);
    assert.match(dock, /runtime: runtime\.status/);
    assert.match(dock, /provider=\{runtime\.provider\}/);
    assert.match(dock, /runtime\.retry\(\)/, 'a failed load can be retried');
    const layer = read(LAYER);
    assert.match(layer, /provider: HandLandmarkProvider \| null/);
    assert.match(layer, /!provider/, 'no provider, no loop');
    assert.ok(!layer.includes('createLandmarkerProvider'), 'the layer does not pick a model');
    const hook = read(RUNTIME_HOOK);
    assert.match(hook, /new HandRuntime\(/);
    assert.match(hook, /loadMediaPipeHandProvider/);
    assert.match(hook, /\.dispose\(\)/, 'the hook disposes the runtime on unmount');
  });
});

describe('hand control is opt-in and never the only way', () => {
  it('is off by default and only mounts the camera when switched on', () => {
    const workspace = read(WORKSPACE);
    assert.match(workspace, /const \[handsEnabled, setHandsEnabled\] = useState\(false\)/);
    assert.match(workspace, /handsEnabled && \(\s*<HandsDock/);
    assert.match(read(HEADER), /pressed=\{handsEnabled\}/, 'the header toggle reflects the state');
    assert.match(read(HEADER), /aria-label="Hand control"/);
  });

  it('leaves keyboard and pointer controls complete', () => {
    const dashboard = read(DASHBOARD);
    assert.match(dashboard, /ArrowLeft/);
    assert.match(dashboard, /onPointerDown/);
    assert.match(dashboard, /reducedMotion="user"/);
    const frame = read(FRAME);
    assert.match(frame, /onKeyDown/);
    assert.match(frame, /aria-describedby=\{instructionsId\}/);
    assert.match(frame, /data-handle="move"/, 'a selection focuses the real move handle');
  });

  it('only grabs what the grid can actually move', () => {
    const dashboard = read(DASHBOARD);
    assert.match(dashboard, /movable/);
    assert.match(dashboard, /matchMedia\(GRID_MEDIA_QUERY\)/);
    assert.match(dashboard, /elementFromPoint/);
  });

  it('turns a confirmed pinch over a normal control into one safe click', () => {
    const dashboard = read(DASHBOARD);
    const gesture = read(GESTURE);
    assert.match(gesture, /changed === 'pinch'/, 'pinch is the only click gesture');
    assert.match(gesture, /never become a drag/, 'pinch must not turn into a move');
    assert.match(dashboard, /\.closest/, 'the hand must find a real interactive control');
    assert.match(dashboard, /button:not/, 'the control allowlist must include buttons');
    assert.match(dashboard, /\.click\(\)/, 'the hand must activate the same control as a pointer click');
    assert.match(dashboard, /data-handle/, 'move and resize handles must not be activated as ordinary clicks');
    assert.match(dashboard, /clickConsumed/, 'a click must not become an accidental widget grab when held');
  });

  it('keeps the cursor visible above JARVIS overlay cards without intercepting input', () => {
    const cursor = read(CURSOR);
    assert.match(cursor, /z-\[110\]/, 'the cursor must remain above the account-inspection overlay');
    assert.match(cursor, /pointer-events-none/);
  });

  it('respects reduced motion and keeps the cursor out of the way of hit testing', () => {
    const cursor = read(CURSOR);
    assert.match(cursor, /pointer-events-none/);
    assert.match(cursor, /motion-reduce:/);
    assert.match(cursor, /aria-hidden/);
  });

  it('shows an explicit local-only camera state with a device picker', () => {
    const dock = read(DOCK);
    assert.match(dock, /Local only/);
    assert.match(dock, /muted/);
    assert.match(dock, /playsInline/);
    assert.match(dock, /selectDevice/);
    assert.match(dock, /handStatus\(/);
  });
});
