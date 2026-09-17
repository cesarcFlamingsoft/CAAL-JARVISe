import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { it } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.startsWith('.') && context.parentURL) {
      const base = fileURLToPath(new URL(specifier, context.parentURL));
      if (existsSync(base + '.ts')) {
        return { url: pathToFileURL(base + '.ts').href, shortCircuit: true };
      }
    }
    return nextResolve(specifier, context);
  },
});

const { analyzeCameraView } = await import('./client.ts');

it('sends one frame through the same-origin BFF and clears it even on failure', async () => {
  const frame = {
    jpegBase64: 'private-frame',
    width: 640,
    height: 360,
    bytes: 128,
    cleared: false,
    clear() {
      this.jpegBase64 = '';
      this.cleared = true;
    },
  };
  let requests = 0;
  await assert.rejects(
    analyzeCameraView({} as HTMLVideoElement, false, {
      capture: async () => frame,
      csrf: async () => 'csrf-token-with-enough-length',
      request: async (url, init) => {
        requests++;
        assert.equal(url, '/api/visual/analyze');
        assert.equal(init.credentials, 'same-origin');
        assert.equal(init.cache, 'no-store');
        assert.equal(new Headers(init.headers).get('x-caal-csrf'), 'csrf-token-with-enough-length');
        assert.deepEqual(JSON.parse(String(init.body)), {
          image: 'private-frame',
          prompt: 'Briefly describe what is visible in this camera view.',
          company_private: false,
        });
        throw new Error('offline');
      },
    }),
    /analysis_unavailable/
  );
  assert.equal(requests, 1);
  assert.equal(frame.cleared, true);
  assert.equal(frame.jpegBase64, '');
});

it('refuses company mode before capture', async () => {
  let captures = 0;
  await assert.rejects(
    analyzeCameraView({} as HTMLVideoElement, true, {
      capture: async () => {
        captures++;
        throw new Error('must not run');
      },
      csrf: async () => 'unused',
      request: async () => new Response(),
    }),
    /company_mode_blocked/
  );
  assert.equal(captures, 0);
});

it('closing Vision during prerequisites prevents capture and upload', async () => {
  const controller = new AbortController();
  let captures = 0;
  await assert.rejects(
    analyzeCameraView({} as HTMLVideoElement, false, {
      signal: controller.signal,
      csrf: async () => {
        controller.abort();
        return 'csrf-token-with-enough-length';
      },
      capture: async () => {
        captures++;
        throw new Error('must not capture');
      },
    }),
    /abort/i
  );
  assert.equal(captures, 0);
});

it('voice capture refuses a changed authenticated user before capturing', async (t) => {
  let captured = false;
  t.mock.method(globalThis, 'fetch', async () =>
    Response.json({
      authenticated: true,
      user: { userId: 'other' },
      csrfToken: 'csrf-token-with-enough-length',
    })
  );
  await assert.rejects(
    analyzeCameraView({} as HTMLVideoElement, false, {
      expectedUser: 'owner',
      capture: async () => {
        captured = true;
        throw new Error('captured');
      },
    }),
    /not_signed_in/
  );
  assert.equal(captured, false);
});
