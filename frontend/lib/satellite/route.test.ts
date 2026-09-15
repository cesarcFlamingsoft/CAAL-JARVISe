import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire, registerHooks } from 'node:module';
import { dirname, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

// Exercise real route/guard/session code. Only the backend network boundary is
// simulated; no production credentials, accounts, or sessions are accessed.
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(import.meta.url);
const ts = require('typescript');
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (
      specifier.startsWith('@/') ||
      (specifier.startsWith('.') && context.parentURL?.startsWith(pathToFileURL(root).href))
    ) {
      const base = specifier.startsWith('@/')
        ? resolve(root, specifier.slice(2))
        : fileURLToPath(new URL(specifier, context.parentURL));
      if (existsSync(base + '.ts'))
        return { url: pathToFileURL(base + '.ts').href, shortCircuit: true };
    }
    if (specifier === 'next/server') return nextResolve('next/server.js', context);
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (
      url.startsWith(pathToFileURL(root).href) &&
      url.endsWith('.ts') &&
      !url.includes('/node_modules/')
    ) {
      return {
        format: 'module',
        source: ts.transpileModule(readFileSync(fileURLToPath(url), 'utf8'), {
          compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
        }).outputText,
        shortCircuit: true,
      };
    }
    return nextLoad(url, context);
  },
});

test('satellite enrollment uses real BFF session/admin/CSRF/origin guards', async (t) => {
  process.env.CAAL_INTERNAL_AUTH_SECRET = 'test-only-secret-'.repeat(4);
  process.env.CAAL_IDENTITY_API_URL = 'http://backend.test';
  process.env.CAAL_PUBLIC_ORIGIN = 'https://jarvis.example.invalid';
  delete process.env.CF_ACCESS_TEAM_DOMAIN;
  delete process.env.CF_ACCESS_AUD;
  const { POST } = await import('../../app/api/admin/satellites/route.ts');
  let role = 'admin';
  let writes = 0;
  const credential = 's'.repeat(43);
  const csrf = 'fixture-csrf-value-with-enough-length';
  const session = 'fixture-session-value-with-enough-length';
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    const target = new URL(String(url));
    assert.equal(target.origin, 'http://backend.test');
    if (target.pathname === '/auth/session')
      return Response.json({
        user_id: 'usr_' + '1'.repeat(24),
        display_name: 'Synthetic admin',
        role,
        status: 'active',
        must_change_password: false,
      });
    assert.equal(target.pathname, '/admin/satellites');
    assert.equal(init.method, 'POST');
    assert.deepEqual(JSON.parse(init.body), {
      satellite_id: 'assist_satellite.living',
      connection_id: 'ha_' + 'b'.repeat(24),
    });
    writes++;
    return Response.json({
      id: 'sat_' + 'a'.repeat(24),
      credential,
      satellite_id: 'assist_satellite.home_assistant_voice_0a3d6b_assist_satellite',
      device_id: '0bf018dfe200b28d8f7cff95e8d2aa75',
    });
  });
  function request(
    headers: Record<string, string> = {},
    body: object = {
      satellite_id: 'assist_satellite.living',
      connection_id: 'ha_' + 'b'.repeat(24),
    }
  ) {
    return new Request('https://jarvis.example.invalid/api/admin/satellites', {
      method: 'POST',
      headers: {
        host: 'jarvis.example.invalid',
        origin: 'https://jarvis.example.invalid',
        'content-type': 'application/json',
        cookie: `caal_session=${session}; caal_csrf=${csrf}`,
        'x-caal-csrf': csrf,
        ...headers,
      },
      body: JSON.stringify(body),
    });
  }
  assert.equal((await POST(request({ cookie: '' }))).status, 401);
  role = 'member';
  assert.equal((await POST(request())).status, 403);
  role = 'admin';
  assert.equal((await POST(request({ origin: 'https://hostile.example' }))).status, 403);
  assert.equal((await POST(request({ 'x-caal-csrf': '' }))).status, 403);
  assert.equal((await POST(request({}, { user_id: 'fake' }))).status, 422);
  assert.equal(writes, 0);
  const result = await POST(request());
  assert.equal(result.status, 200);
  assert.equal(result.headers.get('cache-control'), 'no-store');
  assert.equal((await result.json()).credential, credential);
  assert.equal(writes, 1);
});
