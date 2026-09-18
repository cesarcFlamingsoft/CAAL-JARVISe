import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire, registerHooks } from 'node:module';
import { dirname, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

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

const origin = 'http://localhost:3000';
const csrf = 'visual-reanalyze-test-csrf-token';
function request(body: object, extra: Record<string, string> = {}) {
  return new Request(origin + '/api/visual/reanalyze', {
    method: 'POST',
    headers: {
      host: 'localhost:3000',
      origin,
      referer: origin + '/',
      'content-type': 'application/json',
      cookie: `caal_session=session; caal_csrf=${csrf}`,
      'x-caal-csrf': csrf,
      ...extra,
    },
    body: JSON.stringify(body),
  });
}

test('visual reanalysis BFF authenticates a bounded question and never forwards cookies', async (t) => {
  process.env.CAAL_INTERNAL_AUTH_SECRET = 'visual-reanalyze-test-secret-'.repeat(3);
  process.env.CAAL_IDENTITY_API_URL = 'http://backend.test';
  process.env.CAAL_PASSWORD_LOGIN = 'true';
  delete process.env.CAAL_PUBLIC_ORIGIN;
  let calls = 0;
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    const target = new URL(String(url));
    if (target.pathname === '/auth/session')
      return Response.json({
        user_id: 'usr_' + '2'.repeat(24),
        display_name: 'Viewer',
        role: 'member',
        status: 'active',
        must_change_password: false,
      });
    assert.equal(target.pathname, '/users/me/visual/reanalyze');
    calls++;
    assert.equal(new Headers(init?.headers).has('cookie'), false);
    assert.deepEqual(JSON.parse(String(init?.body)), { question: 'What color is it?' });
    return Response.json({ description: 'Blue.' });
  });
  const { POST } = await import('../../app/api/visual/reanalyze/route.ts');
  for (const body of [
    {},
    { question: '' },
    { question: 'x'.repeat(241) },
    { question: 'bad\ntext' },
    { question: 'ok', extra: true },
  ]) {
    assert.equal((await POST(request(body))).status, 422);
  }
  const response = await POST(request({ question: 'What color is it?' }));
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  assert.deepEqual(await response.json(), { description: 'Blue.' });
  assert.equal(calls, 1);
});
