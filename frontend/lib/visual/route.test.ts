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
const csrf = 'visual-test-csrf-token-long-enough';
const image = 'a'.repeat(100);
const prompt =
  'Briefly describe what is visible in this camera view. Include a visible brand, model, or text only when clear; never guess.';
let index = 0;

function request(
  body: object,
  extra: Record<string, string> = {},
  { session = true }: { session?: boolean } = {}
) {
  return new Request(origin + '/api/visual/analyze', {
    method: 'POST',
    headers: {
      host: 'localhost:3000',
      origin,
      referer: origin + '/',
      'content-type': 'application/json',
      cookie: `${session ? 'caal_session=session;' : ''} caal_csrf=${csrf}`,
      'x-caal-csrf': csrf,
      'cf-connecting-ip': `visual-${index++}`,
      ...extra,
    },
    body: JSON.stringify(body),
  });
}

test('visual BFF enforces auth, origin, CSRF, schema, company mode, and cookie isolation', async (t) => {
  process.env.CAAL_INTERNAL_AUTH_SECRET = 'visual-test-internal-secret-'.repeat(3);
  process.env.CAAL_IDENTITY_API_URL = 'http://backend.test';
  process.env.CAAL_PUBLIC_ORIGIN = origin;
  delete process.env.CAAL_TRUSTED_LOCAL_ORIGINS;
  process.env.CAAL_PASSWORD_LOGIN = 'true';
  delete process.env.CAAL_PUBLIC_ORIGIN;
  delete process.env.CF_ACCESS_TEAM_DOMAIN;
  delete process.env.CF_ACCESS_AUD;

  let visualCalls = 0;
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    const target = new URL(String(url));
    if (target.pathname === '/auth/session') {
      return Response.json({
        user_id: 'usr_' + '1'.repeat(24),
        display_name: 'Viewer',
        role: 'member',
        status: 'active',
        must_change_password: false,
      });
    }
    assert.equal(target.pathname, '/users/me/visual/analyze');
    visualCalls++;
    const headers = new Headers(init?.headers);
    assert.equal(headers.has('cookie'), false, 'browser cookies never cross the trust boundary');
    assert.match(headers.get('authorization') ?? '', /^Bearer /);
    assert.deepEqual(JSON.parse(String(init?.body)), {
      image,
      prompt,
      company_private: false,
    });
    return Response.json({ description: 'A desk.' });
  });

  const { POST } = await import('../../app/api/visual/analyze/route.ts');
  const valid = { image, prompt, company_private: false };

  assert.equal((await POST(request(valid, {}, { session: false }))).status, 401);
  assert.equal((await POST(request(valid, { origin: 'https://evil.example' }))).status, 403);
  assert.equal((await POST(request(valid, { 'x-caal-csrf': 'wrong' }))).status, 403);
  assert.equal((await POST(request({ ...valid, extra: true }))).status, 422);
  assert.equal((await POST(request({ ...valid, prompt: 'x'.repeat(241) }))).status, 422);
  assert.equal(
    (await POST(request({ ...valid, prompt: 'What does our private contract say?' }))).status,
    422
  );

  const companyBody = await POST(request({ ...valid, company_private: true }));
  assert.equal(companyBody.status, 403);
  assert.deepEqual(await companyBody.json(), { error: 'company_mode_blocked' });
  const companyReferer = await POST(request(valid, { referer: origin + '/?company=1' }));
  assert.equal(companyReferer.status, 403);
  assert.deepEqual(await companyReferer.json(), { error: 'company_mode_blocked' });
  assert.equal(visualCalls, 0, 'no company-mode frame reached the agent');

  const wrongVoiceUser = await POST(request(valid, { 'x-caal-visual-user': 'wrong-user' }));
  assert.equal(wrongVoiceUser.status, 403);
  assert.equal(visualCalls, 0);

  const accepted = await POST(request(valid));
  assert.equal(accepted.status, 200);
  assert.deepEqual(await accepted.json(), { description: 'A desk.' });
  assert.equal(accepted.headers.get('cache-control'), 'no-store');
  assert.equal(visualCalls, 1);
});
