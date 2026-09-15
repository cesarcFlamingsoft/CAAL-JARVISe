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

const publicOrigin = 'https://jarvis.mexcantech.io';
const local = 'http://localhost:3000';
const csrf = 'test-only-csrf-token-with-enough-length';
let requestIndex = 0;
function request(origin: string, path: string, body: object, extra: Record<string, string> = {}) {
  return new Request(`${origin}${path}`, {
    method: path === '/api/tts' ? 'PUT' : 'POST',
    headers: {
      host: new URL(origin).host,
      origin,
      'content-type': 'application/json',
      cookie: `caal_csrf=${csrf}`,
      'x-caal-csrf': csrf,
      'cf-connecting-ip': `test-${requestIndex++}`,
      ...extra,
    },
    body: JSON.stringify(body),
  });
}

test('local login passes origin/transport and safely rejects empty credentials', async () => {
  process.env.CAAL_INTERNAL_AUTH_SECRET = 'test-only-internal-secret-'.repeat(3);
  process.env.CAAL_IDENTITY_API_URL = 'http://backend.test';
  process.env.CAAL_PUBLIC_ORIGIN = publicOrigin;
  process.env.CAAL_TRUSTED_LOCAL_ORIGINS = local;
  process.env.CAAL_ALLOW_INSECURE_COOKIES = 'false';
  delete process.env.CF_ACCESS_TEAM_DOMAIN;
  delete process.env.CF_ACCESS_AUD;
  const { POST } = await import('../../app/api/auth/login/route.ts');
  const response = await POST(request(local, '/api/auth/login', {}));
  assert.equal(response.status, 401);
  assert.deepEqual(await response.json(), { error: 'invalid_credentials' });
});

test('local logout retains CSRF checks and clears cookies', async () => {
  const { POST } = await import('../../app/api/auth/logout/route.ts');
  const response = await POST(request(local, '/api/auth/logout', {}));
  assert.equal(response.status, 200);
  assert.match(response.headers.get('set-cookie')!, /caal_session=;.*Max-Age=0/);
  const denied = await POST(request(local, '/api/auth/logout', {}, { 'x-caal-csrf': 'wrong' }));
  assert.equal(denied.status, 403);
  assert.equal((await denied.json()).error, 'csrf');
});

test('local and public sessions retain protected TTS/admin mutations and revocation', async (t) => {
  const login = await import('../../app/api/auth/login/route.ts');
  const logout = await import('../../app/api/auth/logout/route.ts');
  const tts = await import('../../app/api/tts/route.ts');
  assert.ok(existsSync(resolve(root, 'app/api/tts/voicebox/route.ts')));
  const voicebox = await import('../../app/api/tts/voicebox/route.ts');
  const voiceboxTest = await import('../../app/api/tts/voicebox/test/route.ts');
  const admin = await import('../../app/api/admin/users/route.ts');
  const { jwtVerify } = await import('jose');
  const session = 'isolated-test-session-never-valid-in-production';
  const user = {
    user_id: 'usr_' + '1'.repeat(24),
    display_name: 'Test',
    role: 'admin',
    status: 'active',
    must_change_password: false,
  };
  let active = false;
  let writes = 0;
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    const target = new URL(String(url));
    assert.equal(target.origin, 'http://backend.test');
    const authorization = new Headers(init.headers).get('authorization')!;
    const { payload } = await jwtVerify(
      authorization.slice(7),
      new TextEncoder().encode(process.env.CAAL_INTERNAL_AUTH_SECRET)
    );
    const body = init.body ? JSON.parse(init.body) : {};
    if (target.pathname === '/auth/login') {
      assert.equal(payload.sub, 'identity');
      active = true;
      return Response.json({
        ...user,
        session_token: session,
        expires_at: Math.floor(Date.now() / 1000) + 3600,
      });
    }
    if (target.pathname === '/auth/session') {
      assert.equal(payload.sub, 'identity');
      assert.equal(body.session_token, session);
      return active ? Response.json(user) : Response.json({}, { status: 401 });
    }
    if (target.pathname === '/auth/logout') {
      assert.equal(body.session_token, session);
      active = false;
      return Response.json({ ok: true });
    }
    assert.equal(payload.sub, user.user_id);
    assert.equal(payload.aud, 'caal-backend');
    if (target.pathname === '/admin/users') {
      assert.equal(init.method, 'POST');
      assert.equal(body.role, 'member');
      return Response.json(
        { ...user, user_id: 'usr_' + '2'.repeat(24), role: 'member' },
        { status: 201 }
      );
    }
    if (target.pathname.startsWith('/users/me/tts/voicebox')) {
      assert.equal(user.role, 'admin');
      return Response.json({endpoint:'http://127.0.0.1:8000', credential_configured:true, credential:'must-be-stripped'});
    }
    assert.equal(target.pathname, '/users/me/tts');
    if (init.method === 'PUT') {
      writes++;
      assert.equal(body.provider, 'qwen-trial');
    }
    return Response.json({ provider: 'qwen-trial', qwen_configured: true, qwen_voice:'jarvis-designed', source:'personal', applies_to:'new_sessions', voicebox_status:'not_configured', can_configure:user.role === 'admin', profile_id:null, engine:null, model_size:null });
  });
  for (const origin of [local, publicOrigin]) {
    const signedIn = await login.POST(
      request(origin, '/api/auth/login', {
        email: 'test@example.invalid',
        password: 'fixture-only',
      })
    );
    assert.equal(signedIn.status, 200);
    const data = await signedIn.json();
    assert.equal(data.session_token, undefined);
    assert.notEqual(data.csrfToken, csrf);
    const sessionCookie = signedIn.cookies.get('caal_session')!;
    assert.equal(sessionCookie.httpOnly, true);
    assert.equal(sessionCookie.sameSite, 'lax');
    assert.equal(sessionCookie.secure, origin === publicOrigin);
    assert.equal(sessionCookie.domain, undefined);
    const headers = {
      cookie: `caal_session=${session}; caal_csrf=${data.csrfToken}`,
      'x-caal-csrf': data.csrfToken,
    };
    const mutation = () => request(origin, '/api/tts', { provider: 'qwen-trial' }, headers);
    const result = await tts.PUT(mutation());
    assert.equal(result.status, 200);
    assert.equal(result.headers.get('cache-control'), 'no-store');
    assert.equal((await result.json()).provider, 'qwen-trial');
    const count = writes;
    for (const badOrigin of [
      'https://hostile.example',
      'null',
      origin + '/path',
      'http://localhost:3001',
      'http://127.0.0.1:3000',
    ]) {
      const req = mutation();
      req.headers.set('origin', badOrigin);
      req.headers.set('x-forwarded-host', new URL(origin).host);
      const denied = await tts.PUT(req);
      assert.equal(denied.status, 403);
      assert.equal((await denied.json()).error, 'bad_origin');
    }
    const noCsrf = mutation();
    noCsrf.headers.delete('x-caal-csrf');
    assert.equal((await (await tts.PUT(noCsrf)).json()).error, 'csrf');
    assert.equal(writes, count);
    const created = await admin.POST(
      request(
        origin,
        '/api/admin/users',
        {
          email: 'new-test@example.invalid',
          displayName: 'New test',
          role: 'member',
        },
        headers
      )
    );
    assert.equal(created.status, 201);
    assert.equal((await created.json()).user.role, 'member');
    const adminReq = request(origin, '/api/admin/users', {}, headers);
    assert.equal((await admin.POST(adminReq)).status, 422); // reached body validation
    const adminCsrf = request(
      origin,
      '/api/admin/users',
      {},
      { ...headers, 'x-caal-csrf': 'wrong' }
    );
    assert.equal((await (await admin.POST(adminCsrf)).json()).error, 'csrf');
    const configRequest = () => request(origin, '/api/tts/voicebox', {endpoint:'http://127.0.0.1:8000', credential:'test-credential'}, headers);
    const configResult = await voicebox.PUT(configRequest());
    assert.equal(configResult.status, 200);
    assert.ok(!JSON.stringify(await configResult.json()).includes('must-be-stripped'));
    assert.equal((await voiceboxTest.POST(configRequest())).status, 200);
    assert.equal((await voicebox.PUT(request(origin, '/api/tts/voicebox', {}, {...headers, 'x-caal-csrf':'wrong'}))).status, 403);
    user.role = 'member';
    assert.equal((await voicebox.PUT(configRequest())).status, 403);
    assert.equal((await voicebox.GET(configRequest())).status, 403);
    assert.equal((await voiceboxTest.POST(configRequest())).status, 403);
    assert.equal((await tts.PUT(mutation())).status, 200);
    assert.equal((await admin.POST(request(origin, '/api/admin/users', {}, headers))).status, 403);
    user.role = 'admin';
    user.must_change_password = true;
    assert.equal((await (await tts.PUT(mutation())).json()).error, 'password_change_required');
    user.must_change_password = false;
    const signedOut = await logout.POST(request(origin, '/api/auth/logout', {}, headers));
    assert.equal(signedOut.status, 200);
    assert.equal(active, false);
    assert.equal(signedOut.cookies.get('caal_session')?.maxAge, 0);
    assert.equal((await tts.PUT(mutation())).status, 401);
    assert.equal(
      (await tts.PUT(request(origin, '/api/tts', { provider: 'qwen-trial' }))).status,
      401
    );
  }
});

test('login/logout reject hostile or malformed origins and preserve CSRF/rate limits', async () => {
  const login = await import('../../app/api/auth/login/route.ts');
  const logout = await import('../../app/api/auth/logout/route.ts');
  for (const origin of [local, publicOrigin]) {
    for (const route of [login.POST, logout.POST]) {
      for (const bad of [
        'https://hostile.example',
        'null',
        origin + '/',
        origin + '?x',
        origin.replace('://', '://user@'),
      ]) {
        const req = request(origin, '/api/auth/login', {}, { origin: bad });
        const response = await route(req);
        assert.equal(response.status, 403);
        assert.equal((await response.json()).error, 'bad_origin');
      }
      const missing = request(origin, '/api/auth/login', {});
      missing.headers.delete('origin');
      assert.equal((await (await route(missing)).json()).error, 'bad_origin');
      const noCsrf = request(origin, '/api/auth/login', {}, { 'x-caal-csrf': 'wrong' });
      assert.equal((await (await route(noCsrf)).json()).error, 'csrf');
    }
  }
  for (let i = 0; i < 10; i++) {
    assert.equal(
      (
        await login.POST(
          request(local, '/api/auth/login', {}, { 'cf-connecting-ip': 'test-rate-limit' })
        )
      ).status,
      401
    );
  }
  assert.equal(
    (
      await login.POST(
        request(local, '/api/auth/login', {}, { 'cf-connecting-ip': 'test-rate-limit' })
      )
    ).status,
    429
  );
});
