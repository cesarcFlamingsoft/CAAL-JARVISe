import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  SESSION_COOKIE,
  clearedSessionCookie,
  cookieSecurity,
  safeNextPath,
  sessionCookieOptions,
} from './session-cookie.ts';

describe('session cookie', () => {
  it('is HttpOnly, Lax, path-scoped and Secure over TLS', () => {
    const options = sessionCookieOptions({ secure: true, maxAge: 3600 });

    assert.equal(options.name, SESSION_COOKIE);
    assert.equal(options.httpOnly, true);
    assert.equal(options.secure, true);
    // Lax, not Strict: the cookie must survive the top-level navigation that
    // follows sign-in, while still never riding a cross-site POST.
    assert.equal(options.sameSite, 'lax');
    assert.equal(options.path, '/');
    assert.equal(options.maxAge, 3600);
  });

  it('clamps a nonsensical lifetime rather than trusting it', () => {
    assert.ok(sessionCookieOptions({ secure: true, maxAge: -1 }).maxAge >= 0);
    assert.ok(sessionCookieOptions({ secure: true, maxAge: 10 ** 12 }).maxAge <= 90 * 86400);
    assert.ok(sessionCookieOptions({ secure: true, maxAge: Number.NaN }).maxAge >= 0);
  });

  it('expires immediately when cleared and carries no value', () => {
    const cleared = clearedSessionCookie({ secure: false });

    assert.equal(cleared.name, SESSION_COOKIE);
    assert.equal(cleared.value, '');
    assert.equal(cleared.maxAge, 0);
    assert.equal(cleared.httpOnly, true);
  });
});

describe('cookie transport policy', () => {
  const https = new Headers({ 'x-forwarded-proto': 'https' });
  const http = new Headers();

  it('marks cookies Secure whenever the browser reached us over TLS', () => {
    assert.deepEqual(cookieSecurity(https, 'https://x/y', false), { ok: true, secure: true });
    assert.deepEqual(cookieSecurity(http, 'https://x/y', false), { ok: true, secure: true });
  });

  it('refuses to issue a session over plain HTTP by default', () => {
    assert.deepEqual(cookieSecurity(http, 'http://x/y', false), {
      ok: false,
      secure: true,
      reason: 'insecure_transport',
    });
  });

  it('permits a plain-HTTP LAN session only when explicitly opted in', () => {
    assert.deepEqual(cookieSecurity(http, 'http://x/y', true), { ok: true, secure: false });
  });

  it('never downgrades an HTTPS request even when the opt-in is set', () => {
    assert.deepEqual(cookieSecurity(https, 'http://x/y', true), { ok: true, secure: true });
  });
});

describe('safe redirect targets', () => {
  it('keeps an ordinary same-origin path', () => {
    assert.equal(safeNextPath('/account'), '/account');
    assert.equal(safeNextPath('/admin?tab=users'), '/admin?tab=users');
    assert.equal(safeNextPath('/a/b/c#frag'), '/a/b/c#frag');
  });

  it('falls back to the root for anything not a same-origin path', () => {
    for (const hostile of [
      'https://evil.example.com/',
      '//evil.example.com',
      '//evil.example.com/path',
      '/\\evil.example.com',
      '\\\\evil.example.com',
      'http://evil.example.com',
      'javascript:alert(1)',
      'data:text/html,x',
      '/\t/evil.example.com',
      '/\n/evil',
      'account',
      '',
      '   ',
      null,
      undefined,
      42,
      '/'.repeat(2000),
    ] as unknown[]) {
      assert.equal(safeNextPath(hostile), '/', JSON.stringify(hostile));
    }
  });

  it('never returns the login page itself, which would loop', () => {
    assert.equal(safeNextPath('/login'), '/');
    assert.equal(safeNextPath('/login?next=/admin'), '/');
  });

  it('strips a userinfo trick that some parsers treat as a host', () => {
    assert.equal(safeNextPath('/@evil.example.com'), '/@evil.example.com');
    assert.equal(safeNextPath('//user@evil.example.com'), '/');
  });
});
