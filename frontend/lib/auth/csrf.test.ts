import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  CSRF_COOKIE,
  CSRF_HEADER,
  csrfCookieOptions,
  issueCsrfToken,
  readCookie,
  verifyCsrf,
} from './csrf.ts';

function headersWith(entries: Record<string, string>): Headers {
  return new Headers(entries);
}

describe('CSRF double-submit token', () => {
  it('issues long random tokens', () => {
    const a = issueCsrfToken();
    const b = issueCsrfToken();
    assert.ok(a.length >= 32);
    assert.notEqual(a, b);
    assert.match(a, /^[A-Za-z0-9_-]+$/);
  });

  it('is stored HttpOnly, SameSite=Strict, Secure and short lived', () => {
    const options = csrfCookieOptions({ secure: true });
    assert.equal(options.name, CSRF_COOKIE);
    assert.equal(options.httpOnly, true);
    assert.equal(options.sameSite, 'strict');
    assert.equal(options.secure, true);
    assert.equal(options.path, '/');
    assert.ok(options.maxAge > 0 && options.maxAge <= 12 * 3600);
  });

  it('accepts only a header that matches the cookie', () => {
    const token = issueCsrfToken();
    const ok = headersWith({ cookie: `${CSRF_COOKIE}=${token}`, [CSRF_HEADER]: token });
    assert.equal(verifyCsrf(ok), true);

    const cases = [
      headersWith({ cookie: `${CSRF_COOKIE}=${token}` }),
      headersWith({ [CSRF_HEADER]: token }),
      headersWith({ cookie: `${CSRF_COOKIE}=${token}`, [CSRF_HEADER]: issueCsrfToken() }),
      headersWith({ cookie: `${CSRF_COOKIE}=${token}`, [CSRF_HEADER]: token.slice(1) }),
      headersWith({ cookie: `${CSRF_COOKIE}=`, [CSRF_HEADER]: '' }),
      headersWith({ cookie: `other=${token}`, [CSRF_HEADER]: token }),
    ];
    for (const headers of cases) {
      assert.equal(verifyCsrf(headers), false);
    }
  });

  it('reads a cookie out of a cookie header defensively', () => {
    assert.equal(readCookie('a=1; caal_csrf=abc; b=2', 'caal_csrf'), 'abc');
    assert.equal(readCookie('caal_csrf=a%20b', 'caal_csrf'), 'a b');
    assert.equal(readCookie(null, 'caal_csrf'), null);
    assert.equal(readCookie('caal_csrf', 'caal_csrf'), null);
    assert.equal(readCookie('xcaal_csrf=1', 'caal_csrf'), null);
  });
});
