/**
 * Sign out: revoke the session server-side, then clear the cookies.
 *
 * Revocation happens in the CAAL backend's database, so the token is dead for
 * every future request rather than merely forgotten by this browser. The
 * cookie is cleared regardless of whether the backend answered -- a user who
 * asked to sign out must end up signed out locally either way.
 *
 * Sign-out is a state change, so it carries the origin and CSRF checks: a
 * cross-site page must not be able to log a user out at will.
 */
import type { NextRequest } from 'next/server';
import { clientKeyFor, logout } from '@/lib/auth/backend';
import { readIdentityConfig } from '@/lib/auth/config';
import { CSRF_COOKIE, readCookie, verifyCsrf } from '@/lib/auth/csrf';
import { apiError, isSecureRequest, noStoreJson, readLimited } from '@/lib/auth/guard';
import { isTrustedMutationOrigin } from '@/lib/auth/origin';
import { SESSION_COOKIE, clearedSessionCookie } from '@/lib/auth/session-cookie';

export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  const limited = readLimited(req);
  if (limited) return limited;

  const status = readIdentityConfig();
  if (status.status !== 'enabled') {
    return apiError(503, 'identity_not_configured');
  }
  const config = status.config;
  if (!isTrustedMutationOrigin(req.headers, config.publicOrigin)) {
    return apiError(403, 'bad_origin');
  }
  if (!verifyCsrf(req.headers)) {
    return apiError(403, 'csrf');
  }

  const token = readCookie(req.headers.get('cookie'), SESSION_COOKIE);
  if (token) {
    // Best effort: a backend that is down must not trap the user in a session
    // they asked to leave, and an unknown token succeeds anyway.
    await logout(config, token, await clientKeyFor(req.headers));
  }

  const secure = isSecureRequest(req);
  const res = noStoreJson({ ok: true, next: '/login' });
  res.cookies.set(clearedSessionCookie({ secure }));
  res.cookies.set({
    name: CSRF_COOKIE,
    value: '',
    httpOnly: true,
    sameSite: 'strict',
    secure,
    path: '/',
    maxAge: 0,
  });
  return res;
}
