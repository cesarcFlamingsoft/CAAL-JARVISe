/**
 * Who am I, and what may this page do.
 *
 * Also the only place the CSRF token is handed to page scripts: it lives in an
 * HttpOnly cookie the browser cannot read, and is returned in this JSON body,
 * which only same-origin scripts can read. Never returns an email, a phone
 * number, or the session token.
 */
import type { NextRequest } from 'next/server';
import { CSRF_COOKIE, csrfCookieOptions, issueCsrfToken, readCookie } from '@/lib/auth/csrf';
import { clearSession, isSecureRequest, noStoreJson, readLimited } from '@/lib/auth/guard';
import { authenticate } from '@/lib/auth/session';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const limited = readLimited(req);
  if (limited) return limited;

  const auth = await authenticate(req.headers);
  const existing = readCookie(req.headers.get('cookie'), CSRF_COOKIE);
  const csrfToken = existing && existing.length >= 32 ? existing : issueCsrfToken();

  const body: Record<string, unknown> = {
    configured: auth.kind !== 'unconfigured',
    authenticated: auth.kind === 'user',
    csrfToken,
    // The sign-in page needs to know which forms to render.
    passwordLogin: auth.kind === 'unconfigured' ? false : auth.config.passwordLogin,
    cloudflareAccess: auth.kind === 'unconfigured' ? false : auth.config.accessEnabled,
  };
  if (auth.kind === 'user') {
    body.user = {
      userId: auth.user.userId,
      displayName: auth.user.displayName,
      role: auth.user.role,
    };
    body.mustChangePassword = auth.mustChangePassword;
    body.via = auth.via;
  } else if (auth.kind === 'denied') {
    body.reason = auth.reason;
  } else if (auth.kind !== 'unconfigured') {
    body.reason =
      auth.kind === 'invalid'
        ? 'invalid_assertion'
        : auth.kind === 'expired'
          ? 'session_expired'
          : 'not_signed_in';
  }

  const res = noStoreJson(body);
  if (csrfToken !== existing) {
    res.cookies.set({ ...csrfCookieOptions({ secure: isSecureRequest(req) }), value: csrfToken });
  }
  // A cookie the backend rejected is worse than no cookie: clear it so the
  // next request is a clean anonymous one rather than another failed lookup.
  if (auth.kind === 'expired') {
    clearSession(res, req);
  }
  return res;
}
