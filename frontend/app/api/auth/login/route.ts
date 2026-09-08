/**
 * Standalone sign-in.
 *
 * The password is forwarded once to the CAAL backend, which checks it against
 * its own Argon2id hash and returns an opaque session token. That token goes
 * straight into an HttpOnly cookie and is never put in the response body, so
 * no script -- ours or an injected one -- can read it.
 *
 * Sign-in is itself a state change, so it carries the full mutation guard: an
 * `Origin`/`Referer` check, the double-submit CSRF token, and a rate limit at
 * this edge on top of the per-client limit and per-account lockout the backend
 * enforces. The CSRF token is *rotated* on success, so a token planted before
 * sign-in cannot be replayed against the new session.
 */
import type { NextRequest } from 'next/server';
import { clientKeyFor, login } from '@/lib/auth/backend';
import { readIdentityConfig } from '@/lib/auth/config';
import { csrfCookieOptions, issueCsrfToken, verifyCsrf } from '@/lib/auth/csrf';
import {
  apiError,
  cookieSecurity,
  isSecureRequest,
  loginLimited,
  noStoreJson,
  readJsonObject,
} from '@/lib/auth/guard';
import { isTrustedMutationOrigin } from '@/lib/auth/origin';
import { safeNextPath, sessionCookieOptions } from '@/lib/auth/session-cookie';

export const dynamic = 'force-dynamic';

const MAX_EMAIL_LENGTH = 254;
const MAX_PASSWORD_LENGTH = 256;

export async function POST(req: NextRequest) {
  const limited = loginLimited(req);
  if (limited) return limited;

  const status = readIdentityConfig();
  if (status.status !== 'enabled') {
    return apiError(503, 'identity_not_configured');
  }
  const config = status.config;
  if (!config.passwordLogin) {
    return apiError(404, 'not_found');
  }
  if (!isTrustedMutationOrigin(req.headers, config.publicOrigin)) {
    return apiError(403, 'bad_origin');
  }
  if (!verifyCsrf(req.headers)) {
    return apiError(403, 'csrf');
  }

  // Refuse to mint a session that would travel in the clear.
  const transport = cookieSecurity(req.headers, req.url, config.allowInsecureCookies);
  if (!transport.ok) {
    return apiError(400, transport.reason);
  }

  const body = await readJsonObject(req);
  const email = body?.email;
  const password = body?.password;
  if (
    typeof email !== 'string' ||
    typeof password !== 'string' ||
    email.length === 0 ||
    email.length > MAX_EMAIL_LENGTH ||
    password.length === 0 ||
    password.length > MAX_PASSWORD_LENGTH
  ) {
    // The same shape as a wrong password: a malformed body must not become a
    // way to probe which addresses exist.
    return apiError(401, 'invalid_credentials');
  }

  const result = await login(config, email, password, await clientKeyFor(req.headers));
  if (!result.ok) {
    if (result.status === 429) {
      const detail = (result.data as { detail?: unknown } | null)?.detail;
      return apiError(429, detail === 'locked' ? 'locked' : 'rate_limited');
    }
    if (result.status === 401) {
      return apiError(401, 'invalid_credentials');
    }
    if (result.status === 404 || result.status === 503) {
      return apiError(503, 'identity_not_configured');
    }
    return apiError(502, 'backend_unavailable');
  }

  const data = result.data as {
    user_id?: unknown;
    display_name?: unknown;
    role?: unknown;
    must_change_password?: unknown;
    session_token?: unknown;
    expires_at?: unknown;
  } | null;
  const token = typeof data?.session_token === 'string' ? data.session_token : null;
  if (!token || typeof data?.user_id !== 'string') {
    return apiError(502, 'backend_unavailable');
  }

  const mustChange = data.must_change_password === true;
  const expiresAt = typeof data.expires_at === 'number' ? data.expires_at : 0;
  const maxAge = Math.max(0, expiresAt - Math.floor(Date.now() / 1000));
  // A one-time password gets you exactly one place: the change form.
  const next = mustChange ? '/change-password' : safeNextPath(body?.next);
  const rotatedCsrf = issueCsrfToken();

  const res = noStoreJson({
    ok: true,
    user: {
      userId: data.user_id,
      displayName: typeof data.display_name === 'string' ? data.display_name : '',
      role: data.role === 'admin' ? 'admin' : 'member',
    },
    mustChangePassword: mustChange,
    next,
    csrfToken: rotatedCsrf,
  });
  res.cookies.set({ ...sessionCookieOptions({ secure: transport.secure, maxAge }), value: token });
  res.cookies.set({ ...csrfCookieOptions({ secure: isSecureRequest(req) }), value: rotatedCsrf });
  return res;
}
