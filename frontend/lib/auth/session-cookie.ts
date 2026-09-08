/**
 * The browser's half of standalone sign-in: the session cookie and the rules
 * for where a user may be sent after signing in.
 *
 * The cookie holds an opaque token the CAAL backend issued; it is meaningless
 * on its own and is re-resolved against the database on every request. It is
 * `HttpOnly` so no script can read it, `SameSite=Lax` so it survives the
 * top-level navigation after sign-in but never rides a cross-site POST, and
 * `Secure` whenever the browser reached us over TLS.
 *
 * Over plain HTTP the default is to refuse a session outright rather than hand
 * out a bearer token that travels in the clear. A LAN deployment that really
 * wants this must set `CAAL_ALLOW_INSECURE_COOKIES=true`, which is loud in the
 * configuration summary. There is deliberately no silent downgrade.
 */

export const SESSION_COOKIE = 'caal_session';
/** Ninety days, matching the backend's own ceiling on a session's lifetime. */
const MAX_COOKIE_AGE_SECONDS = 90 * 86400;

export interface SessionCookieOptions {
  name: string;
  value?: string;
  httpOnly: true;
  sameSite: 'lax';
  secure: boolean;
  path: string;
  maxAge: number;
}

function clampAge(seconds: number): number {
  if (!Number.isFinite(seconds)) return 0;
  return Math.max(0, Math.min(Math.trunc(seconds), MAX_COOKIE_AGE_SECONDS));
}

export function sessionCookieOptions({
  secure,
  maxAge,
}: {
  secure: boolean;
  maxAge: number;
}): SessionCookieOptions {
  return {
    name: SESSION_COOKIE,
    httpOnly: true,
    sameSite: 'lax',
    secure,
    path: '/',
    maxAge: clampAge(maxAge),
  };
}

/** An immediately-expiring, empty cookie: what sign-out and every failure set. */
export function clearedSessionCookie({
  secure,
}: {
  secure: boolean;
}): SessionCookieOptions & { value: string } {
  return { ...sessionCookieOptions({ secure, maxAge: 0 }), value: '' };
}

export type CookieSecurity =
  | { ok: true; secure: boolean }
  | { ok: false; secure: true; reason: 'insecure_transport' };

/**
 * Decide whether this request may be given a session cookie, and whether that
 * cookie must be `Secure`.
 *
 * TLS is detected from the proxy's `X-Forwarded-Proto` or from the request URL.
 * Trusting that header is safe in the direction that matters: an attacker who
 * forges it only causes a *stricter* cookie. The dangerous direction -- a
 * missing header downgrading a real session to cleartext -- is closed by
 * refusing rather than downgrading.
 */
export function cookieSecurity(
  headers: Headers,
  url: string,
  allowInsecure: boolean
): CookieSecurity {
  const forwarded = headers.get('x-forwarded-proto');
  const firstHop = forwarded?.split(',')[0]?.trim().toLowerCase();
  const isHttps = firstHop === 'https' || url.startsWith('https://');
  if (isHttps) {
    return { ok: true, secure: true };
  }
  if (allowInsecure) {
    return { ok: true, secure: false };
  }
  return { ok: false, secure: true, reason: 'insecure_transport' };
}

const MAX_NEXT_LENGTH = 512;
/** Backslashes, C0/C1 controls and space: authority smuggling and header tricks. */
const HOSTILE = /[\\\u0000-\u0020\u007f-\u009f]/;

/**
 * Reduce an untrusted `?next=` to a safe same-origin path, or `/`.
 *
 * The check is allow-listing, not blocklisting: the value must parse, against
 * a throwaway base, to a URL whose origin is still that base. That closes
 * scheme-relative (`//evil`), backslash (`/\evil`), absolute and `javascript:`
 * targets in one rule rather than a list of patterns to keep up with.
 */
export function safeNextPath(raw: unknown, fallback = '/'): string {
  if (typeof raw !== 'string') return fallback;
  const value = raw.trim();
  if (!value || value.length > MAX_NEXT_LENGTH) return fallback;
  if (!value.startsWith('/')) return fallback;
  if (value.startsWith('//')) return fallback;
  if (HOSTILE.test(value)) return fallback;

  const base = 'https://caal.invalid';
  let url: URL;
  try {
    url = new URL(value, base);
  } catch {
    return fallback;
  }
  if (url.origin !== base) return fallback;
  if (!url.pathname.startsWith('/')) return fallback;
  // Sending someone back to the sign-in page after signing in is a loop.
  if (url.pathname === '/login' || url.pathname.startsWith('/login/')) return fallback;
  return `${url.pathname}${url.search}${url.hash}`;
}
