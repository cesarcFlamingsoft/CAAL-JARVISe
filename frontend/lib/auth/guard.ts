/**
 * Route-handler helpers that enforce the BFF's security posture uniformly.
 *
 * Every identity response is uncacheable; every mutation must come from our
 * own origin, echo the CSRF token, and stay under a per-caller rate limit;
 * every authorization decision is taken from a fresh backend resolution and
 * then enforced again by the backend itself.
 */
import { NextResponse } from 'next/server';
import { backendDetail } from './backend';
import { type IdentityConfig, maskEmail } from './config';
import { verifyCsrf } from './csrf';
import { isTrustedMutationOrigin } from './origin';
import { RateLimiter } from './rate-limit';
import { type AuthResult, type SessionUser, authenticate } from './session';
import { clearedSessionCookie, cookieSecurity } from './session-cookie';

export const MAX_JSON_BODY_BYTES = 16 * 1024;

const mutationLimiter = new RateLimiter({ limit: 60, windowMs: 60_000 });
const readLimiter = new RateLimiter({ limit: 240, windowMs: 60_000 });
/** Sign-in is the one route worth throttling hard at the edge as well. */
const loginLimiter = new RateLimiter({ limit: 10, windowMs: 60_000 });

export function securityHeaders(): Record<string, string> {
  return {
    'Cache-Control': 'no-store',
    Pragma: 'no-cache',
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'same-origin',
  };
}

export function noStoreJson(body: unknown, status = 200): NextResponse {
  return NextResponse.json(body, { status, headers: securityHeaders() });
}

export function apiError(
  status: number,
  code: string,
  extra: Record<string, unknown> = {}
): NextResponse {
  return noStoreJson({ error: code, ...extra }, status);
}

/**
 * Cookies are marked Secure whenever the browser reached us over TLS.
 *
 * For the *session* cookie use `cookieSecurity` instead: it additionally
 * refuses to issue one at all over plain HTTP unless the operator opted in.
 */
export function isSecureRequest(req: Request): boolean {
  const forwarded = req.headers.get('x-forwarded-proto')?.split(',')[0]?.trim().toLowerCase();
  return forwarded === 'https' || req.url.startsWith('https://');
}

/** Attach an immediately-expiring session cookie to a response. */
export function clearSession(res: NextResponse, req: Request): NextResponse {
  res.cookies.set(clearedSessionCookie({ secure: isSecureRequest(req) }));
  return res;
}

export function loginLimited(req: Request): NextResponse | null {
  return loginLimiter.allow(clientKey(req)) ? null : apiError(429, 'rate_limited');
}

export { cookieSecurity };

/** A stable key for rate limiting: the Cloudflare client address, else the first hop. */
export function clientKey(req: Request): string {
  const cf = req.headers.get('cf-connecting-ip');
  if (cf && cf.length <= 64) return `ip:${cf}`;
  const forwarded = req.headers.get('x-forwarded-for');
  const first = forwarded?.split(',')[0]?.trim();
  if (first && first.length <= 64) return `ip:${first}`;
  return 'ip:unknown';
}

export function readLimited(req: Request): NextResponse | null {
  return readLimiter.allow(clientKey(req)) ? null : apiError(429, 'rate_limited');
}

export type Authorized =
  | {
      ok: true;
      config: IdentityConfig;
      user: SessionUser;
      mustChangePassword: boolean;
      sessionToken: string | null;
    }
  | { ok: false; response: NextResponse };

/** Map an authentication outcome to a bounded browser response. */
export function authFailure(auth: AuthResult): NextResponse {
  switch (auth.kind) {
    case 'unconfigured':
      return apiError(503, 'identity_not_configured');
    case 'anonymous':
      return apiError(401, 'not_signed_in');
    case 'expired':
      return apiError(401, 'session_expired');
    case 'invalid':
      return apiError(401, 'invalid_assertion');
    case 'denied':
      return apiError(403, auth.reason);
    default:
      return apiError(401, 'unauthorized');
  }
}

/**
 * Authenticate a request.
 *
 * A user who must change their password is refused everything except the
 * routes that let them do so (`allowPasswordChange`). Enforcing it here rather
 * than only in the UI means a one-time password cannot be used as a working
 * credential by anything that skips the page.
 */
export async function requireUser(
  req: Request,
  { allowPasswordChange = false } = {}
): Promise<Authorized> {
  const limited = readLimited(req);
  if (limited) return { ok: false, response: limited };
  const auth = await authenticate(req.headers);
  if (auth.kind !== 'user') {
    const response = authFailure(auth);
    return {
      ok: false,
      response: auth.kind === 'expired' ? clearSession(response, req) : response,
    };
  }
  if (auth.mustChangePassword && !allowPasswordChange) {
    return { ok: false, response: apiError(403, 'password_change_required') };
  }
  return {
    ok: true,
    config: auth.config,
    user: auth.user,
    mustChangePassword: auth.mustChangePassword,
    sessionToken: auth.sessionToken,
  };
}

export async function requireAdmin(req: Request): Promise<Authorized> {
  const result = await requireUser(req);
  if (!result.ok) return result;
  if (result.user.role !== 'admin') {
    return { ok: false, response: apiError(403, 'forbidden') };
  }
  return result;
}

/** Origin, CSRF and rate-limit checks for a state-changing request. */
export function guardMutation(
  req: Request,
  config: IdentityConfig,
  actorId: string
): NextResponse | null {
  if (!isTrustedMutationOrigin(req.headers, config.publicOrigin)) {
    return apiError(403, 'bad_origin');
  }
  if (!verifyCsrf(req.headers)) {
    return apiError(403, 'csrf');
  }
  if (!mutationLimiter.allow(`user:${actorId}`)) {
    return apiError(429, 'rate_limited');
  }
  return null;
}

/** Parse a small JSON object body; null for anything else. */
export async function readJsonObject(req: Request): Promise<Record<string, unknown> | null> {
  const length = Number(req.headers.get('content-length') ?? '0');
  if (Number.isFinite(length) && length > MAX_JSON_BODY_BYTES) {
    return null;
  }
  let text: string;
  try {
    text = await req.text();
  } catch {
    return null;
  }
  if (text.length > MAX_JSON_BODY_BYTES) {
    return null;
  }
  try {
    const body = JSON.parse(text);
    return body && typeof body === 'object' && !Array.isArray(body)
      ? (body as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

/** Translate a failed backend call into a browser response without leaking internals. */
export function backendFailure(status: number | null, data: unknown): NextResponse {
  const detail = backendDetail(data);
  switch (status) {
    case 401:
      return apiError(401, 'unauthorized');
    case 403:
      return apiError(403, detail ?? 'forbidden');
    case 404:
      return apiError(404, 'not_found');
    case 409:
      return apiError(409, detail ?? 'conflict');
    case 422:
      return apiError(422, detail ?? 'invalid');
    case 429:
      return apiError(429, 'rate_limited');
    case 503:
      return apiError(503, 'identity_not_configured');
    default:
      return apiError(502, 'backend_unavailable');
  }
}

export interface BrowserUser {
  userId: string;
  displayName: string;
  emailHint: string;
  role: string;
  status: string;
  hasCallbackNumber: boolean;
  callbackNumberUpdatedAt: number | null;
  createdAt: number | null;
  updatedAt: number | null;
  lastSeenAt: number | null;
  createdBy?: string | null;
}

/**
 * The shape a browser may see: opaque ids, a masked email hint, safe fields.
 * The raw email and the callback number never appear.
 */
export function browserUser(data: unknown, { admin = false } = {}): BrowserUser | null {
  const row = data as Record<string, unknown> | null;
  if (!row || typeof row.user_id !== 'string') return null;
  const number = (key: string): number | null =>
    typeof row[key] === 'number' && Number.isFinite(row[key]) ? (row[key] as number) : null;
  const user: BrowserUser = {
    userId: row.user_id,
    displayName: typeof row.display_name === 'string' ? row.display_name : '',
    emailHint: maskEmail(row.email),
    role: row.role === 'admin' ? 'admin' : 'member',
    status: row.status === 'suspended' ? 'suspended' : 'active',
    hasCallbackNumber: row.has_callback_number === true,
    callbackNumberUpdatedAt: number('callback_number_updated_at'),
    createdAt: number('created_at'),
    updatedAt: number('updated_at'),
    lastSeenAt: number('last_seen_at'),
  };
  if (admin) {
    user.createdBy = typeof row.created_by === 'string' ? row.created_by : null;
  }
  return user;
}

export function isDisplayName(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    value.trim().length > 0 &&
    value.trim().length <= 80 &&
    !/[\p{Cc}\p{Cf}]/u.test(value)
  );
}

export function isUserId(value: unknown): value is string {
  return typeof value === 'string' && /^usr_[0-9a-f]{24}$/.test(value);
}
