/**
 * Server-only client for the CAAL identity API.
 *
 * Every call carries a fresh, single-use signed principal. Failures are
 * reduced to a status and a small code so a route can map them to a bounded
 * browser response without echoing backend internals. Nothing here logs
 * request or response bodies: they can contain emails and phone numbers.
 */
import type { IdentityConfig } from './config';
import { AUDIENCE_BACKEND, AUDIENCE_IDENTITY, mintPrincipal } from './principal';

const BACKEND_TIMEOUT_MS = 5000;
/** Argon2id is deliberately slow; a sign-in needs more room than a read. */
const LOGIN_TIMEOUT_MS = 15_000;

export type BackendResult =
  | { ok: true; status: number; data: unknown }
  | { ok: false; status: number; data: unknown }
  | { ok: false; status: null; data: null };

interface CallOptions {
  method: 'GET' | 'POST' | 'PATCH' | 'PUT' | 'DELETE';
  body?: unknown;
  headers?: Record<string, string>;
  timeoutMs?: number;
}

async function call(
  config: IdentityConfig,
  path: string,
  init: CallOptions
): Promise<BackendResult> {
  let url: URL;
  try {
    url = new URL(path.replace(/^\//, ''), `${config.apiBaseUrl}/`);
  } catch {
    console.error('[identity] backend URL is not valid');
    return { ok: false, status: null, data: null };
  }
  let response: Response;
  try {
    response = await fetch(url, {
      method: init.method,
      cache: 'no-store',
      redirect: 'error',
      signal: AbortSignal.timeout(init.timeoutMs ?? BACKEND_TIMEOUT_MS),
      headers: {
        Accept: 'application/json',
        ...(init.body === undefined ? {} : { 'Content-Type': 'application/json' }),
        ...init.headers,
      },
      ...(init.body === undefined ? {} : { body: JSON.stringify(init.body) }),
    });
  } catch {
    console.error(`[identity] request to ${path} failed`);
    return { ok: false, status: null, data: null };
  }
  let data: unknown = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  return { ok: response.ok, status: response.status, data };
}

/** Resolve a verified Access identity to its opaque user (bootstrapping the admin once). */
export async function resolveIdentity(
  config: IdentityConfig,
  email: string,
  accessAssertion: string
): Promise<BackendResult> {
  const assertion = await mintPrincipal({
    secret: config.internalAuthSecret,
    subject: 'identity',
    audience: AUDIENCE_IDENTITY,
    claims: { email },
  });
  return call(config, '/auth/resolve', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${assertion}`,
      'Cf-Access-Jwt-Assertion': accessAssertion,
    },
  });
}

/** Call the identity API as `userId`, with a fresh single-use backend principal. */
export async function callAsUser(
  config: IdentityConfig,
  userId: string,
  path: string,
  init: CallOptions
): Promise<BackendResult> {
  const principal = await mintPrincipal({
    secret: config.internalAuthSecret,
    subject: userId,
    audience: AUDIENCE_BACKEND,
  });
  return call(config, path, {
    ...init,
    headers: { ...init.headers, Authorization: `Bearer ${principal}` },
  });
}

/** A backend principal for routes that hand it to another CAAL API (device registry). */
export async function backendPrincipalFor(config: IdentityConfig, userId: string): Promise<string> {
  return mintPrincipal({
    secret: config.internalAuthSecret,
    subject: userId,
    audience: AUDIENCE_BACKEND,
  });
}

/** Read a string field out of an untrusted backend payload. */
export function backendString(data: unknown, key: string): string | null {
  const value = (data as Record<string, unknown> | null)?.[key];
  return typeof value === 'string' && value.length > 0 ? value : null;
}

/** The `detail` code the backend attaches to a refusal, if it is a short token. */
export function backendDetail(data: unknown): string | null {
  const detail = (data as { detail?: unknown } | null)?.detail;
  return typeof detail === 'string' && /^[a-z_]{1,40}$/.test(detail) ? detail : null;
}

// --- standalone password sign-in -------------------------------------------------

/**
 * A stable, non-identifying key for the *browser* making this request.
 *
 * Every sign-in reaches the backend through this one BFF, so the backend's
 * view of the peer address is useless for rate limiting. We therefore sign the
 * browser's own key into the assertion -- but hashed, so the backend's
 * limiter, logs and audit trail never hold a raw client address.
 */
export async function clientKeyFor(headers: Headers): Promise<string> {
  const cf = headers.get('cf-connecting-ip');
  const forwarded = headers.get('x-forwarded-for')?.split(',')[0]?.trim();
  const raw =
    (cf && cf.length <= 64 ? cf : forwarded && forwarded.length <= 64 ? forwarded : '') ||
    'unknown';
  const digest = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(raw));
  return Array.from(new Uint8Array(digest).slice(0, 16))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

async function identityHeaders(
  config: IdentityConfig,
  clientKey: string
): Promise<Record<string, string>> {
  const assertion = await mintPrincipal({
    secret: config.internalAuthSecret,
    subject: 'identity',
    audience: AUDIENCE_IDENTITY,
    claims: { client: clientKey },
  });
  return { Authorization: `Bearer ${assertion}` };
}

/** Check a password. The plaintext is forwarded once and never stored or logged. */
export async function login(
  config: IdentityConfig,
  email: string,
  password: string,
  clientKey: string
): Promise<BackendResult> {
  return call(config, '/auth/login', {
    method: 'POST',
    headers: await identityHeaders(config, clientKey),
    body: { email, password },
    timeoutMs: LOGIN_TIMEOUT_MS,
  });
}

/** Re-resolve an opaque session token against the backend's database. */
export async function readSession(
  config: IdentityConfig,
  sessionToken: string,
  clientKey: string
): Promise<BackendResult> {
  return call(config, '/auth/session', {
    method: 'POST',
    headers: await identityHeaders(config, clientKey),
    body: { session_token: sessionToken },
  });
}

/** Revoke a session server-side. Unknown tokens succeed: sign-out is not a probe. */
export async function logout(
  config: IdentityConfig,
  sessionToken: string,
  clientKey: string
): Promise<BackendResult> {
  return call(config, '/auth/logout', {
    method: 'POST',
    headers: await identityHeaders(config, clientKey),
    body: { session_token: sessionToken },
  });
}

/** Change your own password. The principal, not the body, decides whose. */
export async function changePassword(
  config: IdentityConfig,
  userId: string,
  input: { currentPassword: string; newPassword: string; keepSessionToken?: string | null }
): Promise<BackendResult> {
  const principal = await mintPrincipal({
    secret: config.internalAuthSecret,
    subject: userId,
    audience: AUDIENCE_BACKEND,
  });
  return call(config, '/auth/password', {
    method: 'POST',
    headers: { Authorization: `Bearer ${principal}` },
    body: {
      current_password: input.currentPassword,
      new_password: input.newPassword,
      ...(input.keepSessionToken ? { keep_session_token: input.keepSessionToken } : {}),
    },
    timeoutMs: LOGIN_TIMEOUT_MS,
  });
}
