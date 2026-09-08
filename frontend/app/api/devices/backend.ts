/**
 * Server-only glue between the browser and the CAAL device registry.
 *
 * Everything secret lives here and never leaves the server: the CAAL base URL,
 * the enrollment secret, and the bearer session token minted by the backend.
 * The token is written straight into an HttpOnly cookie (see
 * `@/lib/device-session`) and is never placed in a response body.
 *
 * This module is not a route — Next only routes `route.ts` files.
 */
import { NextResponse } from 'next/server';
import {
  DEVICE_SESSION_COOKIE,
  clampSessionLifetime,
  clearedSessionCookieOptions,
  sessionCookieOptions,
} from '@/lib/device-session';

/** How long we wait on the backend before giving up on a device call. */
const BACKEND_TIMEOUT_MS = 5000;

/** Longest room name we will forward; LiveKit names are far shorter than this. */
const MAX_ROOM_NAME_LENGTH = 128;

/** Printable ASCII only — a room name is machine-generated, not free prose. */
const PRINTABLE_ASCII = /^[\x20-\x7e]+$/;

/**
 * Responses the browser is allowed to see. They are deliberately uniform: a
 * caller cannot tell a misconfigured base URL from a refused enrollment secret
 * from a backend that is simply down.
 */
export const DEVICE_ERROR = {
  invalidRequest: 'Invalid device request.',
  noSession: 'No active device session.',
  unavailable: 'Device service is unavailable.',
} as const;

export function noStoreJson(body: unknown, status = 200): NextResponse {
  return NextResponse.json(body, {
    status,
    headers: { 'Cache-Control': 'no-store' },
  });
}

export function deviceError(
  message: (typeof DEVICE_ERROR)[keyof typeof DEVICE_ERROR],
  status: number
): NextResponse {
  return noStoreJson({ error: message }, status);
}

/** Cookies are only marked Secure when the browser actually reached us over TLS. */
export function isSecureRequest(req: Request): boolean {
  return req.headers.get('x-forwarded-proto') === 'https' || req.url.startsWith('https://');
}

/** Attach the freshly minted session token to the response as an HttpOnly cookie. */
export function setSessionCookie(
  res: NextResponse,
  token: string,
  expiresIn: unknown,
  req: Request
) {
  res.cookies.set({
    ...sessionCookieOptions(clampSessionLifetime(expiresIn), { secure: isSecureRequest(req) }),
    value: token,
  });
}

/** Expire the session cookie; used whenever the backend stops honouring it. */
export function clearSessionCookie(res: NextResponse, req: Request) {
  res.cookies.set({
    ...clearedSessionCookieOptions({ secure: isSecureRequest(req) }),
    value: '',
  });
}

/** The bearer token for this browser, or null when the cookie is absent or empty. */
export function readSessionToken(req: Request): string | null {
  const raw = req.headers.get('cookie');
  if (!raw) {
    return null;
  }

  for (const part of raw.split(';')) {
    const separator = part.indexOf('=');
    if (separator === -1) {
      continue;
    }
    if (part.slice(0, separator).trim() !== DEVICE_SESSION_COOKIE) {
      continue;
    }
    const value = decodeURIComponent(part.slice(separator + 1).trim());
    return value.length > 0 ? value : null;
  }
  return null;
}

/** Validate a room name as bounded, single-line printable text. */
export function validateRoomName(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null;
  }
  const roomName = value.trim();
  if (!roomName || roomName.length > MAX_ROOM_NAME_LENGTH || !PRINTABLE_ASCII.test(roomName)) {
    return null;
  }
  return roomName;
}

export type BackendResult =
  | { ok: true; status: number; data: unknown }
  | { ok: false; status: number; data: unknown }
  | { ok: false; status: null; data: null };

/**
 * Call the CAAL device API. A `status: null` result means we never got an
 * answer (missing configuration, timeout, connection refused) — callers report
 * all of those to the browser identically.
 */
export async function callDeviceApi(
  path: string,
  init: { method: 'GET' | 'POST'; headers?: Record<string, string>; body?: unknown }
): Promise<BackendResult> {
  const baseUrl = process.env.CAAL_DEVICE_API_URL?.trim();
  if (!baseUrl) {
    console.error('[devices] CAAL_DEVICE_API_URL is not set');
    return { ok: false, status: null, data: null };
  }

  let url: URL;
  try {
    url = new URL(path.replace(/^\//, ''), baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`);
  } catch {
    console.error('[devices] CAAL_DEVICE_API_URL is not a valid URL');
    return { ok: false, status: null, data: null };
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    console.error('[devices] CAAL_DEVICE_API_URL must be http or https');
    return { ok: false, status: null, data: null };
  }

  let response: Response;
  try {
    response = await fetch(url, {
      method: init.method,
      cache: 'no-store',
      signal: AbortSignal.timeout(BACKEND_TIMEOUT_MS),
      headers: {
        Accept: 'application/json',
        ...(init.body === undefined ? {} : { 'Content-Type': 'application/json' }),
        ...init.headers,
      },
      ...(init.body === undefined ? {} : { body: JSON.stringify(init.body) }),
    });
  } catch {
    // Deliberately not logging the error: it can echo the configured URL.
    console.error(`[devices] request to ${path} failed`);
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

/** The enrollment secret, or null when it is not configured. */
export function enrollmentToken(): string | null {
  const token = process.env.CAAL_DEVICE_ENROLLMENT_TOKEN?.trim();
  if (!token) {
    console.error('[devices] CAAL_DEVICE_ENROLLMENT_TOKEN is not set');
    return null;
  }
  return token;
}

/** Read a plain string field from an untrusted backend payload. */
export function backendString(data: unknown, key: string): string | null {
  const value = (data as Record<string, unknown> | null)?.[key];
  return typeof value === 'string' && value.length > 0 ? value : null;
}

/** Parse a JSON request body, returning null for anything that is not an object. */
export async function readJsonBody(req: Request): Promise<Record<string, unknown> | null> {
  try {
    const body = await req.json();
    return body && typeof body === 'object' && !Array.isArray(body)
      ? (body as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}
