'use client';

/**
 * Browser helper for the identity routes: fetches the CSRF token once and
 * attaches it to every mutation. Errors are reduced to a short code the UI can
 * explain; response bodies are never logged.
 */

export interface ApiFailure {
  ok: false;
  status: number;
  error: string;
  /** The rest of the error body, for routes that attach bounded detail (never a secret). */
  details?: Record<string, unknown>;
}

export interface ApiSuccess<T> {
  ok: true;
  status: number;
  data: T;
}

export type ApiResult<T> = ApiSuccess<T> | ApiFailure;

let csrfToken: string | null = null;

export async function loadCsrfToken(): Promise<string | null> {
  if (csrfToken) return csrfToken;
  try {
    const response = await fetch('/api/auth/me', { cache: 'no-store', credentials: 'same-origin' });
    if (!response.ok) return null;
    const data = (await response.json()) as { csrfToken?: unknown };
    csrfToken = typeof data.csrfToken === 'string' ? data.csrfToken : null;
    return csrfToken;
  } catch {
    return null;
  }
}

export async function apiRequest<T>(
  path: string,
  init: { method?: 'GET' | 'POST' | 'PATCH' | 'PUT' | 'DELETE'; body?: unknown } = {}
): Promise<ApiResult<T>> {
  const method = init.method ?? 'GET';
  const headers: Record<string, string> = { Accept: 'application/json' };
  if (method !== 'GET') {
    const token = await loadCsrfToken();
    if (!token) {
      return { ok: false, status: 0, error: 'csrf_unavailable' };
    }
    headers['X-CAAL-CSRF'] = token;
  }
  if (init.body !== undefined) {
    headers['Content-Type'] = 'application/json';
  }
  try {
    const response = await fetch(path, {
      method,
      headers,
      cache: 'no-store',
      credentials: 'same-origin',
      ...(init.body === undefined ? {} : { body: JSON.stringify(init.body) }),
    });
    let data: unknown = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }
    if (!response.ok) {
      const body = data && typeof data === 'object' && !Array.isArray(data) ? (data as Record<string, unknown>) : null;
      const error = body?.error;
      return {
        ok: false,
        status: response.status,
        error: typeof error === 'string' ? error : `http_${response.status}`,
        ...(body ? { details: body } : {}),
      };
    }
    return { ok: true, status: response.status, data: data as T };
  } catch {
    return { ok: false, status: 0, error: 'network' };
  }
}

export const ERROR_TEXT: Record<string, string> = {
  identity_not_configured: 'Multi-user identity is not configured on the backend.',
  not_signed_in: 'You are not signed in.',
  session_expired: 'Your session has expired. Sign in again.',
  invalid_assertion: 'Your sign-in could not be verified. Reload the page.',
  // Deliberately identical for a wrong password and an unknown address: the
  // server refuses to say which, and saying it here would undo that.
  invalid_credentials: 'That email and password combination was not recognised.',
  locked: 'Too many failed attempts. Try again in a few minutes.',
  password_policy:
    'That password was refused: use at least 12 characters, and something you have not used here before.',
  password_change_required: 'You must choose a new password before continuing.',
  insecure_transport:
    'Refusing to sign you in over an unencrypted connection. Use HTTPS, or set CAAL_ALLOW_INSECURE_COOKIES=true for a trusted LAN.',
  no_account: 'There is no account for your identity. Ask an administrator to create one.',
  suspended: 'This account is suspended.',
  forbidden: 'You are not allowed to do that.',
  bad_origin: 'Request refused: unexpected origin.',
  csrf: 'Request refused: security token mismatch. Reload the page.',
  csrf_unavailable: 'Could not obtain a security token. Reload the page.',
  rate_limited: 'Too many requests. Please wait a moment.',
  invalid: 'Some fields are invalid.',
  callback_number: 'Enter a full international number, for example +1 780 555 1234.',
  duplicate: 'A user with that email already exists.',
  last_admin: 'At least one active administrator must remain.',
  not_found: 'That user no longer exists.',
  backend_unavailable: 'The backend did not respond. Try again shortly.',
  network: 'Network error. Try again.',
};

export function explain(error: string): string {
  return ERROR_TEXT[error] ?? 'Something went wrong.';
}
