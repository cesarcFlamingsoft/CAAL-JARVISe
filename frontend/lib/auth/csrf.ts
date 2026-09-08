/**
 * Double-submit CSRF protection for the BFF's mutating routes.
 *
 * The token lives in an HttpOnly, SameSite=Strict cookie the browser cannot
 * read, and is handed to page scripts once through the JSON body of
 * `GET /api/auth/me` (which only same-origin scripts can read). A mutation must
 * echo it back in the `X-CAAL-CSRF` header; the two are compared in constant
 * time. Cross-site pages can neither read the token nor make the browser send
 * the header, so a forged request always fails here even if the Access cookie
 * were sent along.
 */

export const CSRF_COOKIE = 'caal_csrf';
export const CSRF_HEADER = 'x-caal-csrf';
export const CSRF_MAX_AGE_SECONDS = 12 * 3600;
const TOKEN_BYTES = 32;

function base64url(bytes: Uint8Array): string {
  let binary = '';
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function issueCsrfToken(): string {
  const buffer = new Uint8Array(TOKEN_BYTES);
  globalThis.crypto.getRandomValues(buffer);
  return base64url(buffer);
}

export interface CsrfCookieOptions {
  name: string;
  httpOnly: true;
  sameSite: 'strict';
  secure: boolean;
  path: string;
  maxAge: number;
}

export function csrfCookieOptions({ secure }: { secure: boolean }): CsrfCookieOptions {
  return {
    name: CSRF_COOKIE,
    httpOnly: true,
    sameSite: 'strict',
    secure,
    path: '/',
    maxAge: CSRF_MAX_AGE_SECONDS,
  };
}

/** Read one cookie value from a raw `Cookie` header; null when absent or empty. */
export function readCookie(header: string | null, name: string): string | null {
  if (!header) {
    return null;
  }
  for (const part of header.split(';')) {
    const separator = part.indexOf('=');
    if (separator === -1) {
      continue;
    }
    if (part.slice(0, separator).trim() !== name) {
      continue;
    }
    let value = part.slice(separator + 1).trim();
    try {
      value = decodeURIComponent(value);
    } catch {
      return null;
    }
    return value.length > 0 ? value : null;
  }
  return null;
}

/** Constant-time equality for two strings of equal length. */
export function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }
  let mismatch = 0;
  for (let index = 0; index < a.length; index++) {
    mismatch |= a.charCodeAt(index) ^ b.charCodeAt(index);
  }
  return mismatch === 0;
}

/** Whether the request carries a header token matching its CSRF cookie. */
export function verifyCsrf(headers: Headers): boolean {
  const cookie = readCookie(headers.get('cookie'), CSRF_COOKIE);
  const header = headers.get(CSRF_HEADER);
  if (!cookie || !header || cookie.length < 16) {
    return false;
  }
  return constantTimeEqual(cookie, header.trim());
}
