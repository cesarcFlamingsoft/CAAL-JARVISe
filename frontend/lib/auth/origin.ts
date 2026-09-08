/**
 * Origin checks for mutating BFF routes.
 *
 * Browsers attach `Origin` to cross-origin and same-origin POST/PUT/PATCH/
 * DELETE requests; a request whose origin is not our own is refused before any
 * other check runs. When the operator configures `CAAL_PUBLIC_ORIGIN` that is
 * the only accepted origin; otherwise it is derived from the forwarded host.
 */

function originOf(value: string | null): string | null {
  if (!value || value === 'null') {
    return null;
  }
  try {
    const url = new URL(value);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      return null;
    }
    return url.origin;
  } catch {
    return null;
  }
}

/** The origin our own pages are served from, as this request sees it. */
export function expectedOrigin(headers: Headers, publicOrigin?: string | null): string | null {
  if (publicOrigin) {
    return originOf(publicOrigin);
  }
  const host = headers.get('host');
  if (!host || !/^[A-Za-z0-9.\-:[\]]+$/.test(host)) {
    return null;
  }
  const forwarded = headers.get('x-forwarded-proto');
  const proto = forwarded === 'https' ? 'https' : 'http';
  return originOf(`${proto}://${host}`);
}

/**
 * True only when the request's `Origin` (or, failing that, `Referer`) matches
 * our own origin exactly. A request with neither header is refused.
 */
export function isTrustedMutationOrigin(headers: Headers, publicOrigin?: string | null): boolean {
  const expected = expectedOrigin(headers, publicOrigin);
  if (!expected) {
    return false;
  }
  const origin = headers.get('origin');
  if (origin !== null) {
    return originOf(origin) === expected;
  }
  const referer = originOf(headers.get('referer'));
  return referer !== null && referer === expected;
}
