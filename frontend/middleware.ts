import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

/**
 * A courtesy redirect, not an authorization check.
 *
 * Middleware runs on the edge and deliberately makes no backend call, so all
 * it can see is whether a session cookie is *present* -- never whether it is
 * valid. Sending someone with no cookie at all straight to the sign-in page
 * saves them a round trip through a page that would only tell them the same
 * thing.
 *
 * Every real decision is taken again, server-side, in the page and in the API
 * route, from a fresh backend resolution. Nothing here may be relied on for
 * protection: a forged cookie gets past this and then fails properly.
 */

const SESSION_COOKIE = 'caal_session';
const ACCESS_ASSERTION_HEADER = 'cf-access-jwt-assertion';

/** Pages that are useless without a user, so an anonymous visitor is redirected. */
const PROTECTED = ['/account', '/admin', '/change-password'];

export function middleware(req: NextRequest) {
  const { pathname, search } = req.nextUrl;
  if (!PROTECTED.some((base) => pathname === base || pathname.startsWith(`${base}/`))) {
    return NextResponse.next();
  }
  // A Cloudflare Access deployment authenticates with a header and no cookie;
  // let the page resolve that itself rather than bouncing the user.
  if (req.headers.has(ACCESS_ASSERTION_HEADER)) {
    return NextResponse.next();
  }
  if (req.cookies.has(SESSION_COOKIE)) {
    return NextResponse.next();
  }

  const url = req.nextUrl.clone();
  url.pathname = '/login';
  url.search = '';
  // `pathname` comes from the router, not from user input, so it is already a
  // same-origin path; the sign-in page validates it again regardless.
  url.searchParams.set('next', `${pathname}${search}`);
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ['/account/:path*', '/admin/:path*', '/change-password/:path*'],
};
