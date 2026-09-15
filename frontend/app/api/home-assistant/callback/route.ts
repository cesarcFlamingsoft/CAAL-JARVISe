import { type NextRequest, NextResponse } from 'next/server';
import { ACCESS_ASSERTION_HEADER } from '@/lib/auth/access';
import { callAsUser } from '@/lib/auth/backend';
import { readCookie } from '@/lib/auth/csrf';
import { isSecureRequest, readLimited, securityHeaders } from '@/lib/auth/guard';
import { authenticate } from '@/lib/auth/session';
import { openFlow, sessionKeyFor } from '@/lib/connections/flow-cookie';

export const dynamic = 'force-dynamic';
function finish(req: NextRequest, status: string) {
  const res = new NextResponse(null, {
    status: 303,
    headers: { ...securityHeaders(), Location: '/home-assistant/result?status=' + status },
  });
  res.cookies.set({
    name: 'caal_ha_flow',
    value: '',
    maxAge: 0,
    httpOnly: true,
    sameSite: 'lax',
    secure: isSecureRequest(req),
    path: '/api/home-assistant/callback',
  });
  return res;
}
export async function GET(req: NextRequest) {
  if (readLimited(req)) return finish(req, 'rate_limited');
  const auth = await authenticate(req.headers);
  if (auth.kind !== 'user') return finish(req, 'sign_in_required');
  try {
    const sessionKey = await sessionKeyFor({
      sessionToken: auth.sessionToken,
      accessAssertion: req.headers.get(ACCESS_ASSERTION_HEADER),
    });
    const flow = await openFlow(
      auth.config.internalAuthSecret,
      readCookie(req.headers.get('cookie'), 'caal_ha_flow'),
      { userId: auth.user.userId, sessionKey, now: Math.floor(Date.now() / 1000) }
    );
    const state = req.nextUrl.searchParams.get('state'),
      code = req.nextUrl.searchParams.get('code');
    if (
      !flow ||
      flow.provider !== 'homeassistant' ||
      flow.stateId !== state ||
      !code ||
      code.length > 4096
    )
      return finish(req, 'not_initiated_here');
    const result = await callAsUser(
      auth.config,
      auth.user.userId,
      '/users/me/home-assistant/callback',
      { method: 'POST', body: { state, code }, timeoutMs: 25000 }
    );
    return finish(req, result.ok ? 'connected' : 'authorization_failed');
  } catch {
    return finish(req, 'authorization_failed');
  }
}
