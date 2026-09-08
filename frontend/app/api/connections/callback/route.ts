/**
 * Where a provider sends the browser back after the user approves (or
 * declines). This is the exact path every provider's `redirect_uri` is built
 * from on the backend (`caal.oauth_providers.OAUTH_CALLBACK_PATH`).
 *
 * The order of checks is deliberate:
 *
 * 1. Who is this? The session cookie is `SameSite=Lax`, so it rides the
 *    provider's top-level redirect. Anyone not signed in is sent to the
 *    result page and nothing is forwarded.
 * 2. Did *this browser session* start a flow? The flow cookie set by the
 *    start route is opened under the signed-in user's id and session key.
 * 3. What did the provider say? The query is parsed into a bounded shape:
 *    a code with a well-formed state, a refusal, a provider error, or
 *    nothing usable. A provider's free-text error is never read.
 * 4. Only a code whose state id matches the flow cookie is forwarded to the
 *    backend, which redeems the state (once, for this user) and answers.
 *
 * Every answer is a 303 to `/connections/result` carrying only an outcome
 * code and a provider name, and clears the flow cookie. The authorization
 * code never appears in a response, a redirect target, or a log line.
 */
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { ACCESS_ASSERTION_HEADER } from '@/lib/auth/access';
import { callAsUser } from '@/lib/auth/backend';
import { readCookie } from '@/lib/auth/csrf';
import { clearSession, isSecureRequest, readLimited, securityHeaders } from '@/lib/auth/guard';
import { authenticate } from '@/lib/auth/session';
import {
  FLOW_COOKIE,
  type OpenedFlow,
  clearedFlowCookie,
  openFlow,
  sessionKeyFor,
} from '@/lib/connections/flow-cookie';
import {
  type Outcome,
  type Provider,
  isProvider,
  outcomeFromAuth,
  outcomeFromBackend,
  parseCallbackQuery,
  resultPath,
} from '@/lib/connections/protocol';

export const dynamic = 'force-dynamic';

/**
 * How long to wait for the backend to redeem the code. Its exchange is
 * bounded at two provider round trips of `caal.oauth_exchange.TIMEOUT_SECONDS`
 * (10 s) each; waiting less would report a connection the backend went on to
 * complete as "backend did not respond".
 */
const CALLBACK_TIMEOUT_MS = 25_000;

/** A 303 to the result page, carrying only codes, that also ends the flow. */
function finish(req: NextRequest, outcome: Outcome, provider: Provider | null): NextResponse {
  const res = new NextResponse(null, {
    status: 303,
    headers: { ...securityHeaders(), Location: resultPath(outcome, provider) },
  });
  res.cookies.set(clearedFlowCookie({ secure: isSecureRequest(req) }));
  return res;
}

export async function GET(req: NextRequest) {
  if (readLimited(req)) return finish(req, 'rate_limited', null);

  const auth = await authenticate(req.headers);
  const refused = outcomeFromAuth(auth);
  if (refused !== null || auth.kind !== 'user') {
    const res = finish(req, refused ?? 'not_signed_in', null);
    return auth.kind === 'expired' ? clearSession(res, req) : res;
  }

  let flow: OpenedFlow | null = null;
  try {
    const sessionKey = await sessionKeyFor({
      sessionToken: auth.sessionToken,
      accessAssertion: req.headers.get(ACCESS_ASSERTION_HEADER),
    });
    flow = await openFlow(
      auth.config.internalAuthSecret,
      readCookie(req.headers.get('cookie'), FLOW_COOKIE),
      { userId: auth.user.userId, sessionKey, now: Math.floor(Date.now() / 1000) }
    );
  } catch {
    flow = null;
  }
  const provider = flow && isProvider(flow.provider) ? flow.provider : null;

  const query = parseCallbackQuery(req.nextUrl.searchParams);
  if (query.kind === 'invalid') return finish(req, 'invalid_callback', provider);
  if (query.kind === 'denied') return finish(req, 'denied', provider);
  if (query.kind === 'unsupported_scope') return finish(req, 'unsupported_scope', provider);
  if (query.kind === 'provider_error') return finish(req, 'provider_error', provider);
  if (!flow || provider === null || flow.stateId !== query.stateId) {
    return finish(req, 'not_initiated_here', provider);
  }

  // Zoho names the data center that issued the code; the backend checks it
  // against the accounts server it is configured for. Only the two-letter
  // code travels -- never a provider-supplied host.
  const location = provider === 'zoho' ? query.location : null;
  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/connections/callback', {
    method: 'POST',
    body: { state: query.state, code: query.code, ...(location ? { location } : {}) },
    timeoutMs: CALLBACK_TIMEOUT_MS,
  });
  return finish(req, outcomeFromBackend(result.status, result.data), provider);
}
