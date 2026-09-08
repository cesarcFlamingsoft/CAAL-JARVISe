/**
 * Start linking a provider account to the signed-in user.
 *
 * The backend mints an opaque, signed, user-bound `state` and the provider's
 * authorization URL. This route hands the URL to the page to navigate to, and
 * sets a short-lived cookie that binds the pending flow to *this* browser
 * session; the callback route refuses a return that does not carry it.
 *
 * A provider the operator has not configured is answered explicitly, naming
 * the missing settings by variable name, so nothing is attempted half set up.
 * No request body is read: there is nothing a browser should be able to tell
 * this route beyond which provider.
 */
import type { NextRequest } from 'next/server';
import { ACCESS_ASSERTION_HEADER } from '@/lib/auth/access';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardMutation,
  isSecureRequest,
  noStoreJson,
  requireUser,
} from '@/lib/auth/guard';
import { flowCookieOptions, sealFlow, sessionKeyFor } from '@/lib/connections/flow-cookie';
import { configurationNeeded, isProvider, parseAuthorization } from '@/lib/connections/protocol';

export const dynamic = 'force-dynamic';

/** Survive a little clock skew between the BFF and the backend. */
const MIN_COOKIE_SECONDS = 60;

type Params = { params: Promise<{ provider: string }> };

export async function POST(req: NextRequest, { params }: Params) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const { provider } = await params;
  if (!isProvider(provider)) return apiError(404, 'unknown_provider');
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const result = await callAsUser(
    auth.config,
    auth.user.userId,
    `/users/me/connections/${provider}/authorize`,
    { method: 'POST' }
  );
  if (!result.ok) {
    const needed = result.status === 503 ? configurationNeeded(result.data) : null;
    if (needed) {
      return apiError(503, 'configuration_needed', {
        provider: needed.provider,
        missing: needed.missing,
      });
    }
    if (result.status === 404) return apiError(404, 'unknown_provider');
    return backendFailure(result.status, result.data);
  }

  const authorization = parseAuthorization(provider, result.data);
  if (!authorization) return apiError(502, 'backend_unavailable');

  let cookieValue: string;
  try {
    const sessionKey = await sessionKeyFor({
      sessionToken: auth.sessionToken,
      accessAssertion: req.headers.get(ACCESS_ASSERTION_HEADER),
    });
    cookieValue = await sealFlow(auth.config.internalAuthSecret, {
      userId: auth.user.userId,
      sessionKey,
      provider,
      stateId: authorization.stateId,
      expiresAt: authorization.expiresAt,
    });
  } catch {
    return apiError(500, 'flow_unavailable');
  }

  const now = Math.floor(Date.now() / 1000);
  const res = noStoreJson({
    provider,
    authorizationUrl: authorization.authorizationUrl,
    expiresAt: authorization.expiresAt,
  });
  res.cookies.set({
    ...flowCookieOptions({
      secure: isSecureRequest(req),
      maxAge: Math.max(MIN_COOKIE_SECONDS, authorization.expiresAt - now),
    }),
    value: cookieValue,
  });
  return res;
}
