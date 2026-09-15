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
import { sealFlow, sessionKeyFor } from '@/lib/connections/flow-cookie';
import { authorizationOriginAllowed } from '@/lib/home-assistant/contract';

export const dynamic = 'force-dynamic';
export async function POST(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;
  if (!authorizationOriginAllowed(req.headers.get('origin'), auth.config.publicOrigin))
    return apiError(409, 'ha_use_public_origin');
  const r = await callAsUser(auth.config, auth.user.userId, '/users/me/home-assistant/authorize', {
    method: 'POST',
  });
  if (!r.ok) return backendFailure(r.status, r.data);
  const d = r.data as Record<string, unknown>;
  if (
    typeof d.authorization_url !== 'string' ||
    d.authorization_url.length > 2000 ||
    typeof d.state_id !== 'string' ||
    !/^[A-Za-z0-9_-]{32}$/.test(d.state_id) ||
    typeof d.expires_at !== 'number'
  )
    return apiError(502, 'backend_unavailable');
  const url = new URL(d.authorization_url);
  if (
    !['http:', 'https:'].includes(url.protocol) ||
    url.pathname != '/auth/authorize' ||
    url.username ||
    url.password
  )
    return apiError(502, 'backend_unavailable');
  const sessionKey = await sessionKeyFor({
    sessionToken: auth.sessionToken,
    accessAssertion: req.headers.get(ACCESS_ASSERTION_HEADER),
  });
  const cookie = await sealFlow(auth.config.internalAuthSecret, {
    userId: auth.user.userId,
    sessionKey,
    provider: 'homeassistant',
    stateId: d.state_id,
    expiresAt: d.expires_at,
  });
  const res = noStoreJson({ authorizationUrl: d.authorization_url });
  res.cookies.set({
    name: 'caal_ha_flow',
    value: cookie,
    httpOnly: true,
    sameSite: 'lax',
    secure: isSecureRequest(req),
    path: '/api/home-assistant/callback',
    maxAge: 600,
  });
  return res;
}
