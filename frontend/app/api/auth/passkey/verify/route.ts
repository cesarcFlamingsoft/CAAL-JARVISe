import type { NextRequest } from 'next/server';
import { clientKeyFor, finishPasskeyLogin } from '@/lib/auth/backend';
import { readIdentityConfig } from '@/lib/auth/config';
import { csrfCookieOptions, issueCsrfToken } from '@/lib/auth/csrf';
import {
  apiError,
  cookieSecurity,
  guardReadPost,
  isSecureRequest,
  loginLimited,
  noStoreJson,
  readJsonObject,
} from '@/lib/auth/guard';
import { safeNextPath, sessionCookieOptions } from '@/lib/auth/session-cookie';

export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  const limited = loginLimited(req);
  if (limited) return limited;
  const status = readIdentityConfig();
  if (
    status.status !== 'enabled' ||
    !status.config.passwordLogin ||
    !status.config.publicOrigin?.startsWith('https://')
  ) {
    return apiError(404, 'not_found');
  }
  const config = status.config;
  const blocked = guardReadPost(req, config);
  if (blocked) return blocked;
  const transport = cookieSecurity(
    req.headers,
    req.url,
    config.allowInsecureCookies,
    config.trustedLocalOrigins
  );
  if (!transport.ok) return apiError(400, transport.reason);
  const body = await readJsonObject(req);
  if (
    typeof body?.ceremonyId !== 'string' ||
    !body.credential ||
    typeof body.credential !== 'object'
  ) {
    return apiError(401, 'invalid_credentials');
  }
  const result = await finishPasskeyLogin(
    config,
    { ceremonyId: body.ceremonyId, credential: body.credential },
    await clientKeyFor(req.headers)
  );
  if (!result.ok) {
    if (result.status === 401) return apiError(401, 'invalid_credentials');
    if (result.status === 429) return apiError(429, 'rate_limited');
    return apiError(502, 'backend_unavailable');
  }
  const data = result.data as Record<string, unknown> | null;
  const token = typeof data?.session_token === 'string' ? data.session_token : null;
  const expiresAt = typeof data?.expires_at === 'number' ? data.expires_at : 0;
  if (!token || typeof data?.user_id !== 'string') return apiError(502, 'backend_unavailable');
  const rotated = issueCsrfToken();
  const res = noStoreJson({ ok: true, next: safeNextPath(body?.next), csrfToken: rotated });
  res.cookies.set({
    ...sessionCookieOptions({
      secure: transport.secure,
      maxAge: Math.max(0, expiresAt - Math.floor(Date.now() / 1000)),
    }),
    value: token,
  });
  res.cookies.set({ ...csrfCookieOptions({ secure: isSecureRequest(req) }), value: rotated });
  return res;
}
