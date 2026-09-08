/**
 * Change your own password.
 *
 * The account is decided by the authenticated session, never by the request
 * body, so this route cannot be aimed at anyone else. Succeeding revokes every
 * *other* session that user holds, which is what makes "change my password"
 * an effective response to a lost or shared device.
 *
 * This is one of the two routes a user with a forced-change flag may still
 * reach; everything else refuses them until they have been through here.
 */
import type { NextRequest } from 'next/server';
import { changePassword } from '@/lib/auth/backend';
import { csrfCookieOptions, issueCsrfToken } from '@/lib/auth/csrf';
import {
  apiError,
  guardMutation,
  isSecureRequest,
  noStoreJson,
  readJsonObject,
  requireUser,
} from '@/lib/auth/guard';

export const dynamic = 'force-dynamic';

const MAX_PASSWORD_LENGTH = 256;

export async function POST(req: NextRequest) {
  const auth = await requireUser(req, { allowPasswordChange: true });
  if (!auth.ok) return auth.response;

  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  if (!auth.config.passwordLogin) {
    return apiError(404, 'not_found');
  }

  const body = await readJsonObject(req);
  const currentPassword = body?.currentPassword;
  const newPassword = body?.newPassword;
  if (
    typeof currentPassword !== 'string' ||
    typeof newPassword !== 'string' ||
    currentPassword.length === 0 ||
    currentPassword.length > MAX_PASSWORD_LENGTH ||
    newPassword.length === 0 ||
    newPassword.length > MAX_PASSWORD_LENGTH
  ) {
    return apiError(422, 'password_policy');
  }

  const result = await changePassword(auth.config, auth.user.userId, {
    currentPassword,
    newPassword,
    // Keeping this session alive means changing a password does not sign you
    // out of the tab you are typing in, while every other session dies.
    keepSessionToken: auth.sessionToken,
  });

  if (!result.ok) {
    if (result.status === 401) return apiError(401, 'invalid_credentials');
    if (result.status === 422) return apiError(422, 'password_policy');
    if (result.status === 429) {
      const detail = (result.data as { detail?: unknown } | null)?.detail;
      return apiError(429, detail === 'locked' ? 'locked' : 'rate_limited');
    }
    if (result.status === 404) return apiError(404, 'not_found');
    return apiError(502, 'backend_unavailable');
  }

  // The credential changed; rotate the CSRF token with it.
  const rotated = issueCsrfToken();
  const res = noStoreJson({ ok: true, next: '/', csrfToken: rotated });
  res.cookies.set({ ...csrfCookieOptions({ secure: isSecureRequest(req) }), value: rotated });
  return res;
}
