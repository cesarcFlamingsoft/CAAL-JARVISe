/**
 * Administrator: issue a one-time password for a user.
 *
 * This is the deliberate replacement for self-service password reset. CAAL has
 * no mail transport, so an emailed reset link would be a recovery channel the
 * deployment cannot secure; instead an administrator generates a random
 * password out of band and hands it over through whatever channel they already
 * trust.
 *
 * The generated password exists in exactly one response, is never stored in
 * the clear, never logged, and never returned again. Issuing one revokes the
 * user's live sessions and forces a change at their next sign-in.
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardMutation,
  isUserId,
  noStoreJson,
  requireAdmin,
} from '@/lib/auth/guard';

export const dynamic = 'force-dynamic';

type Params = { params: Promise<{ id: string }> };

export async function POST(req: NextRequest, { params }: Params) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  const { id } = await params;
  if (!isUserId(id)) return apiError(404, 'not_found');
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;
  if (!auth.config.passwordLogin) return apiError(404, 'not_found');

  const result = await callAsUser(auth.config, auth.user.userId, `/admin/users/${id}/password`, {
    method: 'POST',
  });
  if (!result.ok) return backendFailure(result.status, result.data);

  const issued = (result.data as { one_time_password?: unknown } | null)?.one_time_password;
  if (typeof issued !== 'string' || issued.length < 12) {
    return apiError(502, 'backend_unavailable');
  }
  return noStoreJson({ userId: id, oneTimePassword: issued });
}
