import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardMutation,
  noStoreJson,
  readJsonObject,
  requireUser,
} from '@/lib/auth/guard';

export const dynamic = 'force-dynamic';
const MAX_PASSWORD_LENGTH = 256;

export async function POST(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;
  if (!auth.config.passwordLogin || !auth.config.publicOrigin?.startsWith('https://')) {
    return apiError(404, 'not_found');
  }
  const body = await readJsonObject(req);
  if (
    typeof body?.label !== 'string' ||
    !body.label.trim() ||
    body.label.length > 64 ||
    typeof body?.currentPassword !== 'string' ||
    !body.currentPassword ||
    body.currentPassword.length > MAX_PASSWORD_LENGTH
  ) {
    return apiError(422, 'invalid');
  }
  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/passkeys/options', {
    method: 'POST',
    body: { label: body.label, current_password: body.currentPassword },
    sessionBinding: auth.sessionBinding,
  });
  return result.ok ? noStoreJson(result.data) : backendFailure(result.status, result.data);
}
