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

export async function POST(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;
  const body = await readJsonObject(req);
  if (
    typeof body?.label !== 'string' ||
    body.label.length > 64 ||
    typeof body?.ceremonyId !== 'string' ||
    !body.credential ||
    typeof body.credential !== 'object'
  )
    return apiError(422, 'invalid');
  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/passkeys/verify', {
    method: 'POST',
    body: { label: body.label, ceremonyId: body.ceremonyId, credential: body.credential },
    sessionBinding: auth.sessionBinding,
  });
  return result.ok ? noStoreJson(result.data) : backendFailure(result.status, result.data);
}
