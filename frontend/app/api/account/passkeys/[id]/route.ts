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
const KEY_ID = /^key_[0-9a-f]{24}$/;
const MAX_PASSWORD_LENGTH = 256;

export async function PATCH(req: NextRequest, context: { params: Promise<{ id: string }> }) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;
  const { id } = await context.params;
  const body = await readJsonObject(req);
  if (
    !KEY_ID.test(id) ||
    typeof body?.label !== 'string' ||
    !body.label.trim() ||
    body.label.length > 64
  ) {
    return apiError(422, 'invalid');
  }
  const result = await callAsUser(auth.config, auth.user.userId, `/users/me/passkeys/${id}`, {
    method: 'PATCH',
    body: { label: body.label },
  });
  return result.ok ? noStoreJson({ ok: true }) : backendFailure(result.status, result.data);
}

export async function DELETE(req: NextRequest, context: { params: Promise<{ id: string }> }) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;
  const { id } = await context.params;
  const body = await readJsonObject(req);
  if (
    !KEY_ID.test(id) ||
    typeof body?.currentPassword !== 'string' ||
    !body.currentPassword ||
    body.currentPassword.length > MAX_PASSWORD_LENGTH
  )
    return apiError(422, 'invalid');
  const result = await callAsUser(auth.config, auth.user.userId, `/users/me/passkeys/${id}`, {
    method: 'DELETE',
    body: { current_password: body.currentPassword },
    sessionBinding: auth.sessionBinding,
  });
  return result.ok ? noStoreJson({ ok: true }) : backendFailure(result.status, result.data);
}
