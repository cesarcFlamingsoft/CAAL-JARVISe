import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardMutation,
  noStoreJson,
  requireUser,
} from '@/lib/auth/guard';
import { connectionId } from '@/lib/home-assistant/contract';

export const dynamic = 'force-dynamic';
export async function DELETE(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;
  const { id } = await params;
  if (!connectionId(id)) return apiError(404, 'not_found');
  const r = await callAsUser(
    auth.config,
    auth.user.userId,
    `/users/me/home-assistant/connections/${id}`,
    { method: 'DELETE' }
  );
  return r.ok ? noStoreJson({ status: 'disconnected' }) : backendFailure(r.status, r.data);
}
