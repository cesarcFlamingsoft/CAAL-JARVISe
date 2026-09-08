/**
 * Disconnect one of the signed-in user's own provider connections.
 *
 * The backend wipes the stored tokens and retires the row, and answers
 * `not_found` for an id that belongs to anyone else. The page confirms with
 * the user before calling this; the route itself only insists on a valid id,
 * our own origin, the CSRF token and the per-user budget.
 */
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardMutation,
  requireUser,
  securityHeaders,
} from '@/lib/auth/guard';
import { isConnectionId } from '@/lib/connections/protocol';

export const dynamic = 'force-dynamic';

type Params = { params: Promise<{ connectionId: string }> };

export async function DELETE(req: NextRequest, { params }: Params) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const { connectionId } = await params;
  if (!isConnectionId(connectionId)) return apiError(404, 'not_found');
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const result = await callAsUser(
    auth.config,
    auth.user.userId,
    `/users/me/connections/${connectionId}`,
    { method: 'DELETE' }
  );
  if (!result.ok) return backendFailure(result.status, result.data);
  return new NextResponse(null, { status: 204, headers: securityHeaders() });
}
