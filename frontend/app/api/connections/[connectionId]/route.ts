/**
 * Rename, or disconnect, one of the signed-in user's own provider connections.
 *
 * The backend wipes the stored tokens and retires the row, and answers
 * `not_found` for an id that belongs to anyone else. The page confirms with
 * the user before calling this; the route itself only insists on a valid id,
 * our own origin, the CSRF token and the per-user budget.
 *
 * PATCH saves the names the user gives that account -- a short label and a
 * few aliases, which is how they will ask JARVIS for it. The provider's own
 * account label is not touched, the names are normalized here exactly as the
 * backend normalizes them, and neither the submitted text nor the backend's
 * wording is echoed back: a refusal is a bounded code. An id belonging to
 * anyone else is `not_found`, exactly as it is for DELETE.
 */
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardMutation,
  noStoreJson,
  readJsonObject,
  requireUser,
  securityHeaders,
} from '@/lib/auth/guard';
import {
  accountNamesFrom,
  accountNamesRequest,
  browserConnection,
  isConnectionId,
} from '@/lib/connections/protocol';

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

export async function PATCH(req: NextRequest, { params }: Params) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const { connectionId } = await params;
  if (!isConnectionId(connectionId)) return apiError(404, 'not_found');
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const body = await readJsonObject(req);
  if (!body) return apiError(422, 'invalid_account_name');
  const names = accountNamesFrom(body);
  if (!names.ok) return apiError(422, names.error);

  const result = await callAsUser(
    auth.config,
    auth.user.userId,
    `/users/me/connections/${connectionId}`,
    { method: 'PATCH', body: accountNamesRequest(names.names) }
  );
  if (!result.ok) return backendFailure(result.status, result.data);
  const connection = browserConnection(result.data);
  return connection ? noStoreJson(connection) : apiError(502, 'backend_unavailable');
}
