/** Live text-only message reader. The backend owns connection authorization. */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import { apiError, backendFailure, noStoreJson, requireUser } from '@/lib/auth/guard';
import { browserInboxMessage, messageQuery } from '@/lib/dashboard/provider-data';

export const dynamic = 'force-dynamic';
const MESSAGE_TIMEOUT_MS = 30_000;
const STATES = new Set([
  'reconnect_required',
  'insufficient_scope',
  'not_configured',
  'unsupported',
  'unavailable',
]);

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const query = messageQuery(req.nextUrl.searchParams);
  if (query === null) return apiError(422, 'invalid');
  const result = await callAsUser(
    auth.config,
    auth.user.userId,
    '/users/me/dashboard/inbox/message?' + query,
    { method: 'GET', timeoutMs: MESSAGE_TIMEOUT_MS }
  );
  if (!result.ok) {
    const detail = (result.data as { detail?: unknown } | null)?.detail;
    if (result.status === 502)
      return apiError(
        502,
        typeof detail === 'string' && STATES.has(detail) ? detail : 'unavailable'
      );
    return backendFailure(result.status, result.data);
  }
  const message = browserInboxMessage(result.data);
  if (
    !message ||
    message.connectionId !== req.nextUrl.searchParams.get('connectionId') ||
    message.id !== req.nextUrl.searchParams.get('messageId')
  )
    return apiError(502, 'backend_unavailable');
  return noStoreJson(message);
}
