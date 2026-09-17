/** Owner-only, read-only durable work; no caller-controlled backend scope. */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import { apiError, backendFailure, noStoreJson, requireUser } from '@/lib/auth/guard';
import { browserWork } from '@/lib/dashboard/work';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  if (req.nextUrl.searchParams.size) return apiError(422, 'invalid');
  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/dashboard/work', {
    method: 'GET',
    timeoutMs: 10_000,
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const feed = browserWork(result.data);
  if (!feed) {
    console.error('[dashboard] Work feed unreadable');
    return apiError(502, 'backend_unavailable');
  }
  return noStoreJson(feed);
}
