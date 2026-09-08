/**
 * Server-only helpers for the dashboard routes.
 *
 * Access follows the same posture as the identity routes: every response is
 * uncacheable and every decision comes from a fresh backend resolution. The
 * operator settings are fetched here and reduced *before* they are returned,
 * so a route can never accidentally forward a credential.
 */
import type { NextResponse } from 'next/server';
import { apiError, clearSession, readLimited } from '@/lib/auth/guard';
import { authenticate } from '@/lib/auth/session';
import { type DashboardScope, dashboardScope } from './access';

const WEBHOOK_URL = process.env.WEBHOOK_URL || 'http://agent:8889';
const BACKEND_TIMEOUT_MS = 5000;

export type DashboardAccess =
  | Extract<DashboardScope, { ok: true }>
  | { ok: false; response: NextResponse };

export async function requireDashboardAccess(req: Request): Promise<DashboardAccess> {
  const limited = readLimited(req);
  if (limited) return { ok: false, response: limited };

  const auth = await authenticate(req.headers);
  const scope = dashboardScope(auth);
  if (scope.ok) return scope;

  const response = apiError(scope.status, scope.code);
  return { ok: false, response: auth.kind === 'expired' ? clearSession(response, req) : response };
}

/** The raw operator settings, or null when the backend cannot answer. Never logged. */
export async function fetchBackendSettings(): Promise<unknown | null> {
  try {
    const res = await fetch(`${WEBHOOK_URL}/settings`, {
      cache: 'no-store',
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(BACKEND_TIMEOUT_MS),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}
