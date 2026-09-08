/**
 * Administrator: the bounded, redacted audit trail (opaque ids and actions only).
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import { backendFailure, noStoreJson, requireAdmin } from '@/lib/auth/guard';

export const dynamic = 'force-dynamic';

const MAX_LIMIT = 500;

interface AuditEvent {
  eventId: string;
  occurredAt: number;
  actorId: string;
  actorRole: string;
  action: string;
  targetId: string | null;
  outcome: string;
  detail: Record<string, unknown>;
}

function toEvent(row: unknown): AuditEvent | null {
  const event = row as Record<string, unknown> | null;
  if (!event || typeof event.event_id !== 'string' || typeof event.action !== 'string') {
    return null;
  }
  const detail = event.detail;
  return {
    eventId: event.event_id,
    occurredAt: typeof event.occurred_at === 'number' ? event.occurred_at : 0,
    actorId: typeof event.actor_id === 'string' ? event.actor_id : 'unknown',
    actorRole: typeof event.actor_role === 'string' ? event.actor_role : '',
    action: event.action,
    targetId: typeof event.target_id === 'string' ? event.target_id : null,
    outcome: typeof event.outcome === 'string' ? event.outcome : '',
    detail: detail && typeof detail === 'object' && !Array.isArray(detail) ? (detail as Record<string, unknown>) : {},
  };
}

export async function GET(req: NextRequest) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;

  const requested = Number(req.nextUrl.searchParams.get('limit') ?? '100');
  const limit = Number.isFinite(requested) ? Math.min(Math.max(Math.trunc(requested), 1), MAX_LIMIT) : 100;
  const result = await callAsUser(auth.config, auth.user.userId, `/admin/audit?limit=${limit}`, {
    method: 'GET',
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const rows = (result.data as { events?: unknown[] } | null)?.events;
  const events = Array.isArray(rows) ? rows.map(toEvent).filter((event) => event !== null) : [];
  return noStoreJson({ events });
}
