import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import { backendFailure, noStoreJson, requireUser } from '@/lib/auth/guard';

export const dynamic = 'force-dynamic';

function safePasskeys(data: unknown) {
  const raw = (data as { passkeys?: unknown } | null)?.passkeys;
  if (!Array.isArray(raw)) return null;
  const passkeys = raw.slice(0, 50).flatMap((item) => {
    const value = item as Record<string, unknown>;
    if (
      typeof value.id !== 'string' ||
      !/^key_[0-9a-f]{24}$/.test(value.id) ||
      typeof value.label !== 'string' ||
      typeof value.createdAt !== 'number' ||
      (value.lastUsedAt !== null && typeof value.lastUsedAt !== 'number')
    )
      return [];
    return [
      {
        id: value.id,
        label: value.label.slice(0, 64),
        createdAt: value.createdAt,
        lastUsedAt: value.lastUsedAt,
      },
    ];
  });
  return passkeys;
}

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/passkeys', {
    method: 'GET',
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const passkeys = safePasskeys(result.data);
  return passkeys ? noStoreJson({ passkeys }) : backendFailure(null, null);
}
