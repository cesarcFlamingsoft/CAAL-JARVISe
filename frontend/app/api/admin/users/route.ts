/**
 * Administrator: list users and create a user.
 *
 * Authorization is decided twice: here from a fresh backend resolution of the
 * verified identity, and again by the backend on the principal it receives.
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  browserUser,
  guardMutation,
  isDisplayName,
  noStoreJson,
  readJsonObject,
  requireAdmin,
} from '@/lib/auth/guard';

export const dynamic = 'force-dynamic';

const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export async function GET(req: NextRequest) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;

  const result = await callAsUser(auth.config, auth.user.userId, '/admin/users', { method: 'GET' });
  if (!result.ok) return backendFailure(result.status, result.data);
  const rows = (result.data as { users?: unknown[] } | null)?.users;
  const users = Array.isArray(rows)
    ? rows.map((row) => browserUser(row, { admin: true })).filter((user) => user !== null)
    : [];
  return noStoreJson({ users });
}

export async function POST(req: NextRequest) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const body = await readJsonObject(req);
  const email = typeof body?.email === 'string' ? body.email.trim().toLowerCase() : '';
  const role = body?.role;
  if (
    !body ||
    !email ||
    email.length > 254 ||
    !EMAIL.test(email) ||
    !isDisplayName(body.displayName) ||
    (role !== 'admin' && role !== 'member') ||
    (body.withPassword !== undefined && typeof body.withPassword !== 'boolean') ||
    Object.keys(body).some(
      (key) => !['email', 'displayName', 'role', 'withPassword'].includes(key)
    )
  ) {
    return apiError(422, 'invalid');
  }

  const result = await callAsUser(auth.config, auth.user.userId, '/admin/users', {
    method: 'POST',
    body: {
      email,
      display_name: body.displayName.trim(),
      role,
      with_password: body.withPassword === true,
    },
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const user = browserUser(result.data, { admin: true });
  if (!user) return apiError(502, 'backend_unavailable');
  // Present exactly once, when the administrator asked for one; it is never
  // stored in the clear and can never be fetched again.
  const issued = (result.data as { one_time_password?: unknown } | null)?.one_time_password;
  return noStoreJson(
    { user, ...(typeof issued === 'string' ? { oneTimePassword: issued } : {}) },
    201
  );
}
