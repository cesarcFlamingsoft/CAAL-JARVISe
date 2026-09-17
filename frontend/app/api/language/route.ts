/** Reply-language selection, on the existing signed per-user BFF boundary.
 *
 * Owner-scoped by construction: every call goes out as the verified principal to
 * `/users/me/language`, and there is no deployment-wide language surface to
 * reach, so one account's choice can never move another session.
 */
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
import { type Language, languages, parseLanguageView } from '@/lib/language/contract';

export const dynamic = 'force-dynamic';

function response(data: unknown) {
  const view = parseLanguageView(data);
  return view ? noStoreJson(view) : apiError(502, 'backend_unavailable');
}

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/language', {
    method: 'GET',
    timeoutMs: 10000,
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  return response(result.data);
}

export async function PUT(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;
  const body = await readJsonObject(req);
  if (
    !body ||
    Object.keys(body).some((key) => key !== 'language') ||
    typeof body.language !== 'string' ||
    !languages.some((language) => language === (body.language as Language))
  ) {
    return apiError(422, 'invalid_reply_language');
  }
  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/language', {
    method: 'PUT',
    body: { language: body.language },
    timeoutMs: 10000,
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  return response(result.data);
}
