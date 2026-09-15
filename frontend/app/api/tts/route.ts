/** TTS-only opt-in selection, using the existing signed user BFF boundary. */
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
import { parseTtsView, providers } from '@/lib/tts/contract';

export const dynamic = 'force-dynamic';

function response(data: unknown) {
  const view = parseTtsView(data);
  return view ? noStoreJson(view) : apiError(502, 'backend_unavailable');
}

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/tts', {
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
    Object.keys(body).some(
      (key) => !['provider', 'profile_id', 'engine', 'model_size'].includes(key)
    ) ||
    typeof body.provider !== 'string' ||
    !providers.some((provider) => provider === body.provider)
  ) {
    return apiError(422, 'invalid_tts_provider');
  }
  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/tts', {
    method: 'PUT',
    body,
    timeoutMs: 10000,
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  return response(result.data);
}
