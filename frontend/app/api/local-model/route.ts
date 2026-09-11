/**
 * The local Ollama JARVIS runs on: read it, or (as an administrator) change it.
 *
 * The browser never reaches an Ollama itself. It asks this route, which asks
 * the agent as the signed-in user, and the agent is the only thing that opens
 * a socket -- and only to an address it has already accepted as local. The
 * answer is reduced through the local-model parser before it reaches the
 * browser, so an unexpected backend payload cannot be forwarded by accident.
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardMutation,
  noStoreJson,
  readJsonObject,
  requireAdmin,
  requireUser,
} from '@/lib/auth/guard';
import { browserLocalModel, isModelName, normalizeEndpoint } from '@/lib/local-model/endpoint';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;

  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/local-model', {
    method: 'GET',
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const view = browserLocalModel(result.data);
  return view ? noStoreJson(view) : apiError(502, 'backend_unavailable');
}

export async function PUT(req: NextRequest) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const body = await readJsonObject(req);
  if (!body || Object.keys(body).some((key) => !['endpoint', 'model'].includes(key))) {
    return apiError(422, 'invalid_endpoint');
  }
  const endpoint = normalizeEndpoint(typeof body.endpoint === 'string' ? body.endpoint : '');
  if (!endpoint.ok) return apiError(422, endpoint.code);
  if (!isModelName(body.model)) return apiError(422, 'invalid_model');

  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/local-model', {
    method: 'PUT',
    body: { endpoint: endpoint.endpoint, model: body.model },
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const view = browserLocalModel(result.data);
  return view ? noStoreJson(view) : apiError(502, 'backend_unavailable');
}
