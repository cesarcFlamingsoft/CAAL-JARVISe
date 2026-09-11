/**
 * The models installed at a local Ollama endpoint.
 *
 * A discovery is a request the agent makes on an administrator say-so, so it
 * is guarded like a mutation -- our own origin, the CSRF token, a rate limit --
 * and the address is checked here as well as by the agent. The browser is
 * given a model list and the endpoint it came from, and nothing else: this is
 * not a proxy for arbitrary URLs.
 */
import type { NextRequest } from 'next/server';
import { backendDetail, callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardMutation,
  noStoreJson,
  readJsonObject,
  requireAdmin,
} from '@/lib/auth/guard';
import { browserModels, normalizeEndpoint } from '@/lib/local-model/endpoint';

export const dynamic = 'force-dynamic';

/** Short by design: an Ollama on the same network answers at once or is not there. */
const DISCOVERY_TIMEOUT_MS = 10_000;

export async function POST(req: NextRequest) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const body = await readJsonObject(req);
  if (!body || Object.keys(body).some((key) => key !== 'endpoint')) {
    return apiError(422, 'invalid_endpoint');
  }
  // An empty body means the saved endpoint, which only the agent knows.
  let endpoint: string | null = null;
  if (body.endpoint !== undefined) {
    const checked = normalizeEndpoint(typeof body.endpoint === 'string' ? body.endpoint : '');
    if (!checked.ok) return apiError(422, checked.code);
    endpoint = checked.endpoint;
  }

  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/local-model/models', {
    method: 'POST',
    body: endpoint === null ? {} : { endpoint },
    timeoutMs: DISCOVERY_TIMEOUT_MS,
  });
  if (!result.ok) {
    // Why the agent could not read the endpoint is worth telling the operator:
    // it is a short code about their own network, never an upstream string.
    const detail = result.status === 502 ? backendDetail(result.data) : null;
    return detail ? apiError(502, detail) : backendFailure(result.status, result.data);
  }
  const view = browserModels(result.data);
  return view ? noStoreJson(view) : apiError(502, 'backend_unavailable');
}
