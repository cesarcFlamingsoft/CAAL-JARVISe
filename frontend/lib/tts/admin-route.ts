import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardMutation,
  noStoreJson,
  readJsonObject,
  requireAdmin,
} from '@/lib/auth/guard';
import { parseVoiceboxConfig } from './contract';

export async function voiceboxRoute(req: NextRequest, method: 'GET' | 'PUT' | 'POST') {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  let body: Record<string, unknown> | undefined;
  if (method !== 'GET') {
    const blocked = guardMutation(req, auth.config, auth.user.userId);
    if (blocked) return blocked;
    const value = await readJsonObject(req);
    if (
      !value ||
      Object.keys(value).some((key) => !['endpoint', 'credential'].includes(key)) ||
      typeof value.endpoint !== 'string' ||
      value.endpoint.length > 200 ||
      (value.credential !== undefined &&
        (typeof value.credential !== 'string' || value.credential.length > 4096))
    )
      return apiError(422, 'invalid_voicebox_config');
    body = value;
  }
  const path = method === 'POST' ? '/users/me/tts/voicebox/test' : '/users/me/tts/voicebox';
  const result = await callAsUser(auth.config, auth.user.userId, path, {
    method,
    ...(body ? { body } : {}),
    timeoutMs: 10000,
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const parsed = parseVoiceboxConfig(result.data);
  return parsed ? noStoreJson(parsed) : apiError(502, 'backend_unavailable');
}
