/** Server-owned bridge for a bounded reanalysis of the user's retained visual frame. */
import type { NextRequest } from 'next/server';
import { backendDetail, callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardReadPost,
  noStoreJson,
  requireUser,
} from '@/lib/auth/guard';
import { companySessionRequested } from '@/lib/company/session-mode';

export const dynamic = 'force-dynamic';

const MAX_QUESTION = 240;
const MAX_BODY_BYTES = MAX_QUESTION + 128;
const ANALYSIS_TIMEOUT_MS = 30_000;

function companyReferer(req: Request): boolean {
  const raw = req.headers.get('referer');
  if (!raw) return false;
  try {
    const referer = new URL(raw);
    return referer.origin === new URL(req.url).origin && companySessionRequested(referer.search);
  } catch {
    return false;
  }
}

async function readRequest(req: Request): Promise<{ question: string } | null> {
  const stated = Number(req.headers.get('content-length') ?? '0');
  if (Number.isFinite(stated) && stated > MAX_BODY_BYTES) return null;
  let value: unknown;
  try {
    const text = await req.text();
    if (text.length > MAX_BODY_BYTES) return null;
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const body = value as Record<string, unknown>;
  if (
    Object.keys(body).length !== 1 ||
    typeof body.question !== 'string' ||
    body.question.length < 1 ||
    body.question.length > MAX_QUESTION ||
    /[\x00-\x1f\x7f]/.test(body.question)
  )
    return null;
  return { question: body.question };
}

export async function POST(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const expectedUser = req.headers.get('x-caal-visual-user');
  if (expectedUser !== null && expectedUser !== auth.user.userId)
    return apiError(403, 'not_signed_in');
  const blocked = guardReadPost(req, auth.config);
  if (blocked) return blocked;
  if (companyReferer(req)) return apiError(403, 'company_mode_blocked');
  const body = await readRequest(req);
  if (!body) return apiError(422, 'invalid');
  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/visual/reanalyze', {
    method: 'POST',
    body,
    timeoutMs: ANALYSIS_TIMEOUT_MS,
    sessionBinding: auth.sessionBinding,
  });
  if (!result.ok) {
    const detail = backendDetail(result.data);
    if (result.status === 404 && detail === 'visual_frame_unavailable')
      return apiError(404, 'visual_frame_unavailable');
    if (result.status === 503 && detail === 'vision_unavailable')
      return apiError(503, 'vision_unavailable');
    if (result.status === 502 && detail === 'vision_no_description')
      return apiError(502, 'vision_no_description');
    return backendFailure(result.status, result.data);
  }
  const description = (result.data as { description?: unknown } | null)?.description;
  if (typeof description !== 'string' || description.length > 1200)
    return apiError(502, 'backend_unavailable');
  return noStoreJson({ description });
}
