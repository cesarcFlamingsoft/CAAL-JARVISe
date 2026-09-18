/** Server-owned bridge for one bounded, authenticated local visual analysis. */
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
import { isVisualPrompt } from '@/lib/visual/contract';

export const dynamic = 'force-dynamic';

const MAX_IMAGE_BASE64 = 550_000;
const MAX_BODY_BYTES = 560_000;
const MAX_PROMPT = 240;
const ANALYSIS_TIMEOUT_MS = 30_000;
const BASE64 = /^[A-Za-z0-9+/]+={0,2}$/;

interface VisualRequest {
  image: string;
  prompt: string;
  company_private: boolean;
}

async function readRequest(req: Request): Promise<VisualRequest | null> {
  const stated = Number(req.headers.get('content-length') ?? '0');
  if (Number.isFinite(stated) && stated > MAX_BODY_BYTES) return null;
  let text: string;
  try {
    text = await req.text();
  } catch {
    return null;
  }
  if (text.length > MAX_BODY_BYTES) return null;
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const body = value as Record<string, unknown>;
  if (
    Object.keys(body).some(
      (key) => key !== 'image' && key !== 'prompt' && key !== 'company_private'
    ) ||
    typeof body.image !== 'string' ||
    body.image.length < 4 ||
    body.image.length > MAX_IMAGE_BASE64 ||
    body.image.length % 4 !== 0 ||
    !BASE64.test(body.image) ||
    typeof body.prompt !== 'string' ||
    body.prompt.length > MAX_PROMPT ||
    !isVisualPrompt(body.prompt) ||
    typeof body.company_private !== 'boolean'
  ) {
    return null;
  }
  return body as unknown as VisualRequest;
}

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

export async function POST(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const expectedUser = req.headers.get('x-caal-visual-user');
  if (expectedUser !== null && expectedUser !== auth.user.userId)
    return apiError(403, 'not_signed_in');
  const blocked = guardReadPost(req, auth.config);
  if (blocked) return blocked;

  // The URL check happens before the body is read, so Company Mode does not
  // even materialize a submitted frame in this route.
  if (companyReferer(req)) return apiError(403, 'company_mode_blocked');
  const body = await readRequest(req);
  if (!body) return apiError(422, 'invalid');
  if (body.company_private) return apiError(403, 'company_mode_blocked');

  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/visual/analyze', {
    method: 'POST',
    body,
    timeoutMs: ANALYSIS_TIMEOUT_MS,
    sessionBinding: auth.sessionBinding,
  });
  if (!result.ok) {
    const detail = backendDetail(result.data);
    console.warn(
      `[visual] backend analysis failed status=${result.status ?? 'network'} detail=${detail ?? 'none'}`
    );
    if (result.status === 503 && detail === 'vision_unavailable') {
      return apiError(503, 'vision_unavailable');
    }
    if (result.status === 502 && detail === 'vision_no_description') {
      return apiError(502, 'vision_no_description');
    }
    return backendFailure(result.status, result.data);
  }
  const description = (result.data as { description?: unknown } | null)?.description;
  if (typeof description !== 'string' || description.length > 1200) {
    return apiError(502, 'backend_unavailable');
  }
  return noStoreJson({ description });
}
