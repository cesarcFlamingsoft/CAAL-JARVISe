/**
 * The same-origin BFF for the Company Library.
 *
 * Authorization is decided twice, as everywhere else in this app: here, from a
 * fresh backend resolution of the verified session (`requireAdmin`), and again
 * by the backend on the single-use principal it receives. Ownership of the
 * library is decided a third time, by the library itself.
 *
 * Mutations go through the existing `guardMutation`: same-origin check, CSRF
 * token, rate limit. Nothing here weakens or reimplements any of that.
 *
 * The upload does not use `callAsUser` (which is JSON): it mints the same kind
 * of principal and forwards the raw bytes. The browser never sees a principal,
 * and never sees the company MCP service -- that runs on loopback with its own
 * bearer, which is not present in this process's responses and never reaches a
 * page.
 *
 * The upload is deliberately neither multipart nor query-parameterised. See
 * `./upload.ts` for why, and `readBoundedBody` below for the bound, which is
 * applied to the stream as it arrives rather than after a parser has already
 * read the whole request.
 */
import type { NextRequest } from 'next/server';
import { backendPrincipalFor, callAsUser } from '@/lib/auth/backend';
import type { IdentityConfig } from '@/lib/auth/config';
import {
  apiError,
  backendFailure,
  guardMutation,
  guardReadPost,
  noStoreJson,
  readJsonObject,
  requireAdmin,
} from '@/lib/auth/guard';
import { isClassification, isDocumentId, isSubjectId, isVersionId, isVersionStatus } from './contract';
import {
  MAX_UPLOAD_BYTES,
  METADATA_HEADER,
  decodeUploadMetadata,
  encodeUploadMetadata,
  parseUploadMetadata,
} from './upload';

const MAX_QUERY = 300;
const UPLOAD_TIMEOUT_MS = 60_000;
const UPLOAD_CONTENT_TYPE = 'application/octet-stream';

const BACKEND = '/admin/company';

/**
 * Read the whole body into memory, bounded, counting as it arrives.
 *
 * `null` means "past the bound, stop": the reader is cancelled rather than
 * allowed to finish accumulating something oversized, so a missing or
 * understated `Content-Length` buys nothing. What this does not claim is that
 * the bytes never reach a disk -- they are in this process's memory, and
 * process memory can be swapped. That is stated, not designed around.
 */
async function readBoundedBody(req: NextRequest): Promise<Uint8Array | null> {
  const declared = req.headers.get('content-length');
  if (declared !== null) {
    if (!/^\d{1,12}$/.test(declared)) return null;
    if (Number(declared) > MAX_UPLOAD_BYTES) return null;
  }
  const body = req.body;
  if (!body) return null;
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      total += value.byteLength;
      if (total > MAX_UPLOAD_BYTES) {
        await reader.cancel().catch(() => undefined);
        return null;
      }
      chunks.push(value);
    }
  } catch {
    return null;
  }
  if (declared !== null && total !== Number(declared)) return null;
  const joined = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return joined;
}

async function uploadToBackend(
  config: IdentityConfig,
  userId: string,
  envelope: string,
  body: Uint8Array
): Promise<{ ok: boolean; status: number | null; data: unknown }> {
  let url: URL;
  try {
    url = new URL('admin/company/documents', `${config.apiBaseUrl}/`);
  } catch {
    console.error('[company] backend URL is not valid');
    return { ok: false, status: null, data: null };
  }
  const principal = await backendPrincipalFor(config, userId);
  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      cache: 'no-store',
      redirect: 'error',
      signal: AbortSignal.timeout(UPLOAD_TIMEOUT_MS),
      headers: {
        Accept: 'application/json',
        Authorization: `Bearer ${principal}`,
        'Content-Type': UPLOAD_CONTENT_TYPE,
        'Content-Length': String(body.byteLength),
        [METADATA_HEADER]: envelope,
      },
      // A `BufferSource` body, not a stream: the bytes are already counted and
      // bounded, and a fixed-length body is what the backend's own
      // `Content-Length` check expects.
      body: body.buffer.slice(
        body.byteOffset,
        body.byteOffset + body.byteLength
      ) as ArrayBuffer,
    });
  } catch {
    // Never logged with the filename or the title: both say what a document is.
    console.error('[company] upload to the backend failed');
    return { ok: false, status: null, data: null };
  }
  let data: unknown = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  return { ok: response.ok, status: response.status, data };
}

/** GET /api/admin/company/status */
export async function statusRoute(req: NextRequest) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  const result = await callAsUser(auth.config, auth.user.userId, `${BACKEND}/status`, {
    method: 'GET',
  });
  return result.ok ? noStoreJson(result.data) : backendFailure(result.status, result.data);
}

/** GET /api/admin/company/documents, POST (upload) */
export async function documentsRoute(req: NextRequest, method: 'GET' | 'POST') {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;

  if (method === 'GET') {
    const url = new URL(req.url);
    const classification = url.searchParams.get('classification');
    const status = url.searchParams.get('status');
    const cursor = url.searchParams.get('cursor');
    if (classification && !isClassification(classification)) return apiError(422, 'invalid');
    if (status && !isVersionStatus(status)) return apiError(422, 'invalid');
    if (cursor && !/^\d{1,6}$/.test(cursor)) return apiError(422, 'invalid');
    const query = new URLSearchParams();
    if (classification) query.set('classification', classification);
    if (status) query.set('status', status);
    if (cursor) query.set('cursor', cursor);
    const suffix = query.size ? `?${query.toString()}` : '';
    const result = await callAsUser(
      auth.config,
      auth.user.userId,
      `${BACKEND}/documents${suffix}`,
      { method: 'GET' }
    );
    return result.ok ? noStoreJson(result.data) : backendFailure(result.status, result.data);
  }

  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  // The metadata is read and validated *before* the body is touched, so an
  // eight-mebibyte upload with an unreadable envelope costs nothing.
  const contentType = (req.headers.get('content-type') ?? '').split(';')[0].trim().toLowerCase();
  if (contentType !== UPLOAD_CONTENT_TYPE) return apiError(415, 'invalid_content_type');
  const declared = decodeUploadMetadata(req.headers.get(METADATA_HEADER));
  if (declared === null) return apiError(422, 'invalid_metadata');
  const metadata = parseUploadMetadata(declared);
  if (!metadata.ok) return apiError(422, 'invalid_metadata');

  const body = await readBoundedBody(req);
  if (body === null) return apiError(413, 'too_large');
  if (body.byteLength === 0) return apiError(422, 'empty');

  const result = await uploadToBackend(
    auth.config,
    auth.user.userId,
    encodeUploadMetadata(metadata.value),
    body
  );
  if (!result.ok) return backendFailure(result.status, result.data);
  return noStoreJson(result.data, 201);
}

/** DELETE /api/admin/company/documents/[id] */
export async function documentRoute(req: NextRequest, documentId: string) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  if (!isDocumentId(documentId)) return apiError(404, 'not_found');
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;
  const result = await callAsUser(
    auth.config,
    auth.user.userId,
    `${BACKEND}/documents/${documentId}`,
    { method: 'DELETE' }
  );
  return result.ok ? noStoreJson(result.data) : backendFailure(result.status, result.data);
}

/** PATCH /api/admin/company/versions/[id] */
export async function versionRoute(req: NextRequest, versionId: string) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  if (!isVersionId(versionId)) return apiError(404, 'not_found');
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;
  const body = await readJsonObject(req);
  if (!body || Object.keys(body).length !== 1 || !isVersionStatus(body.status)) {
    return apiError(422, 'invalid');
  }
  const result = await callAsUser(
    auth.config,
    auth.user.userId,
    `${BACKEND}/versions/${versionId}`,
    { method: 'PATCH', body: { status: body.status } }
  );
  return result.ok ? noStoreJson(result.data) : backendFailure(result.status, result.data);
}

/**
 * POST /api/admin/company/search
 *
 * A read, carried as a POST. `?q=what+is+FIXTURE+Person's+severance` is the
 * confidential half of the question, and a query string is recorded by every
 * access log in the path: the Next.js request log, a reverse proxy in combined
 * format, and the browser's own history. A request body is recorded by none of
 * them by default, so the query, the subject id and the filter all travel in a
 * bounded JSON body, here and on the call to the backend.
 *
 * It keeps read semantics deliberately. `requireAdmin` applies the read rate
 * limit; `guardReadPost` applies the same-origin and CSRF checks a POST needs;
 * the mutation budget is untouched, because searching changes nothing and an
 * owner may reasonably search ten times in a row. Nothing about the query is
 * logged or echoed.
 */
export async function searchRoute(req: NextRequest) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  const blocked = guardReadPost(req, auth.config);
  if (blocked) return blocked;
  const body = await readJsonObject(req);
  if (!body) return apiError(422, 'invalid');
  const { query: q, classification, subject } = body;
  if (Object.keys(body).some((key) => !['query', 'classification', 'subject'].includes(key))) {
    return apiError(422, 'invalid');
  }
  if (typeof q !== 'string' || !q.trim() || q.length > MAX_QUERY) return apiError(422, 'invalid');
  if (classification !== undefined && !isClassification(classification)) {
    return apiError(422, 'invalid');
  }
  if (subject !== undefined && !isSubjectId(subject)) return apiError(422, 'invalid');
  const payload: Record<string, unknown> = { query: q };
  if (classification !== undefined) payload.classification = classification;
  if (subject !== undefined) payload.subject = subject;
  const result = await callAsUser(auth.config, auth.user.userId, `${BACKEND}/search`, {
    method: 'POST',
    body: payload,
  });
  return result.ok ? noStoreJson(result.data) : backendFailure(result.status, result.data);
}

/** GET|POST /api/admin/company/people */
export async function peopleRoute(req: NextRequest, method: 'GET' | 'POST') {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  if (method === 'GET') {
    const result = await callAsUser(auth.config, auth.user.userId, `${BACKEND}/people`, {
      method: 'GET',
    });
    return result.ok ? noStoreJson(result.data) : backendFailure(result.status, result.data);
  }
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;
  const body = await readJsonObject(req);
  const displayName = body?.display_name;
  const aliases = body?.aliases;
  if (
    !body ||
    typeof displayName !== 'string' ||
    displayName.trim().length === 0 ||
    displayName.length > 120 ||
    (aliases !== undefined &&
      (!Array.isArray(aliases) ||
        aliases.length > 8 ||
        aliases.some((item) => typeof item !== 'string' || item.length > 120))) ||
    Object.keys(body).some((key) => !['display_name', 'aliases'].includes(key))
  ) {
    return apiError(422, 'invalid');
  }
  const result = await callAsUser(auth.config, auth.user.userId, `${BACKEND}/people`, {
    method: 'POST',
    body: { display_name: displayName.trim(), aliases: Array.isArray(aliases) ? aliases : [] },
  });
  return result.ok ? noStoreJson(result.data, 201) : backendFailure(result.status, result.data);
}
