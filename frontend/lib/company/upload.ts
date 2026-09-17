/**
 * The upload contract, shared by the Company Library page and its BFF route.
 *
 * The bytes of a document travel as the raw request body. Everything else --
 * the title, the filename, the classification, the status, the effective date,
 * the version label, the employee subject ids -- travels as one bounded
 * base64url JSON envelope in the `X-CAAL-Company-Metadata` header, and is
 * validated here before it is sent and again, strictly, by the backend.
 *
 * Two deliberate choices, both of which the backend mirrors:
 *
 * **Not multipart.** `req.formData()` parses the entire request before any
 * bound has been applied, and a large part is buffered to do it. The bound has
 * to come first, so the body is streamed with a running count instead.
 *
 * **Not a query string.** A title like "FIXTURE Person severance agreement" is
 * the confidential part of an upload, and a query string is the part of a
 * request that every access log, proxy log and browser history records. A
 * header is recorded only where something chose to record it.
 *
 * Base64 is not encryption and nothing here pretends it is. The envelope is
 * encoded so that a document title is not accidentally greppable and so that
 * arbitrary Unicode survives a header, not to hide it from anyone holding the
 * request.
 */

export const METADATA_HEADER = 'X-CAAL-Company-Metadata';

/** Hard ceiling on an upload body. The backend enforces its own, identically. */
export const MAX_UPLOAD_BYTES = 8 * 1024 * 1024;

/** Bound on the encoded envelope, matching the backend's header bound. */
export const MAX_METADATA_HEADER_BYTES = 2048;

/** What the library reads. A file outside this list is refused before it is sent. */
export const ACCEPTED_EXTENSIONS = ['.txt', '.md', '.pdf', '.docx'] as const;

const MAX_TITLE = 200;
const MAX_FILENAME = 200;
const MAX_VERSION_LABEL = 60;
const MAX_SUBJECTS = 32;

const CLASSIFICATIONS = ['policy', 'hr', 'contract', 'general'] as const;
const STATUSES = ['current', 'superseded', 'draft'] as const;

export type UploadMetadata = {
  filename: string;
  title: string;
  classification: (typeof CLASSIFICATIONS)[number];
  status: (typeof STATUSES)[number];
  effective_date?: string;
  version_label?: string;
  document_id?: string;
  subjects: string[];
};

export type ParseResult =
  | { ok: true; value: UploadMetadata }
  | { ok: false; field: string };

const isText = (value: unknown, max: number): value is string =>
  typeof value === 'string' && value.trim().length > 0 && value.length <= max;

const KNOWN_FIELDS = new Set([
  'filename',
  'title',
  'classification',
  'status',
  'effective_date',
  'version_label',
  'document_id',
  'subjects',
]);

/** The extension the library will read this file as, or null. */
export function acceptedExtension(filename: unknown): string | null {
  if (typeof filename !== 'string') return null;
  // A name with a path separator in it is not a name. The backend never uses
  // it as a path, but there is no reason to send one either.
  if (filename.includes('/') || filename.includes('\\') || filename.includes('\0')) return null;
  const dot = filename.lastIndexOf('.');
  if (dot <= 0) return null;
  const extension = filename.slice(dot).toLowerCase();
  return (ACCEPTED_EXTENSIONS as readonly string[]).includes(extension) ? extension : null;
}

/**
 * Validate one metadata object, strictly.
 *
 * An unknown field is a refusal, not something to drop: a caller that sends
 * one believes it is asking for something, and storing the document without it
 * would mean storing it under metadata nobody chose.
 */
export function parseUploadMetadata(input: unknown): ParseResult {
  if (!input || typeof input !== 'object' || Array.isArray(input)) return { ok: false, field: '' };
  const row = input as Record<string, unknown>;

  for (const key of Object.keys(row)) {
    if (!KNOWN_FIELDS.has(key)) return { ok: false, field: key };
  }
  if (!isText(row.filename, MAX_FILENAME)) return { ok: false, field: 'filename' };
  if (acceptedExtension(row.filename) === null) return { ok: false, field: 'filename' };
  if (!isText(row.title, MAX_TITLE)) return { ok: false, field: 'title' };
  if (!(CLASSIFICATIONS as readonly unknown[]).includes(row.classification)) {
    return { ok: false, field: 'classification' };
  }
  if (!(STATUSES as readonly unknown[]).includes(row.status)) return { ok: false, field: 'status' };

  const value: UploadMetadata = {
    filename: (row.filename as string).trim(),
    title: (row.title as string).trim(),
    classification: row.classification as UploadMetadata['classification'],
    status: row.status as UploadMetadata['status'],
    subjects: [],
  };

  if (row.effective_date !== undefined && row.effective_date !== null && row.effective_date !== '') {
    if (typeof row.effective_date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(row.effective_date)) {
      return { ok: false, field: 'effective_date' };
    }
    value.effective_date = row.effective_date;
  }
  if (row.version_label !== undefined && row.version_label !== null && row.version_label !== '') {
    if (!isText(row.version_label, MAX_VERSION_LABEL)) return { ok: false, field: 'version_label' };
    value.version_label = (row.version_label as string).trim();
  }
  if (row.document_id !== undefined && row.document_id !== null && row.document_id !== '') {
    if (typeof row.document_id !== 'string' || !/^cdoc_[a-f0-9]{24}$/.test(row.document_id)) {
      return { ok: false, field: 'document_id' };
    }
    value.document_id = row.document_id;
  }
  if (row.subjects !== undefined && row.subjects !== null) {
    if (!Array.isArray(row.subjects) || row.subjects.length > MAX_SUBJECTS) {
      return { ok: false, field: 'subjects' };
    }
    if (row.subjects.some((item) => typeof item !== 'string' || !/^emp_[a-f0-9]{16}$/.test(item))) {
      return { ok: false, field: 'subjects' };
    }
    value.subjects = [...(row.subjects as string[])];
  }
  return { ok: true, value };
}

const toBase64Url = (bytes: Uint8Array): string => {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  const base64 = typeof btoa === 'function' ? btoa(binary) : Buffer.from(bytes).toString('base64');
  return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};

const fromBase64Url = (value: string): Uint8Array | null => {
  if (!/^[A-Za-z0-9_-]+$/.test(value)) return null;
  const base64 = value.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (value.length % 4)) % 4);
  try {
    if (typeof atob === 'function') {
      const binary = atob(base64);
      return Uint8Array.from(binary, (character) => character.charCodeAt(0));
    }
    return new Uint8Array(Buffer.from(base64, 'base64'));
  } catch {
    return null;
  }
};

/** Build the header value. Optional fields the owner left blank are simply absent. */
export function encodeUploadMetadata(metadata: Record<string, unknown>): string {
  const payload: Record<string, unknown> = {};
  for (const key of ['filename', 'title', 'classification', 'status'] as const) {
    payload[key] = metadata[key];
  }
  for (const key of ['effective_date', 'version_label', 'document_id'] as const) {
    const value = metadata[key];
    if (typeof value === 'string' && value.length > 0) payload[key] = value;
  }
  if (Array.isArray(metadata.subjects) && metadata.subjects.length > 0) {
    payload.subjects = metadata.subjects;
  }
  return toBase64Url(new TextEncoder().encode(JSON.stringify(payload)));
}

/** Read a header value back. `null` for anything that is not one. */
export function decodeUploadMetadata(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'string' || value.length === 0) return null;
  if (value.length > MAX_METADATA_HEADER_BYTES) return null;
  const bytes = fromBase64Url(value);
  if (bytes === null) return null;
  try {
    const parsed = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}
