/**
 * The shapes the Company Library page and its BFF routes agree on.
 *
 * Everything the backend sends is parsed here before it reaches a component:
 * an unexpected field is dropped rather than rendered, and every string that
 * is displayed is plain text. Nothing in this module produces HTML, and no
 * value from a document is ever treated as markup.
 */

export const CLASSIFICATIONS = ['policy', 'hr', 'contract', 'general'] as const;
export const VERSION_STATUSES = ['current', 'superseded', 'draft'] as const;

export type Classification = (typeof CLASSIFICATIONS)[number];
export type VersionStatus = (typeof VERSION_STATUSES)[number];

export type Version = {
  version_id: string;
  classification: Classification;
  status: VersionStatus;
  effective_date: string | null;
  version_label: string | null;
  filename: string;
  ingest_status: string;
  reason: string | null;
  page_count: number | null;
  chunk_count: number;
  byte_count: number;
  subjects: string[];
};

export type Document = {
  document_id: string;
  title: string;
  versions: Version[];
};

export type LibraryStatus = {
  company_name: string | null;
  document_count: number;
  indexed_versions: number;
  failed_versions: number;
  passage_count: number;
  subject_count: number;
  conflicting_documents: string[];
  retrieval: string;
  retrieval_note: string;
  accepted_extensions: string[];
  max_upload_bytes: number;
  classification_options: string[];
  status_options: string[];
  unsupported_note: string;
  capacity: { documents: number; passages: number };
};

export type Citation = {
  document_id: string;
  version_id: string;
  chunk_id: string;
  title: string;
  location: string;
  classification: Classification;
  version_status: VersionStatus;
  version_label: string | null;
  effective_date: string | null;
  snippet: string;
};

export type SearchOutcome = {
  status: string;
  results: Citation[];
  conflicts: { document_id: string }[];
  retrieval: string;
  message: string | null;
};

export type Subject = { subject_id: string; display_name: string; aliases: string[] };

const MAX_TEXT = 4000;

const obj = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;

const text = (value: unknown, fallback = ''): string =>
  typeof value === 'string' ? value.slice(0, MAX_TEXT) : fallback;

const maybeText = (value: unknown): string | null =>
  typeof value === 'string' && value.length > 0 ? value.slice(0, MAX_TEXT) : null;

const count = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value) ? Math.trunc(value) : 0;

export const isClassification = (value: unknown): value is Classification =>
  CLASSIFICATIONS.includes(value as Classification);

export const isVersionStatus = (value: unknown): value is VersionStatus =>
  VERSION_STATUSES.includes(value as VersionStatus);

export const isDocumentId = (value: unknown): value is string =>
  typeof value === 'string' && /^cdoc_[a-f0-9]{24}$/.test(value);

export const isVersionId = (value: unknown): value is string =>
  typeof value === 'string' && /^cver_[a-f0-9]{24}$/.test(value);

export const isSubjectId = (value: unknown): value is string =>
  typeof value === 'string' && /^emp_[a-f0-9]{16}$/.test(value);

/** An ISO calendar date, or nothing. A date is never inferred from a file. */
export const isEffectiveDate = (value: unknown): value is string =>
  typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value);

export function parseVersion(value: unknown): Version | null {
  const row = obj(value);
  if (!row || !isVersionId(row.version_id)) return null;
  if (!isClassification(row.classification) || !isVersionStatus(row.status)) return null;
  return {
    version_id: row.version_id,
    classification: row.classification,
    status: row.status,
    effective_date: isEffectiveDate(row.effective_date) ? row.effective_date : null,
    version_label: maybeText(row.version_label),
    filename: text(row.filename, 'upload'),
    ingest_status: text(row.ingest_status, 'unknown'),
    reason: maybeText(row.reason),
    page_count: typeof row.page_count === 'number' ? Math.trunc(row.page_count) : null,
    chunk_count: count(row.chunk_count),
    byte_count: count(row.byte_count),
    subjects: Array.isArray(row.subjects) ? row.subjects.filter(isSubjectId) : [],
  };
}

export function parseDocuments(value: unknown): Document[] {
  const body = obj(value);
  const rows = Array.isArray(body?.documents) ? body.documents : [];
  const documents: Document[] = [];
  for (const entry of rows) {
    const row = obj(entry);
    if (!row || !isDocumentId(row.document_id)) continue;
    const versions = (Array.isArray(row.versions) ? row.versions : [])
      .map(parseVersion)
      .filter((version): version is Version => version !== null);
    documents.push({ document_id: row.document_id, title: text(row.title), versions });
  }
  return documents;
}

export function parseStatus(value: unknown): LibraryStatus | null {
  const row = obj(value);
  if (!row || typeof row.document_count !== 'number') return null;
  const capacity = obj(row.capacity);
  return {
    company_name: maybeText(row.company_name),
    document_count: count(row.document_count),
    indexed_versions: count(row.indexed_versions),
    failed_versions: count(row.failed_versions),
    passage_count: count(row.passage_count),
    subject_count: count(row.subject_count),
    conflicting_documents: Array.isArray(row.conflicting_documents)
      ? row.conflicting_documents.filter(isDocumentId)
      : [],
    retrieval: text(row.retrieval, 'unknown'),
    retrieval_note: text(row.retrieval_note),
    accepted_extensions: Array.isArray(row.accepted_extensions)
      ? row.accepted_extensions.filter((item): item is string => typeof item === 'string')
      : [],
    max_upload_bytes: count(row.max_upload_bytes),
    classification_options: [...CLASSIFICATIONS],
    status_options: [...VERSION_STATUSES],
    unsupported_note: text(row.unsupported_note),
    capacity: {
      documents: count(capacity?.documents),
      passages: count(capacity?.passages),
    },
  };
}

export function parseSearch(value: unknown): SearchOutcome | null {
  const row = obj(value);
  if (!row || typeof row.status !== 'string') return null;
  const rows = Array.isArray(row.results) ? row.results : [];
  const results: Citation[] = [];
  for (const entry of rows) {
    const hit = obj(entry);
    if (!hit || !isDocumentId(hit.document_id) || !isVersionId(hit.version_id)) continue;
    if (!isClassification(hit.classification) || !isVersionStatus(hit.version_status)) continue;
    results.push({
      document_id: hit.document_id,
      version_id: hit.version_id,
      chunk_id: text(hit.chunk_id),
      title: text(hit.title),
      location: text(hit.location),
      classification: hit.classification,
      version_status: hit.version_status,
      version_label: maybeText(hit.version_label),
      effective_date: isEffectiveDate(hit.effective_date) ? hit.effective_date : null,
      snippet: text(hit.snippet),
    });
  }
  const conflicts = (Array.isArray(row.conflicts) ? row.conflicts : [])
    .map(obj)
    .filter((item): item is Record<string, unknown> => item !== null)
    .filter((item) => isDocumentId(item.document_id))
    .map((item) => ({ document_id: item.document_id as string }));
  return {
    status: row.status,
    results,
    conflicts,
    retrieval: text(row.retrieval, 'unknown'),
    message: maybeText(row.message),
  };
}

export function parseSubjects(value: unknown): Subject[] {
  const body = obj(value);
  const rows = Array.isArray(body?.subjects) ? body.subjects : [];
  const subjects: Subject[] = [];
  for (const entry of rows) {
    const row = obj(entry);
    if (!row || !isSubjectId(row.subject_id)) continue;
    subjects.push({
      subject_id: row.subject_id,
      display_name: text(row.display_name),
      aliases: Array.isArray(row.aliases)
        ? row.aliases.filter((item): item is string => typeof item === 'string')
        : [],
    });
  }
  return subjects;
}

/** What the page says about one uploaded file, in the user's words not the code's. */
export function ingestExplanation(version: Pick<Version, 'ingest_status' | 'reason'>): string {
  if (version.ingest_status === 'indexed') return 'Indexed and searchable.';
  switch (version.reason) {
    case 'needs_ocr':
      return 'Stored but not searchable: this looks like a scan and text recognition is not available in this release.';
    case 'password_required':
      return 'Stored but not searchable: the file is password protected.';
    case 'too_large':
      return 'Stored but not searchable: the file is larger than this library reads.';
    case 'timeout':
      return 'Stored but not searchable: reading the file took too long and was stopped.';
    case 'unsupported_type':
      return 'Stored but not searchable: that file type is not supported.';
    default:
      return 'Stored but not searchable: the file could not be read.';
  }
}
