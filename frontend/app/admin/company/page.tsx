'use client';

/**
 * Company Library: where the owner feeds FRIDAY the company's own documents.
 *
 * Everything on this page is the management plane. The model never reaches it;
 * it reads the same library through a read-only MCP service on loopback.
 *
 * The page is deliberately honest about three things the user would otherwise
 * have to discover the hard way: that retrieval is keyword matching and not
 * semantic search, that a scanned or password-protected file is stored but not
 * searchable, and that two versions marked current will make FRIDAY refuse to
 * pick between them.
 *
 * Every string that came from a document is rendered as a text node. Nothing
 * here uses dangerouslySetInnerHTML, and no uploaded text is ever executed,
 * followed, or treated as an instruction.
 */
import { useCallback, useEffect, useState } from 'react';
import { apiRequest, explain, loadCsrfToken } from '@/components/account/api-client';
import {
  CLASSIFICATIONS,
  type Citation,
  type Classification,
  type Document,
  type LibraryStatus,
  type SearchOutcome,
  type Subject,
  VERSION_STATUSES,
  type VersionStatus,
  ingestExplanation,
  parseDocuments,
  parseSearch,
  parseStatus,
  parseSubjects,
} from '@/lib/company/contract';
import { COMPANY_SESSION_HREF } from '@/lib/company/session-mode';
import {
  ACCEPTED_EXTENSIONS,
  MAX_UPLOAD_BYTES,
  METADATA_HEADER,
  encodeUploadMetadata,
  parseUploadMetadata,
} from '@/lib/company/upload';

const CLASSIFICATION_LABELS: Record<Classification, string> = {
  policy: 'Policy',
  hr: 'HR document',
  contract: 'Contract',
  general: 'General',
};
const STATUS_LABELS: Record<VersionStatus, string> = {
  current: 'Current',
  superseded: 'Superseded',
  draft: 'Draft',
};

const field =
  'border-input bg-background text-foreground focus-visible:ring-ring w-full rounded-md border px-3 py-2 text-sm focus-visible:ring-2 focus-visible:outline-none';
const button =
  'border-input hover:bg-accent focus-visible:ring-ring rounded-md border px-3 py-1.5 text-sm font-medium focus-visible:ring-2 focus-visible:outline-none disabled:opacity-50';

function bytes(count: number): string {
  if (count < 1024) return `${count} B`;
  if (count < 1024 * 1024) return `${Math.round(count / 1024)} kB`;
  return `${(count / (1024 * 1024)).toFixed(1)} MB`;
}

export default function CompanyLibraryPage() {
  const [status, setStatus] = useState<LibraryStatus | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [people, setPeople] = useState<Subject[]>([]);
  const [outcome, setOutcome] = useState<SearchOutcome | null>(null);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [reported, listed, subjects] = await Promise.all([
      apiRequest<unknown>('/api/admin/company/status'),
      apiRequest<unknown>('/api/admin/company/documents'),
      apiRequest<unknown>('/api/admin/company/people'),
    ]);
    if (!reported.ok) {
      setMessage(explain(reported.error));
      return;
    }
    setStatus(parseStatus(reported.data));
    if (listed.ok) setDocuments(parseDocuments(listed.data));
    if (subjects.ok) setPeople(parseSubjects(subjects.data));
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const upload = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const element = event.currentTarget;
    const form = new FormData(element);
    const file = form.get('file');
    if (!(file instanceof File) || file.size === 0) {
      setMessage('Choose a file to upload.');
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setMessage(`That file is larger than the ${bytes(MAX_UPLOAD_BYTES)} limit.`);
      return;
    }
    // The metadata is validated here, packed into one header envelope, and
    // never put in the URL: a title and a filename say what a document is, and
    // a query string is the part of a request that access logs record.
    const chosen = form
      .getAll('subjects')
      .filter((item): item is string => typeof item === 'string');
    const metadata = parseUploadMetadata({
      filename: file.name,
      title: String(form.get('title') ?? ''),
      classification: String(form.get('classification') ?? ''),
      status: String(form.get('status') ?? ''),
      effective_date: String(form.get('effective_date') ?? '') || undefined,
      version_label: String(form.get('version_label') ?? '') || undefined,
      document_id: String(form.get('document_id') ?? '') || undefined,
      subjects: chosen,
    });
    if (!metadata.ok) {
      setMessage(
        metadata.field === 'effective_date'
          ? 'The effective date must be a calendar date, such as 2026-01-01, or left blank.'
          : metadata.field === 'filename'
            ? `That file type is not read by this library. Accepted: ${ACCEPTED_EXTENSIONS.join(', ')}.`
            : 'Check the upload details: something there is missing or not a value this page accepts.'
      );
      return;
    }
    setBusy(true);
    setMessage('');
    const token = await loadCsrfToken();
    if (!token) {
      setBusy(false);
      setMessage('This page could not verify your session. Reload and try again.');
      return;
    }
    try {
      const response = await fetch('/api/admin/company/documents', {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'X-CAAL-CSRF': token,
          'Content-Type': 'application/octet-stream',
          [METADATA_HEADER]: encodeUploadMetadata(metadata.value),
        },
        credentials: 'same-origin',
        cache: 'no-store',
        body: file,
      });
      const data = (await response.json().catch(() => null)) as {
        ingest_status?: string;
        reason?: string | null;
        chunk_count?: number;
        error?: string;
      } | null;
      if (!response.ok) {
        setMessage(
          data?.error === 'invalid_effective_date'
            ? 'The effective date must be a calendar date, such as 2026-01-01, or left blank.'
            : explain(data?.error ?? `http_${response.status}`)
        );
      } else if (data?.ingest_status === 'indexed') {
        setMessage(`Indexed. ${data.chunk_count ?? 0} passage(s) are now searchable.`);
        element.reset();
      } else {
        setMessage(
          ingestExplanation({
            ingest_status: data?.ingest_status ?? 'failed',
            reason: data?.reason ?? null,
          })
        );
        element.reset();
      }
      await load();
    } finally {
      setBusy(false);
    }
  };

  const changeStatus = async (versionId: string, next: VersionStatus) => {
    setBusy(true);
    const result = await apiRequest<unknown>(`/api/admin/company/versions/${versionId}`, {
      method: 'PATCH',
      body: { status: next },
    });
    setMessage(result.ok ? `Marked ${STATUS_LABELS[next].toLowerCase()}.` : explain(result.error));
    await load();
    setBusy(false);
  };

  const remove = async (documentId: string) => {
    setBusy(true);
    const result = await apiRequest<{ note?: string }>(
      `/api/admin/company/documents/${documentId}`,
      { method: 'DELETE' }
    );
    setMessage(
      result.ok
        ? (result.data?.note ?? 'Removed from the library index and its encrypted source store.')
        : explain(result.error)
    );
    setConfirming(null);
    await load();
    setBusy(false);
  };

  const search = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const query = String(new FormData(event.currentTarget).get('q') ?? '').trim();
    if (!query) return;
    setBusy(true);
    // A POST, not a GET with `?q=`: the query is the confidential half of the
    // question, and a query string is recorded by the request log, by any
    // proxy in front of it and by this browser's own history.
    const result = await apiRequest<unknown>('/api/admin/company/search', {
      method: 'POST',
      body: { query },
    });
    if (result.ok) setOutcome(parseSearch(result.data));
    else setMessage(explain(result.error));
    setBusy(false);
  };

  const addPerson = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const name = String(new FormData(form).get('display_name') ?? '').trim();
    if (!name) return;
    setBusy(true);
    const result = await apiRequest<unknown>('/api/admin/company/people', {
      method: 'POST',
      body: { display_name: name },
    });
    setMessage(
      result.ok
        ? 'Added. Two people who share a name stay two separate records, and FRIDAY will ask which one you mean.'
        : explain(result.error)
    );
    form.reset();
    await load();
    setBusy(false);
  };

  const heading = status?.company_name ?? 'Company';

  return (
    <main className="friday-page mx-auto max-w-5xl px-4 py-24 md:py-28">
      <h1 className="text-foreground text-2xl font-semibold">{heading} Library</h1>
      <p className="text-muted-foreground mt-2 max-w-3xl text-sm">
        The documents FRIDAY may quote when you ask about your company: policies, HR documents,
        contracts and anything else you upload. They are encrypted on this machine, they belong to
        your account alone, and they are never sent to a cloud model.
      </p>

      {message && (
        <p
          role="status"
          aria-live="polite"
          className="border-input bg-muted/40 text-foreground mt-6 rounded-lg border px-4 py-3 text-sm"
        >
          {message}
        </p>
      )}

      {/* --- how to ask FRIDAY about any of this ------------------------------------ */}
      <section
        aria-labelledby="company-session"
        className="friday-panel border-input bg-muted/30 mt-8 rounded-lg border p-5"
      >
        <h2 id="company-session" className="text-foreground text-lg font-medium">
          Asking FRIDAY about these documents
        </h2>
        <p className="text-muted-foreground mt-2 max-w-3xl text-sm">
          FRIDAY can only read this library in a <strong>company session</strong>. Start one from
          here, ask what you need, and close it when you are done. Your ordinary sessions are
          unchanged: the cloud assistant, your connected mail and calendar accounts, Home Assistant
          and everything else keep working exactly as they did.
        </p>
        <ul className="text-muted-foreground mt-3 max-w-3xl list-disc space-y-1 pl-5 text-sm">
          <li>
            A company session answers with the local model only. Nothing in it is sent to a cloud
            model, and if the local model cannot answer, FRIDAY says so instead of asking anything
            else.
          </li>
          <li>
            <strong className="text-foreground">
              A company session can only answer questions.
            </strong>{' '}
            It has exactly two capabilities: search the library, and read an excerpt from one
            document. There is no email, no calendar, no web, no Home Assistant, no background work
            and no delegation — and also no reminders, no alarms and no notes to memory, because a
            reminder is a message sent later and a note is read back in an ordinary session, where
            the cloud runtime is in the path. A passage in an uploaded document cannot talk FRIDAY
            into forwarding anything, because there is nothing there to forward with.
          </li>
          <li>
            Ask for a reminder in a company session and you will not get one. That is the cost of
            the boundary, and it is the whole cost: an ordinary session still does all of it.
          </li>
          <li>
            It cannot be turned off partway through. Leaving is starting a fresh ordinary session.
          </li>
          <li>
            <strong className="text-foreground">
              Please start a company session before you say anything confidential.
            </strong>{' '}
            In an ordinary session FRIDAY recognises some obvious company wording and stops rather
            than answering, but that check is deliberately simple and does not catch every way of
            asking. The boundary you can rely on is this one: the documents are readable only inside
            a company session.
          </li>
        </ul>
        <a
          className="border-input hover:bg-accent focus-visible:ring-ring mt-4 inline-block rounded-md border px-3 py-1.5 text-sm font-medium focus-visible:ring-2 focus-visible:outline-none"
          href={COMPANY_SESSION_HREF}
        >
          Start a company session
        </a>
      </section>

      {/* --- what is in the library ------------------------------------------------ */}
      <section aria-labelledby="library-status" className="mt-8">
        <h2 id="library-status" className="text-foreground text-lg font-medium">
          Library
        </h2>
        {status && (
          <>
            <dl className="mt-3 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
              {[
                ['Documents', `${status.document_count} of ${status.capacity.documents}`],
                ['Searchable versions', String(status.indexed_versions)],
                ['Not searchable', String(status.failed_versions)],
                ['Passages', String(status.passage_count)],
              ].map(([label, value]) => (
                <div key={label} className="border-input rounded-lg border px-3 py-2">
                  <dt className="text-muted-foreground text-xs">{label}</dt>
                  <dd className="text-foreground font-medium">{value}</dd>
                </div>
              ))}
            </dl>
            <p className="text-muted-foreground mt-3 text-xs">
              Search is keyword matching over the document text (BM25). It is not semantic search: a
              word the document does not contain will not be found. {status.unsupported_note}
            </p>
            {status.conflicting_documents.length > 0 && (
              <p className="border-input bg-muted/40 text-foreground mt-3 rounded-lg border px-3 py-2 text-sm">
                {status.conflicting_documents.length} document(s) have more than one version marked
                current. FRIDAY will say it cannot tell which applies until one is marked
                superseded.
              </p>
            )}
          </>
        )}
        {status?.document_count === 0 && (
          <p className="border-input text-muted-foreground mt-4 rounded-lg border border-dashed px-4 py-6 text-sm">
            The library is empty. Upload a policy, a contract or a handbook below and FRIDAY will be
            able to quote it, with a citation, the next time you ask.
          </p>
        )}
      </section>

      {/* --- upload ----------------------------------------------------------------- */}
      <section aria-labelledby="upload-heading" className="mt-10">
        <h2 id="upload-heading" className="text-foreground text-lg font-medium">
          Add a document
        </h2>
        <form onSubmit={upload} className="mt-3 grid gap-3 sm:grid-cols-2">
          <div className="sm:col-span-2">
            <label htmlFor="file" className="text-foreground text-sm font-medium">
              File
            </label>
            <input
              id="file"
              name="file"
              type="file"
              required
              accept={status?.accepted_extensions.join(',') || '.txt,.md,.pdf,.docx'}
              className={field}
            />
            <p className="text-muted-foreground mt-1 text-xs">
              {status
                ? `${status.accepted_extensions.join(', ')} up to ${bytes(status.max_upload_bytes)}.`
                : 'Text, Markdown, text PDF or Word, up to 8 MB.'}{' '}
              Scanned pages and password-protected PDFs are stored but cannot be searched.
            </p>
          </div>
          <div>
            <label htmlFor="title" className="text-foreground text-sm font-medium">
              Title
            </label>
            <input id="title" name="title" type="text" required maxLength={200} className={field} />
          </div>
          <div>
            <label htmlFor="classification" className="text-foreground text-sm font-medium">
              Kind
            </label>
            <select
              id="classification"
              name="classification"
              defaultValue="policy"
              className={field}
            >
              {CLASSIFICATIONS.map((value) => (
                <option key={value} value={value}>
                  {CLASSIFICATION_LABELS[value]}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="status" className="text-foreground text-sm font-medium">
              Status
            </label>
            <select id="status" name="status" defaultValue="current" className={field}>
              {VERSION_STATUSES.map((value) => (
                <option key={value} value={value}>
                  {STATUS_LABELS[value]}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="effective_date" className="text-foreground text-sm font-medium">
              Effective date <span className="text-muted-foreground">(optional)</span>
            </label>
            <input
              id="effective_date"
              name="effective_date"
              type="date"
              className={field}
              aria-describedby="effective-help"
            />
            <p id="effective-help" className="text-muted-foreground mt-1 text-xs">
              Left blank, no date is recorded. Nothing is ever guessed from the file.
            </p>
          </div>
          <div>
            <label htmlFor="version_label" className="text-foreground text-sm font-medium">
              Version label <span className="text-muted-foreground">(optional)</span>
            </label>
            <input
              id="version_label"
              name="version_label"
              type="text"
              maxLength={60}
              className={field}
            />
          </div>
          <div className="sm:col-span-2">
            <button type="submit" disabled={busy} className={button}>
              {busy ? 'Working…' : 'Upload'}
            </button>
          </div>
        </form>
      </section>

      {/* --- the catalogue ----------------------------------------------------------- */}
      <section aria-labelledby="documents-heading" className="mt-10">
        <h2 id="documents-heading" className="text-foreground text-lg font-medium">
          Documents
        </h2>
        <ul className="mt-3 space-y-3">
          {documents.map((document) => (
            <li key={document.document_id} className="border-input rounded-lg border p-4">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h3 className="text-foreground font-medium">{document.title}</h3>
                {confirming === document.document_id ? (
                  <span className="flex items-center gap-2 text-sm">
                    <span className="text-muted-foreground">
                      Delete this document and every version of it?
                    </span>
                    <button
                      type="button"
                      className={button}
                      disabled={busy}
                      onClick={() => void remove(document.document_id)}
                    >
                      Delete
                    </button>
                    <button type="button" className={button} onClick={() => setConfirming(null)}>
                      Keep
                    </button>
                  </span>
                ) : (
                  <button
                    type="button"
                    className={button}
                    onClick={() => setConfirming(document.document_id)}
                  >
                    Delete
                  </button>
                )}
              </div>
              <ul className="mt-3 space-y-2">
                {document.versions.map((version) => (
                  <li key={version.version_id} className="text-sm">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-foreground font-medium">
                        {CLASSIFICATION_LABELS[version.classification]}
                      </span>
                      <span className="text-muted-foreground">·</span>
                      <span className="text-muted-foreground">{version.filename}</span>
                      {version.version_label && (
                        <span className="text-muted-foreground">({version.version_label})</span>
                      )}
                      {version.effective_date && (
                        <span className="text-muted-foreground">
                          effective {version.effective_date}
                        </span>
                      )}
                      <span className="text-muted-foreground">{bytes(version.byte_count)}</span>
                      <label className="sr-only" htmlFor={`status-${version.version_id}`}>
                        Status of {version.filename}
                      </label>
                      <select
                        id={`status-${version.version_id}`}
                        value={version.status}
                        disabled={busy}
                        onChange={(event) =>
                          void changeStatus(version.version_id, event.target.value as VersionStatus)
                        }
                        className="border-input bg-background text-foreground focus-visible:ring-ring rounded-md border px-2 py-1 text-xs focus-visible:ring-2 focus-visible:outline-none"
                      >
                        {VERSION_STATUSES.map((value) => (
                          <option key={value} value={value}>
                            {STATUS_LABELS[value]}
                          </option>
                        ))}
                      </select>
                    </div>
                    <p className="text-muted-foreground mt-1 text-xs">
                      {ingestExplanation(version)}
                      {version.page_count ? ` ${version.page_count} page(s).` : ''}
                    </p>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      </section>

      {/* --- see what FRIDAY would find ----------------------------------------------- */}
      <section aria-labelledby="search-heading" className="mt-10">
        <h2 id="search-heading" className="text-foreground text-lg font-medium">
          What FRIDAY would find
        </h2>
        <p className="text-muted-foreground mt-1 text-sm">
          Exactly the lookup the assistant performs, with the same citations.
        </p>
        <form onSubmit={search} className="mt-3 flex flex-wrap gap-2">
          <label className="sr-only" htmlFor="q">
            Search the company library
          </label>
          <input
            id="q"
            name="q"
            type="search"
            maxLength={300}
            placeholder="how many days can I work remotely"
            className={`${field} sm:max-w-md`}
          />
          <button type="submit" disabled={busy} className={button}>
            Search
          </button>
        </form>
        {outcome && (
          <div className="mt-4" aria-live="polite">
            {outcome.status === 'conflicting_versions' && (
              <p className="border-input bg-muted/40 text-foreground mb-3 rounded-lg border px-3 py-2 text-sm">
                More than one version of a matching document is marked current, so FRIDAY will say
                it cannot tell which one applies.
              </p>
            )}
            {outcome.results.length === 0 ? (
              <p className="text-muted-foreground text-sm">
                Nothing in the library matches those words. FRIDAY would say so rather than guess.
              </p>
            ) : (
              <ul className="space-y-3">
                {outcome.results.map((hit: Citation) => (
                  <li key={hit.chunk_id} className="border-input rounded-lg border p-3 text-sm">
                    <p className="text-foreground">{hit.snippet}</p>
                    <p className="text-muted-foreground mt-2 text-xs">
                      {hit.title} · {CLASSIFICATION_LABELS[hit.classification]} ·{' '}
                      {STATUS_LABELS[hit.version_status]} · {hit.location}
                      {hit.effective_date ? ` · effective ${hit.effective_date}` : ''}
                      {hit.version_label ? ` · ${hit.version_label}` : ''}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </section>

      {/* --- employee subjects --------------------------------------------------------- */}
      <section aria-labelledby="people-heading" className="mt-10 mb-12">
        <h2 id="people-heading" className="text-foreground text-lg font-medium">
          Employee records
        </h2>
        <p className="text-muted-foreground mt-1 max-w-3xl text-sm">
          A record is an identifier you create, not a name FRIDAY infers. Two people who share a
          name stay two records, and a question that matches both is answered by asking which one
          you mean — never by picking one. FRIDAY only quotes documents you have attached; it makes
          no assessment of anybody.
        </p>
        <form onSubmit={addPerson} className="mt-3 flex flex-wrap gap-2">
          <label className="sr-only" htmlFor="display_name">
            Name
          </label>
          <input
            id="display_name"
            name="display_name"
            type="text"
            maxLength={120}
            placeholder="Name"
            className={`${field} sm:max-w-xs`}
          />
          <button type="submit" disabled={busy} className={button}>
            Add record
          </button>
        </form>
        {people.length > 0 && (
          <ul className="mt-3 space-y-1 text-sm">
            {people.map((person) => (
              <li key={person.subject_id} className="text-foreground">
                {person.display_name}
                <span className="text-muted-foreground text-xs"> · {person.subject_id}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
