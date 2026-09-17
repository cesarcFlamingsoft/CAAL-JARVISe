/**
 * The Company Library contract: what the browser is allowed to believe.
 *
 * The backend is trusted, but the parsers here are the last place a shape is
 * checked before a component renders it, so they are pinned: an unknown
 * classification, a forged identifier or a stray field never reaches a page,
 * and an unreadable upload is explained in words rather than a status code.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  ingestExplanation,
  isDocumentId,
  isEffectiveDate,
  isSubjectId,
  isVersionId,
  parseDocuments,
  parseSearch,
  parseStatus,
  parseSubjects,
} from './contract.ts';

const DOC = 'cdoc_' + 'a'.repeat(24);
const VER = 'cver_' + 'b'.repeat(24);
const EMP = 'emp_' + 'c'.repeat(16);

const version = {
  version_id: VER,
  classification: 'policy',
  status: 'current',
  effective_date: '2026-01-01',
  version_label: null,
  filename: 'FIXTURE-policy.txt',
  ingest_status: 'indexed',
  reason: null,
  page_count: null,
  chunk_count: 3,
  byte_count: 400,
  subjects: [EMP],
};

test('identifiers are checked, not assumed', () => {
  assert.equal(isDocumentId(DOC), true);
  assert.equal(isDocumentId('cdoc_nope'), false);
  assert.equal(isDocumentId('../../etc/passwd'), false);
  assert.equal(isVersionId(VER), true);
  assert.equal(isSubjectId(EMP), true);
  assert.equal(isSubjectId('emp_'), false);
});

test('an effective date is an ISO date or nothing at all', () => {
  assert.equal(isEffectiveDate('2026-01-01'), true);
  assert.equal(isEffectiveDate('last tuesday'), false);
  assert.equal(isEffectiveDate(undefined), false);
});

test('documents parse with their versions', () => {
  const documents = parseDocuments({
    documents: [{ document_id: DOC, title: 'FIXTURE Policy', versions: [version] }],
  });
  assert.equal(documents.length, 1);
  assert.equal(documents[0].versions[0].classification, 'policy');
  assert.deepEqual(documents[0].versions[0].subjects, [EMP]);
});

test('a document with an unknown classification is dropped, not rendered', () => {
  const documents = parseDocuments({
    documents: [
      {
        document_id: DOC,
        title: 'FIXTURE',
        versions: [{ ...version, classification: 'top-secret' }],
      },
    ],
  });
  assert.deepEqual(documents[0].versions, []);
});

test('a forged document id never reaches the page', () => {
  assert.deepEqual(parseDocuments({ documents: [{ document_id: 'cdoc_x', title: 'x' }] }), []);
});

test('status carries the honest retrieval label and the upload limits', () => {
  const status = parseStatus({
    company_name: 'FIXTURE Org',
    document_count: 2,
    indexed_versions: 2,
    failed_versions: 1,
    passage_count: 9,
    subject_count: 0,
    conflicting_documents: [DOC],
    retrieval: 'lexical_bm25',
    retrieval_note: 'Keyword matching.',
    accepted_extensions: ['.txt', '.md'],
    max_upload_bytes: 1024,
    unsupported_note: 'Scans need OCR.',
    capacity: { documents: 200, passages: 20000 },
  });
  assert.equal(status?.retrieval, 'lexical_bm25');
  assert.deepEqual(status?.conflicting_documents, [DOC]);
  assert.equal(status?.capacity.documents, 200);
  assert.equal(status?.max_upload_bytes, 1024);
});

test('an unnamed company stays unnamed', () => {
  const status = parseStatus({ document_count: 0, company_name: null });
  assert.equal(status?.company_name, null);
});

test('search results keep their citation and their snippet as plain text', () => {
  const outcome = parseSearch({
    status: 'ok',
    retrieval: 'lexical_bm25',
    results: [
      {
        document_id: DOC,
        version_id: VER,
        chunk_id: 'cchk_1',
        title: 'FIXTURE Policy',
        location: 'page 2',
        classification: 'policy',
        version_status: 'current',
        version_label: null,
        effective_date: '2026-01-01',
        snippet: '<script>alert(1)</script> three days each week',
      },
    ],
    conflicts: [],
  });
  assert.equal(outcome?.results.length, 1);
  assert.equal(outcome?.results[0].location, 'page 2');
  // Kept verbatim as text. It is rendered as a text node, never as markup.
  assert.match(outcome!.results[0].snippet, /three days each week/);
});

test('a conflicting-versions outcome survives the parse', () => {
  const outcome = parseSearch({
    status: 'conflicting_versions',
    results: [],
    conflicts: [{ document_id: DOC }],
    message: 'Two versions are current.',
  });
  assert.equal(outcome?.status, 'conflicting_versions');
  assert.deepEqual(outcome?.conflicts, [{ document_id: DOC }]);
});

test('subjects parse and a malformed one is dropped', () => {
  const subjects = parseSubjects({
    subjects: [
      { subject_id: EMP, display_name: 'Fixture Person A', aliases: ['A. Fixture'] },
      { subject_id: 'nope', display_name: 'Fixture Person B' },
    ],
  });
  assert.equal(subjects.length, 1);
  assert.equal(subjects[0].display_name, 'Fixture Person A');
});

test('an unreadable upload is explained in words', () => {
  assert.match(
    ingestExplanation({ ingest_status: 'failed', reason: 'needs_ocr' }),
    /scan|recognition/i
  );
  assert.match(
    ingestExplanation({ ingest_status: 'failed', reason: 'password_required' }),
    /password/i
  );
  assert.equal(
    ingestExplanation({ ingest_status: 'indexed', reason: null }),
    'Indexed and searchable.'
  );
  assert.match(ingestExplanation({ ingest_status: 'failed', reason: null }), /could not be read/);
});
