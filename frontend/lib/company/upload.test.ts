/**
 * The browser end of the upload contract.
 *
 * The BFF used to call `req.formData()`, which reads and parses the whole
 * request before anything has checked how big it is -- the bound was applied
 * to `file.size` *after* the parse. That is the bound arriving too late, and
 * on the Node runtime a large multipart part is buffered to do it.
 *
 * So the page sends the same thing the backend now takes: the raw bytes as the
 * body and one bounded base64url JSON envelope in a header. This module is the
 * shared, testable half of that: building an envelope, validating one, and
 * knowing the bound. The route keeps the streaming and the auth.
 *
 * Nothing confidential goes in a URL. A title, a filename and a list of
 * employee subject ids are the confidential part of an upload, and a query
 * string is the part of a request that every access log records.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  MAX_UPLOAD_BYTES,
  METADATA_HEADER,
  decodeUploadMetadata,
  encodeUploadMetadata,
  parseUploadMetadata,
} from './upload.ts';

const DOC = 'cdoc_' + 'a'.repeat(24);
const EMP = 'emp_' + 'c'.repeat(16);
const CANARY = 'FIXTURE_SECRET_zzqx7731';

const valid = {
  filename: 'FIXTURE-policy.txt',
  title: 'FIXTURE Remote Work Policy',
  classification: 'policy',
  status: 'current',
} as const;

test('the header is the one the backend reads', () => {
  assert.equal(METADATA_HEADER, 'X-CAAL-Company-Metadata');
});

test('an envelope round-trips exactly', () => {
  const encoded = encodeUploadMetadata({
    ...valid,
    effective_date: '2026-01-01',
    version_label: 'Rev B',
    document_id: DOC,
    subjects: [EMP],
  });
  assert.deepEqual(decodeUploadMetadata(encoded), {
    ...valid,
    effective_date: '2026-01-01',
    version_label: 'Rev B',
    document_id: DOC,
    subjects: [EMP],
  });
});

test('the envelope is base64url with no padding, so it is a legal header value', () => {
  const encoded = encodeUploadMetadata({ ...valid, title: 'A “smart quoted” tïtle — ok' });
  assert.match(encoded, /^[A-Za-z0-9_-]+$/);
  assert.equal(decodeUploadMetadata(encoded)?.title, 'A “smart quoted” tïtle — ok');
});

test('absent optional fields are absent, not empty strings', () => {
  const decoded = decodeUploadMetadata(encodeUploadMetadata(valid));
  assert.equal('effective_date' in (decoded as object), false);
  assert.equal('version_label' in (decoded as object), false);
  assert.equal('document_id' in (decoded as object), false);
});

test('a valid envelope is accepted by the validator', () => {
  assert.deepEqual(parseUploadMetadata(valid), { ok: true, value: { ...valid, subjects: [] } });
});

test('an unknown field is refused rather than dropped', () => {
  const result = parseUploadMetadata({ ...valid, secret_flag: true });
  assert.equal(result.ok, false);
});

const rejected: Array<[string, Record<string, unknown>]> = [
  ['a bad classification', { ...valid, classification: 'confidential' }],
  ['a bad status', { ...valid, status: 'archived' }],
  ['an empty title', { ...valid, title: '   ' }],
  ['an empty filename', { ...valid, filename: '' }],
  ['an over-long title', { ...valid, title: 'x'.repeat(201) }],
  ['an over-long version label', { ...valid, version_label: 'y'.repeat(61) }],
  ['a free-text date', { ...valid, effective_date: 'January 2026' }],
  ['a forged document id', { ...valid, document_id: 'cdoc_nope' }],
  ['a forged subject id', { ...valid, subjects: ['emp_nope'] }],
  ['subjects that are not a list', { ...valid, subjects: EMP }],
  ['an unsupported extension', { ...valid, filename: 'payload.exe' }],
  ['no extension at all', { ...valid, filename: 'policy' }],
  ['a path in the filename', { ...valid, filename: '../../etc/policy.txt' }],
  ['nothing at all', {}],
];

for (const [name, input] of rejected) {
  test(`the validator refuses ${name}`, () => {
    assert.equal(parseUploadMetadata(input).ok, false);
  });
}

test('the envelope stays inside the bound the backend enforces', () => {
  const encoded = encodeUploadMetadata({
    ...valid,
    title: 'T'.repeat(200),
    filename: 'F'.repeat(190) + '.txt',
    version_label: 'V'.repeat(60),
    document_id: DOC,
    subjects: Array.from({ length: 32 }, () => EMP),
  });
  assert.ok(encoded.length <= 2048, `envelope was ${encoded.length} characters`);
});

test('a malformed envelope decodes to nothing rather than throwing', () => {
  assert.equal(decodeUploadMetadata('not base64 at all!!'), null);
  assert.equal(decodeUploadMetadata(''), null);
  assert.equal(decodeUploadMetadata(Buffer.from('just bytes').toString('base64url')), null);
});

test('the upload bound is a real number the page can show', () => {
  assert.equal(MAX_UPLOAD_BYTES, 8 * 1024 * 1024);
});

test('nothing in a built envelope is readable without decoding it', () => {
  // Not a confidentiality claim -- base64 is not encryption. It is a check
  // that the value cannot be grepped out of a log by accident.
  const encoded = encodeUploadMetadata({ ...valid, title: CANARY });
  assert.equal(encoded.includes(CANARY), false);
});
