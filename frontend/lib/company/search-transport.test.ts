import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

/**
 * Structural guard: the company search query never appears in a URL.
 *
 * `GET /api/admin/company/search?q=...` wrote the confidential half of a
 * company question into the request line, which the Next.js request log, any
 * reverse proxy in front of it and the browser's own history all record. A
 * probe (`reports/company-knowledge/verify-query-log.txt`) found a canary
 * query intact in a real access-log record.
 *
 * So the BFF route, the backend call it makes, and the page that calls the BFF
 * all carry the query in a bounded POST body. These files import Next.js, so
 * they cannot run under `node --test`; what is checked is that none of the
 * three builds a URL out of a query, and that the POST keeps read semantics --
 * the same session and origin guards, the existing read rate limit, and no
 * mutation rate limit, because searching changes nothing.
 */

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = (path: string) => readFileSync(join(ROOT, path), 'utf8');

const BFF = 'lib/company/route.ts';
const ROUTE = 'app/api/admin/company/search/route.ts';
const PAGE = 'app/admin/company/page.tsx';
const BACKEND = join(ROOT, '..', 'src', 'caal', 'company_api.py');

/** Just `searchRoute`, so a neighbouring route's guards cannot satisfy a check. */
function searchRouteSource(): string {
  const source = read(BFF);
  const start = source.indexOf('export async function searchRoute');
  assert.ok(start >= 0, 'searchRoute must exist');
  const next = source.indexOf('export async function', start + 1);
  return source.slice(start, next === -1 ? undefined : next);
}

describe('the company search transport', () => {
  it('exposes POST and not GET on the BFF route', () => {
    const source = read(ROUTE);
    assert.match(source, /export const POST\s*=/);
    assert.ok(!/export const GET\s*=/.test(source), 'the GET form must be gone');
    assert.match(source, /export const dynamic = 'force-dynamic'/);
  });

  it('reads the query from the body and never from the URL', () => {
    const handler = searchRouteSource();
    assert.ok(
      !/searchParams/.test(handler),
      'searchRoute must not read query parameters'
    );
    assert.ok(
      !/URLSearchParams/.test(handler),
      'searchRoute must not build a query string'
    );
    assert.match(handler, /readJsonObject\(req\)/);
  });

  it('calls the backend with a POST body, not a query string', () => {
    const handler = searchRouteSource();
    assert.match(handler, /\$\{BACKEND\}\/search`/);
    assert.ok(!/\/search\?/.test(handler), 'the backend call must carry no query string');
    assert.match(handler, /method: 'POST'/);
  });

  it('keeps read semantics: session and origin guarded, read-limited, not mutation-limited', () => {
    const handler = searchRouteSource();
    // requireAdmin applies readLimited for every caller.
    assert.match(handler, /requireAdmin\(req\)/);
    assert.match(handler, /guardReadPost\(req, auth\.config\)/);
    assert.ok(
      !/guardMutation/.test(handler),
      'searching is a read and must not consume the mutation budget'
    );
  });

  it('validates the body as strictly as the query form was validated', () => {
    const handler = searchRouteSource();
    assert.match(handler, /MAX_QUERY/);
    assert.match(handler, /isClassification/);
    assert.match(handler, /isSubjectId/);
  });

  it('has a read-POST guard that checks origin and CSRF without a mutation budget', () => {
    const guard = read('lib/auth/guard.ts');
    const fn = guard.slice(guard.indexOf('export function guardReadPost'));
    assert.ok(fn.length > 0, 'guardReadPost must exist');
    const body = fn.slice(0, fn.indexOf('\n}\n'));
    assert.match(body, /isTrustedMutationOrigin/);
    assert.match(body, /verifyCsrf/);
    assert.ok(!/mutationLimiter/.test(body), 'a read must not spend the mutation budget');
  });

  it('has the page POST its query instead of interpolating it into a URL', () => {
    const source = read(PAGE);
    assert.ok(
      !/company\/search\?/.test(source),
      'the page must not put the query in the URL'
    );
    const call = source.slice(source.indexOf("'/api/admin/company/search'"));
    assert.ok(call.length > 0, 'the page must call the search route by bare path');
    assert.match(call.slice(0, 600), /method: 'POST'/);
    // `apiRequest` serialises `body` and attaches the CSRF header, which is
    // what `guardReadPost` checks on the other side.
    assert.match(call.slice(0, 600), /body: \{ query/);
  });

  it('matches the backend, which now declares the route as a POST', () => {
    const python = readFileSync(BACKEND, 'utf8');
    assert.match(python, /@router\.post\("\/search"\)/);
    assert.ok(!/@router\.get\("\/search"\)/.test(python), 'the backend GET must be gone');
  });
});
