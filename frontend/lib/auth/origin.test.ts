import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { expectedOrigin, isTrustedMutationOrigin } from './origin.ts';

function headersWith(entries: Record<string, string>): Headers {
  return new Headers(entries);
}

describe('mutation origin checks', () => {
  it('derives the expected origin from the forwarded request when none is configured', () => {
    assert.equal(
      expectedOrigin(headersWith({ host: 'jarvis.example.com', 'x-forwarded-proto': 'https' })),
      'https://jarvis.example.com'
    );
    assert.equal(
      expectedOrigin(headersWith({ host: '192.168.1.5:3000' })),
      'http://192.168.1.5:3000'
    );
    assert.equal(
      expectedOrigin(headersWith({ host: 'evil' }), 'https://jarvis.example.com'),
      'https://jarvis.example.com'
    );
  });

  it('accepts a matching Origin header only', () => {
    const base = { host: 'jarvis.example.com', 'x-forwarded-proto': 'https' };
    assert.equal(
      isTrustedMutationOrigin(headersWith({ ...base, origin: 'https://jarvis.example.com' })),
      true
    );
    assert.equal(
      isTrustedMutationOrigin(headersWith({ ...base, origin: 'https://evil.example.com' })),
      false
    );
    assert.equal(
      isTrustedMutationOrigin(headersWith({ ...base, origin: 'http://jarvis.example.com' })),
      false
    );
    assert.equal(isTrustedMutationOrigin(headersWith({ ...base, origin: 'null' })), false);
  });

  it('falls back to the Referer origin and otherwise fails closed', () => {
    const base = { host: 'jarvis.example.com', 'x-forwarded-proto': 'https' };
    assert.equal(
      isTrustedMutationOrigin(
        headersWith({ ...base, referer: 'https://jarvis.example.com/admin?x=1' })
      ),
      true
    );
    assert.equal(
      isTrustedMutationOrigin(headersWith({ ...base, referer: 'https://evil.example.com/' })),
      false
    );
    assert.equal(isTrustedMutationOrigin(headersWith(base)), false);
    assert.equal(isTrustedMutationOrigin(headersWith({ ...base, referer: 'garbage' })), false);
  });

  it('honours a configured public origin over the Host header', () => {
    const headers = headersWith({
      host: 'frontend:3000',
      origin: 'https://jarvis.example.com',
    });
    assert.equal(isTrustedMutationOrigin(headers, 'https://jarvis.example.com'), true);
    assert.equal(isTrustedMutationOrigin(headers, 'https://other.example.com'), false);
  });
});

it('accepts only explicitly configured local origins alongside the public origin', () => {
  const publicOrigin = 'https://jarvis.mexcantech.io';
  const local = 'http://localhost:3000';
  const headers = new Headers({ host: 'hostile.example', origin: local });
  assert.equal(isTrustedMutationOrigin(headers, publicOrigin), false);
  assert.equal(isTrustedMutationOrigin(headers, publicOrigin, [local]), true);
  headers.set('origin', publicOrigin);
  assert.equal(isTrustedMutationOrigin(headers, publicOrigin, [local]), true);
  for (const origin of [
    'http://localhost:3001',
    'http://127.0.0.1:3000',
    'http://192.168.1.2:3000',
    'https://evil.example',
    'null',
  ]) {
    headers.set('origin', origin);
    assert.equal(isTrustedMutationOrigin(headers, publicOrigin, [local]), false, origin);
  }
});

it('rejects malformed Origin values instead of normalizing them into trust', () => {
  for (const base of ['http://localhost:3000', 'https://jarvis.mexcantech.io']) {
    for (const origin of [
      `${base}/`,
      `${base}/path`,
      `${base}?x=1`,
      `${base}#x`,
      base.replace('://', '://user@'),
      base.replace('://', ':////'),
      `${base}, https://evil.example`,
      base.replace('localhost', 'LOCALHOST') + '/',
    ]) {
      const headers = new Headers({ origin, referer: 'http://localhost:3000/login' });
      assert.equal(
        isTrustedMutationOrigin(headers, 'https://jarvis.mexcantech.io', ['http://localhost:3000']),
        false,
        origin
      );
    }
  }
});
