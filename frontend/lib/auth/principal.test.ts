import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { jwtVerify } from 'jose';

import {
  AUDIENCE_AGENT,
  AUDIENCE_BACKEND,
  AUDIENCE_IDENTITY,
  ISSUER_BFF,
  MAX_TTL_SECONDS,
  MIN_SECRET_LENGTH,
  mintPrincipal,
} from './principal.ts';

const SECRET = 's'.repeat(48);
const NOW = 1_700_000_000;
const USER = 'usr_0123456789abcdef01234567';

async function decode(token: string, audience: string) {
  const { payload, protectedHeader } = await jwtVerify(token, new TextEncoder().encode(SECRET), {
    algorithms: ['HS256'],
    issuer: ISSUER_BFF,
    audience,
    currentDate: new Date((NOW + 1) * 1000),
  });
  return { payload, protectedHeader };
}

describe('internal principal minting', () => {
  it('produces the exact HS256 shape the CAAL backend verifies', async () => {
    const token = await mintPrincipal({
      secret: SECRET,
      subject: USER,
      audience: AUDIENCE_BACKEND,
      now: NOW,
    });

    const { payload, protectedHeader } = await decode(token, AUDIENCE_BACKEND);
    assert.deepEqual(protectedHeader, { alg: 'HS256', typ: 'JWT' });
    assert.equal(payload.iss, ISSUER_BFF);
    assert.equal(payload.aud, AUDIENCE_BACKEND);
    assert.equal(payload.sub, USER);
    assert.equal(payload.iat, NOW);
    assert.equal(payload.nbf, NOW);
    assert.equal(payload.exp, NOW + 60);
    assert.equal(typeof payload.jti, 'string');
    assert.ok((payload.jti as string).length >= 16);
  });

  it('uses a fresh token id every time', async () => {
    const a = await mintPrincipal({ secret: SECRET, subject: USER, audience: AUDIENCE_BACKEND });
    const b = await mintPrincipal({ secret: SECRET, subject: USER, audience: AUDIENCE_BACKEND });
    assert.notEqual(a, b);
  });

  it('carries extra claims such as the room binding for agent principals', async () => {
    const token = await mintPrincipal({
      secret: SECRET,
      subject: USER,
      audience: AUDIENCE_AGENT,
      ttlSeconds: 300,
      claims: { room: 'caal-web-abc' },
      now: NOW,
    });

    const { payload } = await decode(token, AUDIENCE_AGENT);
    assert.equal(payload.room, 'caal-web-abc');
    assert.equal(payload.exp, NOW + 300);
  });

  it('carries an email only for identity assertions', async () => {
    const token = await mintPrincipal({
      secret: SECRET,
      subject: 'identity',
      audience: AUDIENCE_IDENTITY,
      claims: { email: 'cesar@example.com' },
      now: NOW,
    });

    const { payload } = await decode(token, AUDIENCE_IDENTITY);
    assert.equal(payload.email, 'cesar@example.com');
  });

  it('refuses weak secrets, bad subjects, bad audiences and out-of-range lifetimes', async () => {
    const base = { subject: USER, audience: AUDIENCE_BACKEND, now: NOW };
    await assert.rejects(mintPrincipal({ ...base, secret: '' }));
    await assert.rejects(mintPrincipal({ ...base, secret: 'x'.repeat(MIN_SECRET_LENGTH - 1) }));
    await assert.rejects(mintPrincipal({ ...base, secret: SECRET, subject: '' }));
    await assert.rejects(mintPrincipal({ ...base, secret: SECRET, subject: 'x'.repeat(300) }));
    await assert.rejects(mintPrincipal({ ...base, secret: SECRET, audience: '' }));
    await assert.rejects(mintPrincipal({ ...base, secret: SECRET, ttlSeconds: 0 }));
    await assert.rejects(
      mintPrincipal({ ...base, secret: SECRET, ttlSeconds: MAX_TTL_SECONDS + 1 })
    );
  });

  it('never lets extra claims override the reserved ones', async () => {
    await assert.rejects(
      mintPrincipal({
        secret: SECRET,
        subject: USER,
        audience: AUDIENCE_BACKEND,
        claims: { sub: 'usr_other' },
      })
    );
    await assert.rejects(
      mintPrincipal({
        secret: SECRET,
        subject: USER,
        audience: AUDIENCE_BACKEND,
        claims: { exp: 9_999_999_999 },
      })
    );
  });
});
