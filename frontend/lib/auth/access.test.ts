import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { SignJWT, createLocalJWKSet, exportJWK, generateKeyPair } from 'jose';

import { AccessError, normalizeEmail, verifyAccessAssertion } from './access.ts';

const TEAM = 'https://example-team.cloudflareaccess.com';
const AUD = 'd1d09a2c79e964918d59077b9bb5b3a7b67ff76a04a80822eb8a3ce5f46354ac';
const NOW = 1_700_000_000;

const keys = await (async () => {
  const a = await generateKeyPair('RS256');
  const b = await generateKeyPair('RS256');
  const jwkA = { ...(await exportJWK(a.publicKey)), kid: 'kid-a', alg: 'RS256', use: 'sig' };
  const jwkB = { ...(await exportJWK(b.publicKey)), kid: 'kid-b', alg: 'RS256', use: 'sig' };
  return { a, b, jwks: createLocalJWKSet({ keys: [jwkA] }), jwkB };
})();

async function token(
  overrides: Record<string, unknown> = {},
  { key = keys.a.privateKey, kid = 'kid-a', ttl = 600 } = {}
) {
  const claims: Record<string, unknown> = {
    email: 'Cesar@MexcanTech.com',
    type: 'app',
    identity_nonce: 'abc',
    country: 'CA',
    ...overrides,
  };
  let builder = new SignJWT(claims)
    .setProtectedHeader({ alg: 'RS256', kid })
    .setIssuedAt(NOW)
    .setNotBefore(NOW)
    .setExpirationTime(NOW + ttl)
    .setSubject('cf-subject');
  if (!('iss' in overrides)) builder = builder.setIssuer(TEAM);
  if (!('aud' in overrides)) builder = builder.setAudience([AUD]);
  return builder.sign(key);
}

const options = () => ({ teamDomain: TEAM, audience: AUD, jwks: keys.jwks, now: NOW + 5 });

describe('Cloudflare Access assertion verification', () => {
  it('accepts a valid application token and normalizes the email', async () => {
    const identity = await verifyAccessAssertion(await token(), options());

    assert.equal(identity.email, 'cesar@mexcantech.com');
    assert.equal(identity.subject, 'cf-subject');
    assert.equal(identity.expiresAt, NOW + 600);
  });

  it('rejects the wrong issuer, audience, expiry and signing key', async () => {
    for (const bad of [
      token({ iss: 'https://other.cloudflareaccess.com' }),
      token({ aud: ['0'.repeat(64)] }),
      token({}, { ttl: -120 }),
      token({}, { key: keys.b.privateKey, kid: 'kid-a' }),
      token({}, { key: keys.b.privateKey, kid: 'kid-b' }),
    ]) {
      await assert.rejects(verifyAccessAssertion(await bad, options()), AccessError);
    }
  });

  it('rejects service tokens, meta tokens and tokens without a usable email', async () => {
    await assert.rejects(verifyAccessAssertion(await token({ email: undefined }), options()));
    await assert.rejects(
      verifyAccessAssertion(await token({ email: undefined, common_name: 'svc' }), options())
    );
    await assert.rejects(verifyAccessAssertion(await token({ type: 'meta' }), options()));
    await assert.rejects(verifyAccessAssertion(await token({ email: 'not-an-email' }), options()));
    await assert.rejects(verifyAccessAssertion(await token({ email: 42 }), options()));
  });

  it('rejects unsigned or foreign-algorithm tokens and garbage', async () => {
    const [header, body] = (await token()).split('.');
    const unsignedHeader = Buffer.from(JSON.stringify({ alg: 'none', kid: 'kid-a' })).toString(
      'base64url'
    );
    for (const bad of ['', 'a.b.c', `${unsignedHeader}.${body}.`, `${header}.${body}.AAAA`]) {
      await assert.rejects(verifyAccessAssertion(bad, options()), AccessError);
    }
  });

  it('does not accept a bare email header as identity (only tokens are verified)', async () => {
    await assert.rejects(verifyAccessAssertion(undefined, options()), AccessError);
    await assert.rejects(verifyAccessAssertion(null, options()), AccessError);
  });
});

describe('email normalization', () => {
  it('lower-cases and trims', () => {
    assert.equal(normalizeEmail(' Cesar@MexcanTech.com\n'), 'cesar@mexcantech.com');
  });

  it('rejects unusable values', () => {
    for (const bad of ['', 'cesar', 'a@b', 'a b@c.com', 'a@b.com' + String.fromCharCode(0), 'a'.repeat(300), 42]) {
      assert.throws(() => normalizeEmail(bad));
    }
  });
});
