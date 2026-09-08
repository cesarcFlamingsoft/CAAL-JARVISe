import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  FLOW_COOKIE,
  FLOW_COOKIE_PATH,
  MAX_FLOW_SECONDS,
  clearedFlowCookie,
  flowCookieOptions,
  openFlow,
  sealFlow,
  sessionKeyFor,
} from './flow-cookie.ts';

const SECRET = 's'.repeat(48);
const OTHER_SECRET = 'o'.repeat(48);
const USER = 'usr_' + 'a'.repeat(24);
const OTHER_USER = 'usr_' + 'b'.repeat(24);
const STATE_ID = 'abcdefghijklmnopqrstuvwxyz012345';
const NOW = 1_700_000_000;

const binding = {
  userId: USER,
  sessionKey: 'session-key-one',
  provider: 'google',
  stateId: STATE_ID,
  expiresAt: NOW + 600,
};
const expected = { userId: USER, sessionKey: 'session-key-one', now: NOW };

describe('OAuth flow cookie', () => {
  it('seals a binding that only the same user in the same browser session can open', async () => {
    const value = await sealFlow(SECRET, binding);

    assert.ok(!value.includes(USER));
    assert.ok(!value.includes('session-key-one'));
    assert.ok(!value.includes(SECRET));
    assert.deepEqual(await openFlow(SECRET, value, expected), {
      provider: 'google',
      stateId: STATE_ID,
      expiresAt: NOW + 600,
    });
    assert.deepEqual(await openFlow(SECRET, value, { ...expected, now: NOW + 599 }), {
      provider: 'google',
      stateId: STATE_ID,
      expiresAt: NOW + 600,
    });
  });

  it('refuses another user, another session, a tampered field, another secret, and expiry', async () => {
    const value = await sealFlow(SECRET, binding);
    const [version, provider, stateId, expiresAt, signature] = value.split('.');
    const rebuilt = (parts: Partial<Record<'provider' | 'stateId' | 'expiresAt' | 'signature', string>>) =>
      [
        version,
        parts.provider ?? provider,
        parts.stateId ?? stateId,
        parts.expiresAt ?? expiresAt,
        parts.signature ?? signature,
      ].join('.');

    assert.equal(await openFlow(SECRET, value, { ...expected, userId: OTHER_USER }), null);
    assert.equal(await openFlow(SECRET, value, { ...expected, sessionKey: 'session-key-two' }), null);
    assert.equal(await openFlow(OTHER_SECRET, value, expected), null);
    assert.equal(await openFlow(SECRET, rebuilt({ provider: 'zoho' }), expected), null);
    assert.equal(
      await openFlow(SECRET, rebuilt({ stateId: STATE_ID.slice(0, 31) + '9' }), expected),
      null
    );
    assert.equal(await openFlow(SECRET, rebuilt({ expiresAt: String(NOW + 9000) }), expected), null);
    const flipped = (signature[0] === 'A' ? 'B' : 'A') + signature.slice(1);
    assert.equal(await openFlow(SECRET, rebuilt({ signature: flipped }), expected), null);
    assert.equal(await openFlow(SECRET, value, { ...expected, now: NOW + 600 }), null);
    assert.equal(await openFlow(SECRET, value, { ...expected, now: NOW + 601 }), null);
  });

  it('rejects garbage without throwing', async () => {
    const value = await sealFlow(SECRET, binding);
    for (const bad of [
      null,
      undefined,
      '',
      'v1',
      'v1.google',
      'v1.google.' + STATE_ID,
      value + '.extra',
      'v2' + value.slice(2),
      'x'.repeat(5000),
      42,
      { value },
    ]) {
      assert.equal(await openFlow(SECRET, bad, expected), null, String(bad));
    }
  });

  it('refuses to seal with a short secret or a malformed binding', async () => {
    await assert.rejects(sealFlow('short', binding));
    await assert.rejects(sealFlow(SECRET, { ...binding, stateId: 'bad' }));
    await assert.rejects(sealFlow(SECRET, { ...binding, provider: 'Goo gle' }));
    await assert.rejects(sealFlow(SECRET, { ...binding, userId: 'not a user id' }));
    await assert.rejects(sealFlow(SECRET, { ...binding, sessionKey: '' }));
    await assert.rejects(sealFlow(SECRET, { ...binding, expiresAt: Number.NaN }));
    await assert.rejects(sealFlow(SECRET, { ...binding, expiresAt: 1.5 }));
  });

  it('is HttpOnly, Lax, scoped to the callback route and short-lived', () => {
    const options = flowCookieOptions({ secure: true, maxAge: 600 });

    assert.equal(options.name, FLOW_COOKIE);
    assert.equal(options.httpOnly, true);
    // Lax, not Strict: the cookie must ride the provider's top-level redirect
    // back to us, while never riding a cross-site POST.
    assert.equal(options.sameSite, 'lax');
    assert.equal(options.secure, true);
    assert.equal(options.path, FLOW_COOKIE_PATH);
    assert.equal(FLOW_COOKIE_PATH, '/api/connections/callback');
    assert.equal(options.maxAge, 600);
    assert.ok(flowCookieOptions({ secure: true, maxAge: 10 ** 9 }).maxAge <= MAX_FLOW_SECONDS);
    assert.ok(MAX_FLOW_SECONDS <= 3600);
    assert.equal(flowCookieOptions({ secure: true, maxAge: -5 }).maxAge, 0);
    assert.equal(flowCookieOptions({ secure: true, maxAge: Number.NaN }).maxAge, 0);

    const cleared = clearedFlowCookie({ secure: false });
    assert.equal(cleared.name, FLOW_COOKIE);
    assert.equal(cleared.value, '');
    assert.equal(cleared.maxAge, 0);
    assert.equal(cleared.path, FLOW_COOKIE_PATH);
    assert.equal(cleared.httpOnly, true);
  });

  it('derives a session key from the credential without ever returning it', async () => {
    const fromToken = await sessionKeyFor({ sessionToken: 'opaque-session-token', accessAssertion: null });
    assert.ok(!fromToken.includes('opaque-session-token'));
    assert.match(fromToken, /^[0-9a-f]{32,}$/);
    assert.equal(
      fromToken,
      await sessionKeyFor({ sessionToken: 'opaque-session-token', accessAssertion: 'ignored' })
    );
    assert.notEqual(
      fromToken,
      await sessionKeyFor({ sessionToken: 'another-token', accessAssertion: null })
    );

    const fromAccess = await sessionKeyFor({ sessionToken: null, accessAssertion: 'eyJ.assertion' });
    assert.ok(!fromAccess.includes('eyJ.assertion'));
    assert.notEqual(fromAccess, fromToken);
    assert.notEqual(
      fromAccess,
      await sessionKeyFor({ sessionToken: 'eyJ.assertion', accessAssertion: null })
    );

    await assert.rejects(sessionKeyFor({ sessionToken: null, accessAssertion: null }));
    await assert.rejects(sessionKeyFor({ sessionToken: '', accessAssertion: '' }));
  });
});
