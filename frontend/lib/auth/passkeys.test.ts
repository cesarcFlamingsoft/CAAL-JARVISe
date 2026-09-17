import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { creationOptionsFromJson, credentialToJson, requestOptionsFromJson } from './passkeys.ts';

describe('native WebAuthn browser conversion', () => {
  it('decodes server challenges, user ids, and excluded credential ids', () => {
    const options = creationOptionsFromJson({
      challenge: 'AQID',
      rp: { id: 'friday.example.com', name: 'FRIDAY' },
      user: { id: 'dXNyX2FiYw', name: 'usr_abc', displayName: 'A' },
      pubKeyCredParams: [{ type: 'public-key', alg: -7 }],
      excludeCredentials: [{ type: 'public-key', id: 'Y3JlZA', transports: ['internal'] }],
    });
    assert.deepEqual([...new Uint8Array(options.challenge)], [1, 2, 3]);
    assert.equal(new TextDecoder().decode(options.user.id), 'usr_abc');
    assert.equal(new TextDecoder().decode(options.excludeCredentials![0].id), 'cred');
  });

  it('keeps authentication discoverable when allowCredentials is absent', () => {
    const options = requestOptionsFromJson({
      challenge: 'AQID',
      rpId: 'friday.example.com',
      userVerification: 'required',
    });
    assert.equal(options.allowCredentials, undefined);
    assert.equal(options.userVerification, 'required');
  });

  it('serializes assertion bytes without exposing browser objects', () => {
    const response = {
      clientDataJSON: Uint8Array.from([1, 2]).buffer,
      authenticatorData: Uint8Array.from([3]).buffer,
      signature: Uint8Array.from([4]).buffer,
      userHandle: Uint8Array.from([5]).buffer,
    } as AuthenticatorAssertionResponse;
    const credential = {
      id: 'credential-id',
      rawId: Uint8Array.from([6]).buffer,
      type: 'public-key',
      authenticatorAttachment: 'platform',
      response,
      getClientExtensionResults: () => ({}),
    } as PublicKeyCredential;

    assert.deepEqual(credentialToJson(credential), {
      id: 'credential-id',
      rawId: 'Bg',
      type: 'public-key',
      authenticatorAttachment: 'platform',
      response: {
        clientDataJSON: 'AQI',
        authenticatorData: 'Aw',
        signature: 'BA',
        userHandle: 'BQ',
      },
    });
  });
});
