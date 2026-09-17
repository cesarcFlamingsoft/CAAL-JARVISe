/** Native WebAuthn JSON/binary conversion. No camera or biometric data is accessed. */

type JsonRecord = Record<string, unknown>;

function decode(value: unknown): ArrayBuffer {
  if (typeof value !== 'string' || !value || value.length > 131_072) {
    throw new TypeError('Invalid WebAuthn binary value');
  }
  const normalized = value.replace(/-/g, '+').replace(/_/g, '/');
  const binary = atob(normalized + '='.repeat((4 - (normalized.length % 4)) % 4));
  return Uint8Array.from(binary, (character) => character.charCodeAt(0)).buffer;
}

function encode(value: ArrayBuffer): string {
  const binary = Array.from(new Uint8Array(value), (byte) => String.fromCharCode(byte)).join('');
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function creationOptionsFromJson(input: JsonRecord): PublicKeyCredentialCreationOptions {
  const options = structuredClone(input) as unknown as PublicKeyCredentialCreationOptions & {
    challenge: unknown;
    user: PublicKeyCredentialUserEntity & { id: unknown };
    excludeCredentials?: Array<PublicKeyCredentialDescriptor & { id: unknown }>;
  };
  options.challenge = decode(input.challenge);
  options.user.id = decode((input.user as JsonRecord)?.id);
  if (Array.isArray(options.excludeCredentials)) {
    options.excludeCredentials = options.excludeCredentials.map((descriptor) => ({
      ...descriptor,
      id: decode(descriptor.id),
    }));
  }
  return options;
}

export function requestOptionsFromJson(input: JsonRecord): PublicKeyCredentialRequestOptions {
  const options = structuredClone(input) as unknown as PublicKeyCredentialRequestOptions & {
    challenge: unknown;
    allowCredentials?: Array<PublicKeyCredentialDescriptor & { id: unknown }>;
  };
  options.challenge = decode(input.challenge);
  if (Array.isArray(options.allowCredentials)) {
    options.allowCredentials = options.allowCredentials.map((descriptor) => ({
      ...descriptor,
      id: decode(descriptor.id),
    }));
  }
  return options;
}

export function credentialToJson(credential: PublicKeyCredential): JsonRecord {
  const response = credential.response;
  const common = { clientDataJSON: encode(response.clientDataJSON) };
  let serialized: JsonRecord;
  if ('attestationObject' in response) {
    const registration = response as AuthenticatorAttestationResponse;
    serialized = {
      ...common,
      attestationObject: encode(registration.attestationObject),
      transports:
        typeof registration.getTransports === 'function' ? registration.getTransports() : [],
    };
  } else {
    const assertion = response as AuthenticatorAssertionResponse;
    serialized = {
      ...common,
      authenticatorData: encode(assertion.authenticatorData),
      signature: encode(assertion.signature),
      ...(assertion.userHandle ? { userHandle: encode(assertion.userHandle) } : {}),
    };
  }
  return {
    id: credential.id,
    rawId: encode(credential.rawId),
    type: 'public-key',
    ...(credential.authenticatorAttachment
      ? { authenticatorAttachment: credential.authenticatorAttachment }
      : {}),
    response: serialized,
  };
}
