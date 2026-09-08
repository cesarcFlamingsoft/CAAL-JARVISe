/**
 * Short-lived signed principals from the BFF to the CAAL backend and agent.
 *
 * Wire format is the one `caal.internal_auth` verifies: an HS256 JWT with a
 * fixed issuer, a single audience, an opaque subject, `iat`/`nbf`/`exp`, and a
 * random `jti`. HTTP principals are single use on the backend; agent
 * principals are bound to the LiveKit room they were minted for.
 *
 * Only ever called server-side. The secret never reaches a browser.
 */
import { SignJWT } from 'jose';

export const ISSUER_BFF = 'caal-bff';
export const AUDIENCE_BACKEND = 'caal-backend';
export const AUDIENCE_AGENT = 'caal-agent';
export const AUDIENCE_IDENTITY = 'caal-identity';

export const DEFAULT_TTL_SECONDS = 60;
export const MAX_TTL_SECONDS = 300;
export const MIN_SECRET_LENGTH = 32;
const MAX_SUBJECT_LENGTH = 254;
const MAX_AUDIENCE_LENGTH = 64;
const RESERVED = new Set(['iss', 'aud', 'sub', 'iat', 'nbf', 'exp', 'jti']);

export class PrincipalError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PrincipalError';
  }
}

export interface MintOptions {
  readonly secret: string;
  readonly subject: string;
  readonly audience: string;
  readonly ttlSeconds?: number;
  readonly claims?: Record<string, unknown>;
  /** Unix seconds; defaults to the wall clock. */
  readonly now?: number;
}

function base64url(bytes: Uint8Array): string {
  let binary = '';
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/** A random URL-safe token; 16 bytes yields 22 characters. */
export function randomToken(bytes = 16): string {
  const buffer = new Uint8Array(bytes);
  globalThis.crypto.getRandomValues(buffer);
  return base64url(buffer);
}

function requireText(value: unknown, name: string, limit: number): string {
  if (typeof value !== 'string' || !value || value.length > limit || /[\p{Cc}]/u.test(value)) {
    throw new PrincipalError(`${name} is missing or malformed`);
  }
  return value;
}

/** Sign a principal. Throws `PrincipalError` for anything that must not be minted. */
export async function mintPrincipal(options: MintOptions): Promise<string> {
  const { secret } = options;
  if (typeof secret !== 'string' || secret.length < MIN_SECRET_LENGTH) {
    throw new PrincipalError('Internal auth secret is missing or too short');
  }
  const subject = requireText(options.subject, 'subject', MAX_SUBJECT_LENGTH);
  const audience = requireText(options.audience, 'audience', MAX_AUDIENCE_LENGTH);
  const ttl = Math.trunc(options.ttlSeconds ?? DEFAULT_TTL_SECONDS);
  if (!Number.isFinite(ttl) || ttl < 1 || ttl > MAX_TTL_SECONDS) {
    throw new PrincipalError(`ttl must be between 1 and ${MAX_TTL_SECONDS} seconds`);
  }
  const extra: Record<string, unknown> = {};
  for (const [name, value] of Object.entries(options.claims ?? {})) {
    if (RESERVED.has(name)) {
      throw new PrincipalError('claims may not override reserved names');
    }
    extra[name] = value;
  }
  const issued = Math.trunc(options.now ?? Date.now() / 1000);

  return new SignJWT({ ...extra })
    .setProtectedHeader({ alg: 'HS256', typ: 'JWT' })
    .setIssuer(ISSUER_BFF)
    .setAudience(audience)
    .setSubject(subject)
    .setIssuedAt(issued)
    .setNotBefore(issued)
    .setExpirationTime(issued + ttl)
    .setJti(randomToken(16))
    .sign(new TextEncoder().encode(secret));
}
