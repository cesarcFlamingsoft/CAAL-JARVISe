/**
 * Cloudflare Access application-token verification for the BFF.
 *
 * Cloudflare Access fronts the public portal and attaches a signed JWT to every
 * request it lets through (`Cf-Access-Jwt-Assertion`). Only that signature is
 * identity. The plain `Cf-Access-Authenticated-User-Email` header is never
 * consulted: anyone who can reach this server directly could set it.
 *
 * Verification pins the algorithm, the issuer (the team domain), the
 * application audience tag and the lifetime, and requires a verified `email`
 * claim, which service tokens and login "meta" tokens do not carry. Keys come
 * from the team's JWKS endpoint with caching and rotation handled by `jose`;
 * tests inject a local key set.
 */
import { createRemoteJWKSet, jwtVerify } from 'jose';
import type { JWTVerifyGetKey } from 'jose';

export const ACCESS_ASSERTION_HEADER = 'cf-access-jwt-assertion';
export const MAX_TOKEN_LENGTH = 8192;
export const MAX_EMAIL_LENGTH = 254;
const CLOCK_TOLERANCE_SECONDS = 30;
const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export class AccessError extends Error {
  readonly code: string;

  constructor(code: string, message?: string) {
    super(message ?? code);
    this.name = 'AccessError';
    this.code = code;
  }
}

export interface AccessIdentity {
  readonly email: string;
  readonly subject: string;
  readonly issuedAt: number;
  readonly expiresAt: number;
}

export interface VerifyOptions {
  readonly teamDomain: string;
  readonly audience: string;
  readonly jwks: JWTVerifyGetKey;
  /** Unix seconds; defaults to the wall clock. */
  readonly now?: number;
}

/** Lower-cased, trimmed email or a thrown `AccessError`. */
export function normalizeEmail(raw: unknown): string {
  if (typeof raw !== 'string') {
    throw new AccessError('email_missing', 'email must be text');
  }
  const value = raw.trim().toLowerCase();
  if (!value || value.length > MAX_EMAIL_LENGTH || /[\p{Cc}\p{Cf}]/u.test(value)) {
    throw new AccessError('email_invalid', 'email is empty, too long, or has control characters');
  }
  if (!EMAIL.test(value) || value.split('@').length !== 2) {
    throw new AccessError('email_invalid', 'email is not well formed');
  }
  return value;
}

/**
 * The team's key set with `jose`'s built-in cache: keys are refreshed on an
 * unknown key id (rotation) but never more than once per cooldown, and a
 * cached key set is reused for up to an hour.
 */
export function createAccessKeySet(teamDomain: string): JWTVerifyGetKey {
  return createRemoteJWKSet(new URL(`${teamDomain}/cdn-cgi/access/certs`), {
    cooldownDuration: 30_000,
    cacheMaxAge: 3_600_000,
    timeoutDuration: 5_000,
  });
}

/** Verify an Access assertion or throw `AccessError`. Never logs the token. */
export async function verifyAccessAssertion(
  token: unknown,
  options: VerifyOptions
): Promise<AccessIdentity> {
  if (typeof token !== 'string' || !token || token.length > MAX_TOKEN_LENGTH) {
    throw new AccessError('token_missing', 'Access assertion is missing or malformed');
  }
  let payload;
  try {
    ({ payload } = await jwtVerify(token, options.jwks, {
      algorithms: ['RS256'],
      issuer: options.teamDomain,
      audience: options.audience,
      clockTolerance: CLOCK_TOLERANCE_SECONDS,
      requiredClaims: ['exp', 'iat', 'sub'],
      ...(options.now === undefined ? {} : { currentDate: new Date(options.now * 1000) }),
    }));
  } catch (error) {
    const code = (error as { code?: string })?.code ?? 'token_invalid';
    throw new AccessError(code, 'Access assertion failed verification');
  }
  if (payload.type !== undefined && payload.type !== 'app') {
    throw new AccessError('token_type', 'Access assertion is not an application token');
  }
  if (typeof payload.sub !== 'string' || !payload.sub) {
    throw new AccessError('subject_missing', 'Access assertion has no subject');
  }
  return {
    email: normalizeEmail(payload.email),
    subject: payload.sub,
    issuedAt: Number(payload.iat),
    expiresAt: Number(payload.exp),
  };
}
