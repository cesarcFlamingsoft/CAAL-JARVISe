/**
 * Binding an OAuth round trip to the browser session that started it.
 *
 * The backend already binds each authorization `state` to the user who asked
 * for it. This cookie adds the other half: the *browser session* that asked.
 * When the provider sends the browser back, the callback route opens the
 * cookie under the signed-in user's id and session key; if the cookie is
 * missing, was minted for someone else, for another session, for another
 * state, or has expired, the callback is refused before anything is forwarded.
 *
 * The value is `v1.<provider>.<stateId>.<expiresAt>.<signature>` where the
 * signature is an HMAC-SHA256, under a key derived from the internal auth
 * secret for this purpose alone, over the user id, the session key and the
 * three visible fields. The user id and the session key are inputs to the
 * signature, never part of the value. The cookie is HttpOnly, SameSite=Lax
 * (it must ride the provider's top-level redirect back to us), scoped to the
 * callback route's path only, and lives no longer than the state does.
 *
 * Free of Next.js so it can be unit tested; the route reads and sets the
 * cookie header itself.
 */

export const FLOW_COOKIE = 'caal_oauth_flow';
export const FLOW_COOKIE_PATH = '/api/connections/callback';
/** Matches the backend's ceiling on a state's lifetime. */
export const MAX_FLOW_SECONDS = 3600;

const VERSION = 'v1';
const MIN_SECRET_LENGTH = 32;
const MAX_VALUE_LENGTH = 512;
const KEY_DOMAIN = 'caal.bff.oauth_flow.v1';
const USER_ID = /^usr_[0-9a-f]{24}$/;
const PROVIDER = /^[a-z]{2,32}$/;
const STATE_ID = /^[A-Za-z0-9_-]{32}$/;
const SIGNATURE = /^[A-Za-z0-9_-]{43}$/;
/** Printable, no whitespace: it is joined into the signed message by newline. */
const SESSION_KEY = /^[\x21-\x7e]{1,256}$/;
const EXPIRY = /^[0-9]{1,12}$/;

export interface FlowBinding {
  userId: string;
  sessionKey: string;
  provider: string;
  stateId: string;
  expiresAt: number;
}

export interface OpenedFlow {
  provider: string;
  stateId: string;
  expiresAt: number;
}

export interface FlowCookieOptions {
  name: string;
  value?: string;
  httpOnly: true;
  sameSite: 'lax';
  secure: boolean;
  path: string;
  maxAge: number;
}

const encoder = new TextEncoder();

function base64url(bytes: Uint8Array): string {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function hex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

async function hmac(key: Uint8Array, message: string): Promise<Uint8Array> {
  const cryptoKey = await globalThis.crypto.subtle.importKey(
    'raw',
    key as BufferSource,
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );
  const signature = await globalThis.crypto.subtle.sign('HMAC', cryptoKey, encoder.encode(message));
  return new Uint8Array(signature);
}

/** A key for this purpose alone, derived from -- never equal to -- the shared secret. */
async function flowKey(secret: string): Promise<Uint8Array> {
  if (typeof secret !== 'string' || secret.length < MIN_SECRET_LENGTH) {
    throw new Error('Internal auth secret is missing or too short');
  }
  return hmac(encoder.encode(secret), KEY_DOMAIN);
}

function validBinding(binding: FlowBinding): boolean {
  return (
    USER_ID.test(binding.userId) &&
    SESSION_KEY.test(binding.sessionKey) &&
    PROVIDER.test(binding.provider) &&
    STATE_ID.test(binding.stateId) &&
    Number.isInteger(binding.expiresAt) &&
    binding.expiresAt > 0 &&
    EXPIRY.test(String(binding.expiresAt))
  );
}

async function sign(key: Uint8Array, binding: FlowBinding): Promise<string> {
  const message = [
    binding.userId,
    binding.sessionKey,
    binding.provider,
    binding.stateId,
    String(binding.expiresAt),
  ].join('\n');
  return base64url(await hmac(key, message));
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let mismatch = 0;
  for (let index = 0; index < a.length; index++) {
    mismatch |= a.charCodeAt(index) ^ b.charCodeAt(index);
  }
  return mismatch === 0;
}

/**
 * A stable, non-identifying key for the credential this browser session
 * presents: the hash of the standalone session token, else of the Cloudflare
 * Access assertion. The credential itself is never returned or stored.
 */
export async function sessionKeyFor(input: {
  sessionToken: string | null;
  accessAssertion: string | null;
}): Promise<string> {
  let material: string;
  if (input.sessionToken) {
    material = `session\n${input.sessionToken}`;
  } else if (input.accessAssertion) {
    material = `access\n${input.accessAssertion}`;
  } else {
    throw new Error('No session credential to bind the flow to');
  }
  const digest = await globalThis.crypto.subtle.digest('SHA-256', encoder.encode(material));
  return hex(new Uint8Array(digest));
}

/** Seal a binding into a cookie value. Throws rather than seal anything malformed. */
export async function sealFlow(secret: string, binding: FlowBinding): Promise<string> {
  const key = await flowKey(secret);
  if (!validBinding(binding)) {
    throw new Error('Flow binding is malformed');
  }
  const signature = await sign(key, binding);
  return [VERSION, binding.provider, binding.stateId, String(binding.expiresAt), signature].join('.');
}

/**
 * Open a cookie value for the given user and session, at time `now` (Unix
 * seconds). Null for anything that does not verify; never throws on input.
 */
export async function openFlow(
  secret: string,
  value: unknown,
  expected: { userId: string; sessionKey: string; now: number }
): Promise<OpenedFlow | null> {
  if (typeof value !== 'string' || !value || value.length > MAX_VALUE_LENGTH) return null;
  const parts = value.split('.');
  if (parts.length !== 5 || parts[0] !== VERSION) return null;
  const [, provider, stateId, rawExpiry, signature] = parts;
  if (!PROVIDER.test(provider) || !STATE_ID.test(stateId)) return null;
  if (!EXPIRY.test(rawExpiry) || !SIGNATURE.test(signature)) return null;
  const expiresAt = Number(rawExpiry);
  if (!Number.isInteger(expected.now) || expiresAt <= expected.now) return null;
  const binding: FlowBinding = {
    userId: expected.userId,
    sessionKey: expected.sessionKey,
    provider,
    stateId,
    expiresAt,
  };
  if (!validBinding(binding)) return null;
  let key: Uint8Array;
  try {
    key = await flowKey(secret);
  } catch {
    return null;
  }
  const expectedSignature = await sign(key, binding);
  if (!constantTimeEqual(expectedSignature, signature)) return null;
  return { provider, stateId, expiresAt };
}

function clampAge(seconds: number): number {
  if (!Number.isFinite(seconds)) return 0;
  return Math.max(0, Math.min(Math.trunc(seconds), MAX_FLOW_SECONDS));
}

export function flowCookieOptions({
  secure,
  maxAge,
}: {
  secure: boolean;
  maxAge: number;
}): FlowCookieOptions {
  return {
    name: FLOW_COOKIE,
    httpOnly: true,
    sameSite: 'lax',
    secure,
    path: FLOW_COOKIE_PATH,
    maxAge: clampAge(maxAge),
  };
}

/** An immediately-expiring, empty cookie: what every callback answer sets. */
export function clearedFlowCookie({ secure }: { secure: boolean }): FlowCookieOptions & {
  value: string;
} {
  return { ...flowCookieOptions({ secure, maxAge: 0 }), value: '' };
}
