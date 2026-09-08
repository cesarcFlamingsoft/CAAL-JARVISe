/**
 * Who is making this request.
 *
 * The primary answer comes from the standalone path: an opaque session token
 * in an HttpOnly cookie, re-resolved against the CAAL backend's database on
 * every request. Nothing about the user is trusted from the cookie itself --
 * role, status, and the forced-password-change flag are whatever the database
 * says right now, so suspending or resetting someone takes effect on their
 * very next request.
 *
 * Cloudflare Access is an optional *additional* provider. When it is
 * configured and no session cookie resolves, a verified Access assertion is
 * accepted instead. When it is not configured, that path does not exist and
 * nothing about local sign-in is weakened.
 *
 * The email never leaves the server side of this boundary.
 */
import type { JWTVerifyGetKey } from 'jose';
import {
  ACCESS_ASSERTION_HEADER,
  type AccessIdentity,
  createAccessKeySet,
  verifyAccessAssertion,
} from './access';
import {
  backendDetail,
  backendString,
  clientKeyFor,
  readSession,
  resolveIdentity,
} from './backend';
import { type IdentityConfig, type IdentityStatus, readIdentityConfig } from './config';
import { readCookie } from './csrf';
import { SESSION_COOKIE } from './session-cookie';

export type Role = 'admin' | 'member';
export type Status = 'active' | 'suspended';

export interface SessionUser {
  readonly userId: string;
  readonly role: Role;
  readonly status: Status;
  readonly displayName: string;
}

export type AuthResult =
  | { kind: 'unconfigured'; status: IdentityStatus }
  | { kind: 'anonymous'; config: IdentityConfig }
  /** A session cookie was presented but is expired, revoked, or unknown. */
  | { kind: 'expired'; config: IdentityConfig }
  | { kind: 'invalid'; config: IdentityConfig }
  | { kind: 'denied'; config: IdentityConfig; reason: 'no_account' | 'suspended' | 'unavailable' }
  | {
      kind: 'user';
      config: IdentityConfig;
      user: SessionUser;
      via: 'password' | 'access';
      mustChangePassword: boolean;
      sessionToken: string | null;
      identity?: AccessIdentity;
    };

const keySets = new Map<string, JWTVerifyGetKey>();

function keySetFor(teamDomain: string): JWTVerifyGetKey {
  let keys = keySets.get(teamDomain);
  if (!keys) {
    keys = createAccessKeySet(teamDomain);
    keySets.set(teamDomain, keys);
  }
  return keys;
}

const USER_ID = /^usr_[0-9a-f]{24}$/;

function parseUser(data: unknown): SessionUser | null {
  const userId = backendString(data, 'user_id');
  const role = backendString(data, 'role');
  const status = backendString(data, 'status');
  const displayName = backendString(data, 'display_name') ?? '';
  if (!userId || !USER_ID.test(userId)) return null;
  if (role !== 'admin' && role !== 'member') return null;
  if (status !== 'active' && status !== 'suspended') return null;
  return { userId, role, status, displayName };
}

/** Authenticate a request from its headers. Never throws; never logs identities. */
export async function authenticate(
  headers: Headers,
  env: Record<string, string | undefined> = process.env
): Promise<AuthResult> {
  const status = readIdentityConfig(env);
  if (status.status !== 'enabled') {
    return { kind: 'unconfigured', status };
  }
  const config = status.config;

  // --- 1. the standalone session cookie ----------------------------------------
  const sessionToken = readCookie(headers.get('cookie'), SESSION_COOKIE);
  let sessionFailed = false;
  if (sessionToken && config.passwordLogin) {
    const clientKey = await clientKeyFor(headers);
    const resolved = await readSession(config, sessionToken, clientKey);
    if (resolved.ok) {
      const user = parseUser(resolved.data);
      if (user && user.status === 'active') {
        const mustChange =
          (resolved.data as { must_change_password?: unknown } | null)?.must_change_password ===
          true;
        return {
          kind: 'user',
          config,
          user,
          via: 'password',
          mustChangePassword: mustChange,
          sessionToken,
        };
      }
      return { kind: 'denied', config, reason: 'unavailable' };
    }
    if (resolved.status === null) {
      // The backend did not answer at all. Do not silently treat a live
      // session as signed out; say so, so the caller can retry rather than
      // clear a valid cookie.
      return { kind: 'denied', config, reason: 'unavailable' };
    }
    sessionFailed = true;
  }

  // --- 2. Cloudflare Access, only if the operator configured it -----------------
  const assertion = config.accessEnabled ? headers.get(ACCESS_ASSERTION_HEADER) : null;
  if (assertion && config.teamDomain && config.audience) {
    let identity: AccessIdentity;
    try {
      identity = await verifyAccessAssertion(assertion, {
        teamDomain: config.teamDomain,
        audience: config.audience,
        jwks: keySetFor(config.teamDomain),
      });
    } catch {
      return { kind: 'invalid', config };
    }

    const resolved = await resolveIdentity(config, identity.email, assertion);
    if (resolved.ok) {
      const user = parseUser(resolved.data);
      if (user && user.status === 'active') {
        return {
          kind: 'user',
          config,
          user,
          via: 'access',
          mustChangePassword: false,
          sessionToken: null,
          identity,
        };
      }
      return { kind: 'denied', config, reason: 'unavailable' };
    }
    if (resolved.status === 403) {
      const detail = backendDetail(resolved.data);
      if (detail === 'no_account' || detail === 'suspended') {
        return { kind: 'denied', config, reason: detail };
      }
    }
    if (resolved.status === 401) {
      return { kind: 'invalid', config };
    }
    return { kind: 'denied', config, reason: 'unavailable' };
  }

  // A cookie was presented and the backend rejected it: the caller is signed
  // out and the stale cookie should be cleared.
  if (sessionFailed) {
    return { kind: 'expired', config };
  }
  return { kind: 'anonymous', config };
}
