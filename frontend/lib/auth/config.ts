/**
 * Server-only configuration for multi-user identity in the BFF.
 *
 * Mirrors the backend's `caal.security_config`: identity is *enabled* only when
 * every value validates. Anything less is reported by variable name, never by
 * value, and every identity-aware route fails closed.
 *
 * Sign-in is standalone by default -- a password checked by the CAAL backend
 * against its own Argon2id hashes, with an opaque server-side session this BFF
 * keeps in an HttpOnly cookie. Cloudflare Access is an *optional additional*
 * provider: set both `CF_ACCESS_TEAM_DOMAIN` and `CF_ACCESS_AUD` to enable it,
 * leave both unset to run without it. Setting one without the other is an
 * error rather than a silent downgrade.
 *
 * Nothing here reads `process.env` implicitly; callers pass the environment in,
 * which is what keeps it unit-testable.
 */

export const ENV_TEAM_DOMAIN = 'CF_ACCESS_TEAM_DOMAIN';
export const ENV_AUDIENCE = 'CF_ACCESS_AUD';
export const ENV_INTERNAL_SECRET = 'CAAL_INTERNAL_AUTH_SECRET';
export const ENV_API_URL = 'CAAL_IDENTITY_API_URL';
export const ENV_PUBLIC_ORIGIN = 'CAAL_PUBLIC_ORIGIN';
export const ENV_REQUIRE_IDENTITY = 'CAAL_REQUIRE_IDENTITY_FOR_SESSIONS';
export const ENV_PASSWORD_LOGIN = 'CAAL_PASSWORD_LOGIN';
export const ENV_ALLOW_INSECURE_COOKIES = 'CAAL_ALLOW_INSECURE_COOKIES';

export const MIN_SECRET_LENGTH = 32;

const TEAM_DOMAIN = /^https:\/\/[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.cloudflareaccess\.com$/;
const AUDIENCE = /^[0-9a-f]{64}$/;
const TRUE = ['true', '1', 'yes', 'on'];
const FALSE = ['false', '0', 'no', 'off'];

export interface IdentityConfig {
  /** Null when Cloudflare Access is not configured, which is the default. */
  readonly teamDomain: string | null;
  readonly audience: string | null;
  readonly internalAuthSecret: string;
  readonly apiBaseUrl: string;
  readonly publicOrigin: string | null;
  readonly requireIdentityForSessions: boolean;
  /** Whether local password sign-in is offered. */
  readonly passwordLogin: boolean;
  /**
   * Whether a session cookie may be issued over plain HTTP. Off by default:
   * without it, a non-TLS request is refused a session rather than handed one
   * that travels in the clear.
   */
  readonly allowInsecureCookies: boolean;
  readonly accessEnabled: boolean;
  /** Operator-facing summary that never includes a secret. */
  describe(): string;
}

export type IdentityStatus =
  | { status: 'enabled'; config: IdentityConfig; problems: string[] }
  | { status: 'disabled' | 'invalid'; config?: undefined; problems: string[] };

type Env = Record<string, string | undefined>;

function present(env: Env, name: string): string | null {
  const value = env[name];
  if (value === undefined || value === '') {
    return null;
  }
  return value;
}

function normalizeOrigin(raw: string): string | null {
  try {
    const url = new URL(raw.trim());
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      return null;
    }
    return url.origin;
  } catch {
    return null;
  }
}

/** Read a boolean flag; returns `null` when the value is not a boolean at all. */
function flag(env: Env, name: string, fallback: boolean): boolean | null {
  const raw = present(env, name);
  if (raw === null) return fallback;
  const value = raw.trim().toLowerCase();
  if (TRUE.includes(value)) return true;
  if (FALSE.includes(value)) return false;
  return null;
}

/** Validate the identity configuration. Never logs or returns secret values. */
export function readIdentityConfig(env: Env = process.env): IdentityStatus {
  const problems: string[] = [];
  let attempted = false;

  const secret = present(env, ENV_INTERNAL_SECRET);
  if (secret === null) {
    problems.push(`${ENV_INTERNAL_SECRET} (missing)`);
  } else {
    attempted = true;
    if (secret.trim().length < MIN_SECRET_LENGTH || secret !== secret.trim()) {
      problems.push(
        `${ENV_INTERNAL_SECRET} (invalid: at least ${MIN_SECRET_LENGTH} characters, no surrounding whitespace)`
      );
    }
  }

  const rawApi =
    present(env, ENV_API_URL) ?? present(env, 'CAAL_DEVICE_API_URL') ?? present(env, 'WEBHOOK_URL');
  let apiBaseUrl: string | null = null;
  if (rawApi === null) {
    problems.push(`${ENV_API_URL} (missing)`);
  } else {
    attempted = true;
    apiBaseUrl = normalizeOrigin(rawApi);
    if (apiBaseUrl === null) {
      problems.push(`${ENV_API_URL} (invalid: expected an http(s) URL)`);
    } else {
      // Keep any path prefix the operator configured, minus a trailing slash.
      apiBaseUrl = rawApi.trim().replace(/\/+$/, '');
    }
  }

  // Cloudflare Access: optional, but both halves or neither. Half a pair is an
  // error rather than a silent downgrade to "no provider".
  const teamDomain = present(env, ENV_TEAM_DOMAIN);
  const audience = present(env, ENV_AUDIENCE);
  const hasAccess = teamDomain !== null && audience !== null;
  if (teamDomain !== null || audience !== null) {
    attempted = true;
    if (teamDomain === null) {
      problems.push(`${ENV_TEAM_DOMAIN} (missing: required alongside ${ENV_AUDIENCE})`);
    } else if (!TEAM_DOMAIN.test(teamDomain.trim())) {
      problems.push(`${ENV_TEAM_DOMAIN} (invalid: expected https://<team>.cloudflareaccess.com)`);
    }
    if (audience === null) {
      problems.push(`${ENV_AUDIENCE} (missing: required alongside ${ENV_TEAM_DOMAIN})`);
    } else if (!AUDIENCE.test(audience.trim())) {
      problems.push(`${ENV_AUDIENCE} (invalid: expected the 64-character hex application tag)`);
    }
  }

  const rawOrigin = present(env, ENV_PUBLIC_ORIGIN);
  let publicOrigin: string | null = null;
  if (rawOrigin !== null) {
    publicOrigin = normalizeOrigin(rawOrigin);
    if (publicOrigin === null) {
      problems.push(`${ENV_PUBLIC_ORIGIN} (invalid: expected an http(s) origin)`);
    }
  }

  const requireIdentity = flag(env, ENV_REQUIRE_IDENTITY, false);
  if (requireIdentity === null) {
    problems.push(`${ENV_REQUIRE_IDENTITY} (invalid: expected true or false)`);
  }
  const passwordLogin = flag(env, ENV_PASSWORD_LOGIN, true);
  if (passwordLogin === null) {
    problems.push(`${ENV_PASSWORD_LOGIN} (invalid: expected true or false)`);
  }
  const allowInsecureCookies = flag(env, ENV_ALLOW_INSECURE_COOKIES, false);
  if (allowInsecureCookies === null) {
    problems.push(`${ENV_ALLOW_INSECURE_COOKIES} (invalid: expected true or false)`);
  }

  // With no password sign-in and no identity provider there is no way in at
  // all; refuse rather than serve a portal nobody can enter.
  if (passwordLogin === false && !hasAccess) {
    problems.push(
      `${ENV_PASSWORD_LOGIN} (invalid: cannot be false with no identity provider configured: that leaves no way to sign in)`
    );
  }

  if (problems.length > 0) {
    return { status: attempted ? 'invalid' : 'disabled', problems };
  }

  const config: IdentityConfig = {
    teamDomain: teamDomain === null ? null : teamDomain.trim(),
    audience: audience === null ? null : audience.trim(),
    internalAuthSecret: secret!,
    apiBaseUrl: apiBaseUrl!,
    publicOrigin,
    requireIdentityForSessions: requireIdentity!,
    passwordLogin: passwordLogin!,
    allowInsecureCookies: allowInsecureCookies!,
    accessEnabled: hasAccess,
    describe() {
      const providers: string[] = [];
      if (this.passwordLogin) providers.push('local password sign-in');
      if (this.accessEnabled) providers.push(`Cloudflare Access team ${this.teamDomain}`);
      return (
        `Multi-user identity enabled (${providers.join(' and ')}), backend ${this.apiBaseUrl}` +
        (this.requireIdentityForSessions ? ', anonymous voice sessions refused' : '') +
        (this.allowInsecureCookies ? ', INSECURE cookies permitted over plain HTTP' : '')
      );
    },
  };
  return { status: 'enabled', config, problems: [] };
}

/**
 * Reduce an email to a recognizable hint for the browser: first character of
 * the local part plus the domain. Raw emails never leave the server.
 */
export function maskEmail(value: unknown): string {
  if (typeof value !== 'string') {
    return '•••••';
  }
  const at = value.indexOf('@');
  if (at < 1 || at !== value.lastIndexOf('@') || at === value.length - 1) {
    return '•••••';
  }
  return `${value[0]}•••••@${value.slice(at + 1)}`;
}
