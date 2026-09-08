/**
 * The browser-facing contract for a user's own provider connections.
 *
 * Everything the BFF routes and the Settings panel need to agree on lives
 * here, free of Next.js and React so it can be unit tested: the provider
 * catalogue, the shapes a browser may see, how a provider's redirect back to
 * us is parsed, and the bounded set of outcomes the callback can end in.
 *
 * Two rules run through it. Nothing here forwards a field it does not
 * recognise, so a token, a raw user id or a provider's free-text error can
 * never reach a page by accident. And nothing here claims more than the
 * backend did: an authorization the backend could not turn into tokens is
 * reported as exactly that, never as "connected".
 */
import type { AuthResult } from '../auth/session';

export const PROVIDERS = ['google', 'microsoft', 'zoho'] as const;
export type Provider = (typeof PROVIDERS)[number];

/** Must match `caal.oauth_providers.OAUTH_CALLBACK_PATH`; every redirect_uri is built from it. */
export const OAUTH_CALLBACK_PATH = '/api/connections/callback';

export const MAX_CODE_LENGTH = 4096;
const MAX_MISSING_NAMES = 12;
const MAX_LABEL_LENGTH = 254;
const MAX_SCOPES = 50;
const MAX_CONNECTIONS = 100;

/**
 * The backend's bounded reasons for a failed exchange, and the outcome each
 * one is shown as. Anything else is a plain `exchange_failed`.
 */
const EXCHANGE_REASON_OUTCOMES: Record<string, Outcome> = {
  insufficient_scope: 'unsupported_scope',
  datacenter_mismatch: 'datacenter_mismatch',
  identity_unavailable: 'identity_unavailable',
};

const PROVIDER_LABELS: Record<Provider, string> = {
  google: 'Google',
  microsoft: 'Microsoft / Outlook',
  zoho: 'Zoho',
};

/** Where each provider's authorization endpoint may live; null means any https host. */
const AUTHORIZATION_HOSTS: Record<Provider, RegExp | null> = {
  google: /^accounts\.google\.com$/,
  microsoft: /^login\.microsoftonline\.com$/,
  // The Zoho accounts host is operator-configured (accounts.zoho.com, .eu, .in, ...).
  zoho: null,
};

const CONNECTION_ID = /^con_[0-9a-f]{24}$/;
const STATE = /^([A-Za-z0-9_-]{32})\.([A-Za-z0-9_-]{43})$/;
const CODE = /^[\x21-\x7e]+$/;
const ENV_NAME = /^[A-Z][A-Z0-9_]{0,63}$/;
const SCOPE = /^[A-Za-z0-9_.:/\-]{1,256}$/;
const SHORT_CODE = /^[a-z_]{1,40}$/;
/** Zoho's data-center code on its redirect (`us`, `eu`, `in`, ...). */
const ZOHO_LOCATION = /^[a-z]{2}$/;

// --- validators ------------------------------------------------------------------

export function isProvider(value: unknown): value is Provider {
  return typeof value === 'string' && (PROVIDERS as readonly string[]).includes(value);
}

export function providerLabel(provider: Provider | null): string {
  return provider ? PROVIDER_LABELS[provider] : 'the provider';
}

export function isConnectionId(value: unknown): value is string {
  return typeof value === 'string' && CONNECTION_ID.test(value);
}

/** Split a state into its opaque id and signature; null for any other shape. */
export function parseState(value: unknown): { state: string; stateId: string } | null {
  if (typeof value !== 'string') return null;
  const match = STATE.exec(value);
  return match ? { state: value, stateId: match[1] } : null;
}

export function isAuthorizationCode(value: unknown): value is string {
  return typeof value === 'string' && value.length <= MAX_CODE_LENGTH && CODE.test(value);
}

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function unixSeconds(value: unknown): number | null {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : null;
}

function shortCode(value: unknown): string | null {
  return typeof value === 'string' && SHORT_CODE.test(value) ? value : null;
}

// --- browser shapes ----------------------------------------------------------------

export interface BrowserConnection {
  connectionId: string;
  provider: Provider;
  status: 'connected';
  accountLabel: string | null;
  scopes: string[];
  tokenExpiresAt: number | null;
  hasRefreshToken: boolean;
  connectedAt: number | null;
  updatedAt: number | null;
}

export interface ProviderAvailability {
  provider: Provider;
  displayName: string;
  configured: boolean;
}

export interface BrowserConnectionList {
  connections: BrowserConnection[];
  providers: ProviderAvailability[];
  /**
   * Whether the backend can finish an authorization by exchanging the code
   * for tokens. Reported by the backend itself (`token_exchange_available`
   * on its list answer), never assumed here: a missing or non-boolean claim
   * reads as unavailable, so the panel can only ever understate.
   */
  tokenExchangeAvailable: boolean;
}

/** One live connection in browser shape, or null for anything not exactly that. */
export function browserConnection(data: unknown): BrowserConnection | null {
  const row = record(data);
  if (!row) return null;
  if (!isConnectionId(row.connection_id) || !isProvider(row.provider)) return null;
  if (row.status !== 'connected') return null;
  if (!Array.isArray(row.scopes) || row.scopes.length > MAX_SCOPES) return null;
  const scopes: string[] = [];
  for (const scope of row.scopes) {
    if (typeof scope !== 'string' || !SCOPE.test(scope)) return null;
    scopes.push(scope);
  }
  let accountLabel: string | null = null;
  if (row.account_label !== null && row.account_label !== undefined) {
    if (typeof row.account_label !== 'string') return null;
    const label = row.account_label.trim().replace(/\s+/g, ' ');
    if (/[\p{Cc}\p{Cf}]/u.test(label) || label.length > MAX_LABEL_LENGTH) return null;
    accountLabel = label || null;
  }
  return {
    connectionId: row.connection_id,
    provider: row.provider,
    status: 'connected',
    accountLabel,
    scopes,
    tokenExpiresAt: unixSeconds(row.token_expires_at),
    hasRefreshToken: row.has_refresh_token === true,
    connectedAt: unixSeconds(row.connected_at),
    updatedAt: unixSeconds(row.updated_at),
  };
}

/**
 * The backend's list, reduced to what a browser may see. Malformed or foreign
 * rows are dropped; a payload without both lists is refused outright so a
 * contract drift surfaces as an error rather than as an empty panel.
 */
export function browserConnectionList(data: unknown): BrowserConnectionList | null {
  const body = record(data);
  if (!body || !Array.isArray(body.connections) || !Array.isArray(body.providers)) return null;
  const connections: BrowserConnection[] = [];
  for (const entry of body.connections) {
    const connection = browserConnection(entry);
    if (connection) connections.push(connection);
    if (connections.length === MAX_CONNECTIONS) break;
  }
  const configured = new Set<Provider>();
  for (const entry of body.providers) {
    const row = record(entry);
    if (row && isProvider(row.provider) && row.configured === true) configured.add(row.provider);
  }
  return {
    connections,
    providers: PROVIDERS.map((provider) => ({
      provider,
      displayName: PROVIDER_LABELS[provider],
      configured: configured.has(provider),
    })),
    tokenExchangeAvailable: body.token_exchange_available === true,
  };
}

export interface ConfigurationNeeded {
  provider: Provider;
  missing: string[];
}

/** The backend's explicit "not set up" answer, reduced to variable names. */
export function configurationNeeded(data: unknown): ConfigurationNeeded | null {
  const body = record(data);
  if (!body || body.detail !== 'provider_not_configured') return null;
  if (!isProvider(body.provider) || !Array.isArray(body.missing)) return null;
  const missing: string[] = [];
  for (const name of body.missing) {
    if (typeof name === 'string' && ENV_NAME.test(name)) missing.push(name);
    if (missing.length === MAX_MISSING_NAMES) break;
  }
  return { provider: body.provider, missing };
}

export interface Authorization {
  authorizationUrl: string;
  stateId: string;
  expiresAt: number;
}

/**
 * The backend's authorize answer, accepted only when it points at the
 * provider's own https endpoint and carries a well-formed state.
 */
export function parseAuthorization(provider: Provider, data: unknown): Authorization | null {
  const body = record(data);
  if (!body || body.provider !== provider) return null;
  if (typeof body.authorization_url !== 'string' || body.authorization_url.length > 4096) return null;
  let url: URL;
  try {
    url = new URL(body.authorization_url);
  } catch {
    return null;
  }
  if (url.protocol !== 'https:' || !url.hostname) return null;
  const host = AUTHORIZATION_HOSTS[provider];
  if (host && !host.test(url.hostname)) return null;
  const states = url.searchParams.getAll('state');
  const state = states.length === 1 ? parseState(states[0]) : null;
  if (!state) return null;
  const expiresAt = unixSeconds(body.expires_at);
  if (expiresAt === null) return null;
  return { authorizationUrl: body.authorization_url, stateId: state.stateId, expiresAt };
}

// --- callback ------------------------------------------------------------------------

export type CallbackQuery =
  /**
   * A code to redeem. `location` is Zoho's data-center code when the redirect
   * carried a well-formed one; the backend checks it against the accounts
   * server it is configured for and fails closed on a mismatch.
   */
  | { kind: 'code'; state: string; stateId: string; code: string; location: string | null }
  /** The user declined at the provider. */
  | { kind: 'denied' }
  /** The provider rejected the scopes this server asked for. */
  | { kind: 'unsupported_scope' }
  /** The provider reported some other error; its wording is never read. */
  | { kind: 'provider_error' }
  | { kind: 'invalid' };

function single(params: URLSearchParams, name: string): string | null {
  const values = params.getAll(name);
  return values.length === 1 ? values[0] : null;
}

/** Parse the provider's redirect back to us. Extra parameters are ignored. */
export function parseCallbackQuery(params: URLSearchParams): CallbackQuery {
  if (params.has('error')) {
    const error = single(params, 'error');
    if (error === 'access_denied') return { kind: 'denied' };
    if (error === 'invalid_scope') return { kind: 'unsupported_scope' };
    return { kind: 'provider_error' };
  }
  if (params.has('state') && params.getAll('state').length !== 1) return { kind: 'invalid' };
  if (params.has('code') && params.getAll('code').length !== 1) return { kind: 'invalid' };
  const state = parseState(single(params, 'state'));
  const code = single(params, 'code');
  if (!state || !isAuthorizationCode(code)) return { kind: 'invalid' };
  const rawLocation = single(params, 'location');
  const location = rawLocation !== null && ZOHO_LOCATION.test(rawLocation) ? rawLocation : null;
  return { kind: 'code', state: state.state, stateId: state.stateId, code, location };
}

export const OUTCOMES = [
  'connected',
  'token_exchange_unavailable',
  'denied',
  'provider_error',
  'invalid_callback',
  'not_initiated_here',
  'invalid_state',
  'configuration_needed',
  'exchange_failed',
  'unsupported_scope',
  'datacenter_mismatch',
  'identity_unavailable',
  'not_signed_in',
  'unauthorized',
  'identity_not_configured',
  'rate_limited',
  'backend_unavailable',
] as const;
export type Outcome = (typeof OUTCOMES)[number];

export function isOutcome(value: unknown): value is Outcome {
  return typeof value === 'string' && (OUTCOMES as readonly string[]).includes(value);
}

/** The outcome for a request that is not a usable signed-in user; null when it is. */
export function outcomeFromAuth(auth: AuthResult): Outcome | null {
  switch (auth.kind) {
    case 'user':
      return auth.mustChangePassword ? 'unauthorized' : null;
    case 'unconfigured':
      return 'identity_not_configured';
    case 'denied':
      return 'unauthorized';
    default:
      return 'not_signed_in';
  }
}

/** The outcome of the backend's callback answer. */
export function outcomeFromBackend(status: number | null, data: unknown): Outcome {
  const detail = shortCode(record(data)?.detail);
  switch (status) {
    case 200:
      return 'connected';
    case 400:
      return detail === 'invalid_state' ? 'invalid_state' : 'invalid_callback';
    case 401:
    case 403:
      return 'unauthorized';
    case 422:
      return 'invalid_callback';
    case 429:
      return 'rate_limited';
    case 502: {
      if (detail !== 'token_exchange_failed') return 'exchange_failed';
      const reason = shortCode(record(data)?.reason);
      return (reason && EXCHANGE_REASON_OUTCOMES[reason]) || 'exchange_failed';
    }
    case 503:
      if (detail === 'token_exchange_unavailable') return 'token_exchange_unavailable';
      if (detail === 'provider_not_configured') return 'configuration_needed';
      return 'identity_not_configured';
    default:
      return 'backend_unavailable';
  }
}

export const RESULT_PATH = '/connections/result';

export function resultPath(outcome: Outcome, provider: Provider | null): string {
  const params = new URLSearchParams({ outcome });
  if (provider) params.set('provider', provider);
  return `${RESULT_PATH}?${params.toString()}`;
}

export interface ResultView {
  title: string;
  detail: string;
  connected: boolean;
}

const START_AGAIN = 'Start again from Settings → Integrations → Connected accounts.';

/** Plain words for an outcome. Nothing here is interpolated from a provider or a request. */
export function describeOutcome(outcome: Outcome, provider: Provider | null): ResultView {
  const name = providerLabel(provider);
  const Name = provider ? name : 'The provider';
  switch (outcome) {
    case 'connected':
      return {
        title: `${Name} is connected`,
        detail: `JARVIS can now use the ${name} account you approved. You can review or disconnect it under Settings → Integrations → Connected accounts.`,
        connected: true,
      };
    case 'token_exchange_unavailable':
      return {
        title: 'Authorization received, but not completed',
        detail: `${Name} approved the request, but the JARVIS backend reports that it cannot perform the final token exchange, so the ${name} account is not connected and nothing was stored. The approval was discarded. Ask the operator to configure a provider, then connect again.`,
        connected: false,
      };
    case 'unsupported_scope':
      return {
        title: `${Name} rejected the requested permissions`,
        detail: `${Name} did not accept the set of permissions this JARVIS server asks for, so the account could not be linked. Nothing was connected. Ask the operator to review the ${name} scope setting on the server.`,
        connected: false,
      };
    case 'datacenter_mismatch':
      return {
        title: `Your ${name} account lives in a different data center`,
        detail: `${Name} reported that your account is served from a different data center than the one this JARVIS server is configured for, so the approval could not be redeemed. Nothing was connected. Ask the operator to point the ${name} accounts domain at your data center.`,
        connected: false,
      };
    case 'identity_unavailable':
      return {
        title: `${Name} did not say which account was approved`,
        detail: `${Name} approved the request but did not say which account it was for, and JARVIS will not guess. Nothing was connected. ${START_AGAIN}`,
        connected: false,
      };
    case 'denied':
      return {
        title: 'Access was declined',
        detail: `You declined the request at ${name}. Nothing was connected.`,
        connected: false,
      };
    case 'provider_error':
      return {
        title: `${Name} reported a problem`,
        detail: `${Name} did not complete the authorization. Nothing was connected. ${START_AGAIN}`,
        connected: false,
      };
    case 'invalid_callback':
      return {
        title: 'The return from the provider was not usable',
        detail: `The provider's redirect was missing or malformed, so it was ignored. Nothing was connected. ${START_AGAIN}`,
        connected: false,
      };
    case 'not_initiated_here':
      return {
        title: 'This browser did not start this connection',
        detail: `The connection attempt was not started from this browser session, or it took longer than the allowed time. Nothing was connected. ${START_AGAIN}`,
        connected: false,
      };
    case 'invalid_state':
      return {
        title: 'The authorization expired or was already used',
        detail: `Each connection attempt can be completed once, within a few minutes. Nothing was connected. ${START_AGAIN}`,
        connected: false,
      };
    case 'configuration_needed':
      return {
        title: `${Name} is not set up on this server`,
        detail: `The operator has not configured the ${name} OAuth client on this JARVIS server. Nothing was connected.`,
        connected: false,
      };
    case 'exchange_failed':
      return {
        title: `${Name} refused the token exchange`,
        detail: `The authorization could not be exchanged for access. Nothing was connected. ${START_AGAIN}`,
        connected: false,
      };
    case 'not_signed_in':
      return {
        title: 'You are not signed in',
        detail: 'Sign in to JARVIS first, then start again from Settings → Integrations → Connected accounts.',
        connected: false,
      };
    case 'unauthorized':
      return {
        title: 'Your account cannot do this right now',
        detail: 'Your JARVIS account is not allowed to connect accounts at the moment. Nothing was connected.',
        connected: false,
      };
    case 'identity_not_configured':
      return {
        title: 'Multi-user identity is not configured',
        detail: 'Per-user connected accounts need multi-user identity, which is not configured on this JARVIS server.',
        connected: false,
      };
    case 'rate_limited':
      return {
        title: 'Too many requests',
        detail: `Please wait a moment. Nothing was connected. ${START_AGAIN}`,
        connected: false,
      };
    case 'backend_unavailable':
      return {
        title: 'The backend did not respond',
        detail: `JARVIS could not reach its backend to finish the connection. Nothing was connected. ${START_AGAIN}`,
        connected: false,
      };
  }
}

// --- panel ------------------------------------------------------------------------------

export type ConnectionRowStatus = 'connected' | 'not_connected' | 'not_configured';

export interface ConnectionRow {
  provider: Provider;
  label: string;
  status: ConnectionRowStatus;
  configured: boolean;
  /**
   * Connect (or "connect another account") is offered only for a configured
   * provider, and only while the backend says it can finish the exchange.
   * Several accounts from one provider may be linked side by side.
   */
  canConnect: boolean;
  /** Every live account linked at this provider, oldest first. */
  connections: BrowserConnection[];
}

/** One row per provider, in catalogue order, from the browser list. */
export function panelRows(list: BrowserConnectionList): ConnectionRow[] {
  return PROVIDERS.map((provider) => {
    const availability = list.providers.find((entry) => entry.provider === provider);
    const configured = availability?.configured === true;
    const connections = list.connections.filter((entry) => entry.provider === provider);
    const status: ConnectionRowStatus =
      connections.length > 0 ? 'connected' : configured ? 'not_connected' : 'not_configured';
    return {
      provider,
      label: PROVIDER_LABELS[provider],
      status,
      configured,
      canConnect: configured && list.tokenExchangeAvailable,
      connections,
    };
  });
}

export const CONNECTION_ERROR_TEXT: Record<string, string> = {
  configuration_needed: 'This provider is not set up on this server.',
  unknown_provider: 'That provider is not supported.',
  invalid_state: 'The authorization expired or was already used. Start again.',
  token_exchange_unavailable:
    'The backend reports that it cannot complete the token exchange, so the account was not connected.',
  exchange_failed: 'The provider refused the token exchange. Nothing was connected.',
  unsupported_scope:
    'The provider rejected the permissions this server asks for. Nothing was connected.',
  datacenter_mismatch:
    'Your account is served from a different data center than this server is configured for. Nothing was connected.',
  identity_unavailable:
    'The provider did not say which account was approved, so nothing was connected.',
  not_found: 'That connection no longer exists. Refresh the list.',
  backend_unavailable: 'The backend did not respond. Try again shortly.',
};

function settingNames(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((name): name is string => typeof name === 'string' && ENV_NAME.test(name))
    .slice(0, MAX_MISSING_NAMES);
}

/**
 * Plain words for a connections error code, naming the settings involved by
 * variable name when the backend told us which. Null for codes this slice
 * does not own, so the caller can fall back to the shared identity text.
 */
export function explainConnectionError(
  error: string,
  extra: { missing?: unknown; settings?: unknown } = {}
): string | null {
  const text = CONNECTION_ERROR_TEXT[error];
  if (!text) return null;
  if (error === 'configuration_needed') {
    const names = settingNames(extra.missing);
    if (names.length > 0) return `${text} Ask the operator to set: ${names.join(', ')}.`;
  }
  if (error === 'unsupported_scope' || error === 'datacenter_mismatch') {
    const names = settingNames(extra.settings);
    if (names.length > 0) return `${text} Ask the operator to review: ${names.join(', ')}.`;
  }
  return text;
}
