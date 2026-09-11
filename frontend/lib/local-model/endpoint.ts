/**
 * The local Ollama endpoint and model, as the browser is allowed to see them.
 *
 * The browser never talks to an Ollama itself: it asks the BFF, which asks the
 * agent, which is the only thing that opens a socket. The checks here are the
 * same narrow allowance the backend enforces -- plain http, an explicit port,
 * no credentials, no path, and a host that is either a named local alias or an
 * IP literal in a loopback / RFC1918 / IPv6 ULA-link-local range -- repeated
 * on this side so an operator is told what is wrong as they type. The backend
 * decides; this only saves a round trip and reduces what a route hands on.
 */

/** The two names a local Ollama is reached by, inside and outside a container. */
const LOCAL_ALIASES = new Set(['localhost', 'host.docker.internal']);
const MAX_ENDPOINT_LENGTH = 200;
const MAX_MODEL_NAME = 120;
export const MAX_MODELS = 100;

const MODEL_NAME = /^[A-Za-z0-9][A-Za-z0-9._:/+@-]*$/;
const ASCII_PRINTABLE = /^[\x21-\x7e]+$/;
const APPLIES_TO = new Set(['new_sessions']);

export type EndpointCode =
  | 'invalid_endpoint'
  | 'scheme_not_http'
  | 'credentials_not_allowed'
  | 'path_not_allowed'
  | 'port_required'
  | 'invalid_port'
  | 'host_not_local';

export type EndpointCheck = { ok: true; endpoint: string } | { ok: false; code: EndpointCode };

export interface LocalModelRouting {
  primary: string;
  escalation: string;
  coding: string;
}

export interface LocalModelView {
  endpoint: string;
  model: string;
  localOnly: boolean;
  appliesTo: string;
  routing: LocalModelRouting;
}

export interface ModelsView {
  endpoint: string;
  models: string[];
}

const refuse = (code: EndpointCode): EndpointCheck => ({ ok: false, code });

function isLocalIpv4(host: string): boolean | null {
  const parts = host.split('.');
  if (parts.length !== 4) return null;
  const octets: number[] = [];
  for (const part of parts) {
    if (!/^\d{1,3}$/.test(part)) return null;
    const value = Number(part);
    if (value > 255) return null;
    octets.push(value);
  }
  const [a, b] = octets;
  if (a === 127) return true;
  if (a === 10) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  return false;
}

function isLocalIpv6(host: string): boolean {
  // An IPv4 address wearing an IPv6 coat is refused rather than unwrapped.
  if (host.includes('.') || host.includes('%')) return false;
  if (!/^[0-9a-f:]+$/.test(host)) return false;
  if (/^0*:(:0*)*:?0*1$/.test(host.replace(/::/, ':0:'))) return true;
  if (host === '::1') return true;
  const first = host.startsWith('::') ? '0' : (host.split(':')[0] ?? '');
  const value = Number.parseInt(first || '0', 16);
  if (Number.isNaN(value)) return false;
  // Unique local addresses (fc00::/7) and link-local ones (fe80::/10).
  if (value >= 0xfc00 && value <= 0xfdff) return true;
  return value >= 0xfe80 && value <= 0xfebf;
}

/**
 * The one stored form of an acceptable endpoint, or why it is not one.
 *
 * The authority is taken apart by hand rather than with URL, which normalizes
 * away a default port (http://x:80 loses its 80) and would turn a refusal
 * into the wrong reason.
 */
export function normalizeEndpoint(raw: string): EndpointCheck {
  if (typeof raw !== 'string') return refuse('invalid_endpoint');
  const text = raw.trim();
  if (!text || text.length > MAX_ENDPOINT_LENGTH || !ASCII_PRINTABLE.test(text)) {
    return refuse('invalid_endpoint');
  }
  const scheme = text.slice(0, Math.max(text.indexOf(':'), 0)).toLowerCase();
  if (scheme !== 'http') return refuse('scheme_not_http');
  if (!text.slice(0, 7).toLowerCase().startsWith('http://')) return refuse('invalid_endpoint');

  const rest = text.slice(7);
  const cut = rest.search(/[/?#]/);
  const authority = (cut === -1 ? rest : rest.slice(0, cut)).toLowerCase();
  const remainder = cut === -1 ? '' : rest.slice(cut);
  if (remainder !== '' && remainder !== '/') return refuse('path_not_allowed');
  if (authority.includes('@')) return refuse('credentials_not_allowed');

  let host = '';
  let portText = '';
  let bracketed = false;
  if (authority.startsWith('[')) {
    const close = authority.indexOf(']');
    if (close < 0) return refuse('invalid_endpoint');
    bracketed = true;
    host = authority.slice(1, close);
    const tail = authority.slice(close + 1);
    if (tail === '') return refuse('port_required');
    if (!/^:\d+$/.test(tail)) return refuse('invalid_port');
    portText = tail.slice(1);
  } else {
    const parts = authority.split(':');
    if (parts.length > 2) return refuse('invalid_endpoint');
    host = parts[0] ?? '';
    portText = parts[1] ?? '';
    if (host && parts.length === 1) return refuse('port_required');
  }
  if (!host) return refuse('invalid_endpoint');
  if (portText === '') return refuse('port_required');
  if (!/^\d+$/.test(portText)) return refuse('invalid_port');
  const port = Number(portText);
  if (port < 1 || port > 65535) return refuse('invalid_port');

  if (!bracketed && LOCAL_ALIASES.has(host)) {
    return { ok: true, endpoint: 'http://' + host + ':' + port };
  }
  const ipv4 = bracketed ? null : isLocalIpv4(host);
  if (ipv4 === true) return { ok: true, endpoint: 'http://' + host + ':' + port };
  if (ipv4 === false) return refuse('host_not_local');
  if (bracketed && isLocalIpv6(host)) {
    return { ok: true, endpoint: 'http://[' + host + ']:' + port };
  }
  return refuse('host_not_local');
}

/** Whether this is plausibly an Ollama model name, e.g. qwen3:8b. */
export function isModelName(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    value.length > 0 &&
    value.length <= MAX_MODEL_NAME &&
    MODEL_NAME.test(value)
  );
}

const string = (row: Record<string, unknown>, key: string): string | null =>
  typeof row[key] === 'string' ? (row[key] as string) : null;

function routingOf(value: unknown): LocalModelRouting | null {
  const row = value as Record<string, unknown> | null;
  if (!row || typeof row !== 'object') return null;
  const primary = string(row, 'primary');
  const escalation = string(row, 'escalation');
  const coding = string(row, 'coding');
  if (primary === null || escalation === null || coding === null) return null;
  return { primary, escalation, coding };
}

/**
 * The current choice, reduced to the fields the browser is promised.
 *
 * An endpoint the backend would no longer accept, or a model name that is not
 * one, means the answer is not usable: nothing is shown rather than something
 * misleading being offered as what JARVIS is running on.
 */
export function browserLocalModel(data: unknown): LocalModelView | null {
  const row = data as Record<string, unknown> | null;
  if (!row || typeof row !== 'object') return null;
  const endpoint = string(row, 'endpoint');
  const model = string(row, 'model');
  const appliesTo = string(row, 'applies_to');
  const routing = routingOf(row.routing);
  if (endpoint === null || model === null || routing === null) return null;
  if (!normalizeEndpoint(endpoint).ok) return null;
  if (model !== '' && !isModelName(model)) return null;
  if (appliesTo === null || !APPLIES_TO.has(appliesTo)) return null;
  return {
    endpoint,
    model,
    localOnly: row.local_only === true,
    appliesTo,
    routing,
  };
}

/** A discovered model list: plausible names only, unique, sorted and capped. */
export function browserModels(data: unknown): ModelsView | null {
  const row = data as Record<string, unknown> | null;
  if (!row || typeof row !== 'object') return null;
  const endpoint = string(row, 'endpoint');
  if (endpoint === null || !normalizeEndpoint(endpoint).ok) return null;
  const rows = Array.isArray(row.models) ? row.models : null;
  if (rows === null) return null;
  const models = [...new Set(rows.filter(isModelName))].sort().slice(0, MAX_MODELS);
  return { endpoint, models };
}

const SENTENCES: Record<string, string> = {
  invalid_endpoint: 'That does not look like a URL. Use the form http://192.168.1.50:11434.',
  scheme_not_http: 'Only plain http addresses are allowed; a local Ollama does not use https.',
  credentials_not_allowed: 'Remove the username and password from the address.',
  path_not_allowed: 'Give the base address only, with no path, query or fragment.',
  port_required: 'Add the port, usually 11434.',
  invalid_port: 'That port number is not valid.',
  host_not_local:
    'Only localhost, host.docker.internal and private network addresses are allowed. ' +
    'This setting is local-network-only.',
  unresolvable: 'That name could not be resolved on this network.',
  invalid_model: 'That is not a model name. Refresh the list and pick one.',
  unreachable:
    'Could not reach Ollama there. Check that it is running and that it listens on the ' +
    'network, not only on its own loopback.',
  timeout: 'Ollama did not answer in time.',
  upstream_error: 'Ollama answered, but not with a model list.',
  unexpected_response: 'That address answered, but it does not look like Ollama.',
  forbidden: 'Only an administrator can change the local model.',
  unauthorized: 'Your session has expired. Sign in again.',
  rate_limited: 'Too many attempts just now. Wait a moment and try again.',
  backend_unavailable: 'JARVIS is not answering just now. Try again in a moment.',
};

const FALLBACK = 'That did not work. Check the address and try again.';

/** A sentence for a backend code. An unknown code is never shown back. */
export function describeEndpointCode(code: unknown): string {
  return typeof code === 'string' && code in SENTENCES ? SENTENCES[code] : FALLBACK;
}
