/**
 * Browser-safe rules for a CAAL web device session.
 *
 * The bearer session token minted by `POST /devices/register` deliberately does
 * not appear anywhere in this module. It is held by the server-side routes under
 * `app/api/devices/*` in an HttpOnly cookie, so nothing that reaches the browser
 * — including this file — can read, render, or log it. What lives here is only
 * the identity, label, and formatting logic that both sides need to agree on.
 */

/** Stable per-browser identity, persisted so a reload keeps the same session. */
export const DEVICE_ID_STORAGE_KEY = 'caal.device.id';
export const DEVICE_LABEL_STORAGE_KEY = 'caal.device.label';

/** Name of the HttpOnly cookie holding the device's bearer session token. */
export const DEVICE_SESSION_COOKIE = 'caal_device_session';

/** Mirrors `device_registry.MAX_LABEL_LENGTH` so the UI rejects what the backend would. */
export const MAX_DEVICE_LABEL_LENGTH = 40;

/** Bounds on the cookie lifetime we will accept from the backend. */
const MIN_SESSION_LIFETIME_SECONDS = 60;
const MAX_SESSION_LIFETIME_SECONDS = 3600;

const DEVICE_ID_PATTERN =
  /^web-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

/** Control and format characters, which the backend refuses in a label. */
const UNPRINTABLE = /[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}]/u;

/** The slice of `Storage` we use, so tests can hand in a plain object. */
export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export type LabelValidation = { ok: true; label: string } | { ok: false; error: string };

export interface ActiveDevice {
  label: string;
  transport: string;
  lastSeen: number;
  isSelf: boolean;
}

/** Transports the backend can report; anything else is not shown. */
const KNOWN_TRANSPORTS = new Set(['web', 'mobile', 'phone']);

function randomUuid(): string {
  const bytes = new Uint8Array(16);
  // `crypto.randomUUID` is secure-context only, and CAAL is routinely served
  // over plain HTTP on a LAN, so build the v4 UUID from raw random bytes.
  const webcrypto = globalThis.crypto;
  if (typeof webcrypto?.getRandomValues === 'function') {
    webcrypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < bytes.length; i++) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;

  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20),
  ].join('-');
}

/** Return true for an id this browser could have minted itself. */
export function isDeviceId(value: unknown): value is string {
  return typeof value === 'string' && DEVICE_ID_PATTERN.test(value);
}

/**
 * Return the browser's stable device id, minting and persisting one if the
 * stored value is missing or does not look like an id we issued.
 */
export function ensureDeviceId(store: KeyValueStore): string {
  const stored = store.getItem(DEVICE_ID_STORAGE_KEY);
  if (isDeviceId(stored)) {
    return stored;
  }

  const deviceId = `web-${randomUuid()}`;
  store.setItem(DEVICE_ID_STORAGE_KEY, deviceId);
  return deviceId;
}

/** Normalize a user-typed device name, or explain why it cannot be used. */
export function validateDeviceLabel(value: unknown): LabelValidation {
  if (typeof value !== 'string') {
    return { ok: false, error: 'Device name must be text.' };
  }

  const label = value.trim().replace(/\s+/g, ' ');
  if (!label) {
    return { ok: false, error: 'Device name is required.' };
  }
  if (label.length > MAX_DEVICE_LABEL_LENGTH) {
    return {
      ok: false,
      error: `Device name must be at most ${MAX_DEVICE_LABEL_LENGTH} characters.`,
    };
  }
  if (UNPRINTABLE.test(label)) {
    return { ok: false, error: 'Device name must not contain control characters.' };
  }

  return { ok: true, label };
}

/** Persist a valid device name and return it; invalid names are never stored. */
export function storeDeviceLabel(store: KeyValueStore, value: unknown): string {
  const result = validateDeviceLabel(value);
  if (!result.ok) {
    throw new Error(result.error);
  }

  store.setItem(DEVICE_LABEL_STORAGE_KEY, result.label);
  return result.label;
}

/** Return the persisted device name, or null when absent or no longer valid. */
export function readStoredLabel(store: KeyValueStore): string | null {
  const result = validateDeviceLabel(store.getItem(DEVICE_LABEL_STORAGE_KEY));
  return result.ok ? result.label : null;
}

const BROWSER_NAMES: [RegExp, string][] = [
  [/\bEdg(?:e|A|iOS)?\//, 'Edge'],
  [/\bOPR\/|\bOpera\b/, 'Opera'],
  [/\b(?:Chrome|CriOS)\//, 'Chrome'],
  [/\b(?:Firefox|FxiOS)\//, 'Firefox'],
  [/\bSafari\//, 'Safari'],
];

const PLATFORM_NAMES: [RegExp, string][] = [
  [/\bMac OS X\b|\bMacintosh\b/, 'macOS'],
  [/\bWindows\b/, 'Windows'],
  [/\bAndroid\b/, 'Android'],
  [/\b(?:iPhone|iPad|iPod)\b/, 'iOS'],
  [/\bCrOS\b/, 'ChromeOS'],
  [/\bLinux\b/, 'Linux'],
];

function firstMatch(candidates: [RegExp, string][], userAgent: string): string | null {
  for (const [pattern, name] of candidates) {
    if (pattern.test(userAgent)) {
      return name;
    }
  }
  return null;
}

/**
 * Suggest a friendly name for this browser. Only ever a hint — the user can
 * rename the device, and the result is bounded like any other label.
 */
export function defaultDeviceLabel(userAgent: unknown): string {
  const agent = typeof userAgent === 'string' ? userAgent : '';
  const browser = firstMatch(BROWSER_NAMES, agent);
  const platform = firstMatch(PLATFORM_NAMES, agent);

  let label = 'Web browser';
  if (browser && platform) {
    label = `${browser} on ${platform}`;
  } else if (browser) {
    label = browser;
  } else if (platform) {
    label = `Browser on ${platform}`;
  }

  return label.slice(0, MAX_DEVICE_LABEL_LENGTH);
}

/** Clamp a backend-provided lifetime into a range we are willing to honour. */
export function clampSessionLifetime(expiresIn: unknown): number {
  const seconds = typeof expiresIn === 'number' ? Math.floor(expiresIn) : Number.NaN;
  if (!Number.isFinite(seconds) || seconds < MIN_SESSION_LIFETIME_SECONDS) {
    return MIN_SESSION_LIFETIME_SECONDS;
  }
  return Math.min(seconds, MAX_SESSION_LIFETIME_SECONDS);
}

export interface SessionCookieOptions {
  name: string;
  httpOnly: true;
  sameSite: 'lax';
  secure: boolean;
  path: string;
  maxAge: number;
}

/**
 * Cookie attributes for the device session token. HttpOnly keeps it out of
 * scripts entirely, and SameSite=Lax keeps a third-party page from spending it.
 */
export function sessionCookieOptions(
  expiresIn: number,
  { secure = false }: { secure?: boolean } = {}
): SessionCookieOptions {
  return {
    name: DEVICE_SESSION_COOKIE,
    httpOnly: true,
    sameSite: 'lax',
    secure,
    path: '/',
    maxAge: clampSessionLifetime(expiresIn),
  };
}

/** Same cookie, expired immediately — used whenever the session stops being valid. */
export function clearedSessionCookieOptions({ secure = false }: { secure?: boolean } = {}) {
  return { ...sessionCookieOptions(MIN_SESSION_LIFETIME_SECONDS, { secure }), maxAge: 0 };
}

/**
 * Read the backend's active-device list defensively: anything malformed or on
 * an unknown transport is dropped rather than rendered.
 */
export function normalizeActiveDevices(payload: unknown): ActiveDevice[] {
  const devices = (payload as { devices?: unknown })?.devices;
  if (!Array.isArray(devices)) {
    return [];
  }

  const normalized: ActiveDevice[] = [];
  for (const entry of devices) {
    const device = entry as Record<string, unknown>;
    const label = validateDeviceLabel(device?.label);
    const transport = typeof device?.transport === 'string' ? device.transport : '';
    const lastSeen = typeof device?.last_seen === 'number' ? device.last_seen : Number.NaN;

    if (!label.ok || !KNOWN_TRANSPORTS.has(transport) || !Number.isFinite(lastSeen)) {
      continue;
    }

    normalized.push({
      label: label.label,
      transport,
      lastSeen: Math.floor(lastSeen),
      isSelf: device?.is_self === true,
    });
  }
  return normalized;
}

/** Render a unix timestamp as coarse relative time; clock skew reads as "just now". */
export function formatLastSeen(lastSeen: number, now: number = Date.now() / 1000): string {
  if (!Number.isFinite(lastSeen) || !Number.isFinite(now)) {
    return 'unknown';
  }

  const elapsed = Math.max(0, Math.floor(now - lastSeen));
  if (elapsed < 60) {
    return 'just now';
  }
  if (elapsed < 3600) {
    return `${Math.floor(elapsed / 60)}m ago`;
  }
  if (elapsed < 86_400) {
    return `${Math.floor(elapsed / 3600)}h ago`;
  }
  return `${Math.floor(elapsed / 86_400)}d ago`;
}
