import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  DEVICE_SESSION_COOKIE,
  MAX_DEVICE_LABEL_LENGTH,
  defaultDeviceLabel,
  ensureDeviceId,
  formatLastSeen,
  readStoredLabel,
  sessionCookieOptions,
  storeDeviceLabel,
  validateDeviceLabel,
} from './device-session.ts';

function memoryStore(seed: Record<string, string> = {}) {
  const data = new Map(Object.entries(seed));
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, value),
    snapshot: () => Object.fromEntries(data),
  };
}

describe('device identity', () => {
  it('creates a stable id and reuses it across calls', () => {
    const store = memoryStore();

    const first = ensureDeviceId(store);
    const second = ensureDeviceId(store);

    assert.equal(first, second);
    assert.match(first, /^web-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
  });

  it('replaces a malformed stored id instead of trusting it', () => {
    const store = memoryStore({ 'caal.device.id': 'not a device id' });

    const deviceId = ensureDeviceId(store);

    assert.notEqual(deviceId, 'not a device id');
    assert.equal(ensureDeviceId(store), deviceId);
  });
});

describe('friendly label', () => {
  it('accepts a label and collapses surrounding whitespace', () => {
    assert.deepEqual(validateDeviceLabel('  Studio   Laptop '), {
      ok: true,
      label: 'Studio Laptop',
    });
  });

  it('rejects empty, overlong, and control-character labels', () => {
    for (const bad of ['', '   ', 'x'.repeat(MAX_DEVICE_LABEL_LENGTH + 1), 'bell\u0007']) {
      const result = validateDeviceLabel(bad);
      assert.equal(result.ok, false, `expected ${JSON.stringify(bad)} to be rejected`);
      assert.equal(typeof (result as { error: string }).error, 'string');
    }
  });

  it('persists a valid label and refuses to persist an invalid one', () => {
    const store = memoryStore();

    assert.equal(storeDeviceLabel(store, ' Studio  Laptop '), 'Studio Laptop');
    assert.equal(readStoredLabel(store), 'Studio Laptop');
    assert.throws(() => storeDeviceLabel(store, '   '));
    assert.equal(readStoredLabel(store), 'Studio Laptop');
  });

  it('derives a bounded default label from the user agent', () => {
    assert.equal(
      defaultDeviceLabel(
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
      ),
      'Chrome on macOS'
    );
    assert.equal(defaultDeviceLabel(''), 'Web browser');
    assert.ok(defaultDeviceLabel('x'.repeat(500)).length <= MAX_DEVICE_LABEL_LENGTH);
  });
});

describe('session cookie', () => {
  it('is http-only and same-site so scripts can never read the token', () => {
    const options = sessionCookieOptions(300, { secure: true });

    assert.equal(options.name, DEVICE_SESSION_COOKIE);
    assert.equal(options.httpOnly, true);
    assert.equal(options.sameSite, 'lax');
    assert.equal(options.secure, true);
    assert.equal(options.path, '/');
    assert.equal(options.maxAge, 300);
  });

  it('clamps an implausible lifetime from the backend', () => {
    assert.equal(sessionCookieOptions(0).maxAge, 60);
    assert.equal(sessionCookieOptions(999_999).maxAge, 3600);
    assert.equal(sessionCookieOptions(Number.NaN).maxAge, 60);
    assert.equal(sessionCookieOptions(300).secure, false);
  });
});

describe('last seen formatting', () => {
  it('reads as friendly relative time', () => {
    assert.equal(formatLastSeen(1000, 1005), 'just now');
    assert.equal(formatLastSeen(1000, 1000 + 120), '2m ago');
    assert.equal(formatLastSeen(1000, 1000 + 7200), '2h ago');
    assert.equal(formatLastSeen(1000, 900), 'just now');
  });
});
