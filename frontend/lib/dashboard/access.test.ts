import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import type { AuthResult } from '../auth/session.ts';
import { dashboardScope } from './access.ts';

const USER_ID = 'usr_0123456789abcdef01234567';

// Only the discriminants matter to `dashboardScope`; the config object is opaque to it.
const config = {} as Extract<AuthResult, { kind: 'anonymous' }>['config'];

function signedIn(overrides: Partial<Extract<AuthResult, { kind: 'user' }>> = {}): AuthResult {
  return {
    kind: 'user',
    config,
    user: { userId: USER_ID, role: 'member', status: 'active', displayName: 'Cesar' },
    via: 'password',
    mustChangePassword: false,
    sessionToken: 'opaque',
    ...overrides,
  };
}

describe('dashboard access scope', () => {
  it('is local for a legacy single-user deployment', () => {
    const scope = dashboardScope({
      kind: 'unconfigured',
      status: { status: 'disabled' },
    } as AuthResult);

    assert.deepEqual(scope, { ok: true, scope: 'local', userId: null });
  });

  it('is bound to the opaque user id for a signed-in user', () => {
    assert.deepEqual(dashboardScope(signedIn()), { ok: true, scope: 'user', userId: USER_ID });
  });

  it('refuses a one-time password session until the password is changed', () => {
    const scope = dashboardScope(signedIn({ mustChangePassword: true }));

    assert.deepEqual(scope, { ok: false, status: 403, code: 'password_change_required' });
  });

  it('refuses anyone who is not a user once identity is configured', () => {
    assert.deepEqual(dashboardScope({ kind: 'anonymous', config }), {
      ok: false,
      status: 401,
      code: 'not_signed_in',
    });
    assert.deepEqual(dashboardScope({ kind: 'expired', config }), {
      ok: false,
      status: 401,
      code: 'session_expired',
    });
    assert.deepEqual(dashboardScope({ kind: 'invalid', config }), {
      ok: false,
      status: 401,
      code: 'invalid_assertion',
    });
    assert.deepEqual(dashboardScope({ kind: 'denied', config, reason: 'suspended' }), {
      ok: false,
      status: 403,
      code: 'suspended',
    });
  });
});
