import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { maskEmail, readIdentityConfig } from './config.ts';

const VALID = {
  CF_ACCESS_TEAM_DOMAIN: 'https://flamingsoftinc.cloudflareaccess.com',
  CF_ACCESS_AUD: 'd1d09a2c79e964918d59077b9bb5b3a7b67ff76a04a80822eb8a3ce5f46354ac',
  CAAL_INTERNAL_AUTH_SECRET: 'k'.repeat(48),
  CAAL_IDENTITY_API_URL: 'http://agent:8889',
};

describe('identity configuration', () => {
  it('is enabled only when every value validates', () => {
    const status = readIdentityConfig(VALID);
    assert.equal(status.status, 'enabled');
    assert.equal(status.config?.teamDomain, VALID.CF_ACCESS_TEAM_DOMAIN);
    assert.equal(status.config?.audience, VALID.CF_ACCESS_AUD);
    assert.equal(status.config?.apiBaseUrl, 'http://agent:8889');
    assert.equal(status.config?.requireIdentityForSessions, false);
    assert.deepEqual(status.problems, []);
  });

  it('reports legacy mode when nothing is configured', () => {
    const status = readIdentityConfig({});
    assert.equal(status.status, 'disabled');
    // Only the two genuinely required values: the shared secret and the
    // backend URL. Cloudflare Access is optional.
    assert.deepEqual(status.problems, [
      'CAAL_INTERNAL_AUTH_SECRET (missing)',
      'CAAL_IDENTITY_API_URL (missing)',
    ]);
  });

  it('fails closed and names the problem without echoing values', () => {
    for (const [name, value] of [
      ['CF_ACCESS_TEAM_DOMAIN', 'http://flamingsoftinc.cloudflareaccess.com'],
      ['CF_ACCESS_TEAM_DOMAIN', 'https://example.com'],
      ['CF_ACCESS_AUD', 'not-hex'],
      ['CAAL_INTERNAL_AUTH_SECRET', 'short'],
      ['CAAL_IDENTITY_API_URL', 'ftp://agent'],
    ] as const) {
      const status = readIdentityConfig({ ...VALID, [name]: value });
      assert.equal(status.status, 'invalid', name);
      assert.ok(
        status.problems.some((p) => p.includes(name)),
        name
      );
      assert.ok(!status.problems.join(' ').includes(value), name);
      assert.equal(status.config, undefined);
    }
  });

  it('falls back to the device/webhook URL for the backend and reads the strict flag', () => {
    const status = readIdentityConfig({
      ...VALID,
      CAAL_IDENTITY_API_URL: undefined,
      WEBHOOK_URL: 'http://agent:8889/',
      CAAL_REQUIRE_IDENTITY_FOR_SESSIONS: 'true',
      CAAL_PUBLIC_ORIGIN: 'https://jarvis.example.com/',
    });
    assert.equal(status.status, 'enabled');
    assert.equal(status.config?.apiBaseUrl, 'http://agent:8889');
    assert.equal(status.config?.requireIdentityForSessions, true);
    assert.equal(status.config?.publicOrigin, 'https://jarvis.example.com');
  });

  it('never renders secrets', () => {
    const status = readIdentityConfig(VALID);
    assert.ok(!JSON.stringify(status.problems).includes('k'.repeat(48)));
    assert.equal(status.config?.describe().includes('k'.repeat(48)), false);
  });
});

describe('email masking for browser responses', () => {
  it('keeps only a hint of the local part and the domain', () => {
    assert.equal(maskEmail('cesarc@mexcantech.com'), 'c•••••@mexcantech.com');
    assert.equal(maskEmail('a@b.co'), 'a•••••@b.co');
    assert.equal(maskEmail('not-an-email'), '•••••');
    assert.equal(maskEmail(42), '•••••');
  });
});

// --- standalone (Cloudflare-free) operation ------------------------------------

const LOCAL = {
  CAAL_INTERNAL_AUTH_SECRET: 'k'.repeat(48),
  CAAL_IDENTITY_API_URL: 'http://agent:8889',
};

describe('standalone identity configuration', () => {
  it('is enabled with no Cloudflare variables at all', () => {
    const status = readIdentityConfig(LOCAL);

    assert.equal(status.status, 'enabled');
    assert.equal(status.config?.accessEnabled, false);
    assert.equal(status.config?.teamDomain, null);
    assert.equal(status.config?.audience, null);
    assert.equal(status.config?.passwordLogin, true);
  });

  it('refuses a half-configured Cloudflare pair instead of downgrading', () => {
    for (const half of ['CF_ACCESS_TEAM_DOMAIN', 'CF_ACCESS_AUD'] as const) {
      const status = readIdentityConfig({ ...LOCAL, [half]: VALID[half] });
      assert.equal(status.status, 'invalid', half);
      assert.equal(status.config, undefined, half);
    }
  });

  it('defaults to refusing cookies over plain HTTP', () => {
    assert.equal(readIdentityConfig(LOCAL).config?.allowInsecureCookies, false);
    assert.equal(
      readIdentityConfig({ ...LOCAL, CAAL_ALLOW_INSECURE_COOKIES: 'true' }).config
        ?.allowInsecureCookies,
      true
    );
  });

  it('refuses a configuration with no way to sign in', () => {
    const status = readIdentityConfig({ ...LOCAL, CAAL_PASSWORD_LOGIN: 'false' });

    assert.equal(status.status, 'invalid');
    assert.ok(status.problems.some((p) => p.includes('no way to sign in')));
  });

  it('allows password login to be switched off when Access is configured', () => {
    const status = readIdentityConfig({ ...VALID, CAAL_PASSWORD_LOGIN: 'false' });

    assert.equal(status.status, 'enabled');
    assert.equal(status.config?.passwordLogin, false);
    assert.equal(status.config?.accessEnabled, true);
  });

  it('rejects non-boolean flags by name', () => {
    for (const name of [
      'CAAL_PASSWORD_LOGIN',
      'CAAL_ALLOW_INSECURE_COOKIES',
      'CAAL_REQUIRE_IDENTITY_FOR_SESSIONS',
    ] as const) {
      const status = readIdentityConfig({ ...LOCAL, [name]: 'sometimes' });
      assert.equal(status.status, 'invalid', name);
      assert.ok(
        status.problems.some((p) => p.startsWith(name)),
        name
      );
    }
  });

  it('describes the active providers without naming Cloudflare when it is off', () => {
    const local = readIdentityConfig(LOCAL).config!.describe();
    const both = readIdentityConfig(VALID).config!.describe();

    assert.ok(local.includes('password'));
    assert.ok(!local.includes('Cloudflare'));
    assert.ok(both.includes('Cloudflare'));
  });

  it('warns in its description when insecure cookies are permitted', () => {
    const status = readIdentityConfig({ ...LOCAL, CAAL_ALLOW_INSECURE_COOKIES: 'yes' });

    assert.ok(status.config!.describe().includes('INSECURE'));
  });
});
