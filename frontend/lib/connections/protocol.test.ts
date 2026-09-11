import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  MAX_ALIASES,
  MAX_CODE_LENGTH,
  OAUTH_CALLBACK_PATH,
  OUTCOMES,
  PROVIDERS,
  accountNamesFrom,
  accountNamesRequest,
  browserConnection,
  browserConnectionList,
  cleanNames,
  configurationNeeded,
  describeOutcome,
  explainConnectionError,
  isAuthorizationCode,
  isConnectionId,
  isOutcome,
  isProvider,
  outcomeFromAuth,
  outcomeFromBackend,
  nameKey,
  panelRows,
  parseAliasInput,
  parseAuthorization,
  parseCallbackQuery,
  parseState,
  resultPath,
} from './protocol.ts';

const STATE_ID = 'abcdefghijklmnopqrstuvwxyz012345';
const SIGNATURE = 'S1gnatureS1gnatureS1gnatureS1gnatureS1gnat_';
const STATE = `${STATE_ID}.${SIGNATURE}`;
const CODE = '4/0AX4XfWh-AUTHORIZATION-CODE';
const CONNECTION_ID = 'con_' + '0'.repeat(24);
const NOW = 1_700_000_000;

const backendRow = {
  connection_id: CONNECTION_ID,
  provider: 'google',
  status: 'connected',
  account_label: 'ana@gmail.com',
  scopes: ['openid', 'email'],
  token_expires_at: NOW + 3600,
  has_refresh_token: true,
  connected_at: NOW,
  updated_at: NOW,
};

const backendProviders = [
  { provider: 'google', display_name: 'Google', configured: true },
  { provider: 'microsoft', display_name: 'Microsoft', configured: false },
  { provider: 'zoho', display_name: 'Zoho', configured: false },
];

const googleUrl = (state: string) =>
  `https://accounts.google.com/o/oauth2/v2/auth?client_id=id&redirect_uri=https%3A%2F%2Fjarvis.example%2Fapi%2Fconnections%2Fcallback&response_type=code&scope=openid&state=${state}&code_challenge=c&code_challenge_method=S256`;

describe('provider catalogue and validators', () => {
  it('knows exactly the three providers and the fixed callback path', () => {
    assert.deepEqual([...PROVIDERS], ['google', 'microsoft', 'zoho']);
    assert.equal(OAUTH_CALLBACK_PATH, '/api/connections/callback');
  });

  it('accepts only exact provider names', () => {
    for (const good of ['google', 'microsoft', 'zoho']) assert.equal(isProvider(good), true, good);
    for (const bad of ['GOOGLE', 'apple', 'google ', '', null, undefined, 0, {}]) {
      assert.equal(isProvider(bad), false, String(bad));
    }
  });

  it('accepts only well-formed connection ids', () => {
    assert.equal(isConnectionId(CONNECTION_ID), true);
    for (const bad of ['con_' + '0'.repeat(23), 'usr_' + '0'.repeat(24), 'con_' + 'Z'.repeat(24), '', 7]) {
      assert.equal(isConnectionId(bad), false, String(bad));
    }
  });

  it('splits a state into its opaque id and signature and refuses anything else', () => {
    assert.deepEqual(parseState(STATE), { state: STATE, stateId: STATE_ID });
    for (const bad of [
      STATE_ID,
      `${STATE}.`,
      `${STATE_ID}.${SIGNATURE.slice(1)}`,
      `${STATE_ID.slice(1)}.${SIGNATURE}`,
      `${STATE_ID}.${SIGNATURE}/`,
      ' ' + STATE,
      STATE.replace('a', 'ä'),
      'x'.repeat(300),
      '',
      null,
      42,
    ]) {
      assert.equal(parseState(bad), null, String(bad));
    }
  });

  it('accepts an authorization code as visible ASCII within the backend limit', () => {
    assert.equal(MAX_CODE_LENGTH, 4096);
    assert.equal(isAuthorizationCode(CODE), true);
    assert.equal(isAuthorizationCode('1000.abc-DEF_ghi.jkl'), true);
    assert.equal(isAuthorizationCode('a'.repeat(MAX_CODE_LENGTH)), true);
    for (const bad of ['', 'has space', 'tab\there', 'ünïcode', 'a'.repeat(MAX_CODE_LENGTH + 1), null]) {
      assert.equal(isAuthorizationCode(bad), false, String(bad));
    }
  });
});

describe('browser shapes', () => {
  it('reduces the backend list to the browser shape and never forwards unknown fields', () => {
    const list = browserConnectionList({
      connections: [{ ...backendRow, access_token_enc: 'enc:v1:SECRET', user_id: 'usr_x' }],
      providers: backendProviders,
      token_exchange_available: true,
      extra: 'ignored',
    });

    assert.deepEqual(list, {
      connections: [
        {
          connectionId: CONNECTION_ID,
          provider: 'google',
          status: 'connected',
          accountLabel: 'ana@gmail.com',
          userLabel: null,
          aliases: [],
          scopes: ['openid', 'email'],
          tokenExpiresAt: NOW + 3600,
          hasRefreshToken: true,
          connectedAt: NOW,
          updatedAt: NOW,
        },
      ],
      providers: [
        { provider: 'google', displayName: 'Google', configured: true },
        { provider: 'microsoft', displayName: 'Microsoft / Outlook', configured: false },
        { provider: 'zoho', displayName: 'Zoho', configured: false },
      ],
      tokenExchangeAvailable: true,
    });
    const rendered = JSON.stringify(list);
    assert.ok(!rendered.includes('enc:v1'));
    assert.ok(!rendered.includes('usr_x'));
    assert.ok(!rendered.includes('access_token'));
  });

  it('believes the backend, and only the backend, about whether an exchange can be completed', () => {
    const base = { connections: [], providers: backendProviders };
    assert.equal(browserConnectionList({ ...base, token_exchange_available: true })?.tokenExchangeAvailable, true);
    for (const claim of [false, undefined, null, 'true', 1, {}]) {
      const list = browserConnectionList({ ...base, token_exchange_available: claim });
      assert.equal(list?.tokenExchangeAvailable, false, String(claim));
    }
  });

  it('drops malformed, foreign and non-live rows, and fails closed on a bad payload', () => {
    const list = browserConnectionList({
      connections: [
        backendRow,
        { ...backendRow, provider: 'apple' },
        { ...backendRow, status: 'revoked' },
        { ...backendRow, connection_id: 'nope' },
        { ...backendRow, scopes: 'openid' },
        { ...backendRow, account_label: null, scopes: [] },
        'garbage',
      ],
      providers: [{ provider: 'google', display_name: 'Google', configured: true }],
    });
    assert.ok(list);
    assert.equal(list.connections.length, 2);
    assert.equal(list.connections[1].accountLabel, null);
    assert.deepEqual(
      list.providers.map((entry) => [entry.provider, entry.configured]),
      [
        ['google', true],
        ['microsoft', false],
        ['zoho', false],
      ]
    );

    for (const bad of [null, {}, { connections: [] }, { providers: [] }, { connections: {}, providers: [] }, 'x']) {
      assert.equal(browserConnectionList(bad), null, JSON.stringify(bad));
    }
  });

  it('accepts only the provider’s own https authorization endpoint carrying a well-formed state', () => {
    const good = parseAuthorization('google', {
      provider: 'google',
      authorization_url: googleUrl(STATE),
      expires_at: NOW + 600,
    });
    assert.deepEqual(good, {
      authorizationUrl: googleUrl(STATE),
      stateId: STATE_ID,
      expiresAt: NOW + 600,
    });

    const microsoft = parseAuthorization('microsoft', {
      provider: 'microsoft',
      authorization_url: `https://login.microsoftonline.com/common/oauth2/v2.0/authorize?state=${STATE}`,
      expires_at: NOW + 600,
    });
    assert.equal(microsoft?.stateId, STATE_ID);
    const zoho = parseAuthorization('zoho', {
      provider: 'zoho',
      authorization_url: `https://accounts.zoho.eu/oauth/v2/auth?state=${STATE}`,
      expires_at: NOW + 600,
    });
    assert.equal(zoho?.stateId, STATE_ID);

    const base = { provider: 'google', authorization_url: googleUrl(STATE), expires_at: NOW + 600 };
    for (const bad of [
      { ...base, authorization_url: googleUrl(STATE).replace('https://', 'http://') },
      { ...base, authorization_url: googleUrl(STATE).replace('accounts.google.com', 'accounts.google.com.evil.example') },
      { ...base, authorization_url: googleUrl(STATE).replace('accounts.google.com', 'evil.example') },
      { ...base, authorization_url: 'javascript:alert(1)' },
      { ...base, authorization_url: googleUrl('').replace('&state=', '&nostate=') },
      { ...base, authorization_url: googleUrl('bad.state') },
      { ...base, provider: 'zoho' },
      { ...base, expires_at: undefined },
      { ...base, expires_at: 'soon' },
      { ...base, expires_at: -1 },
      { ...base, expires_at: 1.5 },
      null,
      'x',
    ]) {
      assert.equal(parseAuthorization('google', bad), null, JSON.stringify(bad));
    }
    assert.equal(
      parseAuthorization('zoho', {
        provider: 'zoho',
        authorization_url: `http://accounts.zoho.com/oauth/v2/auth?state=${STATE}`,
        expires_at: NOW + 600,
      }),
      null
    );
  });

  it('extracts a configuration-needed answer as variable names only, bounded', () => {
    assert.deepEqual(
      configurationNeeded({
        detail: 'provider_not_configured',
        status: 'configuration_needed',
        provider: 'zoho',
        missing: ['CAAL_OAUTH_ZOHO_CLIENT_ID', 'CAAL_OAUTH_ZOHO_CLIENT_SECRET'],
      }),
      { provider: 'zoho', missing: ['CAAL_OAUTH_ZOHO_CLIENT_ID', 'CAAL_OAUTH_ZOHO_CLIENT_SECRET'] }
    );
    const noisy = configurationNeeded({
      detail: 'provider_not_configured',
      provider: 'google',
      missing: ['lowercase', 'HAS SPACE', 'X'.repeat(65), 'sk-live-LEAK', 7, 'CAAL_PUBLIC_ORIGIN'],
    });
    assert.deepEqual(noisy, { provider: 'google', missing: ['CAAL_PUBLIC_ORIGIN'] });
    const many = configurationNeeded({
      detail: 'provider_not_configured',
      provider: 'google',
      missing: Array.from({ length: 40 }, (_, i) => `VAR_${i}`),
    });
    assert.ok(many && many.missing.length <= 12);
    for (const bad of [
      { detail: 'token_exchange_unavailable' },
      { detail: 'provider_not_configured', provider: 'apple', missing: [] },
      { detail: 'provider_not_configured', provider: 'google', missing: 'X' },
      null,
    ]) {
      assert.equal(configurationNeeded(bad), null, JSON.stringify(bad));
    }
  });
});

describe('callback parsing', () => {
  it('reads a successful redirect and ignores the extras providers add', () => {
    const query = parseCallbackQuery(
      new URLSearchParams({ state: STATE, code: CODE, scope: 'openid email', authuser: '0', prompt: 'consent' })
    );
    assert.deepEqual(query, { kind: 'code', state: STATE, stateId: STATE_ID, code: CODE, location: null });
  });

  it('carries Zoho’s data-center code through when it is well formed, and drops it otherwise', () => {
    const withLocation = parseCallbackQuery(
      new URLSearchParams({ state: STATE, code: CODE, location: 'eu', 'accounts-server': 'https://accounts.zoho.eu' })
    );
    assert.deepEqual(withLocation, { kind: 'code', state: STATE, stateId: STATE_ID, code: CODE, location: 'eu' });
    for (const bad of ['EU', 'usa', 'u', '', 'e-u', 'https://accounts.zoho.eu']) {
      const query = parseCallbackQuery(new URLSearchParams({ state: STATE, code: CODE, location: bad }));
      assert.equal(query.kind, 'code', bad);
      assert.equal(query.kind === 'code' ? query.location : 'x', null, bad);
    }
    const repeated = parseCallbackQuery(new URLSearchParams(`state=${STATE}&code=${CODE}&location=us&location=eu`));
    assert.equal(repeated.kind === 'code' ? repeated.location : 'x', null);
  });

  it('recognises a rejected scope as its own outcome, never as a code', () => {
    assert.deepEqual(
      parseCallbackQuery(new URLSearchParams({ error: 'invalid_scope', state: STATE, code: CODE })),
      { kind: 'unsupported_scope' }
    );
  });

  it('treats a provider error as a refusal and never as a code, even when a code is present', () => {
    assert.deepEqual(
      parseCallbackQuery(new URLSearchParams({ error: 'access_denied', state: STATE })),
      { kind: 'denied' }
    );
    assert.deepEqual(
      parseCallbackQuery(new URLSearchParams({ error: 'access_denied', state: STATE, code: CODE })),
      { kind: 'denied' }
    );
    assert.deepEqual(
      parseCallbackQuery(
        new URLSearchParams({
          error: 'server_error',
          error_description: '<script>alert(1)</script>',
          state: STATE,
        })
      ),
      { kind: 'provider_error' }
    );
  });

  it('is invalid without a well-formed state and code, or with repeated parameters', () => {
    for (const params of [
      new URLSearchParams({ code: CODE }),
      new URLSearchParams({ state: STATE }),
      new URLSearchParams({ state: STATE, code: '' }),
      new URLSearchParams({ state: STATE, code: 'has space' }),
      new URLSearchParams({ state: STATE_ID, code: CODE }),
      new URLSearchParams({ state: 'x'.repeat(300), code: CODE }),
      new URLSearchParams({ state: STATE, code: 'a'.repeat(MAX_CODE_LENGTH + 1) }),
      new URLSearchParams(`state=${STATE}&state=${STATE}&code=${CODE}`),
      new URLSearchParams(`state=${STATE}&code=${CODE}&code=other`),
      new URLSearchParams(),
    ]) {
      assert.deepEqual(parseCallbackQuery(params), { kind: 'invalid' }, params.toString());
    }
  });
});

describe('outcomes', () => {
  it('maps the backend callback answer to a bounded outcome', () => {
    assert.equal(outcomeFromBackend(200, { provider: 'google', status: 'connected' }), 'connected');
    assert.equal(
      outcomeFromBackend(503, { detail: 'token_exchange_unavailable' }),
      'token_exchange_unavailable'
    );
    assert.equal(
      outcomeFromBackend(503, { detail: 'provider_not_configured', missing: ['X'] }),
      'configuration_needed'
    );
    assert.equal(outcomeFromBackend(503, { detail: 'Multi-user identity is not configured' }), 'identity_not_configured');
    assert.equal(outcomeFromBackend(400, { detail: 'invalid_state' }), 'invalid_state');
    assert.equal(outcomeFromBackend(400, { detail: 'other' }), 'invalid_callback');
    assert.equal(outcomeFromBackend(422, { detail: [] }), 'invalid_callback');
    assert.equal(outcomeFromBackend(502, { detail: 'token_exchange_failed' }), 'exchange_failed');
    assert.equal(
      outcomeFromBackend(502, { detail: 'token_exchange_failed', reason: 'provider_refused' }),
      'exchange_failed'
    );
    assert.equal(
      outcomeFromBackend(502, { detail: 'token_exchange_failed', reason: 'insufficient_scope', settings: ['CAAL_OAUTH_ZOHO_SCOPES'] }),
      'unsupported_scope'
    );
    assert.equal(
      outcomeFromBackend(502, { detail: 'token_exchange_failed', reason: 'datacenter_mismatch' }),
      'datacenter_mismatch'
    );
    assert.equal(
      outcomeFromBackend(502, { detail: 'token_exchange_failed', reason: 'identity_unavailable' }),
      'identity_unavailable'
    );
    assert.equal(
      outcomeFromBackend(502, { detail: 'token_exchange_failed', reason: '<script>' }),
      'exchange_failed'
    );
    assert.equal(outcomeFromBackend(502, { detail: 'other', reason: 'insufficient_scope' }), 'exchange_failed');
    assert.equal(outcomeFromBackend(429, { detail: 'rate_limited' }), 'rate_limited');
    assert.equal(outcomeFromBackend(401, { detail: 'unauthorized' }), 'unauthorized');
    assert.equal(outcomeFromBackend(403, { detail: 'suspended' }), 'unauthorized');
    assert.equal(outcomeFromBackend(500, null), 'backend_unavailable');
    assert.equal(outcomeFromBackend(null, null), 'backend_unavailable');
    for (const outcome of OUTCOMES) assert.equal(isOutcome(outcome), true, outcome);
    assert.equal(isOutcome('nope'), false);
    assert.equal(isOutcome(undefined), false);
  });

  it('maps an authentication result to an outcome, or to nothing for a usable user', () => {
    const user = { kind: 'user', user: { userId: 'usr_' + 'a'.repeat(24) }, mustChangePassword: false };
    assert.equal(outcomeFromAuth(user as never), null);
    assert.equal(outcomeFromAuth({ ...user, mustChangePassword: true } as never), 'unauthorized');
    assert.equal(outcomeFromAuth({ kind: 'unconfigured' } as never), 'identity_not_configured');
    assert.equal(outcomeFromAuth({ kind: 'anonymous' } as never), 'not_signed_in');
    assert.equal(outcomeFromAuth({ kind: 'expired' } as never), 'not_signed_in');
    assert.equal(outcomeFromAuth({ kind: 'invalid' } as never), 'not_signed_in');
    assert.equal(outcomeFromAuth({ kind: 'denied', reason: 'suspended' } as never), 'unauthorized');
  });

  it('builds a result path that carries only the outcome and provider codes', () => {
    assert.equal(resultPath('connected', 'google'), '/connections/result?outcome=connected&provider=google');
    assert.equal(resultPath('not_signed_in', null), '/connections/result?outcome=not_signed_in');
  });

  it('describes every outcome, and is truthful that an unfinished exchange is not a connection', () => {
    for (const outcome of OUTCOMES) {
      const view = describeOutcome(outcome, 'google');
      assert.ok(view.title.length > 0 && view.detail.length > 0, outcome);
      assert.equal(view.connected, outcome === 'connected', outcome);
    }
    const unfinished = describeOutcome('token_exchange_unavailable', 'google');
    assert.match(unfinished.detail, /not connected/i);
    assert.match(unfinished.detail, /Google/);
    assert.match(unfinished.detail, /token exchange/i);
    assert.equal(describeOutcome('connected', 'microsoft').connected, true);
    assert.match(describeOutcome('connected', 'microsoft').title, /Microsoft/);
    assert.match(describeOutcome('denied', null).detail, /provider/i);
    assert.match(describeOutcome('not_initiated_here', null).detail, /Settings/);
    assert.match(describeOutcome('unsupported_scope', 'zoho').detail, /permission|scope/i);
    assert.match(describeOutcome('unsupported_scope', 'zoho').detail, /operator/i);
    assert.match(describeOutcome('datacenter_mismatch', 'zoho').detail, /data cent(er|re)/i);
    assert.match(describeOutcome('datacenter_mismatch', 'zoho').detail, /operator/i);
    assert.match(describeOutcome('identity_unavailable', 'google').detail, /which account|identity/i);
    assert.match(describeOutcome('token_exchange_unavailable', 'google').detail, /not connected/i);
    for (const outcome of ['unsupported_scope', 'datacenter_mismatch', 'identity_unavailable'] as const) {
      assert.equal(describeOutcome(outcome, null).connected, false);
      assert.match(describeOutcome(outcome, null).detail, /Nothing was connected/);
    }
  });
});

describe('panel view model', () => {
  const SECOND_ID = 'con_' + '1'.repeat(24);
  const providers = [
    { provider: 'google', display_name: 'Google', configured: true },
    { provider: 'microsoft', display_name: 'Microsoft', configured: true },
    { provider: 'zoho', display_name: 'Zoho', configured: false },
  ];
  const list = browserConnectionList({
    connections: [
      backendRow,
      { ...backendRow, connection_id: SECOND_ID, account_label: 'cesar.work@gmail.com', connected_at: NOW + 10 },
    ],
    providers,
    token_exchange_available: true,
  });

  it('shows one row per provider in a fixed order, listing every linked account', () => {
    assert.ok(list);
    const rows = panelRows(list);
    assert.deepEqual(
      rows.map((row) => [row.provider, row.label, row.status, row.canConnect]),
      [
        ['google', 'Google', 'connected', true],
        ['microsoft', 'Microsoft / Outlook', 'not_connected', true],
        ['zoho', 'Zoho', 'not_configured', false],
      ]
    );
    assert.deepEqual(
      rows[0].connections.map((c) => [c.connectionId, c.accountLabel]),
      [
        [CONNECTION_ID, 'ana@gmail.com'],
        [SECOND_ID, 'cesar.work@gmail.com'],
      ]
    );
    assert.deepEqual(rows[1].connections, []);
    assert.deepEqual(rows[2].connections, []);
  });

  it('offers Connect only when the backend can really complete the exchange', () => {
    const unavailable = browserConnectionList({ connections: [backendRow], providers });
    assert.ok(unavailable);
    assert.equal(unavailable.tokenExchangeAvailable, false);
    const rows = panelRows(unavailable);
    assert.deepEqual(
      rows.map((row) => [row.provider, row.status, row.configured, row.canConnect]),
      [
        ['google', 'connected', true, false],
        ['microsoft', 'not_connected', true, false],
        ['zoho', 'not_configured', false, false],
      ]
    );
  });

  it('explains connection errors in plain words, naming missing settings by variable name', () => {
    const text = explainConnectionError('configuration_needed', {
      missing: ['CAAL_OAUTH_ZOHO_CLIENT_ID', 'CAAL_OAUTH_ZOHO_CLIENT_SECRET'],
    });
    assert.ok(text);
    assert.match(text, /CAAL_OAUTH_ZOHO_CLIENT_ID/);
    assert.match(text, /CAAL_OAUTH_ZOHO_CLIENT_SECRET/);
    assert.ok(explainConnectionError('configuration_needed'));
    assert.ok(explainConnectionError('unknown_provider'));
    const scope = explainConnectionError('unsupported_scope', { settings: ['CAAL_OAUTH_ZOHO_SCOPES', 'nope', 7] });
    assert.ok(scope);
    assert.match(scope, /CAAL_OAUTH_ZOHO_SCOPES/);
    assert.ok(!scope.includes('nope'));
    assert.match(explainConnectionError('datacenter_mismatch', { settings: ['CAAL_OAUTH_ZOHO_ACCOUNTS_DOMAIN'] }) ?? '', /CAAL_OAUTH_ZOHO_ACCOUNTS_DOMAIN/);
    assert.ok(explainConnectionError('identity_unavailable'));
    assert.match(explainConnectionError('not_found') ?? '', /connection/i);
    assert.ok(!/user/i.test(explainConnectionError('not_found') ?? ''));
    assert.equal(explainConnectionError('nope'), null);
  });
});


describe('the names a user gives their own accounts', () => {
  it('carries the user names beside the provider label, never instead of it', () => {
    const named = browserConnection({
      ...backendRow,
      user_label: '  Work  ',
      aliases: ['Office', 'office', '', 'day job'],
    });
    assert.ok(named);
    assert.equal(named.accountLabel, 'ana@gmail.com');
    assert.equal(named.userLabel, 'Work');
    assert.deepEqual(named.aliases, ['Office', 'day job']);

    // An unusable name is dropped rather than failing the whole row.
    const messy = browserConnection({
      ...backendRow,
      user_label: 'line\nbreak',
      aliases: ['ok', 42, 'x'.repeat(200), null],
    });
    assert.ok(messy);
    assert.equal(messy.userLabel, null);
    assert.deepEqual(messy.aliases, ['ok']);
    const bare = browserConnection(backendRow);
    assert.equal(bare?.userLabel, null);
    assert.deepEqual(bare?.aliases, []);
  });

  it('folds case, spacing, accents and apostrophes into one key', () => {
    assert.equal(nameKey('  W\u00f6rk\u2019s  Mail '), 'works mail');
    assert.equal(nameKey('WORK'), nameKey('work'));
    assert.equal(nameKey('###'), '');
  });

  it('reads one comma-separated field into bounded, deduplicated names', () => {
    assert.deepEqual(parseAliasInput('work, Work ,  office ,'), ['work', 'office']);
    assert.deepEqual(parseAliasInput(''), []);
    assert.deepEqual(parseAliasInput('   ,  '), []);
    const many = parseAliasInput(Array.from({ length: 20 }, (_, i) => `n${i}`).join(','));
    assert.equal(many.length, MAX_ALIASES);
    assert.deepEqual(cleanNames('not a list'), []);
  });

  it('validates a submitted update the way the backend will, without echoing it', () => {
    const ok = accountNamesFrom({ user_label: ' work ', aliases: ['Office', 'office'] });
    assert.equal(ok.ok, true);
    if (ok.ok) {
      assert.deepEqual(ok.names, { userLabel: 'work', aliases: ['Office'] });
      assert.deepEqual(accountNamesRequest(ok.names), {
        user_label: 'work',
        aliases: ['Office'],
      });
    }
    // Clearing both names is a legitimate update.
    const cleared = accountNamesFrom({ user_label: '', aliases: [] });
    assert.equal(cleared.ok && cleared.names.userLabel, null);

    for (const bad of [
      { user_label: 'x'.repeat(49) },
      { user_label: 'line\nbreak' },
      { user_label: '-' },
      { user_label: 7 },
      { aliases: 'work' },
      { aliases: [7] },
      { aliases: ['x'.repeat(49)] },
      { aliases: ['#'] },
      null,
      'garbage',
    ]) {
      const refused = accountNamesFrom(bad);
      assert.equal(refused.ok, false, JSON.stringify(bad));
      if (!refused.ok) {
        assert.equal(refused.error, 'invalid_account_name');
        // The refusal is a code: nothing the user typed travels with it.
        assert.ok(!JSON.stringify(refused).includes('line'));
      }
    }
    const tooMany = accountNamesFrom({
      aliases: Array.from({ length: 40 }, (_, i) => `name${i}`),
    });
    assert.equal(tooMany.ok, false);
    if (!tooMany.ok) assert.equal(tooMany.error, 'too_many_names');
  });

  it('has plain words for every naming failure, and never repeats the input', () => {
    for (const code of ['invalid_account_name', 'too_many_names', 'no_change']) {
      const text = explainConnectionError(code);
      assert.ok(text && text.length > 10, code);
    }
  });
});
