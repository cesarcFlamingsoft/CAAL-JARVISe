import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

/**
 * Structural guard for the connections BFF and its panel. These files import
 * Next.js and React, so they cannot be executed under `node --test`; what can
 * be checked is that every route keeps the posture the rest of the identity
 * BFF has, and that the panel keeps the promises the slice makes: confirm
 * before disconnecting, connect only when the provider is configured, never a
 * password field, and never a claim of "connected" this build cannot back.
 */

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = (path: string) => readFileSync(join(ROOT, path), 'utf8');

const ROUTES = {
  list: 'app/api/connections/route.ts',
  start: 'app/api/connections/start/[provider]/route.ts',
  disconnect: 'app/api/connections/[connectionId]/route.ts',
  callback: 'app/api/connections/callback/route.ts',
};
const PANEL = 'components/settings/connected-accounts.tsx';
const SETTINGS = 'components/settings/settings-panel.tsx';
const RESULT_PAGE = 'app/(app)/connections/result/page.tsx';

describe('connections BFF routes', () => {
  it('exist where the backend fixes the callback and where the panel calls them', () => {
    for (const path of Object.values(ROUTES)) {
      assert.ok(existsSync(join(ROOT, path)), `${path} is missing`);
    }
    // The backend builds every provider's redirect_uri from this exact path.
    const python = join(ROOT, '..', 'src', 'caal', 'oauth_providers.py');
    if (existsSync(python)) {
      assert.match(readFileSync(python, 'utf8'), /OAUTH_CALLBACK_PATH = "\/api\/connections\/callback"/);
    }
  });

  it('authenticate the browser session and proxy to the backend as that user only', () => {
    for (const key of ['list', 'start', 'disconnect'] as const) {
      const source = read(ROUTES[key]);
      assert.match(source, /requireUser\(req/, `${key} must use requireUser`);
      assert.match(source, /callAsUser\(\s*auth\.config,\s*auth\.user\.userId/, `${key} must call as the session user`);
      assert.match(source, /['`]\/users\/me\/connections/, `${key} must target the user's own connections`);
      assert.ok(!source.includes('/admin/'), `${key} must not reach admin routes`);
      assert.match(source, /export const dynamic = 'force-dynamic'/);
    }
    const callback = read(ROUTES.callback);
    assert.match(callback, /authenticate\(req\.headers\)/);
    assert.match(
      callback,
      /callAsUser\(\s*auth\.config,\s*auth\.user\.userId,\s*'\/users\/me\/connections\/callback'/
    );
    assert.match(callback, /export const dynamic = 'force-dynamic'/);
  });

  it('guard every mutation with the origin, CSRF and rate-limit checks', () => {
    for (const key of ['start', 'disconnect'] as const) {
      const source = read(ROUTES[key]);
      const guards = source.match(/guardMutation\(req, auth\.config, auth\.user\.userId\)/g) ?? [];
      const mutations = source.match(/export async function (POST|PATCH|PUT|DELETE)\(/g) ?? [];
      assert.equal(guards.length, mutations.length, `${key} must guard every mutation`);
      assert.ok(guards.length > 0, key);
    }
    assert.match(read(ROUTES.start), /isProvider\(/);
    assert.match(read(ROUTES.disconnect), /isConnectionId\(/);
    assert.match(read(ROUTES.disconnect), /export async function DELETE\(/);
    // Renaming an account is the one other thing that route does, and it is a
    // mutation on the caller's own connection like the delete beside it.
    const disconnect = read(ROUTES.disconnect);
    assert.match(disconnect, /export async function PATCH\(/);
    assert.match(disconnect, /accountNamesFrom\(/);
    assert.match(disconnect, /accountNamesRequest\(/);
    assert.match(disconnect, /readJsonObject\(req\)/);
    // The names are validated here; the backend body carries nothing else.
    assert.ok(!/user_label:\s*body/.test(disconnect));
    assert.ok(!/export async function (GET|POST|PUT)\(/.test(disconnect));
    assert.ok(!/export async function (GET|PATCH|PUT|DELETE)\(/.test(read(ROUTES.start)));
    assert.ok(!/export async function (POST|PATCH|PUT|DELETE)\(/.test(read(ROUTES.list)));
  });

  it('bind the callback to the browser session that started the flow, and never render the code', () => {
    const start = read(ROUTES.start);
    assert.match(start, /sealFlow\(/);
    assert.match(start, /flowCookieOptions\(/);
    assert.match(start, /parseAuthorization\(/);
    assert.match(start, /configurationNeeded\(/);
    assert.match(start, /sessionKeyFor\(/);

    const callback = read(ROUTES.callback);
    assert.match(callback, /export async function GET\(/);
    assert.ok(!/export async function (POST|PATCH|PUT|DELETE)\(/.test(callback));
    assert.match(callback, /parseCallbackQuery\(/);
    assert.match(callback, /openFlow\(/);
    assert.match(callback, /sessionKeyFor\(/);
    assert.match(callback, /clearedFlowCookie\(/);
    assert.match(callback, /resultPath\(/);
    assert.match(callback, /outcomeFromBackend\(/);
    assert.match(callback, /outcomeFromAuth\(/);
    assert.match(callback, /status: 303/);
    assert.ok(!callback.includes('error_description'), 'provider error text must never be read');
    assert.match(callback, /location/, 'Zoho’s data-center code must reach the backend so the exchange fails closed');
    assert.ok(!callback.includes('accounts-server'), 'a provider-supplied host must never be trusted');
    // The backend's exchange may take two bounded provider round trips; the
    // BFF must wait at least that long or it would report a completed
    // connection as "backend did not respond".
    const budget = /CALLBACK_TIMEOUT_MS = (\d[\d_]*)/.exec(callback);
    assert.ok(budget, 'the callback must declare its own timeout');
    const millis = Number(budget![1].replace(/_/g, ''));
    assert.ok(millis >= 20_000 && millis <= 60_000, budget![1]);
    assert.match(callback, /timeoutMs: CALLBACK_TIMEOUT_MS/);
    assert.ok(!/NextResponse\.json\(/.test(callback) || !callback.includes('code'), 'the code must not be echoed');
  });

  it('never model a password or an app password anywhere in the slice', () => {
    // The word itself (a field, a body key, a label), not the identity BFF's
    // camel-cased forced-change flag that the callback must honour.
    for (const path of [...Object.values(ROUTES), PANEL, RESULT_PAGE, 'lib/connections/protocol.ts', 'lib/connections/flow-cookie.ts']) {
      const source = read(path);
      assert.ok(!/\bpassword\b/i.test(source), `${path} must not mention a password`);
      assert.ok(!/app_password|imap_password|smtp_password/.test(source), path);
      assert.ok(!source.includes('type="password"'), path);
    }
  });
});

describe('connected accounts panel', () => {
  it('is mounted inside Settings → Integrations', () => {
    const settings = read(SETTINGS);
    assert.match(settings, /import \{ ConnectedAccounts \} from '@\/components\/settings\/connected-accounts'/);
    const start = settings.indexOf('const renderIntegrationsTab');
    const end = settings.indexOf('const renderWakeWordTab');
    assert.ok(start > 0 && end > start, 'integrations tab must still exist');
    assert.ok(settings.slice(start, end).includes('<ConnectedAccounts'), 'panel must render in the tab');
  });

  it('confirms before disconnecting, enables Connect only for configured providers, and is honest about the exchange', () => {
    const panel = read(PANEL);
    assert.match(panel, /disabled=\{!row\.canConnect/);
    assert.match(panel, /confirming/);
    assert.equal((panel.match(/method: 'DELETE'/g) ?? []).length, 1, 'exactly one DELETE call');
    assert.match(panel, /async function confirmDisconnect[\s\S]*?method: 'DELETE'/);
    assert.ok(!panel.includes('window.confirm'), 'use an explicit in-page confirmation');
    assert.match(panel, /tokenExchangeAvailable/);
    assert.ok(!panel.includes('type="password"'));
    // Several accounts per provider: each linked account is listed and can be
    // disconnected on its own, and another can be connected alongside.
    assert.match(panel, /row\.connections\.map\(/);
    assert.match(panel, /Connect another/);
    assert.match(panel, /panelRows\(/);
    assert.match(panel, /explainConnectionError\(/);
    assert.match(panel, /'\/api\/connections'/);
    assert.match(panel, /\/api\/connections\/start\//);
  });

  it('lets the owner name each account, and says what the names are for', () => {
    const panel = read(PANEL);
    // One PATCH, on the caller's own connection, with the checked names only.
    assert.equal((panel.match(/method: 'PATCH'/g) ?? []).length, 1, 'exactly one PATCH call');
    assert.match(panel, /async function saveNames[\s\S]*?method: 'PATCH'/);
    assert.match(panel, /accountNamesFrom\(/);
    assert.match(panel, /parseAliasInput\(/);
    // The control is reachable and labelled, and cancelling is offered.
    assert.match(panel, /aria-expanded=\{isNaming\}/);
    assert.match(panel, /htmlFor=/);
    assert.match(panel, /Name this account/);
    assert.match(panel, /Edit names/);
    assert.match(panel, /Cancel/);
    assert.match(panel, /role="alert"/);
    // The promise the slice makes to the user, in the panel's own words.
    assert.match(panel, /provider account identity remains unchanged/i);
    // Connect and Disconnect are still there, and no token ever is.
    assert.match(panel, /Disconnect/);
    assert.match(panel, /Connect another/);
    assert.ok(!/accessToken|access_token|refresh/i.test(panel));
  });

  it('renders only a validated outcome on the result page', () => {
    const page = read(RESULT_PAGE);
    assert.match(page, /isOutcome\(/);
    assert.match(page, /isProvider\(/);
    assert.match(page, /describeOutcome\(/);
    assert.match(page, /export const dynamic = 'force-dynamic'/);
  });

  it('builds on the existing dependency set only', () => {
    const pkg = JSON.parse(read('package.json')) as {
      dependencies: Record<string, string>;
      devDependencies: Record<string, string>;
    };
    const declared = new Set([...Object.keys(pkg.dependencies), ...Object.keys(pkg.devDependencies)]);
    for (const path of [...Object.values(ROUTES), PANEL, RESULT_PAGE, 'lib/connections/protocol.ts', 'lib/connections/flow-cookie.ts']) {
      const imports = [...read(path).matchAll(/from '([^'.@][^']*|@[^/']+\/[^/']+)'/g)].map((m) => m[1]);
      for (const spec of imports) {
        const pkgName = spec.startsWith('@') ? spec.split('/').slice(0, 2).join('/') : spec.split('/')[0];
        if (pkgName === 'react' || pkgName === 'next') continue;
        assert.ok(declared.has(pkgName), `${path} imports ${pkgName}, which is not a declared dependency`);
      }
    }
  });
});
