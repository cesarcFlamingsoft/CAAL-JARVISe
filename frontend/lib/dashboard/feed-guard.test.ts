import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

/**
 * Structural guard for the dashboard feed BFF routes, the hook and the two
 * widgets that show connected-account data. These files import Next.js and
 * React, so they cannot run under node --test; what can be checked is that
 * every route keeps the identity BFF's posture, that the browser only ever
 * sees the parsed shapes, and that the widgets stay honest about each
 * account's state and never gate on the voice session.
 */

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = (path: string) => readFileSync(join(ROOT, path), 'utf8');

const ROUTES = {
  calendar: 'app/api/dashboard/calendar/route.ts',
  inbox: 'app/api/dashboard/inbox/route.ts',
};
const HOOK = 'hooks/useDashboardFeed.ts';
const CALENDAR_WIDGET = 'components/dashboard/widgets/calendar-widget.tsx';
const INBOX_WIDGET = 'components/dashboard/widgets/inbox-widget.tsx';
const ISSUES = 'components/dashboard/account-issues.tsx';
const SECTION = 'components/dashboard/account-section.tsx';
const WORKSPACE = 'components/dashboard/workspace.tsx';
const PANEL = 'components/settings/connected-accounts.tsx';
const SESSION_READS = ['useSessionContext', 'useVoiceAssistant', 'useRoomContext', 'isConnected'];

describe('dashboard feed BFF routes', () => {
  it('exist, authenticate the browser session, and proxy as that user only', () => {
    for (const [key, path] of Object.entries(ROUTES)) {
      assert.ok(existsSync(join(ROOT, path)), path + ' is missing');
      const source = read(path);
      assert.match(source, /requireUser\(req/, key + ' must use requireUser');
      assert.match(
        source,
        /callAsUser\(\s*auth\.config,\s*auth\.user\.userId,\s*'\/users\/me\/dashboard\//,
        key + ' must call the backend as the session user'
      );
      assert.match(source, /export const dynamic = 'force-dynamic'/, key);
      assert.match(source, /export async function GET\(/, key);
      assert.ok(
        !/export async function (POST|PATCH|PUT|DELETE)\(/.test(source),
        key + ' is read-only'
      );
      assert.ok(!source.includes('/admin/'), key);
      assert.match(source, /noStoreJson\(/, key);
      assert.match(source, /feedQuery\(/, key + ' must validate its query parameters');
      const budget = /FEED_TIMEOUT_MS = (\d[\d_]*)/.exec(source);
      assert.ok(budget, key + ' must declare its own timeout');
      const millis = Number(budget![1].replace(/_/g, ''));
      assert.ok(millis >= 20_000 && millis <= 60_000, budget![1]);
      assert.match(source, /timeoutMs: FEED_TIMEOUT_MS/, key);
    }
    assert.match(read(ROUTES.calendar), /browserCalendarFeed\(/);
    assert.match(read(ROUTES.inbox), /browserInboxFeed\(/);
  });
});

describe('dashboard feed hook', () => {
  it('fetches the BFF with the session cookie, never caches, and refreshes when accounts change', () => {
    const source = read(HOOK);
    assert.match(source, /'\/api\/dashboard\//);
    assert.match(source, /credentials: 'same-origin'/);
    assert.match(source, /cache: 'no-store'/);
    assert.match(source, /'connections-updated'/);
    assert.match(source, /'settings-updated'/);
    assert.match(source, /identity_not_configured/);
    for (const name of SESSION_READS) assert.ok(!source.includes(name), name);
  });
});

describe('calendar and inbox widgets', () => {
  it('render parsed feed items only, report each account state, and never gate on the call', () => {
    const calendar = read(CALENDAR_WIDGET);
    assert.match(calendar, /groupEventsByDay\(/);
    assert.match(calendar, /accountIssues\(/);
    const inbox = read(INBOX_WIDGET);
    assert.match(inbox, /accountIssues\(/);
    // Each account's state is reported by its own section, never feed-wide.
    for (const source of [calendar, inbox]) assert.match(source, /<AccountSection\b/);
    assert.match(read(SECTION), /<AccountIssues\b/);
    assert.match(read(ISSUES), /describeAccountStatus\(/);
    assert.match(inbox, /rel=.noopener noreferrer./);
    assert.match(inbox, /target=._blank./);
    for (const source of [calendar, inbox]) {
      assert.ok(!source.includes('dangerouslySetInnerHTML'));
      for (const name of SESSION_READS) assert.ok(!source.includes(name), name);
      assert.ok(!/\bpassword\b/i.test(source));
    }
  });

  it('group every item under the account it came from, never one mixed stream', () => {
    const calendar = read(CALENDAR_WIDGET);
    const inbox = read(INBOX_WIDGET);
    assert.match(calendar, /groupEventsByAccount\(/, 'events must be grouped per account');
    assert.match(calendar, /groupEventsByDay\(group\.events/, 'days are grouped within an account');
    assert.match(inbox, /groupMessagesByAccount\(/, 'messages must be grouped per account');
    for (const source of [calendar, inbox]) {
      assert.match(source, /<AccountSection\b/, 'each account gets its own labelled section');
      assert.ok(
        !/\bfeed\.data\.(events|messages)\.map\(/.test(source),
        'no widget may render the merged feed list directly'
      );
    }
  });

  it('keeps each account section owning its label, count and state', () => {
    assert.ok(existsSync(join(ROOT, SECTION)), SECTION + ' is missing');
    const section = read(SECTION);
    assert.match(section, /accountName\(/, 'a section is titled by its own account');
    assert.match(section, /<AccountIssues\b/, 'a failing account keeps its own state and remedy');
    assert.match(section, /issues=\{/, 'the issue state is per account, not feed-wide');
    assert.match(section, /count/, 'a section carries its own item count');
    // Movable widgets are sized by their frame: a section may never pin a width.
    assert.ok(!/\bw-\[\d/.test(section), 'sections must stay fluid inside the widget frame');
    assert.ok(!/\bmin-w-\[\d/.test(section), 'sections must stay fluid inside the widget frame');
    for (const name of SESSION_READS) assert.ok(!section.includes(name), name);
  });

  it('are mounted in the workspace with the inbox as a movable widget', () => {
    const workspace = read(WORKSPACE);
    assert.match(workspace, /case 'inbox':/);
    assert.match(workspace, /<InboxWidget\b/);
    assert.match(workspace, /<CalendarWidget\b/);
    assert.match(workspace, /useDashboardFeed\(/);
  });

  it('is told when an account is disconnected in Settings', () => {
    assert.match(read(PANEL), /'connections-updated'/);
  });

  it('builds on the existing dependency set only', () => {
    const pkg = JSON.parse(read('package.json')) as {
      dependencies: Record<string, string>;
      devDependencies: Record<string, string>;
    };
    const declared = new Set([
      ...Object.keys(pkg.dependencies),
      ...Object.keys(pkg.devDependencies),
    ]);
    const sources = [
      ...Object.values(ROUTES),
      HOOK,
      CALENDAR_WIDGET,
      INBOX_WIDGET,
      ISSUES,
      'lib/dashboard/provider-data.ts',
    ];
    for (const path of sources) {
      const imports = [...read(path).matchAll(/from '([^'.@][^']*|@[^/']+\/[^/']+)'/g)].map(
        (m) => m[1]
      );
      for (const spec of imports) {
        const pkgName = spec.startsWith('@')
          ? spec.split('/').slice(0, 2).join('/')
          : spec.split('/')[0];
        if (pkgName === 'react' || pkgName === 'next') continue;
        assert.ok(
          declared.has(pkgName),
          path + ' imports ' + pkgName + ', which is not a declared dependency'
        );
      }
    }
  });
});
