import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

/**
 * Structural guard for the reminder BFF routes and the widget that shows
 * them. These files import Next.js and React, so they cannot run under
 * node --test; what can be checked is that the routes keep the identity BFF
 * posture, that the browser only ever sees the parsed shape, that the one
 * thing a page can change is the default for future reminders, and that
 * nothing in the widget gates a private list on a voice session or offers a
 * delivery for a reminder with no time on it.
 */

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = (path: string) => readFileSync(join(ROOT, path), 'utf8');

const FEED_ROUTE = 'app/api/dashboard/reminders/route.ts';
const DELIVERY_ROUTE = 'app/api/dashboard/reminders/delivery/route.ts';
const WIDGET = 'components/dashboard/widgets/reminders-widget.tsx';
const WORKSPACE = 'components/dashboard/workspace.tsx';
const HOOK = 'hooks/useDashboardFeed.ts';
const SESSION_READS = ['useSessionContext', 'useVoiceAssistant', 'useRoomContext', 'isConnected'];

describe('the reminders feed route', () => {
  it('exists, authenticates the browser session, and proxies as that user only', () => {
    assert.ok(existsSync(join(ROOT, FEED_ROUTE)), FEED_ROUTE + ' is missing');
    const source = read(FEED_ROUTE);
    assert.match(source, /requireUser\(req/);
    assert.match(
      source,
      /callAsUser\(auth\.config, auth\.user\.userId, '\/users\/me\/dashboard\/reminders'/
    );
    assert.match(source, /export const dynamic = 'force-dynamic'/);
    assert.match(source, /noStoreJson\(/);
    assert.match(source, /browserReminders\(/);
  });

  it('is read-only, and never reaches an admin surface', () => {
    const source = read(FEED_ROUTE);
    assert.ok(!/export async function (POST|PATCH|PUT|DELETE)\(/.test(source));
    assert.ok(!source.includes('/admin/'));
  });

  it('never forwards a query parameter that could widen the scope', () => {
    assert.ok(!read(FEED_ROUTE).includes('searchParams'));
  });

  it('records the branch where the backend answered but the answer was unreadable', () => {
    // That branch is why the dashboard once said "the backend did not answer"
    // while the backend was answering, with nothing in any log to say so.
    const source = read(FEED_ROUTE);
    const logs = source.match(/console\.(error|warn|info|log)\([^;]*\);/g) ?? [];
    assert.equal(logs.length, 1, 'the unreadable-answer branch must leave exactly one trace');
    assert.match(logs[0], /GET \/users\/me\/dashboard\/reminders/);
    assert.match(logs[0], /result\.status/);
    // A reminder is private: the trace may carry the call and its outcome and
    // nothing that belongs to the person who made it.
    for (const forbidden of [
      'result.data',
      'feed',
      'auth.user',
      'userId',
      'title',
      'JSON.stringify',
    ]) {
      assert.ok(!logs[0].includes(forbidden), forbidden + ' must never be logged');
    }
  });
});

describe('the delivery default route', () => {
  it('guards every mutation and validates the body before the backend sees it', () => {
    assert.ok(existsSync(join(ROOT, DELIVERY_ROUTE)), DELIVERY_ROUTE + ' is missing');
    const source = read(DELIVERY_ROUTE);
    assert.match(source, /export async function GET\(/);
    assert.match(source, /export async function PUT\(/);
    assert.match(source, /requireUser\(req/);
    assert.match(source, /guardMutation\(req, auth\.config, auth\.user\.userId\)/);
    assert.match(source, /deliveryChoiceRequest\(body\.delivery\)/);
    assert.match(source, /apiError\(422, 'invalid'\)/);
    assert.match(source, /browserDeliveryDefaults\(/);
  });

  it('has no way to name an owner, a number or a chat', () => {
    // The prose explains why; the code is what must not contain one.
    const code = read(DELIVERY_ROUTE).replace(/\/\*[\s\S]*?\*\//g, '');
    for (const forbidden of ['userId:', 'user_id', 'phone', 'chat_id', 'destination']) {
      assert.ok(!code.includes(forbidden), forbidden + ' must not appear');
    }
  });

  it('never deletes or creates a reminder from the browser', () => {
    const source = read(DELIVERY_ROUTE);
    assert.ok(!/export async function (POST|DELETE|PATCH)\(/.test(source));
  });
});

describe('the reminders widget', () => {
  const source = read(WIDGET);

  it('renders every honest state and never a placeholder list', () => {
    for (const state of ['loading', 'unauthorized', 'unconfigured', 'error']) {
      assert.ok(source.includes("'" + state + "'"), state + ' state is missing');
    }
    assert.match(source, /WidgetLoading/);
    assert.match(source, /WidgetSignIn/);
    assert.match(source, /WidgetError/);
    // The empty state covers the whole widget now: no reminders *and* no
    // alarms. It still says so plainly rather than showing a placeholder.
    assert.match(source, /Nothing scheduled yet/);
    assert.match(source, /reminders\.length === 0 && alarms\.length === 0/);
  });

  it('no longer claims the list is missing from the dashboard', () => {
    assert.ok(!source.includes('not exposed to the dashboard'));
    assert.ok(!source.includes('WidgetBlocked'));
  });

  it('shows the delivery of a timed reminder and never of an undated one', () => {
    assert.match(source, /if \(!reminder\.timed\) \{/);
    assert.match(source, /List item, no alert/);
    assert.match(source, /dueLabel\(reminder\.due, now\)/);
  });

  it('tells a reminder that has been and gone from one that is not armed', () => {
    assert.match(source, /Nothing left to deliver/);
    assert.match(source, /No delivery armed/);
  });

  it('says when a call or a message actually happens', () => {
    assert.match(source, /comes due, not now/);
  });

  it('does not gate a private list on a voice session', () => {
    for (const read of SESSION_READS) {
      assert.ok(!source.includes(read), read + ' must not gate the reminders widget');
    }
  });
});

describe('the workspace', () => {
  it('feeds the widget from the BFF and keeps it on the movable grid', () => {
    const source = read(WORKSPACE);
    assert.match(source, /useDashboardFeed\('\/api\/dashboard\/reminders', browserReminders\)/);
    assert.match(source, /<RemindersWidget feed=\{reminders\}/);
    assert.match(source, /case 'reminders':/);
  });

  it('teaches the shared feed hook the new path, so a refresh picks it up', () => {
    assert.match(read(HOOK), /'\/api\/dashboard\/reminders'/);
    assert.match(read(HOOK), /REFRESH_INTERVAL_MS/);
  });
});

describe('the reminders widget shows the alarms and timers too', () => {
  it('renders them as their own labelled kind, with their own due time and state', () => {
    const source = read(WIDGET);
    assert.match(source, /feed\.data\.alarms|alarms\b/, 'the widget must read the alarm list');
    assert.match(source, /ALARM_KIND_LABELS/, 'an alarm and a timer must say which they are');
    assert.match(source, /ALARM_STATE_LABELS/, 'delivered, missed and waiting must read as such');
    assert.match(source, /dueLabel\(/, 'an alarm must show when it is due, in the reader zone');
    // One truthful view: the two lists are shown side by side, never merged
    // into invented rows and never claimed to be each other.
    assert.ok(!source.includes('synthetic'));
    for (const name of SESSION_READS) assert.ok(!source.includes(name), name);
  });

  it('never offers a delivery choice for an alarm', () => {
    const source = read(WIDGET);
    const start = source.indexOf('function AlarmList');
    const alarms = source.slice(start, source.indexOf('\nfunction ', start + 1));
    assert.ok(alarms.length > 0, 'the alarm list must be its own component');
    for (const name of ['CHANNEL_CHOICES', 'apiRequest', 'PUT']) {
      assert.ok(!alarms.includes(name), name + ' must not appear in the alarm list');
    }
  });
});
