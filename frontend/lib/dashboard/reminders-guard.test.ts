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
    assert.match(source, /No reminders yet/);
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
