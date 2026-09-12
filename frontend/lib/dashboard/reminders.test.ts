import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  CHANNEL_LABELS,
  DELIVERY_CHANNELS,
  browserDeliveryDefaults,
  browserReminders,
  deliveryChoiceRequest,
  deliverySummary,
  dueLabel,
  orderChannels,
} from './reminders.ts';

/**
 * The browser whole view of the reminder slice. Everything a page renders
 * passes through here first: a payload that is not exactly the backend own
 * shape becomes null rather than a half-parsed object, and nothing a browser
 * sends reaches the backend without being checked here as well.
 */

const ID = '11111111-2222-4333-8444-555555555555';
const OTHER = '99999999-2222-4333-8444-555555555555';

const REMINDER = {
  id: ID,
  title: 'Call the clinic',
  due: '2023-11-14T23:30:00Z',
  timed: true,
  list_name: 'Reminders',
  notes: 'ask about the results',
  completed: false,
  created_at: '2023-11-14T22:00:00+00:00',
  delivery: [
    { channel: 'telegram', state: 'pending' },
    { channel: 'speak', state: 'delivered' },
  ],
};

const FEED = {
  generated_at: 1_700_000_000,
  reminders: [REMINDER],
  available: ['speak', 'telegram'],
  defaults: ['speak'],
};

describe('browserReminders', () => {
  it('parses the backend feed into the shape a page renders', () => {
    const feed = browserReminders(FEED);
    assert.ok(feed);
    assert.equal(feed.generatedAt, 1_700_000_000);
    assert.equal(feed.reminders.length, 1);
    const item = feed.reminders[0];
    assert.equal(item.id, ID);
    assert.equal(item.title, 'Call the clinic');
    assert.equal(item.timed, true);
    assert.equal(item.list, 'Reminders');
    assert.deepEqual(feed.available, ['speak', 'telegram']);
    assert.deepEqual(feed.defaults, ['speak']);
  });

  it('puts the channels in one order however they arrive', () => {
    const feed = browserReminders(FEED);
    assert.deepEqual(
      feed!.reminders[0].delivery.map((entry) => entry.channel),
      ['speak', 'telegram']
    );
    assert.deepEqual(
      feed!.reminders[0].delivery.map((entry) => entry.state),
      ['delivered', 'pending']
    );
  });

  it('is null for anything that is not the backend shape', () => {
    assert.equal(browserReminders(null), null);
    assert.equal(browserReminders([]), null);
    assert.equal(browserReminders({ reminders: [] }), null);
    assert.equal(browserReminders({ generated_at: 1, reminders: {} }), null);
  });

  it('drops a reminder it cannot read rather than rendering a half-parsed one', () => {
    const feed = browserReminders({
      ...FEED,
      reminders: [{ ...REMINDER, id: 'not-an-id' }, { ...REMINDER, id: OTHER, title: '  ' }, 7],
    });
    assert.deepEqual(feed!.reminders, []);
  });

  it('never shows a delivery on a reminder with no time on it', () => {
    const feed = browserReminders({
      ...FEED,
      reminders: [{ ...REMINDER, due: null, timed: false }],
    });
    assert.equal(feed!.reminders[0].timed, false);
    assert.equal(feed!.reminders[0].due, null);
    assert.deepEqual(feed!.reminders[0].delivery, []);
  });

  it('drops an unknown channel or state instead of showing it', () => {
    const feed = browserReminders({
      ...FEED,
      reminders: [
        {
          ...REMINDER,
          delivery: [
            { channel: 'sms', state: 'pending' },
            { channel: 'call', state: 'exploded' },
            { channel: 'call', state: 'failed' },
          ],
        },
      ],
    });
    assert.deepEqual(feed!.reminders[0].delivery, [{ channel: 'call', state: 'failed' }]);
  });

  it('strips control characters and bounds every text field', () => {
    const feed = browserReminders({
      ...FEED,
      reminders: [{ ...REMINDER, title: 'a' + String.fromCharCode(10) + 'b'.repeat(500) }],
    });
    const title = feed!.reminders[0].title;
    assert.ok(title.length <= 200);
    assert.ok(!title.includes(String.fromCharCode(10)));
  });
});

/**
 * The path the deployed dashboard actually takes, end to end, with no browser
 * and no session: the BFF route reduces the backend answer through
 * `browserReminders` and serialises *that*, and the feed hook then parses the
 * body it receives through `browserReminders` again before the widget sees
 * it. A parser that only understands the backend own spelling therefore reads
 * its own answer as malformed and the widget says the backend did not answer,
 * even though the backend answered perfectly. These tests stand in for that
 * two-step and would have caught it.
 */
describe('the BFF answer, re-parsed by the browser', () => {
  /** What `noStoreJson(feed)` puts on the wire and `response.json()` returns. */
  const overTheWire = (value: unknown) => JSON.parse(JSON.stringify(value));

  it('is still a feed after the round trip the dashboard performs', () => {
    const served = browserReminders(FEED);
    assert.ok(served, 'the route could not parse the backend feed');
    const rendered = browserReminders(overTheWire(served));
    assert.ok(rendered, 'the widget read the BFF answer as malformed');
    assert.deepEqual(rendered, served, 'the round trip changed the feed');
  });

  it('keeps every field of a reminder the browser renders', () => {
    const backend = { ...FEED, reminders: [{ ...REMINDER, list_name: 'Groceries' }] };
    const served = browserReminders(backend)!;
    const rendered = browserReminders(overTheWire(served))!;
    const item = rendered.reminders[0];
    assert.equal(rendered.generatedAt, 1_700_000_000);
    assert.equal(item.id, ID);
    assert.equal(item.title, 'Call the clinic');
    assert.equal(item.due, '2023-11-14T23:30:00Z');
    assert.equal(item.timed, true);
    // The list name is the reminder own, not the fallback a dropped field
    // would leave behind.
    assert.equal(item.list, 'Groceries');
    assert.equal(item.notes, 'ask about the results');
    assert.deepEqual(item.delivery, [
      { channel: 'speak', state: 'delivered' },
      { channel: 'telegram', state: 'pending' },
    ]);
    assert.deepEqual(rendered.available, ['speak', 'telegram']);
    assert.deepEqual(rendered.defaults, ['speak']);
  });

  it('is still refused when the answer is not a feed at all', () => {
    // Tolerating the browser own spelling must not become tolerating
    // anything: a drifted or empty backend answer has to stay an error.
    assert.equal(browserReminders({ generatedAt: 1, reminders: {} }), null);
    assert.equal(browserReminders({ generatedAt: 'soon', reminders: [] }), null);
    assert.equal(browserReminders({ reminders: [] }), null);
    assert.equal(browserReminders({ generated_at: 1 }), null);
  });
});

describe('dueLabel', () => {
  const now = new Date('2023-11-14T22:13:20Z');

  it('reads as the wall clock of the reader own timezone', () => {
    assert.equal(dueLabel('2023-11-14T23:30:00Z', now, 'UTC'), 'Today 23:30');
    assert.equal(dueLabel('2023-11-14T23:30:00Z', now, 'America/Mexico_City'), 'Today 17:30');
    // Already the 15th in Tokyo, for the reader and for the reminder alike.
    assert.equal(dueLabel('2023-11-14T23:30:00Z', now, 'Asia/Tokyo'), 'Today 08:30');
    assert.equal(dueLabel('2023-11-15T23:30:00Z', now, 'UTC'), 'Tomorrow 23:30');
  });

  it('names a day further out', () => {
    assert.equal(dueLabel('2023-11-20T09:00:00Z', now, 'UTC'), 'Mon 20 Nov, 09:00');
  });

  it('says so when the time has already passed', () => {
    assert.match(dueLabel('2023-11-14T20:00:00Z', now, 'UTC'), /^Overdue/);
  });

  it('is not a time at all when there is none', () => {
    assert.equal(dueLabel(null, now, 'UTC'), 'No time');
    assert.equal(dueLabel('whenever', now, 'UTC'), 'No time');
  });
});

describe('the delivery choice', () => {
  it('accepts any combination of the fixed vocabulary, in one order', () => {
    assert.deepEqual(deliveryChoiceRequest(['call', 'speak']), { delivery: ['speak', 'call'] });
    assert.deepEqual(deliveryChoiceRequest([...DELIVERY_CHANNELS]), {
      delivery: ['speak', 'telegram', 'call'],
    });
  });

  it('refuses anything that is not a channel, including a destination', () => {
    for (const value of [null, 'speak', [], ['sms'], ['speak', 'sms'], ['+15551230000'], [7]]) {
      assert.equal(deliveryChoiceRequest(value), null, JSON.stringify(value));
    }
  });

  it('parses the saved default, and never invents one', () => {
    assert.deepEqual(browserDeliveryDefaults({ delivery: ['speak'], available: [], saved: true }), {
      delivery: ['speak'],
      available: [],
      saved: true,
    });
    assert.equal(browserDeliveryDefaults({ delivery: ['speak'] }), null);
    assert.equal(browserDeliveryDefaults(null), null);
  });

  it('orders and deduplicates a set of channels', () => {
    assert.deepEqual(orderChannels(['call', 'speak', 'call', 'nope']), ['speak', 'call']);
  });
});

describe('deliverySummary', () => {
  it('reads as a sentence and never as a destination', () => {
    assert.equal(deliverySummary([]), '');
    assert.equal(deliverySummary([{ channel: 'call', state: 'pending' }]), CHANNEL_LABELS.call);
    assert.equal(
      deliverySummary([
        { channel: 'speak', state: 'pending' },
        { channel: 'telegram', state: 'pending' },
        { channel: 'call', state: 'pending' },
      ]),
      'Spoken here, Telegram and Phone call'
    );
  });
});

/**
 * Alarms and timers ride in on the same feed as the reminders and stay their
 * own kind of thing all the way to the page: the widget shows one truthful
 * list, never one list pretending the two are the same.
 */
describe('browserReminders and the alarms beside them', () => {
  const ALARM = {
    id: OTHER,
    label: 'Wake up',
    kind: 'alarm',
    due: '2023-11-15T07:00:00Z',
    state: 'pending',
  };

  const withAlarms = (alarms: unknown) => browserReminders({ ...FEED, alarms });

  it('reads the alarms and timers of this owner', () => {
    const feed = withAlarms([ALARM, { ...ALARM, id: ID, label: 'Laundry', kind: 'timer' }]);
    assert.ok(feed);
    assert.deepEqual(
      feed!.alarms.map((item) => [item.label, item.kind, item.state]),
      [
        ['Wake up', 'alarm', 'pending'],
        ['Laundry', 'timer', 'pending'],
      ]
    );
  });

  it('is an empty list, not an error, for a backend that sent none', () => {
    const feed = browserReminders(FEED);
    assert.ok(feed);
    assert.deepEqual(feed!.alarms, []);
  });

  it('drops an alarm that is not exactly the shape it claims to be', () => {
    const refused = [
      { ...ALARM, id: 'not-a-uuid' },
      { ...ALARM, label: '' },
      { ...ALARM, kind: 'reminder' },
      { ...ALARM, kind: 'anything' },
      { ...ALARM, state: 'sent' },
      { ...ALARM, due: null },
      { ...ALARM, due: 'whenever' },
      'a string',
      null,
    ];
    for (const entry of refused) {
      const feed = withAlarms([entry]);
      assert.ok(feed, JSON.stringify(entry));
      assert.deepEqual(feed!.alarms, [], JSON.stringify(entry));
    }
  });

  it('carries no field the backend did not promise', () => {
    const feed = withAlarms([{ ...ALARM, owner: 'usr_somebody', notes: 'private' }]);
    assert.ok(feed);
    assert.deepEqual(Object.keys(feed!.alarms[0]).sort(), ['due', 'id', 'kind', 'label', 'state']);
  });

  it('survives the BFF-to-browser round trip unchanged', () => {
    const served = withAlarms([ALARM]);
    assert.ok(served);
    const rendered = browserReminders(JSON.parse(JSON.stringify(served)));
    assert.deepEqual(rendered, served);
  });
});
