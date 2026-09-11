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
