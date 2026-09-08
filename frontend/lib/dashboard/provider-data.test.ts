import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  ACCOUNT_STATUSES,
  accountIssues,
  browserCalendarFeed,
  browserInboxFeed,
  dayLabel,
  describeAccountStatus,
  feedQuery,
  groupEventsByAccount,
  groupEventsByDay,
  groupMessagesByAccount,
} from './provider-data.ts';

const CONNECTION = 'con_0123456789abcdef01234567';
const OTHER = 'con_fedcba9876543210fedcba98';
const TOKEN = 'ya29.SHOULD-NEVER-BE-HERE';
const BELL = String.fromCharCode(7);

const CALENDAR_PAYLOAD = {
  generated_at: 1_700_000_600,
  window_start: '2023-11-14T22:23:20Z',
  window_end: '2023-11-21T22:23:20Z',
  accounts: [
    {
      connection_id: CONNECTION,
      provider: 'google',
      account_label: 'ana@gmail.example',
      status: 'ok',
      reason: null,
      count: 2,
      access_token: TOKEN,
    },
    {
      connection_id: OTHER,
      provider: 'microsoft',
      account_label: null,
      status: 'reconnect_required',
      reason: 'reconnect_required',
      count: 0,
    },
    { connection_id: 'con_bad', provider: 'google', status: 'ok', count: 1 },
    {
      connection_id: CONNECTION,
      provider: 'zoho',
      account_label: 'z',
      status: 'made_up',
      reason: 'x',
      count: -3,
    },
  ],
  events: [
    {
      id: 'e2',
      connection_id: CONNECTION,
      provider: 'google',
      title: 'Later',
      start: '2023-11-16T09:00:00Z',
      end: '2023-11-16T10:00:00Z',
      all_day: false,
      location: null,
      link: 'https://www.google.com/calendar/event?eid=1',
      status: 'confirmed',
      description: 'SECRET body',
    },
    {
      id: 'e1',
      connection_id: CONNECTION,
      provider: 'google',
      title: '  Soon' + BELL + '  now ',
      start: '2023-11-15T01:00:00Z',
      end: null,
      all_day: false,
      location: 'Room',
      link: 'javascript:alert(1)',
      status: 'weird',
    },
    {
      id: 'e3',
      connection_id: OTHER,
      provider: 'microsoft',
      title: null,
      start: '2023-11-17',
      end: '2023-11-18',
      all_day: true,
      location: null,
      link: null,
      status: null,
    },
    {
      id: 'bad-start',
      connection_id: CONNECTION,
      provider: 'google',
      title: 'x',
      start: 'tomorrow',
      all_day: false,
    },
    {
      id: 'foreign',
      connection_id: 'usr_0123456789abcdef01234567',
      provider: 'google',
      start: '2023-11-17T09:00:00Z',
      all_day: false,
    },
    'garbage',
  ],
};

const INBOX_PAYLOAD = {
  generated_at: 1_700_000_600,
  accounts: [
    {
      connection_id: CONNECTION,
      provider: 'google',
      account_label: 'ana@gmail.example',
      status: 'ok',
      reason: null,
      count: 2,
    },
    {
      connection_id: OTHER,
      provider: 'zoho',
      account_label: 'ana@zoho.example',
      status: 'unsupported',
      reason: 'unsupported',
      count: 0,
    },
  ],
  messages: [
    {
      id: 'm-old',
      connection_id: CONNECTION,
      provider: 'google',
      subject: 'Older',
      sender: 'Billing',
      preview: 'P'.repeat(500),
      received_at: '2023-11-14T20:00:00Z',
      unread: false,
      link: null,
      body: 'SECRET body',
    },
    {
      id: 'm-new',
      connection_id: CONNECTION,
      provider: 'google',
      subject: null,
      sender: 'Ana <script>',
      preview: null,
      received_at: '2023-11-14T22:00:00Z',
      unread: true,
      link: 'https://mail.example/open?id=1',
    },
    {
      id: 'no-date',
      connection_id: CONNECTION,
      provider: 'google',
      subject: 'x',
      received_at: 'yesterday',
      unread: true,
    },
  ],
  unread_count: 99,
};

describe('calendar feed', () => {
  it('keeps only well-formed accounts and events, sorted by start, with nothing extra', () => {
    const feed = browserCalendarFeed(CALENDAR_PAYLOAD);
    assert.ok(feed);
    assert.equal(feed.generatedAt, 1_700_000_600);
    assert.equal(feed.windowStart, '2023-11-14T22:23:20Z');
    assert.equal(feed.windowEnd, '2023-11-21T22:23:20Z');

    assert.deepEqual(feed.accounts, [
      {
        connectionId: CONNECTION,
        provider: 'google',
        accountLabel: 'ana@gmail.example',
        status: 'ok',
        reason: null,
        count: 2,
      },
      {
        connectionId: OTHER,
        provider: 'microsoft',
        accountLabel: null,
        status: 'reconnect_required',
        reason: 'reconnect_required',
        count: 0,
      },
      {
        connectionId: CONNECTION,
        provider: 'zoho',
        accountLabel: 'z',
        status: 'unavailable',
        reason: 'x',
        count: 0,
      },
    ]);

    assert.deepEqual(
      feed.events.map((event) => event.id),
      ['e1', 'e2', 'e3']
    );
    assert.deepEqual(feed.events[0], {
      id: 'e1',
      connectionId: CONNECTION,
      provider: 'google',
      title: 'Soon now',
      start: '2023-11-15T01:00:00Z',
      end: null,
      allDay: false,
      location: 'Room',
      link: null,
      status: null,
    });
    assert.equal(feed.events[1].link, 'https://www.google.com/calendar/event?eid=1');
    assert.equal(feed.events[1].status, 'confirmed');
    assert.equal(feed.events[2].allDay, true);
    assert.equal(feed.events[2].title, null);

    const wire = JSON.stringify(feed);
    assert.ok(!wire.includes(TOKEN), 'a token must never survive parsing');
    assert.ok(!wire.includes('SECRET'), 'unknown fields are dropped');
    assert.ok(!wire.includes('javascript:'));
  });

  it('refuses a payload that does not carry both lists', () => {
    for (const bad of [
      null,
      'x',
      [],
      {},
      { accounts: [] },
      { events: [] },
      { accounts: 'x', events: [] },
    ]) {
      assert.equal(browserCalendarFeed(bad), null, JSON.stringify(bad));
    }
    const empty = browserCalendarFeed({
      generated_at: 0,
      window_start: '',
      window_end: '',
      accounts: [],
      events: [],
    });
    assert.deepEqual(empty, {
      generatedAt: 0,
      windowStart: null,
      windowEnd: null,
      accounts: [],
      events: [],
    });
  });

  it('preserves account and event rows after the BFF has reduced them', () => {
    const bffPayload = browserCalendarFeed(CALENDAR_PAYLOAD)!;
    const browserPayload = browserCalendarFeed(bffPayload)!;
    assert.equal(browserPayload.accounts.length, bffPayload.accounts.length);
    assert.equal(browserPayload.events.length, bffPayload.events.length);
  });

  it('groups events by local day and names today and tomorrow', () => {
    const feed = browserCalendarFeed(CALENDAR_PAYLOAD)!;
    const now = new Date('2023-11-15T00:30:00Z');
    const groups = groupEventsByDay(feed.events, 'UTC');
    assert.deepEqual(
      groups.map((group) => [group.day, group.events.map((event) => event.id)]),
      [
        ['2023-11-15', ['e1']],
        ['2023-11-16', ['e2']],
        ['2023-11-17', ['e3']],
      ]
    );
    assert.equal(dayLabel('2023-11-15', now, 'UTC'), 'Today');
    assert.equal(dayLabel('2023-11-16', now, 'UTC'), 'Tomorrow');
    const later = dayLabel('2023-11-17', now, 'UTC');
    assert.ok(later.length > 0 && later !== 'Today' && later !== 'Tomorrow');
  });
});

describe('inbox feed', () => {
  it('keeps only well-formed messages, newest first, bounded and plain', () => {
    const feed = browserInboxFeed(INBOX_PAYLOAD);
    assert.ok(feed);
    assert.deepEqual(
      feed.messages.map((message) => message.id),
      ['m-new', 'm-old']
    );
    assert.deepEqual(feed.messages[0], {
      id: 'm-new',
      connectionId: CONNECTION,
      provider: 'google',
      subject: null,
      sender: 'Ana <script>',
      preview: null,
      receivedAt: '2023-11-14T22:00:00Z',
      unread: true,
      link: 'https://mail.example/open?id=1',
    });
    assert.equal(feed.messages[1].preview!.length, 160);
    assert.equal(feed.unreadCount, 1, 'unread is counted from the messages shown, not trusted');
    assert.deepEqual(
      feed.accounts.map((account) => account.status),
      ['ok', 'unsupported']
    );
    assert.ok(!JSON.stringify(feed).includes('SECRET'));
  });

  it('preserves account and message rows after the BFF has reduced them', () => {
    const bffPayload = browserInboxFeed(INBOX_PAYLOAD)!;
    const browserPayload = browserInboxFeed(bffPayload)!;
    assert.equal(browserPayload.accounts.length, bffPayload.accounts.length);
    assert.equal(browserPayload.messages.length, bffPayload.messages.length);
  });

  it('refuses a payload without both lists', () => {
    for (const bad of [null, {}, { accounts: [] }, { messages: [] }]) {
      assert.equal(browserInboxFeed(bad), null);
    }
  });
});

describe('account status words', () => {
  it('has plain words for every status and lists only the accounts that need attention', () => {
    for (const status of ACCOUNT_STATUSES) {
      const words = describeAccountStatus({
        connectionId: CONNECTION,
        provider: 'google',
        accountLabel: null,
        status,
        reason: null,
        count: 0,
      });
      assert.ok(words.length > 0, status);
    }
    const feed = browserCalendarFeed(CALENDAR_PAYLOAD)!;
    assert.deepEqual(
      accountIssues(feed.accounts).map((account) => account.status),
      ['reconnect_required', 'unavailable']
    );
  });
});

describe('feed query', () => {
  it('passes only bounded days and limit to the backend, with defaults', () => {
    const calendar = { days: 7, limit: 25 };
    assert.equal(feedQuery(new URLSearchParams(''), calendar), 'days=7&limit=25');
    assert.equal(
      feedQuery(new URLSearchParams('days=3&limit=10&extra=1'), calendar),
      'days=3&limit=10'
    );
    assert.equal(feedQuery(new URLSearchParams('limit=5'), { limit: 20 }), 'limit=5');
    assert.equal(feedQuery(new URLSearchParams('days=3'), { limit: 20 }), 'limit=20');
    for (const bad of [
      'days=0',
      'days=32',
      'limit=0',
      'limit=51',
      'limit=abc',
      'days=1.5',
      'days=7&days=8',
      'limit=1e1',
    ]) {
      assert.equal(feedQuery(new URLSearchParams(bad), calendar), null, bad);
    }
  });
});

describe('grouping by connected account', () => {
  const THIRD = 'con_00112233445566778899aabb';

  const GROUPED_CALENDAR = {
    generated_at: 1_700_000_600,
    window_start: '2023-11-14T22:23:20Z',
    window_end: '2023-11-21T22:23:20Z',
    accounts: [
      {
        connection_id: CONNECTION,
        provider: 'google',
        account_label: 'ana@gmail.example',
        status: 'ok',
        reason: null,
        count: 2,
      },
      {
        connection_id: OTHER,
        provider: 'microsoft',
        account_label: 'ana@work.example',
        status: 'ok',
        reason: null,
        count: 1,
      },
      {
        connection_id: THIRD,
        provider: 'zoho',
        account_label: 'ana@zoho.example',
        status: 'reconnect_required',
        reason: 'reconnect_required',
        count: 0,
      },
    ],
    events: [
      {
        id: 'g1',
        connection_id: CONNECTION,
        provider: 'google',
        title: 'Standup',
        start: '2023-11-15T09:00:00Z',
        end: null,
        all_day: false,
        location: null,
        link: null,
        status: null,
      },
      {
        id: 'm1',
        connection_id: OTHER,
        provider: 'microsoft',
        title: 'Review',
        start: '2023-11-15T10:00:00Z',
        end: null,
        all_day: false,
        location: null,
        link: null,
        status: null,
      },
      {
        id: 'g2',
        connection_id: CONNECTION,
        provider: 'google',
        title: 'Later',
        start: '2023-11-16T09:00:00Z',
        end: null,
        all_day: false,
        location: null,
        link: null,
        status: null,
      },
    ],
  };

  const GROUPED_INBOX = {
    generated_at: 1_700_000_600,
    accounts: GROUPED_CALENDAR.accounts,
    messages: [
      {
        id: 'gm1',
        connection_id: CONNECTION,
        provider: 'google',
        subject: 'One',
        sender: 'Ana',
        preview: null,
        received_at: '2023-11-14T22:00:00Z',
        unread: true,
        link: null,
      },
      {
        id: 'mm1',
        connection_id: OTHER,
        provider: 'microsoft',
        subject: 'Two',
        sender: 'Bo',
        preview: null,
        received_at: '2023-11-14T21:00:00Z',
        unread: false,
        link: null,
      },
      {
        id: 'gm2',
        connection_id: CONNECTION,
        provider: 'google',
        subject: 'Three',
        sender: 'Cy',
        preview: null,
        received_at: '2023-11-14T20:00:00Z',
        unread: true,
        link: null,
      },
    ],
    unread_count: 2,
  };

  it('gives every account its own events, in account order, never one mixed list', () => {
    const feed = browserCalendarFeed(GROUPED_CALENDAR)!;
    const groups = groupEventsByAccount(feed);
    assert.deepEqual(
      groups.map((group) => [group.account.connectionId, group.events.map((event) => event.id)]),
      [
        [CONNECTION, ['g1', 'g2']],
        [OTHER, ['m1']],
        [THIRD, []],
      ]
    );
    assert.equal(groups.length, feed.accounts.length, 'one section per connected account');
  });

  it('keeps each section its own state and count, including an account that failed', () => {
    const groups = groupEventsByAccount(browserCalendarFeed(GROUPED_CALENDAR)!);
    const zoho = groups[2];
    assert.equal(zoho.account.status, 'reconnect_required');
    assert.equal(zoho.account.accountLabel, 'ana@zoho.example');
    assert.equal(zoho.account.provider, 'zoho');
    assert.deepEqual(zoho.events, [], 'an account that could not answer contributes nothing');
    assert.equal(groups[0].account.status, 'ok');
    assert.equal(groups[0].events.length, 2, 'the section counts only its own account');
  });

  it('attributes nothing to an account the feed did not name', () => {
    const stray = {
      ...GROUPED_CALENDAR,
      accounts: [GROUPED_CALENDAR.accounts[0]],
    };
    const groups = groupEventsByAccount(browserCalendarFeed(stray)!);
    assert.deepEqual(
      groups.map((group) => group.events.map((event) => event.id)),
      [['g1', 'g2']]
    );
  });

  it('gives every account its own messages and its own unread count', () => {
    const feed = browserInboxFeed(GROUPED_INBOX)!;
    const groups = groupMessagesByAccount(feed);
    assert.deepEqual(
      groups.map((group) => [
        group.account.connectionId,
        group.messages.map((message) => message.id),
        group.unreadCount,
      ]),
      [
        [CONNECTION, ['gm1', 'gm2'], 2],
        [OTHER, ['mm1'], 0],
        [THIRD, [], 0],
      ]
    );
  });
});
