/**
 * The browser-facing contract for the dashboard's connected-account feeds.
 *
 * The BFF proxies the backend's calendar and inbox feeds for the signed-in
 * user and reduces each answer through these parsers before anything reaches
 * a page. The rules mirror the backend's: nothing unrecognised is forwarded,
 * every text field is plain and bounded, a link is https or nothing, and an
 * account's state is one of a fixed vocabulary with plain words for each.
 * Free of Next.js and React so it can be unit tested.
 */
import type { Provider } from '../connections/protocol';

/** Mirrors the connections protocol; kept local so this module runs under node --test. */
const PROVIDERS: readonly Provider[] = ['google', 'microsoft', 'zoho'];
const CONNECTION_ID = /^con_[0-9a-f]{24}$/;

function isProvider(value: unknown): value is Provider {
  return typeof value === 'string' && (PROVIDERS as readonly string[]).includes(value);
}

function isConnectionId(value: unknown): value is string {
  return typeof value === 'string' && CONNECTION_ID.test(value);
}

export const ACCOUNT_STATUSES = [
  'ok',
  'reconnect_required',
  'insufficient_scope',
  'not_configured',
  'unsupported',
  'unavailable',
  /** The provider did not answer; the items shown are what the backend last indexed. */
  'stale',
] as const;
export type AccountStatus = (typeof ACCOUNT_STATUSES)[number];

export interface FeedAccount {
  connectionId: string;
  provider: Provider;
  accountLabel: string | null;
  status: AccountStatus;
  /** The backend's bounded reason code, for diagnostics; never shown raw. */
  reason: string | null;
  /** Items this account contributed before the merged list was cut. */
  count: number;
}

export interface CalendarEventItem {
  id: string;
  connectionId: string;
  provider: Provider;
  title: string | null;
  /** UTC ISO 8601, or a YYYY-MM-DD date for an all-day event. */
  start: string;
  end: string | null;
  allDay: boolean;
  location: string | null;
  link: string | null;
  status: 'confirmed' | 'tentative' | null;
}

export interface CalendarFeed {
  generatedAt: number;
  windowStart: string | null;
  windowEnd: string | null;
  accounts: FeedAccount[];
  events: CalendarEventItem[];
}

export interface InboxMessageItem {
  id: string;
  connectionId: string;
  provider: Provider;
  subject: string | null;
  sender: string | null;
  preview: string | null;
  receivedAt: string;
  unread: boolean;
  link: string | null;
}

export interface InboxFeed {
  generatedAt: number;
  accounts: FeedAccount[];
  messages: InboxMessageItem[];
  /** Counted from the messages shown, never trusted from the wire. */
  unreadCount: number;
}

const MAX_ACCOUNTS = 100;
const MAX_ITEMS = 50;
const MAX_TITLE = 200;
const MAX_PREVIEW = 160;
const MAX_SENDER = 120;
const MAX_LABEL = 254;
const MAX_ID = 512;
const MAX_LINK = 2048;
const DATE_ONLY = /^\d{4}-\d{2}-\d{2}$/;
const DATE_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/;
const SHORT_CODE = /^[a-z_]{1,40}$/;
const EVENT_STATUSES = new Set(['confirmed', 'tentative']);
const DAY_MS = 86_400_000;

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/** Plain, single-line, bounded text; null for anything else or for nothing. */
function text(value: unknown, limit: number): string | null {
  if (typeof value !== 'string') return null;
  const plain = value
    .replace(/\p{Cc}/gu, ' ')
    .replace(/\p{Cf}/gu, '')
    .trim()
    .replace(/\s+/g, ' ');
  return plain ? plain.slice(0, limit) : null;
}

function idLike(value: unknown): string | null {
  if (typeof value !== 'string' || value.length === 0 || value.length > MAX_ID) return null;
  return /[\s\p{Cc}]/u.test(value) ? null : value;
}

/** The provider's own https link to an item, or null. Never anything else. */
function httpsLink(value: unknown): string | null {
  if (typeof value !== 'string' || value.length === 0 || value.length > MAX_LINK) return null;
  if (/[\s\p{Cc}]/u.test(value)) return null;
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    return null;
  }
  if (url.protocol !== 'https:' || !url.hostname || url.username || url.password) return null;
  return value;
}

function whenDate(value: unknown): string | null {
  return typeof value === 'string' && (DATE_ONLY.test(value) || DATE_TIME.test(value))
    ? value
    : null;
}

function whenTime(value: unknown): string | null {
  return typeof value === 'string' && DATE_TIME.test(value) ? value : null;
}

function count(value: unknown): number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : 0;
}

function compare(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

function list<T>(value: unknown[], parse: (entry: unknown) => T | null, max: number): T[] {
  const items: T[] = [];
  for (const entry of value) {
    const item = parse(entry);
    if (item) items.push(item);
    if (items.length === max) break;
  }
  return items;
}

function account(entry: unknown): FeedAccount | null {
  const row = record(entry);
  const connectionId = row?.connection_id ?? row?.connectionId;
  if (!row || !isConnectionId(connectionId) || !isProvider(row.provider)) return null;
  const status = (ACCOUNT_STATUSES as readonly string[]).includes(row.status as string)
    ? (row.status as AccountStatus)
    : 'unavailable';
  const reason = typeof row.reason === 'string' && SHORT_CODE.test(row.reason) ? row.reason : null;
  return {
    connectionId,
    provider: row.provider,
    accountLabel: text(row.account_label ?? row.accountLabel, MAX_LABEL),
    status,
    reason,
    count: count(row.count),
  };
}

function event(entry: unknown): CalendarEventItem | null {
  const row = record(entry);
  if (!row) return null;
  const id = idLike(row.id);
  const start = whenDate(row.start);
  const connectionId = row.connection_id ?? row.connectionId;
  if (!id || !start || !isConnectionId(connectionId) || !isProvider(row.provider)) {
    return null;
  }
  const status =
    typeof row.status === 'string' && EVENT_STATUSES.has(row.status)
      ? (row.status as 'confirmed' | 'tentative')
      : null;
  return {
    id,
    connectionId,
    provider: row.provider,
    title: text(row.title, MAX_TITLE),
    start,
    end: whenDate(row.end),
    allDay: row.all_day === true || row.allDay === true,
    location: text(row.location, MAX_TITLE),
    link: httpsLink(row.link),
    status,
  };
}

function message(entry: unknown): InboxMessageItem | null {
  const row = record(entry);
  if (!row) return null;
  const id = idLike(row.id);
  const receivedAt = whenTime(row.received_at ?? row.receivedAt);
  const connectionId = row.connection_id ?? row.connectionId;
  if (!id || !receivedAt || !isConnectionId(connectionId) || !isProvider(row.provider)) {
    return null;
  }
  return {
    id,
    connectionId,
    provider: row.provider,
    subject: text(row.subject, MAX_TITLE),
    sender: text(row.sender, MAX_SENDER),
    preview: text(row.preview, MAX_PREVIEW),
    receivedAt,
    unread: row.unread === true,
    link: httpsLink(row.link),
  };
}

/**
 * The backend's calendar feed, reduced to what a browser may see. A payload
 * without both lists is refused outright so a contract drift surfaces as an
 * error rather than as an empty widget.
 */
export function browserCalendarFeed(data: unknown): CalendarFeed | null {
  const body = record(data);
  if (!body || !Array.isArray(body.accounts) || !Array.isArray(body.events)) return null;
  const events = list(body.events, event, MAX_ITEMS).sort((a, b) => compare(a.start, b.start));
  return {
    generatedAt: count(body.generated_at ?? body.generatedAt),
    windowStart: whenTime(body.window_start ?? body.windowStart),
    windowEnd: whenTime(body.window_end ?? body.windowEnd),
    accounts: list(body.accounts, account, MAX_ACCOUNTS),
    events,
  };
}

export function browserInboxFeed(data: unknown): InboxFeed | null {
  const body = record(data);
  if (!body || !Array.isArray(body.accounts) || !Array.isArray(body.messages)) return null;
  const messages = list(body.messages, message, MAX_ITEMS).sort((a, b) =>
    compare(b.receivedAt, a.receivedAt)
  );
  return {
    generatedAt: count(body.generated_at ?? body.generatedAt),
    accounts: list(body.accounts, account, MAX_ACCOUNTS),
    messages,
    unreadCount: messages.filter((item) => item.unread).length,
  };
}

// --- grouping by account ---------------------------------------------------------------------

/** One connected account's slice of a feed: never merged with another account's. */
export interface AccountEventGroup {
  account: FeedAccount;
  events: CalendarEventItem[];
}

export interface AccountInboxGroup {
  account: FeedAccount;
  messages: InboxMessageItem[];
  /** Unread among this account's own messages, counted here, never from the wire. */
  unreadCount: number;
}

/**
 * Items bucketed under the account they came from, one bucket per account the
 * feed named, in the feed's account order. An account that answered with
 * nothing, or could not answer at all, still gets its bucket so its state is
 * never silently missing; an item naming an account the feed did not report is
 * attributed to none, because there is no account to show it under.
 */
function byAccount<T extends { connectionId: string }>(
  accounts: FeedAccount[],
  items: T[]
): Array<[FeedAccount, T[]]> {
  const groups: Array<[FeedAccount, T[]]> = accounts.map((account) => [account, []]);
  const first = new Map<string, T[]>();
  for (const [account, bucket] of groups) {
    if (!first.has(account.connectionId)) first.set(account.connectionId, bucket);
  }
  for (const item of items) {
    first.get(item.connectionId)?.push(item);
  }
  return groups;
}

export function groupEventsByAccount(feed: CalendarFeed): AccountEventGroup[] {
  return byAccount(feed.accounts, feed.events).map(([account, events]) => ({ account, events }));
}

export function groupMessagesByAccount(feed: InboxFeed): AccountInboxGroup[] {
  return byAccount(feed.accounts, feed.messages).map(([account, messages]) => ({
    account,
    messages,
    unreadCount: messages.filter((message) => message.unread).length,
  }));
}

/** The accounts whose state the user should know about. */
export function accountIssues(accounts: FeedAccount[]): FeedAccount[] {
  return accounts.filter((entry) => entry.status !== 'ok');
}

/** Plain words for an account state. Nothing here is interpolated from a provider. */
export function describeAccountStatus(entry: FeedAccount): string {
  switch (entry.status) {
    case 'ok':
      return 'Connected';
    case 'reconnect_required':
      return 'Needs reconnecting: JARVIS no longer holds working access. Reconnect it under Settings.';
    case 'insufficient_scope':
      return 'Missing permission: this account was linked without access to this data. Reconnect it under Settings to grant it.';
    case 'not_configured':
      return 'This provider is no longer configured on the server, so its stored access cannot be renewed.';
    case 'unsupported':
      return 'Reading this data is not available yet for this provider.';
    case 'unavailable':
      return 'The provider did not answer just now.';
    case 'stale':
      return 'The provider did not answer just now, so this shows what JARVIS last saw.';
  }
}

// --- query ----------------------------------------------------------------------------------

export interface FeedQueryDefaults {
  /** Present for the calendar feed only; the inbox feed takes no window. */
  days?: number;
  limit: number;
}

const MAX_WINDOW_DAYS = 31;
const WHOLE_NUMBER = /^\d{1,3}$/;

function bounded(params: URLSearchParams, name: string, fallback: number, max: number) {
  const values = params.getAll(name);
  if (values.length === 0) return fallback;
  if (values.length > 1 || !WHOLE_NUMBER.test(values[0])) return null;
  const value = Number(values[0]);
  return value >= 1 && value <= max ? value : null;
}

/**
 * The query string for the backend feed: only days and limit, each a whole
 * number within the backend's bounds, defaults applied, everything else
 * ignored. Null for a value out of bounds or given twice, so the route can
 * refuse rather than guess.
 */
export function feedQuery(params: URLSearchParams, defaults: FeedQueryDefaults): string | null {
  const out = new URLSearchParams();
  if (defaults.days !== undefined) {
    const days = bounded(params, 'days', defaults.days, MAX_WINDOW_DAYS);
    if (days === null) return null;
    out.set('days', String(days));
  }
  const limit = bounded(params, 'limit', defaults.limit, MAX_ITEMS);
  if (limit === null) return null;
  out.set('limit', String(limit));
  return out.toString();
}

// --- days and times ---------------------------------------------------------------------

/** YYYY-MM-DD for an instant in timeZone (the browser's zone when omitted). */
export function localDay(value: Date, timeZone?: string): string {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(value);
  const part = (type: string) => parts.find((entry) => entry.type === type)?.value ?? '';
  return part('year') + '-' + part('month') + '-' + part('day');
}

export interface DayGroup {
  day: string;
  events: CalendarEventItem[];
}

/** Events bucketed by the local day they start on, in day order. */
export function groupEventsByDay(events: CalendarEventItem[], timeZone?: string): DayGroup[] {
  const groups = new Map<string, CalendarEventItem[]>();
  for (const item of events) {
    const day = DATE_ONLY.test(item.start) ? item.start : localDay(new Date(item.start), timeZone);
    const bucket = groups.get(day);
    if (bucket) bucket.push(item);
    else groups.set(day, [item]);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => compare(a, b))
    .map(([day, items]) => ({ day, events: items }));
}

/** Today, Tomorrow, or the weekday and date. */
export function dayLabel(day: string, now: Date, timeZone?: string): string {
  if (day === localDay(now, timeZone)) return 'Today';
  if (day === localDay(new Date(now.getTime() + DAY_MS), timeZone)) return 'Tomorrow';
  const [year, month, date] = day.split('-').map(Number);
  return new Intl.DateTimeFormat(undefined, {
    weekday: 'long',
    month: 'short',
    day: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(Date.UTC(year, month - 1, date)));
}

export function formatClockTime(iso: string, timeZone?: string): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: 'numeric',
    minute: '2-digit',
    timeZone,
  }).format(new Date(iso));
}

/** The time for something from today, otherwise a short date. */
export function formatDayOrTime(iso: string, now: Date, timeZone?: string): string {
  if (localDay(new Date(iso), timeZone) === localDay(now, timeZone)) {
    return formatClockTime(iso, timeZone);
  }
  return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', timeZone }).format(
    new Date(iso)
  );
}
