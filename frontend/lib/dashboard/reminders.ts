/**
 * The browser-facing contract for the signed-in user own local reminders.
 *
 * The BFF proxies the backend own per-user reminder routes and reduces every
 * answer through these parsers before anything reaches a page, and checks
 * everything a page sends before it reaches the backend. The rules mirror the
 * backend own: a channel is one of a fixed vocabulary, a state is one of a
 * fixed vocabulary, every text field is plain and bounded, and a due time is
 * one instant or it is nothing at all.
 *
 * A reminder is private, so nothing here has a way to name an owner: the only
 * reminders that exist for this module are the ones the backend returned for
 * the session user. Nothing a page can send names a phone number or a chat.
 *
 * Free of Next.js and React so it can be unit tested.
 */

export const DELIVERY_CHANNELS = ['speak', 'telegram', 'call'] as const;
export type DeliveryChannel = (typeof DELIVERY_CHANNELS)[number];

export const DELIVERY_STATES = ['pending', 'delivered', 'failed'] as const;
export type DeliveryState = (typeof DELIVERY_STATES)[number];

/** Matches the backend own bounds. */
const MAX_TITLE = 200;
const MAX_NOTES = 2000;
const MAX_LIST = 80;
const MAX_REMINDERS = 200;
const REMINDER_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

/** How each channel reads on the dashboard, as the words JARVIS uses out loud. */
export const CHANNEL_LABELS: Record<DeliveryChannel, string> = {
  speak: 'Spoken here',
  telegram: 'Telegram',
  call: 'Phone call',
};

/** The choice, phrased the way the question is asked. */
export const CHANNEL_CHOICES: Record<DeliveryChannel, string> = {
  speak: 'Say it here',
  telegram: 'Message my Telegram',
  call: 'Call me',
};

export const STATE_LABELS: Record<DeliveryState, string> = {
  pending: 'waiting',
  delivered: 'delivered',
  failed: 'failed',
};

export interface ReminderDelivery {
  channel: DeliveryChannel;
  state: DeliveryState;
}

export interface ReminderItem {
  id: string;
  title: string;
  /** UTC ISO 8601, or null for a reminder with no time on it. */
  due: string | null;
  timed: boolean;
  list: string;
  notes: string;
  completed: boolean;
  delivery: ReminderDelivery[];
}

export interface RemindersFeed {
  generatedAt: number;
  reminders: ReminderItem[];
  /** Channels this owner is allowed to use at all, decided by the backend. */
  available: DeliveryChannel[];
  /** What a new reminder of theirs uses when they do not say. */
  defaults: DeliveryChannel[];
}

export interface DeliveryDefaults {
  delivery: DeliveryChannel[];
  available: DeliveryChannel[];
  /** Whether they have ever chosen; false means JARVIS still asks out loud. */
  saved: boolean;
}

// --- bounded values ---------------------------------------------------------------------

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/** Plain, single-line, bounded text; empty for anything else. */
function text(value: unknown, limit: number): string {
  if (typeof value !== 'string') return '';
  return value
    .replace(/\p{Cc}/gu, ' ')
    .replace(/\p{Cf}/gu, '')
    .trim()
    .slice(0, limit);
}

function isChannel(value: unknown): value is DeliveryChannel {
  return typeof value === 'string' && (DELIVERY_CHANNELS as readonly string[]).includes(value);
}

function isState(value: unknown): value is DeliveryState {
  return typeof value === 'string' && (DELIVERY_STATES as readonly string[]).includes(value);
}

/** One instant, or null. A value that does not parse is not shown as a time. */
function instant(value: unknown): string | null {
  if (typeof value !== 'string' || !value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? value : null;
}

/** Canonical order, deduplicated: the same set always reads the same way. */
export function orderChannels(values: readonly unknown[]): DeliveryChannel[] {
  const chosen = new Set(values.filter(isChannel));
  return DELIVERY_CHANNELS.filter((channel) => chosen.has(channel));
}

function channelList(value: unknown): DeliveryChannel[] {
  return Array.isArray(value) ? orderChannels(value) : [];
}

function delivery(value: unknown): ReminderDelivery[] {
  if (!Array.isArray(value)) return [];
  const seen = new Map<DeliveryChannel, DeliveryState>();
  for (const entry of value) {
    const item = record(entry);
    if (!item || !isChannel(item.channel) || !isState(item.state)) continue;
    if (!seen.has(item.channel)) seen.set(item.channel, item.state);
  }
  return DELIVERY_CHANNELS.filter((channel) => seen.has(channel)).map((channel) => ({
    channel,
    state: seen.get(channel) as DeliveryState,
  }));
}

function reminder(value: unknown): ReminderItem | null {
  const raw = record(value);
  if (!raw) return null;
  const id = typeof raw.id === 'string' && REMINDER_ID.test(raw.id) ? raw.id : null;
  const title = text(raw.title, MAX_TITLE);
  if (!id || !title) return null;
  const due = instant(raw.due);
  return {
    id,
    title,
    due,
    timed: raw.timed === true && due !== null,
    list: text(raw.list_name, MAX_LIST) || 'Reminders',
    notes: text(raw.notes, MAX_NOTES),
    completed: raw.completed === true,
    // An undated reminder is a list item: it can never show a delivery, even
    // if something upstream ever claimed one.
    delivery: raw.timed === true && due !== null ? delivery(raw.delivery) : [],
  };
}

export function browserReminders(data: unknown): RemindersFeed | null {
  const raw = record(data);
  if (!raw || typeof raw.generated_at !== 'number' || !Array.isArray(raw.reminders)) return null;
  const reminders: ReminderItem[] = [];
  for (const entry of raw.reminders.slice(0, MAX_REMINDERS)) {
    const item = reminder(entry);
    if (item) reminders.push(item);
  }
  return {
    generatedAt: raw.generated_at,
    reminders,
    available: channelList(raw.available),
    defaults: channelList(raw.defaults),
  };
}

export function browserDeliveryDefaults(data: unknown): DeliveryDefaults | null {
  const raw = record(data);
  if (!raw || !Array.isArray(raw.delivery) || !Array.isArray(raw.available)) return null;
  return {
    delivery: channelList(raw.delivery),
    available: channelList(raw.available),
    saved: raw.saved === true,
  };
}

/** What a page may send: a non-empty set from the fixed vocabulary, and nothing else. */
export function deliveryChoiceRequest(value: unknown): { delivery: DeliveryChannel[] } | null {
  if (!Array.isArray(value)) return null;
  if (value.length !== value.filter(isChannel).length) return null;
  const chosen = orderChannels(value);
  return chosen.length > 0 ? { delivery: chosen } : null;
}

// --- how a due time reads ---------------------------------------------------------------

/**
 * The time a reminder is due, in the reader own timezone.
 *
 * A stored reminder is one UTC instant; a reader is somewhere. Every label
 * here is produced in `timeZone` (the browser own by default), so the same
 * reminder reads as the wall-clock time that person will actually be at when
 * it goes off, and a reminder set on a phone in another zone still reads
 * correctly here. A time that has already passed says so rather than being
 * shown as an ordinary future time.
 */
export function dueLabel(iso: string | null, now: Date, timeZone?: string): string {
  if (!iso) return 'No time';
  const at = new Date(iso);
  if (!Number.isFinite(at.getTime())) return 'No time';
  const zone = timeZone || undefined;
  const clock = new Intl.DateTimeFormat('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    timeZone: zone,
  }).format(at);
  const today = dayKey(now, zone);
  const day = dayKey(at, zone);
  const tomorrow = dayKey(new Date(now.getTime() + 86_400_000), zone);
  const overdue = at.getTime() <= now.getTime();
  let when: string;
  if (day === today) when = 'Today ' + clock;
  else if (day === tomorrow) when = 'Tomorrow ' + clock;
  else
    when =
      new Intl.DateTimeFormat('en-GB', {
        weekday: 'short',
        day: 'numeric',
        month: 'short',
        timeZone: zone,
      }).format(at) +
      ', ' +
      clock;
  return overdue ? 'Overdue - was ' + when : when;
}

/** The calendar day an instant falls on *there*, not here. */
function dayKey(at: Date, timeZone?: string): string {
  return new Intl.DateTimeFormat('en-CA', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    timeZone,
  }).format(at);
}

/** "Spoken here and Telegram", for a reminder that is armed on both. */
export function deliverySummary(delivery: readonly ReminderDelivery[]): string {
  if (delivery.length === 0) return '';
  const names = delivery.map((item) => CHANNEL_LABELS[item.channel]);
  if (names.length === 1) return names[0];
  return names.slice(0, -1).join(', ') + ' and ' + names[names.length - 1];
}
