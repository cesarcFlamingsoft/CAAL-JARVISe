/**
 * What the dashboard is allowed to know about the deployment.
 *
 * The operator settings the backend returns from `GET /settings` include every
 * credential JARVIS uses. The dashboard needs none of them: it only needs to
 * know *whether* a calendar or reminder source exists so it can show an honest
 * empty state instead of an invented one. This module is the boundary that
 * keeps it that way: it is the only shape the browser ever receives.
 *
 * Where the backend has no HTTP endpoint yet, the capability says so by name
 * rather than the widget pretending to have data.
 */

/** Backend endpoints the dashboard needs that do not exist yet. */
export const MISSING_ENDPOINTS = {
  weather: 'GET /weather/current',
  calendarEvents: 'GET /calendar/events',
  reminders: 'GET /reminders',
  backgroundTasks: 'GET /tasks',
} as const;

export type CalendarProvider =
  | 'zoho_caldav'
  | 'caldav'
  | 'ics'
  | 'google'
  | 'microsoft'
  | 'icloud'
  | 'other';

export interface CalendarSourceSummary {
  id: string;
  label: string;
  provider: CalendarProvider;
  isDefault: boolean;
  writable: boolean;
}

export type RemindersProvider = 'local' | 'apple';

export interface DashboardCapabilities {
  weather: { configured: false; missingEndpoint: string };
  calendar: {
    configured: boolean;
    sources: CalendarSourceSummary[];
    eventsEndpoint: string;
  };
  reminders: {
    configured: boolean;
    provider: RemindersProvider | null;
    listEndpoint: string;
  };
  alarms: { enabled: boolean };
  work: { tasksEndpoint: string };
}

const CALENDAR_PROVIDERS = new Set<CalendarProvider>([
  'zoho_caldav',
  'caldav',
  'ics',
  'google',
  'microsoft',
  'icloud',
]);

const REMINDER_PROVIDERS = new Set<RemindersProvider>(['local', 'apple']);

const MAX_LABEL = 60;

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function shortText(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const text = value.trim().replace(/\s+/g, ' ');
  if (!text || /[\p{Cc}\p{Cf}]/u.test(text)) return null;
  return text.slice(0, MAX_LABEL);
}

function calendarSource(entry: unknown): CalendarSourceSummary | null {
  const source = record(entry);
  const id = shortText(source?.id);
  if (!source || !id) return null;
  const provider = typeof source.provider === 'string' ? source.provider : '';
  return {
    id,
    label: shortText(source.display_name) ?? id,
    provider: CALENDAR_PROVIDERS.has(provider as CalendarProvider)
      ? (provider as CalendarProvider)
      : 'other',
    isDefault: source.default === true,
    writable: source.writable === true,
  };
}

/** Reduce the raw operator settings to the credential-free capability summary. */
export function capabilitiesFromSettings(payload: unknown): DashboardCapabilities {
  const settings = record(record(payload)?.settings);

  const sources: CalendarSourceSummary[] = [];
  const rawSources = settings?.calendar_sources;
  if (Array.isArray(rawSources)) {
    for (const entry of rawSources) {
      const source = calendarSource(entry);
      if (source) sources.push(source);
    }
  }

  const rawReminders = settings?.reminders_provider;
  const remindersProvider = REMINDER_PROVIDERS.has(rawReminders as RemindersProvider)
    ? (rawReminders as RemindersProvider)
    : null;

  return {
    weather: { configured: false, missingEndpoint: MISSING_ENDPOINTS.weather },
    calendar: {
      configured: sources.length > 0,
      sources,
      eventsEndpoint: MISSING_ENDPOINTS.calendarEvents,
    },
    reminders: {
      configured: remindersProvider !== null,
      provider: remindersProvider,
      listEndpoint: MISSING_ENDPOINTS.reminders,
    },
    alarms: { enabled: settings?.alarms_enabled === true },
    work: { tasksEndpoint: MISSING_ENDPOINTS.backgroundTasks },
  };
}
