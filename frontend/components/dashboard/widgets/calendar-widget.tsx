'use client';

/**
 * Upcoming events from the signed-in user's connected accounts, grouped by
 * day, followed by the operator-level calendar sources JARVIS can read by
 * voice. Every event shown came from the backend feed as a bounded summary;
 * each account that could not answer is named with its state; nothing is
 * invented. Nothing here reads the voice session.
 */
import { Button } from '@/components/livekit/button';
import type { CapabilitiesState } from '@/hooks/useDashboardCapabilities';
import type { FeedController } from '@/hooks/useDashboardFeed';
import type { CalendarProvider } from '@/lib/dashboard/capabilities';
import {
  type AccountEventGroup,
  type CalendarEventItem,
  type CalendarFeed,
  accountIssues,
  dayLabel,
  formatClockTime,
  groupEventsByAccount,
  groupEventsByDay,
} from '@/lib/dashboard/provider-data';
import { cn } from '@/lib/utils';
import { AccountSection } from '../account-section';
import {
  WidgetBlocked,
  WidgetEmpty,
  WidgetError,
  WidgetLoading,
  WidgetSignIn,
} from '../widget-notice';

const PROVIDER_LABEL: Record<CalendarProvider, string> = {
  zoho_caldav: 'Zoho',
  caldav: 'CalDAV',
  ics: 'ICS feed',
  google: 'Google',
  microsoft: 'Microsoft',
  icloud: 'iCloud',
  other: 'Calendar',
};
const ELLIPSIS = '…';
const UNTITLED = '(no title)';
const ALL_DAY = 'All day';
const RANGE = ' - ';
const DATE_FORMAT: Intl.DateTimeFormatOptions = { weekday: 'long', month: 'long', day: 'numeric' };

interface CalendarWidgetProps {
  capabilities: CapabilitiesState & { reload: () => void };
  feed: FeedController<CalendarFeed>;
  passwordLogin: boolean;
  today: Date | null;
  onOpenSettings: () => void;
}

export function CalendarWidget({
  capabilities,
  feed,
  passwordLogin,
  today,
  onOpenSettings,
}: CalendarWidgetProps) {
  const dateLabel = today ? new Intl.DateTimeFormat(undefined, DATE_FORMAT).format(today) : null;

  return (
    <div className="space-y-3">
      <p className="text-sm">
        <span className="text-muted-foreground">Today </span>
        <span className="font-medium">{dateLabel ?? ELLIPSIS}</span>
      </p>
      <Upcoming
        feed={feed}
        passwordLogin={passwordLogin}
        today={today}
        onOpenSettings={onOpenSettings}
      />
      <VoiceSources capabilities={capabilities} />
    </div>
  );
}

interface UpcomingProps {
  feed: FeedController<CalendarFeed>;
  passwordLogin: boolean;
  today: Date | null;
  onOpenSettings: () => void;
}

function Upcoming({ feed, passwordLogin, today, onOpenSettings }: UpcomingProps) {
  if (feed.status === 'loading') {
    return <WidgetLoading label="Reading your calendars…" />;
  }
  if (feed.status === 'unauthorized') {
    return <WidgetSignIn passwordLogin={passwordLogin} what="your calendar" />;
  }
  if (feed.status === 'unconfigured') {
    return (
      <WidgetEmpty
        title="Connected accounts need multi-user identity"
        detail="This JARVIS server runs in single-user mode, so there are no per-user calendars to read."
      />
    );
  }
  if (feed.status === 'error') {
    return (
      <WidgetError
        title="Upcoming events are unavailable"
        detail="The backend did not answer."
        onRetry={feed.reload}
      />
    );
  }

  const { accounts, events } = feed.data;
  if (accounts.length === 0) {
    return (
      <WidgetEmpty
        title="No accounts connected"
        detail="Connect a Google or Microsoft account under Settings → Integrations → Connected accounts and your upcoming events appear here."
        action={
          <Button variant="outline" size="sm" onClick={onOpenSettings}>
            Open settings
          </Button>
        }
      />
    );
  }

  const issues = accountIssues(accounts);
  const now = today ?? new Date(feed.data.generatedAt * 1000);
  const groups = groupEventsByAccount(feed.data);
  const quiet = issues.length === accounts.length;

  return (
    <section aria-label="Upcoming events" className="space-y-2.5">
      {events.length === 0 && !quiet && (
        <p className="text-muted-foreground text-sm">Nothing scheduled in the next seven days.</p>
      )}
      {groups.map((group) => (
        <AccountGroup
          key={group.account.connectionId}
          group={group}
          now={now}
          onOpenSettings={onOpenSettings}
        />
      ))}
    </section>
  );
}

function eventCount(total: number): string {
  return total === 1 ? '1 event' : total + ' events';
}

interface AccountGroupProps {
  group: AccountEventGroup;
  now: Date;
  onOpenSettings: () => void;
}

/** One connected account's own upcoming events, by day. Never another's. */
function AccountGroup({ group, now, onOpenSettings }: AccountGroupProps) {
  const days = groupEventsByDay(group.events);
  return (
    <AccountSection
      account={group.account}
      count={eventCount(group.events.length)}
      emptyLabel="Nothing scheduled."
      onOpenSettings={onOpenSettings}
    >
      {days.length > 0 ? (
        <div className="space-y-2">
          {days.map((day) => (
            <div key={day.day}>
              <p className="text-muted-foreground mb-1 text-xs font-medium tracking-wider uppercase">
                {dayLabel(day.day, now)}
              </p>
              <ol className="divide-border/60 divide-y">
                {day.events.map((event) => (
                  <li
                    key={event.connectionId + ':' + event.id}
                    className="py-1 first:pt-0 last:pb-0"
                  >
                    <EventRow event={event} />
                  </li>
                ))}
              </ol>
            </div>
          ))}
        </div>
      ) : null}
    </AccountSection>
  );
}

function eventTime(event: CalendarEventItem): string {
  if (event.allDay) return ALL_DAY;
  const start = formatClockTime(event.start);
  return event.end ? start + RANGE + formatClockTime(event.end) : start;
}

function EventRow({ event }: { event: CalendarEventItem }) {
  const tentative = event.status === 'tentative';
  const body = (
    <div className="flex items-baseline gap-3">
      <span className="text-muted-foreground w-28 shrink-0 text-xs tabular-nums">
        {eventTime(event)}
      </span>
      <span className="min-w-0 flex-1">
        <span className={cn('block truncate text-sm', tentative ? 'italic' : 'font-medium')}>
          {event.title ?? UNTITLED}
        </span>
        {event.location && (
          <span className="text-muted-foreground block truncate text-xs">{event.location}</span>
        )}
      </span>
    </div>
  );
  if (!event.link) return <div className="-mx-2 px-2 py-1">{body}</div>;
  return (
    <a
      href={event.link}
      target="_blank"
      rel="noopener noreferrer"
      className="hover:bg-muted/60 -mx-2 block rounded-md px-2 py-1"
    >
      {body}
    </a>
  );
}

function VoiceSources({ capabilities }: { capabilities: CapabilitiesState }) {
  if (capabilities.status !== 'ready' || !capabilities.data.calendar.configured) return null;
  const { calendar } = capabilities.data;
  return (
    <div>
      <p className="text-muted-foreground mb-1.5 text-xs font-medium tracking-wider uppercase">
        Voice sources
      </p>
      <ul className="flex flex-wrap gap-1.5">
        {calendar.sources.map((source) => (
          <li
            key={source.id}
            className="bg-muted inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs"
          >
            <span className="font-medium">{source.label}</span>
            <span className="text-muted-foreground">{PROVIDER_LABEL[source.provider]}</span>
            {source.isDefault && <span className="text-muted-foreground">· default</span>}
          </li>
        ))}
      </ul>
      <WidgetBlocked
        className="mt-2"
        title="Events from these sources"
        detail="JARVIS reads them by voice; they are not shown on the dashboard yet."
        endpoint={calendar.eventsEndpoint}
      />
    </div>
  );
}
