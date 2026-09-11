'use client';

/**
 * The signed-in user own local reminders, and how each one will reach them.
 *
 * Every reminder shown here came from the backend for this session user; there
 * is no other way into this widget and nothing in it can name another owner.
 * Due times are rendered in the reader own timezone, so a reminder always
 * reads as the wall clock it will actually go off at. A reminder with no time
 * on it is a list item and is shown as one -- it never claims a delivery.
 *
 * The only thing that can be changed from here is which channels *future*
 * reminders use. It touches nothing that is already armed, and it can name no
 * phone number and no chat: those come from the profile, server-side.
 */
import { useCallback, useEffect, useState } from 'react';
import { apiRequest } from '@/components/account/api-client';
import { Button } from '@/components/livekit/button';
import type { FeedController } from '@/hooks/useDashboardFeed';
import {
  CHANNEL_CHOICES,
  CHANNEL_LABELS,
  type DeliveryChannel,
  type DeliveryDefaults,
  type ReminderItem,
  type RemindersFeed,
  STATE_LABELS,
  browserDeliveryDefaults,
  dueLabel,
} from '@/lib/dashboard/reminders';
import { WidgetEmpty, WidgetError, WidgetLoading, WidgetSignIn } from '../widget-notice';

interface RemindersWidgetProps {
  feed: FeedController<RemindersFeed>;
  passwordLogin: boolean;
  /** Null until the page has mounted, so the first paint shows no clock at all. */
  now: Date | null;
}

const STATE_CLASS: Record<string, string> = {
  pending: 'bg-muted text-muted-foreground',
  delivered: 'bg-muted text-foreground',
  failed: 'bg-destructive/10 text-destructive',
};

function DeliveryTags({ reminder, now }: { reminder: ReminderItem; now: Date | null }) {
  if (!reminder.timed) {
    return <span className="text-muted-foreground text-xs">List item, no alert</span>;
  }
  if (reminder.delivery.length === 0) {
    // A reminder can have no channel for two honest reasons: its time is still
    // coming and nothing was armed, or it has already been and gone. Saying
    // which is the difference between a warning and a fact.
    const past = now !== null && reminder.due !== null && Date.parse(reminder.due) <= now.getTime();
    return (
      <span className="text-muted-foreground text-xs">
        {past ? 'Nothing left to deliver' : 'No delivery armed'}
      </span>
    );
  }
  return (
    <ul className="flex flex-wrap gap-1">
      {reminder.delivery.map((entry) => (
        <li
          key={entry.channel}
          className={'rounded-full px-2 py-0.5 text-xs ' + (STATE_CLASS[entry.state] ?? '')}
        >
          {CHANNEL_LABELS[entry.channel]} &middot; {STATE_LABELS[entry.state]}
        </li>
      ))}
    </ul>
  );
}

/** The minimal edit surface: what the *next* reminder will use. */
function DeliveryDefaultsEditor({
  available,
  onSaved,
}: {
  available: DeliveryChannel[];
  onSaved: () => void;
}) {
  const [state, setState] = useState<DeliveryDefaults | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const result = await apiRequest<unknown>('/api/dashboard/reminders/delivery');
    setState(result.ok ? browserDeliveryDefaults(result.data) : null);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const toggle = async (channel: DeliveryChannel) => {
    if (!state || busy) return;
    const chosen = state.delivery.includes(channel)
      ? state.delivery.filter((entry) => entry !== channel)
      : [...state.delivery, channel];
    if (chosen.length === 0) {
      setError('Pick at least one way to be reminded.');
      return;
    }
    setBusy(true);
    setError(null);
    const result = await apiRequest<unknown>('/api/dashboard/reminders/delivery', {
      method: 'PUT',
      body: { delivery: chosen },
    });
    setBusy(false);
    const saved = result.ok ? browserDeliveryDefaults(result.data) : null;
    if (saved) {
      setState(saved);
      onSaved();
    } else {
      setError('That could not be saved.');
    }
  };

  if (!state) return null;
  const offered = state.available.length > 0 ? state.available : available;
  return (
    <div className="border-border/70 mt-1 border-t pt-2">
      <p className="text-muted-foreground text-xs">New reminders reach me by</p>
      <ul className="mt-1.5 flex flex-wrap gap-1.5">
        {offered.map((channel) => {
          const on = state.delivery.includes(channel);
          return (
            <li key={channel}>
              <Button
                type="button"
                size="sm"
                variant={on ? 'primary' : 'outline'}
                aria-pressed={on}
                disabled={busy}
                onClick={() => void toggle(channel)}
              >
                {CHANNEL_CHOICES[channel]}
              </Button>
            </li>
          );
        })}
      </ul>
      <p className="text-muted-foreground mt-1.5 text-xs">
        {error ?? 'A message or a call happens when the reminder comes due, not now.'}
      </p>
    </div>
  );
}

export function RemindersWidget({ feed, passwordLogin, now }: RemindersWidgetProps) {
  if (feed.status === 'loading') {
    return <WidgetLoading label="Loading your reminders…" />;
  }
  if (feed.status === 'unauthorized') {
    return <WidgetSignIn passwordLogin={passwordLogin} what="your reminders" />;
  }
  if (feed.status === 'unconfigured') {
    return (
      <WidgetEmpty
        title="Reminders are per-person"
        detail="This deployment has no signed-in profiles, so there is no private list to show."
      />
    );
  }
  if (feed.status === 'error') {
    return (
      <WidgetError
        title="Reminders are unavailable"
        detail="The backend did not answer."
        onRetry={feed.reload}
      />
    );
  }

  const { reminders, available } = feed.data;
  return (
    <div className="space-y-3">
      {reminders.length === 0 ? (
        <WidgetEmpty
          title="No reminders yet"
          detail="Ask JARVIS to remind you about something and it will appear here."
        />
      ) : (
        <ul className="space-y-2">
          {reminders.map((reminder) => (
            <li key={reminder.id} className="space-y-1">
              <p className="text-sm leading-snug font-medium">{reminder.title}</p>
              <p className="text-muted-foreground text-xs">
                {!reminder.timed
                  ? 'No time on it'
                  : now
                    ? dueLabel(reminder.due, now)
                    : 'Reading the time\u2026'}
                {reminder.completed ? ' · done' : ''}
              </p>
              <DeliveryTags reminder={reminder} now={now} />
            </li>
          ))}
        </ul>
      )}
      <DeliveryDefaultsEditor available={available} onSaved={feed.reload} />
    </div>
  );
}
