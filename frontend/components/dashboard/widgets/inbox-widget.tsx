'use client';

/**
 * Recent mail from the signed-in user's connected accounts. Every message
 * shown came from the backend feed as a bounded, plain-text summary (sender,
 * subject, a short preview, when it arrived) with the provider's own link to
 * open it. Each account that could not answer is named with its state, and
 * nothing is invented. Nothing here reads the voice session.
 */
import { Button } from '@/components/livekit/button';
import type { FeedController } from '@/hooks/useDashboardFeed';
import {
  type AccountInboxGroup,
  type InboxFeed,
  type InboxMessageItem,
  accountIssues,
  formatDayOrTime,
  groupMessagesByAccount,
} from '@/lib/dashboard/provider-data';
import { cn } from '@/lib/utils';
import { AccountSection } from '../account-section';
import { WidgetEmpty, WidgetError, WidgetLoading, WidgetSignIn } from '../widget-notice';

const UNKNOWN_SENDER = 'Unknown sender';
const NO_SUBJECT = '(no subject)';

interface InboxWidgetProps {
  feed: FeedController<InboxFeed>;
  passwordLogin: boolean;
  now: Date | null;
  onOpenSettings: () => void;
}

export function InboxWidget({ feed, passwordLogin, now, onOpenSettings }: InboxWidgetProps) {
  if (feed.status === 'loading') {
    return <WidgetLoading label="Reading your inbox…" />;
  }
  if (feed.status === 'unauthorized') {
    return <WidgetSignIn passwordLogin={passwordLogin} what="your inbox" />;
  }
  if (feed.status === 'unconfigured') {
    return (
      <WidgetEmpty
        title="Connected accounts need multi-user identity"
        detail="This JARVIS server runs in single-user mode, so there are no per-user mail accounts to read."
      />
    );
  }
  if (feed.status === 'error') {
    return (
      <WidgetError
        title="Inbox is unavailable"
        detail="The backend did not answer."
        onRetry={feed.reload}
      />
    );
  }

  const { accounts, messages } = feed.data;
  if (accounts.length === 0) {
    return (
      <WidgetEmpty
        title="No accounts connected"
        detail="Connect a Google or Microsoft account under Settings → Integrations → Connected accounts and your recent mail appears here."
        action={
          <Button variant="outline" size="sm" onClick={onOpenSettings}>
            Open settings
          </Button>
        }
      />
    );
  }

  const issues = accountIssues(accounts);
  const clock = now ?? new Date(feed.data.generatedAt * 1000);
  const groups = groupMessagesByAccount(feed.data);
  const quiet = issues.length === accounts.length;

  return (
    <div className="space-y-2.5">
      {messages.length === 0 && !quiet && (
        <p className="text-muted-foreground text-sm">No recent messages.</p>
      )}
      {groups.map((group) => (
        <AccountGroup
          key={group.account.connectionId}
          group={group}
          now={clock}
          onOpenSettings={onOpenSettings}
        />
      ))}
    </div>
  );
}

function messageCount(group: AccountInboxGroup): string {
  const total = group.messages.length;
  const shown = total === 1 ? '1 message' : total + ' messages';
  return group.unreadCount > 0 ? shown + ', ' + group.unreadCount + ' unread' : shown;
}

interface AccountGroupProps {
  group: AccountInboxGroup;
  now: Date;
  onOpenSettings: () => void;
}

/** One connected account's own recent mail, with its own unread count. */
function AccountGroup({ group, now, onOpenSettings }: AccountGroupProps) {
  return (
    <AccountSection
      account={group.account}
      count={messageCount(group)}
      emptyLabel="No recent messages."
      onOpenSettings={onOpenSettings}
    >
      {group.messages.length > 0 ? (
        <ol aria-label="Recent messages" className="divide-border/60 divide-y">
          {group.messages.map((message) => (
            <li key={message.connectionId + ':' + message.id} className="py-2 first:pt-0 last:pb-0">
              <MessageRow message={message} now={now} />
            </li>
          ))}
        </ol>
      ) : null}
    </AccountSection>
  );
}

function MessageRow({ message, now }: { message: InboxMessageItem; now: Date }) {
  const body = (
    <>
      <div className="flex items-baseline justify-between gap-3">
        <p
          className={cn(
            'min-w-0 truncate text-sm',
            message.unread ? 'font-semibold' : 'font-medium'
          )}
        >
          {message.unread && (
            <span
              aria-hidden
              className="bg-primary mr-1.5 inline-block size-1.5 rounded-full align-middle"
            />
          )}
          {message.sender ?? UNKNOWN_SENDER}
        </p>
        <time
          dateTime={message.receivedAt}
          className="text-muted-foreground shrink-0 text-xs tabular-nums"
        >
          {formatDayOrTime(message.receivedAt, now)}
        </time>
      </div>
      <p className="truncate text-sm">{message.subject ?? NO_SUBJECT}</p>
      {message.preview && (
        <p className="text-muted-foreground line-clamp-2 text-xs">{message.preview}</p>
      )}
      {message.unread && <span className="sr-only">Unread</span>}
    </>
  );
  if (!message.link) return <div className="-mx-2 px-2 py-1">{body}</div>;
  return (
    <a
      href={message.link}
      target="_blank"
      rel="noopener noreferrer"
      className="hover:bg-muted/60 -mx-2 block rounded-md px-2 py-1"
    >
      {body}
    </a>
  );
}
