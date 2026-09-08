'use client';

/**
 * One connected account's own slice of a dashboard feed: a heading naming the
 * account and its provider, that account's own item count, and either its
 * items or, when it could not answer, its own state with the remedy. Sections
 * stack, so a widget never merges two accounts into one undifferentiated
 * stream, and nothing here is fixed-width: the movable widget frame owns the
 * size, so a section reflows with it.
 *
 * Nothing here is interpolated from a provider and nothing reads the voice
 * session.
 */
import type { ReactNode } from 'react';
import type { FeedAccount } from '@/lib/dashboard/provider-data';
import { AccountIssues, PROVIDER_LABEL, accountName } from './account-issues';

const SEPARATOR = ' \u00b7 ';

interface AccountSectionProps {
  account: FeedAccount;
  /** This account's own item count, already in words, e.g. "2 events". */
  count: string | null;
  /** Shown when this account answered with nothing at all. */
  emptyLabel: string;
  onOpenSettings: () => void;
  children?: ReactNode;
}

export function AccountSection(props: AccountSectionProps) {
  const { account, count, emptyLabel, onOpenSettings, children } = props;
  const answered = account.status === 'ok';
  const provider = PROVIDER_LABEL[account.provider];
  const meta = count && answered ? provider + SEPARATOR + count : provider;
  const empty = <p className="text-muted-foreground text-sm">{emptyLabel}</p>;
  return (
    <section
      aria-label={accountName(account)}
      className="border-border/60 rounded-lg border px-2.5 py-2 sm:px-3"
    >
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-2 gap-y-0.5">
        <p className="text-foreground min-w-0 flex-1 truncate text-xs font-semibold">
          {account.accountLabel ?? provider}
        </p>
        <p className="text-muted-foreground shrink-0 text-[11px] tracking-wider uppercase">
          {meta}
        </p>
      </div>
      {answered ? (
        (children ?? empty)
      ) : (
        <AccountIssues issues={[account]} onOpenSettings={onOpenSettings} />
      )}
    </section>
  );
}
