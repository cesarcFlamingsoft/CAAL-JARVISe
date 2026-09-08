'use client';

/**
 * The connected accounts that could not answer a dashboard feed, each named
 * with its state in plain words. Rendered by the calendar and inbox widgets so
 * an account that needs reconnecting is never silently missing from either.
 * Nothing here is interpolated from a provider.
 */
import { Button } from '@/components/livekit/button';
import { type FeedAccount, describeAccountStatus } from '@/lib/dashboard/provider-data';

export const PROVIDER_LABEL = { google: 'Google', microsoft: 'Microsoft', zoho: 'Zoho' } as const;

/** Whether the person can put this right themselves under Settings. */
function fixable(status: FeedAccount['status']): boolean {
  return status === 'reconnect_required' || status === 'insufficient_scope';
}

export function accountName(account: FeedAccount): string {
  const provider = PROVIDER_LABEL[account.provider];
  return account.accountLabel ? provider + ' - ' + account.accountLabel : provider;
}

interface AccountIssuesProps {
  issues: FeedAccount[];
  onOpenSettings: () => void;
}

export function AccountIssues({ issues, onOpenSettings }: AccountIssuesProps) {
  if (issues.length === 0) return null;
  return (
    <ul aria-label="Accounts needing attention" className="space-y-1.5">
      {issues.map((account) => (
        <li
          key={account.connectionId}
          className="border-border/70 rounded-lg border border-dashed px-3 py-2 text-xs"
        >
          <p className="text-foreground font-medium">{accountName(account)}</p>
          <p className="text-muted-foreground mt-0.5">{describeAccountStatus(account)}</p>
          {fixable(account.status) && (
            <div className="mt-1.5">
              <Button variant="outline" size="sm" onClick={onOpenSettings}>
                Open settings
              </Button>
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}
