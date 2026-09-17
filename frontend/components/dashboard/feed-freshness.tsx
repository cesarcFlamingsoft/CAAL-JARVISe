import type { FeedController } from '@/hooks/useDashboardFeed';

export function FeedFreshness({ feed, now }: { feed: FeedController<unknown>; now: Date | null }) {
  if (feed.status !== 'ready') return null;
  const stale =
    feed.stale || (!!feed.updatedAt && !!now && now.getTime() - feed.updatedAt > 120_000);
  return (
    <div className="feed-freshness" data-stale={stale || undefined}>
      <span role="status">
        {stale
          ? 'Stale · last available snapshot'
          : feed.refreshing
            ? 'Refreshing…'
            : 'Updated ' +
              (feed.updatedAt
                ? new Date(feed.updatedAt).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })
                : 'just now')}
      </span>
      <button type="button" onClick={feed.reload} disabled={feed.refreshing}>
        Refresh
      </button>
    </div>
  );
}
