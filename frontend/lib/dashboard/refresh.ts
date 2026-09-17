/** Transport-independent refresh policy. No identity data survives unmount. */
export type FeedState<T> =
  | { status: 'loading' }
  | { status: 'ready'; data: T; updatedAt?: number; stale?: boolean; refreshing?: boolean }
  | { status: 'unauthorized'; code: string }
  | { status: 'unconfigured' }
  | { status: 'error'; code: string };

export function reduceFeed<T>(previous: FeedState<T>, next: FeedState<T>): FeedState<T> {
  if (next.status === 'error' && previous.status === 'ready') {
    return { ...previous, stale: true, refreshing: false };
  }
  return next;
}

export function createRefreshGate(
  load: () => Promise<void>,
  visible: () => boolean,
  now: () => number = Date.now
) {
  let pending: Promise<void> | undefined;
  let last = -Infinity;
  return {
    refresh() {
      if (pending) return pending;
      if (!visible() || now() - last < 5000) return Promise.resolve();
      last = now();
      pending = load().finally(() => {
        pending = undefined;
      });
      return pending;
    },
  };
}
