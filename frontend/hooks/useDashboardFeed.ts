'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { type FeedState, createRefreshGate, reduceFeed } from '@/lib/dashboard/refresh';
import { SCHEDULED_EVENT_NAME } from '@/lib/dashboard/scheduled-events';
import { workRefreshInterval } from '@/lib/dashboard/work';

export type { FeedState } from '@/lib/dashboard/refresh';
export type FeedController<T> = FeedState<T> & { reload: () => void };
export type FeedPath =
  | '/api/dashboard/calendar'
  | '/api/dashboard/inbox'
  | '/api/dashboard/reminders'
  | '/api/dashboard/work';
const REFRESH_INTERVAL_MS = 60_000;

// Only in-flight responses are shared; no personal data is cached between mounts.
const requests = new Map<string, Promise<Response>>();
function request(path: string) {
  let pending = requests.get(path);
  if (!pending) {
    pending = fetch(path, {
      cache: 'no-store',
      credentials: 'same-origin',
      signal: AbortSignal.timeout(45_000),
    });
    requests.set(path, pending);
    void pending
      .finally(() => {
        if (requests.get(path) === pending) requests.delete(path);
      })
      .catch(() => {});
  }
  return pending.then((response) => response.clone());
}

/** Stable timers, one in-flight request, bounded focus refresh, and no hidden-tab polling. */
export function useDashboardFeed<T>(
  path: FeedPath,
  parse: (data: unknown) => T | null
): FeedController<T> {
  const [state, setState] = useState<FeedState<T>>({ status: 'loading' });
  const refreshRef = useRef<() => void>(() => {});
  const reload = useCallback(() => refreshRef.current(), []);
  useEffect(() => {
    let cancelled = false;
    let generation = 0;
    let interval = REFRESH_INTERVAL_MS;
    let timer: number;
    const load = async () => {
      const version = generation;
      setState((current) =>
        current.status === 'ready' ? { ...current, refreshing: true } : current
      );
      let next: FeedState<T>;
      try {
        const response = await request(path);
        const body = await response.json().catch(() => null);
        const code = typeof body?.error === 'string' ? body.error : 'http_' + response.status;
        if (response.status === 401 || response.status === 403)
          next = { status: 'unauthorized', code };
        else if (response.status === 503 && code === 'identity_not_configured')
          next = { status: 'unconfigured' };
        else if (!response.ok) next = { status: 'error', code };
        else {
          const data = parse(body);
          next = data
            ? { status: 'ready', data, updatedAt: Date.now() }
            : { status: 'error', code: 'malformed' };
        }
      } catch {
        next = { status: 'error', code: 'network' };
      }
      if (!cancelled && version === generation) {
        setState((current) => reduceFeed(current, next));
        if (path === '/api/dashboard/work' && next.status === 'ready') {
          const nextInterval = workRefreshInterval(next.data);
          if (nextInterval !== interval) {
            interval = nextInterval;
            window.clearInterval(timer);
            timer = window.setInterval(refresh, interval);
          }
        }
      }
    };
    const visible = () => document.visibilityState === 'visible';
    let gate = createRefreshGate(load, visible);
    const refresh = () => {
      void gate.refresh();
    };
    // Changed account scope must immediately discard old summaries and old responses.
    const invalidate = () => {
      generation++;
      requests.delete(path);
      setState({ status: 'loading' });
      gate = createRefreshGate(load, visible);
      refresh();
    };
    refreshRef.current = refresh;
    refresh();
    timer = window.setInterval(refresh, interval);
    document.addEventListener('visibilitychange', refresh);
    window.addEventListener('focus', refresh);
    window.addEventListener('settings-updated', invalidate);
    window.addEventListener('connections-updated', invalidate);
    if (path === '/api/dashboard/reminders') window.addEventListener(SCHEDULED_EVENT_NAME, refresh);
    return () => {
      cancelled = true;
      refreshRef.current = () => {};
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', refresh);
      window.removeEventListener('focus', refresh);
      window.removeEventListener('settings-updated', invalidate);
      window.removeEventListener('connections-updated', invalidate);
      window.removeEventListener(SCHEDULED_EVENT_NAME, refresh);
    };
  }, [path, parse]);
  return { ...state, reload };
}
