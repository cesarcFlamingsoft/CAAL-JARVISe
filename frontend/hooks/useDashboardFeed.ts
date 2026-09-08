'use client';

import { useCallback, useEffect, useState } from 'react';

export type FeedState<T> =
  | { status: 'loading' }
  | { status: 'ready'; data: T }
  /** Identity is configured and this browser is not a signed-in user. */
  | { status: 'unauthorized'; code: string }
  /** A single-user deployment: there are no per-user accounts to read. */
  | { status: 'unconfigured' }
  | { status: 'error'; code: string };

export type FeedController<T> = FeedState<T> & { reload: () => void };

export type FeedPath = '/api/dashboard/calendar' | '/api/dashboard/inbox';

/** How often a visible dashboard re-reads a feed on its own. */
const REFRESH_INTERVAL_MS = 60_000;

/**
 * One dashboard feed from the BFF, reduced by `parse`, re-read on a timer
 * while the page is visible, whenever settings or connected accounts change,
 * and on demand. Nothing here reads the voice session: a call is docked
 * inside the dashboard, never a condition for showing someone their mail.
 */
export function useDashboardFeed<T>(
  path: FeedPath,
  parse: (data: unknown) => T | null
): FeedController<T> {
  const [state, setState] = useState<FeedState<T>>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);

  const reload = useCallback(() => setAttempt((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch(path, { cache: 'no-store', credentials: 'same-origin' });
        let body: unknown = null;
        try {
          body = await response.json();
        } catch {
          body = null;
        }
        if (cancelled) return;
        const code = (body as { error?: unknown } | null)?.error;
        const errorCode = typeof code === 'string' ? code : 'http_' + response.status;
        if (response.status === 401 || response.status === 403) {
          setState({ status: 'unauthorized', code: errorCode });
          return;
        }
        if (response.status === 503 && errorCode === 'identity_not_configured') {
          setState({ status: 'unconfigured' });
          return;
        }
        if (!response.ok) {
          setState({ status: 'error', code: errorCode });
          return;
        }
        const data = parse(body);
        setState(data ? { status: 'ready', data } : { status: 'error', code: 'malformed' });
      } catch {
        if (!cancelled) setState({ status: 'error', code: 'network' });
      }
    };
    void load();
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') reload();
    }, REFRESH_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [attempt, path, parse, reload]);

  // Refresh immediately when the user comes back, rather than waiting for the
  // next minute tick after a laptop wake, tab switch, or mobile app resume.
  useEffect(() => {
    const refreshWhenVisible = () => {
      if (document.visibilityState === 'visible') reload();
    };
    document.addEventListener('visibilitychange', refreshWhenVisible);
    window.addEventListener('focus', refreshWhenVisible);
    return () => {
      document.removeEventListener('visibilitychange', refreshWhenVisible);
      window.removeEventListener('focus', refreshWhenVisible);
    };
  }, [reload]);

  // Settings saves and account connects or disconnects announce themselves.
  useEffect(() => {
    window.addEventListener('settings-updated', reload);
    window.addEventListener('connections-updated', reload);
    return () => {
      window.removeEventListener('settings-updated', reload);
      window.removeEventListener('connections-updated', reload);
    };
  }, [reload]);

  return { ...state, reload };
}
