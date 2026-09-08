'use client';

import { useCallback, useEffect, useState } from 'react';
import type { DashboardCapabilities } from '@/lib/dashboard/capabilities';

export type CapabilitiesState =
  | { status: 'loading' }
  | { status: 'ready'; data: DashboardCapabilities }
  /** Identity is configured and this browser is not a signed-in user. */
  | { status: 'unauthorized'; code: string }
  | { status: 'error'; code: string };

/** The credential-free capability summary, refreshed whenever settings are saved. */
export function useDashboardCapabilities(): CapabilitiesState & { reload: () => void } {
  const [state, setState] = useState<CapabilitiesState>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);

  const reload = useCallback(() => setAttempt((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch('/api/dashboard/capabilities', {
          cache: 'no-store',
          credentials: 'same-origin',
        });
        let body: unknown = null;
        try {
          body = await response.json();
        } catch {
          body = null;
        }
        if (cancelled) return;
        const code = (body as { error?: unknown } | null)?.error;
        const errorCode = typeof code === 'string' ? code : `http_${response.status}`;
        if (response.status === 401 || response.status === 403) {
          setState({ status: 'unauthorized', code: errorCode });
          return;
        }
        if (!response.ok) {
          setState({ status: 'error', code: errorCode });
          return;
        }
        const data = (body as { capabilities?: DashboardCapabilities } | null)?.capabilities;
        setState(data ? { status: 'ready', data } : { status: 'error', code: 'malformed' });
      } catch {
        if (!cancelled) setState({ status: 'error', code: 'network' });
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  // The settings panel announces a save with this event; sources may have changed.
  useEffect(() => {
    window.addEventListener('settings-updated', reload);
    return () => window.removeEventListener('settings-updated', reload);
  }, [reload]);

  return { ...state, reload };
}
