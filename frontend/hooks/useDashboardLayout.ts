'use client';

import { useCallback, useEffect, useState } from 'react';
import {
  DEFAULT_LAYOUT,
  type Layout,
  type LayoutScope,
  layoutStorageKey,
  parseLayout,
  serializeLayout,
} from '@/lib/dashboard/layout';

function readStored(key: string): Layout {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? parseLayout(JSON.parse(raw)) : DEFAULT_LAYOUT;
  } catch {
    return DEFAULT_LAYOUT;
  }
}

function writeStored(key: string, layout: Layout): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(serializeLayout(layout)));
  } catch {
    // Private mode or quota: the arrangement still applies to this page view.
  }
}

function clearStored(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // Nothing to clear, or storage is unavailable.
  }
}

export interface DashboardLayoutController {
  layout: Layout;
  /** False until the layout for the current scope has been read. */
  ready: boolean;
  /** False when there is no scope to save under, so edits last for this view only. */
  persisted: boolean;
  /** Show an in-progress arrangement without saving it (drag and resize). */
  preview: (layout: Layout) => void;
  /** Apply and save an arrangement. */
  commit: (layout: Layout) => void;
  reset: () => void;
}

/**
 * The widget arrangement for one scope, kept in this browser's storage.
 *
 * `scope` is `undefined` while identity is still resolving (nothing is shown
 * yet, to avoid flashing the wrong person's layout), `null` when it could not
 * be resolved (defaults, unsaved), or a scope key to load and save under.
 */
export function useDashboardLayout(
  scope: LayoutScope | null | undefined
): DashboardLayoutController {
  const [layout, setLayout] = useState<Layout>(DEFAULT_LAYOUT);
  const [loaded, setLoaded] = useState<LayoutScope | null | undefined>(undefined);

  useEffect(() => {
    if (scope === undefined) return;
    if (scope === null) {
      setLayout(DEFAULT_LAYOUT);
      setLoaded(null);
      return;
    }
    const key = layoutStorageKey(scope);
    setLayout(readStored(key));
    setLoaded(scope);

    // Another tab of the same browser rearranged the same scope.
    const onStorage = (event: StorageEvent) => {
      if (event.key === key) setLayout(readStored(key));
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, [scope]);

  const commit = useCallback(
    (next: Layout) => {
      setLayout(next);
      if (scope) writeStored(layoutStorageKey(scope), next);
    },
    [scope]
  );

  const reset = useCallback(() => {
    setLayout(DEFAULT_LAYOUT);
    if (scope) clearStored(layoutStorageKey(scope));
  }, [scope]);

  return {
    layout,
    ready: scope !== undefined && loaded === scope,
    persisted: typeof scope === 'string',
    preview: setLayout,
    commit,
    reset,
  };
}
