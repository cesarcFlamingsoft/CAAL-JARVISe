import type { MeState } from '@/hooks/useMe';

export type WorkspaceMode = 'loading' | 'signed-out' | 'personal';

/** Resolve access before any account-scoped dashboard component is mounted. */
export function workspaceMode(me: MeState): WorkspaceMode {
  if (me.status === 'loading') return 'loading';
  if (me.status === 'ready' && me.me.configured && !me.me.authenticated) return 'signed-out';
  return 'personal';
}
