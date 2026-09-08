'use client';

import { useEffect, useState } from 'react';

/** The browser-safe identity summary from `/api/auth/me`: no email, no token. */
export interface Me {
  configured: boolean;
  authenticated: boolean;
  passwordLogin: boolean;
  mustChangePassword: boolean;
  user?: { userId: string; displayName: string; role: 'admin' | 'member' };
  reason?: string;
}

export type MeState = { status: 'loading' } | { status: 'ready'; me: Me } | { status: 'error' };

function normalize(data: unknown): Me {
  const raw = (data ?? {}) as Record<string, unknown>;
  const user = raw.user as Record<string, unknown> | undefined;
  const me: Me = {
    configured: raw.configured === true,
    authenticated: raw.authenticated === true,
    passwordLogin: raw.passwordLogin === true,
    mustChangePassword: raw.mustChangePassword === true,
  };
  if (me.authenticated && user && typeof user.userId === 'string' && user.userId) {
    me.user = {
      userId: user.userId,
      displayName: typeof user.displayName === 'string' ? user.displayName : '',
      role: user.role === 'admin' ? 'admin' : 'member',
    };
  } else {
    me.authenticated = false;
  }
  if (typeof raw.reason === 'string') me.reason = raw.reason;
  return me;
}

/** Who is signed in, resolved once per page load. */
export function useMe(): MeState {
  const [state, setState] = useState<MeState>({ status: 'loading' });

  useEffect(() => {
    let cancelled = false;
    fetch('/api/auth/me', { cache: 'no-store', credentials: 'same-origin' })
      .then(async (response) => (response.ok ? normalize(await response.json()) : null))
      .then((me) => {
        if (!cancelled) setState(me ? { status: 'ready', me } : { status: 'error' });
      })
      .catch(() => {
        if (!cancelled) setState({ status: 'error' });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
