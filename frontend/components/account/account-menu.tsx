'use client';

/**
 * Top-right account pill: who is signed in (display name, role) and links to
 * the account page and, for administrators, the admin panel.
 *
 * Renders nothing in legacy single-user deployments. Everything it shows comes
 * from `/api/auth/me`, which never includes an email or phone number.
 */
import { useEffect, useState } from 'react';
import { UserCircle } from '@phosphor-icons/react/dist/ssr';
import { apiRequest } from './api-client';

interface Me {
  configured: boolean;
  authenticated: boolean;
  passwordLogin?: boolean;
  mustChangePassword?: boolean;
  user?: { userId: string; displayName: string; role: 'admin' | 'member' };
  reason?: string;
}

const REASON_TEXT: Record<string, string> = {
  not_signed_in: 'Not signed in',
  session_expired: 'Session expired',
  invalid_assertion: 'Sign-in could not be verified',
  no_account: 'No account for this identity',
  suspended: 'Account suspended',
  unavailable: 'Identity service unavailable',
};

export function AccountMenu() {
  const [me, setMe] = useState<Me | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/auth/me', { cache: 'no-store', credentials: 'same-origin' })
      .then((response) => (response.ok ? response.json() : null))
      .then((data: Me | null) => {
        if (!cancelled) setMe(data);
      })
      .catch(() => {
        if (!cancelled) setMe(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function signOut() {
    const result = await apiRequest<{ next?: string }>('/api/auth/logout', { method: 'POST' });
    // Whether or not the backend answered, this browser is done with the
    // session: a full navigation drops any in-memory state with it.
    window.location.assign(result.ok ? (result.data.next ?? '/login') : '/login');
  }

  if (!me || !me.configured) {
    return null;
  }

  const label =
    me.authenticated && me.user ? me.user.displayName : (REASON_TEXT[me.reason ?? ''] ?? 'Not signed in');

  return (
    <nav
      aria-label="Account"
      className="bg-background/80 border-input text-muted-foreground fixed top-4 right-20 z-40 flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs backdrop-blur"
    >
      <UserCircle className="size-4" weight="bold" aria-hidden />
      <span className="text-foreground max-w-40 truncate font-medium">{label}</span>
      {me.authenticated && (
        <a href="/account" className="hover:text-foreground underline-offset-4 hover:underline">
          Account
        </a>
      )}
      {me.authenticated && me.user?.role === 'admin' && (
        <a href="/admin" className="hover:text-foreground underline-offset-4 hover:underline">
          Admin
        </a>
      )}
      {me.authenticated ? (
        <button
          type="button"
          onClick={signOut}
          className="hover:text-foreground underline-offset-4 hover:underline"
        >
          Sign out
        </button>
      ) : (
        me.passwordLogin && (
          <a href="/login" className="hover:text-foreground underline-offset-4 hover:underline">
            Sign in
          </a>
        )
      )}
    </nav>
  );
}
