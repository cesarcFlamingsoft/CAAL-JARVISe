'use client';

/**
 * The sign-in form.
 *
 * The password is posted once and never kept: no local storage, no query
 * string, no browser history. The response carries no session token -- the
 * server sets an HttpOnly cookie the page cannot read -- so there is nothing
 * here for an injected script to steal.
 *
 * Every refusal renders the same message. The server already refuses to say
 * whether an address exists, and repeating a distinct message here would
 * undo that.
 */
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiRequest, explain, loadCsrfToken } from '@/components/account/api-client';

interface LoginResponse {
  ok: boolean;
  next?: string;
  mustChangePassword?: boolean;
}

export function LoginForm({ next }: { next: string }) {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch the CSRF token up front so the first submission is not a round trip
  // slower than the rest.
  useEffect(() => {
    void loadCsrfToken();
  }, []);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);

    const result = await apiRequest<LoginResponse>('/api/auth/login', {
      method: 'POST',
      body: { email, password, next },
    });

    setPassword('');
    if (!result.ok) {
      setError(explain(result.error));
      setBusy(false);
      return;
    }
    // A full navigation, not a client-side push: the new session cookie must
    // be picked up by a fresh server render.
    window.location.assign(result.data.next ?? '/');
    void router;
  }

  return (
    <form onSubmit={onSubmit} className="flex flex-col gap-4" noValidate>
      <div className="flex flex-col gap-1.5">
        <label htmlFor="email" className="text-sm font-medium">
          Email
        </label>
        <input
          id="email"
          name="email"
          type="email"
          autoComplete="username"
          required
          autoFocus
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="border-input bg-background focus-visible:ring-ring rounded-md border px-3 py-2 text-sm focus-visible:ring-2 focus-visible:outline-none"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <label htmlFor="password" className="text-sm font-medium">
          Password
        </label>
        <input
          id="password"
          name="password"
          type="password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="border-input bg-background focus-visible:ring-ring rounded-md border px-3 py-2 text-sm focus-visible:ring-2 focus-visible:outline-none"
        />
      </div>

      {error && (
        <p role="alert" className="text-destructive text-sm">
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={busy || !email || !password}
        className="bg-primary text-primary-foreground rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50"
      >
        {busy ? 'Signing in…' : 'Sign in'}
      </button>

      <p className="text-muted-foreground text-xs">
        Forgotten your password? There is no self-service reset: ask an administrator to issue you a
        new one-time password.
      </p>
    </form>
  );
}
