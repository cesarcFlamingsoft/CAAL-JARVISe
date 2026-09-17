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
import { credentialToJson, requestOptionsFromJson } from '@/lib/auth/passkeys';

interface LoginResponse {
  ok: boolean;
  next?: string;
  mustChangePassword?: boolean;
}

export function LoginForm({ next, passkeys }: { next: string; passkeys: boolean }) {
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

  async function onPasskey() {
    if (busy) return;
    if (!window.PublicKeyCredential || !navigator.credentials) {
      setError(
        'Passkeys are not available in this browser. You can still sign in with your password.'
      );
      return;
    }
    setBusy(true);
    setError(null);
    const begun = await apiRequest<{ ceremonyId: string; publicKey: Record<string, unknown> }>(
      '/api/auth/passkey/options',
      { method: 'POST' }
    );
    if (!begun.ok) {
      setError(explain(begun.error));
      setBusy(false);
      return;
    }
    try {
      const credential = (await navigator.credentials.get({
        publicKey: requestOptionsFromJson(begun.data.publicKey),
      })) as PublicKeyCredential | null;
      if (!credential) throw new Error('cancelled');
      const verified = await apiRequest<{ next?: string }>('/api/auth/passkey/verify', {
        method: 'POST',
        body: {
          ceremonyId: begun.data.ceremonyId,
          credential: credentialToJson(credential),
          next,
        },
      });
      if (!verified.ok) {
        setError(explain(verified.error));
        setBusy(false);
        return;
      }
      window.location.assign(verified.data.next ?? '/');
    } catch {
      setError(
        'Passkey sign-in was cancelled or could not be completed. Your password still works.'
      );
      setBusy(false);
    }
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

      {passkeys && (
        <>
          <div className="flex items-center gap-3" aria-hidden="true">
            <span className="bg-border h-px flex-1" />
            <span className="text-muted-foreground text-xs uppercase">or</span>
            <span className="bg-border h-px flex-1" />
          </div>

          <button
            type="button"
            disabled={busy}
            onClick={onPasskey}
            className="border-primary/50 text-primary hover:bg-primary/10 rounded-md border px-4 py-2 text-sm font-medium disabled:opacity-50"
          >
            Sign in with passkey
          </button>

          <p className="text-muted-foreground text-xs">
            Passkeys use your device&apos;s secure biometric or screen-lock prompt. Face ID, Touch
            ID, and other device checks stay on your device—FRIDAY never receives biometric data.
            Password sign-in remains available.
          </p>
        </>
      )}

      <p className="text-muted-foreground text-xs">
        Forgotten your password? There is no self-service reset: ask an administrator to issue you a
        new one-time password.
      </p>
    </form>
  );
}
