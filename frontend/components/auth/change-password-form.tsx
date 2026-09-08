'use client';

/**
 * Change-password form, used both for a routine change and for the forced
 * change after a one-time password.
 *
 * The confirmation field is checked here purely to save a round trip; the
 * policy that matters -- length, variety, and "must differ from the current
 * one" -- is enforced by the backend, which is the only place it can be.
 */
import { useState } from 'react';
import { apiRequest, explain } from '@/components/account/api-client';

const MIN_LENGTH = 12;

interface ChangeResponse {
  ok: boolean;
  next?: string;
}

export function ChangePasswordForm({ forced }: { forced: boolean }) {
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [confirm, setConfirm] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const mismatch = confirm.length > 0 && next !== confirm;
  const tooShort = next.length > 0 && next.length < MIN_LENGTH;

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (busy) return;
    if (next !== confirm) {
      setError('The two new passwords do not match.');
      return;
    }
    setBusy(true);
    setError(null);

    const result = await apiRequest<ChangeResponse>('/api/auth/password', {
      method: 'POST',
      body: { currentPassword: current, newPassword: next },
    });

    setCurrent('');
    setNext('');
    setConfirm('');
    if (!result.ok) {
      setError(explain(result.error));
      setBusy(false);
      return;
    }
    setDone(true);
    // Reload from the server so the forced-change flag clears everywhere.
    window.location.assign(result.data.next ?? '/');
  }

  if (done) {
    return <p className="text-sm">Password changed. Taking you back to JARVIS…</p>;
  }

  return (
    <form onSubmit={onSubmit} className="flex flex-col gap-4" noValidate>
      <div className="flex flex-col gap-1.5">
        <label htmlFor="current" className="text-sm font-medium">
          {forced ? 'One-time password' : 'Current password'}
        </label>
        <input
          id="current"
          type="password"
          autoComplete="current-password"
          required
          autoFocus
          value={current}
          onChange={(e) => setCurrent(e.target.value)}
          className="border-input bg-background focus-visible:ring-ring rounded-md border px-3 py-2 text-sm focus-visible:ring-2 focus-visible:outline-none"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <label htmlFor="next" className="text-sm font-medium">
          New password
        </label>
        <input
          id="next"
          type="password"
          autoComplete="new-password"
          required
          minLength={MIN_LENGTH}
          value={next}
          onChange={(e) => setNext(e.target.value)}
          className="border-input bg-background focus-visible:ring-ring rounded-md border px-3 py-2 text-sm focus-visible:ring-2 focus-visible:outline-none"
        />
        <p className="text-muted-foreground text-xs">
          At least {MIN_LENGTH} characters. A passphrase of a few words is stronger than a short
          password with symbols in it.
        </p>
      </div>

      <div className="flex flex-col gap-1.5">
        <label htmlFor="confirm" className="text-sm font-medium">
          Confirm new password
        </label>
        <input
          id="confirm"
          type="password"
          autoComplete="new-password"
          required
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          className="border-input bg-background focus-visible:ring-ring rounded-md border px-3 py-2 text-sm focus-visible:ring-2 focus-visible:outline-none"
        />
        {mismatch && (
          <p className="text-destructive text-xs">The two new passwords do not match.</p>
        )}
        {tooShort && (
          <p className="text-destructive text-xs">Must be at least {MIN_LENGTH} characters.</p>
        )}
      </div>

      {error && (
        <p role="alert" className="text-destructive text-sm">
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={busy || !current || !next || next !== confirm || next.length < MIN_LENGTH}
        className="bg-primary text-primary-foreground rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50"
      >
        {busy ? 'Changing…' : 'Change password'}
      </button>

      <p className="text-muted-foreground text-xs">
        Changing your password signs out every other device.
      </p>
    </form>
  );
}
