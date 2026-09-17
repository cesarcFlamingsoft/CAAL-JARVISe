'use client';

import { useCallback, useEffect, useState } from 'react';
import { Button } from '@/components/livekit/button';
import { creationOptionsFromJson, credentialToJson } from '@/lib/auth/passkeys';
import { apiRequest, explain } from './api-client';

interface Passkey {
  id: string;
  label: string;
  createdAt: number;
  lastUsedAt: number | null;
}

function date(value: number | null) {
  return value === null ? 'Never' : new Date(value * 1000).toLocaleDateString();
}

export function PasskeyPanel() {
  const [passkeys, setPasskeys] = useState<Passkey[]>([]);
  const [label, setLabel] = useState('This device');
  const [currentPassword, setCurrentPassword] = useState('');
  const [removeTarget, setRemoveTarget] = useState<Passkey | null>(null);
  const [removePassword, setRemovePassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [available, setAvailable] = useState(true);
  const [message, setMessage] = useState<string | null>(null);

  const load = useCallback(async () => {
    const result = await apiRequest<{ passkeys: Passkey[] }>('/api/account/passkeys');
    if (result.ok) setPasskeys(result.data.passkeys);
    else if (result.status === 404) setAvailable(false);
    else setMessage(explain(result.error));
  }, []);

  useEffect(() => void load(), [load]);

  async function add() {
    if (busy || !label.trim() || !currentPassword) return;
    if (!window.PublicKeyCredential || !navigator.credentials) {
      setMessage('Passkeys are not available in this browser.');
      return;
    }
    setBusy(true);
    setMessage(null);
    const begun = await apiRequest<{ ceremonyId: string; publicKey: Record<string, unknown> }>(
      '/api/account/passkeys/options',
      { method: 'POST', body: { label, currentPassword } }
    );
    setCurrentPassword('');
    if (!begun.ok) {
      setMessage(explain(begun.error));
      setBusy(false);
      return;
    }
    try {
      const credential = (await navigator.credentials.create({
        publicKey: creationOptionsFromJson(begun.data.publicKey),
      })) as PublicKeyCredential | null;
      if (!credential) throw new Error('cancelled');
      const finished = await apiRequest('/api/account/passkeys/verify', {
        method: 'POST',
        body: {
          ceremonyId: begun.data.ceremonyId,
          label,
          credential: credentialToJson(credential),
        },
      });
      if (!finished.ok) setMessage(explain(finished.error));
      else {
        setMessage('Passkey added. Your password remains available.');
        await load();
      }
    } catch {
      setMessage('The device prompt was cancelled or could not be completed.');
    }
    setBusy(false);
  }

  async function rename(passkey: Passkey) {
    const next = window.prompt('Passkey name', passkey.label)?.trim();
    if (!next || next === passkey.label) return;
    const result = await apiRequest(`/api/account/passkeys/${passkey.id}`, {
      method: 'PATCH',
      body: { label: next },
    });
    if (result.ok) await load();
    else setMessage(explain(result.error));
  }

  async function revoke() {
    if (!removeTarget || !removePassword || busy) return;
    setBusy(true);
    const target = removeTarget;
    const result = await apiRequest(`/api/account/passkeys/${target.id}`, {
      method: 'DELETE',
      body: { currentPassword: removePassword },
    });
    setRemovePassword('');
    if (result.ok) await load();
    else setMessage(explain(result.error));
    if (result.ok) setRemoveTarget(null);
    setBusy(false);
  }

  if (!available) return null;
  return (
    <section className="friday-panel space-y-4" aria-labelledby="passkeys-heading">
      <div>
        <h2 id="passkeys-heading" className="text-sm font-semibold tracking-wider uppercase">
          Passkeys
        </h2>
        <p className="text-muted-foreground mt-1 text-xs">
          Use this device&apos;s Face ID, Touch ID, or screen lock. The biometric check stays on the
          device; FRIDAY stores only a public key. Your password remains a fallback.
        </p>
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        <div>
          <label className="text-muted-foreground text-xs" htmlFor="passkey-label">
            Passkey name
          </label>
          <input
            id="passkey-label"
            value={label}
            maxLength={64}
            onChange={(event) => setLabel(event.target.value)}
            className="border-input bg-background mt-1 w-full rounded-lg border px-3 py-2 text-sm"
          />
        </div>
        <div>
          <label className="text-muted-foreground text-xs" htmlFor="passkey-current-password">
            Current password
          </label>
          <input
            id="passkey-current-password"
            type="password"
            autoComplete="current-password"
            value={currentPassword}
            maxLength={256}
            onChange={(event) => setCurrentPassword(event.target.value)}
            className="border-input bg-background mt-1 w-full rounded-lg border px-3 py-2 text-sm"
          />
        </div>
        <Button
          variant="primary"
          size="sm"
          onClick={add}
          disabled={busy || !label.trim() || !currentPassword}
        >
          {busy ? 'Waiting…' : 'Add passkey'}
        </Button>
      </div>
      {passkeys.length === 0 ? (
        <p className="text-muted-foreground text-xs">No passkeys added.</p>
      ) : (
        <ul className="space-y-2">
          {passkeys.map((passkey) => (
            <li
              key={passkey.id}
              className="border-border flex items-center justify-between rounded-lg border p-3 text-sm"
            >
              <div>
                <p className="font-medium">{passkey.label}</p>
                <p className="text-muted-foreground text-xs">
                  Added {date(passkey.createdAt)} · Last used {date(passkey.lastUsedAt)}
                </p>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={() => rename(passkey)}>
                  Rename
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setRemovePassword('');
                    setRemoveTarget(passkey);
                  }}
                >
                  Remove
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}
      {removeTarget && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="remove-passkey-heading"
          className="border-border space-y-3 rounded-lg border p-3"
        >
          <h3 id="remove-passkey-heading" className="text-sm font-medium">
            Remove {removeTarget.label}?
          </h3>
          <p className="text-muted-foreground text-xs">
            Enter your current password to remove this passkey. Your password will still work.
          </p>
          <label className="text-muted-foreground block text-xs" htmlFor="remove-passkey-password">
            Current password
          </label>
          <input
            id="remove-passkey-password"
            type="password"
            autoComplete="current-password"
            autoFocus
            value={removePassword}
            maxLength={256}
            onChange={(event) => setRemovePassword(event.target.value)}
            className="border-input bg-background w-full rounded-lg border px-3 py-2 text-sm"
          />
          <div className="flex gap-2">
            <Button variant="primary" size="sm" onClick={revoke} disabled={busy || !removePassword}>
              {busy ? 'Removing…' : 'Remove passkey'}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setRemovePassword('');
                setRemoveTarget(null);
              }}
              disabled={busy}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}
      {message && (
        <p role="status" className="text-muted-foreground text-xs">
          {message}
        </p>
      )}
    </section>
  );
}
