'use client';

import { useCallback, useEffect, useState } from 'react';
import { apiRequest, explain } from '@/components/account/api-client';
import { Button } from '@/components/livekit/button';
import { type HAAccess, parseAccess } from '@/lib/home-assistant/contract';

const statusText = {
  denied: 'Access not granted',
  connection_required: 'Connection required',
  connected: 'Authenticated HA connection saved',
  service_account: 'Existing administrator service credential',
};
export function HomeAssistantAccess({
  userId,
  onChanged,
}: {
  userId?: string;
  onChanged?: () => void;
}) {
  const [access, setAccess] = useState<HAAccess | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const path = userId ? `/api/admin/users/${userId}/home-assistant` : '/api/home-assistant';
  const load = useCallback(async () => {
    const r = await apiRequest<unknown>(path);
    const data = r.ok ? parseAccess(r.data) : null;
    if (data) {
      setAccess(data);
      setError('');
    } else setError(r.ok ? 'Home Assistant status unavailable' : explain(r.error));
  }, [path]);
  useEffect(() => {
    void load();
  }, [load]);
  const save = async (enabled: boolean, connection_id: string | null) => {
    setBusy(true);
    const r = await apiRequest<unknown>(path, { method: 'PUT', body: { enabled, connection_id } });
    setBusy(false);
    if (r.ok) {
      await load();
      onChanged?.();
    } else setError(explain(r.error));
  };
  const connect = async () => {
    setBusy(true);
    const r = await apiRequest<{ authorizationUrl: string }>('/api/home-assistant/start', {
      method: 'POST',
    });
    if (r.ok) window.location.assign(r.data.authorizationUrl);
    else {
      setError(
        r.error === 'ha_use_public_origin'
          ? 'Open the configured public Jarvis URL and sign in there before connecting Home Assistant.'
          : explain(r.error)
      );
      setBusy(false);
    }
  };
  const disconnect = async (id: string) => {
    if (
      !window.confirm(
        'Disconnect this authenticated Home Assistant identity? Accounts using it will need another connection.'
      )
    )
      return;
    setBusy(true);
    const r = await apiRequest(`/api/home-assistant/connections/${id}`, { method: 'DELETE' });
    setBusy(false);
    if (r.ok) await load();
    else setError(explain(r.error));
  };
  return (
    <section className="mt-3 max-w-md space-y-2 text-sm" aria-label="Home Assistant access">
      <h3 className="font-medium">Home Assistant</h3>
      {access ? (
        <>
          {userId && (
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={access.enabled}
                disabled={busy}
                onChange={(e) => void save(e.target.checked, access.connection_id)}
              />
              Allow HA access
            </label>
          )}
          <p role="status" className="text-muted-foreground text-xs">
            {statusText[access.status]}
          </p>
          {userId && (
            <label className="flex flex-col gap-1 text-xs">
              Authorized HA identity
              <select
                aria-label="Authorized HA identity"
                className="border-input bg-background rounded-lg border p-2"
                disabled={busy}
                value={access.connection_id ?? ''}
                onChange={(e) => void save(access.enabled, e.target.value || null)}
              >
                <option value="">User needs to connect an HA identity</option>
                {access.connections.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                    {c.shared ? ' (shared by this admin)' : ''}
                    {c.ha_admin ? ' · HA administrator' : ''}
                  </option>
                ))}
              </select>
            </label>
          )}
          <p className="text-muted-foreground text-xs">
            Device-state reads and light on/off only, subject to the authenticated HA user’s
            permissions. Security, lock and garage actions are unavailable. Jarvis roles do not
            change HA roles.
          </p>
          {access.status === 'service_account' && (
            <p className="text-xs">
              Uses the configured HA service identity, which may have owner access. It is never
              automatically used for member accounts.
            </p>
          )}
          {!userId && (
            <>
              {!access.enabled && (
                <p className="text-xs">
                  You may connect now; an administrator must grant access before Jarvis can use it.
                </p>
              )}
              <Button size="sm" variant="secondary" disabled={busy} onClick={() => void connect()}>
                Connect Home Assistant
              </Button>
              <p className="text-muted-foreground text-xs">
                Start and finish in the same Jarvis browser session at the configured public URL.
                Sign in to the intended HA user in Home Assistant’s login screen.
              </p>
              {access.connections
                .filter((c) => !c.shared)
                .map((c) => (
                  <div key={c.id} className="flex items-center gap-2">
                    <span>{c.label}</span>
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={busy}
                      onClick={() => void disconnect(c.id)}
                    >
                      Disconnect
                    </Button>
                  </div>
                ))}
            </>
          )}
        </>
      ) : (
        <p>{error ? '' : 'Loading HA access…'}</p>
      )}
      {error && (
        <p role="alert" className="text-destructive text-xs">
          {error}
        </p>
      )}
    </section>
  );
}
