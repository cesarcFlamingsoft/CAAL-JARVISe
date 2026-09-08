'use client';

/**
 * Settings → Integrations → Connected accounts.
 *
 * The signed-in user's own provider accounts, linked by approving access at
 * the provider itself. One row per provider listing every linked account (a
 * person may link two Google accounts, say); Connect -- or "Connect another
 * account" -- is offered only when the operator has configured that provider
 * and the backend says it can finish the exchange. Disconnect applies to one
 * account and asks for an explicit confirmation before anything is sent.
 * What the panel says about the token exchange comes from the backend, not
 * from optimism: while the backend reports it cannot complete one, the panel
 * says so and offers nothing it cannot deliver.
 */
import { useCallback, useEffect, useState } from 'react';
import { CircleNotch } from '@phosphor-icons/react/dist/ssr';
import { apiRequest, explain } from '@/components/account/api-client';
import { Button } from '@/components/livekit/button';
import {
  type BrowserConnection,
  type BrowserConnectionList,
  type ConnectionRow,
  type Provider,
  explainConnectionError,
  panelRows,
} from '@/lib/connections/protocol';

interface StartResponse {
  provider: Provider;
  authorizationUrl: string;
  expiresAt: number;
}

interface Notice {
  provider: Provider;
  tone: 'error' | 'info';
  text: string;
}

function describeError(error: string, details?: Record<string, unknown>): string {
  return (
    explainConnectionError(error, { missing: details?.missing, settings: details?.settings }) ??
    explain(error)
  );
}

function formatWhen(seconds: number | null): string | null {
  if (seconds === null) return null;
  try {
    return new Date(seconds * 1000).toLocaleString();
  } catch {
    return null;
  }
}

export function ConnectedAccounts() {
  const [list, setList] = useState<BrowserConnectionList | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [starting, setStarting] = useState<Provider | null>(null);
  /** The connection id whose Disconnect is awaiting the user's confirmation. */
  const [confirming, setConfirming] = useState<string | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    const result = await apiRequest<BrowserConnectionList>('/api/connections');
    setLoading(false);
    if (result.ok) {
      setList(result.data);
      setLoadError(null);
    } else {
      setList(null);
      setLoadError(describeError(result.error, result.details));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function connect(row: ConnectionRow) {
    if (!row.canConnect || starting !== null) return;
    setStarting(row.provider);
    setNotice(null);
    const result = await apiRequest<StartResponse>(`/api/connections/start/${row.provider}`, {
      method: 'POST',
    });
    if (!result.ok || !/^https:\/\//.test(result.data.authorizationUrl ?? '')) {
      setStarting(null);
      setNotice({
        provider: row.provider,
        tone: 'error',
        text: result.ok
          ? describeError('backend_unavailable')
          : describeError(result.error, result.details),
      });
      return;
    }
    // A top-level navigation to the provider. The round trip ends on
    // /connections/result, which says exactly what happened.
    window.location.assign(result.data.authorizationUrl);
  }

  async function confirmDisconnect(row: ConnectionRow, connection: BrowserConnection) {
    if (confirming !== connection.connectionId || removing !== null) return;
    setRemoving(connection.connectionId);
    setNotice(null);
    const result = await apiRequest<null>(`/api/connections/${connection.connectionId}`, {
      method: 'DELETE',
    });
    setRemoving(null);
    setConfirming(null);
    // The dashboard feeds re-read themselves when an account goes away.
    if (result.ok) window.dispatchEvent(new Event('connections-updated'));
    setNotice(
      result.ok
        ? {
            provider: row.provider,
            tone: 'info',
            text: `${row.label}${connection.accountLabel ? ` (${connection.accountLabel})` : ''} was disconnected. JARVIS no longer holds any access for it.`,
          }
        : { provider: row.provider, tone: 'error', text: describeError(result.error, result.details) }
    );
    await load();
  }

  const rows = list ? panelRows(list) : [];

  return (
    <div className="overflow-hidden rounded-xl border">
      <div className="bg-muted/50 border-b px-4 py-3">
        <span className="font-semibold">Connected accounts</span>
        <p className="text-muted-foreground text-xs">
          Your own Google, Microsoft / Outlook and Zoho accounts, linked by approving access at the
          provider. You can link more than one account from the same provider. JARVIS never asks
          you for your sign-in details here.
        </p>
      </div>

      <div className="space-y-3 p-4">
        {list && !list.tokenExchangeAvailable && (
          <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3">
            <p className="text-sm font-medium text-amber-200">Connecting is not available</p>
            <p className="text-muted-foreground mt-1 text-xs">
              The JARVIS backend reports that it cannot complete the token exchange with any
              provider right now, usually because no provider has been configured by the operator.
              Nothing can be connected until that changes.
            </p>
          </div>
        )}

        {loading && !list && (
          <p className="text-muted-foreground flex items-center gap-2 text-sm">
            <CircleNotch className="h-4 w-4 animate-spin" aria-hidden />
            Loading your connected accounts…
          </p>
        )}

        {loadError && !list && <p className="text-destructive text-sm">{loadError}</p>}

        {rows.map((row) => {
          const rowNotice = notice && notice.provider === row.provider ? notice : null;
          const connectLabel = row.connections.length > 0 ? 'Connect another account' : 'Connect';
          return (
            <div key={row.provider} className="space-y-2 rounded-lg border p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-sm font-medium">{row.label}</p>
                  <p className="text-muted-foreground truncate text-xs">
                    {row.status === 'connected' &&
                      (row.connections.length === 1
                        ? '1 account connected'
                        : `${row.connections.length} accounts connected`)}
                    {row.status === 'not_connected' && 'Not connected'}
                    {row.status === 'not_configured' &&
                      'Not set up on this server. Ask the operator to configure this provider.'}
                  </p>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <Button
                    variant={row.status === 'connected' ? 'secondary' : 'primary'}
                    size="sm"
                    onClick={() => void connect(row)}
                    disabled={!row.canConnect || starting !== null}
                    title={
                      row.configured
                        ? row.canConnect
                          ? undefined
                          : 'The backend cannot complete the token exchange right now'
                        : 'This provider is not configured on this server'
                    }
                  >
                    {starting === row.provider ? (
                      <>
                        <CircleNotch className="h-4 w-4 animate-spin" aria-hidden />
                        Opening…
                      </>
                    ) : (
                      connectLabel
                    )}
                  </Button>
                </div>
              </div>

              {row.connections.map((connection) => {
                const isConfirming = confirming === connection.connectionId;
                const isRemoving = removing === connection.connectionId;
                const since = formatWhen(connection.connectedAt);
                return (
                  <div
                    key={connection.connectionId}
                    className="bg-muted/30 space-y-2 rounded-md border px-3 py-2"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-muted-foreground min-w-0 truncate text-xs">
                        Connected{connection.accountLabel ? ` as ${connection.accountLabel}` : ''}
                        {since ? ` · since ${since}` : ''}
                      </p>
                      {!isConfirming && (
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => {
                            setNotice(null);
                            setConfirming(connection.connectionId);
                          }}
                          disabled={removing !== null}
                        >
                          Disconnect…
                        </Button>
                      )}
                    </div>

                    {isConfirming && (
                      <div
                        role="alertdialog"
                        aria-label={`Disconnect ${row.label}`}
                        className="border-destructive/40 bg-destructive/5 space-y-2 rounded-lg border p-3"
                      >
                        <p className="text-sm">
                          Disconnect {row.label}
                          {connection.accountLabel ? ` (${connection.accountLabel})` : ''}? JARVIS
                          will delete the access it holds for this account. Your other accounts
                          stay connected. You can connect it again later.
                        </p>
                        <div className="flex gap-2">
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={() => void confirmDisconnect(row, connection)}
                            disabled={isRemoving}
                          >
                            {isRemoving ? 'Disconnecting…' : 'Yes, disconnect'}
                          </Button>
                          <Button
                            variant="secondary"
                            size="sm"
                            onClick={() => setConfirming(null)}
                            disabled={isRemoving}
                          >
                            Cancel
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}

              {rowNotice && (
                <p
                  className={`text-xs ${
                    rowNotice.tone === 'error' ? 'text-destructive' : 'text-muted-foreground'
                  }`}
                >
                  {rowNotice.text}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
