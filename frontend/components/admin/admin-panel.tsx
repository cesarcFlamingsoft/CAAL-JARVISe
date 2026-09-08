'use client';

/**
 * Administrator panel: users, roles, status, approved callback numbers, audit.
 *
 * This UI only ever renders what the BFF returns: opaque ids, masked email
 * hints, roles, statuses and whether a callback number is on file. Every
 * change goes through a CSRF-protected same-origin route and is authorized
 * again by the backend. Hiding a control here is never what protects it.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from '@/components/livekit/button';
import { apiRequest, explain } from '@/components/account/api-client';

interface AdminUser {
  userId: string;
  displayName: string;
  emailHint: string;
  role: 'admin' | 'member';
  status: 'active' | 'suspended';
  hasCallbackNumber: boolean;
  callbackNumberUpdatedAt: number | null;
  createdAt: number | null;
  lastSeenAt: number | null;
  createdBy: string | null;
}

interface AuditEvent {
  eventId: string;
  occurredAt: number;
  actorId: string;
  actorRole: string;
  action: string;
  targetId: string | null;
  outcome: string;
  detail: Record<string, unknown>;
}

function formatTime(seconds: number | null): string {
  if (!seconds) return '—';
  return new Date(seconds * 1000).toLocaleString();
}

function shortId(id: string | null): string {
  if (!id) return '—';
  return id.length > 12 ? `${id.slice(0, 8)}…${id.slice(-4)}` : id;
}

const inputClass = 'border-input bg-background rounded-lg border px-3 py-2 text-sm';

export function AdminPanel({ selfId }: { selfId: string }) {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const [newEmail, setNewEmail] = useState('');
  const [newName, setNewName] = useState('');
  const [newRole, setNewRole] = useState<'admin' | 'member'>('member');
  const [newWithPassword, setNewWithPassword] = useState(true);
  // A freshly issued one-time password. Held in memory only, shown once, and
  // dropped as soon as the administrator dismisses it: it is never persisted,
  // never re-fetchable, and never written anywhere it could be recovered.
  const [issued, setIssued] = useState<{ userId: string; password: string } | null>(null);

  const [numberDrafts, setNumberDrafts] = useState<Record<string, string>>({});
  const [nameDrafts, setNameDrafts] = useState<Record<string, string>>({});

  const refresh = useCallback(async () => {
    const [usersResult, auditResult] = await Promise.all([
      apiRequest<{ users: AdminUser[] }>('/api/admin/users'),
      apiRequest<{ events: AuditEvent[] }>('/api/admin/audit?limit=100'),
    ]);
    if (usersResult.ok) {
      setUsers(usersResult.data.users);
      setError(null);
    } else {
      setError(explain(usersResult.error));
    }
    if (auditResult.ok) {
      setEvents(auditResult.data.events);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const run = async (key: string, action: () => Promise<{ ok: boolean; error?: string }>, done: string) => {
    setBusy(key);
    setError(null);
    setNotice(null);
    const result = await action();
    setBusy(null);
    if (result.ok) {
      setNotice(done);
      await refresh();
    } else {
      setError(explain(result.error ?? 'unknown'));
    }
  };

  const createUser = () =>
    run(
      'create',
      async () => {
        const result = await apiRequest<{ user: AdminUser; oneTimePassword?: string }>(
          '/api/admin/users',
          {
            method: 'POST',
            body: {
              email: newEmail.trim(),
              displayName: newName.trim(),
              role: newRole,
              withPassword: newWithPassword,
            },
          }
        );
        if (result.ok) {
          setNewEmail('');
          setNewName('');
          setNewRole('member');
          if (result.data.oneTimePassword) {
            setIssued({
              userId: result.data.user.userId,
              password: result.data.oneTimePassword,
            });
          }
        }
        return result;
      },
      'User created.'
    );

  const patchUser = (user: AdminUser, body: Record<string, string>, done: string) =>
    run(user.userId, () => apiRequest(`/api/admin/users/${user.userId}`, { method: 'PATCH', body }), done);

  const setNumber = (user: AdminUser) =>
    run(
      `${user.userId}:number`,
      async () => {
        const result = await apiRequest(`/api/admin/users/${user.userId}/callback-number`, {
          method: 'PUT',
          body: { number: (numberDrafts[user.userId] ?? '').trim() },
        });
        if (result.ok) {
          setNumberDrafts((drafts) => ({ ...drafts, [user.userId]: '' }));
        }
        return result;
      },
      'Callback number approved. It is stored encrypted and never displayed.'
    );

  const clearNumber = (user: AdminUser) =>
    run(
      `${user.userId}:number`,
      () => apiRequest(`/api/admin/users/${user.userId}/callback-number`, { method: 'DELETE' }),
      'Callback number cleared.'
    );

  const resetPassword = (user: AdminUser) =>
    run(
      `${user.userId}:password`,
      async () => {
        const result = await apiRequest<{ oneTimePassword: string }>(
          `/api/admin/users/${user.userId}/password`,
          { method: 'POST' }
        );
        if (result.ok) {
          setIssued({ userId: user.userId, password: result.data.oneTimePassword });
        }
        return result;
      },
      'One-time password issued. It is shown once and cannot be retrieved again.'
    );

  const activeAdmins = useMemo(
    () => users.filter((user) => user.role === 'admin' && user.status === 'active').length,
    [users]
  );

  return (
    <div className="space-y-8">
      {(error || notice) && (
        <p className={error ? 'text-destructive text-sm' : 'text-muted-foreground text-sm'} role="status">
          {error ?? notice}
        </p>
      )}

      {issued && (
        <div className="border-input bg-muted/40 rounded-lg border p-4" role="status">
          <p className="text-sm font-medium">
            One-time password for {shortId(issued.userId)}
          </p>
          <code className="mt-2 block font-mono text-base break-all select-all">
            {issued.password}
          </code>
          <p className="text-muted-foreground mt-2 text-xs">
            Shown once and never again. Give it to the user over a channel you trust; they must
            choose their own password before they can do anything else. Their existing sessions
            have already been signed out.
          </p>
          <Button variant="secondary" size="sm" className="mt-3" onClick={() => setIssued(null)}>
            Done, hide it
          </Button>
        </div>
      )}

      <section className="space-y-3">
        <h2 className="text-sm font-semibold tracking-wider uppercase">Create user</h2>
        <form
          className="flex flex-wrap items-end gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            void createUser();
          }}
        >
          <label className="flex flex-col gap-1 text-xs">
            Email (must match their Cloudflare Access login)
            <input
              type="email"
              required
              value={newEmail}
              maxLength={254}
              onChange={(event) => setNewEmail(event.target.value)}
              className={`${inputClass} w-72`}
              autoComplete="off"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            Display name
            <input
              type="text"
              required
              value={newName}
              maxLength={80}
              onChange={(event) => setNewName(event.target.value)}
              className={`${inputClass} w-48`}
              autoComplete="off"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            Role
            <select
              value={newRole}
              onChange={(event) => setNewRole(event.target.value as 'admin' | 'member')}
              className={inputClass}
            >
              <option value="member">member</option>
              <option value="admin">admin</option>
            </select>
          </label>
          <label className="flex items-center gap-2 self-end pb-2 text-xs">
            <input
              type="checkbox"
              checked={newWithPassword}
              onChange={(event) => setNewWithPassword(event.target.checked)}
            />
            Issue a one-time password
          </label>
          <Button type="submit" variant="primary" size="sm" disabled={busy !== null}>
            {busy === 'create' ? 'Creating…' : 'Create'}
          </Button>
        </form>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold tracking-wider uppercase">
          Users <span className="text-muted-foreground font-normal">({users.length})</span>
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-muted-foreground text-xs uppercase">
              <tr>
                <th className="py-2 pr-4">User</th>
                <th className="py-2 pr-4">Role</th>
                <th className="py-2 pr-4">Status</th>
                <th className="py-2 pr-4">Callback number</th>
                <th className="py-2 pr-4">Last seen</th>
              </tr>
            </thead>
            <tbody className="divide-border divide-y">
              {users.map((user) => {
                const isSelf = user.userId === selfId;
                const lastActiveAdmin =
                  user.role === 'admin' && user.status === 'active' && activeAdmins <= 1;
                const nameDraft = nameDrafts[user.userId];
                return (
                  <tr key={user.userId} className="align-top">
                    <td className="py-3 pr-4">
                      <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            aria-label={`Display name for ${user.emailHint}`}
                            value={nameDraft ?? user.displayName}
                            maxLength={80}
                            onChange={(event) =>
                              setNameDrafts((drafts) => ({ ...drafts, [user.userId]: event.target.value }))
                            }
                            className={`${inputClass} w-44`}
                          />
                          {nameDraft !== undefined && nameDraft.trim() !== user.displayName && (
                            <Button
                              variant="ghost"
                              size="sm"
                              disabled={busy !== null}
                              onClick={() =>
                                void patchUser(user, { displayName: nameDraft.trim() }, 'Name updated.').then(
                                  () => setNameDrafts((drafts) => ({ ...drafts, [user.userId]: undefined as never }))
                                )
                              }
                            >
                              Save
                            </Button>
                          )}
                        </div>
                        <span className="text-muted-foreground font-mono text-xs">
                          {user.emailHint} · {shortId(user.userId)}
                          {isSelf && ' · you'}
                        </span>
                      </div>
                    </td>
                    <td className="py-3 pr-4">
                      <select
                        aria-label={`Role for ${user.emailHint}`}
                        value={user.role}
                        disabled={busy !== null || lastActiveAdmin}
                        onChange={(event) =>
                          void patchUser(user, { role: event.target.value }, 'Role updated.')
                        }
                        className={inputClass}
                      >
                        <option value="member">member</option>
                        <option value="admin">admin</option>
                      </select>
                    </td>
                    <td className="py-3 pr-4">
                      <div className="flex items-center gap-2">
                        <span className="capitalize">{user.status}</span>
                        <Button
                          variant={user.status === 'active' ? 'destructive' : 'secondary'}
                          size="sm"
                          disabled={busy !== null || lastActiveAdmin}
                          onClick={() =>
                            void patchUser(
                              user,
                              { status: user.status === 'active' ? 'suspended' : 'active' },
                              user.status === 'active' ? 'User suspended.' : 'User reactivated.'
                            )
                          }
                        >
                          {user.status === 'active' ? 'Suspend' : 'Activate'}
                        </Button>
                        <Button
                          variant="secondary"
                          size="sm"
                          disabled={busy !== null}
                          onClick={() => void resetPassword(user)}
                          title="Issue a random one-time password; ends this user's sessions"
                        >
                          Reset password
                        </Button>
                      </div>
                    </td>
                    <td className="py-3 pr-4">
                      <div className="flex flex-col gap-2">
                        <span className="text-muted-foreground text-xs">
                          {user.hasCallbackNumber
                            ? `Approved (set ${formatTime(user.callbackNumberUpdatedAt)})`
                            : 'None approved'}
                        </span>
                        <div className="flex items-center gap-2">
                          <input
                            type="tel"
                            aria-label={`New callback number for ${user.emailHint}`}
                            placeholder="+1 780 555 1234"
                            value={numberDrafts[user.userId] ?? ''}
                            maxLength={32}
                            autoComplete="off"
                            onChange={(event) =>
                              setNumberDrafts((drafts) => ({ ...drafts, [user.userId]: event.target.value }))
                            }
                            className={`${inputClass} w-40`}
                          />
                          <Button
                            variant="primary"
                            size="sm"
                            disabled={busy !== null || !(numberDrafts[user.userId] ?? '').trim()}
                            onClick={() => void setNumber(user)}
                          >
                            {user.hasCallbackNumber ? 'Replace' : 'Approve'}
                          </Button>
                          {user.hasCallbackNumber && (
                            <Button
                              variant="ghost"
                              size="sm"
                              disabled={busy !== null}
                              onClick={() => void clearNumber(user)}
                            >
                              Clear
                            </Button>
                          )}
                        </div>
                      </div>
                    </td>
                    <td className="text-muted-foreground py-3 pr-4 text-xs">
                      {formatTime(user.lastSeenAt)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold tracking-wider uppercase">Audit trail</h2>
        <p className="text-muted-foreground text-xs">
          Most recent first. Actors and targets are opaque ids; emails, phone numbers and memory
          contents are never recorded.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-muted-foreground uppercase">
              <tr>
                <th className="py-2 pr-4">When</th>
                <th className="py-2 pr-4">Action</th>
                <th className="py-2 pr-4">Actor</th>
                <th className="py-2 pr-4">Target</th>
                <th className="py-2 pr-4">Detail</th>
              </tr>
            </thead>
            <tbody className="divide-border divide-y font-mono">
              {events.map((event) => (
                <tr key={event.eventId}>
                  <td className="py-1.5 pr-4 whitespace-nowrap">{formatTime(event.occurredAt)}</td>
                  <td className="py-1.5 pr-4">{event.action}</td>
                  <td className="py-1.5 pr-4">
                    {shortId(event.actorId)} <span className="text-muted-foreground">({event.actorRole})</span>
                  </td>
                  <td className="py-1.5 pr-4">{shortId(event.targetId)}</td>
                  <td className="py-1.5 pr-4">{JSON.stringify(event.detail)}</td>
                </tr>
              ))}
              {events.length === 0 && (
                <tr>
                  <td colSpan={5} className="text-muted-foreground py-3">
                    No events yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
