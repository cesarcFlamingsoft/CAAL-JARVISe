'use client';

/**
 * The signed-in user's own profile. Only the display name is editable here;
 * role, status and the approved callback number are administrator-controlled,
 * and the number itself is never shown, only whether one is on file.
 */
import { useCallback, useEffect, useState } from 'react';
import { Button } from '@/components/livekit/button';
import { apiRequest, explain } from './api-client';

interface BrowserUser {
  userId: string;
  displayName: string;
  emailHint: string;
  role: string;
  status: string;
  hasCallbackNumber: boolean;
  callbackNumberUpdatedAt: number | null;
}

export function AccountPanel() {
  const [user, setUser] = useState<BrowserUser | null>(null);
  const [draft, setDraft] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    const result = await apiRequest<{ user: BrowserUser }>('/api/me');
    if (result.ok) {
      setUser(result.data.user);
      setDraft(result.data.user.displayName);
      setError(null);
    } else {
      setError(explain(result.error));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const save = async () => {
    setSaving(true);
    setSaved(false);
    const result = await apiRequest<{ user: BrowserUser }>('/api/me', {
      method: 'PATCH',
      body: { displayName: draft },
    });
    setSaving(false);
    if (result.ok) {
      setUser(result.data.user);
      setDraft(result.data.user.displayName);
      setError(null);
      setSaved(true);
    } else {
      setError(explain(result.error));
    }
  };

  if (!user) {
    return (
      <p className="text-muted-foreground text-sm">{error ?? 'Loading your profile…'}</p>
    );
  }

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <h2 className="text-sm font-semibold tracking-wider uppercase">Profile</h2>
        <dl className="grid grid-cols-[8rem_1fr] gap-y-2 text-sm">
          <dt className="text-muted-foreground">Signed in as</dt>
          <dd className="font-mono">{user.emailHint}</dd>
          <dt className="text-muted-foreground">Role</dt>
          <dd className="capitalize">{user.role}</dd>
          <dt className="text-muted-foreground">Status</dt>
          <dd className="capitalize">{user.status}</dd>
          <dt className="text-muted-foreground">Callback number</dt>
          <dd>
            {user.hasCallbackNumber
              ? 'An approved number is on file. JARVIS will only ever call that number.'
              : 'None on file. Ask an administrator to approve one before asking JARVIS to call you.'}
          </dd>
        </dl>
      </section>

      <section className="space-y-2">
        <label className="text-sm font-medium" htmlFor="display-name">
          Display name
        </label>
        <div className="flex gap-2">
          <input
            id="display-name"
            type="text"
            value={draft}
            maxLength={80}
            onChange={(event) => {
              setDraft(event.target.value);
              setSaved(false);
            }}
            className="border-input bg-background w-full rounded-lg border px-3 py-2 text-sm"
          />
          <Button
            variant="primary"
            size="sm"
            onClick={save}
            disabled={saving || draft.trim() === user.displayName || !draft.trim()}
          >
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </div>
        {saved && <p className="text-muted-foreground text-xs">Saved.</p>}
        {error && <p className="text-destructive text-xs">{error}</p>}
      </section>
    </div>
  );
}
