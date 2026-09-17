'use client';

import { useEffect, useState } from 'react';
import { apiRequest } from '@/components/account/api-client';
import {
  type Language,
  type LanguageView,
  languageLabels,
  languages,
  parseLanguageView,
} from '@/lib/language/contract';

const inputClass = 'border-input bg-background w-full min-w-0 rounded-lg border px-3 py-2 text-sm';
const buttonClass = 'rounded-lg border px-3 py-2 text-sm disabled:opacity-50';

export function LanguageSettings() {
  const [view, setView] = useState<LanguageView | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    let active = true;
    void apiRequest<unknown>('/api/language').then((result) => {
      if (!active) return;
      const parsed = result.ok ? parseLanguageView(result.data) : null;
      setView(parsed);
      setError(!parsed);
    });
    return () => {
      active = false;
    };
  }, []);
  return view ? (
    <LanguageForm view={view} onSaved={setView} />
  ) : (
    <p role="status">{error ? 'Reply language unavailable.' : 'Loading reply language…'}</p>
  );
}

export function LanguageForm({
  view,
  onSaved,
}: {
  view: LanguageView;
  onSaved: (view: LanguageView) => void;
}) {
  const [language, setLanguage] = useState<Language>(view.language);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  const save = async () => {
    setBusy(true);
    setNote('');
    const result = await apiRequest<unknown>('/api/language', { method: 'PUT', body: { language } });
    const parsed = result.ok ? parseLanguageView(result.data) : null;
    if (parsed) {
      onSaved(parsed);
      setNote('Saved for your new sessions.');
    } else setNote('Could not save your reply language. Check your sign-in and try again.');
    setBusy(false);
  };
  return (
    <section aria-label="Reply language" className="space-y-3 rounded-xl border p-4">
      <h3 className="font-semibold">Reply language</h3>
      <p className="text-muted-foreground text-xs">
        Your choice applies to your new sessions only, and to nobody else. Current choice:{' '}
        {languageLabels[view.language]} (
        {view.source === 'default' ? 'deployment default' : 'personal'}).
      </p>
      <label className="block space-y-1 text-sm">
        Answer me in
        <select
          className={inputClass}
          value={language}
          disabled={busy}
          onChange={(e) => setLanguage(e.target.value as Language)}
        >
          {languages.map((item) => (
            <option key={item} value={item}>
              {languageLabels[item]}
            </option>
          ))}
        </select>
      </label>
      <p className="text-muted-foreground text-xs">
        Automatic follows the language you speak and keeps it through short answers like “yes” or
        “sí”. You can still switch inside a conversation by asking — “español por favor”, “English
        please” — which lasts for that session without changing this setting.
      </p>
      <button type="button" className={buttonClass} onClick={save} disabled={busy}>
        {busy ? 'Saving…' : 'Save my language choice'}
      </button>
      {note && (
        <p role="status" className="text-sm">
          {note}
        </p>
      )}
    </section>
  );
}
