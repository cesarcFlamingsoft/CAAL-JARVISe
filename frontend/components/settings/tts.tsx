'use client';

import { useEffect, useState } from 'react';
import { apiRequest } from '@/components/account/api-client';
import {
  type Provider,
  type TtsView,
  engines,
  parseTtsView,
  parseVoiceboxConfig,
} from '@/lib/tts/contract';

const inputClass = 'border-input bg-background w-full min-w-0 rounded-lg border px-3 py-2 text-sm';
const buttonClass = 'rounded-lg border px-3 py-2 text-sm disabled:opacity-50';
const statusText = {
  not_configured: 'Voicebox is not configured.',
  unavailable: 'Voicebox is unavailable. Check the shared connection.',
  api_verified: 'Voicebox API verified. Audio synthesis has not been tested here.',
  configuration_changed:
    'Voicebox connection changed. Select your profile again; Kokoro is the fallback.',
};
function describeError(code: string) {
  if (code === 'invalid_voicebox_config')
    return 'Use an explicitly approved local URL with a port and no path or credentials.';
  if (code === 'voicebox_profile_unavailable')
    return 'Profile or cached model unavailable, incompatible, or unsupported. Check the profile ID and model in Voicebox.';
  if (code === 'voicebox_unavailable')
    return 'Voicebox is unavailable or its API could not be verified. Nothing was saved.';
  if (code === 'qwen_trial_unavailable') return 'Qwen streaming is unavailable. Nothing was saved.';
  return 'Could not save or load TTS settings. Check your sign-in and try again.';
}

export function TtsSettings() {
  const [view, setView] = useState<TtsView | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    let active = true;
    void apiRequest<unknown>('/api/tts').then((result) => {
      if (!active) return;
      const parsed = result.ok ? parseTtsView(result.data) : null;
      setView(parsed);
      setError(!parsed);
    });
    return () => {
      active = false;
    };
  }, []);
  return view ? (
    <TtsForm view={view} onSaved={setView} />
  ) : (
    <p role="status">{error ? 'TTS settings unavailable.' : 'Loading TTS settings…'}</p>
  );
}

export function TtsForm({ view, onSaved }: { view: TtsView; onSaved: (view: TtsView) => void }) {
  const [provider, setProvider] = useState<Provider>(view.provider);
  const [profile, setProfile] = useState(view.profile_id ?? '');
  const [engine, setEngine] = useState(view.engine ?? '');
  const [size, setSize] = useState(view.model_size ?? '');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  const save = async () => {
    setBusy(true);
    setNote('');
    const body =
      provider === 'voicebox'
        ? { provider, profile_id: profile, engine, model_size: size }
        : { provider };
    const result = await apiRequest<unknown>('/api/tts', { method: 'PUT', body });
    const parsed = result.ok ? parseTtsView(result.data) : null;
    if (parsed) {
      onSaved(parsed);
      setNote('Saved for your new sessions.');
    } else setNote(describeError(result.ok ? 'schema' : result.error));
    setBusy(false);
  };
  return (
    <section aria-label="Voice output" className="space-y-3 rounded-xl border p-4">
      <h3 className="font-semibold">Voice output</h3>
      <p className="text-muted-foreground text-xs">
        Your choice applies to new sessions. Current choice:{' '}
        {view.provider === 'qwen-trial' ? 'Qwen streaming' : view.provider} (
        {view.source === 'default' ? 'deployment default' : 'personal'}).
      </p>
      <label className="block space-y-1 text-sm">
        Speech backend
        <select
          className={inputClass}
          value={provider}
          disabled={busy}
          onChange={(e) => setProvider(e.target.value as Provider)}
        >
          <option value="kokoro">Kokoro — built-in</option>
          <option value="qwen-trial" disabled={!view.qwen_configured}>
            Qwen streaming — current local service
          </option>
          <option
            value="voicebox"
            disabled={
              view.voicebox_status !== 'api_verified' &&
              view.voicebox_status !== 'configuration_changed'
            }
          >
            Voicebox — external API
          </option>
          <option value="piper">Piper — legacy deployment option</option>
        </select>
      </label>
      <p className="text-muted-foreground text-xs">
        Kokoro uses the deployment endpoint. No URL needed. Qwen streaming is the custom local
        service. Voicebox sends audio after full synthesis.
      </p>
      <p role="status" className="text-muted-foreground text-xs">
        {statusText[view.voicebox_status]}
      </p>
      {provider === 'voicebox' && (
        <div className="grid min-w-0 gap-3 sm:grid-cols-2">
          <label className="space-y-1 text-sm sm:col-span-2">
            Voicebox profile ID
            <input
              className={inputClass}
              value={profile}
              maxLength={100}
              onChange={(e) => setProfile(e.target.value)}
              placeholder="Choose a profile you are authorized to use"
            />
          </label>
          <label className="space-y-1 text-sm">
            Engine
            <select
              className={inputClass}
              aria-label="Engine"
              value={engine}
              onChange={(e) => {
                setEngine(e.target.value);
                setSize('');
              }}
            >
              <option value="">Choose engine</option>
              {engines.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-sm">
            Model size
            <select
              aria-label="Model size"
              className={inputClass}
              value={size}
              onChange={(e) => setSize(e.target.value)}
            >
              <option value="">Choose model</option>
              {(engine === 'tada'
                ? ['1B', '3B']
                : engine.startsWith('qwen')
                  ? ['1.7B', '0.6B']
                  : ['1.7B']
              ).map((item) => (
                <option key={item} value={item}>
                  {engine && !engine.startsWith('qwen') && engine !== 'tada'
                    ? 'Engine default'
                    : item}
                </option>
              ))}
            </select>
          </label>
          <p className="text-muted-foreground text-xs sm:col-span-2">
            Enter an explicitly chosen English preset or cloned profile. Private profiles are not
            listed or automatically selected. Designed profiles are not supported by this
            integration.
          </p>
        </div>
      )}
      <button
        type="button"
        className={buttonClass}
        onClick={save}
        disabled={
          busy ||
          (provider === 'voicebox' &&
            (!profile ||
              !engine ||
              !size ||
              view.voicebox_status === 'unavailable' ||
              view.voicebox_status === 'not_configured'))
        }
      >
        {busy ? 'Saving…' : 'Save my voice choice'}
      </button>
      {note && (
        <p role="status" className="text-sm">
          {note}
        </p>
      )}
      {view.can_configure && (
        <VoiceboxAdmin
          onSaved={async () => {
            const result = await apiRequest<unknown>('/api/tts');
            const parsed = result.ok ? parseTtsView(result.data) : null;
            if (parsed) onSaved(parsed);
          }}
        />
      )}
    </section>
  );
}

function VoiceboxAdmin({ onSaved }: { onSaved: () => void }) {
  const [endpoint, setEndpoint] = useState('');
  const [credential, setCredential] = useState('');
  const [configured, setConfigured] = useState(false);
  const [clearCredential, setClearCredential] = useState(false);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  useEffect(() => {
    let active = true;
    void apiRequest<unknown>('/api/tts/voicebox').then((result) => {
      if (!active) return;
      const parsed = result.ok ? parseVoiceboxConfig(result.data) : null;
      if (parsed) {
        setEndpoint(parsed.endpoint);
        setConfigured(parsed.credential_configured);
      } else setNote('Shared Voicebox settings unavailable.');
    });
    return () => {
      active = false;
    };
  }, []);
  const submit = async (test: boolean) => {
    setBusy(true);
    setNote('');
    const result = await apiRequest<unknown>(
      test ? '/api/tts/voicebox/test' : '/api/tts/voicebox',
      {
        method: test ? 'POST' : 'PUT',
        body: {
          endpoint,
          ...(credential || clearCredential
            ? { credential: clearCredential ? '' : credential }
            : {}),
        },
      }
    );
    const parsed = result.ok ? parseVoiceboxConfig(result.data) : null;
    setCredential('');
    setClearCredential(false);
    if (parsed) {
      setConfigured(parsed.credential_configured);
      setNote(
        test
          ? 'API connection verified. No synthesis or selection performed.'
          : 'Shared connection saved. Your voice choice is unchanged.'
      );
      if (!test) onSaved();
    } else setNote(describeError(result.ok ? 'schema' : result.error));
    setBusy(false);
  };
  return (
    <fieldset disabled={busy} className="min-w-0 space-y-3 border-t pt-3">
      <legend className="text-sm font-medium">Administrator · shared Voicebox connection</legend>
      <label className="block space-y-1 text-sm">
        Voicebox URL
        <input
          type="url"
          className={inputClass}
          value={endpoint}
          onChange={(e) => setEndpoint(e.target.value)}
          maxLength={200}
          placeholder="http://host.docker.internal:8000"
        />
      </label>
      <p className="text-muted-foreground text-xs">
        Only explicitly approved local hosts, addresses and ports are accepted. The server connects
        to Voicebox; your browser does not.
      </p>
      <label className="block space-y-1 text-sm">
        Proxy bearer credential (optional)
        <input
          type="password"
          autoComplete="new-password"
          className={inputClass}
          value={credential}
          maxLength={4096}
          onChange={(e) => setCredential(e.target.value)}
          placeholder={
            configured
              ? 'Stored securely; blank keeps it for this URL'
              : 'Local Voicebox has no native bearer authentication'
          }
        />
      </label>
      {configured && (
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={clearCredential}
            onChange={(e) => setClearCredential(e.target.checked)}
          />
          Remove stored credential
        </label>
      )}
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          className={buttonClass}
          disabled={!endpoint || busy}
          onClick={() => submit(true)}
        >
          Test connection
        </button>
        <button
          type="button"
          className={buttonClass}
          disabled={!endpoint || busy}
          onClick={() => submit(false)}
        >
          Save connection
        </button>
      </div>
      {note && (
        <p role="status" className="text-sm">
          {note}
        </p>
      )}
    </fieldset>
  );
}
