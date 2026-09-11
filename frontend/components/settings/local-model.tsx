'use client';

/**
 * Choosing the local Ollama JARVIS runs on, from the settings panel.
 *
 * The operator types an address on their own network, refreshes the list of
 * models installed there, picks one and saves. The browser never talks to the
 * Ollama: every request goes to our own authenticated route, which asks the
 * agent. What is typed is checked here as it is typed, and again by the route,
 * and again by the agent -- only plain http on a local address is ever
 * reached. An error is a sentence about the network, never an upstream string.
 */
import { useCallback, useEffect, useState } from 'react';
import { ArrowsClockwise, CircleNotch, FloppyDisk } from '@phosphor-icons/react/dist/ssr';
import { apiRequest } from '@/components/account/api-client';
import {
  type LocalModelView,
  type ModelsView,
  describeEndpointCode,
  normalizeEndpoint,
} from '@/lib/local-model/endpoint';

type Note = { kind: 'error' | 'ok'; text: string } | null;

const PLACEHOLDER = 'http://192.168.1.50:11434';

export function LocalModelSettings() {
  const [current, setCurrent] = useState<LocalModelView | null>(null);
  const [endpoint, setEndpoint] = useState('');
  const [model, setModel] = useState('');
  const [models, setModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [discovering, setDiscovering] = useState(false);
  const [saving, setSaving] = useState(false);
  const [note, setNote] = useState<Note>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const result = await apiRequest<LocalModelView>('/api/local-model');
      if (cancelled) return;
      if (result.ok) {
        setCurrent(result.data);
        setEndpoint(result.data.endpoint);
        setModel(result.data.model);
      } else {
        setNote({ kind: 'error', text: describeEndpointCode(result.error) });
      }
      setLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const checked = normalizeEndpoint(endpoint);
  const typedProblem = endpoint.trim() && !checked.ok ? describeEndpointCode(checked.code) : null;

  const refresh = useCallback(async () => {
    const parsed = normalizeEndpoint(endpoint);
    if (!parsed.ok) {
      setNote({ kind: 'error', text: describeEndpointCode(parsed.code) });
      return;
    }
    setDiscovering(true);
    setNote(null);
    const result = await apiRequest<ModelsView>('/api/local-model/models', {
      method: 'POST',
      body: { endpoint: parsed.endpoint },
    });
    setDiscovering(false);
    if (!result.ok) {
      setModels([]);
      setNote({ kind: 'error', text: describeEndpointCode(result.error) });
      return;
    }
    setModels(result.data.models);
    if (result.data.models.length === 0) {
      setNote({ kind: 'error', text: 'That Ollama answered, but it has no models installed.' });
      return;
    }
    if (!result.data.models.includes(model)) setModel(result.data.models[0]);
    setNote({ kind: 'ok', text: result.data.models.length + ' models found on that machine.' });
  }, [endpoint, model]);

  const save = useCallback(async () => {
    const parsed = normalizeEndpoint(endpoint);
    if (!parsed.ok) {
      setNote({ kind: 'error', text: describeEndpointCode(parsed.code) });
      return;
    }
    setSaving(true);
    const result = await apiRequest<LocalModelView>('/api/local-model', {
      method: 'PUT',
      body: { endpoint: parsed.endpoint, model },
    });
    setSaving(false);
    if (!result.ok) {
      setNote({ kind: 'error', text: describeEndpointCode(result.error) });
      return;
    }
    setCurrent(result.data);
    setEndpoint(result.data.endpoint);
    setModel(result.data.model);
    setNote({
      kind: 'ok',
      text: 'Saved. The next JARVIS session uses it; a conversation already in progress keeps the model it started with.',
    });
  }, [endpoint, model]);

  const unchanged =
    current !== null &&
    checked.ok &&
    checked.endpoint === current.endpoint &&
    model === current.model;

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <label className="text-muted-foreground text-xs font-bold tracking-wide uppercase">
          Local model
        </label>
        <p className="text-muted-foreground text-xs">
          JARVIS answers ordinary turns on this Ollama. Work that needs an agent harness is
          escalated to Hermes, and anything about code goes to Hermes coding delegation. Choosing a
          model here does not change that routing.
        </p>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium" htmlFor="ollama-endpoint">
          Ollama address
        </label>
        <div className="flex gap-2">
          <input
            id="ollama-endpoint"
            type="text"
            inputMode="url"
            autoComplete="off"
            spellCheck={false}
            value={endpoint}
            disabled={loading}
            onChange={(event) => setEndpoint(event.target.value)}
            placeholder={PLACEHOLDER}
            className="border-input bg-background flex-1 rounded-lg border px-4 py-3 text-sm"
          />
          <button
            type="button"
            onClick={refresh}
            disabled={loading || discovering || !checked.ok}
            className="bg-muted hover:bg-muted/80 flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50"
          >
            {discovering ? (
              <CircleNotch size={16} className="animate-spin" />
            ) : (
              <ArrowsClockwise size={16} />
            )}
            Refresh models
          </button>
        </div>
        <p className="text-muted-foreground text-xs">
          Local network only: localhost, host.docker.internal, or a private address such as{' '}
          <code>http://192.168.1.50:11434</code> or <code>http://10.0.0.12:11434</code>. Include the
          port. Addresses on the public internet and https are refused.
        </p>
        {typedProblem && <p className="text-xs text-amber-500">{typedProblem}</p>}
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium" htmlFor="ollama-model">
          Model
        </label>
        <select
          id="ollama-model"
          value={model}
          disabled={loading || (models.length === 0 && !model)}
          onChange={(event) => setModel(event.target.value)}
          className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
        >
          {models.length === 0 && model && <option value={model}>{model}</option>}
          {models.length === 0 && !model && (
            <option value="">Refresh to see installed models</option>
          )}
          {models.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        {models.length === 0 && (
          <p className="text-muted-foreground text-xs">
            Refresh to read the models installed on that machine.
          </p>
        )}
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={save}
          disabled={loading || saving || !checked.ok || !model || unchanged}
          className="bg-primary text-primary-foreground flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50"
        >
          {saving ? <CircleNotch size={16} className="animate-spin" /> : <FloppyDisk size={16} />}
          Save local model
        </button>
        {current && (
          <span className="text-muted-foreground text-xs">
            In use now: {current.model || 'no model chosen'} at {current.endpoint}
          </span>
        )}
      </div>

      {note && (
        <p className={note.kind === 'error' ? 'text-xs text-red-500' : 'text-xs text-green-500'}>
          {note.text}
        </p>
      )}
    </div>
  );
}
