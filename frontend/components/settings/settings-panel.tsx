'use client';

import { useCallback, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Check, CircleNotch, FloppyDisk, X } from '@phosphor-icons/react/dist/ssr';
import { Button } from '@/components/livekit/button';

// =============================================================================
// Types
// =============================================================================

interface Settings {
  agent_name: string;
  prompt: string;
  wake_greetings: string[];
  // Providers
  llm_provider: 'ollama' | 'groq';
  ollama_host: string;
  ollama_model: string;
  groq_api_key: string;
  groq_model: string;
  tts_provider: 'kokoro' | 'piper';
  tts_voice_kokoro: string;
  tts_voice_piper: string;
  // LLM settings
  temperature: number;
  num_ctx: number;
  max_turns: number;
  tool_cache_size: number;
  // Integrations
  hass_enabled: boolean;
  hass_host: string;
  hass_token: string;
  hass_agent_id: string;
  n8n_enabled: boolean;
  n8n_url: string;
  n8n_token: string;
  // Wake word
  wake_word_enabled: boolean;
  wake_word_model: string;
  wake_word_threshold: number;
  wake_word_timeout: number;
  // Turn detection
  allow_interruptions: boolean;
  min_endpointing_delay: number;
  // Noise suppression
  noise_suppression_enabled: boolean;
  noise_suppression_atten_db: number;
}

type TestStatus = 'idle' | 'testing' | 'success' | 'error';

type TabId = 'agent' | 'prompt' | 'providers' | 'llm' | 'integrations' | 'wake';

// =============================================================================
// Constants
// =============================================================================

const DEFAULT_SETTINGS: Settings = {
  agent_name: 'Cal',
  prompt: 'default',
  wake_greetings: ["Hey, what's up?", "What's up?", 'How can I help?'],
  llm_provider: 'ollama',
  ollama_host: 'http://localhost:11434',
  ollama_model: '',
  groq_api_key: '',
  groq_model: '',
  tts_provider: 'kokoro',
  tts_voice_kokoro: 'am_puck',
  tts_voice_piper: 'speaches-ai/piper-en_US-ryan-high',
  temperature: 0.7,
  num_ctx: 8192,
  max_turns: 20,
  tool_cache_size: 3,
  hass_enabled: false,
  hass_host: '',
  hass_token: '',
  hass_agent_id: 'conversation.home_assistant',
  n8n_enabled: false,
  n8n_url: '',
  n8n_token: '',
  wake_word_enabled: false,
  wake_word_model: 'models/hey_cal.onnx',
  wake_word_threshold: 0.5,
  wake_word_timeout: 3.0,
  allow_interruptions: true,
  min_endpointing_delay: 0.5,
  noise_suppression_enabled: false,
  noise_suppression_atten_db: 100,
};

const DEFAULT_PROMPT = `# Voice Assistant

You are a helpful, conversational voice assistant.
{{CURRENT_DATE_CONTEXT}}

# Tool Priority

Always prefer using tools to answer questions when possible.
`;

const TABS: { id: TabId; label: string }[] = [
  { id: 'agent', label: 'Agent' },
  { id: 'prompt', label: 'Prompt' },
  { id: 'providers', label: 'Providers' },
  { id: 'llm', label: 'LLM Settings' },
  { id: 'integrations', label: 'Integrations' },
  { id: 'wake', label: 'Wake Word' },
];

// =============================================================================
// Component
// =============================================================================

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsPanel({ isOpen, onClose }: SettingsPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId>('agent');
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [promptContent, setPromptContent] = useState('');
  const [voices, setVoices] = useState<string[]>([]);
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [groqModels, setGroqModels] = useState<string[]>([]);
  const [wakeWordModels, setWakeWordModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  // Test states
  const [ollamaTest, setOllamaTest] = useState<{ status: TestStatus; error: string | null }>({
    status: 'idle',
    error: null,
  });
  const [groqTest, setGroqTest] = useState<{ status: TestStatus; error: string | null }>({
    status: 'idle',
    error: null,
  });
  const [hassTest, setHassTest] = useState<{
    status: TestStatus;
    error: string | null;
    info: string | null;
  }>({
    status: 'idle',
    error: null,
    info: null,
  });
  const [n8nTest, setN8nTest] = useState<{
    status: TestStatus;
    error: string | null;
    info: string | null;
  }>({
    status: 'idle',
    error: null,
    info: null,
  });
  const [hassAgents, setHassAgents] = useState<{ id: string; name: string }[]>([]);

  // ---------------------------------------------------------------------------
  // Load settings
  // ---------------------------------------------------------------------------

  const loadSettings = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // First load settings to get the correct tts_provider
      const settingsRes = await fetch('/api/settings');
      let ttsProvider = 'kokoro';

      if (settingsRes.ok) {
        const data = await settingsRes.json();
        const loadedSettings = data.settings || DEFAULT_SETTINGS;
        setSettings(loadedSettings);
        setPromptContent(data.prompt_content || DEFAULT_PROMPT);
        ttsProvider = loadedSettings.tts_provider || 'kokoro';
      } else {
        setSettings(DEFAULT_SETTINGS);
        setPromptContent(DEFAULT_PROMPT);
      }

      // Now fetch voices with correct provider, plus wake word models
      const [voicesRes, wakeWordModelsRes] = await Promise.all([
        fetch(`/api/voices?provider=${ttsProvider}`),
        fetch('/api/wake-word/models'),
      ]);

      if (voicesRes.ok) {
        const data = await voicesRes.json();
        setVoices(data.voices || []);
      }

      if (wakeWordModelsRes.ok) {
        const data = await wakeWordModelsRes.json();
        setWakeWordModels(data.models || []);
      }
    } catch (err) {
      console.error('Error loading settings:', err);
      setError('Failed to load settings');
      setSettings(DEFAULT_SETTINGS);
      setPromptContent(DEFAULT_PROMPT);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      loadSettings();
    }
  }, [isOpen, loadSettings]);

  // ---------------------------------------------------------------------------
  // Test connections
  // ---------------------------------------------------------------------------

  const testOllama = useCallback(async () => {
    if (!settings.ollama_host) return;
    setOllamaTest({ status: 'testing', error: null });

    try {
      const res = await fetch('/api/setup/test-ollama', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ host: settings.ollama_host }),
      });
      const result = await res.json();

      if (result.success) {
        setOllamaTest({ status: 'success', error: null });
        setOllamaModels(result.models || []);
        if (!settings.ollama_model && result.models?.length > 0) {
          setSettings((s) => ({ ...s, ollama_model: result.models[0] }));
        }
      } else {
        setOllamaTest({ status: 'error', error: result.error || 'Connection failed' });
      }
    } catch {
      setOllamaTest({ status: 'error', error: 'Failed to connect' });
    }
  }, [settings.ollama_host, settings.ollama_model]);

  const testGroq = useCallback(async () => {
    if (!settings.groq_api_key) return;
    setGroqTest({ status: 'testing', error: null });

    try {
      const res = await fetch('/api/setup/test-groq', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: settings.groq_api_key }),
      });
      const result = await res.json();

      if (result.success) {
        setGroqTest({ status: 'success', error: null });
        setGroqModels(result.models || []);
        if (!settings.groq_model && result.models?.length > 0) {
          const preferredModel = 'llama-3.3-70b-versatile';
          const selectedModel = result.models.includes(preferredModel)
            ? preferredModel
            : result.models[0];
          setSettings((s) => ({ ...s, groq_model: selectedModel }));
        }
      } else {
        setGroqTest({ status: 'error', error: result.error || 'Invalid API key' });
      }
    } catch {
      setGroqTest({ status: 'error', error: 'Failed to validate' });
    }
  }, [settings.groq_api_key, settings.groq_model]);

  // Auto-fetch Groq models when API key is available and models not yet loaded
  useEffect(() => {
    if (isOpen && settings.groq_api_key && groqModels.length === 0 && !loading) {
      testGroq();
    }
  }, [isOpen, settings.groq_api_key, groqModels.length, loading, testGroq]);

  const fetchHassAgents = useCallback(async () => {
    if (!settings.hass_host || !settings.hass_token) return;
    try {
      const res = await fetch('/api/setup/hass-agents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ host: settings.hass_host, token: settings.hass_token }),
      });
      const result = await res.json();
      if (result.success && result.agents) {
        setHassAgents(result.agents);
        // Set default agent if not already set
        if (!settings.hass_agent_id && result.agents.length > 0) {
          setSettings((prev) => ({ ...prev, hass_agent_id: result.agents[0].id }));
        }
      }
    } catch {
      // Silently fail
    }
  }, [settings.hass_host, settings.hass_token, settings.hass_agent_id]);

  const testHass = useCallback(async () => {
    if (!settings.hass_host || !settings.hass_token) return;
    setHassTest({ status: 'testing', error: null, info: null });

    try {
      const res = await fetch('/api/setup/test-hass', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ host: settings.hass_host, token: settings.hass_token }),
      });
      const result = await res.json();

      if (result.success) {
        setHassTest({
          status: 'success',
          error: null,
          info: `Connected - ${result.device_count} entities`,
        });
        // Fetch available agents on successful connection
        fetchHassAgents();
      } else {
        setHassTest({ status: 'error', error: result.error || 'Connection failed', info: null });
        setHassAgents([]);
      }
    } catch {
      setHassTest({ status: 'error', error: 'Failed to connect', info: null });
      setHassAgents([]);
    }
  }, [settings.hass_host, settings.hass_token, fetchHassAgents]);

  const testN8n = useCallback(async () => {
    if (!settings.n8n_url || !settings.n8n_token) return;
    setN8nTest({ status: 'testing', error: null, info: null });

    const mcpUrl = getN8nMcpUrl(settings.n8n_url);

    try {
      const res = await fetch('/api/setup/test-n8n', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: mcpUrl, token: settings.n8n_token }),
      });
      const result = await res.json();

      if (result.success) {
        setN8nTest({ status: 'success', error: null, info: 'Connected' });
      } else {
        setN8nTest({ status: 'error', error: result.error || 'Connection failed', info: null });
      }
    } catch {
      setN8nTest({ status: 'error', error: 'Failed to connect', info: null });
    }
  }, [settings.n8n_url, settings.n8n_token]);

  // ---------------------------------------------------------------------------
  // Save
  // ---------------------------------------------------------------------------

  const handleSave = async () => {
    setSaving(true);
    setSaveStatus('Saving...');
    setError(null);

    try {
      // Transform n8n URL and filter empty wake greetings
      const finalSettings = {
        ...settings,
        n8n_url: settings.n8n_enabled ? getN8nMcpUrl(settings.n8n_url) : settings.n8n_url,
        wake_greetings: settings.wake_greetings.filter((g) => g.trim()),
      };

      // Save settings
      const settingsRes = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings: finalSettings }),
      });

      if (!settingsRes.ok) {
        throw new Error('Failed to save settings');
      }

      // Save prompt if custom
      if (settings.prompt === 'custom') {
        const promptRes = await fetch('/api/prompt', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: promptContent }),
        });

        if (!promptRes.ok) {
          throw new Error('Failed to save prompt');
        }
      }

      // Download Piper model if using Piper
      if (settings.tts_provider === 'piper' && settings.tts_voice_piper) {
        setSaveStatus('Downloading voice model...');
        await fetch('/api/download-piper-model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_id: settings.tts_voice_piper }),
        });
      }

      onClose();
    } catch (err) {
      console.error('Error saving settings:', err);
      setError(err instanceof Error ? err.message : 'Failed to save');
    } finally {
      setSaving(false);
      setSaveStatus('');
    }
  };

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  const getN8nMcpUrl = (host: string) => {
    if (!host) return '';
    const baseUrl = host.replace(/\/$/, '');
    if (baseUrl.includes('/mcp-server')) return baseUrl;
    return `${baseUrl}/mcp-server/http`;
  };

  const handleWakeGreetingsChange = (value: string) => {
    // Keep empty lines while editing so Enter key works
    // Empty lines are filtered out when saving
    const greetings = value.split('\n');
    setSettings({ ...settings, wake_greetings: greetings });
  };

  const handleTtsProviderChange = async (provider: 'kokoro' | 'piper') => {
    if (provider === settings.tts_provider) return;

    setSettings({ ...settings, tts_provider: provider });

    // Fetch voices for the new provider
    try {
      const res = await fetch(`/api/voices?provider=${provider}`);
      if (res.ok) {
        const data = await res.json();
        setVoices(data.voices || []);
      }
    } catch (err) {
      console.error('Failed to fetch voices for provider:', err);
    }
  };

  const handlePiperVoiceChange = (voice: string) => {
    setSettings({ ...settings, tts_voice_piper: voice });
  };

  // ---------------------------------------------------------------------------
  // Render helpers
  // ---------------------------------------------------------------------------

  const TestStatusIcon = ({ status }: { status: TestStatus }) => {
    switch (status) {
      case 'testing':
        return <CircleNotch className="h-4 w-4 animate-spin text-blue-500" />;
      case 'success':
        return <Check className="h-4 w-4 text-green-500" weight="bold" />;
      case 'error':
        return <X className="h-4 w-4 text-red-500" weight="bold" />;
      default:
        return null;
    }
  };

  const Toggle = ({
    enabled,
    onToggle,
    disabled = false,
  }: {
    enabled: boolean;
    onToggle: () => void;
    disabled?: boolean;
  }) => (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      onClick={onToggle}
      disabled={disabled}
      className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:ring-2 focus:ring-offset-2 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50 ${
        enabled ? 'bg-primary' : 'bg-muted'
      }`}
    >
      <span
        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
          enabled ? 'translate-x-5' : 'translate-x-0'
        }`}
      />
    </button>
  );

  // ---------------------------------------------------------------------------
  // Tab content
  // ---------------------------------------------------------------------------

  const renderAgentTab = () => (
    <div className="space-y-6">
      <div className="space-y-2">
        <label className="text-sm font-medium">Agent Name</label>
        <input
          type="text"
          value={settings.agent_name}
          onChange={(e) => setSettings({ ...settings, agent_name: e.target.value })}
          className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Voice</label>
        <select
          value={
            settings.tts_provider === 'piper' ? settings.tts_voice_piper : settings.tts_voice_kokoro
          }
          onChange={(e) => {
            if (settings.tts_provider === 'piper') {
              handlePiperVoiceChange(e.target.value);
            } else {
              setSettings({ ...settings, tts_voice_kokoro: e.target.value });
            }
          }}
          className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
        >
          {voices.length > 0 ? (
            voices.map((voice) => (
              <option key={voice} value={voice}>
                {voice}
              </option>
            ))
          ) : (
            <option
              value={
                settings.tts_provider === 'piper'
                  ? settings.tts_voice_piper
                  : settings.tts_voice_kokoro
              }
            >
              {settings.tts_provider === 'piper'
                ? settings.tts_voice_piper
                : settings.tts_voice_kokoro}
            </option>
          )}
        </select>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">
          Wake Greetings{' '}
          <span className="text-muted-foreground text-xs font-normal">(one per line)</span>
        </label>
        <textarea
          value={settings.wake_greetings.join('\n')}
          onChange={(e) => handleWakeGreetingsChange(e.target.value)}
          rows={5}
          className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
        />
      </div>
    </div>
  );

  const renderPromptTab = () => (
    <div className="space-y-4">
      <div className="bg-muted inline-flex rounded-lg p-1">
        <button
          onClick={() => setSettings({ ...settings, prompt: 'default' })}
          className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            settings.prompt === 'default'
              ? 'bg-background text-foreground shadow'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          Default
        </button>
        <button
          onClick={() => setSettings({ ...settings, prompt: 'custom' })}
          className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            settings.prompt === 'custom'
              ? 'bg-background text-foreground shadow'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          Custom
        </button>
      </div>

      <textarea
        value={promptContent}
        onChange={(e) => setPromptContent(e.target.value)}
        readOnly={settings.prompt === 'default'}
        rows={12}
        className={`border-input bg-background w-full rounded-lg border px-4 py-3 font-mono text-sm ${
          settings.prompt === 'default' ? 'cursor-not-allowed opacity-60' : ''
        }`}
      />
    </div>
  );

  const renderProvidersTab = () => (
    <div className="space-y-8">
      {/* LLM Provider */}
      <div className="space-y-4">
        <label className="text-muted-foreground text-xs font-bold tracking-wide uppercase">
          LLM Provider
        </label>
        <div className="bg-muted inline-flex rounded-lg p-1">
          <button
            onClick={() => setSettings({ ...settings, llm_provider: 'ollama' })}
            className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              settings.llm_provider === 'ollama'
                ? 'bg-background text-foreground shadow'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Ollama
          </button>
          <button
            onClick={() => setSettings({ ...settings, llm_provider: 'groq' })}
            className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              settings.llm_provider === 'groq'
                ? 'bg-background text-foreground shadow'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Groq
          </button>
        </div>

        {settings.llm_provider === 'ollama' ? (
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Host URL</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={settings.ollama_host}
                  onChange={(e) => setSettings({ ...settings, ollama_host: e.target.value })}
                  placeholder="http://localhost:11434"
                  className="border-input bg-background flex-1 rounded-lg border px-4 py-3 text-sm"
                />
                <button
                  onClick={testOllama}
                  disabled={!settings.ollama_host || ollamaTest.status === 'testing'}
                  className="bg-muted hover:bg-muted/80 flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50"
                >
                  <TestStatusIcon status={ollamaTest.status} />
                  Test
                </button>
              </div>
              {ollamaTest.error && <p className="text-xs text-red-500">{ollamaTest.error}</p>}
              {ollamaTest.status === 'success' && (
                <p className="text-xs text-green-500">{ollamaModels.length} models available</p>
              )}
            </div>

            {/* Show model dropdown if we have models from test OR if model is already configured */}
            {(ollamaModels.length > 0 || settings.ollama_model) && (
              <div className="space-y-2">
                <label className="text-sm font-medium">Model</label>
                <select
                  value={settings.ollama_model}
                  onChange={(e) => setSettings({ ...settings, ollama_model: e.target.value })}
                  className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
                >
                  {ollamaModels.length > 0 ? (
                    <>
                      <option value="">Select a model...</option>
                      {ollamaModels.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </>
                  ) : (
                    <option value={settings.ollama_model}>{settings.ollama_model}</option>
                  )}
                </select>
                {ollamaModels.length === 0 && settings.ollama_model && (
                  <p className="text-muted-foreground text-xs">
                    Test connection to see all available models
                  </p>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key</label>
              <div className="flex gap-2">
                <input
                  type="password"
                  value={settings.groq_api_key}
                  onChange={(e) => setSettings({ ...settings, groq_api_key: e.target.value })}
                  placeholder="gsk_..."
                  className="border-input bg-background flex-1 rounded-lg border px-4 py-3 text-sm"
                />
                <button
                  onClick={testGroq}
                  disabled={!settings.groq_api_key || groqTest.status === 'testing'}
                  className="bg-muted hover:bg-muted/80 flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50"
                >
                  <TestStatusIcon status={groqTest.status} />
                  Test
                </button>
              </div>
              {groqTest.error && <p className="text-xs text-red-500">{groqTest.error}</p>}
              {groqTest.status === 'success' && (
                <p className="text-xs text-green-500">{groqModels.length} models available</p>
              )}
              <p className="text-muted-foreground text-xs">
                Get your API key at{' '}
                <a
                  href="https://console.groq.com/keys"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline"
                >
                  console.groq.com
                </a>
              </p>
            </div>

            {/* Show model dropdown if we have models from test OR if model is already configured */}
            {(groqModels.length > 0 || settings.groq_model) && (
              <div className="space-y-2">
                <label className="text-sm font-medium">Model</label>
                <select
                  value={settings.groq_model}
                  onChange={(e) => setSettings({ ...settings, groq_model: e.target.value })}
                  className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
                >
                  {groqModels.length > 0 ? (
                    <>
                      <option value="">Select a model...</option>
                      {[...groqModels].sort().map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </>
                  ) : (
                    <option value={settings.groq_model}>{settings.groq_model}</option>
                  )}
                </select>
                {groqModels.length === 0 && settings.groq_model && (
                  <p className="text-muted-foreground text-xs">
                    Enter API key and test to see all available models
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {/* STT Info */}
        <div className="rounded-lg border border-blue-500/30 bg-blue-500/10 p-3">
          <p className="text-sm text-blue-200">
            <span className="font-semibold">STT Provider:</span>{' '}
            {settings.llm_provider === 'ollama' ? 'Speaches (local)' : 'Groq Whisper'}
            <br />
            <span className="text-xs opacity-70">
              Automatically selected based on LLM provider.
            </span>
          </p>
        </div>
      </div>

      {/* TTS Provider */}
      <div className="space-y-4">
        <label className="text-muted-foreground text-xs font-bold tracking-wide uppercase">
          TTS Provider
        </label>
        <div className="bg-muted inline-flex rounded-lg p-1">
          <button
            onClick={() => handleTtsProviderChange('kokoro')}
            className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              settings.tts_provider === 'kokoro'
                ? 'bg-background text-foreground shadow'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Kokoro
          </button>
          <button
            onClick={() => handleTtsProviderChange('piper')}
            className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              settings.tts_provider === 'piper'
                ? 'bg-background text-foreground shadow'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Piper
          </button>
        </div>
        <p className="text-muted-foreground text-xs">
          {settings.tts_provider === 'kokoro'
            ? 'High-quality neural TTS (requires Kokoro container)'
            : 'Lightweight CPU-friendly TTS with 35+ languages'}
        </p>
      </div>
    </div>
  );

  const renderLLMTab = () => (
    <div className="space-y-6">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">Temperature</label>
          <span className="text-muted-foreground text-sm">{settings.temperature}</span>
        </div>
        <input
          type="range"
          min="0"
          max="2"
          step="0.1"
          value={settings.temperature}
          onChange={(e) => setSettings({ ...settings, temperature: parseFloat(e.target.value) })}
          className="bg-muted accent-primary h-2 w-full cursor-pointer appearance-none rounded-lg"
        />
        <div className="text-muted-foreground flex justify-between text-xs">
          <span>Precise</span>
          <span>Creative</span>
        </div>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Context Size</label>
        <input
          type="number"
          min="1024"
          max="131072"
          step="1024"
          value={settings.num_ctx}
          onChange={(e) => setSettings({ ...settings, num_ctx: parseInt(e.target.value) || 8192 })}
          className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Max Turns</label>
        <input
          type="number"
          min="1"
          max="100"
          value={settings.max_turns}
          onChange={(e) => setSettings({ ...settings, max_turns: parseInt(e.target.value) || 20 })}
          className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Tool Cache Size</label>
        <input
          type="number"
          min="0"
          max="10"
          value={settings.tool_cache_size}
          onChange={(e) =>
            setSettings({ ...settings, tool_cache_size: parseInt(e.target.value) || 3 })
          }
          className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
        />
      </div>

      {/* Turn Detection Section */}
      <div className="border-t pt-6">
        <h3 className="mb-4 text-sm font-semibold">Turn Detection</h3>
        <p className="text-muted-foreground mb-4 text-xs">
          Control how the agent detects when you&apos;re done speaking
        </p>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium">Allow Interruptions</label>
              <p className="text-muted-foreground text-xs">Interrupt the agent while speaking</p>
            </div>
            <Toggle
              enabled={settings.allow_interruptions}
              onToggle={() =>
                setSettings({ ...settings, allow_interruptions: !settings.allow_interruptions })
              }
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Endpointing Delay</label>
              <span className="text-muted-foreground text-sm">
                {settings.min_endpointing_delay}s
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.1"
              value={settings.min_endpointing_delay}
              onChange={(e) =>
                setSettings({ ...settings, min_endpointing_delay: parseFloat(e.target.value) })
              }
              className="bg-muted accent-primary h-2 w-full cursor-pointer appearance-none rounded-lg"
            />
            <p className="text-muted-foreground text-xs">
              How long to wait after you stop speaking before responding
            </p>
          </div>
        </div>
      </div>

      {/* Noise Suppression Section */}
      <div className="border-t pt-6">
        <h3 className="mb-4 text-sm font-semibold">Noise Suppression</h3>
        <p className="text-muted-foreground mb-4 text-xs">
          Filter background noise from your microphone using DeepFilterNet AI
        </p>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium">Enable Noise Suppression</label>
              <p className="text-muted-foreground text-xs">
                Requires deepfilternet package installed
              </p>
            </div>
            <Toggle
              enabled={settings.noise_suppression_enabled}
              onToggle={() =>
                setSettings({
                  ...settings,
                  noise_suppression_enabled: !settings.noise_suppression_enabled,
                })
              }
            />
          </div>

          {settings.noise_suppression_enabled && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Suppression Strength</label>
                <span className="text-muted-foreground text-sm">
                  {settings.noise_suppression_atten_db}dB
                </span>
              </div>
              <input
                type="range"
                min="20"
                max="100"
                step="10"
                value={settings.noise_suppression_atten_db}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    noise_suppression_atten_db: parseFloat(e.target.value),
                  })
                }
                className="bg-muted accent-primary h-2 w-full cursor-pointer appearance-none rounded-lg"
              />
              <div className="text-muted-foreground flex justify-between text-xs">
                <span>Conservative</span>
                <span>Aggressive</span>
              </div>
              <p className="text-muted-foreground text-xs">
                Higher values remove more noise but may affect voice quality
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderIntegrationsTab = () => (
    <div className="space-y-4">
      {/* Home Assistant */}
      <div className="overflow-hidden rounded-xl border">
        <div className="bg-muted/50 flex items-center justify-between border-b px-4 py-3">
          <span className="font-semibold">Home Assistant</span>
          <Toggle
            enabled={settings.hass_enabled}
            onToggle={() => setSettings({ ...settings, hass_enabled: !settings.hass_enabled })}
          />
        </div>

        {settings.hass_enabled && (
          <div className="space-y-4 p-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Host URL</label>
              <input
                type="text"
                value={settings.hass_host}
                onChange={(e) => setSettings({ ...settings, hass_host: e.target.value })}
                placeholder="http://homeassistant.local:8123"
                className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Long-lived Access Token</label>
              <div className="flex gap-2">
                <input
                  type="password"
                  value={settings.hass_token}
                  onChange={(e) => setSettings({ ...settings, hass_token: e.target.value })}
                  placeholder="eyJ0eX..."
                  className="border-input bg-background flex-1 rounded-lg border px-4 py-3 text-sm"
                />
                <button
                  onClick={testHass}
                  disabled={
                    !settings.hass_host || !settings.hass_token || hassTest.status === 'testing'
                  }
                  className="bg-muted hover:bg-muted/80 flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50"
                >
                  <TestStatusIcon status={hassTest.status} />
                  Test
                </button>
              </div>
              {hassTest.error && <p className="text-xs text-red-500">{hassTest.error}</p>}
              {hassTest.info && <p className="text-xs text-green-500">{hassTest.info}</p>}
            </div>
            {hassAgents.length > 0 && (
              <div className="space-y-2">
                <label className="text-sm font-medium">AI Assistant</label>
                <select
                  value={settings.hass_agent_id || ''}
                  onChange={(e) => setSettings({ ...settings, hass_agent_id: e.target.value })}
                  className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
                >
                  {hassAgents.map((agent) => (
                    <option key={agent.id} value={agent.id}>
                      {agent.name}
                    </option>
                  ))}
                </select>
                <p className="text-muted-foreground text-xs">
                  Choose which AI assistant handles smart home requests
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* n8n */}
      <div
        className={`overflow-hidden rounded-xl border ${!settings.n8n_enabled ? 'opacity-60' : ''}`}
      >
        <div className="bg-muted/50 flex items-center justify-between border-b px-4 py-3">
          <span className="font-semibold">n8n</span>
          <Toggle
            enabled={settings.n8n_enabled}
            onToggle={() => setSettings({ ...settings, n8n_enabled: !settings.n8n_enabled })}
          />
        </div>

        {settings.n8n_enabled && (
          <div className="space-y-4 p-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Host URL</label>
              <input
                type="text"
                value={settings.n8n_url}
                onChange={(e) => setSettings({ ...settings, n8n_url: e.target.value })}
                placeholder="http://n8n:5678"
                className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
              />
              <p className="text-muted-foreground text-xs">
                /mcp-server/http will be appended automatically
              </p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Access Token</label>
              <div className="flex gap-2">
                <input
                  type="password"
                  value={settings.n8n_token}
                  onChange={(e) => setSettings({ ...settings, n8n_token: e.target.value })}
                  placeholder="n8n_api_..."
                  className="border-input bg-background flex-1 rounded-lg border px-4 py-3 text-sm"
                />
                <button
                  onClick={testN8n}
                  disabled={
                    !settings.n8n_url || !settings.n8n_token || n8nTest.status === 'testing'
                  }
                  className="bg-muted hover:bg-muted/80 flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50"
                >
                  <TestStatusIcon status={n8nTest.status} />
                  Test
                </button>
              </div>
              {n8nTest.error && <p className="text-xs text-red-500">{n8nTest.error}</p>}
              {n8nTest.info && <p className="text-xs text-green-500">{n8nTest.info}</p>}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderWakeWordTab = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between rounded-xl border p-4">
        <label className="text-sm font-medium">Enable Wake Word</label>
        <Toggle
          enabled={settings.wake_word_enabled}
          onToggle={() =>
            setSettings({ ...settings, wake_word_enabled: !settings.wake_word_enabled })
          }
        />
      </div>

      {settings.wake_word_enabled && (
        <>
          <div className="space-y-2">
            <label className="text-sm font-medium">Wake Word Model</label>
            <select
              value={settings.wake_word_model}
              onChange={(e) => setSettings({ ...settings, wake_word_model: e.target.value })}
              className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
            >
              {wakeWordModels.length > 0 ? (
                wakeWordModels.map((model) => (
                  <option key={model} value={model}>
                    {model.replace('models/', '').replace('.onnx', '').replace(/_/g, ' ')}
                  </option>
                ))
              ) : (
                <option value={settings.wake_word_model}>
                  {settings.wake_word_model.replace('models/', '').replace('.onnx', '')}
                </option>
              )}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Threshold</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={settings.wake_word_threshold}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    wake_word_threshold: parseFloat(e.target.value) || 0.5,
                  })
                }
                className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Silence Timeout (s)</label>
              <input
                type="number"
                min="1"
                max="30"
                step="0.5"
                value={settings.wake_word_timeout}
                onChange={(e) =>
                  setSettings({ ...settings, wake_word_timeout: parseFloat(e.target.value) || 3.0 })
                }
                className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
              />
            </div>
          </div>
        </>
      )}
    </div>
  );

  // ---------------------------------------------------------------------------
  // Main render
  // ---------------------------------------------------------------------------

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex">
      {/* Overlay */}
      <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={onClose} />

      {/* Panel */}
      <div className="bg-background absolute inset-y-0 right-0 flex w-full flex-col shadow-2xl sm:w-[80%] sm:max-w-4xl">
        {/* Header */}
        <header className="shrink-0 border-b">
          <div className="flex items-center justify-between px-6 py-5">
            <h1 className="text-2xl font-bold">Settings</h1>
            <button
              onClick={onClose}
              className="text-muted-foreground hover:text-foreground hover:bg-muted rounded-full p-2 transition-colors"
            >
              <X className="h-6 w-6" weight="bold" />
            </button>
          </div>

          {/* Tabs */}
          <div className="overflow-x-auto px-6">
            <div className="flex gap-6">
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`border-b-2 pt-1 pb-3 text-sm font-semibold whitespace-nowrap transition-colors ${
                    activeTab === tab.id
                      ? 'border-primary text-primary'
                      : 'text-muted-foreground hover:text-foreground border-transparent'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-y-auto p-6">
          <div className="mx-auto max-w-3xl">
            {loading ? (
              <div className="text-muted-foreground py-8 text-center">Loading settings...</div>
            ) : (
              <>
                {error && (
                  <div className="mb-4 rounded-md bg-red-500/10 p-3 text-red-500">{error}</div>
                )}

                {activeTab === 'agent' && renderAgentTab()}
                {activeTab === 'prompt' && renderPromptTab()}
                {activeTab === 'providers' && renderProvidersTab()}
                {activeTab === 'llm' && renderLLMTab()}
                {activeTab === 'integrations' && renderIntegrationsTab()}
                {activeTab === 'wake' && renderWakeWordTab()}
              </>
            )}
          </div>
        </main>

        {/* Footer */}
        <div className="shrink-0 border-t p-6">
          <div className="mx-auto max-w-3xl">
            <Button
              variant="primary"
              onClick={handleSave}
              disabled={loading || saving}
              className="w-full py-3"
            >
              <FloppyDisk className="h-4 w-4" weight="bold" />
              {saving ? saveStatus || 'Saving...' : 'Save Changes'}
            </Button>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
