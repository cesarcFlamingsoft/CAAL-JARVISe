'use client';

import { useCallback, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  Check,
  CircleNotch,
  FloppyDisk,
  Microphone,
  MicrophoneSlash,
  Trash,
  UserPlus,
  X,
} from '@phosphor-icons/react/dist/ssr';
import { Button } from '@/components/livekit/button';

// =============================================================================
// Types
// =============================================================================

interface EmailAccount {
  id: string;
  provider: 'zoho' | 'generic_imap_smtp';
  display_name?: string;
  email: string;
  imap_host?: string;
  imap_port?: number;
  imap_ssl?: boolean;
  imap_username?: string;
  imap_password?: string;
  smtp_host?: string;
  smtp_port?: number;
  smtp_starttls?: boolean;
  smtp_username?: string;
  smtp_password?: string;
  default?: boolean;
}

interface CalendarSource {
  id: string;
  provider: 'zoho_caldav' | 'caldav' | 'ics' | 'google' | 'microsoft' | 'icloud';
  display_name?: string;
  url?: string;
  username?: string;
  password?: string;
  default?: boolean;
  writable?: boolean;
}

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
  native_tools_enabled: boolean;
  email_accounts: EmailAccount[];
  calendar_sources: CalendarSource[];
  reminders_provider: 'local' | 'apple';
  alarms_enabled: boolean;
  // Friday assistant
  friday_enabled: boolean;
  friday_host: string;
  friday_token: string;
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
  // Energy gate
  energy_gate_enabled: boolean;
  energy_gate_threshold_db: number;
  // TV/Media rejection
  tv_rejection_enabled: boolean;
  tv_rejection_min_liveness: number;
  media_noise_music_detection: boolean;
  media_noise_stereo_detection: boolean;
  media_noise_playback_voice_detection: boolean;
  // Speaker recognition
  speaker_recognition_enabled: boolean;
  speaker_recognition_threshold: number;
  speaker_recognition_required: boolean;
}

type TestStatus = 'idle' | 'testing' | 'success' | 'error';

type TabId = 'agent' | 'prompt' | 'providers' | 'llm' | 'audio' | 'integrations' | 'wake';

interface SpeakerInfo {
  name: string;
  enrollment_samples: number;
  created_at: number;
}

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
  native_tools_enabled: true,
  email_accounts: [],
  calendar_sources: [],
  reminders_provider: 'local',
  alarms_enabled: true,
  friday_enabled: false,
  friday_host: '',
  friday_token: '',
  wake_word_enabled: false,
  wake_word_model: 'models/hey_cal.onnx',
  wake_word_threshold: 0.5,
  wake_word_timeout: 3.0,
  allow_interruptions: true,
  min_endpointing_delay: 0.5,
  noise_suppression_enabled: false,
  noise_suppression_atten_db: 100,
  energy_gate_enabled: true,
  energy_gate_threshold_db: -35,
  // TV/Media rejection
  tv_rejection_enabled: true,
  tv_rejection_min_liveness: 0.2,
  media_noise_music_detection: true,
  media_noise_stereo_detection: true,
  media_noise_playback_voice_detection: true,
  // Speaker recognition
  speaker_recognition_enabled: false,
  speaker_recognition_threshold: 0.75,
  speaker_recognition_required: false,
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
  { id: 'audio', label: 'Audio Filter' },
  { id: 'integrations', label: 'Integrations' },
  { id: 'wake', label: 'Wake Word' },
];

const REDACTED_SECRET = '********';

const applyEmailProviderDefaults = (account: EmailAccount): EmailAccount => {
  const withProvider = { ...account };
  if (withProvider.provider === 'zoho') {
    withProvider.imap_host ||= 'imap.zoho.com';
    withProvider.imap_port ||= 993;
    withProvider.imap_ssl ??= true;
    withProvider.smtp_host ||= 'smtp.zoho.com';
    withProvider.smtp_port ||= 587;
    withProvider.smtp_starttls ??= true;
  }
  if (withProvider.email) {
    withProvider.imap_username ||= withProvider.email;
    withProvider.smtp_username ||= withProvider.email;
  }
  return withProvider;
};

const newEmailAccount = (index: number): EmailAccount =>
  applyEmailProviderDefaults({
    id: index === 0 ? 'personal' : `email-${index + 1}`,
    provider: 'zoho',
    display_name: index === 0 ? 'Personal' : `Email ${index + 1}`,
    email: '',
    default: index === 0,
  });

const newCalendarSource = (index: number): CalendarSource => ({
  id: index === 0 ? 'primary' : `calendar-${index + 1}`,
  provider: 'zoho_caldav',
  display_name: index === 0 ? 'Primary calendar' : `Calendar ${index + 1}`,
  url: '',
  username: '',
  password: '',
  default: index === 0,
  writable: true,
});

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
  const [fridayTest, setFridayTest] = useState<{
    status: TestStatus;
    error: string | null;
    info: string | null;
  }>({
    status: 'idle',
    error: null,
    info: null,
  });
  const [emailTests, setEmailTests] = useState<
    Record<string, { status: TestStatus; error: string | null; info: string | null }>
  >({});
  const [calendarTests, setCalendarTests] = useState<
    Record<string, { status: TestStatus; error: string | null; info: string | null }>
  >({});
  const [hassAgents, setHassAgents] = useState<{ id: string; name: string }[]>([]);

  const updateEmailAccount = (index: number, patch: Partial<EmailAccount>) => {
    const accounts = [...(settings.email_accounts || [])];
    const current = accounts[index] || newEmailAccount(index);
    const updated = applyEmailProviderDefaults({ ...current, ...patch });
    if (patch.default) {
      accounts.forEach((account, accountIndex) => {
        accounts[accountIndex] = { ...account, default: accountIndex === index };
      });
    }
    accounts[index] = updated;
    setSettings({ ...settings, email_accounts: accounts });
  };

  const addEmailAccount = () => {
    const accounts = settings.email_accounts || [];
    setSettings({ ...settings, email_accounts: [...accounts, newEmailAccount(accounts.length)] });
  };

  const removeEmailAccount = (index: number) => {
    setSettings({
      ...settings,
      email_accounts: (settings.email_accounts || []).filter((_, accountIndex) => accountIndex !== index),
    });
  };

  const updateCalendarSource = (index: number, patch: Partial<CalendarSource>) => {
    const sources = [...(settings.calendar_sources || [])];
    const updated = { ...(sources[index] || newCalendarSource(index)), ...patch };
    if (patch.default) {
      sources.forEach((source, sourceIndex) => {
        sources[sourceIndex] = { ...source, default: sourceIndex === index };
      });
    }
    sources[index] = updated;
    setSettings({ ...settings, calendar_sources: sources });
  };

  const addCalendarSource = () => {
    const sources = settings.calendar_sources || [];
    setSettings({ ...settings, calendar_sources: [...sources, newCalendarSource(sources.length)] });
  };

  const removeCalendarSource = (index: number) => {
    setSettings({
      ...settings,
      calendar_sources: (settings.calendar_sources || []).filter(
        (_, sourceIndex) => sourceIndex !== index
      ),
    });
  };

  // Speaker enrollment state
  const [speakers, setSpeakers] = useState<SpeakerInfo[]>([]);
  const [speakersLoading, setSpeakersLoading] = useState(false);
  const [newSpeakerName, setNewSpeakerName] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [audioSamples, setAudioSamples] = useState<string[]>([]);
  const [recordingTime, setRecordingTime] = useState(0);
  const [enrolling, setEnrolling] = useState(false);
  const [enrollmentError, setEnrollmentError] = useState<string | null>(null);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);

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
  // Speaker Management
  // ---------------------------------------------------------------------------

  const loadSpeakers = useCallback(async () => {
    if (!settings.speaker_recognition_enabled) return;

    setSpeakersLoading(true);
    try {
      const res = await fetch('/api/speakers');
      if (res.ok) {
        const data = await res.json();
        setSpeakers(data.speakers || []);
      }
    } catch (err) {
      console.error('Failed to load speakers:', err);
    } finally {
      setSpeakersLoading(false);
    }
  }, [settings.speaker_recognition_enabled]);

  useEffect(() => {
    if (isOpen && settings.speaker_recognition_enabled) {
      loadSpeakers();
    }
  }, [isOpen, settings.speaker_recognition_enabled, loadSpeakers]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());

        // Convert WebM to WAV using AudioContext
        const audioBlob = new Blob(chunks, { type: 'audio/webm' });
        const arrayBuffer = await audioBlob.arrayBuffer();

        try {
          const audioContext = new AudioContext({ sampleRate: 16000 });
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

          // Convert to WAV
          const wavBuffer = audioBufferToWav(audioBuffer);
          const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });

          // Convert to base64
          const reader = new FileReader();
          reader.onloadend = () => {
            const base64 = reader.result as string;
            setAudioSamples((prev) => [...prev, base64]);
          };
          reader.readAsDataURL(wavBlob);

          audioContext.close();
        } catch (err) {
          console.error('Failed to process audio:', err);
          setEnrollmentError('Failed to process audio recording');
        }
      };

      setMediaRecorder(recorder);
      setIsRecording(true);
      setRecordingTime(0);
      setEnrollmentError(null);
      recorder.start();

      // Update recording time every second
      const interval = setInterval(() => {
        setRecordingTime((t) => t + 1);
      }, 1000);

      // Auto-stop after 10 seconds
      setTimeout(() => {
        clearInterval(interval);
        if (recorder.state === 'recording') {
          recorder.stop();
          setIsRecording(false);
        }
      }, 10000);

      // Store interval for cleanup
      (recorder as MediaRecorder & { _interval?: NodeJS.Timeout })._interval = interval;
    } catch (err) {
      console.error('Failed to start recording:', err);
      setEnrollmentError('Microphone access denied');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      const recorder = mediaRecorder as MediaRecorder & { _interval?: NodeJS.Timeout };
      if (recorder._interval) {
        clearInterval(recorder._interval);
      }
      mediaRecorder.stop();
      setIsRecording(false);
    }
  }, [mediaRecorder]);

  const enrollSpeaker = useCallback(async () => {
    if (!newSpeakerName.trim()) {
      setEnrollmentError('Please enter a speaker name');
      return;
    }
    if (audioSamples.length === 0) {
      setEnrollmentError('Please record at least one audio sample');
      return;
    }

    setEnrolling(true);
    setEnrollmentError(null);

    try {
      const res = await fetch('/api/speakers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newSpeakerName.trim(),
          audio_data: audioSamples,
        }),
      });

      const result = await res.json();

      if (result.success) {
        setNewSpeakerName('');
        setAudioSamples([]);
        loadSpeakers();
      } else {
        setEnrollmentError(result.message || 'Failed to enroll speaker');
      }
    } catch (err) {
      console.error('Failed to enroll speaker:', err);
      setEnrollmentError('Failed to connect to server');
    } finally {
      setEnrolling(false);
    }
  }, [newSpeakerName, audioSamples, loadSpeakers]);

  const removeSpeaker = useCallback(
    async (name: string) => {
      try {
        const res = await fetch(`/api/speakers/${encodeURIComponent(name)}`, {
          method: 'DELETE',
        });

        if (res.ok) {
          loadSpeakers();
        }
      } catch (err) {
        console.error('Failed to remove speaker:', err);
      }
    },
    [loadSpeakers]
  );

  // Helper function to convert AudioBuffer to WAV
  function audioBufferToWav(audioBuffer: AudioBuffer): ArrayBuffer {
    const numChannels = 1; // Mono
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    // Get audio data (mono - use first channel or average)
    let samples: Float32Array;
    if (audioBuffer.numberOfChannels === 1) {
      samples = audioBuffer.getChannelData(0);
    } else {
      // Average channels for mono
      const left = audioBuffer.getChannelData(0);
      const right = audioBuffer.getChannelData(1);
      samples = new Float32Array(left.length);
      for (let i = 0; i < left.length; i++) {
        samples[i] = (left[i] + right[i]) / 2;
      }
    }

    const dataLength = samples.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    // WAV header
    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, dataLength, true);

    // Audio data
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      const sample = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }

    return buffer;
  }

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

  const testFriday = useCallback(async () => {
    if (!settings.friday_host || !settings.friday_token) return;
    setFridayTest({ status: 'testing', error: null, info: null });

    try {
      const res = await fetch('/api/setup/test-friday', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ host: settings.friday_host, token: settings.friday_token }),
      });
      const result = await res.json();

      if (result.success) {
        setFridayTest({ status: 'success', error: null, info: 'Connected' });
      } else {
        setFridayTest({ status: 'error', error: result.error || 'Connection failed', info: null });
      }
    } catch {
      setFridayTest({ status: 'error', error: 'Failed to connect', info: null });
    }
  }, [settings.friday_host, settings.friday_token]);

  const testEmailAccount = useCallback(async (account: EmailAccount) => {
    if (!account.id) return;
    setEmailTests((prev) => ({ ...prev, [account.id]: { status: 'testing', error: null, info: null } }));

    try {
      const res = await fetch('/api/setup/test-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: account.id }),
      });
      const result = await res.json();
      if (result.success) {
        const checks = result.data?.checks ? Object.keys(result.data.checks).join(' + ') : 'Connected';
        setEmailTests((prev) => ({
          ...prev,
          [account.id]: { status: 'success', error: null, info: checks || 'Connected' },
        }));
      } else {
        setEmailTests((prev) => ({
          ...prev,
          [account.id]: { status: 'error', error: result.error || 'Connection failed', info: null },
        }));
      }
    } catch {
      setEmailTests((prev) => ({
        ...prev,
        [account.id]: { status: 'error', error: 'Failed to connect', info: null },
      }));
    }
  }, []);

  const testCalendarSource = useCallback(async (source: CalendarSource) => {
    if (!source.id) return;
    setCalendarTests((prev) => ({
      ...prev,
      [source.id]: { status: 'testing', error: null, info: null },
    }));

    try {
      const res = await fetch('/api/setup/test-calendar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: source.id }),
      });
      const result = await res.json();
      if (result.success) {
        setCalendarTests((prev) => ({
          ...prev,
          [source.id]: { status: 'success', error: null, info: 'Connected' },
        }));
      } else {
        setCalendarTests((prev) => ({
          ...prev,
          [source.id]: { status: 'error', error: result.error || 'Connection failed', info: null },
        }));
      }
    } catch {
      setCalendarTests((prev) => ({
        ...prev,
        [source.id]: { status: 'error', error: 'Failed to connect', info: null },
      }));
    }
  }, []);

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

      {/* Energy Gate */}
      <div className="overflow-hidden rounded-xl border">
        <div className="bg-muted/50 flex items-center justify-between border-b px-4 py-3">
          <div>
            <span className="font-semibold">Energy Gate</span>
            <p className="text-muted-foreground text-xs">
              Filter quiet/distant sounds (TV, background chatter)
            </p>
          </div>
          <Toggle
            enabled={settings.energy_gate_enabled}
            onToggle={() =>
              setSettings({ ...settings, energy_gate_enabled: !settings.energy_gate_enabled })
            }
          />
        </div>

        <div className="space-y-4 p-4">
          <p className="text-muted-foreground text-sm">
            Filters out audio below a volume threshold, preventing distant sounds like TV or
            background conversations from triggering the wake word.
          </p>

          {settings.energy_gate_enabled && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Threshold</label>
                <span className="text-muted-foreground text-sm">
                  {settings.energy_gate_threshold_db}dB
                </span>
              </div>
              <input
                type="range"
                min="-50"
                max="-20"
                step="1"
                value={settings.energy_gate_threshold_db}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    energy_gate_threshold_db: parseFloat(e.target.value),
                  })
                }
                className="bg-muted accent-primary h-2 w-full cursor-pointer appearance-none rounded-lg"
              />
              <div className="text-muted-foreground flex justify-between text-xs">
                <span>More Sensitive (-50)</span>
                <span>Less Sensitive (-20)</span>
              </div>
              <p className="text-muted-foreground text-xs">
                -40dB = quiet room, -35dB = normal speech, -30dB = loud speech
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderIntegrationsTab = () => (
    <div className="space-y-4">
      {/* Native Assistant Tools */}
      <div className="overflow-hidden rounded-xl border border-green-500/30">
        <div className="bg-green-500/10 flex items-center justify-between border-b px-4 py-3">
          <div>
            <span className="font-semibold">Native Assistant Tools</span>
            <p className="text-muted-foreground text-xs">
              Preferred JARVIS capabilities: email, calendars, reminders, alarms and timers.
            </p>
          </div>
          <Toggle
            enabled={settings.native_tools_enabled}
            onToggle={() =>
              setSettings({ ...settings, native_tools_enabled: !settings.native_tools_enabled })
            }
          />
        </div>

        {settings.native_tools_enabled && (
          <div className="space-y-5 p-4">
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="space-y-3 rounded-lg border p-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <label className="text-sm font-medium">Email accounts</label>
                    <p className="text-muted-foreground text-xs">
                      Runtime SMTP + IMAP accounts. Zoho custom domains work out of the box.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={addEmailAccount}
                    className="bg-muted hover:bg-muted/80 rounded-lg px-3 py-2 text-xs font-medium"
                  >
                    Add email
                  </button>
                </div>

                {(settings.email_accounts || []).length === 0 && (
                  <p className="text-muted-foreground rounded-lg border border-dashed p-3 text-xs">
                    No email accounts configured. Add your Zoho Mail account here — no Docker edits.
                  </p>
                )}

                {(settings.email_accounts || []).map((account, index) => (
                  <div key={`${account.id}-${index}`} className="space-y-3 rounded-lg border p-3">
                    <div className="flex items-center justify-between gap-2">
                      <input
                        type="text"
                        value={account.display_name || ''}
                        onChange={(e) => updateEmailAccount(index, { display_name: e.target.value })}
                        placeholder="Display name"
                        className="border-input bg-background flex-1 rounded-lg border px-3 py-2 text-sm"
                      />
                      <button
                        type="button"
                        onClick={() => removeEmailAccount(index)}
                        className="text-muted-foreground hover:text-red-500 rounded-lg p-2"
                        aria-label="Remove email account"
                      >
                        <Trash size={16} />
                      </button>
                    </div>
                    <div className="grid gap-2 md:grid-cols-2">
                      <input
                        type="text"
                        value={account.id || ''}
                        onChange={(e) => updateEmailAccount(index, { id: e.target.value })}
                        placeholder="account id, e.g. personal"
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm"
                      />
                      <select
                        value={account.provider || 'zoho'}
                        onChange={(e) =>
                          updateEmailAccount(index, {
                            provider: e.target.value as EmailAccount['provider'],
                          })
                        }
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm"
                      >
                        <option value="zoho">Zoho Mail</option>
                        <option value="generic_imap_smtp">Generic IMAP/SMTP</option>
                      </select>
                      <input
                        type="email"
                        value={account.email || ''}
                        onChange={(e) =>
                          updateEmailAccount(index, {
                            email: e.target.value,
                            imap_username: e.target.value,
                            smtp_username: e.target.value,
                          })
                        }
                        placeholder="name@yourdomain.com"
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm md:col-span-2"
                      />
                    </div>
                    <div className="grid gap-2 md:grid-cols-2">
                      <input
                        type="text"
                        value={account.imap_host || ''}
                        onChange={(e) => updateEmailAccount(index, { imap_host: e.target.value })}
                        placeholder="IMAP host"
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm"
                      />
                      <input
                        type="number"
                        value={account.imap_port || 993}
                        onChange={(e) => updateEmailAccount(index, { imap_port: Number(e.target.value) })}
                        placeholder="IMAP port"
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm"
                      />
                      <input
                        type="text"
                        value={account.smtp_host || ''}
                        onChange={(e) => updateEmailAccount(index, { smtp_host: e.target.value })}
                        placeholder="SMTP host"
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm"
                      />
                      <input
                        type="number"
                        value={account.smtp_port || 587}
                        onChange={(e) => updateEmailAccount(index, { smtp_port: Number(e.target.value) })}
                        placeholder="SMTP port"
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm"
                      />
                      <input
                        type="password"
                        value={account.imap_password || ''}
                        onChange={(e) => updateEmailAccount(index, { imap_password: e.target.value })}
                        placeholder={`IMAP app password (${REDACTED_SECRET} keeps existing)`}
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm"
                      />
                      <input
                        type="password"
                        value={account.smtp_password || ''}
                        onChange={(e) => updateEmailAccount(index, { smtp_password: e.target.value })}
                        placeholder={`SMTP app password (${REDACTED_SECRET} keeps existing)`}
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm"
                      />
                    </div>
                    <div className="flex flex-wrap gap-4 text-xs">
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={account.default || false}
                          onChange={(e) => updateEmailAccount(index, { default: e.target.checked })}
                        />
                        Default account
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={account.imap_ssl ?? true}
                          onChange={(e) => updateEmailAccount(index, { imap_ssl: e.target.checked })}
                        />
                        IMAP SSL
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={account.smtp_starttls ?? true}
                          onChange={(e) => updateEmailAccount(index, { smtp_starttls: e.target.checked })}
                        />
                        SMTP STARTTLS
                      </label>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => testEmailAccount(account)}
                        disabled={!account.id || emailTests[account.id]?.status === 'testing'}
                        className="bg-muted hover:bg-muted/80 flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium disabled:opacity-50"
                      >
                        <TestStatusIcon status={emailTests[account.id]?.status || 'idle'} />
                        Test email
                      </button>
                      {emailTests[account.id]?.info && (
                        <p className="text-xs text-green-500">{emailTests[account.id]?.info}</p>
                      )}
                      {emailTests[account.id]?.error && (
                        <p className="text-xs text-red-500">{emailTests[account.id]?.error}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              <div className="space-y-3 rounded-lg border p-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <label className="text-sm font-medium">Calendar sources</label>
                    <p className="text-muted-foreground text-xs">
                      Add Zoho CalDAV, iCloud, generic CalDAV, ICS, Google, or Outlook sources.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={addCalendarSource}
                    className="bg-muted hover:bg-muted/80 rounded-lg px-3 py-2 text-xs font-medium"
                  >
                    Add calendar
                  </button>
                </div>

                {(settings.calendar_sources || []).length === 0 && (
                  <p className="text-muted-foreground rounded-lg border border-dashed p-3 text-xs">
                    No calendars configured. Add a source here and JARVIS can read it at runtime.
                  </p>
                )}

                {(settings.calendar_sources || []).map((source, index) => (
                  <div key={`${source.id}-${index}`} className="space-y-3 rounded-lg border p-3">
                    <div className="flex items-center justify-between gap-2">
                      <input
                        type="text"
                        value={source.display_name || ''}
                        onChange={(e) => updateCalendarSource(index, { display_name: e.target.value })}
                        placeholder="Display name"
                        className="border-input bg-background flex-1 rounded-lg border px-3 py-2 text-sm"
                      />
                      <button
                        type="button"
                        onClick={() => removeCalendarSource(index)}
                        className="text-muted-foreground hover:text-red-500 rounded-lg p-2"
                        aria-label="Remove calendar source"
                      >
                        <Trash size={16} />
                      </button>
                    </div>
                    <div className="grid gap-2 md:grid-cols-2">
                      <input
                        type="text"
                        value={source.id || ''}
                        onChange={(e) => updateCalendarSource(index, { id: e.target.value })}
                        placeholder="source id, e.g. work"
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm"
                      />
                      <select
                        value={source.provider || 'zoho_caldav'}
                        onChange={(e) =>
                          updateCalendarSource(index, {
                            provider: e.target.value as CalendarSource['provider'],
                          })
                        }
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm"
                      >
                        <option value="zoho_caldav">Zoho Calendar / CalDAV</option>
                        <option value="caldav">Generic CalDAV</option>
                        <option value="icloud">Apple iCloud CalDAV</option>
                        <option value="ics">ICS feed (read-only)</option>
                        <option value="google">Google Calendar</option>
                        <option value="microsoft">Outlook / Microsoft 365</option>
                      </select>
                      <input
                        type="text"
                        value={source.url || ''}
                        onChange={(e) => updateCalendarSource(index, { url: e.target.value })}
                        placeholder="CalDAV or ICS URL"
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm md:col-span-2"
                      />
                      <input
                        type="text"
                        value={source.username || ''}
                        onChange={(e) => updateCalendarSource(index, { username: e.target.value })}
                        placeholder="username / email"
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm"
                      />
                      <input
                        type="password"
                        value={source.password || ''}
                        onChange={(e) => updateCalendarSource(index, { password: e.target.value })}
                        placeholder={`app password (${REDACTED_SECRET} keeps existing)`}
                        className="border-input bg-background rounded-lg border px-3 py-2 text-sm"
                      />
                    </div>
                    <div className="flex flex-wrap gap-4 text-xs">
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={source.default || false}
                          onChange={(e) => updateCalendarSource(index, { default: e.target.checked })}
                        />
                        Default calendar
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={source.writable ?? true}
                          onChange={(e) => updateCalendarSource(index, { writable: e.target.checked })}
                        />
                        Writable
                      </label>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => testCalendarSource(source)}
                        disabled={!source.id || calendarTests[source.id]?.status === 'testing'}
                        className="bg-muted hover:bg-muted/80 flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium disabled:opacity-50"
                      >
                        <TestStatusIcon status={calendarTests[source.id]?.status || 'idle'} />
                        Test calendar
                      </button>
                      {calendarTests[source.id]?.info && (
                        <p className="text-xs text-green-500">{calendarTests[source.id]?.info}</p>
                      )}
                      {calendarTests[source.id]?.error && (
                        <p className="text-xs text-red-500">{calendarTests[source.id]?.error}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2 rounded-lg border p-3">
                <label className="text-sm font-medium">Reminders provider</label>
                <select
                  value={settings.reminders_provider}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      reminders_provider: e.target.value as Settings['reminders_provider'],
                    })
                  }
                  className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
                >
                  <option value="local">Local CAAL reminders</option>
                  <option value="apple">Apple Reminders via remindctl</option>
                </select>
                <p className="text-muted-foreground text-xs">
                  Local works in Docker; Apple requires host-side bridge access.
                </p>
              </div>

              <div className="flex items-center justify-between rounded-lg border p-3">
                <div>
                  <label className="text-sm font-medium">Alarms and timers</label>
                  <p className="text-muted-foreground text-xs">
                    Let JARVIS schedule audible alarms and timer announcements.
                  </p>
                </div>
                <Toggle
                  enabled={settings.alarms_enabled}
                  onToggle={() => setSettings({ ...settings, alarms_enabled: !settings.alarms_enabled })}
                />
              </div>
            </div>
          </div>
        )}
      </div>

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

      {/* Legacy n8n */}
      <div
        className={`overflow-hidden rounded-xl border ${!settings.n8n_enabled ? 'opacity-60' : ''}`}
      >
        <div className="bg-muted/50 flex items-center justify-between border-b px-4 py-3">
          <div>
            <span className="font-semibold">n8n Legacy Workflows</span>
            <p className="text-muted-foreground text-xs">
              Optional fallback. Native Assistant Tools are the primary path.
            </p>
          </div>
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

      {/* Friday Assistant */}
      <div
        className={`overflow-hidden rounded-xl border ${!settings.friday_enabled ? 'opacity-60' : ''}`}
      >
        <div className="bg-muted/50 flex items-center justify-between border-b px-4 py-3">
          <span className="font-semibold">Friday Assistant</span>
          <Toggle
            enabled={settings.friday_enabled}
            onToggle={() => setSettings({ ...settings, friday_enabled: !settings.friday_enabled })}
          />
        </div>

        {settings.friday_enabled && (
          <div className="space-y-4 p-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Host URL</label>
              <input
                type="text"
                value={settings.friday_host}
                onChange={(e) => setSettings({ ...settings, friday_host: e.target.value })}
                placeholder="http://10.0.0.55:18789"
                className="border-input bg-background w-full rounded-lg border px-4 py-3 text-sm"
              />
              <p className="text-muted-foreground text-xs">
                Clawdbot API endpoint for advanced AI assistance
              </p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">API Token</label>
              <div className="flex gap-2">
                <input
                  type="password"
                  value={settings.friday_token}
                  onChange={(e) => setSettings({ ...settings, friday_token: e.target.value })}
                  placeholder="your_api_token"
                  className="border-input bg-background flex-1 rounded-lg border px-4 py-3 text-sm"
                />
                <button
                  onClick={testFriday}
                  disabled={
                    !settings.friday_host || !settings.friday_token || fridayTest.status === 'testing'
                  }
                  className="bg-muted hover:bg-muted/80 flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50"
                >
                  <TestStatusIcon status={fridayTest.status} />
                  Test
                </button>
              </div>
              {fridayTest.error && <p className="text-xs text-red-500">{fridayTest.error}</p>}
              {fridayTest.info && <p className="text-xs text-green-500">{fridayTest.info}</p>}
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

  const renderAudioTab = () => (
    <div className="space-y-6">
      {/* TV/Media Rejection */}
      <div className="overflow-hidden rounded-xl border">
        <div className="bg-muted/50 flex items-center justify-between border-b px-4 py-3">
          <div>
            <span className="font-semibold">TV / Media Rejection</span>
            <p className="text-muted-foreground text-xs">
              Filter out TV, radio, and video playback
            </p>
          </div>
          <Toggle
            enabled={settings.tv_rejection_enabled}
            onToggle={() =>
              setSettings({ ...settings, tv_rejection_enabled: !settings.tv_rejection_enabled })
            }
          />
        </div>

        {settings.tv_rejection_enabled && (
          <div className="space-y-4 p-4">
            <p className="text-muted-foreground text-sm">
              Uses AI-powered audio analysis to distinguish live speech from media playback,
              preventing the assistant from responding to TV shows, music, or videos.
            </p>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Liveness Threshold</label>
                <span className="text-muted-foreground text-sm">
                  {settings.tv_rejection_min_liveness}
                </span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.5"
                step="0.05"
                value={settings.tv_rejection_min_liveness}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    tv_rejection_min_liveness: parseFloat(e.target.value),
                  })
                }
                className="bg-muted accent-primary h-2 w-full cursor-pointer appearance-none rounded-lg"
              />
              <div className="text-muted-foreground flex justify-between text-xs">
                <span>More Permissive</span>
                <span>Stricter</span>
              </div>
            </div>

            <div className="space-y-3 pt-2">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium">Music Detection</label>
                  <p className="text-muted-foreground text-xs">Detect radio and video music</p>
                </div>
                <Toggle
                  enabled={settings.media_noise_music_detection}
                  onToggle={() =>
                    setSettings({
                      ...settings,
                      media_noise_music_detection: !settings.media_noise_music_detection,
                    })
                  }
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium">Stereo Detection</label>
                  <p className="text-muted-foreground text-xs">
                    Detect wide stereo (media vs live speech)
                  </p>
                </div>
                <Toggle
                  enabled={settings.media_noise_stereo_detection}
                  onToggle={() =>
                    setSettings({
                      ...settings,
                      media_noise_stereo_detection: !settings.media_noise_stereo_detection,
                    })
                  }
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium">Playback Voice Detection</label>
                  <p className="text-muted-foreground text-xs">
                    Detect voices from video/podcast playback
                  </p>
                </div>
                <Toggle
                  enabled={settings.media_noise_playback_voice_detection}
                  onToggle={() =>
                    setSettings({
                      ...settings,
                      media_noise_playback_voice_detection:
                        !settings.media_noise_playback_voice_detection,
                    })
                  }
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Speaker Recognition */}
      <div className="overflow-hidden rounded-xl border">
        <div className="bg-muted/50 flex items-center justify-between border-b px-4 py-3">
          <div>
            <span className="font-semibold">Speaker Recognition</span>
            <p className="text-muted-foreground text-xs">
              Voice biometrics like Alexa/Google Voice Match
            </p>
          </div>
          <Toggle
            enabled={settings.speaker_recognition_enabled}
            onToggle={() =>
              setSettings({
                ...settings,
                speaker_recognition_enabled: !settings.speaker_recognition_enabled,
              })
            }
          />
        </div>

        {settings.speaker_recognition_enabled && (
          <div className="space-y-4 p-4">
            <div className="rounded-lg border border-blue-500/30 bg-blue-500/10 p-3">
              <p className="text-sm text-blue-200">
                <span className="font-semibold">Requires:</span> resemblyzer package
                <br />
                <span className="text-xs opacity-70">
                  Run ./start-apple.sh to auto-install, or: pip install resemblyzer
                </span>
              </p>
            </div>

            {/* Enrolled Speakers List */}
            <div className="space-y-2">
              <label className="text-sm font-semibold">Enrolled Speakers</label>
              {speakersLoading ? (
                <div className="text-muted-foreground flex items-center gap-2 text-sm">
                  <CircleNotch className="h-4 w-4 animate-spin" />
                  Loading speakers...
                </div>
              ) : speakers.length === 0 ? (
                <p className="text-muted-foreground text-sm">
                  No speakers enrolled yet. Add a speaker below.
                </p>
              ) : (
                <div className="space-y-2">
                  {speakers.map((speaker) => (
                    <div
                      key={speaker.name}
                      className="bg-muted/50 flex items-center justify-between rounded-lg px-3 py-2"
                    >
                      <div>
                        <span className="font-medium">{speaker.name}</span>
                        <span className="text-muted-foreground ml-2 text-xs">
                          ({speaker.enrollment_samples} sample
                          {speaker.enrollment_samples !== 1 ? 's' : ''})
                        </span>
                      </div>
                      <button
                        onClick={() => removeSpeaker(speaker.name)}
                        className="text-muted-foreground hover:text-red-400"
                        title="Remove speaker"
                      >
                        <Trash className="h-4 w-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Enroll New Speaker */}
            <div className="border-t pt-4">
              <label className="text-sm font-semibold">Enroll New Speaker</label>
              <p className="text-muted-foreground mb-3 text-xs">
                Record 3-10 seconds of speech to create a voice profile
              </p>

              {enrollmentError && (
                <div className="mb-3 rounded-lg border border-red-500/30 bg-red-500/10 p-2 text-sm text-red-300">
                  {enrollmentError}
                </div>
              )}

              <div className="space-y-3">
                <input
                  type="text"
                  placeholder="Speaker name"
                  value={newSpeakerName}
                  onChange={(e) => setNewSpeakerName(e.target.value)}
                  className="bg-muted border-input w-full rounded-lg border px-3 py-2 text-sm"
                />

                <div className="flex items-center gap-3">
                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={enrolling}
                    className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                      isRecording
                        ? 'bg-red-500 text-white hover:bg-red-600'
                        : 'bg-primary text-primary-foreground hover:bg-primary/90'
                    }`}
                  >
                    {isRecording ? (
                      <>
                        <MicrophoneSlash className="h-4 w-4" />
                        Stop ({10 - recordingTime}s)
                      </>
                    ) : (
                      <>
                        <Microphone className="h-4 w-4" />
                        Record Sample
                      </>
                    )}
                  </button>

                  {audioSamples.length > 0 && (
                    <span className="text-muted-foreground text-sm">
                      {audioSamples.length} sample{audioSamples.length !== 1 ? 's' : ''} recorded
                    </span>
                  )}
                </div>

                {audioSamples.length > 0 && (
                  <div className="flex gap-2">
                    <button
                      onClick={enrollSpeaker}
                      disabled={enrolling || !newSpeakerName.trim()}
                      className="bg-primary text-primary-foreground hover:bg-primary/90 flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
                    >
                      {enrolling ? (
                        <>
                          <CircleNotch className="h-4 w-4 animate-spin" />
                          Enrolling...
                        </>
                      ) : (
                        <>
                          <UserPlus className="h-4 w-4" />
                          Enroll Speaker
                        </>
                      )}
                    </button>
                    <button
                      onClick={() => setAudioSamples([])}
                      disabled={enrolling}
                      className="text-muted-foreground hover:text-foreground text-sm"
                    >
                      Clear samples
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Settings */}
            <div className="space-y-4 border-t pt-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Verification Threshold</label>
                  <span className="text-muted-foreground text-sm">
                    {settings.speaker_recognition_threshold}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="0.9"
                  step="0.05"
                  value={settings.speaker_recognition_threshold}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      speaker_recognition_threshold: parseFloat(e.target.value),
                    })
                  }
                  className="bg-muted accent-primary h-2 w-full cursor-pointer appearance-none rounded-lg"
                />
                <div className="text-muted-foreground flex justify-between text-xs">
                  <span>More Lenient (0.5)</span>
                  <span>Stricter (0.9)</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium">Require Recognized Speaker</label>
                  <p className="text-muted-foreground text-xs">
                    Only respond to enrolled speakers
                  </p>
                </div>
                <Toggle
                  enabled={settings.speaker_recognition_required}
                  onToggle={() =>
                    setSettings({
                      ...settings,
                      speaker_recognition_required: !settings.speaker_recognition_required,
                    })
                  }
                />
              </div>
            </div>
          </div>
        )}
      </div>
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
                {activeTab === 'audio' && renderAudioTab()}
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
