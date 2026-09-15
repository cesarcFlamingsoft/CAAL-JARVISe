/** Shared BFF reducer/browser parser. Infrastructure and credentials never pass through. */
export const providers = ['kokoro', 'qwen-trial', 'voicebox', 'piper'] as const;
export const engines = [
  'qwen',
  'qwen_custom_voice',
  'luxtts',
  'chatterbox',
  'chatterbox_turbo',
  'tada',
  'kokoro',
] as const;
export type Provider = (typeof providers)[number];
export interface TtsView {
  provider: Provider;
  source: 'personal' | 'default';
  qwen_configured: boolean;
  qwen_voice: string;
  applies_to: 'new_sessions';
  voicebox_status: 'not_configured' | 'unavailable' | 'api_verified' | 'configuration_changed';
  can_configure: boolean;
  profile_id: string | null;
  engine: string | null;
  model_size: string | null;
}
export function parseTtsView(value: unknown): TtsView | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const v = value as Record<string, unknown>;
  if (
    !providers.includes(v.provider as Provider) ||
    !['personal', 'default'].includes(v.source as string) ||
    typeof v.qwen_configured !== 'boolean' ||
    typeof v.can_configure !== 'boolean' ||
    v.applies_to !== 'new_sessions' ||
    typeof v.qwen_voice !== 'string' ||
    v.qwen_voice.length > 100 ||
    !['not_configured', 'unavailable', 'api_verified', 'configuration_changed'].includes(
      v.voicebox_status as string
    )
  )
    return null;
  if (
    v.profile_id !== null &&
    (typeof v.profile_id !== 'string' || !/^[A-Za-z0-9_-]{1,100}$/.test(v.profile_id))
  )
    return null;
  if (v.engine !== null && !engines.includes(v.engine as (typeof engines)[number])) return null;
  if (v.model_size !== null && !['1.7B', '0.6B', '1B', '3B'].includes(v.model_size as string))
    return null;
  if (v.provider === 'voicebox' && (!v.profile_id || !v.engine || !v.model_size)) return null;
  return {
    provider: v.provider as Provider,
    source: v.source as TtsView['source'],
    qwen_configured: v.qwen_configured,
    qwen_voice: v.qwen_voice,
    applies_to: 'new_sessions',
    voicebox_status: v.voicebox_status as TtsView['voicebox_status'],
    can_configure: v.can_configure,
    profile_id: v.profile_id as string | null,
    engine: v.engine as string | null,
    model_size: v.model_size as string | null,
  };
}
export function parseVoiceboxConfig(
  value: unknown
): { endpoint: string; credential_configured: boolean } | null {
  if (!value || typeof value !== 'object') return null;
  const v = value as Record<string, unknown>;
  return typeof v.endpoint === 'string' &&
    v.endpoint.length <= 200 &&
    typeof v.credential_configured === 'boolean'
    ? { endpoint: v.endpoint, credential_configured: v.credential_configured }
    : null;
}
