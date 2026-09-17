/**
 * The "running work" widget's only live inputs are the LiveKit room itself:
 * the agent's state and the `tool_status` data packets it publishes after each
 * response. Both arrive untrusted, so they are parsed defensively here and
 * reduced into a small, bounded feed the widget can render.
 */

export const MAX_ACTIVITY_ENTRIES = 12;
const MAX_TOOL_NAME = 80;

export interface ToolStatusPacket {
  toolUsed: boolean;
  toolNames: string[];
}

export interface ToolActivity {
  id: string;
  tools: string[];
  /** Unix milliseconds. */
  at: number;
}

/** Read a `tool_status` packet; null for anything that is not one. */
export function parseToolStatus(payload: unknown): ToolStatusPacket | null {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return null;
  const data = payload as Record<string, unknown>;
  if (typeof data.tool_used !== 'boolean') return null;
  const toolNames = Array.isArray(data.tool_names)
    ? data.tool_names
        .filter((name): name is string => typeof name === 'string' && name.trim().length > 0)
        .map((name) => name.trim().slice(0, MAX_TOOL_NAME))
    : [];
  return { toolUsed: data.tool_used, toolNames };
}

let sequence = 0;

/** Prepend a tool call to the feed; responses that used no tool are not work. */
export function appendToolActivity(
  feed: ToolActivity[],
  packet: ToolStatusPacket,
  at: number,
  limit = MAX_ACTIVITY_ENTRIES
): ToolActivity[] {
  if (!packet.toolUsed || packet.toolNames.length === 0) return feed;
  sequence += 1;
  const entry: ToolActivity = { id: `act_${at}_${sequence}`, tools: [...packet.toolNames], at };
  return [entry, ...feed].slice(0, Math.max(1, limit));
}

export type VoiceTone = 'idle' | 'busy' | 'live' | 'error';

export interface VoiceStatusInput {
  isConnected: boolean;
  connecting: boolean;
  agentState?: string;
  connectionState?: string;
}

export interface VoiceStatus {
  label: string;
  tone: VoiceTone;
}

/** A short, truthful description of the docked voice capability. */
export function voiceStatus({
  isConnected,
  connecting,
  agentState,
  connectionState,
}: VoiceStatusInput): VoiceStatus {
  if (connectionState === 'reconnecting' || connectionState === 'signalReconnecting') {
    return { label: 'Reconnecting voice…', tone: 'busy' };
  }
  if (connectionState === 'disconnected') isConnected = false;
  if (!isConnected) {
    return connecting
      ? { label: 'Connecting to FRIDAY…', tone: 'busy' }
      : { label: 'Voice is idle', tone: 'idle' };
  }
  switch (agentState) {
    case 'listening':
      return { label: 'FRIDAY is listening', tone: 'live' };
    case 'thinking':
      return { label: 'FRIDAY is thinking', tone: 'busy' };
    case 'speaking':
      return { label: 'FRIDAY is speaking', tone: 'live' };
    case 'idle':
      return { label: 'FRIDAY is on the call', tone: 'live' };
    case 'failed':
      return { label: 'FRIDAY did not join the call', tone: 'error' };
    default:
      return { label: 'Waiting for FRIDAY to join…', tone: 'busy' };
  }
}
