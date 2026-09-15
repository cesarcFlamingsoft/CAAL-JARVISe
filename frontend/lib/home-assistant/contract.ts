export interface HAAccess {
  enabled: boolean;
  status: 'denied' | 'connection_required' | 'connected' | 'service_account';
  connection_id: string | null;
  scope: 'states_and_lights';
  can_connect: boolean;
  connections: { id: string; label: string; shared: boolean; ha_admin: boolean }[];
}
export const connectionId = (v: unknown): v is string =>
  typeof v === 'string' && /^ha_[a-f0-9]{24}$/.test(v);
export function parseAccess(value: unknown): HAAccess | null {
  if (!value || typeof value !== 'object') return null;
  const v = value as Record<string, unknown>;
  if (
    typeof v.enabled !== 'boolean' ||
    !['denied', 'connection_required', 'connected', 'service_account'].includes(String(v.status)) ||
    (v.connection_id !== null && !connectionId(v.connection_id)) ||
    v.scope !== 'states_and_lights' ||
    typeof v.can_connect !== 'boolean' ||
    !Array.isArray(v.connections) ||
    v.connections.length > 32
  )
    return null;
  const connections: HAAccess['connections'] = [];
  for (const c of v.connections) {
    if (
      !c ||
      !connectionId(c.id) ||
      typeof c.label !== 'string' ||
      c.label.length > 80 ||
      typeof c.shared !== 'boolean' ||
      typeof c.ha_admin !== 'boolean'
    )
      return null;
    connections.push({ id: c.id, label: c.label, shared: c.shared, ha_admin: c.ha_admin });
  }
  return {
    enabled: v.enabled,
    status: v.status as HAAccess['status'],
    connection_id: v.connection_id as string | null,
    scope: 'states_and_lights',
    can_connect: v.can_connect,
    connections,
  };
}

export function authorizationOriginAllowed(
  origin: string | null,
  publicOrigin: string | null
): boolean {
  return Boolean(publicOrigin?.startsWith('https://') && origin === publicOrigin);
}
