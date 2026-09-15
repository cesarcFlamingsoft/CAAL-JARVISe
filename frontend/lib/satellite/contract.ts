export type Enrollment = {
  id: string;
  credential: string;
  satellite_id: string;
  device_id: string;
};
export type Scope = 'conversation' | 'states' | 'states_and_lights';
export type Registered = {
  id: string;
  satellite_id: string;
  device_id: string;
  connection_id: string | null;
  scope: Scope;
  active: number;
};
export type Device = {
  satellite_id: string;
  device_id: string;
  name: string;
  enrollment: Registered | null;
};
export type Status = {
  devices: Device[];
  enrollments: Registered[];
  connections: { id: string; label: string; shared: boolean; ha_admin: boolean }[];
  connection_id: string | null;
  provider_identity: { is_admin: boolean; is_owner: boolean } | null;
  personal_data: false;
  error: string | null;
};
export const satelliteId = (v: unknown): v is string =>
  typeof v === 'string' && /^sat_[a-f0-9]{24}$/.test(v);
export const connectionId = (v: unknown): v is string =>
  typeof v === 'string' && /^ha_[a-f0-9]{24}$/.test(v);
export const entityId = (v: unknown): v is string =>
  typeof v === 'string' && /^assist_satellite\.[a-z0-9_]{1,150}$/.test(v);
const deviceId = (v: unknown): v is string => typeof v === 'string' && /^[a-f0-9]{32}$/.test(v);
export const scopeValue = (v: unknown): v is Scope =>
  ['conversation', 'states', 'states_and_lights'].includes(String(v));
export function parseEnrollment(value: unknown): Enrollment | null {
  if (!value || typeof value !== 'object') return null;
  const v = value as Record<string, unknown>;
  return satelliteId(v.id) &&
    typeof v.credential === 'string' &&
    /^[A-Za-z0-9_-]{43}$/.test(v.credential) &&
    entityId(v.satellite_id) &&
    deviceId(v.device_id)
    ? { id: v.id, credential: v.credential, satellite_id: v.satellite_id, device_id: v.device_id }
    : null;
}
function registered(v: unknown): v is Registered {
  if (!v || typeof v !== 'object') return false;
  const r = v as Record<string, unknown>;
  return (
    satelliteId(r.id) &&
    entityId(r.satellite_id) &&
    deviceId(r.device_id) &&
    (r.connection_id === null || connectionId(r.connection_id)) &&
    scopeValue(r.scope) &&
    r.active === 1
  );
}
export function parseStatus(value: unknown): Status | null {
  if (!value || typeof value !== 'object') return null;
  const v = value as Record<string, unknown>;
  if (
    v.personal_data !== false ||
    !Array.isArray(v.devices) ||
    !Array.isArray(v.enrollments) ||
    !Array.isArray(v.connections) ||
    !(v.connection_id === null || connectionId(v.connection_id)) ||
    !(v.error === null || typeof v.error === 'string')
  )
    return null;
  if (
    !v.enrollments.every(registered) ||
    !v.devices.every(
      (d) =>
        d &&
        entityId(d.satellite_id) &&
        deviceId(d.device_id) &&
        typeof d.name === 'string' &&
        (d.enrollment === null || registered(d.enrollment))
    )
  )
    return null;
  if (
    !v.connections.every(
      (c) =>
        c &&
        connectionId(c.id) &&
        typeof c.label === 'string' &&
        typeof c.shared === 'boolean' &&
        typeof c.ha_admin === 'boolean'
    )
  )
    return null;
  const account = v.provider_identity as Record<string, unknown> | null;
  if (
    account !== null &&
    (!account || typeof account.is_admin !== 'boolean' || typeof account.is_owner !== 'boolean')
  )
    return null;
  return value as Status;
}
