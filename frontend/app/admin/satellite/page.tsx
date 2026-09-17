'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { apiRequest, explain } from '@/components/account/api-client';
import {
  type Device,
  type Enrollment,
  type Scope,
  type Status,
  parseEnrollment,
  parseStatus,
} from '@/lib/satellite/contract';

export default function SatellitePage() {
  const [status, setStatus] = useState<Status | null>(null);
  const [selected, setSelected] = useState('');
  const [issued, setIssued] = useState<Enrollment | null>(null);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [scopes, setScopes] = useState<Record<string, Scope>>({});
  const load = async (connection = selected) => {
    const r = await apiRequest<unknown>(
      '/api/admin/satellites' +
        (connection ? '?connection_id=' + encodeURIComponent(connection) : '')
    );
    if (r.ok) {
      const parsed = parseStatus(r.data);
      if (parsed) {
        setStatus(parsed);
        setMessage(parsed.error || '');
      } else setMessage('The device inventory could not be read.');
    } else setMessage(explain(r.error));
  };
  useEffect(() => {
    void load('');
  }, []);
  const action = async (method: 'POST' | 'PUT' | 'DELETE', device: Device) => {
    if (method !== 'POST' && !device.enrollment) return;
    setBusy(true);
    setMessage('');
    setIssued(null);
    const path = '/api/admin/satellites' + (method === 'POST' ? '' : '/' + device.enrollment!.id);
    const body =
      method === 'POST'
        ? { satellite_id: device.satellite_id, connection_id: selected }
        : method === 'PUT'
          ? {
              connection_id: selected,
              scope: scopes[device.satellite_id] || device.enrollment!.scope,
            }
          : undefined;
    const r = await apiRequest<unknown>(path, { method, ...(body ? { body } : {}) });
    if (r.ok) {
      if (method === 'POST') setIssued(parseEnrollment(r.data));
      await load();
      setMessage(
        method === 'POST'
          ? 'Credential issued. Finish setup in Home Assistant.'
          : method === 'PUT'
            ? 'Device permissions saved.'
            : 'Device access revoked.'
      );
    } else setMessage(explain(r.error));
    setBusy(false);
  };
  const devices = status?.devices || [];
  const missing =
    status?.enrollments.filter((e) => !devices.some((d) => d.satellite_id === e.satellite_id)) ||
    [];
  return (
    <main className="mx-auto max-w-3xl space-y-6 p-6">
      <h1 className="text-2xl font-semibold">FRIDAY voice satellites</h1>
      <p>
        Manage your Home Assistant voice devices. Each speaker has its own credential and home
        permissions. Personal email, calendar, memory, scheduling and administrator tools are
        unavailable.
      </p>
      <section className="space-y-3 rounded border p-4">
        <label className="block" htmlFor="ha-connection">
          Home Assistant connection
        </label>
        <select
          id="ha-connection"
          className="w-full rounded border p-2"
          value={selected}
          disabled={busy}
          onChange={(e) => {
            setSelected(e.target.value);
            setIssued(null);
            void load(e.target.value);
          }}
        >
          <option value="">Select a connected HA account</option>
          {status?.connections.map((c) => (
            <option key={c.id} value={c.id}>
              {c.label}
              {c.ha_admin ? ' — HA administrator' : ''}
            </option>
          ))}
        </select>
        <p className="text-sm">
          Home requests use this account’s real Home Assistant identity. FRIDAY limits each speaker
          to the permissions below; these limits do not reduce the HA account’s own privileges. No
          default service credential is used.
        </p>
        {status?.provider_identity && (
          <p className="text-sm">
            Verified HA identity:{' '}
            {status.provider_identity.is_owner
              ? 'owner'
              : status.provider_identity.is_admin
                ? 'administrator'
                : 'member'}
            . HA records this account for light commands.
          </p>
        )}
        {status?.connections.length === 0 && (
          <p>Connect Home Assistant from your FRIDAY account settings, then return here.</p>
        )}
        <button
          className="rounded border px-4 py-2"
          disabled={busy || !selected}
          onClick={() => void load()}
        >
          Refresh voice devices
        </button>
      </section>
      {selected && status && (
        <p>
          {devices.length} registry-verified voice device{devices.length === 1 ? '' : 's'}.{' '}
          {devices.filter((d) => d.enrollment).length} enrolled. Enrollment does not select an HA
          pipeline.
        </p>
      )}
      {devices.map((device) => (
        <section key={device.satellite_id} className="space-y-3 rounded border p-4">
          <h2 className="font-semibold">{device.name}</h2>
          <p className="text-sm break-all">{device.satellite_id}</p>
          <p>{device.enrollment ? 'Credential enrolled' : 'Not enrolled'}</p>
          <button
            className="rounded border px-4 py-2"
            disabled={busy || !selected}
            onClick={() => void action('POST', device)}
          >
            {device.enrollment ? 'Replace this device’s credential' : 'Enroll device'}
          </button>
          {device.enrollment && (
            <>
              <label className="block">
                Home permissions
                <select
                  className="ml-2 rounded border p-2"
                  value={scopes[device.satellite_id] || device.enrollment.scope}
                  disabled={busy}
                  onChange={(e) =>
                    setScopes({ ...scopes, [device.satellite_id]: e.target.value as Scope })
                  }
                >
                  <option value="conversation">Conversation only</option>
                  <option value="states">Device state reads</option>
                  <option value="states_and_lights">Device states and light on/off</option>
                </select>
              </label>
              <p className="text-sm">
                Reads cover lights, switches, fans, climate and environmental sensors. Locks,
                alarms, garage controls and generic services are unavailable. Install FRIDAY Voice
                Satellites 0.2.0 in HA before enabling home controls.
              </p>
              <button
                className="rounded border px-4 py-2"
                disabled={busy || !selected}
                onClick={() => void action('PUT', device)}
              >
                Save permissions using selected account
              </button>
              <button
                className="ml-2 rounded border px-4 py-2"
                disabled={busy}
                onClick={() => void action('DELETE', device)}
              >
                Revoke this device
              </button>
            </>
          )}
        </section>
      ))}
      {missing.map((e) => (
        <section key={e.id} className="space-y-2 rounded border p-4">
          <p className="break-all">Existing enrollment: {e.satellite_id}</p>
          <p>Select its HA connection to verify the device and configure permissions.</p>
          <button
            className="rounded border px-4 py-2"
            disabled={busy}
            onClick={() =>
              void action('DELETE', {
                satellite_id: e.satellite_id,
                device_id: e.device_id,
                name: e.satellite_id,
                enrollment: e,
              })
            }
          >
            Revoke this device
          </button>
        </section>
      ))}
      {issued && (
        <section className="space-y-3 rounded border p-4">
          <h2 className="font-semibold">Finish setup in Home Assistant</h2>
          <p className="break-all">
            Credential for {issued.satellite_id}. Copy it into the FRIDAY Voice Satellites
            integration with your private FRIDAY backend address. It is shown once and is not saved
            in this browser.
          </p>
          <input
            aria-label="Satellite credential"
            type="password"
            readOnly
            value={issued.credential}
            className="w-full rounded border p-2"
          />
          <button
            className="rounded border px-4 py-2"
            onClick={() => void navigator.clipboard.writeText(issued.credential)}
          >
            Copy credential
          </button>
          <button className="ml-2 rounded border px-4 py-2" onClick={() => setIssued(null)}>
            Dismiss credential
          </button>
          <p>
            After the integration loads, select its conversation and voice entities in a dedicated
            HA assistant with local intent preference disabled, then select that assistant only as
            this device’s primary assistant. Keep the secondary assistant and wake words.
          </p>
        </section>
      )}
      <p className="text-sm">
        Audio uses Home Assistant’s native random bearer URLs with provider file caching disabled.
      </p>
      {message && <p role="status">{message}</p>}
      <Link href="/">Return to FRIDAY</Link>
    </main>
  );
}
