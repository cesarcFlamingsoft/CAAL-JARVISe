'use client';

/**
 * Browser-side device presence: enroll this tab with the CAAL device registry,
 * keep the session warm, and show which devices are currently attached.
 *
 * No credential passes through this file. Enrollment is done by the same-origin
 * routes under `app/api/devices/*`, which hold the enrollment secret server-side
 * and keep the returned bearer token in an HttpOnly cookie. This component only
 * ever sees its own device id, its friendly name, and the sanitized device list,
 * and it never logs a response body.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSessionContext } from '@livekit/components-react';
import { Button } from '@/components/livekit/button';
import {
  type ActiveDevice,
  DEVICE_ID_STORAGE_KEY,
  type KeyValueStore,
  MAX_DEVICE_LABEL_LENGTH,
  defaultDeviceLabel,
  ensureDeviceId,
  formatLastSeen,
  normalizeActiveDevices,
  readStoredLabel,
  storeDeviceLabel,
  validateDeviceLabel,
} from '@/lib/device-session';
import { cn } from '@/lib/utils';

/** How often we tell the backend this device is still here. */
const HEARTBEAT_INTERVAL_MS = 60_000;

/**
 * How many times we will enroll before giving up and waiting for the user.
 * A session that keeps coming back unauthorized would otherwise re-enroll
 * forever; the budget is only refilled by a heartbeat that actually succeeds.
 */
const MAX_ENROLLMENTS = 3;

type CallOutcome = 'ok' | 'unauthorized' | 'error';

type PresenceStatus = 'idle' | 'registering' | 'active' | 'offline' | 'unavailable';

const STATUS_TEXT: Record<PresenceStatus, string> = {
  idle: 'Waiting for a session',
  registering: 'Registering this device…',
  active: 'Registered',
  offline: 'Device service is unreachable',
  unavailable: 'Device session could not be kept',
};

/** Used when `localStorage` is unavailable, so the panel still works for this tab. */
function createMemoryStore(): KeyValueStore {
  const data = new Map<string, string>();
  return {
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => void data.set(key, value),
  };
}

/** `localStorage` when the browser allows it, otherwise a per-tab fallback. */
function openDeviceStore(): KeyValueStore {
  try {
    const store = window.localStorage;
    store.getItem(DEVICE_ID_STORAGE_KEY);
    return store;
  } catch {
    return createMemoryStore();
  }
}

/**
 * Call one of our own device routes. Failures are reduced to an outcome so the
 * caller cannot accidentally surface a backend message, and 401 is separated out
 * because it is the one failure that means "enroll again".
 */
async function callDeviceRoute(path: string, init?: RequestInit): Promise<
  { outcome: 'ok'; data: unknown } | { outcome: 'unauthorized' | 'error' }
> {
  try {
    const response = await fetch(path, {
      cache: 'no-store',
      credentials: 'same-origin',
      ...init,
    });
    if (!response.ok) {
      return { outcome: response.status === 401 ? 'unauthorized' : 'error' };
    }
    return { outcome: 'ok', data: await response.json() };
  } catch {
    return { outcome: 'error' };
  }
}

export function DevicePresence() {
  const session = useSessionContext();
  // The registry keys a device to the room it is attached to, so enrollment can
  // only happen once LiveKit has actually given us a room.
  const roomName = session.isConnected ? (session.room.name ?? '') : '';

  const storeRef = useRef<KeyValueStore | null>(null);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const [label, setLabel] = useState('');
  const [devices, setDevices] = useState<ActiveDevice[]>([]);
  const [status, setStatus] = useState<PresenceStatus>('idle');
  const [retryCount, setRetryCount] = useState(0);

  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState<string | null>(null);
  const [labelError, setLabelError] = useState<string | null>(null);

  // Identity is read after mount: it lives in the browser, so touching it while
  // rendering would disagree with the server-rendered markup.
  useEffect(() => {
    const store = openDeviceStore();
    storeRef.current = store;
    setDeviceId(ensureDeviceId(store));
    setLabel(readStoredLabel(store) ?? defaultDeviceLabel(window.navigator.userAgent));
  }, []);

  useEffect(() => {
    if (!deviceId || !roomName || !label) {
      setDevices((current) => (current.length === 0 ? current : []));
      setStatus('idle');
      return;
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let registered = false;
    let enrollments = 0;

    const schedule = () => {
      timer = setTimeout(() => void step(), HEARTBEAT_INTERVAL_MS);
    };

    const enroll = async (): Promise<CallOutcome> =>
      (
        await callDeviceRoute('/api/devices/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            device_id: deviceId,
            label,
            room_name: roomName,
            transport: 'web',
          }),
        })
      ).outcome;

    const heartbeat = async (): Promise<CallOutcome> =>
      (await callDeviceRoute('/api/devices/heartbeat', { method: 'POST' })).outcome;

    const refreshActive = async (): Promise<CallOutcome> => {
      const result = await callDeviceRoute('/api/devices/active');
      if (result.outcome === 'ok' && !cancelled) {
        setDevices(normalizeActiveDevices(result.data));
      }
      return result.outcome;
    };

    /** Forget the session locally; the route already expired the cookie itself. */
    const dropSession = () => {
      registered = false;
      setDevices([]);
      setStatus('registering');
    };

    const step = async () => {
      if (cancelled) return;

      if (!registered) {
        if (enrollments >= MAX_ENROLLMENTS) {
          setStatus('unavailable');
          return;
        }
        enrollments += 1;
        setStatus('registering');

        const enrolled = await enroll();
        if (cancelled) return;

        if (enrolled !== 'ok') {
          // The backend is down or refused us; retry on the normal cadence,
          // still inside the enrollment budget.
          setStatus('offline');
          schedule();
          return;
        }

        registered = true;
        setStatus('active');
        await refreshActive();
        if (cancelled) return;
        schedule();
        return;
      }

      const beat = await heartbeat();
      if (cancelled) return;

      if (beat === 'unauthorized') {
        dropSession();
        void step();
        return;
      }

      if (beat !== 'ok') {
        setStatus('offline');
        schedule();
        return;
      }

      // A session the backend still honours; allow a fresh enrollment budget if
      // it is ever dropped later.
      enrollments = 0;
      setStatus('active');

      if ((await refreshActive()) === 'unauthorized' && !cancelled) {
        dropSession();
        void step();
        return;
      }
      if (cancelled) return;
      schedule();
    };

    void step();

    return () => {
      cancelled = true;
      if (timer) {
        clearTimeout(timer);
      }
    };
  }, [deviceId, roomName, label, retryCount]);

  const saveLabel = useCallback(() => {
    const result = validateDeviceLabel(draft);
    if (!result.ok) {
      setLabelError(result.error);
      return;
    }

    const store = storeRef.current;
    if (store) {
      try {
        storeDeviceLabel(store, result.label);
      } catch {
        // Storage refused the write (private mode, quota). The name still
        // applies to this session; it just will not survive a reload.
      }
    }

    setLabelError(null);
    setDraft(null);
    // Changing the name restarts enrollment, so the registry shows the new one.
    setLabel(result.label);
  }, [draft]);

  const now = Date.now() / 1000;
  const statusText = STATUS_TEXT[status];
  const canRetry = status === 'unavailable' || status === 'offline';
  const otherDevices = useMemo(() => devices.filter((device) => !device.isSelf).length, [devices]);

  if (!deviceId) {
    return null;
  }

  return (
    <div className="fixed top-4 left-4 z-40 w-64 text-left">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className={cn(
          'bg-background/80 border-input text-muted-foreground hover:text-foreground',
          'flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs backdrop-blur',
          'cursor-pointer transition-colors'
        )}
      >
        <span
          aria-hidden
          className={cn(
            'size-1.5 rounded-full',
            status === 'active' ? 'bg-primary' : 'bg-muted-foreground/50'
          )}
        />
        Devices
        {devices.length > 0 && <span className="tabular-nums">{devices.length}</span>}
      </button>

      {open && (
        <div className="bg-background/95 border-input mt-2 space-y-3 rounded-lg border p-3 text-sm backdrop-blur">
          <div className="space-y-2">
            <p className="text-muted-foreground text-xs uppercase">This device</p>
            {draft === null ? (
              <div className="flex items-center justify-between gap-2">
                <span className="truncate font-medium">{label}</span>
                <Button variant="ghost" size="sm" onClick={() => setDraft(label)}>
                  Rename
                </Button>
              </div>
            ) : (
              <div className="space-y-2">
                <input
                  type="text"
                  value={draft}
                  autoFocus
                  maxLength={MAX_DEVICE_LABEL_LENGTH}
                  aria-label="Device name"
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') saveLabel();
                    if (event.key === 'Escape') {
                      setDraft(null);
                      setLabelError(null);
                    }
                  }}
                  className="border-input bg-background w-full rounded-lg border px-3 py-2 text-sm"
                />
                {labelError && <p className="text-destructive text-xs">{labelError}</p>}
                <div className="flex gap-2">
                  <Button variant="primary" size="sm" onClick={saveLabel}>
                    Save
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setDraft(null);
                      setLabelError(null);
                    }}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between gap-2">
            <p className="text-muted-foreground text-xs">{statusText}</p>
            {canRetry && (
              <Button variant="ghost" size="sm" onClick={() => setRetryCount((n) => n + 1)}>
                Retry
              </Button>
            )}
          </div>

          {devices.length > 0 && (
            <ul className="space-y-2">
              {devices.map((device) => (
                <li
                  key={`${device.transport}:${device.label}:${device.lastSeen}`}
                  className="flex items-baseline justify-between gap-2"
                >
                  <span className="min-w-0">
                    <span className="block truncate">{device.label}</span>
                    <span className="text-muted-foreground text-xs">
                      {device.transport}
                      {device.isSelf && ' · this device'}
                    </span>
                  </span>
                  <span className="text-muted-foreground shrink-0 text-xs">
                    {formatLastSeen(device.lastSeen, now)}
                  </span>
                </li>
              ))}
            </ul>
          )}

          {status === 'active' && otherDevices === 0 && (
            <p className="text-muted-foreground text-xs">No other devices are attached.</p>
          )}
        </div>
      )}
    </div>
  );
}
