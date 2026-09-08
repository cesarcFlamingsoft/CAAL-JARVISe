/**
 * List the devices currently attached to this user's assistant.
 *
 * Authenticated by the same HttpOnly cookie as the heartbeat. Entries are
 * validated here before they reach the browser, so a malformed or unknown
 * device from the backend is dropped instead of rendered.
 */
import type { NextRequest } from 'next/server';
import { normalizeActiveDevices } from '@/lib/device-session';
import {
  DEVICE_ERROR,
  callDeviceApi,
  clearSessionCookie,
  deviceError,
  noStoreJson,
  readSessionToken,
} from '../backend';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const token = readSessionToken(req);
  if (!token) {
    return deviceError(DEVICE_ERROR.noSession, 401);
  }

  const result = await callDeviceApi('/devices/active', {
    method: 'GET',
    headers: { Authorization: `Bearer ${token}` },
  });

  if (result.status === 401) {
    const expired = deviceError(DEVICE_ERROR.noSession, 401);
    clearSessionCookie(expired, req);
    return expired;
  }
  if (!result.ok) {
    return deviceError(DEVICE_ERROR.unavailable, 503);
  }

  // Re-emit the backend's wire shape so the client keeps using one parser, but
  // only for entries that survived validation.
  const devices = normalizeActiveDevices(result.data).map((device) => ({
    label: device.label,
    transport: device.transport,
    last_seen: device.lastSeen,
    is_self: device.isSelf,
  }));

  return noStoreJson({ devices });
}
