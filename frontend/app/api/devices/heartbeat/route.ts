/**
 * Keep this browser's device session alive.
 *
 * The bearer token comes from the HttpOnly cookie, never from the request body,
 * so a page script can drive the heartbeat without ever seeing the credential.
 * A session the backend has stopped honouring is cleared here rather than left
 * to expire, so the client is told to enroll again immediately.
 */
import type { NextRequest } from 'next/server';
import { clampSessionLifetime, validateDeviceLabel } from '@/lib/device-session';
import {
  DEVICE_ERROR,
  backendString,
  callDeviceApi,
  clearSessionCookie,
  deviceError,
  noStoreJson,
  readSessionToken,
  setSessionCookie,
} from '../backend';

export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  const token = readSessionToken(req);
  if (!token) {
    return deviceError(DEVICE_ERROR.noSession, 401);
  }

  const result = await callDeviceApi('/devices/heartbeat', {
    method: 'POST',
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

  const status = backendString(result.data, 'status');
  const transport = backendString(result.data, 'transport');
  const label = validateDeviceLabel(backendString(result.data, 'label'));
  if (!status || !transport || !label.ok) {
    console.error('[devices] heartbeat returned an unusable payload');
    return deviceError(DEVICE_ERROR.unavailable, 503);
  }

  const expiresIn = clampSessionLifetime((result.data as { expires_in?: unknown }).expires_in);
  const res = noStoreJson({ status, label: label.label, transport, expires_in: expiresIn });
  // Roll the cookie forward on every successful beat, matching the lifetime the
  // backend just quoted.
  setSessionCookie(res, token, expiresIn, req);
  return res;
}
