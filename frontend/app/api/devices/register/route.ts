/**
 * Enroll this browser with the CAAL device registry.
 *
 * The enrollment secret is added here, server-side, so the browser never holds
 * it. The session token that comes back is written to an HttpOnly cookie and
 * omitted from the response body — the client only learns its own identity.
 */
import type { NextRequest } from 'next/server';
import { backendPrincipalFor } from '@/lib/auth/backend';
import { authenticate } from '@/lib/auth/session';
import { clampSessionLifetime, isDeviceId, validateDeviceLabel } from '@/lib/device-session';
import {
  DEVICE_ERROR,
  backendString,
  callDeviceApi,
  deviceError,
  enrollmentToken,
  noStoreJson,
  readJsonBody,
  setSessionCookie,
  validateRoomName,
} from '../backend';

export const dynamic = 'force-dynamic';

/** This proxy speaks for a browser, so it only ever enrolls a web device. */
const TRANSPORT = 'web';

export async function POST(req: NextRequest) {
  const body = await readJsonBody(req);
  if (!body) {
    return deviceError(DEVICE_ERROR.invalidRequest, 400);
  }

  const label = validateDeviceLabel(body.label);
  const roomName = validateRoomName(body.room_name);
  if (
    !isDeviceId(body.device_id) ||
    !label.ok ||
    !roomName ||
    (body.transport !== undefined && body.transport !== TRANSPORT)
  ) {
    return deviceError(DEVICE_ERROR.invalidRequest, 400);
  }

  const secret = enrollmentToken();
  if (!secret) {
    return deviceError(DEVICE_ERROR.unavailable, 503);
  }

  // Bind the device to the verified user, so it only ever sees that user's
  // other devices. An Access-verified caller without an account is refused; a
  // request with no assertion (LAN) stays an anonymous legacy device unless
  // the operator requires identity for sessions.
  const auth = await authenticate(req.headers);
  const identityHeaders: Record<string, string> = {};
  if (auth.kind === 'user') {
    identityHeaders['X-CAAL-Principal'] = await backendPrincipalFor(
      auth.config,
      auth.user.userId
    );
  } else if (auth.kind === 'denied' || auth.kind === 'invalid') {
    return deviceError(DEVICE_ERROR.noSession, 401);
  } else if (auth.kind === 'anonymous' && auth.config.requireIdentityForSessions) {
    return deviceError(DEVICE_ERROR.noSession, 401);
  }

  const result = await callDeviceApi('/devices/register', {
    method: 'POST',
    headers: { 'X-CAAL-Device-Token': secret, ...identityHeaders },
    body: {
      device_id: body.device_id,
      room_name: roomName,
      label: label.label,
      transport: TRANSPORT,
    },
  });

  if (!result.ok) {
    // A 400 means the backend rejected the same fields we just checked; every
    // other failure — including a refused enrollment secret — reads as "down",
    // so a caller cannot probe our configuration.
    return deviceError(
      result.status === 400 ? DEVICE_ERROR.invalidRequest : DEVICE_ERROR.unavailable,
      result.status === 400 ? 400 : 503
    );
  }

  const sessionToken = backendString(result.data, 'session_token');
  const deviceId = backendString(result.data, 'device_id');
  const transport = backendString(result.data, 'transport');
  const registeredLabel = validateDeviceLabel(backendString(result.data, 'label'));
  if (!sessionToken || !deviceId || !transport || !registeredLabel.ok) {
    console.error('[devices] register returned an unusable payload');
    return deviceError(DEVICE_ERROR.unavailable, 503);
  }

  // Report the lifetime the cookie actually got, so the client's heartbeat
  // schedule matches when the session really stops working.
  const expiresIn = clampSessionLifetime((result.data as { expires_in?: unknown }).expires_in);
  const res = noStoreJson({
    device_id: deviceId,
    label: registeredLabel.label,
    transport,
    expires_in: expiresIn,
  });
  setSessionCookie(res, sessionToken, expiresIn, req);
  return res;
}
