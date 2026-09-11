/**
 * How future reminders of the signed-in user will be delivered.
 *
 * The minimal edit surface for the reminder slice: GET reads the channels
 * their next reminder will use, PUT chooses them. It names no reminder, so it
 * can never re-arm, cancel or re-send something that is already waiting, and
 * it names no destination: a phone number and a Telegram chat come from the
 * profile an administrator controls. The body is checked against a fixed
 * vocabulary here before it reaches the backend, and a channel this owner is
 * not authorised for is refused there; the refusal is a bounded code.
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardMutation,
  noStoreJson,
  readJsonObject,
  requireUser,
} from '@/lib/auth/guard';
import { browserDeliveryDefaults, deliveryChoiceRequest } from '@/lib/dashboard/reminders';

export const dynamic = 'force-dynamic';

const DELIVERY_TIMEOUT_MS = 10_000;
const PATH = '/users/me/dashboard/reminders/delivery';

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;

  const result = await callAsUser(auth.config, auth.user.userId, PATH, {
    method: 'GET',
    timeoutMs: DELIVERY_TIMEOUT_MS,
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const defaults = browserDeliveryDefaults(result.data);
  return defaults ? noStoreJson(defaults) : apiError(502, 'backend_unavailable');
}

export async function PUT(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const body = await readJsonObject(req);
  const choice = body === null ? null : deliveryChoiceRequest(body.delivery);
  if (choice === null) return apiError(422, 'invalid');

  const result = await callAsUser(auth.config, auth.user.userId, PATH, {
    method: 'PUT',
    timeoutMs: DELIVERY_TIMEOUT_MS,
    body: choice,
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const defaults = browserDeliveryDefaults(result.data);
  return defaults ? noStoreJson(defaults) : apiError(502, 'backend_unavailable');
}
