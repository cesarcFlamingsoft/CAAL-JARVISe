/**
 * The one packet that says a person own scheduled items changed.
 *
 * A reminder or an alarm set by voice has to appear on the dashboard now, not
 * at the end of the next polling interval. The agent publishes a constant on
 * its own LiveKit data topic -- a version and a kind, and nothing else -- and
 * this is the browser half of that contract.
 *
 * What arrives is treated as hostile until it is exactly the constant: the
 * topic must match, the payload must be small, it must decode as UTF-8, parse
 * as a plain object, and carry those two fields and no others. Anything else
 * is ignored in silence. The packet is a nudge and carries no capability: the
 * dashboard still re-reads its own authenticated feed, which is the surface
 * that decides what that person may see. No title, time, id, owner or
 * destination is ever in it, so there is nothing here to leak even if it were
 * forged.
 *
 * Free of React and LiveKit so it can be unit tested.
 */

/** The LiveKit data topic. Validated exactly; nothing else is listened to. */
export const SCHEDULED_TOPIC = 'scheduled_changed';

/** The browser-local event a validated packet is turned into. */
export const SCHEDULED_EVENT_NAME = 'scheduled-items-updated';

export const SCHEDULED_KIND = 'scheduled_changed';
export const SCHEDULED_VERSION = 1;

/** The constant is 34 bytes. Anything larger is not it. */
const MAX_PAYLOAD_BYTES = 64;
const FATAL_UTF8 = new TextDecoder('utf-8', { fatal: true });

/** Whether a received data packet is exactly the scheduled-change constant. */
export function isScheduledChange(topic: string | undefined, payload: Uint8Array): boolean {
  if (topic !== SCHEDULED_TOPIC) return false;
  if (!(payload instanceof Uint8Array)) return false;
  if (payload.length === 0 || payload.length > MAX_PAYLOAD_BYTES) return false;
  let parsed: unknown;
  try {
    parsed = JSON.parse(FATAL_UTF8.decode(payload));
  } catch {
    return false;
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return false;
  const keys = Object.keys(parsed as Record<string, unknown>);
  if (keys.length !== 2 || !keys.includes('v') || !keys.includes('kind')) return false;
  const event = parsed as { v: unknown; kind: unknown };
  return event.v === SCHEDULED_VERSION && event.kind === SCHEDULED_KIND;
}
