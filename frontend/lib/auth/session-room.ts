/**
 * Naming for identified LiveKit sessions.
 *
 * An identified user gets a fresh, unguessable room per connection, minted
 * here rather than taken from the browser, so no client can pick another
 * user's room. Neither the room name nor the participant identity carries the
 * user id; the agent learns who the session is for only from the signed
 * principal in the room configuration.
 */

const ROOM = /^caal-web-[0-9a-f]{32}$/;

function randomHex(bytes: number): string {
  const buffer = new Uint8Array(bytes);
  globalThis.crypto.getRandomValues(buffer);
  return Array.from(buffer, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export function newSessionRoomName(): string {
  return `caal-web-${randomHex(16)}`;
}

export function isSessionRoomName(value: unknown): value is string {
  return typeof value === 'string' && ROOM.test(value);
}

/** An opaque per-connection participant identity; unrelated to the user id. */
export function participantIdentityFor(userId: string): string {
  if (typeof userId !== 'string' || !userId) {
    throw new Error('userId is required');
  }
  return `user-${randomHex(12)}`;
}
