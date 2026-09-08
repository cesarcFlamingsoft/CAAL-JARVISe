import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { isSessionRoomName, newSessionRoomName, participantIdentityFor } from './session-room.ts';

describe('identified session rooms', () => {
  it('mints unguessable room names that carry no user information', () => {
    const a = newSessionRoomName();
    const b = newSessionRoomName();
    assert.match(a, /^caal-web-[0-9a-f]{32}$/);
    assert.notEqual(a, b);
    assert.equal(isSessionRoomName(a), true);
    assert.equal(isSessionRoomName('voice_assistant_room_x'), false);
  });

  it('derives an opaque participant identity that never embeds the user id', () => {
    const identity = participantIdentityFor('usr_0123456789abcdef01234567');
    assert.match(identity, /^user-[0-9a-f]{24}$/);
    assert.ok(!identity.includes('0123456789abcdef01234567'));
    assert.notEqual(identity, participantIdentityFor('usr_0123456789abcdef01234567'));
  });
});
