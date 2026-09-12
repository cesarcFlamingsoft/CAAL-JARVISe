'use client';

/**
 * Reloads the scheduled dashboard feed the moment JARVIS changes it.
 *
 * The agent publishes one constant on this browser own LiveKit room when a
 * reminder or an alarm of this person is created or changed. This hook listens
 * for exactly that, only while its own room is connected, validates it against
 * the contract in lib/dashboard/scheduled-events, and turns it into a
 * browser-local event. It reads nothing out of the packet -- there is nothing
 * in one to read -- and it fetches nothing: the feed hook re-reads the
 * authenticated route, which is what decides what this person may see.
 *
 * Everything else stays as it was. A browser with no call running, or one that
 * misses a packet, still has the visibility, focus and interval refreshes.
 */
import { useEffect } from 'react';
import { ConnectionState, RoomEvent } from 'livekit-client';
import { useMaybeRoomContext } from '@livekit/components-react';
import { SCHEDULED_EVENT_NAME, isScheduledChange } from '@/lib/dashboard/scheduled-events';

export function useScheduledUpdates(): void {
  const room = useMaybeRoomContext();

  useEffect(() => {
    if (!room) return;
    const onData = (
      payload: Uint8Array,
      _participant?: unknown,
      _kind?: unknown,
      topic?: string
    ) => {
      if (room.state !== ConnectionState.Connected) return;
      if (!isScheduledChange(topic, payload)) return;
      window.dispatchEvent(new Event(SCHEDULED_EVENT_NAME));
    };
    room.on(RoomEvent.DataReceived, onData);
    return () => {
      room.off(RoomEvent.DataReceived, onData);
    };
  }, [room]);
}
