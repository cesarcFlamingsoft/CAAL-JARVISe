'use client';

import { useEffect, useState } from 'react';
import { RoomEvent } from 'livekit-client';
import { useRoomContext } from '@livekit/components-react';
import { type ToolActivity, appendToolActivity, parseToolStatus } from '@/lib/dashboard/activity';

/**
 * Every tool call the agent reports over the room's `tool_status` topic,
 * newest first and bounded. Survives the end of a call so the workspace can
 * still show what JARVIS did after voice is docked again.
 */
export function useToolActivity(): ToolActivity[] {
  const room = useRoomContext();
  const [feed, setFeed] = useState<ToolActivity[]>([]);

  useEffect(() => {
    if (!room) return;

    const handleDataReceived = (
      payload: Uint8Array,
      _participant: unknown,
      _kind: unknown,
      topic?: string
    ) => {
      if (topic !== 'tool_status') return;
      try {
        const packet = parseToolStatus(JSON.parse(new TextDecoder().decode(payload)));
        if (packet) {
          const now = Date.now();
          setFeed((current) => appendToolActivity(current, packet, now));
        }
      } catch {
        // A packet we could not read is not work we can show.
      }
    };

    room.on(RoomEvent.DataReceived, handleDataReceived);
    return () => {
      room.off(RoomEvent.DataReceived, handleDataReceived);
    };
  }, [room]);

  return feed;
}
