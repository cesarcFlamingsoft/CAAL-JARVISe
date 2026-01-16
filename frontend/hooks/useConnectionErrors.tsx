'use client';

import { useEffect } from 'react';
import { RoomEvent } from 'livekit-client';
import { useRoomContext } from '@livekit/components-react';
import { toastAlert } from '@/components/livekit/alert-toast';

/**
 * Hook to display MCP connection errors from the agent.
 * Listens for data packets with topic "mcp_error" and shows toast alerts.
 */
export function useConnectionErrors() {
  const room = useRoomContext();

  useEffect(() => {
    if (!room) return;

    const handleDataReceived = (
      payload: Uint8Array,
      participant: unknown,
      kind: unknown,
      topic?: string
    ) => {
      // Only handle mcp_error messages
      if (topic !== 'mcp_error') return;

      try {
        const decoder = new TextDecoder();
        const data = JSON.parse(decoder.decode(payload));

        if (data.type === 'mcp_error' && Array.isArray(data.errors)) {
          // Show toast for each error
          toastAlert({
            title: 'Connection Error',
            description: (
              <ul className="list-inside list-disc">
                {data.errors.map((error: string, i: number) => (
                  <li key={i}>{error}</li>
                ))}
              </ul>
            ),
          });
        }
      } catch (error) {
        console.error('[useConnectionErrors] Failed to parse error:', error);
      }
    };

    room.on(RoomEvent.DataReceived, handleDataReceived);

    return () => {
      room.off(RoomEvent.DataReceived, handleDataReceived);
    };
  }, [room]);
}
