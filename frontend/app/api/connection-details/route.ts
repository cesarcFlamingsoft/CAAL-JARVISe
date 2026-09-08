import { NextResponse } from 'next/server';
import { AccessToken, type AccessTokenOptions, type VideoGrant } from 'livekit-server-sdk';
import { RoomConfiguration } from '@livekit/protocol';
import { securityHeaders } from '@/lib/auth/guard';
import { AUDIENCE_AGENT, mintPrincipal } from '@/lib/auth/principal';
import { authenticate } from '@/lib/auth/session';
import { newSessionRoomName, participantIdentityFor } from '@/lib/auth/session-room';

type ConnectionDetails = {
  serverUrl: string;
  roomName: string;
  participantName: string;
  participantToken: string;
};

// NOTE: you are expected to define the following environment variables in `.env.local`:
const API_KEY = process.env.LIVEKIT_API_KEY;
const API_SECRET = process.env.LIVEKIT_API_SECRET;
// Internal URL for token generation (Docker network or localhost)
const LIVEKIT_URL = process.env.LIVEKIT_URL;
// External URL for browser connection (set to 'auto' for dynamic detection)
const LIVEKIT_PUBLIC_URL = process.env.NEXT_PUBLIC_LIVEKIT_URL;

/** How long the agent has to pick up the signed session principal. */
const AGENT_PRINCIPAL_TTL_SECONDS = 300;

// don't cache the results
export const revalidate = 0;

export async function POST(req: Request) {
  try {
    if (LIVEKIT_URL === undefined) {
      throw new Error('LIVEKIT_URL is not defined');
    }
    if (API_KEY === undefined) {
      throw new Error('LIVEKIT_API_KEY is not defined');
    }
    if (API_SECRET === undefined) {
      throw new Error('LIVEKIT_API_SECRET is not defined');
    }

    // Parse agent configuration from request body
    const body = await req.json();
    const agentName: string =
      body?.agentName ??
      body?.agent_name ??
      body?.room_config?.agents?.[0]?.agent_name ??
      body?.roomConfiguration?.agents?.[0]?.agentName ??
      body?.roomConfiguration?.agents?.[0]?.agent_name;
    const clientId: string =
      body?.client_id ??
      body?.participant_identity ??
      body?.participantIdentity; // Optional client ID for unique rooms

    // Who is connecting, from the verified Cloudflare Access assertion. With
    // multi-user identity configured, an identified user gets a fresh
    // server-named room and a signed principal in the room configuration; the
    // agent learns the user only from that principal. A caller Access verified
    // but who has no account is refused. A request with no assertion at all
    // (LAN access) keeps the legacy shared behaviour unless the operator set
    // CAAL_REQUIRE_IDENTITY_FOR_SESSIONS.
    const auth = await authenticate(req.headers);
    let roomName: string;
    let participantIdentity: string;
    let agentMetadata: string | undefined;
    const participantName = body?.participant_name ?? body?.participantName ?? 'user';

    if (auth.kind === 'user' && auth.mustChangePassword) {
      // A one-time password buys exactly one thing: the change form. It must
      // not also buy a voice session with the user's memories attached.
      return NextResponse.json(
        { error: 'password_change_required' },
        { status: 403, headers: securityHeaders() }
      );
    } else if (auth.kind === 'user') {
      roomName = newSessionRoomName();
      participantIdentity = participantIdentityFor(auth.user.userId);
      const principal = await mintPrincipal({
        secret: auth.config.internalAuthSecret,
        subject: auth.user.userId,
        audience: AUDIENCE_AGENT,
        ttlSeconds: AGENT_PRINCIPAL_TTL_SECONDS,
        claims: { room: roomName },
      });
      agentMetadata = JSON.stringify({ caal_principal: principal });
    } else if (auth.kind === 'denied' || auth.kind === 'invalid') {
      return NextResponse.json(
        { error: auth.kind === 'denied' ? auth.reason : 'invalid_assertion' },
        { status: auth.kind === 'denied' ? 403 : 401, headers: securityHeaders() }
      );
    } else if (
      (auth.kind === 'anonymous' || auth.kind === 'expired') &&
      auth.config.requireIdentityForSessions
    ) {
      return NextResponse.json({ error: 'not_signed_in' }, { status: 401, headers: securityHeaders() });
    } else {
      // Legacy single-user behaviour: client-chosen room, random identity.
      participantIdentity =
        body?.participant_identity ??
        body?.participantIdentity ??
        `voice_assistant_user_${Math.floor(Math.random() * 10_000)}`;
      roomName = clientId ? `voice_assistant_room_${clientId}` : 'voice_assistant_room';
    }

    const participantToken = await createParticipantToken(
      { identity: participantIdentity, name: participantName },
      roomName,
      agentName,
      agentMetadata
    );

    // Determine the WebSocket URL for the client
    // Priority:
    // 1. If HTTPS request and NEXT_PUBLIC_LIVEKIT_URL is set, use it (secure mode)
    // 2. Otherwise, derive ws:// from request hostname (LAN/mobile HTTP access)
    let serverUrl: string;
    const forwardedProto = req.headers.get('x-forwarded-proto');
    const isHttps = forwardedProto === 'https' || req.url.startsWith('https://');

    if (isHttps && LIVEKIT_PUBLIC_URL) {
      // HTTPS request - use configured secure URL (Tailscale/distributed mode)
      serverUrl = LIVEKIT_PUBLIC_URL;
    } else {
      // HTTP request - derive ws:// from request host for LAN/mobile access
      const host = req.headers.get('host') || 'localhost';
      const hostname = host.split(':')[0]; // Remove port if present
      serverUrl = `ws://${hostname}:7880`;
    }

    // Return connection details
    const data: ConnectionDetails = {
      serverUrl,
      roomName,
      participantToken: participantToken,
      participantName,
    };
    return NextResponse.json(data, { headers: securityHeaders() });
  } catch (error) {
    if (error instanceof Error) {
      console.error(error);
      return new NextResponse(error.message, { status: 500 });
    }
  }
}

function createParticipantToken(
  userInfo: AccessTokenOptions,
  roomName: string,
  agentName?: string,
  agentMetadata?: string
): Promise<string> {
  const at = new AccessToken(API_KEY, API_SECRET, {
    ...userInfo,
    ttl: '15m',
  });
  const grant: VideoGrant = {
    room: roomName,
    roomJoin: true,
    canPublish: true,
    canPublishData: true,
    canSubscribe: true,
  };
  at.addGrant(grant);

  // Always set room config with fast departure timeout for quick reconnect
  // departureTimeout: seconds to keep room open after last participant leaves (default 20s)
  // The agent dispatch metadata (when present) carries the signed session
  // principal; it is part of the server-signed token and never visible to
  // participants.
  at.roomConfig = new RoomConfiguration({
    departureTimeout: 1, // Close room 1 second after disconnect for fast reconnect
    ...(agentName && {
      agents: [{ agentName, ...(agentMetadata !== undefined && { metadata: agentMetadata }) }],
    }),
  });

  return at.toJwt();
}
