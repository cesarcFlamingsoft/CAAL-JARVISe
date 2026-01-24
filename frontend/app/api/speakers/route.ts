import { NextRequest, NextResponse } from 'next/server';

const WEBHOOK_URL = process.env.WEBHOOK_URL || 'http://agent:8889';

// Note: App Router handles body parsing automatically with larger limits
// For very large payloads, configure in next.config.js: experimental.serverActions.bodySizeLimit

/**
 * GET /api/speakers - List all enrolled speakers
 */
export async function GET() {
  try {
    const res = await fetch(`${WEBHOOK_URL}/speakers`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!res.ok) {
      const text = await res.text();
      console.error('[/api/speakers] Backend error:', res.status, text);
      return NextResponse.json({ error: text || 'Backend error' }, { status: res.status });
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('[/api/speakers] Error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

/**
 * POST /api/speakers - Enroll a new speaker
 * Body: { name: string, audio_data: string[] }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Use AbortController for timeout (60s for processing multiple audio samples)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000);

    const res = await fetch(`${WEBHOOK_URL}/speakers`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      const text = await res.text();
      console.error('[/api/speakers POST] Backend error:', res.status, text);
      return NextResponse.json({ error: text || 'Backend error' }, { status: res.status });
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('[/api/speakers POST] Error:', error);

    // Check for abort/timeout
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        { error: 'Request timed out - try with fewer/shorter samples' },
        { status: 504 }
      );
    }

    // Check for fetch error (connection refused, etc.)
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        { error: 'Failed to connect to agent server' },
        { status: 503 }
      );
    }

    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}
