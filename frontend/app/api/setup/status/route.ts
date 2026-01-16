import { NextResponse } from 'next/server';

const WEBHOOK_URL = process.env.WEBHOOK_URL || 'http://agent:8889';

export async function GET() {
  try {
    const res = await fetch(`${WEBHOOK_URL}/setup/status`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!res.ok) {
      const text = await res.text();
      console.error('[/api/setup/status] Backend error:', res.status, text);
      return NextResponse.json({ error: text || 'Backend error' }, { status: res.status });
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('[/api/setup/status] Error:', error);
    // If backend is not reachable, assume setup not completed
    return NextResponse.json({ completed: false });
  }
}
