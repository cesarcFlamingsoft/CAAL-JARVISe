import { NextRequest, NextResponse } from 'next/server';

const WEBHOOK_URL = process.env.WEBHOOK_URL || 'http://agent:8889';

/**
 * DELETE /api/speakers/[name] - Remove an enrolled speaker
 */
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ name: string }> }
) {
  try {
    const { name } = await params;
    const encodedName = encodeURIComponent(name);

    const res = await fetch(`${WEBHOOK_URL}/speakers/${encodedName}`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!res.ok) {
      const text = await res.text();
      console.error('[/api/speakers/[name]] Backend error:', res.status, text);
      return NextResponse.json({ error: text || 'Backend error' }, { status: res.status });
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('[/api/speakers/[name]] Error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}
