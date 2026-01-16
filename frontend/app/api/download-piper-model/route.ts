import { NextRequest, NextResponse } from 'next/server';

const WEBHOOK_URL = process.env.WEBHOOK_URL || 'http://agent:8889';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { model_id } = body;

    if (!model_id) {
      return NextResponse.json({ success: false, message: 'model_id required' }, { status: 400 });
    }

    const res = await fetch(`${WEBHOOK_URL}/download-piper-model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id }),
    });

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Failed to download Piper model:', error);
    return NextResponse.json(
      { success: false, message: 'Failed to download model' },
      { status: 500 }
    );
  }
}
