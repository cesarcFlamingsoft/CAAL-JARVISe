import type { NextRequest } from 'next/server';
import { accessRoute } from '@/lib/home-assistant/route';

export const dynamic = 'force-dynamic';
type Params = { params: Promise<{ id: string }> };
export async function GET(req: NextRequest, { params }: Params) {
  return accessRoute(req, 'GET', (await params).id);
}
export async function PUT(req: NextRequest, { params }: Params) {
  return accessRoute(req, 'PUT', (await params).id);
}
