import type { NextRequest } from 'next/server';
import { accessRoute } from '@/lib/home-assistant/route';

export const dynamic = 'force-dynamic';
export async function GET(req: NextRequest) {
  return accessRoute(req, 'GET');
}
