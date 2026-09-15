import type { NextRequest } from 'next/server';
import { satelliteRoute } from '@/lib/satellite/route';

export const dynamic = 'force-dynamic';
export async function DELETE(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  return satelliteRoute(req, 'DELETE', (await params).id);
}

export async function PUT(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  return satelliteRoute(req, 'PUT', (await params).id);
}
