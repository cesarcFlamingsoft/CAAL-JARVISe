import type { NextRequest } from 'next/server';
import { satelliteRoute } from '@/lib/satellite/route';

export const dynamic = 'force-dynamic';
export const GET = (req: NextRequest) => satelliteRoute(req, 'GET');
export const POST = (req: NextRequest) => satelliteRoute(req, 'POST');
