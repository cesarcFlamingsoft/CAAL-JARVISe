import type { NextRequest } from 'next/server';
import { peopleRoute } from '@/lib/company/route';

export const dynamic = 'force-dynamic';

export const GET = (req: NextRequest) => peopleRoute(req, 'GET');
export const POST = (req: NextRequest) => peopleRoute(req, 'POST');
