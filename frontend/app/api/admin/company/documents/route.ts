import type { NextRequest } from 'next/server';
import { documentsRoute } from '@/lib/company/route';

export const dynamic = 'force-dynamic';

export const GET = (req: NextRequest) => documentsRoute(req, 'GET');
export const POST = (req: NextRequest) => documentsRoute(req, 'POST');
