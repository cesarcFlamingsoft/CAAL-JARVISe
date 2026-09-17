import type { NextRequest } from 'next/server';
import { statusRoute } from '@/lib/company/route';

export const dynamic = 'force-dynamic';

export const GET = (req: NextRequest) => statusRoute(req);
