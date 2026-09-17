import type { NextRequest } from 'next/server';
import { searchRoute } from '@/lib/company/route';

export const dynamic = 'force-dynamic';

// POST, not GET: the query is the confidential part and must not reach a URL.
// See `searchRoute` in @/lib/company/route.
export const POST = (req: NextRequest) => searchRoute(req);
