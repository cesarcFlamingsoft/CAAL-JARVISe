import type { NextRequest } from 'next/server';
import { documentRoute } from '@/lib/company/route';

export const dynamic = 'force-dynamic';

export const DELETE = async (req: NextRequest, ctx: { params: Promise<{ id: string }> }) =>
  documentRoute(req, (await ctx.params).id);
