import type { NextRequest } from 'next/server';
import { versionRoute } from '@/lib/company/route';

export const dynamic = 'force-dynamic';

export const PATCH = async (req: NextRequest, ctx: { params: Promise<{ id: string }> }) =>
  versionRoute(req, (await ctx.params).id);
