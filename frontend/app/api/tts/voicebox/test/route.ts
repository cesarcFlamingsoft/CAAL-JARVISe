import type { NextRequest } from 'next/server';
import { voiceboxRoute } from '@/lib/tts/admin-route';

export const dynamic = 'force-dynamic';
export const POST = (req: NextRequest) => voiceboxRoute(req, 'POST');
