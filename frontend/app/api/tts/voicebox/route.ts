import type { NextRequest } from 'next/server';
import { voiceboxRoute } from '@/lib/tts/admin-route';

export const dynamic = 'force-dynamic';
export const GET = (req: NextRequest) => voiceboxRoute(req, 'GET');
export const PUT = (req: NextRequest) => voiceboxRoute(req, 'PUT');
