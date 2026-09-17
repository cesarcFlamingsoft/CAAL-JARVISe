import type { NextRequest } from 'next/server';
import { beginPasskeyLogin, clientKeyFor } from '@/lib/auth/backend';
import { readIdentityConfig } from '@/lib/auth/config';
import { apiError, guardReadPost, loginLimited, noStoreJson } from '@/lib/auth/guard';

export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  const limited = loginLimited(req);
  if (limited) return limited;
  const status = readIdentityConfig();
  if (
    status.status !== 'enabled' ||
    !status.config.passwordLogin ||
    !status.config.publicOrigin?.startsWith('https://')
  ) {
    return apiError(404, 'not_found');
  }
  const blocked = guardReadPost(req, status.config);
  if (blocked) return blocked;
  const result = await beginPasskeyLogin(status.config, await clientKeyFor(req.headers));
  if (!result.ok)
    return apiError(
      result.status === 429 ? 429 : 502,
      result.status === 429 ? 'rate_limited' : 'backend_unavailable'
    );
  return noStoreJson(result.data);
}
