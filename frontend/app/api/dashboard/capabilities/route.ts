/**
 * What the Monitor dashboard may show for this deployment.
 *
 * Returns only the credential-free capability summary (see
 * `lib/dashboard/capabilities.ts`). Requires a signed-in user once identity is
 * configured; a legacy single-user deployment is served as `local`.
 */
import type { NextRequest } from 'next/server';
import { apiError, noStoreJson } from '@/lib/auth/guard';
import { capabilitiesFromSettings } from '@/lib/dashboard/capabilities';
import { fetchBackendSettings, requireDashboardAccess } from '@/lib/dashboard/server';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const access = await requireDashboardAccess(req);
  if (!access.ok) return access.response;

  const settings = await fetchBackendSettings();
  if (settings === null) return apiError(502, 'backend_unavailable');

  return noStoreJson({ scope: access.scope, capabilities: capabilitiesFromSettings(settings) });
}
