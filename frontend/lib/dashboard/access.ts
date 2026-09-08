/**
 * Who may read dashboard data, and under which scope.
 *
 * A legacy single-user deployment has no accounts, so the dashboard is local.
 * Once identity is configured the dashboard is a signed-in surface: anonymous
 * LAN visitors may still talk to JARVIS, but they get no one's calendar or
 * work, and a one-time password buys the change-password form and nothing else.
 *
 * This file stays free of Next.js imports so the rule can be unit tested; the
 * route helper that turns a refusal into a response lives in `server.ts`.
 */
import type { AuthResult } from '../auth/session';

export type DashboardScope =
  | { ok: true; scope: 'local'; userId: null }
  | { ok: true; scope: 'user'; userId: string }
  | { ok: false; status: 401 | 403; code: string };

export function dashboardScope(auth: AuthResult): DashboardScope {
  switch (auth.kind) {
    case 'unconfigured':
      return { ok: true, scope: 'local', userId: null };
    case 'user':
      if (auth.mustChangePassword) {
        return { ok: false, status: 403, code: 'password_change_required' };
      }
      return { ok: true, scope: 'user', userId: auth.user.userId };
    case 'anonymous':
      return { ok: false, status: 401, code: 'not_signed_in' };
    case 'expired':
      return { ok: false, status: 401, code: 'session_expired' };
    case 'invalid':
      return { ok: false, status: 401, code: 'invalid_assertion' };
    case 'denied':
      return { ok: false, status: 403, code: auth.reason };
  }
}
