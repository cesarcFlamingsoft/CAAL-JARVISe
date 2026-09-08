import { headers } from 'next/headers';
import Link from 'next/link';
import { notFound, redirect } from 'next/navigation';
import { AdminPanel } from '@/components/admin/admin-panel';
import { authenticate } from '@/lib/auth/session';

export const dynamic = 'force-dynamic';

/**
 * Administrator panel. The page is gated on a fresh backend resolution of the
 * verified Cloudflare Access identity; non-administrators get a 404 so the
 * route reveals nothing. Every action inside is authorized again by the
 * backend, so the gate here is a courtesy, not the protection.
 */
export default async function AdminPage() {
  const auth = await authenticate(await headers());

  if (auth.kind === 'unconfigured') {
    return (
      <main className="mx-auto max-w-3xl px-6 py-24">
        <h1 className="text-xl font-semibold">Administration is not available</h1>
        <p className="text-muted-foreground mt-2 text-sm">
          Multi-user identity is not configured for this deployment. See
          docs/MULTI-USER-IDENTITY.md for the required settings.
        </p>
      </main>
    );
  }
  if (auth.kind === 'user' && auth.mustChangePassword) {
    // Signed in on a one-time password: the API refuses everything here
    // anyway, so render the change form rather than a panel of failures.
    redirect('/change-password');
  }
  if (auth.kind !== 'user' || auth.user.role !== 'admin') {
    notFound();
  }

  return (
    <main className="mx-auto max-w-6xl px-6 py-20">
      <header className="mb-8 flex items-baseline justify-between">
        <div>
          <h1 className="text-xl font-semibold">JARVIS administration</h1>
          <p className="text-muted-foreground mt-1 text-sm">
            Users, roles, approved callback numbers and the audit trail.
          </p>
        </div>
        <Link
          href="/"
          className="text-muted-foreground hover:text-foreground text-sm underline-offset-4 hover:underline"
        >
          Back to JARVIS
        </Link>
      </header>
      <AdminPanel selfId={auth.user.userId} />
    </main>
  );
}
