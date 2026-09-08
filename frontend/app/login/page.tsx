import { headers } from 'next/headers';
import { redirect } from 'next/navigation';
import { LoginForm } from '@/components/auth/login-form';
import { authenticate } from '@/lib/auth/session';
import { safeNextPath } from '@/lib/auth/session-cookie';

export const dynamic = 'force-dynamic';

interface PageProps {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}

/**
 * The sign-in page.
 *
 * Deliberately outside the `(app)` group so it renders without the account
 * menu and without any of the session-bound chrome. An already-signed-in
 * visitor is sent on rather than shown the form again.
 *
 * The `?next=` parameter is reduced to a safe same-origin path before it is
 * ever rendered or acted on, so this page cannot be used as an open redirect.
 */
export default async function LoginPage({ searchParams }: PageProps) {
  const params = await searchParams;
  const raw = params.next;
  const next = safeNextPath(Array.isArray(raw) ? raw[0] : raw);

  const auth = await authenticate(await headers());

  if (auth.kind === 'unconfigured') {
    return (
      <main className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-6">
        <h1 className="text-xl font-semibold">Sign-in is not available</h1>
        <p className="text-muted-foreground mt-2 text-sm">
          This deployment runs in single-user mode. See docs/STANDALONE-AUTH.md for the settings
          that enable accounts.
        </p>
      </main>
    );
  }

  if (auth.kind === 'user') {
    redirect(auth.mustChangePassword ? '/change-password' : next);
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-6 py-20">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold">Sign in to JARVIS</h1>
        <p className="text-muted-foreground mt-1 text-sm">
          Use the email and password your administrator gave you.
        </p>
      </header>

      {auth.config.passwordLogin ? (
        <LoginForm next={next} />
      ) : (
        <p className="text-muted-foreground text-sm">
          Password sign-in is disabled on this deployment; sign in through your identity provider.
        </p>
      )}
    </main>
  );
}
