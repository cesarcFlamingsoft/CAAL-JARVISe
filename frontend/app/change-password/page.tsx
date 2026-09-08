import { headers } from 'next/headers';
import Link from 'next/link';
import { redirect } from 'next/navigation';
import { ChangePasswordForm } from '@/components/auth/change-password-form';
import { authenticate } from '@/lib/auth/session';

export const dynamic = 'force-dynamic';

/**
 * Change your password, and the only page a forced-change session can reach.
 *
 * Outside the `(app)` group on purpose: a user still holding a one-time
 * password should not be shown the assistant, and the API refuses them
 * everything else anyway, so rendering the app chrome would only produce a
 * page full of failed requests.
 */
export default async function ChangePasswordPage() {
  const auth = await authenticate(await headers());

  if (auth.kind !== 'user') {
    redirect('/login?next=%2Fchange-password');
  }
  if (!auth.config.passwordLogin) {
    redirect('/');
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-6 py-20">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold">
          {auth.mustChangePassword ? 'Choose a new password' : 'Change your password'}
        </h1>
        {auth.mustChangePassword ? (
          <p className="text-muted-foreground mt-1 text-sm">
            You signed in with a one-time password. Choose your own before continuing; nothing else
            is available until you do.
          </p>
        ) : (
          <p className="text-muted-foreground mt-1 text-sm">
            Signed in as {auth.user.displayName}.
          </p>
        )}
      </header>

      <ChangePasswordForm forced={auth.mustChangePassword} />

      {!auth.mustChangePassword && (
        <Link
          href="/account"
          className="text-muted-foreground hover:text-foreground mt-6 text-sm underline-offset-4 hover:underline"
        >
          Back to your account
        </Link>
      )}
    </main>
  );
}
