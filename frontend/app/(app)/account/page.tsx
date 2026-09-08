import { headers } from 'next/headers';
import Link from 'next/link';
import { redirect } from 'next/navigation';
import { AccountPanel } from '@/components/account/account-panel';
import { authenticate } from '@/lib/auth/session';

export const dynamic = 'force-dynamic';

const DENIED_TEXT: Record<string, string> = {
  no_account: 'Your identity was verified, but there is no JARVIS account for it yet. Ask an administrator to create one.',
  suspended: 'This account is suspended. Ask an administrator if you think this is a mistake.',
  unavailable: 'The identity service is unavailable right now. Try again shortly.',
};

/** The signed-in user's own account page. */
export default async function AccountPage() {
  const auth = await authenticate(await headers());

  if (auth.kind === 'user' && auth.mustChangePassword) {
    redirect('/change-password');
  }

  let body: React.ReactNode;
  if (auth.kind === 'unconfigured') {
    body = (
      <p className="text-muted-foreground text-sm">
        This deployment runs in single-user mode; there are no per-user accounts to manage.
      </p>
    );
  } else if (auth.kind === 'anonymous') {
    body = (
      <p className="text-muted-foreground text-sm">
        You are not signed in through Cloudflare Access, so this session has no user profile.
        Memory, phone handoff and callbacks are unavailable.
      </p>
    );
  } else if (auth.kind === 'invalid') {
    body = <p className="text-destructive text-sm">Your sign-in could not be verified. Reload the page.</p>;
  } else if (auth.kind === 'denied') {
    body = <p className="text-destructive text-sm">{DENIED_TEXT[auth.reason]}</p>;
  } else {
    body = <AccountPanel />;
  }

  return (
    <main className="mx-auto max-w-2xl px-6 py-20">
      <header className="mb-8 flex items-baseline justify-between">
        <h1 className="text-xl font-semibold">Your account</h1>
        <Link
          href="/"
          className="text-muted-foreground hover:text-foreground text-sm underline-offset-4 hover:underline"
        >
          Back to JARVIS
        </Link>
      </header>
      {body}
    </main>
  );
}
