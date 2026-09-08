import Link from 'next/link';
import { type Provider, describeOutcome, isOutcome, isProvider } from '@/lib/connections/protocol';

export const dynamic = 'force-dynamic';

type SearchParams = Promise<Record<string, string | string[] | undefined>>;

const UNKNOWN = {
  title: 'Nothing to show',
  detail:
    'This page reports the result of connecting an account. Start from Settings → Integrations → Connected accounts.',
  connected: false,
};

/**
 * The end of an account-connection round trip. The callback route sends the
 * browser here with two codes and nothing else; anything that is not a known
 * code renders as "nothing to show".
 */
export default async function ConnectionResultPage({ searchParams }: { searchParams: SearchParams }) {
  const params = await searchParams;
  const outcome = isOutcome(params.outcome) ? params.outcome : null;
  const provider: Provider | null = isProvider(params.provider) ? params.provider : null;
  const view = outcome ? describeOutcome(outcome, provider) : UNKNOWN;

  return (
    <main className="mx-auto max-w-2xl px-6 py-20">
      <header className="mb-8 flex items-baseline justify-between">
        <h1 className="text-xl font-semibold">Connected accounts</h1>
        <Link
          href="/"
          className="text-muted-foreground hover:text-foreground text-sm underline-offset-4 hover:underline"
        >
          Back to JARVIS
        </Link>
      </header>
      <section
        aria-live="polite"
        className={`rounded-xl border p-4 ${
          view.connected ? 'border-green-500/30 bg-green-500/10' : 'border-input'
        }`}
      >
        <h2 className="text-base font-semibold">{view.title}</h2>
        <p className="text-muted-foreground mt-2 text-sm">{view.detail}</p>
      </section>
      <p className="text-muted-foreground mt-6 text-xs">
        Open JARVIS, then Settings → Integrations → Connected accounts to see the current state of
        every provider.
      </p>
    </main>
  );
}
