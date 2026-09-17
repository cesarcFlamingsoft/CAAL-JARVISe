import Link from 'next/link';

export default async function HAResult({
  searchParams,
}: {
  searchParams: Promise<{ status?: string }>;
}) {
  const { status } = await searchParams;
  return (
    <main className="friday-page mx-auto max-w-lg space-y-4 p-8">
      <h1 className="text-xl font-semibold">Home Assistant connection</h1>
      <p className="friday-panel border-primary/20 border p-5 text-sm leading-6">
        {status === 'connected'
          ? 'Your authenticated HA connection was saved. An administrator must grant or select it before FRIDAY can use it.'
          : 'The connection could not be completed. Start again from your account at the configured public FRIDAY URL, and finish in the same signed-in browser session.'}
      </p>
      <Link
        href="/"
        className="border-primary/30 bg-primary/10 text-primary hover:bg-primary/15 inline-flex rounded-lg border px-4 py-2 text-sm"
      >
        Return to FRIDAY
      </Link>
    </main>
  );
}
