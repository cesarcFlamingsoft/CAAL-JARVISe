import Link from 'next/link';

export default async function HAResult({
  searchParams,
}: {
  searchParams: Promise<{ status?: string }>;
}) {
  const { status } = await searchParams;
  return (
    <main className="mx-auto max-w-lg space-y-4 p-8">
      <h1 className="text-xl font-semibold">Home Assistant connection</h1>
      <p>
        {status === 'connected'
          ? 'Your authenticated HA connection was saved. An administrator must grant or select it before FRIDAY can use it.'
          : 'The connection could not be completed. Start again from your account at the configured public FRIDAY URL, and finish in the same signed-in browser session.'}
      </p>
      <Link href="/">Return to FRIDAY</Link>
    </main>
  );
}
