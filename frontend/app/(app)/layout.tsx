import { headers } from 'next/headers';
import Link from 'next/link';
import { AccountMenu } from '@/components/account/account-menu';
import { getAppConfig } from '@/lib/utils';

interface LayoutProps {
  children: React.ReactNode;
}

const BRAND_NAME = 'MEXCANTECH';
const BRAND_TAGLINE = 'Learn. Build. Empower.';

export default async function Layout({ children }: LayoutProps) {
  const hdrs = await headers();
  const { companyName, logo, logoDark } = await getAppConfig(hdrs);

  return (
    <>
      {/* Brand - top left */}
      <header className="fixed top-0 left-0 z-40 hidden p-6 md:block">
        <Link
          href="/"
          aria-label={`${BRAND_NAME} — ${companyName} home`}
          className="focus-visible:ring-ring flex items-center gap-3 rounded-md transition-opacity duration-200 hover:opacity-80 focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:outline-none"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={logoDark ?? logo}
            alt=""
            aria-hidden="true"
            className="size-8 rounded-md shadow-sm"
          />
          <span className="flex flex-col leading-tight">
            <span className="text-foreground font-mono text-sm font-semibold tracking-widest">
              {BRAND_NAME}
            </span>
            <span className="text-muted-foreground font-mono text-[10px] tracking-wider uppercase">
              {BRAND_TAGLINE}
            </span>
          </span>
        </Link>
      </header>

      {/* Who is signed in (multi-user deployments only) */}
      <AccountMenu />

      {children}

      {/* Branding - bottom right */}
      <footer className="fixed right-0 bottom-0 z-40 hidden p-6 md:block">
        <p className="text-muted-foreground font-mono text-xs font-medium tracking-wider uppercase">
          {companyName} by {BRAND_NAME}
        </p>
      </footer>
    </>
  );
}
