'use client';

/**
 * A compact, per-account dashboard summary. Opening it presents that account's
 * own data in a focused inspection card above the persistent workspace.
 */
import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { ArrowUpRight, X } from '@phosphor-icons/react/dist/ssr';
import { AnimatePresence, motion } from 'motion/react';
import type { FeedAccount } from '@/lib/dashboard/provider-data';
import { AccountIssues, PROVIDER_LABEL, accountName } from './account-issues';

interface AccountSectionProps {
  account: FeedAccount;
  /** This account's own item count, already in words, e.g. "2 events". */
  count: string | null;
  /** Shown when this account answered with nothing at all. */
  emptyLabel: string;
  onOpenSettings: () => void;
  children?: ReactNode;
}

export function AccountSection(props: AccountSectionProps) {
  const { account, count, emptyLabel, onOpenSettings, children } = props;
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const answered = account.status === 'ok';
  const provider = PROVIDER_LABEL[account.provider];
  const label = account.accountLabel ?? provider;
  const summaryCount = answered && count ? count : 'Needs attention';
  const empty = <p className="text-muted-foreground text-sm">{emptyLabel}</p>;

  const close = () => {
    setOpen(false);
    requestAnimationFrame(() => triggerRef.current?.focus());
  };

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close();
    };
    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', onKeyDown);
    requestAnimationFrame(() => dialogRef.current?.focus());
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  const overlay = (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[100] grid place-items-center p-4 sm:p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
        >
          <motion.button
            type="button"
            aria-label={'Close ' + label + ' details'}
            className="bg-background/80 absolute inset-0 cursor-default backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={close}
          />
          <motion.div
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby={titleId}
            tabIndex={-1}
            initial={{ opacity: 0, y: 20, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.98 }}
            transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
            className="border-border/80 bg-card relative z-10 flex max-h-[min(46rem,calc(100vh-2rem))] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border shadow-2xl sm:max-h-[calc(100vh-3rem)]"
          >
            <header className="border-border/70 bg-muted/25 flex items-center gap-3 border-b px-4 py-3 sm:px-5">
              <span className="bg-primary/10 text-primary grid size-10 shrink-0 place-items-center rounded-xl font-mono text-xs font-bold tracking-wider">
                {provider.slice(0, 2).toUpperCase()}
              </span>
              <div className="min-w-0 flex-1">
                <h2 id={titleId} className="truncate text-base font-semibold">
                  {label}
                </h2>
                <p className="text-muted-foreground text-[11px] font-medium tracking-[0.16em] uppercase">
                  {provider} · {summaryCount}
                </p>
              </div>
              <button
                type="button"
                onClick={close}
                className="hover:bg-muted focus-visible:ring-ring grid size-10 place-items-center rounded-full outline-none transition-colors focus-visible:ring-2"
                aria-label="Close account details"
              >
                <X className="size-4" weight="bold" />
              </button>
            </header>
            <div className="overflow-y-auto px-4 py-4 sm:px-5 sm:py-5">
              {answered ? children ?? empty : <AccountIssues issues={[account]} onOpenSettings={onOpenSettings} />}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );

  return (
    <section
      aria-label={accountName(account)}
      className="border-border/70 bg-card/45 overflow-hidden rounded-xl border shadow-sm transition-shadow duration-300 hover:shadow-md"
    >
      <button
        ref={triggerRef}
        type="button"
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-controls={titleId}
        onClick={() => setOpen(true)}
        className="group flex min-h-14 w-full items-center gap-3 px-3 text-left outline-none transition-colors duration-200 hover:bg-muted/55 focus-visible:bg-muted/65 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
      >
        <span className="bg-primary/10 text-primary grid size-8 shrink-0 place-items-center rounded-lg font-mono text-[10px] font-bold tracking-wider">
          {provider.slice(0, 2).toUpperCase()}
        </span>
        <span className="min-w-0 flex-1">
          <span className="text-foreground block truncate text-sm font-semibold">{label}</span>
          <span className="text-muted-foreground block text-[10px] font-medium tracking-[0.16em] uppercase">
            {provider}
          </span>
        </span>
        <span className="text-muted-foreground shrink-0 text-xs font-medium tabular-nums">
          {summaryCount}
        </span>
        <ArrowUpRight
          aria-hidden
          className="text-muted-foreground size-4 shrink-0 transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
          weight="bold"
        />
      </button>
      {typeof document !== 'undefined' ? createPortal(overlay, document.body) : null}
    </section>
  );
}
