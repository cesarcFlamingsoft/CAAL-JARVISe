'use client';

import { type ReactNode, useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  type InboxMessageDetail,
  type InboxMessageItem,
  browserInboxMessage,
} from '@/lib/dashboard/provider-data';

const FAILURES: Record<string, string> = {
  unauthorized: 'Sign in again to read this message.',
  not_found: 'This message or connected account is no longer available.',
  reconnect_required: 'Reconnect this account in Settings to read this message.',
  insufficient_scope: 'This account needs permission to read mail. Reconnect it in Settings.',
  not_configured: 'This mail provider is not configured on the server.',
  unsupported: 'Reading messages is not supported for this provider.',
};

const URL = /https?:\/\/[^\s<>]+/g;
const INVITE_LABEL =
  /^(When|Where|Guests|Join with Google Meet|Meeting link|More phone numbers|PIN):\s*/i;

function linkedLine(line: string): ReactNode {
  const nodes: ReactNode[] = [];
  let cursor = 0;
  for (const match of line.matchAll(URL)) {
    const raw = match[0];
    const href = raw.replace(/[)>.,;]+$/, '');
    const start = match.index ?? 0;
    if (start > cursor) nodes.push(line.slice(cursor, start));
    nodes.push(
      <a
        key={start}
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-primary decoration-primary/40 hover:decoration-primary font-medium break-all underline underline-offset-4"
      >
        {href.length > 70 ? 'Open secure link' : href}
      </a>
    );
    cursor = start + raw.length;
  }
  if (cursor < line.length) nodes.push(line.slice(cursor));
  return nodes.length ? nodes : line;
}

function EmailBody({ body }: { body: string }) {
  const lines = body.split('\n');
  return (
    <article className="mail-reader-body text-foreground/90 text-[15px] leading-7">
      {lines.map((line, index) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={index} className="h-4" aria-hidden="true" />;
        if (/^-{3,}.*-{3,}$/.test(trimmed) || trimmed === 'Original Appointment')
          return (
            <h3
              key={index}
              className="text-muted-foreground mt-7 border-t pt-5 text-xs font-semibold tracking-[0.16em] uppercase"
            >
              {trimmed.replace(/-/g, '').trim() || 'Details'}
            </h3>
          );
        if (INVITE_LABEL.test(trimmed))
          return (
            <p
              key={index}
              className="mail-reader-detail border-primary/15 bg-primary/[0.045] rounded-md border px-3 py-2 text-sm"
            >
              {linkedLine(trimmed)}
            </p>
          );
        return (
          <p key={index} className="break-words whitespace-pre-wrap">
            {linkedLine(line)}
          </p>
        );
      })}
    </article>
  );
}

export function MessageReader({
  message,
  onClose,
  onOpened,
}: {
  message: InboxMessageItem;
  onClose: () => void;
  onOpened?: (detail: InboxMessageDetail) => void;
}) {
  const titleId = useId();
  const dialog = useRef<HTMLDivElement>(null);
  const overlay = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  const opened = useRef(onOpened);
  opened.current = onOpened;
  const [attempt, setAttempt] = useState(0);
  const [detail, setDetail] = useState<InboxMessageDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const overflow = document.body.style.overflow;
    const siblings = [...document.body.children].filter(
      (node): node is HTMLElement => node instanceof HTMLElement && node !== overlay.current
    );
    const inert = siblings.map((node) => node.inert);
    siblings.forEach((node) => {
      node.inert = true;
    });
    document.body.style.overflow = 'hidden';
    dialog.current?.focus();
    const keyboard = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopImmediatePropagation();
        close.current();
      }
      if (event.key === 'Tab') {
        const items = [
          ...(dialog.current?.querySelectorAll<HTMLElement>('button, a[href], [tabindex="0"]') ??
            []),
        ].filter((node) => !node.hasAttribute('disabled'));
        const first = items[0];
        const last = items[items.length - 1];
        if (
          event.shiftKey &&
          (document.activeElement === first || document.activeElement === dialog.current)
        ) {
          event.preventDefault();
          last?.focus();
        } else if (
          !event.shiftKey &&
          (document.activeElement === last || document.activeElement === dialog.current)
        ) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    const focus = (event: FocusEvent) => {
      if (event.target instanceof Node && !dialog.current?.contains(event.target))
        dialog.current?.focus();
    };
    window.addEventListener('keydown', keyboard, true);
    document.addEventListener('focusin', focus);
    return () => {
      window.removeEventListener('keydown', keyboard, true);
      document.removeEventListener('focusin', focus);
      siblings.forEach((node, i) => {
        node.inert = inert[i];
      });
      document.body.style.overflow = overflow;
      if (previous?.isConnected) previous.focus();
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    const timeout = setTimeout(() => controller.abort(), 35_000);
    setDetail(null);
    setError(null);
    const query = new URLSearchParams({
      connectionId: message.connectionId,
      messageId: message.id,
    });
    void (async () => {
      try {
        const response = await fetch('/api/dashboard/inbox/message?' + query, {
          credentials: 'same-origin',
          cache: 'no-store',
          signal: controller.signal,
        });
        const data = await response.json();
        if (!active) return;
        if (!response.ok) {
          setError(FAILURES[data?.error] ?? 'This message is unavailable right now. Try again.');
          return;
        }
        const parsed = browserInboxMessage(data);
        if (!parsed || parsed.id !== message.id || parsed.connectionId !== message.connectionId)
          throw new Error('invalid');
        setDetail(parsed);
        opened.current?.(parsed);
      } catch {
        if (active) setError('This message is unavailable right now. Try again.');
      } finally {
        clearTimeout(timeout);
      }
    })();
    return () => {
      active = false;
      clearTimeout(timeout);
      controller.abort();
    };
  }, [message.connectionId, message.id, attempt]);

  return createPortal(
    <div
      ref={overlay}
      className="fixed inset-0 z-[120] grid place-items-center bg-black/60 p-4"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        className="mail-reader friday-panel bg-card text-card-foreground border-primary/25 flex max-h-[calc(100dvh-2rem)] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border shadow-[0_24px_80px_rgba(0,0,0,.6)]"
      >
        <header className="border-primary/20 from-primary/[0.10] via-card to-card border-b bg-gradient-to-r px-6 py-5 sm:px-8">
          <div className="flex items-start justify-between gap-5">
            <div className="min-w-0">
              <p className="text-primary mb-2 text-[10px] font-semibold tracking-[0.22em] uppercase">
                Secure message reader
              </p>
              <h2
                id={titleId}
                className="text-xl leading-snug font-semibold break-words sm:text-2xl"
              >
                {detail?.subject ?? message.subject ?? '(no subject)'}
              </h2>
            </div>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close message"
              className="border-primary/25 bg-background/50 hover:bg-primary/10 shrink-0 rounded-lg border px-3 py-1.5 text-sm"
            >
              Close
            </button>
          </div>
          <div className="border-primary/15 mt-5 grid gap-x-8 gap-y-3 border-t pt-4 text-sm sm:grid-cols-[minmax(0,1fr)_auto]">
            <div className="min-w-0">
              <span className="text-muted-foreground mr-2 text-xs font-semibold tracking-wider uppercase">
                From
              </span>
              {detail?.sender ?? message.sender ?? 'Unknown sender'}
            </div>
            <time
              dateTime={(detail ?? message).receivedAt}
              className="text-muted-foreground sm:text-right"
            >
              {new Date((detail ?? message).receivedAt).toLocaleString()}
            </time>
            {detail && (
              <div className="min-w-0 sm:col-span-2">
                <span className="text-muted-foreground mr-2 text-xs font-semibold tracking-wider uppercase">
                  To
                </span>
                {detail.recipients.join(', ') || 'Not provided'}
              </div>
            )}
          </div>
        </header>
        <div className="min-h-0 overflow-y-auto px-6 py-6 sm:px-8 sm:py-7">
          <div className="mx-auto max-w-3xl">
            {!detail && !error && (
              <p role="status" className="text-muted-foreground py-8 text-sm">
                Loading message…
              </p>
            )}
            {error && (
              <div
                role="alert"
                className="rounded-xl border border-red-400/30 bg-red-400/5 p-4 text-red-200"
              >
                <p>{error}</p>
                <button
                  type="button"
                  className="mt-3 rounded border px-3 py-1"
                  onClick={() => setAttempt((n) => n + 1)}
                >
                  Try again
                </button>
              </div>
            )}
            {detail && (
              <>
                <EmailBody body={detail.body || 'This message has no text content.'} />
                {detail.link && (
                  <a
                    href={detail.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="border-primary/30 bg-primary/[0.06] text-primary hover:bg-primary/[0.12] mt-8 inline-flex rounded-lg border px-4 py-2 text-sm font-medium"
                  >
                    Open in mail provider
                  </a>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
