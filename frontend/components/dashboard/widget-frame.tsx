'use client';

/**
 * The chrome around one dashboard widget: a titled region with a move handle,
 * a resize handle, and a scrolling body. Both handles are real buttons so the
 * keyboard can do everything the pointer can.
 */
import { type CSSProperties, type ReactNode, useId } from 'react';
import { motion } from 'motion/react';
import { ArrowsOutSimple, DotsSixVertical } from '@phosphor-icons/react/dist/ssr';
import type { WidgetPlacement } from '@/lib/dashboard/layout';
import { cn } from '@/lib/utils';

export type GestureKind = 'move' | 'resize';

/** How the hand cursor relates to this widget, if hand control is on. */
export type HandState = 'hover' | 'selected';

export interface WidgetHandleProps {
  onPointerDown: (event: React.PointerEvent<HTMLButtonElement>) => void;
  onPointerMove: (event: React.PointerEvent<HTMLButtonElement>) => void;
  onPointerUp: (event: React.PointerEvent<HTMLButtonElement>) => void;
  onPointerCancel: (event: React.PointerEvent<HTMLButtonElement>) => void;
  onKeyDown: (event: React.KeyboardEvent<HTMLButtonElement>) => void;
}

interface WidgetFrameProps {
  placement: WidgetPlacement;
  title: string;
  icon?: ReactNode;
  /** Short live status shown beside the title, e.g. "2 sources". */
  meta?: ReactNode;
  instructionsId: string;
  active: GestureKind | null;
  hand?: HandState | null;
  moveHandle: WidgetHandleProps;
  resizeHandle: WidgetHandleProps;
  children: ReactNode;
}

const HANDLE_CLASSES = cn(
  'text-muted-foreground hover:text-foreground hover:bg-muted rounded-md transition-colors',
  'focus-visible:ring-ring/60 outline-none focus-visible:ring-[3px]',
  'touch-none select-none',
  // Rearranging needs the grid; on a stacked phone layout the order is fixed.
  'hidden md:inline-flex'
);

export function WidgetFrame({
  placement,
  title,
  icon,
  meta,
  instructionsId,
  active,
  hand = null,
  moveHandle,
  resizeHandle,
  children,
}: WidgetFrameProps) {
  const headingId = useId();
  const { x, y, w, h } = placement;
  const style = {
    '--widget-column': `${x + 1} / span ${w}`,
    '--widget-row': `${y + 1} / span ${h}`,
  } as CSSProperties;

  return (
    <motion.section
      layout
      transition={{ layout: { type: 'spring', stiffness: 700, damping: 60, mass: 1 } }}
      aria-labelledby={headingId}
      data-widget={placement.id}
      data-active={active ?? undefined}
      data-hand={hand ?? undefined}
      style={style}
      className={cn(
        'bg-card text-card-foreground border-border relative flex min-h-56 min-w-0 flex-col overflow-hidden rounded-2xl border shadow-sm',
        'md:[grid-column:var(--widget-column)] md:[grid-row:var(--widget-row)] md:min-h-0',
        'transition-shadow duration-150 motion-reduce:transition-none',
        !active && hand === 'hover' && 'ring-ring/40 ring-1',
        !active && hand === 'selected' && 'ring-foreground/50 ring-2',
        active && 'ring-ring/60 z-10 shadow-xl ring-2'
      )}
    >
      <header className="flex items-center gap-2 border-b px-3 py-2">
        <button
          type="button"
          aria-label={`Move ${title}`}
          aria-describedby={instructionsId}
          data-handle="move"
          className={cn(HANDLE_CLASSES, 'cursor-grab p-1 active:cursor-grabbing')}
          {...moveHandle}
        >
          <DotsSixVertical aria-hidden className="size-4" weight="bold" />
        </button>
        {icon && (
          <span aria-hidden className="text-muted-foreground inline-flex shrink-0">
            {icon}
          </span>
        )}
        <h2 id={headingId} className="truncate text-sm font-semibold tracking-tight">
          {title}
        </h2>
        {meta && <span className="text-muted-foreground ml-auto truncate text-xs">{meta}</span>}
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">{children}</div>

      <button
        type="button"
        aria-label={`Resize ${title}`}
        aria-describedby={instructionsId}
        data-handle="resize"
        className={cn(HANDLE_CLASSES, 'absolute right-1 bottom-1 cursor-nwse-resize p-1')}
        {...resizeHandle}
      >
        <ArrowsOutSimple aria-hidden className="size-3.5" weight="bold" />
      </button>
    </motion.section>
  );
}
