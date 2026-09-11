'use client';

/**
 * The intent point. A thin ring follows the open hand; a closed fist fills
 * the centre and draws the dwell as an arc; a grab thickens the ring and
 * names what is moving. It is purely decorative (announcements live on the
 * grid) and lets every hit test pass through it.
 */
import type { CSSProperties } from 'react';
import type { GesturePhase, Point } from '@/lib/hands/gesture';
import { cn } from '@/lib/utils';

export interface HandCursorState {
  phase: GesturePhase;
  /** Viewport pixels. */
  point: Point | null;
  /** 0..1 while holding a fist; 1 while grabbing. */
  dwellProgress: number;
  /** What the hand is over, and whether a grab could move it. */
  target: { title: string; movable: boolean } | null;
}

const SIZE = 36;
const RADIUS = 15;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export function HandCursor({ phase, point, dwellProgress, target }: HandCursorState) {
  if (!point || phase === 'idle') return null;
  const grabbing = phase === 'grabbing';
  const holding = phase === 'holding';
  const blocked = grabbing && target !== null && !target.movable;
  const style: CSSProperties = {
    transform: 'translate(' + (point.x - SIZE / 2) + 'px, ' + (point.y - SIZE / 2) + 'px)',
    width: SIZE,
    height: SIZE,
  };
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 z-[110]">
      <div data-phase={phase} style={style} className="absolute top-0 left-0 will-change-transform">
        <svg viewBox={'0 0 ' + SIZE + ' ' + SIZE} className="size-full overflow-visible">
          <circle
            cx={SIZE / 2}
            cy={SIZE / 2}
            r={RADIUS}
            className={cn(
              'stroke-foreground fill-none transition-[stroke-width,opacity] duration-150 motion-reduce:transition-none',
              grabbing ? 'opacity-90' : 'opacity-60'
            )}
            strokeWidth={grabbing ? 2.5 : 1.25}
          />
          {(holding || grabbing) && (
            <circle
              cx={SIZE / 2}
              cy={SIZE / 2}
              r={RADIUS}
              className={cn('fill-none', blocked ? 'stroke-destructive' : 'stroke-foreground')}
              strokeWidth={2.5}
              strokeLinecap="round"
              strokeDasharray={CIRCUMFERENCE}
              strokeDashoffset={CIRCUMFERENCE * (1 - Math.min(1, Math.max(0, dwellProgress)))}
              transform={'rotate(-90 ' + SIZE / 2 + ' ' + SIZE / 2 + ')'}
            />
          )}
          <circle
            cx={SIZE / 2}
            cy={SIZE / 2}
            r={holding || grabbing ? 3.5 : 1.5}
            className={cn(
              'fill-foreground transition-[r] duration-150 motion-reduce:transition-none',
              blocked && 'fill-destructive'
            )}
          />
        </svg>
        {grabbing && target && (
          <span
            className={cn(
              'bg-background/90 text-foreground absolute top-full left-1/2 mt-1 -translate-x-1/2 rounded-full border px-2 py-0.5 font-mono text-[10px] tracking-wider whitespace-nowrap uppercase shadow-sm',
              blocked && 'text-destructive'
            )}
          >
            {blocked ? 'Not movable' : 'Moving ' + target.title}
          </span>
        )}
      </div>
    </div>
  );
}
