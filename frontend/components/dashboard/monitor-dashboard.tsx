'use client';

/**
 * The Monitor grid. Widgets are placed on a 12-column CSS grid from the
 * layout engine; on narrow screens the same widgets stack in reading order.
 *
 * Pointer gestures on a widget's handles preview a new arrangement while the
 * pointer is down and commit (compacted) on release. Arrow keys on the same
 * handles move or resize one cell at a time, and every change is announced.
 */
import { type ReactNode, useCallback, useId, useRef, useState } from 'react';
import { MotionConfig } from 'motion/react';
import {
  type Layout,
  WIDGETS,
  type WidgetId,
  compactLayout,
  growWidget,
  moveWidget,
  nudgeWidget,
  resizeWidget,
  stackedOrder,
} from '@/lib/dashboard/layout';
import { type GestureKind, WidgetFrame, type WidgetHandleProps } from './widget-frame';

/** Matches the `md:` breakpoint, below which the grid is stacked and fixed. */
const GRID_MEDIA_QUERY = '(min-width: 768px)';

/** Fallbacks if computed grid metrics cannot be read; match the classes below. */
const FALLBACK_COLUMN_PX = 64;
const FALLBACK_ROW_PX = 72;
const FALLBACK_GAP_PX = 12;

const ARROWS: Record<string, [number, number]> = {
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
  ArrowUp: [0, -1],
  ArrowDown: [0, 1],
};

interface Gesture {
  kind: GestureKind;
  id: WidgetId;
  pointerId: number;
  startX: number;
  startY: number;
  origin: Layout;
  item: Layout[number];
  columnPx: number;
  rowPx: number;
  gapPx: number;
  preview: Layout;
}

interface GridMetrics {
  columnPx: number;
  rowPx: number;
  gapPx: number;
}

function readMetrics(grid: HTMLElement): GridMetrics {
  const style = getComputedStyle(grid);
  const column = parseFloat(style.gridTemplateColumns.split(' ')[0] ?? '');
  const row = parseFloat(style.gridAutoRows);
  const gap = parseFloat(style.columnGap);
  return {
    columnPx: Number.isFinite(column) && column > 0 ? column : FALLBACK_COLUMN_PX,
    rowPx: Number.isFinite(row) && row > 0 ? row : FALLBACK_ROW_PX,
    gapPx: Number.isFinite(gap) && gap >= 0 ? gap : FALLBACK_GAP_PX,
  };
}

function sameLayout(a: Layout, b: Layout): boolean {
  return (
    a.length === b.length &&
    a.every((item, index) => {
      const other = b[index];
      return (
        item.id === other.id &&
        item.x === other.x &&
        item.y === other.y &&
        item.w === other.w &&
        item.h === other.h
      );
    })
  );
}

function describe(layout: Layout, id: WidgetId, kind: GestureKind): string {
  const item = layout.find((entry) => entry.id === id);
  if (!item) return '';
  const title = WIDGETS[id].title;
  return kind === 'move'
    ? `${title} moved to column ${item.x + 1}, row ${item.y + 1}.`
    : `${title} resized to ${item.w} columns by ${item.h} rows.`;
}

export interface MonitorDashboardProps {
  layout: Layout;
  onPreview: (layout: Layout) => void;
  onCommit: (layout: Layout) => void;
  renderWidget: (id: WidgetId) => { icon?: ReactNode; meta?: ReactNode; body: ReactNode };
}

export function MonitorDashboard({
  layout,
  onPreview,
  onCommit,
  renderWidget,
}: MonitorDashboardProps) {
  const gridRef = useRef<HTMLDivElement>(null);
  const gestureRef = useRef<Gesture | null>(null);
  const instructionsId = useId();
  const [active, setActive] = useState<{ id: WidgetId; kind: GestureKind } | null>(null);
  const [announcement, setAnnouncement] = useState('');

  const announce = useCallback((message: string) => {
    // Re-announce identical messages by clearing first.
    setAnnouncement('');
    requestAnimationFrame(() => setAnnouncement(message));
  }, []);

  const begin = useCallback(
    (kind: GestureKind, id: WidgetId) => (event: React.PointerEvent<HTMLButtonElement>) => {
      if (event.pointerType === 'mouse' && event.button !== 0) return;
      if (!window.matchMedia(GRID_MEDIA_QUERY).matches) return;
      const grid = gridRef.current;
      const item = layout.find((entry) => entry.id === id);
      if (!grid || !item) return;

      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      gestureRef.current = {
        kind,
        id,
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        origin: layout,
        item,
        preview: layout,
        ...readMetrics(grid),
      };
      setActive({ id, kind });
    },
    [layout]
  );

  const move = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => {
      const gesture = gestureRef.current;
      if (!gesture || event.pointerId !== gesture.pointerId) return;
      const columns = Math.round(
        (event.clientX - gesture.startX) / (gesture.columnPx + gesture.gapPx)
      );
      const rows = Math.round((event.clientY - gesture.startY) / (gesture.rowPx + gesture.gapPx));
      const { item, id, origin } = gesture;
      const next =
        gesture.kind === 'move'
          ? moveWidget(origin, id, item.x + columns, item.y + rows)
          : resizeWidget(origin, id, item.w + columns, item.h + rows);
      if (!sameLayout(next, gesture.preview)) {
        gesture.preview = next;
        onPreview(next);
      }
    },
    [onPreview]
  );

  const finish = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => {
      const gesture = gestureRef.current;
      if (!gesture || event.pointerId !== gesture.pointerId) return;
      gestureRef.current = null;
      setActive(null);
      const settled = compactLayout(gesture.preview);
      onCommit(settled);
      if (!sameLayout(settled, gesture.origin)) {
        announce(describe(settled, gesture.id, gesture.kind));
      }
    },
    [announce, onCommit]
  );

  const cancel = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => {
      const gesture = gestureRef.current;
      if (!gesture || event.pointerId !== gesture.pointerId) return;
      gestureRef.current = null;
      setActive(null);
      onPreview(gesture.origin);
    },
    [onPreview]
  );

  const keyboard = useCallback(
    (kind: GestureKind, id: WidgetId) => (event: React.KeyboardEvent<HTMLButtonElement>) => {
      const delta = ARROWS[event.key];
      if (!delta) return;
      event.preventDefault();
      const resize = kind === 'resize' || event.shiftKey;
      const [dx, dy] = delta;
      const next = resize ? growWidget(layout, id, dx, dy) : nudgeWidget(layout, id, dx, dy);
      if (sameLayout(next, layout)) return;
      onCommit(next);
      announce(describe(next, id, resize ? 'resize' : 'move'));
    },
    [announce, layout, onCommit]
  );

  const handlesFor = (kind: GestureKind, id: WidgetId): WidgetHandleProps => ({
    onPointerDown: begin(kind, id),
    onPointerMove: move,
    onPointerUp: finish,
    onPointerCancel: cancel,
    onKeyDown: keyboard(kind, id),
  });

  const byId = new Map(layout.map((item) => [item.id, item]));

  return (
    <MotionConfig reducedMotion="user">
      <p id={instructionsId} className="sr-only">
        Use the arrow keys to move this widget one cell at a time. Hold Shift with the arrow keys to
        resize it. Drag with a pointer to do the same.
      </p>
      <div aria-live="polite" aria-atomic className="sr-only">
        {announcement}
      </div>

      <div
        ref={gridRef}
        role="list"
        aria-label="Dashboard widgets"
        className="grid grid-cols-1 gap-3 md:auto-rows-[4.5rem] md:grid-cols-12"
      >
        {stackedOrder(layout).map((id) => {
          const placement = byId.get(id);
          if (!placement) return null;
          const { icon, meta, body } = renderWidget(id);
          return (
            <div key={id} role="listitem" className="contents">
              <WidgetFrame
                placement={placement}
                title={WIDGETS[id].title}
                icon={icon}
                meta={meta}
                instructionsId={instructionsId}
                active={active?.id === id ? active.kind : null}
                moveHandle={handlesFor('move', id)}
                resizeHandle={handlesFor('resize', id)}
              >
                {body}
              </WidgetFrame>
            </div>
          );
        })}
      </div>
    </MotionConfig>
  );
}
