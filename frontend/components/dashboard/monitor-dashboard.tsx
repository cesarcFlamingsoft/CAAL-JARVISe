'use client';

/**
 * The Monitor grid. Widgets are placed on a 12-column CSS grid from the
 * layout engine; on narrow screens the same widgets stack in reading order.
 *
 * Pointer gestures on a widget's handles preview a new arrangement while the
 * pointer is down and commit (compacted) on release. Arrow keys on the same
 * handles move or resize one cell at a time, and every change is announced.
 *
 * Hand control drives the very same move gesture through a small controller
 * (hover, select, grab, move, release, cancel) exposed on a ref, so a hand
 * can only ever do what a pointer already can, and only to widgets the grid
 * can actually move.
 */
import {
  type ReactNode,
  type Ref,
  useCallback,
  useEffect,
  useId,
  useImperativeHandle,
  useRef,
  useState,
} from 'react';
import { MotionConfig } from 'motion/react';
import {
  type Layout,
  WIDGETS,
  type WidgetId,
  compactLayout,
  growWidget,
  isWidgetId,
  moveWidget,
  nudgeWidget,
  resizeWidget,
  stackedOrder,
} from '@/lib/dashboard/layout';
import type { Point } from '@/lib/hands/gesture';
import type { HandSurfaceController, HandTarget } from '@/lib/hands/surface';
import {
  type GestureKind,
  type HandState,
  WidgetFrame,
  type WidgetHandleProps,
} from './widget-frame';

/** Matches the `md:` breakpoint, below which the grid is stacked and fixed. */
const GRID_MEDIA_QUERY = '(min-width: 768px)';

/** Fallbacks if computed grid metrics cannot be read; match the classes below. */
const FALLBACK_COLUMN_PX = 64;
const FALLBACK_ROW_PX = 72;
const FALLBACK_GAP_PX = 12;

/** Pointer ids are non-negative; the hand borrows one that never collides. */
const HAND_POINTER_ID = -1;

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

interface WidgetTarget extends HandTarget {
  id: WidgetId;
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
    ? title + ' moved to column ' + (item.x + 1) + ', row ' + (item.y + 1) + '.'
    : title + ' resized to ' + item.w + ' columns by ' + item.h + ' rows.';
}

export interface MonitorDashboardProps {
  layout: Layout;
  onPreview: (layout: Layout) => void;
  onCommit: (layout: Layout) => void;
  renderWidget: (id: WidgetId) => { icon?: ReactNode; meta?: ReactNode; body: ReactNode };
  /** Receives the hand controller while mounted; absent when hand control is not wired. */
  handController?: Ref<HandSurfaceController | null>;
}

export function MonitorDashboard({
  layout,
  onPreview,
  onCommit,
  renderWidget,
  handController,
}: MonitorDashboardProps) {
  const gridRef = useRef<HTMLDivElement>(null);
  const gestureRef = useRef<Gesture | null>(null);
  const clickConsumedRef = useRef(false);
  const layoutRef = useRef(layout);
  const instructionsId = useId();
  const [active, setActive] = useState<{ id: WidgetId; kind: GestureKind } | null>(null);
  const [handHover, setHandHover] = useState<WidgetId | null>(null);
  const [handSelected, setHandSelected] = useState<WidgetId | null>(null);
  const [announcement, setAnnouncement] = useState('');

  useEffect(() => {
    layoutRef.current = layout;
  }, [layout]);

  const announce = useCallback((message: string) => {
    // Re-announce identical messages by clearing first.
    setAnnouncement('');
    requestAnimationFrame(() => setAnnouncement(message));
  }, []);

  /** Start a move or resize from a viewport point. False when the grid cannot rearrange now. */
  const beginGesture = useCallback(
    (kind: GestureKind, id: WidgetId, pointerId: number, clientX: number, clientY: number) => {
      if (gestureRef.current) return false;
      if (!window.matchMedia(GRID_MEDIA_QUERY).matches) return false;
      const grid = gridRef.current;
      const current = layoutRef.current;
      const item = current.find((entry) => entry.id === id);
      if (!grid || !item) return false;
      gestureRef.current = {
        kind,
        id,
        pointerId,
        startX: clientX,
        startY: clientY,
        origin: current,
        item,
        preview: current,
        ...readMetrics(grid),
      };
      setActive({ id, kind });
      return true;
    },
    []
  );

  const updateGesture = useCallback(
    (pointerId: number, clientX: number, clientY: number) => {
      const gesture = gestureRef.current;
      if (!gesture || pointerId !== gesture.pointerId) return;
      const columns = Math.round((clientX - gesture.startX) / (gesture.columnPx + gesture.gapPx));
      const rows = Math.round((clientY - gesture.startY) / (gesture.rowPx + gesture.gapPx));
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

  const finishGesture = useCallback(
    (pointerId: number) => {
      const gesture = gestureRef.current;
      if (!gesture || pointerId !== gesture.pointerId) return;
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

  const cancelGesture = useCallback(
    (pointerId: number) => {
      const gesture = gestureRef.current;
      if (!gesture || pointerId !== gesture.pointerId) return false;
      gestureRef.current = null;
      setActive(null);
      onPreview(gesture.origin);
      return true;
    },
    [onPreview]
  );

  const begin = useCallback(
    (kind: GestureKind, id: WidgetId) => (event: React.PointerEvent<HTMLButtonElement>) => {
      if (event.pointerType === 'mouse' && event.button !== 0) return;
      if (!beginGesture(kind, id, event.pointerId, event.clientX, event.clientY)) return;
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [beginGesture]
  );

  const move = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) =>
      updateGesture(event.pointerId, event.clientX, event.clientY),
    [updateGesture]
  );

  const finish = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => finishGesture(event.pointerId),
    [finishGesture]
  );

  const cancel = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => {
      cancelGesture(event.pointerId);
    },
    [cancelGesture]
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

  /** The widget under a viewport point, and whether the grid could move it right now. */
  const targetAt = useCallback((point: Point): WidgetTarget | null => {
    const element = document.elementFromPoint(point.x, point.y);
    const id = element?.closest('[data-widget]')?.getAttribute('data-widget');
    if (!isWidgetId(id)) return null;
    const movable =
      window.matchMedia(GRID_MEDIA_QUERY).matches &&
      layoutRef.current.some((item) => item.id === id);
    return { id, title: WIDGETS[id].title, movable };
  }, []);

  /** A fist may activate the same real controls a pointer would, never drag handles. */
  const controlAt = useCallback((point: Point): HTMLElement | null => {
    const element = document.elementFromPoint(point.x, point.y);
    if (!(element instanceof HTMLElement)) return null;
    const control = element.closest<HTMLElement>(
      'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [role="button"]:not([aria-disabled="true"])'
    );
    return control && !control.closest('[data-handle]') ? control : null;
  }, []);

  useImperativeHandle(
    handController,
    (): HandSurfaceController => ({
      hover(point) {
        const target = targetAt(point);
        setHandHover(target?.id ?? null);
        return target;
      },
      hoverEnd() {
        setHandHover(null);
        setHandSelected(null);
      },
      select(point) {
        const target = targetAt(point);
        const control = controlAt(point);
        clickConsumedRef.current = control !== null;
        setHandSelected(target?.id ?? null);
        if (control) {
          control.focus({ preventScroll: true });
          control.click();
          announce('Activated control.');
          return target ? { ...target, movable: false } : null;
        }
        if (target) {
          // The real handle takes focus, so keyboard and screen reader follow the hand.
          gridRef.current
            ?.querySelector<HTMLElement>('[data-widget="' + target.id + '"] [data-handle="move"]')
            ?.focus({ preventScroll: true });
          announce(target.title + ' selected.');
        }
        return target;
      },
      grab(point) {
        const target = targetAt(point);
        if (clickConsumedRef.current) {
          clickConsumedRef.current = false;
          return target ? { ...target, movable: false } : null;
        }
        if (!target) return null;
        setHandSelected(target.id);
        if (!target.movable) {
          announce(target.title + ' cannot be moved on this screen.');
          return target;
        }
        if (!beginGesture('move', target.id, HAND_POINTER_ID, point.x, point.y)) {
          return { ...target, movable: false };
        }
        announce('Moving ' + target.title + '. Open your hand to drop it.');
        return target;
      },
      move(point) {
        updateGesture(HAND_POINTER_ID, point.x, point.y);
      },
      release() {
        finishGesture(HAND_POINTER_ID);
      },
      cancel() {
        if (cancelGesture(HAND_POINTER_ID)) {
          announce('Move cancelled. The layout is back where it was.');
        }
      },
    }),
    [announce, beginGesture, cancelGesture, controlAt, finishGesture, targetAt, updateGesture]
  );

  const handlesFor = (kind: GestureKind, id: WidgetId): WidgetHandleProps => ({
    onPointerDown: begin(kind, id),
    onPointerMove: move,
    onPointerUp: finish,
    onPointerCancel: cancel,
    onKeyDown: keyboard(kind, id),
  });

  const handStateFor = (id: WidgetId): HandState | null =>
    handSelected === id ? 'selected' : handHover === id ? 'hover' : null;

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
                hand={handStateFor(id)}
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
