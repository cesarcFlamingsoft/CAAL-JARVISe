/**
 * Layout engine for the Monitor dashboard.
 *
 * Widgets live on a fixed-column grid. Every operation here is a pure function
 * over a `Layout` so the React side only has to render what it is given and the
 * rules (no overlaps, nothing outside the grid, nothing floating above a gap)
 * can be tested without a browser.
 *
 * Persistence is deliberately per *scope*: a signed-in user's arrangement is
 * keyed by their opaque id, so two people sharing a browser never see each
 * other's workspace, and a legacy single-user deployment keeps one local key.
 */

export type WidgetId = 'weather' | 'calendar' | 'reminders' | 'work' | 'inbox';

export interface WidgetPlacement {
  id: WidgetId;
  /** Column index, 0-based. */
  x: number;
  /** Row index, 0-based. */
  y: number;
  /** Width in columns. */
  w: number;
  /** Height in rows. */
  h: number;
}

export type Layout = WidgetPlacement[];

export interface WidgetDefinition {
  title: string;
  minW: number;
  minH: number;
  maxH: number;
}

export const GRID_COLUMNS = 12;
export const LAYOUT_VERSION = 1;

/** Sizing rules per widget. Titles are the user-facing names. */
export const WIDGETS: Record<WidgetId, WidgetDefinition> = {
  weather: { title: 'Weather', minW: 3, minH: 2, maxH: 8 },
  calendar: { title: 'Calendar', minW: 3, minH: 3, maxH: 12 },
  reminders: { title: 'Reminders', minW: 3, minH: 3, maxH: 12 },
  work: { title: 'Running work', minW: 3, minH: 3, maxH: 12 },
  inbox: { title: 'Inbox', minW: 3, minH: 3, maxH: 12 },
};

export const WIDGET_IDS = Object.keys(WIDGETS) as WidgetId[];

export const DEFAULT_LAYOUT: Layout = [
  { id: 'weather', x: 0, y: 0, w: 3, h: 3 },
  { id: 'calendar', x: 3, y: 0, w: 5, h: 5 },
  { id: 'reminders', x: 8, y: 0, w: 4, h: 5 },
  { id: 'work', x: 0, y: 3, w: 3, h: 4 },
  { id: 'inbox', x: 3, y: 5, w: 9, h: 5 },
];

const STORAGE_PREFIX = 'caal.dashboard.layout';

export function isWidgetId(value: unknown): value is WidgetId {
  return typeof value === 'string' && Object.prototype.hasOwnProperty.call(WIDGETS, value);
}

/** True when two placements share at least one cell. */
export function overlaps(a: WidgetPlacement, b: WidgetPlacement): boolean {
  if (a.id === b.id) return false;
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

/** Keep a placement inside the grid and inside its widget's size limits. */
export function clampPlacement(item: WidgetPlacement, cols = GRID_COLUMNS): WidgetPlacement {
  const def = WIDGETS[item.id];
  const w = Math.max(def.minW, Math.min(cols, Math.round(item.w)));
  const h = Math.max(def.minH, Math.min(def.maxH, Math.round(item.h)));
  const x = Math.max(0, Math.min(cols - w, Math.round(item.x)));
  const y = Math.max(0, Math.round(item.y));
  return { id: item.id, x, y, w, h };
}

function byPosition(a: WidgetPlacement, b: WidgetPlacement): number {
  return a.y - b.y || a.x - b.x;
}

function collidesWithAny(item: WidgetPlacement, placed: WidgetPlacement[]): boolean {
  return placed.some((other) => overlaps(item, other));
}

/**
 * Let one item fall upward until it rests on something (or row 0), then
 * fall downward until it no longer overlaps anything already placed.
 */
function settle(item: WidgetPlacement, placed: WidgetPlacement[]): WidgetPlacement {
  let y = item.y;
  while (y > 0 && !collidesWithAny({ ...item, y: y - 1 }, placed)) {
    y -= 1;
  }
  while (collidesWithAny({ ...item, y }, placed)) {
    y += 1;
  }
  return { ...item, y };
}

/** Restore the canonical id order so React keys and diffs stay stable. */
function canonical(layout: Layout): Layout {
  const byId = new Map(layout.map((item) => [item.id, item]));
  return WIDGET_IDS.flatMap((id) => {
    const item = byId.get(id);
    return item ? [item] : [];
  });
}

/**
 * Place `pinned` exactly where it is and let every other widget settle around
 * it in reading order. Used while a widget is being dragged or resized.
 */
export function placeWidget(layout: Layout, pinned: WidgetPlacement, cols = GRID_COLUMNS): Layout {
  const anchor = clampPlacement(pinned, cols);
  const placed: WidgetPlacement[] = [anchor];
  const others = layout
    .filter((item) => item.id !== anchor.id)
    .map((item) => clampPlacement(item, cols))
    .sort(byPosition);
  for (const item of others) {
    placed.push(settle(item, placed));
  }
  return canonical(placed);
}

/** Remove vertical gaps: every widget rests on the one below it or on row 0. */
export function compactLayout(layout: Layout, cols = GRID_COLUMNS): Layout {
  const placed: WidgetPlacement[] = [];
  const sorted = layout.map((item) => clampPlacement(item, cols)).sort(byPosition);
  for (const item of sorted) {
    placed.push(settle(item, placed));
  }
  return canonical(placed);
}

export function moveWidget(
  layout: Layout,
  id: WidgetId,
  x: number,
  y: number,
  cols = GRID_COLUMNS
): Layout {
  const item = layout.find((entry) => entry.id === id);
  if (!item) return layout;
  return placeWidget(layout, { ...item, x, y }, cols);
}

export function resizeWidget(
  layout: Layout,
  id: WidgetId,
  w: number,
  h: number,
  cols = GRID_COLUMNS
): Layout {
  const item = layout.find((entry) => entry.id === id);
  if (!item) return layout;
  const def = WIDGETS[id];
  const width = Math.max(def.minW, Math.min(cols - item.x, Math.round(w)));
  const height = Math.max(def.minH, Math.min(def.maxH, Math.round(h)));
  return placeWidget(layout, { ...item, w: width, h: height }, cols);
}

/**
 * Keyboard move: shift a widget by whole cells.
 *
 * Moving up or sideways is a plain move followed by compaction. Moving down
 * needs more care: gravity would pull a widget dropped one row lower straight
 * back to where it was, so "down" means stepping past the nearest widget
 * beneath it instead. With nothing beneath, there is nowhere lower to go.
 */
export function nudgeWidget(
  layout: Layout,
  id: WidgetId,
  dx: number,
  dy: number,
  cols = GRID_COLUMNS
): Layout {
  const item = layout.find((entry) => entry.id === id);
  if (!item) return layout;

  let y = item.y + dy;
  if (dy > 0) {
    const beneath = layout
      .filter(
        (other) =>
          other.id !== id &&
          other.y >= item.y + item.h &&
          other.x < item.x + item.w &&
          item.x < other.x + other.w
      )
      .sort(byPosition)[0];
    if (!beneath) return layout;
    y = beneath.y + beneath.h;
  }
  return compactLayout(moveWidget(layout, id, item.x + dx, y, cols), cols);
}

/** Keyboard resize: grow or shrink a widget by whole cells. */
export function growWidget(
  layout: Layout,
  id: WidgetId,
  dw: number,
  dh: number,
  cols = GRID_COLUMNS
): Layout {
  const item = layout.find((entry) => entry.id === id);
  if (!item) return layout;
  return compactLayout(resizeWidget(layout, id, item.w + dw, item.h + dh, cols), cols);
}

/** Total rows used, for sizing the grid container. */
export function layoutHeight(layout: Layout): number {
  return layout.reduce((max, item) => Math.max(max, item.y + item.h), 0);
}

/** Reading order for narrow screens where the grid collapses to one column. */
export function stackedOrder(layout: Layout): WidgetId[] {
  return [...layout].sort(byPosition).map((item) => item.id);
}

export interface StoredLayout {
  version: number;
  items: Layout;
}

export function serializeLayout(layout: Layout): StoredLayout {
  return {
    version: LAYOUT_VERSION,
    items: layout.map(({ id, x, y, w, h }) => ({ id, x, y, w, h })),
  };
}

function isCell(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

/**
 * Read a stored layout defensively. Anything that is not exactly what we
 * wrote falls back to the defaults; unknown widgets are dropped, widgets added
 * since the layout was saved are appended, and overlaps are repaired.
 */
export function parseLayout(raw: unknown): Layout {
  const stored = raw as { version?: unknown; items?: unknown } | null;
  if (!stored || typeof stored !== 'object' || Array.isArray(stored)) return DEFAULT_LAYOUT;
  if (stored.version !== LAYOUT_VERSION || !Array.isArray(stored.items)) return DEFAULT_LAYOUT;

  const seen = new Set<WidgetId>();
  const items: Layout = [];
  for (const entry of stored.items as unknown[]) {
    const item = entry as Record<string, unknown> | null;
    if (!item || typeof item !== 'object') return DEFAULT_LAYOUT;
    if (!isWidgetId(item.id)) continue;
    if (seen.has(item.id)) return DEFAULT_LAYOUT;
    if (!isCell(item.x) || !isCell(item.y) || !isCell(item.w) || !isCell(item.h)) {
      return DEFAULT_LAYOUT;
    }
    seen.add(item.id);
    items.push({ id: item.id, x: item.x, y: item.y, w: item.w, h: item.h });
  }
  if (items.length === 0) return DEFAULT_LAYOUT;

  const bottom = layoutHeight(items);
  for (const fallback of DEFAULT_LAYOUT) {
    if (!seen.has(fallback.id)) {
      items.push({ ...fallback, y: bottom + fallback.y });
    }
  }
  return compactLayout(items);
}

/** The slice of `/api/auth/me` that decides whose layout this is. */
export interface LayoutIdentity {
  configured: boolean;
  authenticated: boolean;
  user?: { userId: string };
}

export type LayoutScope = 'local' | 'anonymous' | `user:${string}`;

/**
 * Whose layout to load. A single-user deployment has one local layout; with
 * identity configured, each user gets their own and an anonymous visitor gets
 * a separate one that never mixes with any account.
 */
export function layoutScopeFor(me: LayoutIdentity | null | undefined): LayoutScope {
  if (!me || !me.configured) return 'local';
  if (me.authenticated && me.user && typeof me.user.userId === 'string' && me.user.userId) {
    return `user:${me.user.userId}`;
  }
  return 'anonymous';
}

export function layoutStorageKey(scope: LayoutScope): string {
  return `${STORAGE_PREFIX}.v${LAYOUT_VERSION}.${scope}`;
}
