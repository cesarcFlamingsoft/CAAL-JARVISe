import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  DEFAULT_LAYOUT,
  GRID_COLUMNS,
  LAYOUT_VERSION,
  type Layout,
  WIDGETS,
  type WidgetId,
  compactLayout,
  growWidget,
  layoutHeight,
  layoutScopeFor,
  layoutStorageKey,
  moveWidget,
  nudgeWidget,
  overlaps,
  parseLayout,
  resizeWidget,
  serializeLayout,
  stackedOrder,
} from './layout.ts';

const ALL_IDS: WidgetId[] = ['weather', 'calendar', 'reminders', 'work', 'inbox'];

function find(layout: Layout, id: WidgetId) {
  const item = layout.find((entry) => entry.id === id);
  assert.ok(item, `expected ${id} in layout`);
  return item;
}

function assertWellFormed(layout: Layout) {
  assert.deepEqual(
    layout.map((item) => item.id).sort(),
    [...ALL_IDS].sort(),
    'every widget appears exactly once'
  );
  for (const item of layout) {
    const def = WIDGETS[item.id];
    assert.ok(Number.isInteger(item.x) && item.x >= 0, `${item.id}.x is a non-negative integer`);
    assert.ok(Number.isInteger(item.y) && item.y >= 0, `${item.id}.y is a non-negative integer`);
    assert.ok(item.w >= def.minW, `${item.id} is at least its minimum width`);
    assert.ok(item.h >= def.minH, `${item.id} is at least its minimum height`);
    assert.ok(item.x + item.w <= GRID_COLUMNS, `${item.id} stays inside the grid`);
  }
  for (const a of layout) {
    for (const b of layout) {
      if (a.id !== b.id) {
        assert.equal(overlaps(a, b), false, `${a.id} must not overlap ${b.id}`);
      }
    }
  }
}

describe('default layout', () => {
  it('places every widget inside the grid with no overlaps', () => {
    assertWellFormed(DEFAULT_LAYOUT);
  });
});

describe('moving a widget', () => {
  it('lands where asked and pushes whatever it covers downward', () => {
    const calendar = find(DEFAULT_LAYOUT, 'calendar');

    const next = moveWidget(DEFAULT_LAYOUT, 'weather', calendar.x, calendar.y);

    assertWellFormed(next);
    const weather = find(next, 'weather');
    assert.equal(weather.x, calendar.x);
    assert.equal(weather.y, calendar.y);
    assert.ok(find(next, 'calendar').y >= weather.y + weather.h, 'calendar moved below weather');
  });

  it('clamps a move that would leave the grid', () => {
    const next = moveWidget(DEFAULT_LAYOUT, 'weather', GRID_COLUMNS + 5, -3);

    assertWellFormed(next);
    const weather = find(next, 'weather');
    assert.equal(weather.x + weather.w, GRID_COLUMNS);
    assert.equal(weather.y, 0);
  });

  it('is a no-op for an unknown widget and never mutates its input', () => {
    const snapshot = JSON.stringify(DEFAULT_LAYOUT);

    const next = moveWidget(DEFAULT_LAYOUT, 'nope' as WidgetId, 1, 1);

    assert.deepEqual(next, DEFAULT_LAYOUT);
    assert.equal(JSON.stringify(DEFAULT_LAYOUT), snapshot);
  });

  it('can be nudged one cell at a time from the keyboard', () => {
    const before = find(DEFAULT_LAYOUT, 'reminders');

    const next = nudgeWidget(DEFAULT_LAYOUT, 'reminders', -1, 0);

    assertWellFormed(next);
    assert.equal(find(next, 'reminders').x, before.x - 1);
  });

  it('swaps with the neighbour below when nudged down, instead of floating back up', () => {
    // Weather sits directly above Running work in the default layout.
    const weather = find(DEFAULT_LAYOUT, 'weather');
    const work = find(DEFAULT_LAYOUT, 'work');
    assert.equal(work.y, weather.y + weather.h);

    const next = nudgeWidget(DEFAULT_LAYOUT, 'weather', 0, 1);

    assertWellFormed(next);
    assert.equal(find(next, 'work').y, 0, 'work takes the top slot');
    assert.equal(find(next, 'weather').y, find(next, 'work').h, 'weather rests below work');

    // And back up again restores the original order.
    const restored = nudgeWidget(next, 'weather', 0, -1);
    assert.deepEqual(restored, DEFAULT_LAYOUT);

    // With nothing beneath it, a downward nudge is a no-op.
    assert.deepEqual(nudgeWidget(next, 'weather', 0, 1), next);
  });
});

describe('resizing a widget', () => {
  it('honours the widget minimum size', () => {
    const next = resizeWidget(DEFAULT_LAYOUT, 'calendar', 0, 0);

    assertWellFormed(next);
    const calendar = find(next, 'calendar');
    assert.equal(calendar.w, WIDGETS.calendar.minW);
    assert.equal(calendar.h, WIDGETS.calendar.minH);
  });

  it('cannot grow past the right edge of the grid', () => {
    const before = find(DEFAULT_LAYOUT, 'reminders');

    const next = growWidget(DEFAULT_LAYOUT, 'reminders', 99, 0);

    assertWellFormed(next);
    const reminders = find(next, 'reminders');
    assert.equal(reminders.x, before.x);
    assert.equal(reminders.x + reminders.w, GRID_COLUMNS);
  });

  it('pushes neighbours down instead of overlapping them', () => {
    const next = growWidget(DEFAULT_LAYOUT, 'weather', 0, 6);

    assertWellFormed(next);
  });
});

describe('compaction', () => {
  it('closes vertical gaps so nothing floats', () => {
    const floating: Layout = DEFAULT_LAYOUT.map((item) => ({ ...item, y: item.y + 7 }));

    const next = compactLayout(floating);

    assertWellFormed(next);
    assert.equal(Math.min(...next.map((item) => item.y)), 0);
    assert.equal(layoutHeight(next), layoutHeight(DEFAULT_LAYOUT));
  });

  it('orders widgets top-to-bottom, left-to-right for a single-column screen', () => {
    const layout: Layout = [
      { id: 'work', x: 6, y: 4, w: 6, h: 3 },
      { id: 'reminders', x: 0, y: 4, w: 6, h: 3 },
      { id: 'calendar', x: 6, y: 0, w: 6, h: 4 },
      { id: 'weather', x: 0, y: 0, w: 6, h: 4 },
    ];

    assert.deepEqual(stackedOrder(layout), ['weather', 'calendar', 'reminders', 'work']);
  });
});

describe('persistence', () => {
  it('round-trips through its serialized form', () => {
    const moved = moveWidget(DEFAULT_LAYOUT, 'work', 0, 0);

    const restored = parseLayout(JSON.parse(JSON.stringify(serializeLayout(moved))));

    assert.deepEqual(restored, moved);
    assert.equal(serializeLayout(moved).version, LAYOUT_VERSION);
  });

  it('falls back to the default layout for anything malformed', () => {
    for (const bad of [null, undefined, 'x', 42, [], {}, { version: LAYOUT_VERSION }]) {
      assert.deepEqual(parseLayout(bad), DEFAULT_LAYOUT, `expected defaults for ${String(bad)}`);
    }
    assert.deepEqual(
      parseLayout({ version: LAYOUT_VERSION + 1, items: DEFAULT_LAYOUT }),
      DEFAULT_LAYOUT
    );
    assert.deepEqual(
      parseLayout({
        version: LAYOUT_VERSION,
        items: [{ id: 'weather', x: 1.5, y: 0, w: 4, h: 3 }],
      }),
      DEFAULT_LAYOUT
    );
  });

  it('appends the inbox below a layout saved before it existed, without moving the rest', () => {
    const stored = {
      version: LAYOUT_VERSION,
      items: [
        { id: 'weather', x: 0, y: 0, w: 3, h: 3 },
        { id: 'calendar', x: 3, y: 0, w: 5, h: 5 },
        { id: 'reminders', x: 8, y: 0, w: 4, h: 5 },
        { id: 'work', x: 0, y: 3, w: 3, h: 4 },
      ],
    };

    const restored = parseLayout(stored);

    assertWellFormed(restored);
    for (const item of stored.items) {
      assert.deepEqual(find(restored, item.id as WidgetId), item);
    }
    const inbox = find(restored, 'inbox');
    assert.ok(inbox.y >= 5, 'inbox lands below the widgets that were already placed');
    assert.ok(inbox.w >= WIDGETS.inbox.minW && inbox.h >= WIDGETS.inbox.minH);
  });

  it('drops unknown widgets, adds missing ones, and repairs overlaps', () => {
    const stored = {
      version: LAYOUT_VERSION,
      items: [
        { id: 'weather', x: 0, y: 0, w: 6, h: 3 },
        { id: 'calendar', x: 0, y: 0, w: 6, h: 3 },
        { id: 'stocks', x: 6, y: 0, w: 6, h: 3 },
      ],
    };

    const restored = parseLayout(stored);

    assertWellFormed(restored);
    assert.equal(find(restored, 'weather').y, 0);
  });
});

describe('per-user scoping', () => {
  it('keys the stored layout by the signed-in user, never by anything shared', () => {
    const userScope = layoutScopeFor({
      configured: true,
      authenticated: true,
      user: { userId: 'usr_0123456789abcdef01234567' },
    });
    assert.equal(userScope, 'user:usr_0123456789abcdef01234567');
    assert.equal(layoutScopeFor({ configured: false, authenticated: false }), 'local');
    assert.equal(layoutScopeFor({ configured: true, authenticated: false }), 'anonymous');
    assert.equal(layoutScopeFor(null), 'local');
  });

  it('builds distinct storage keys per scope and version', () => {
    const a = layoutStorageKey('user:usr_0123456789abcdef01234567');
    const b = layoutStorageKey('user:usr_fedcba9876543210fedcba98');

    assert.notEqual(a, b);
    assert.ok(a.includes(`v${LAYOUT_VERSION}`));
    assert.ok(a.startsWith('caal.dashboard.layout.'));
  });
});
