import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { test } from 'node:test';
import { createRefreshGate, reduceFeed } from './refresh.ts';
import { WORK_STATUS, browserWork, workRefreshInterval } from './work.ts';

const item = {
  title: 'Research report',
  status: 'queued',
  created_at: 1700000000,
  updated_at: 1700000001,
  started_at: null,
  finished_at: null,
};
const feed = { generated_at: 1700000002, items: [item] };
const read = (path: string) => readFileSync(new URL('../../' + path, import.meta.url), 'utf8');

test('work parser reduces extra fields at both levels and bounds all input', () => {
  assert.deepEqual(
    browserWork({
      ...feed,
      owner: 'excluded',
      items: [
        {
          ...item,
          task_id: 'excluded',
          session_key: 'excluded',
          request: 'excluded',
          result: 'excluded',
          error: 'excluded',
          callback: 'excluded',
          attempts: 3,
          lease: 'excluded',
        },
      ],
    }),
    feed
  );
  assert.ok(browserWork({ ...feed, items: Array(12).fill(item) }));
  for (const value of [
    null,
    {},
    { ...feed, generated_at: '1700000002' },
    { ...feed, items: Array(13).fill(item) },
    { ...feed, items: [null] },
  ]) {
    assert.equal(browserWork(value), null);
  }
  for (const change of [
    { title: 'x'.repeat(121) },
    { title: '' },
    { title: 4 },
    { status: 'unknown' },
    { created_at: 1.5 },
    { updated_at: -1 },
    { started_at: 'yesterday' },
    { finished_at: Infinity },
    { created_at: 9e15 },
  ]) {
    assert.equal(browserWork({ ...feed, items: [{ ...item, ...change }] }), null);
  }
});

test('every lifecycle state survives parsing and only active work polls at 15 seconds', () => {
  for (const status of ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted']) {
    const data = { ...feed, items: [{ ...item, status }] };
    assert.ok(browserWork(data));
    assert.equal(workRefreshInterval(data), ['queued', 'running'].includes(status) ? 15000 : 60000);
  }
  assert.equal(workRefreshInterval({ ...feed, items: [] }), 60000);
  assert.equal(workRefreshInterval(null), 60000);
  assert.deepEqual(
    Object.values(WORK_STATUS).map((entry) => entry.label),
    [
      'Queued for FRIDAY',
      'FRIDAY is working',
      'Completed',
      'Couldn’t complete',
      'Cancelled',
      'Interrupted — retry if needed',
    ]
  );
  assert.equal(WORK_STATUS.queued.detail, 'FRIDAY will start when a worker is free');
  assert.equal(WORK_STATUS.running.detail, 'I’m actively working on this now');
});

test('BFF authenticates, fixes its path, reduces payload and never logs content', () => {
  const source = read('app/api/dashboard/work/route.ts');
  assert.match(source, /requireUser\(req\)/);
  assert.match(source, /callAsUser\(auth.config, auth.user.userId, '\/users\/me\/dashboard\/work'/);
  assert.match(source, /browserWork\(result.data\)/);
  assert.match(source, /noStoreJson\(feed\)/);
  assert.match(source, /force-dynamic/);
  assert.match(source, /searchParams.size/);
  assert.doesNotMatch(source, /console\.(?:error|log)\([^;]*(?:result|userId|title)/);
  assert.doesNotMatch(source, /export async function (?:POST|PUT|DELETE|PATCH)/);
});

test('work is wired into PersonalWorkspace with visible-only adaptive refresh and honest states', () => {
  const hook = read('hooks/useDashboardFeed.ts');
  assert.match(hook, /'\/api\/dashboard\/work'/);
  assert.match(hook, /workRefreshInterval\(next.data\)/);
  assert.match(hook, /document.visibilityState === 'visible'/);
  assert.match(hook, /createRefreshGate\(load, visible\)/);
  assert.match(hook, /cache: 'no-store'/);
  assert.match(hook, /window.clearInterval\(timer\)/);
  const workspace = read('components/dashboard/workspace.tsx');
  assert.ok(
    workspace.indexOf("useDashboardFeed('/api/dashboard/work'") >
      workspace.indexOf('function PersonalWorkspace')
  );
  assert.match(workspace, /<WorkWidget feed=\{work\}/);
  const widget = read('components/dashboard/widgets/work-widget.tsx');
  assert.match(widget, /FRIDAY is working in the background/);
  assert.match(widget, /activeCount/);
  assert.match(widget, /WORK_STATUS\[item.status\]/);
  assert.match(widget, /dateTime=/);
  assert.match(widget, /formatDayOrTime/);
  for (const state of ['loading', 'unauthorized', 'unconfigured', 'error'])
    assert.ok(widget.includes("feed.status === '" + state + "'"));
  assert.match(widget, /FeedFreshness/);
  assert.doesNotMatch(widget, /dangerouslySetInnerHTML|tasksEndpoint|no endpoint yet/);
});

test('mounted hook switches polling cadence, skips hidden fetches and cleans up', async () => {
  const ts = createRequire(import.meta.url)('typescript');
  let cleanup: () => void = () => {};
  let state: unknown;
  let now = 10000;
  let visibilityState = 'visible';
  let calls = 0;
  let status = 'queued';
  let timerId = 0;
  const timers = new Map<number, { callback: () => void; delay: number }>();
  const listeners = new Map<string, () => void>();
  const events = {
    addEventListener: (name: string, callback: () => void) => listeners.set(name, callback),
    removeEventListener: (name: string) => listeners.delete(name),
  };
  const imports: Record<string, unknown> = {
    react: {
      useCallback: (callback: unknown) => callback,
      useRef: (current: unknown) => ({ current }),
      useState: (initial: unknown) => {
        state = initial;
        return [
          state,
          (next: unknown) => {
            state = typeof next === 'function' ? next(state) : next;
          },
        ];
      },
      useEffect: (effect: () => () => void) => {
        cleanup = effect();
      },
    },
    '@/lib/dashboard/refresh': {
      reduceFeed,
      createRefreshGate: (load: () => Promise<void>, visible: () => boolean) =>
        createRefreshGate(load, visible, () => now),
    },
    '@/lib/dashboard/work': { workRefreshInterval },
    '@/lib/dashboard/scheduled-events': { SCHEDULED_EVENT_NAME: 'scheduled' },
  };
  const compiled = ts.transpileModule(read('hooks/useDashboardFeed.ts'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const exports: { useDashboardFeed?: (path: string, parse: typeof browserWork) => unknown } = {};
  new Function('exports', 'require', 'fetch', 'window', 'document', compiled)(
    exports,
    (name: string) => {
      assert.ok(name in imports);
      return imports[name];
    },
    async (_path: string, options: RequestInit) => {
      calls++;
      assert.equal(options.cache, 'no-store');
      return new Response(JSON.stringify({ ...feed, items: [{ ...item, status }] }));
    },
    {
      ...events,
      setInterval: (callback: () => void, delay: number) => {
        timers.set(++timerId, { callback, delay });
        return timerId;
      },
      clearInterval: (id: number) => timers.delete(id),
    },
    {
      ...events,
      get visibilityState() {
        return visibilityState;
      },
    }
  );
  const settle = () => new Promise<void>((resolve) => setImmediate(resolve));
  const tick = async () => {
    assert.equal(timers.size, 1);
    const timer = [...timers.values()][0];
    now += timer.delay;
    timer.callback();
    await settle();
  };
  assert.ok(exports.useDashboardFeed);
  exports.useDashboardFeed('/api/dashboard/work', browserWork);
  await settle();
  assert.equal(calls, 1);
  assert.equal([...timers.values()][0].delay, 15000);
  visibilityState = 'hidden';
  await tick();
  assert.equal(calls, 1);
  visibilityState = 'visible';
  status = 'running';
  await tick();
  assert.equal(calls, 2);
  assert.equal([...timers.values()][0].delay, 15000);
  status = 'succeeded';
  await tick();
  assert.equal([...timers.values()][0].delay, 60000);
  status = 'queued';
  await tick();
  assert.equal([...timers.values()][0].delay, 15000);
  cleanup();
  assert.equal(timers.size, 0);
  assert.equal(listeners.size, 0);
});
