/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-function-type -- dynamic framework-boundary test harness */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { test } from 'node:test';
import {
  accountIssues,
  browserInboxMessage,
  groupMessagesByAccount,
  messageQuery,
} from './provider-data.ts';

const require = createRequire(import.meta.url);
const ts = require('typescript');
const root = new URL('../../', import.meta.url);
const read = (path: string) => readFileSync(new URL(path, root), 'utf8');
// Execute the real route/component with framework boundaries substituted; no new dependencies.
function compile(
  path: string,
  imports: Record<string, unknown>,
  globals: Record<string, unknown> = {}
) {
  const source = ts.transpileModule(read(path), {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;
  const exports: Record<string, any> = {};
  new Function('exports', 'require', ...Object.keys(globals), source)(
    exports,
    (name: string) => {
      if (!(name in imports)) throw new Error('Unexpected import: ' + name);
      return imports[name];
    },
    ...Object.values(globals)
  );
  return exports;
}
const connectionId = 'con_' + 'a'.repeat(24);
const raw = {
  id: 'A+/=',
  connection_id: connectionId,
  provider: 'microsoft',
  subject: 'Subject',
  sender: 'Sender',
  received_at: '2023-11-14T22:13:20Z',
  unread: true,
  recipients: ['to@example.com'],
  body: 'First\nSecond',
  link: 'https://outlook.office.com/mail/',
  token: 'secret',
  html: '<script>bad</script>',
};

test('detail parser bounds and allowlists text contract and preserves line breaks', () => {
  const parsed = browserInboxMessage(raw)!;
  assert.equal(parsed.body, 'First\nSecond');
  assert.ok(!('token' in parsed) && !('html' in parsed) && !('preview' in parsed));
  assert.deepEqual(browserInboxMessage(JSON.parse(JSON.stringify(parsed))), parsed);
  const large = browserInboxMessage({
    ...raw,
    body: 'x'.repeat(100000),
    recipients: Array(100).fill('x'.repeat(400)),
    link: 'javascript:alert(1)',
  })!;
  assert.equal(large.body.length, 65536);
  assert.equal(large.recipients.length, 50);
  assert.equal(large.recipients[0].length, 254);
  assert.equal(large.link, null);
  for (const change of [
    { body: {} },
    { recipients: null },
    { provider: 'unknown' },
    { received_at: 'invalid' },
  ])
    assert.equal(browserInboxMessage({ ...raw, ...change }), null);
});

test('query only accepts one connection and opaque message ID', () => {
  const query = new URLSearchParams({ connectionId, messageId: raw.id });
  assert.equal(messageQuery(query), query.toString());
  for (const suffix of ['&messageId=other', '&connectionId=other', '&url=https://evil'])
    assert.equal(messageQuery(new URLSearchParams(query + suffix)), null);
  for (const id of ['', '..', '../x', 'a?x', 'a#x', 'x'.repeat(513)])
    assert.equal(messageQuery(new URLSearchParams({ connectionId, messageId: id })), null);
});

test('BFF executes auth, identity binding, bounded timeout, parser and safe failure paths', async () => {
  let authorized = true;
  let result: any = { ok: true, data: raw };
  const calls: any[] = [];
  const reply = (status: number, data: unknown) => ({ status, data, cache: 'no-store' });
  const { GET } = compile('app/api/dashboard/inbox/message/route.ts', {
    '@/lib/auth/backend': {
      callAsUser: async (...args: unknown[]) => {
        calls.push(args);
        return result;
      },
    },
    '@/lib/auth/guard': {
      requireUser: async () =>
        authorized
          ? { ok: true, config: {}, user: { userId: 'signed-in' } }
          : { ok: false, response: reply(401, 'unauthorized') },
      apiError: (status: number, code: string) => reply(status, { error: code }),
      noStoreJson: (data: unknown) => reply(200, data),
      backendFailure: (status: number) => reply(status, { error: 'safe' }),
    },
    '@/lib/dashboard/provider-data': { browserInboxMessage, messageQuery },
  });
  const request = (query = new URLSearchParams({ connectionId, messageId: raw.id })) => ({
    nextUrl: { searchParams: query },
  });
  authorized = false;
  assert.equal((await GET(request())).status, 401);
  assert.equal(calls.length, 0);
  authorized = true;
  assert.equal((await GET(request(new URLSearchParams()))).status, 422);
  assert.equal(calls.length, 0);
  const response = await GET(request());
  assert.equal(response.status, 200);
  assert.equal(response.cache, 'no-store');
  assert.equal(response.data.token, undefined);
  result = { ok: true, data: { ...raw, unread: false } };
  assert.equal((await GET(request())).data.unread, false);
  assert.equal(calls[0][1], 'signed-in');
  assert.equal(calls[0][3].timeoutMs, 30000);
  assert.match(calls[0][2], /messageId=A%2B%2F%3D/);
  result = { ok: true, data: { ...raw, id: 'other' } };
  assert.equal((await GET(request())).status, 502);
  result = { ok: false, status: 502, data: { detail: 'private provider payload' } };
  assert.deepEqual((await GET(request())).data, { error: 'unavailable' });
  result.data.detail = 'insufficient_scope';
  assert.deepEqual((await GET(request())).data, { error: 'insufficient_scope' });
});

test('modal mounts in a portal, isolates focus, handles Escape and restores background', () => {
  const effects: Array<() => () => void> = [];
  const handlers = new Map<string, Function>();
  let closed = 0;
  class Element {
    inert = false;
    isConnected = true;
    focusCount = 0;
    focus() {
      this.focusCount++;
      doc.activeElement = this;
    }
    contains(node: unknown) {
      return node === this;
    }
    hasAttribute() {
      return false;
    }
    querySelectorAll() {
      return [first, last];
    }
  }
  const trigger = new Element();
  const background = new Element();
  const first = new Element();
  const last = new Element();
  const dialog = new Element();
  const overlay = new Element();
  const refs = [dialog, overlay, null];
  const doc = {
    activeElement: trigger,
    body: { children: [background, overlay], style: { overflow: 'auto' } },
    addEventListener: (name: string, handler: Function) => handlers.set(name, handler),
    removeEventListener: (name: string) => handlers.delete(name),
  };
  const jsx = (type: unknown, props: any) => ({ type, props });
  const { MessageReader } = compile(
    'components/dashboard/message-reader.tsx',
    {
      react: {
        useEffect: (fn: any) => effects.push(fn),
        useId: () => 'title',
        useRef: () => ({ current: refs.shift() }),
        useState: (value: unknown) => [value, () => {}],
      },
      'react/jsx-runtime': { jsx, jsxs: jsx },
      'react-dom': { createPortal: (child: unknown, target: unknown) => ({ child, target }) },
      '@/lib/dashboard/provider-data': { browserInboxMessage },
    },
    { document: doc, window: doc, HTMLElement: Element, Node: Element }
  );
  const portal = MessageReader({ message: browserInboxMessage(raw), onClose: () => closed++ });
  assert.equal(portal.target, doc.body);
  assert.equal(portal.child.props.children.props.role, 'dialog');
  assert.equal(portal.child.props.children.props['aria-modal'], 'true');
  const cleanup = effects[0]();
  assert.equal(background.inert, true);
  assert.equal(overlay.inert, false);
  assert.equal(doc.activeElement, dialog);
  assert.equal(doc.body.style.overflow, 'hidden');
  const tab = (shiftKey: boolean) =>
    handlers.get('keydown')!({ key: 'Tab', shiftKey, preventDefault() {} });
  tab(false);
  assert.equal(doc.activeElement, first);
  tab(true);
  assert.equal(doc.activeElement, last);
  tab(false);
  assert.equal(doc.activeElement, first);
  let stopped = false;
  handlers.get('keydown')!({
    key: 'Escape',
    preventDefault() {},
    stopImmediatePropagation() {
      stopped = true;
    },
  });
  assert.equal(closed, 1);
  assert.equal(stopped, true);
  portal.child.props.onClick({ target: overlay, currentTarget: overlay });
  assert.equal(closed, 2);
  cleanup();
  assert.equal(background.inert, false);
  assert.equal(doc.body.style.overflow, 'auto');
  assert.equal(doc.activeElement, trigger);
  assert.equal(handlers.size, 0);
});

test('reader fetch uses no-store, aborts on close and ignores late private data', async () => {
  const effects: Array<() => () => void> = [];
  const updates: unknown[] = [];
  let resolve: (value: unknown) => void = () => {};
  let signal: AbortSignal | undefined;
  const jsx = (type: unknown, props: unknown) => ({ type, props });
  const { MessageReader } = compile(
    'components/dashboard/message-reader.tsx',
    {
      react: {
        useEffect: (fn: any) => effects.push(fn),
        useId: () => 'title',
        useRef: (value: unknown) => ({ current: value }),
        useState: (value: unknown) => [value, (next: unknown) => updates.push(next)],
      },
      'react/jsx-runtime': { jsx, jsxs: jsx },
      'react-dom': { createPortal: (child: unknown) => child },
      '@/lib/dashboard/provider-data': { browserInboxMessage },
    },
    {
      document: { body: {} },
      fetch: (url: string, options: RequestInit) => {
        assert.match(url, /messageId=A%2B%2F%3D/);
        assert.equal(options.cache, 'no-store');
        assert.equal(options.credentials, 'same-origin');
        signal = options.signal as AbortSignal;
        return new Promise((done) => {
          resolve = done;
        });
      },
    }
  );
  MessageReader({ message: browserInboxMessage(raw), onClose() {} });
  const cleanup = effects[1]();
  assert.deepEqual(updates, [null, null]);
  cleanup();
  assert.equal(signal?.aborted, true);
  resolve({ ok: true, json: async () => raw });
  await new Promise((done) => setImmediate(done));
  assert.deepEqual(updates, [null, null]);
});

for (const outcome of ['read', 'unread', 'error', 'mismatch']) {
  test(`reader reports only validated provider read state: ${outcome}`, async () => {
    const effects: Array<() => () => void> = [];
    const opened: unknown[] = [];
    const jsx = (type: unknown, props: unknown) => ({ type, props });
    const answer = {
      ...raw,
      unread: outcome !== 'read',
      ...(outcome === 'mismatch' ? { id: 'other' } : {}),
    };
    const { MessageReader } = compile(
      'components/dashboard/message-reader.tsx',
      {
        react: {
          useEffect: (fn: any) => effects.push(fn),
          useId: () => 'title',
          useRef: (value: unknown) => ({ current: value }),
          useState: (value: unknown) => [value, () => {}],
        },
        'react/jsx-runtime': { jsx, jsxs: jsx },
        'react-dom': { createPortal: (child: unknown) => child },
        '@/lib/dashboard/provider-data': { browserInboxMessage },
      },
      {
        document: { body: {} },
        fetch: async () => ({ ok: outcome !== 'error', json: async () => answer }),
      }
    );
    MessageReader({
      message: browserInboxMessage(raw),
      onClose() {},
      onOpened: (detail: unknown) => opened.push(detail),
    });
    const cleanup = effects[1]();
    await new Promise((done) => setImmediate(done));
    assert.deepEqual(
      opened,
      ['error', 'mismatch'].includes(outcome) ? [] : [browserInboxMessage(answer)]
    );
    cleanup();
  });
}

test('email-open contract propagates confirmed state into inbox counts without optimistic marking', () => {
  const reader = read('components/dashboard/message-reader.tsx');
  const inbox = read('components/dashboard/widgets/inbox-widget.tsx');
  assert.match(reader, /onOpened/);
  assert.ok(reader.indexOf('setDetail(parsed)') < reader.indexOf('opened.current?.(parsed)'));
  assert.match(inbox, /onOpened=/);
  assert.match(inbox, /groupMessagesByAccount\(visibleFeed\)/);
  assert.doesNotMatch(reader, /unread:\s*false/);
});

test('inbox updates only the opened account and yields to the next live snapshot', () => {
  let cursor = 0;
  const state: any[] = [];
  const jsx = (type: unknown, props: any) => ({ type, props });
  const { InboxWidget } = compile('components/dashboard/widgets/inbox-widget.tsx', {
    react: {
      useEffect() {},
      useState: (initial: unknown) => {
        const index = cursor++;
        if (!(index in state)) state[index] = initial;
        return [
          state[index],
          (next: any) => {
            state[index] = typeof next === 'function' ? next(state[index]) : next;
          },
        ];
      },
    },
    'react/jsx-runtime': { jsx, jsxs: jsx },
    '@/components/livekit/button': { Button: 'button' },
    '@/lib/dashboard/provider-data': { accountIssues, groupMessagesByAccount },
    '@/lib/utils': { cn: (...classes: string[]) => classes.join(' ') },
    '../account-section': { AccountSection: 'section' },
    '../message-reader': { MessageReader: 'reader' },
    '../widget-notice': {},
  });
  const message = { ...browserInboxMessage(raw), preview: '' };
  const other = { ...message, connectionId: 'con_' + 'b'.repeat(24) };
  let data = {
    generatedAt: 1,
    unreadCount: 2,
    messages: [message, other],
    accounts: [message, other].map((m) => ({
      connectionId: m.connectionId,
      provider: m.provider,
      status: 'ok',
      count: 1,
    })),
  };
  function render() {
    cursor = 0;
    return InboxWidget({
      feed: { status: 'ready', data, reload() {} },
      passwordLogin: true,
      now: null,
      onOpenSettings() {},
    });
  }
  const groups = (tree: any) => tree.props.children[2];
  let tree = render();
  groups(tree)[0].props.onRead(message);
  tree = render();
  assert.equal(groups(tree)[0].props.group.unreadCount, 1);
  tree.props.children[0].props.onOpened({ ...message, unread: false });
  tree = render();
  assert.equal(groups(tree)[0].props.group.unreadCount, 0);
  assert.equal(groups(tree)[1].props.group.unreadCount, 1);
  assert.equal(data.messages[0].unread, true);
  data = { ...data, generatedAt: 2 };
  assert.equal(groups(render())[0].props.group.unreadCount, 1);
});
