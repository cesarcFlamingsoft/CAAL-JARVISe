import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire, registerHooks } from 'node:module';
import { resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

// Render the actual components and Next Link. Only AccountMenu's initial
// auth/me state is supplied as a fixture; this does not authenticate any user.
const root = fileURLToPath(new URL('../../', import.meta.url));
const require = createRequire(import.meta.url);
const ts = require('typescript');
let me: unknown = null;
(globalThis as typeof globalThis & { __navigationMe?: () => unknown }).__navigationMe = () => me;
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === 'react' && context.parentURL?.endsWith('/account-menu.tsx')) {
      return { url: 'fixture:menu-react', shortCircuit: true };
    }
    if (specifier === 'next/link') return nextResolve('next/link.js', context);
    if (specifier.startsWith('@/') || (specifier.startsWith('.') && context.parentURL?.startsWith(pathToFileURL(root).href))) {
      const base = specifier.startsWith('@/') ? resolve(root, specifier.slice(2)) : fileURLToPath(new URL(specifier, context.parentURL));
      for (const ext of ['.ts', '.tsx']) {
        if (existsSync(base + ext)) return { url: pathToFileURL(base + ext).href, shortCircuit: true };
      }
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url === 'fixture:menu-react') return { format: 'module', shortCircuit: true, source: `export {useEffect} from ${JSON.stringify(pathToFileURL(require.resolve('react')).href)}; export const useState = () => [globalThis.__navigationMe(), () => {}];` };
    if (url.startsWith(pathToFileURL(root).href) && /\.tsx?$/.test(url) && !url.includes('/node_modules/')) {
      return { format: 'module', shortCircuit: true, source: ts.transpileModule(readFileSync(fileURLToPath(url), 'utf8'), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX } }).outputText };
    }
    return nextLoad(url, context);
  },
});

const { AccountMenu } = await import('../../components/account/account-menu.tsx');
const { AdminPanel } = await import('../../components/admin/admin-panel.tsx');
const voiceLink = /<a\b[^>]*href="\/admin\/satellite"[^>]*>Home Assistant Voice<\/a>/;

test('Administrator panel renders a visible Home Assistant Voice link', () => {
  assert.match(renderToStaticMarkup(createElement(AdminPanel, { selfId: 'fixture-admin' })), voiceLink);
});
test('top account navigation renders Home Assistant Voice for administrators', () => {
  me = { configured: true, authenticated: true, user: { role: 'admin', displayName: 'Fixture admin' } };
  assert.match(renderToStaticMarkup(createElement(AccountMenu)), voiceLink);
});
test('members, signed-out, loading and unconfigured navigation expose no admin configuration', () => {
  for (const state of [null, { configured: false }, { configured: true, authenticated: false, user: { role: 'admin' } }, { configured: true, authenticated: true, user: { role: 'member', displayName: 'Fixture member' } }]) {
    me = state;
    const html = renderToStaticMarkup(createElement(AccountMenu));
    assert.doesNotMatch(html, /href="\/admin|Home Assistant Voice/);
  }
});
