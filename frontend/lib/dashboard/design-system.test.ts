import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { test } from 'node:test';

const root = new URL('../../', import.meta.url);
function sources(directory: URL): URL[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    if (entry.name.startsWith('.') || entry.name === 'node_modules') return [];
    const path = new URL(entry.name + (entry.isDirectory() ? '/' : ''), directory);
    return entry.isDirectory()
      ? sources(path)
      : /\.(tsx|css)$/.test(entry.name) && !/\.test\./.test(entry.name)
        ? [path]
        : [];
  });
}

test('production UI contains no legacy green status utilities or hardcoded green colors', () => {
  const violations: string[] = [];
  for (const file of [...sources(root), new URL('app-config.ts', root)]) {
    const source = readFileSync(file, 'utf8');
    if (/\b(?:green|emerald|lime)(?:-\d|\b)/i.test(source)) violations.push(file.pathname);
    for (const match of source.matchAll(
      /#([\da-f]{8}|[\da-f]{6}|[\da-f]{4}|[\da-f]{3})(?![\da-f])/gi
    )) {
      let hex = match[1];
      if (hex.length <= 4) hex = [...hex].map((c) => c + c).join('');
      const [r, g, b] = [0, 2, 4].map((offset) => parseInt(hex.slice(offset, offset + 2), 16));
      if (g > r * 1.15 && g > b * 1.15) violations.push(`${file.pathname}: ${match[0]}`);
    }
    for (const match of source.matchAll(/rgba?\(\s*(\d+)[, ]+\s*(\d+)[, ]+\s*(\d+)/gi)) {
      const [r, g, b] = match.slice(1).map(Number);
      if (g > r * 1.15 && g > b * 1.15) violations.push(`${file.pathname}: ${match[0]}`);
    }
    for (const match of source.matchAll(/hsla?\(\s*([\d.]+)/gi)) {
      if (Number(match[1]) >= 75 && Number(match[1]) <= 165)
        violations.push(`${file.pathname}: ${match[0]}`);
    }
    for (const match of source.matchAll(/oklch\([^)]*?\s([\d.]+)\s*\)/g)) {
      const hue = Number(match[1]);
      if (hue >= 100 && hue <= 170) violations.push(`${file.pathname}: ${match[0]}`);
    }
  }
  assert.deepEqual(violations, []);
});

test('shared cinematic tokens reach pages and body-portaled readers and settings', () => {
  const css = readFileSync(new URL('styles/globals.css', root), 'utf8');
  assert.match(readFileSync(new URL('app/layout.tsx', root), 'utf8'), /friday-theme/);
  for (const token of [
    'background',
    'foreground',
    'card',
    'primary',
    'border',
    'input',
    'ring',
    'success',
  ])
    assert.match(css, new RegExp(`--${token}:`));
  for (const style of ['friday-theme', 'friday-page', 'friday-panel', 'mail-reader'])
    assert.ok(css.includes(`.${style}`), style);
});

test('authenticated page shells and password form use shared cinematic surfaces', () => {
  for (const path of [
    'app/(app)/account/page.tsx',
    'app/(app)/admin/page.tsx',
    'app/(app)/connections/result/page.tsx',
    'app/admin/company/page.tsx',
    'app/admin/satellite/page.tsx',
    'app/change-password/page.tsx',
    'app/home-assistant/result/page.tsx',
  ]) {
    assert.match(readFileSync(new URL(path, root), 'utf8'), /friday-page/);
  }
  assert.match(
    readFileSync(new URL('components/auth/change-password-form.tsx', root), 'utf8'),
    /friday-panel/
  );
});
