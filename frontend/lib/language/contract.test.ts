import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire, registerHooks } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import test from 'node:test';
const require = createRequire(import.meta.url);
const ts = require('typescript');
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
registerHooks({
 resolve(specifier, context, next) {
  if (specifier.startsWith('@/') || (specifier.startsWith('.') && context.parentURL?.startsWith(pathToFileURL(root).href))) {
   const base = specifier.startsWith('@/') ? resolve(root, specifier.slice(2)) : fileURLToPath(new URL(specifier, context.parentURL));
   for (const suffix of ['', '.ts', '.tsx']) if (existsSync(base + suffix) && /\.tsx?$/.test(base+suffix)) return {url:pathToFileURL(base+suffix).href,shortCircuit:true};
  }
  return next(specifier,context);
 },
 load(url, context, next) {
  if (/\.tsx?$/.test(url)) return {format:'module', shortCircuit:true,source:ts.transpileModule(readFileSync(fileURLToPath(url),'utf8'),{compilerOptions:{module:ts.ModuleKind.ESNext,jsx:ts.JsxEmit.ReactJSX}}).outputText};
  return next(url,context);
 }
});

const routeSource = () => readFileSync(resolve(root, 'app/api/language/route.ts'), 'utf8');

test('the reply-language view parser accepts only the three supported choices', async () => {
  const { parseLanguageView, languages } = await import('./contract.ts');
  assert.deepEqual([...languages], ['auto', 'en', 'es']);
  for (const language of ['auto', 'en', 'es']) {
    assert.deepEqual(
      parseLanguageView({ language, source: language === 'auto' ? 'default' : 'personal', applies_to: 'new_sessions' }),
      { language, source: language === 'auto' ? 'default' : 'personal', applies_to: 'new_sessions' }
    );
  }
  for (const bad of [
    null,
    undefined,
    'es',
    [],
    { language: 'fr', source: 'personal', applies_to: 'new_sessions' },
    { language: 'en', source: 'global', applies_to: 'new_sessions' },
    { language: 'en', source: 'personal', applies_to: 'every_session' },
    { language: 'en', source: 'personal' },
  ]) {
    assert.equal(parseLanguageView(bad), null, `expected ${JSON.stringify(bad)} to be rejected`);
  }
});

test('reply language uses authenticated owner-scoped reads and CSRF-protected writes', () => {
  const source = routeSource();
  assert.match(source, /await requireUser\(req\)/);
  assert.doesNotMatch(source, /requireAdmin/);
  assert.match(source, /guardMutation\(req, auth\.config, auth\.user\.userId\)/);
  assert.match(source, /callAsUser\(auth\.config, auth\.user\.userId, '\/users\/me\/language'/g);
  assert.match(source, /noStoreJson/);
  assert.match(source, /force-dynamic/);
  assert.doesNotMatch(source, /process\.env|console\./);
  // The route must never reach a deployment-wide settings surface.
  assert.doesNotMatch(source, /\/settings|admin/);
});

test('the write route refuses anything but the three owner choices', () => {
  const source = routeSource();
  assert.match(source, /Object\.keys\(body\)/);
  assert.match(source, /languages\.some/);
  assert.match(source, /422/);
});

test('the selector offers Auto, English and Espanol and marks the saved one', async () => {
  const { createElement } = await import('react');
  const { renderToStaticMarkup } = await import('react-dom/server');
  const { LanguageForm } = await import('../../components/settings/language.tsx');
  const html = renderToStaticMarkup(
    createElement(LanguageForm, {
      view: { language: 'es', source: 'personal', applies_to: 'new_sessions' },
      onSaved: () => {},
    })
  );
  for (const label of ['Reply language', 'Automatic', 'English', 'Español', 'new sessions'])
    assert.ok(html.includes(label), `missing ${label}`);
  assert.match(html, /value="es" selected=""/);
});

test('an account that never chose a language still shows the automatic default', async () => {
  const { createElement } = await import('react');
  const { renderToStaticMarkup } = await import('react-dom/server');
  const { LanguageForm } = await import('../../components/settings/language.tsx');
  const html = renderToStaticMarkup(
    createElement(LanguageForm, {
      view: { language: 'auto', source: 'default', applies_to: 'new_sessions' },
      onSaved: () => {},
    })
  );
  assert.match(html, /value="auto" selected=""/);
});

test('the selector is mounted in the existing voice settings panel', () => {
  const panel = readFileSync(resolve(root, 'components/settings/settings-panel.tsx'), 'utf8');
  assert.ok(panel.includes('<LanguageSettings />'));
  assert.match(panel, /from '@\/components\/settings\/language'/);
  // It must not be turned into a global/deployment setting.
  assert.doesNotMatch(panel, /for \(const owned of \[[^\]]*'reply_language'/);
});
