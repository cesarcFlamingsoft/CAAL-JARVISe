import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire, registerHooks } from 'node:module';
import { resolve, dirname } from 'node:path';
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
test('compact settings render honest provider choices and admin-only configuration', async () => {
 assert.ok(existsSync(resolve(root,'components/settings/tts.tsx')));
 const {createElement} = await import('react');
 const {renderToStaticMarkup} = await import('react-dom/server');
 const {TtsForm} = await import('../../components/settings/tts.tsx');
 const view = {provider:'qwen-trial', source:'default', qwen_configured:true, qwen_voice:'jarvis-designed', applies_to:'new_sessions', voicebox_status:'not_configured', can_configure:false, profile_id:null, engine:null, model_size:null};
 const html = renderToStaticMarkup(createElement(TtsForm,{view,onSaved:()=>{}}));
 for (const label of ['Kokoro', 'Qwen streaming', 'Voicebox', 'No URL needed', 'new sessions']) assert.ok(html.includes(label));
 assert.match(html, /value="voicebox" disabled=""/);
 assert.match(html, /value="qwen-trial" selected=""/);
 assert.ok(!html.includes('Voicebox URL'));
 const admin = renderToStaticMarkup(createElement(TtsForm,{view:{...view,can_configure:true},onSaved:()=>{}}));
 assert.ok(admin.includes('Voicebox URL') && admin.includes('Test connection') && admin.includes('Save connection'));
 assert.match(admin, /type="password"/);
});

test('live settings panel mounts personal voice controls instead of global provider buttons', () => {
 const panel = readFileSync(resolve(root,'components/settings/settings-panel.tsx'),'utf8');
 assert.ok(panel.includes('<TtsSettings />'));
 assert.ok(!panel.includes("handleTtsProviderChange('kokoro')"));
});

test('general settings saves cannot overwrite the existing global voice provider', () => {
 const panel = readFileSync(resolve(root,'components/settings/settings-panel.tsx'),'utf8');
 assert.match(panel, /for \(const owned of \[[^\]]*'tts_provider'/);
});
