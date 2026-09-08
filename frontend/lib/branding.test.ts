import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { extname, join, relative } from 'node:path';
import { describe, it } from 'node:test';

const FRONTEND_ROOT = process.cwd();

const SOURCE_DIRS = ['app', 'components', 'hooks', 'lib', 'styles'];
const SOURCE_FILES = ['app-config.ts', 'middleware.ts', 'next.config.ts'];
const SOURCE_EXTENSIONS = new Set(['.ts', '.tsx', '.js', '.jsx', '.mjs', '.css']);
const SKIPPED_DIRS = new Set(['node_modules', '.next', 'dist', 'build', '.turbo']);

/** Test files are allowed to mention retired branding, since they assert its absence. */
function isTestFile(path: string) {
  return /\.test\.[jt]sx?$/.test(path);
}

function walk(dir: string, out: string[]) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      if (!SKIPPED_DIRS.has(entry)) walk(full, out);
      continue;
    }
    if (SOURCE_EXTENSIONS.has(extname(entry)) && !isTestFile(entry)) out.push(full);
  }
}

function productionSourceFiles() {
  const files: string[] = [];
  for (const dir of SOURCE_DIRS) {
    const full = join(FRONTEND_ROOT, dir);
    if (existsSync(full)) walk(full, files);
  }
  for (const file of SOURCE_FILES) {
    const full = join(FRONTEND_ROOT, file);
    if (existsSync(full)) files.push(full);
  }
  return files;
}

function readSource(relPath: string) {
  return readFileSync(join(FRONTEND_ROOT, relPath), 'utf8');
}

function findMatches(pattern: RegExp) {
  const hits: string[] = [];
  for (const file of productionSourceFiles()) {
    const contents = readFileSync(file, 'utf8');
    contents.split('\n').forEach((line, index) => {
      if (pattern.test(line)) {
        hits.push(`${relative(FRONTEND_ROOT, file)}:${index + 1}: ${line.trim()}`);
      }
    });
  }
  return hits;
}

const RETIRED_BRAND_PATTERNS: Array<[string, RegExp]> = [
  ['CoreWorxLab name', /coreworx/i],
  ['cwl logo asset', /cwl-logo/i],
  ['CoreWorxLab GitHub link', /github\.com\/coreworxlab/i],
  ['CoreWorxLab YouTube link', /youtube\.com\/@?coreworxlab/i],
];

describe('frontend branding', () => {
  for (const [label, pattern] of RETIRED_BRAND_PATTERNS) {
    it(`has no ${label} in production sources`, () => {
      assert.deepEqual(findMatches(pattern), []);
    });
  }

  it('ships no retired cwl logo asset in public/', () => {
    const publicFiles = readdirSync(join(FRONTEND_ROOT, 'public'));
    assert.deepEqual(
      publicFiles.filter((name) => /cwl/i.test(name)),
      []
    );
  });

  it('points app config at local MexcanTech logo assets that exist and are optimized', () => {
    const config = readSource('app-config.ts');
    const logoPaths = [...config.matchAll(/^\s*logo(?:Dark)?: '([^']+)'/gm)].map((m) => m[1]);

    assert.equal(logoPaths.length, 2, 'expected both logo and logoDark to be configured');

    for (const logoPath of logoPaths) {
      assert.match(logoPath, /^\/mexcantech-/, `${logoPath} should be a local MexcanTech asset`);
      const assetPath = join(FRONTEND_ROOT, 'public', logoPath.replace(/^\//, ''));
      assert.ok(existsSync(assetPath), `${logoPath} should exist in public/`);
      assert.ok(
        statSync(assetPath).size < 150 * 1024,
        `${logoPath} should be optimized to well under 150KB`
      );
    }
  });

  it('ships a full MexcanTech wordmark asset alongside the square header mark', () => {
    const wordmark = join(FRONTEND_ROOT, 'public', 'mexcantech-logo.png');
    assert.ok(existsSync(wordmark), 'public/mexcantech-logo.png should exist');
    assert.ok(statSync(wordmark).size < 400 * 1024, 'wordmark should be optimized under 400KB');
  });

  it('renders accessible MexcanTech header and footer branding', () => {
    const layout = readSource('app/(app)/layout.tsx');

    assert.match(layout, /MEXCANTECH/);
    assert.match(layout, /Learn\. Build\. Empower\./);
    assert.match(layout, /aria-label=/, 'brand links should carry an accessible label');
    assert.match(layout, /alt=/, 'logo images should carry alt text');
  });

  it('keeps JARVIS as the user-facing product name', () => {
    const config = readSource('app-config.ts');
    assert.match(config, /companyName: 'JARVIS'/);
    assert.match(config, /pageTitle: 'JARVIS Voice Assistant'/);
    assert.match(config, /startButtonText: 'Talk to JARVIS'/);
  });

  it('generates the Open Graph image without LiveKit or CoreWorx asset assumptions', () => {
    const og = readSource('app/(app)/opengraph-image.tsx');

    assert.doesNotMatch(og, /lk-logo/);
    assert.doesNotMatch(og, /lk-wordmark/);
    assert.doesNotMatch(og, /About Acme/);
    assert.match(og, /MexcanTech/i, 'OG metadata should describe the MexcanTech brand');

    // Local assets must be resolved from public/ rather than assumed remote.
    const referenced = [...og.matchAll(/'public\/([^']+)'/g)].map((m) => m[1]);
    assert.ok(referenced.length > 0, 'expected local public/ assets to be referenced');
    for (const asset of referenced) {
      assert.ok(
        existsSync(join(FRONTEND_ROOT, 'public', asset)),
        `public/${asset} referenced by opengraph-image.tsx should exist`
      );
    }
  });
});
