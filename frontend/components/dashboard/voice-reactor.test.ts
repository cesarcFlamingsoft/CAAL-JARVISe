import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it } from 'node:test';

const source = readFileSync(join(process.cwd(), 'components/dashboard/voice-reactor.tsx'), 'utf8');

describe('voice reactor identity', () => {
  it('renders F.R.I.D.A.Y. rather than a legacy J monogram', () => {
    assert.match(source, />F\.R\.I\.D\.A\.Y\.<\/span>/);
    assert.match(source, /VOICE INTERFACE \/ F\.R\.I\.D\.A\.Y\./);
    assert.doesNotMatch(source, /reactor-monogram">J</);
    assert.doesNotMatch(source, /J\.A\.R\.V\.I\.S\./);
  });
});
