/**
 * How the browser asks for a company-private session, and how it leaves one.
 *
 * Entering is a navigation, not a toggle, and that is the point: a session's
 * kind is fixed when the room is created, so changing it has to mean getting a
 * new room. `/?company=1` loads the app fresh and asks the BFF to mint a
 * principal carrying the request; `/` loads it fresh and does not. Leaving a
 * company session therefore really is a fresh ordinary session, rather than a
 * flag flipped inside a conversation that already has company passages in it.
 *
 * Nothing here decides anything. The request is honoured only if the BFF signs
 * it into the room-bound principal and the worker matches the user against the
 * provisioned library owner.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  COMPANY_SESSION_HREF,
  ORDINARY_SESSION_HREF,
  companySessionRequested,
} from './session-mode.ts';

test('the company entry link asks for a company session', () => {
  assert.equal(companySessionRequested(new URL(`https://x${COMPANY_SESSION_HREF}`).search), true);
});

test('the exit link asks for an ordinary session', () => {
  assert.equal(companySessionRequested(new URL(`https://x${ORDINARY_SESSION_HREF}`).search), false);
});

test('an ordinary page load asks for nothing', () => {
  for (const search of ['', '?', '?foo=bar', '?company=0', '?company=', '?company=true']) {
    assert.equal(companySessionRequested(search), false, search);
  }
});

test('only the exact opt-in counts', () => {
  assert.equal(companySessionRequested('?company=1'), true);
  assert.equal(companySessionRequested('?company=1&else=2'), true);
  assert.equal(companySessionRequested('?COMPANY=1'), false);
});

test('a non-string is not an opt-in', () => {
  for (const value of [undefined, null, 1, {}]) {
    assert.equal(companySessionRequested(value as unknown as string), false);
  }
});

test('the exit href goes somewhere that is not the company entry', () => {
  assert.notEqual(COMPANY_SESSION_HREF, ORDINARY_SESSION_HREF);
});
