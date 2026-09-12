import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { SCHEDULED_EVENT_NAME, SCHEDULED_TOPIC, isScheduledChange } from './scheduled-events.ts';

/**
 * The browser half of the scheduled-change packet: what it accepts, what it
 * refuses, and the structural guards for the two files that cannot run under
 * node --test because they import React and LiveKit.
 */

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = (path: string) => readFileSync(join(ROOT, path), 'utf8');

const HOOK = 'hooks/useScheduledUpdates.ts';
const FEED_HOOK = 'hooks/useDashboardFeed.ts';
const WORKSPACE = 'components/dashboard/workspace.tsx';

const bytes = (text: string) => new TextEncoder().encode(text);
const CONSTANT = bytes('{"v":1,"kind":"scheduled_changed"}');

describe('the scheduled-change packet', () => {
  it('is accepted only as the exact constant on its own topic', () => {
    assert.equal(SCHEDULED_TOPIC, 'scheduled_changed');
    assert.equal(isScheduledChange(SCHEDULED_TOPIC, CONSTANT), true);
    // Field order is not part of the contract; the field set is.
    const reordered = bytes('{"kind":"scheduled_changed","v":1}');
    assert.equal(isScheduledChange(SCHEDULED_TOPIC, reordered), true);
  });

  it('ignores another topic, however well formed the payload is', () => {
    assert.equal(isScheduledChange('tool_status', CONSTANT), false);
    assert.equal(isScheduledChange(undefined, CONSTANT), false);
    assert.equal(isScheduledChange('', CONSTANT), false);
  });

  it('ignores anything that is not exactly those two fields', () => {
    const refused = [
      '{"v":1,"kind":"scheduled_changed","title":"Biopsy results"}',
      '{"v":1}',
      '{"kind":"scheduled_changed"}',
      '{"v":2,"kind":"scheduled_changed"}',
      '{"v":"1","kind":"scheduled_changed"}',
      '{"v":1,"kind":"reload_everything"}',
      '[{"v":1,"kind":"scheduled_changed"}]',
      '"scheduled_changed"',
      'null',
      'not json at all',
      '',
    ];
    for (const payload of refused) {
      assert.equal(isScheduledChange(SCHEDULED_TOPIC, bytes(payload)), false, payload);
    }
  });

  it('ignores a payload too large to be the constant', () => {
    const padded = '{"v":1,"kind":"scheduled_changed"' + ' '.repeat(64) + '}';
    assert.equal(isScheduledChange(SCHEDULED_TOPIC, bytes(padded)), false);
  });

  it('ignores bytes that are not valid UTF-8', () => {
    assert.equal(isScheduledChange(SCHEDULED_TOPIC, new Uint8Array([0xff, 0xfe, 0xfd])), false);
  });
});

describe('the scheduled-change listener', () => {
  it('names the browser-local event once, for both halves to share', () => {
    assert.equal(SCHEDULED_EVENT_NAME, 'scheduled-items-updated');
  });

  it('listens on this browser own room, only while it is connected', () => {
    assert.ok(existsSync(join(ROOT, HOOK)), HOOK + ' is missing');
    const source = read(HOOK);
    assert.match(source, /useMaybeRoomContext\(/, 'it must attach to this browser own room');
    assert.match(source, /RoomEvent\.DataReceived/);
    assert.match(source, /isScheduledChange\(/, 'every packet must be validated');
    assert.match(source, /ConnectionState\.Connected/, 'only a connected room is listened to');
    // The event name is the shared constant, so the two halves cannot drift.
    assert.match(source, /SCHEDULED_EVENT_NAME/);
    assert.match(source, /from '@\/lib\/dashboard\/scheduled-events'/);
    assert.match(source, /dispatchEvent\(/);
    // A nudge, not a carrier: nothing is ever read out of the packet.
    assert.ok(!source.includes('JSON.parse'), 'the hook must not read the packet itself');
    assert.ok(!source.includes('fetch('), 'the hook must not fetch; the feed does that');
  });

  it('reloads the scheduled feed and leaves email and calendar alone', () => {
    const source = read(FEED_HOOK);
    assert.match(source, /SCHEDULED_EVENT_NAME/);
    assert.match(source, /from '@\/lib\/dashboard\/scheduled-events'/);
    assert.match(
      source,
      /path [!=]== '\/api\/dashboard\/reminders'/,
      'only the scheduled feed may react to a scheduled change'
    );
    // The ordinary fallbacks stay exactly where they were.
    assert.match(source, /'visibilitychange'/);
    assert.match(source, /REFRESH_INTERVAL_MS/);
  });

  it('is mounted in the workspace', () => {
    assert.match(read(WORKSPACE), /useScheduledUpdates\(/);
  });
});
