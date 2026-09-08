import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  MAX_ACTIVITY_ENTRIES,
  type ToolActivity,
  appendToolActivity,
  parseToolStatus,
  voiceStatus,
} from './activity.ts';

describe('tool status packets', () => {
  it('reads a well-formed packet from the agent', () => {
    const status = parseToolStatus({
      tool_used: true,
      tool_names: ['list_calendar_events', 'create_reminder'],
      tool_params: [{ range: 'today' }, {}],
    });

    assert.deepEqual(status, {
      toolUsed: true,
      toolNames: ['list_calendar_events', 'create_reminder'],
    });
  });

  it('ignores anything that is not a packet and drops non-string names', () => {
    assert.equal(parseToolStatus(null), null);
    assert.equal(parseToolStatus('tool_used'), null);
    assert.equal(parseToolStatus({ tool_used: 'yes' }), null);
    assert.deepEqual(parseToolStatus({ tool_used: true, tool_names: ['ok', 7, null] }), {
      toolUsed: true,
      toolNames: ['ok'],
    });
    assert.deepEqual(parseToolStatus({ tool_used: false }), { toolUsed: false, toolNames: [] });
  });
});

describe('activity feed', () => {
  it('records tool calls newest-first and skips responses that used no tool', () => {
    let feed: ToolActivity[] = [];

    feed = appendToolActivity(feed, { toolUsed: true, toolNames: ['a'] }, 1000);
    feed = appendToolActivity(feed, { toolUsed: false, toolNames: [] }, 2000);
    feed = appendToolActivity(feed, { toolUsed: true, toolNames: ['b', 'c'] }, 3000);

    assert.deepEqual(
      feed.map((entry) => ({ tools: entry.tools, at: entry.at })),
      [
        { tools: ['b', 'c'], at: 3000 },
        { tools: ['a'], at: 1000 },
      ]
    );
    assert.notEqual(feed[0].id, feed[1].id);
  });

  it('is bounded so a long session cannot grow without limit', () => {
    let feed: ToolActivity[] = [];
    for (let i = 0; i < MAX_ACTIVITY_ENTRIES + 5; i++) {
      feed = appendToolActivity(feed, { toolUsed: true, toolNames: [`tool_${i}`] }, i);
    }

    assert.equal(feed.length, MAX_ACTIVITY_ENTRIES);
    assert.deepEqual(feed[0].tools, [`tool_${MAX_ACTIVITY_ENTRIES + 4}`]);
  });
});

describe('voice status', () => {
  it('describes the docked voice capability honestly for every phase', () => {
    assert.deepEqual(voiceStatus({ isConnected: false, connecting: false }), {
      label: 'Voice is idle',
      tone: 'idle',
    });
    assert.deepEqual(voiceStatus({ isConnected: false, connecting: true }), {
      label: 'Connecting to JARVIS…',
      tone: 'busy',
    });
    assert.deepEqual(
      voiceStatus({ isConnected: true, connecting: false, agentState: 'listening' }),
      {
        label: 'JARVIS is listening',
        tone: 'live',
      }
    );
    assert.deepEqual(
      voiceStatus({ isConnected: true, connecting: false, agentState: 'thinking' }),
      {
        label: 'JARVIS is thinking',
        tone: 'busy',
      }
    );
    assert.deepEqual(
      voiceStatus({ isConnected: true, connecting: false, agentState: 'speaking' }),
      {
        label: 'JARVIS is speaking',
        tone: 'live',
      }
    );
    assert.deepEqual(
      voiceStatus({ isConnected: true, connecting: false, agentState: 'connecting' }),
      {
        label: 'Waiting for JARVIS to join…',
        tone: 'busy',
      }
    );
    assert.deepEqual(voiceStatus({ isConnected: true, connecting: false, agentState: 'failed' }), {
      label: 'JARVIS did not join the call',
      tone: 'error',
    });
  });
});
