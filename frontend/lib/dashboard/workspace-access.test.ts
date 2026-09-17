import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { describe, it } from 'node:test';
import type { MeState } from '../../hooks/useMe.ts';
import { workspaceMode } from './workspace-access.ts';

const source = readFileSync(
  new URL('../../components/dashboard/workspace.tsx', import.meta.url),
  'utf8'
);

describe('workspace access surface', () => {
  it('shows access only for a resolved anonymous visitor when local auth is configured', () => {
    const states: Array<[MeState, ReturnType<typeof workspaceMode>]> = [
      [{ status: 'loading' }, 'loading'],
      [{ status: 'error' }, 'personal'],
      [
        {
          status: 'ready',
          me: {
            configured: true,
            authenticated: false,
            passwordLogin: true,
            mustChangePassword: false,
          },
        },
        'signed-out',
      ],
      [
        {
          status: 'ready',
          me: {
            configured: false,
            authenticated: false,
            passwordLogin: false,
            mustChangePassword: false,
          },
        },
        'personal',
      ],
      [
        {
          status: 'ready',
          me: {
            configured: true,
            authenticated: true,
            passwordLogin: true,
            mustChangePassword: false,
            user: { userId: 'user-1', displayName: 'Cesar', role: 'member' },
          },
        },
        'personal',
      ],
    ];

    for (const [state, expected] of states) assert.equal(workspaceMode(state), expected);
  });

  it('keeps feed hooks behind the personal workspace component boundary', () => {
    assert.match(source, /function PersonalWorkspace\(/);
    assert.match(source, /mode === 'signed-out'/);
    assert.match(source, /<SignedOutAccess\b/);
    assert.match(source, /<PersonalWorkspace\b/);

    const wrapper = source.slice(source.indexOf('export function Workspace('));
    const boundary = wrapper.indexOf('function PersonalWorkspace(');
    assert.ok(boundary > 0, 'personal workspace component must follow the access wrapper');
    const accessWrapper = wrapper.slice(0, boundary);
    for (const hook of [
      'useDashboardCapabilities(',
      'useDashboardFeed(',
      'useDashboardLayout(',
      'useScheduledUpdates(',
      'useWeather(',
    ]) {
      assert.ok(!accessWrapper.includes(hook), `${hook} must not run before access is resolved`);
    }
  });

  it('provides an accessible private-workspace explanation and password sign-in action', () => {
    assert.match(source, /<section[^>]+aria-labelledby=/s);
    assert.match(source, /Your workspace is private/);
    assert.match(source, /href="\/login"/);
    assert.match(source, />\s*Sign in\s*</);
    assert.match(source, /passwordLogin/);
  });
});
