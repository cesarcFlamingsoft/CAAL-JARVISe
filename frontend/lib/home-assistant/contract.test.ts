import assert from 'node:assert/strict';
import test from 'node:test';
import { parseAccess } from './contract.ts';

test('backend grant payload reaches browser without secrets or invented identities', () => {
  const data = {
    enabled: true,
    status: 'connection_required',
    connection_id: null,
    scope: 'states_and_lights',
    can_connect: false,
    connections: [
      {
        id: 'ha_' + 'a'.repeat(24),
        label: 'Authenticated HA user',
        shared: true,
        ha_admin: false,
        access_token: 'private',
      },
    ],
    refresh_token: 'private',
  };
  const parsed = parseAccess(data);
  assert.equal(parsed?.status, 'connection_required');
  assert.equal(parsed?.connections.length, 1);
  assert.ok(!JSON.stringify(parsed).includes('private'));
  assert.equal(parseAccess({ ...data, status: 'invented' }), null);
  assert.equal(parseAccess({ ...data, enabled: 'true' }), null);
});

test('HA authorization must return to the browser origin that started it', async () => {
  const { authorizationOriginAllowed } = await import('./contract.ts');
  assert.equal(
    authorizationOriginAllowed('http://localhost:3000', 'https://jarvis.example.com'),
    false
  );
  assert.equal(authorizationOriginAllowed(null, 'https://jarvis.example.com'), false);
  assert.equal(
    authorizationOriginAllowed('https://jarvis.example.com', 'https://jarvis.example.com'),
    true
  );
});
