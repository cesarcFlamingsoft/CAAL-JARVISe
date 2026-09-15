import assert from 'node:assert/strict';
import test from 'node:test';
import { parseEnrollment, parseStatus } from './contract.ts';

test('accept only restricted enrollment and opaque credentials', () => {
  assert.equal(
    parseEnrollment({
      id: 'sat_' + 'a'.repeat(24),
      credential: 's'.repeat(43),
      satellite_id: 'assist_satellite.home_assistant_voice_0a3d6b_assist_satellite',
      device_id: '0bf018dfe200b28d8f7cff95e8d2aa75',
    })?.credential.length,
    43
  );
  assert.equal(parseEnrollment({ id: 'person', credential: 'secret' }), null);
  assert.equal(
    parseStatus({
      enrolled: true,
      id: 'sat_' + 'a'.repeat(24),
      personal_data: true,
      device_actions: false,
    }),
    null
  );
  assert.equal(
    parseStatus({ enrolled: false, id: null, personal_data: false, device_actions: false }),
    null
  );
});

test('accept configurable device identities and enrollment inventory', () => {
  const item = {
    id: 'sat_' + 'a'.repeat(24),
    credential: 's'.repeat(43),
    satellite_id: 'assist_satellite.living',
    device_id: 'b'.repeat(32),
  };
  assert.equal(parseEnrollment(item)?.satellite_id, item.satellite_id);
  assert.equal(parseEnrollment({ ...item, satellite_id: 'light.desk' }), null);
  const status = {
    personal_data: false,
    devices: [
      {
        satellite_id: item.satellite_id,
        device_id: item.device_id,
        name: 'Living',
        enrollment: null,
      },
    ],
    enrollments: [],
    connections: [],
    connection_id: null,
    provider_identity: null,
    error: null,
  };
  assert.equal(parseStatus(status)?.devices.length, 1);
  assert.equal(parseStatus({ ...status, personal_data: true }), null);
});
