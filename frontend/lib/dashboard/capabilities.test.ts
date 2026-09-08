import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { MISSING_ENDPOINTS, capabilitiesFromSettings } from './capabilities.ts';

const SECRET_PASSWORD = 'hunter2-super-secret';
const SECRET_URL = 'https://caldav.example.com/private/feed.ics';

const SETTINGS_PAYLOAD = {
  settings: {
    agent_name: 'JARVIS',
    calendar_sources: [
      {
        id: 'work',
        provider: 'zoho_caldav',
        display_name: 'Work',
        url: SECRET_URL,
        username: 'cesar@example.com',
        password: SECRET_PASSWORD,
        default: true,
        writable: true,
      },
      { id: 'holidays', provider: 'ics', url: SECRET_URL },
      { provider: 'ics', url: SECRET_URL },
    ],
    reminders_provider: 'apple',
    alarms_enabled: true,
    hass_token: SECRET_PASSWORD,
  },
  prompt_content: 'You are JARVIS',
  custom_prompt_exists: false,
};

describe('dashboard capabilities', () => {
  it('describes calendar sources without any credential or address', () => {
    const caps = capabilitiesFromSettings(SETTINGS_PAYLOAD);

    assert.equal(caps.calendar.configured, true);
    assert.deepEqual(caps.calendar.sources, [
      { id: 'work', label: 'Work', provider: 'zoho_caldav', isDefault: true, writable: true },
      { id: 'holidays', label: 'holidays', provider: 'ics', isDefault: false, writable: false },
    ]);

    const wire = JSON.stringify(caps);
    assert.ok(!wire.includes(SECRET_PASSWORD), 'password must not reach the browser');
    assert.ok(!wire.includes(SECRET_URL), 'source URL must not reach the browser');
    assert.ok(!wire.includes('cesar@example.com'), 'username must not reach the browser');
  });

  it('reports reminders and alarms from the operator settings', () => {
    const caps = capabilitiesFromSettings(SETTINGS_PAYLOAD);

    assert.equal(caps.reminders.configured, true);
    assert.equal(caps.reminders.provider, 'apple');
    assert.equal(caps.alarms.enabled, true);
  });

  it('never invents a weather source and names the endpoint that is missing', () => {
    const caps = capabilitiesFromSettings(SETTINGS_PAYLOAD);

    assert.equal(caps.weather.configured, false);
    assert.equal(caps.weather.missingEndpoint, MISSING_ENDPOINTS.weather);
    assert.equal(caps.calendar.eventsEndpoint, MISSING_ENDPOINTS.calendarEvents);
    assert.equal(caps.reminders.listEndpoint, MISSING_ENDPOINTS.reminders);
    assert.equal(caps.work.tasksEndpoint, MISSING_ENDPOINTS.backgroundTasks);
  });

  it('treats a missing or malformed settings body as nothing configured', () => {
    for (const bad of [null, undefined, 'nope', [], {}, { settings: 'x' }, { settings: {} }]) {
      const caps = capabilitiesFromSettings(bad);
      assert.equal(caps.calendar.configured, false, `calendar for ${JSON.stringify(bad)}`);
      assert.deepEqual(caps.calendar.sources, []);
      assert.equal(caps.reminders.configured, false);
      assert.equal(caps.reminders.provider, null);
      assert.equal(caps.alarms.enabled, false);
    }
    assert.equal(
      capabilitiesFromSettings({ settings: { reminders_provider: 'unknown-thing' } }).reminders
        .configured,
      false
    );
  });
});
