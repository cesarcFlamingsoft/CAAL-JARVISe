import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

/**
 * Structural guard for the weather slice: the four BFF routes, the hook, the
 * widget and the settings section. These files import Next.js and React, so
 * they cannot run under node --test; what can be checked is that they keep
 * the identity BFF posture, that the browser never speaks to Open-Meteo
 * itself, that a position is only ever read after the user asks for it, and
 * that the widget stays honest about how old a reading is.
 */

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = (path: string) => readFileSync(join(ROOT, path), 'utf8');

const READ_ROUTES = {
  current: 'app/api/dashboard/weather/route.ts',
  cities: 'app/api/dashboard/weather/cities/route.ts',
  location: 'app/api/dashboard/weather/location/route.ts',
};
const WRITE_ROUTES = {
  city: 'app/api/dashboard/weather/city/route.ts',
  position: 'app/api/dashboard/weather/browser-location/route.ts',
};
const HOOK = 'hooks/useWeather.ts';
const WIDGET = 'components/dashboard/widgets/weather-widget.tsx';
const SETTINGS = 'components/settings/weather-location.tsx';
const PANEL = 'components/settings/settings-panel.tsx';
const PARSER = 'lib/dashboard/weather.ts';
const SESSION_READS = ['useSessionContext', 'useVoiceAssistant', 'useRoomContext', 'isConnected'];

const ALL = [
  ...Object.values(READ_ROUTES),
  ...Object.values(WRITE_ROUTES),
  HOOK,
  WIDGET,
  SETTINGS,
  PARSER,
];

describe('the weather slice never leaves this deployment', () => {
  it('exists', () => {
    for (const path of ALL) assert.ok(existsSync(join(ROOT, path)), path + ' is missing');
  });

  it('never names Open-Meteo outside the backend', () => {
    for (const path of ALL) {
      const source = read(path);
      assert.ok(!/open-meteo/i.test(source), path + ' must not reach Open-Meteo directly');
      // The reverse lookup that names a shared position is a backend errand
      // too: the browser must never send its own position to Nominatim.
      assert.ok(!/nominatim|openstreetmap/i.test(source), path + ' must not reach Nominatim');
      assert.ok(!/https?:\/\/(?!localhost)/i.test(source), path + ' must not name an absolute URL');
    }
  });
});

describe('weather BFF routes', () => {
  it('authenticate the browser session and proxy as that user only', () => {
    for (const [key, path] of Object.entries({ ...READ_ROUTES, ...WRITE_ROUTES })) {
      const source = read(path);
      assert.match(source, /requireUser\(req/, key + ' must use requireUser');
      assert.match(
        source,
        /callAsUser\(\s*auth\.config,\s*auth\.user\.userId,\s*[`'"]\/users\/me\/weather/,
        key + ' must call the backend as the session user'
      );
      assert.match(source, /export const dynamic = 'force-dynamic'/, key);
      assert.ok(!source.includes('/admin/'), key);
    }
  });

  it('keeps the read routes read-only and bounded', () => {
    for (const [key, path] of Object.entries(READ_ROUTES)) {
      const source = read(path);
      assert.match(source, /export async function GET\(/, key);
      assert.ok(!/export async function (POST|PATCH|PUT|DELETE)\(/.test(source), key);
      assert.match(source, /noStoreJson\(/, key);
    }
    assert.match(read(READ_ROUTES.current), /browserWeather\(/);
    assert.match(read(READ_ROUTES.cities), /browserCitySearch\(/);
    assert.match(read(READ_ROUTES.location), /browserWeatherLocation\(/);
    // A search the browser mistyped never becomes an upstream lookup.
    assert.match(read(READ_ROUTES.cities), /citySearchQuery\(/);
  });

  it('guards every mutation with origin, CSRF and the per-user budget', () => {
    for (const [key, path] of Object.entries(WRITE_ROUTES)) {
      const source = read(path);
      assert.match(source, /guardMutation\(req, auth\.config, auth\.user\.userId\)/, key);
      assert.match(source, /export async function PUT\(/, key);
      assert.match(source, /export async function DELETE\(/, key);
      assert.ok(!/export async function GET\(/.test(source), key + ' must not read');
      assert.match(source, /readJsonObject\(req\)/, key);
      assert.match(source, /browserWeatherLocation\(/, key);
    }
    // Only the reduced shapes reach the backend.
    assert.match(read(WRITE_ROUTES.city), /citySelection\(/);
    assert.match(read(WRITE_ROUTES.position), /consentedPosition\(/);
  });
});

describe('weather hook', () => {
  it('reads the BFF with the session cookie and never caches', () => {
    const source = read(HOOK);
    assert.match(source, /'\/api\/dashboard\/weather'/);
    assert.match(source, /credentials: 'same-origin'/);
    assert.match(source, /cache: 'no-store'/);
    assert.match(source, /browserWeather/);
    // A saved city or a shared position must show up without a reload.
    assert.match(source, /'weather-location-updated'/);
    for (const name of SESSION_READS) assert.ok(!source.includes(name), name);
  });
});

const HONEST_STATES = ['no_location', 'unavailable', 'stale'];
const EFFECT_END = String.fromCharCode(125) + ', [';

describe('weather widget', () => {
  it('shows only parsed readings, says how old they are, and never gates on the call', () => {
    const source = read(WIDGET);
    assert.match(source, /freshnessLabel\(/, 'the widget must date its reading');
    assert.match(source, /conditionsLine\(/, 'the widget must show the practical detail');
    assert.match(source, /temperatureLabel\(/);
    // Every honest state is rendered by name rather than as an empty box.
    for (const state of HONEST_STATES) {
      assert.ok(source.includes(state), 'the widget must handle ' + state);
    }
    assert.match(source, /WidgetLoading/);
    assert.match(source, /WidgetSignIn/);
    for (const name of SESSION_READS) assert.ok(!source.includes(name), name);
    // Nothing is fabricated when there is no reading.
    assert.ok(!/Math\.random/.test(source));
  });

  it('names the place a reading came from, and says when it is only approximate', () => {
    const source = read(WIDGET);
    // The place line is the parsed one, never assembled from a coordinate.
    assert.match(source, /placeLine\(/, 'the widget must show where the reading came from');
    assert.match(source, /placeHint\(/, 'a shared browser position must be marked approximate');
    assert.ok(!/latitude|longitude|coords/i.test(source), 'the widget must never see a position');
    // A wrong place must be replaceable from the widget itself.
    assert.match(source, /onOpenSettings/);
  });
});

describe('weather settings', () => {
  const source = read(SETTINGS);

  it('asks the browser for a position only when the user presses the button', () => {
    const geolocation = source.indexOf('navigator.geolocation');
    const handler = source.indexOf('async function shareLocation(');
    assert.ok(handler > 0, 'the position must be read inside its own handler');
    assert.ok(geolocation > handler, 'the position must not be read anywhere else');
    assert.equal(source.split('navigator.geolocation').length - 1, 1);
    assert.match(source, /onClick=... => void shareLocation...$/m);
    // Never on mount, never as a side effect of rendering.
    for (const chunk of source.split('useEffect(').slice(1)) {
      const body = chunk.slice(0, Math.max(0, chunk.indexOf(EFFECT_END)));
      assert.ok(!body.includes('geolocation'), 'no effect may read the position');
    }
  });

  it('says what the position is for before asking for it', () => {
    assert.match(source, /used only to show your weather/i);
    // The short life of a shared position is stated, not hidden.
    assert.match(source, /expires/i);
  });

  it('does not ask for more precision than a forecast needs', () => {
    assert.match(source, /enableHighAccuracy: false/);
  });

  it('offers the whole choice: search a city, pick one, or clear it', () => {
    assert.match(source, /.\/api\/dashboard\/weather\/cities/);
    assert.match(source, /'PUT'/);
    assert.match(source, /'DELETE'/);
    assert.match(source, /latitude/);
    // Typing must not become a lookup per keystroke.
    assert.match(source, /DEBOUNCE_MS/);
    assert.match(source, /disabled=/, 'controls must be disabled while they cannot be used');
    assert.match(source, /weather-location-updated/);
  });

  it('is offered from the settings panel', () => {
    const panel = read(PANEL);
    assert.match(panel, /WeatherLocation/);
    assert.match(panel, /components\/settings\/weather-location/);
  });
});
