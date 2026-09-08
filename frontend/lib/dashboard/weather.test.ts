import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  MAX_CITY_QUERY_LENGTH,
  MAX_CITY_RESULTS,
  browserCitySearch,
  browserWeather,
  browserWeatherLocation,
  citySearchQuery,
  citySelection,
  conditionsLine,
  consentedPosition,
  freshnessLabel,
  placeHint,
  placeLine,
  temperatureLabel,
} from './weather.ts';

/**
 * The browser's whole view of the weather slice. Everything a page renders
 * passes through here first: a payload that is not exactly the backend's
 * shape becomes null rather than a half-parsed object, and nothing a browser
 * sends reaches the backend without being checked here as well.
 */

const NEWLINE = String.fromCharCode(10);

const OBSERVATION = {
  observed_at: '2023-11-14T23:00',
  temperature_c: 7.4,
  apparent_temperature_c: 4.2,
  relative_humidity_pct: 81,
  is_day: false,
  precipitation_mm: 0.1,
  rain_mm: 0.1,
  showers_mm: 0,
  snowfall_cm: 0,
  weather_code: 61,
  cloud_cover_pct: 100,
  wind_speed_kmh: 14.4,
  wind_direction_deg: 230,
  wind_gusts_kmh: 31,
};

const WEATHER = {
  generated_at: 1_700_000_000,
  state: 'ok',
  stale: false,
  reason: null,
  location: {
    source: 'city',
    label: 'Paris',
    region: 'Ile-de-France',
    country: 'France',
    timezone: 'Europe/Paris',
  },
  observation: OBSERVATION,
  conditions: 'Slight rain',
  advice: 'Rain about: take a coat.',
  cached_at: 1_700_000_000,
  expires_at: 1_700_003_600,
};

describe('current weather parser', () => {
  it('reduces the backend payload to the shape a page renders', () => {
    const weather = browserWeather(WEATHER);
    assert.ok(weather);
    assert.equal(weather.state, 'ok');
    assert.equal(weather.stale, false);
    assert.equal(weather.location?.source, 'city');
    assert.equal(weather.location?.label, 'Paris');
    assert.equal(weather.location?.region, 'Ile-de-France');
    assert.equal(weather.observation?.temperatureC, 7.4);
    assert.equal(weather.observation?.apparentTemperatureC, 4.2);
    assert.equal(weather.observation?.relativeHumidityPct, 81);
    assert.equal(weather.observation?.isDay, false);
    assert.equal(weather.observation?.windGustsKmh, 31);
    assert.equal(weather.conditions, 'Slight rain');
    assert.equal(weather.advice, 'Rain about: take a coat.');
    assert.equal(weather.cachedAt, 1_700_000_000);
    assert.equal(weather.expiresAt, 1_700_003_600);
  });

  it('preserves a BFF-reduced reading when the client validates it again', () => {
    const bff = browserWeather(WEATHER);
    assert.ok(bff);
    const browser = browserWeather(bff);
    assert.equal(browser?.observation?.temperatureC, 7.4);
    assert.equal(browser?.observation?.weatherCode, 61);
    assert.equal(browser?.cachedAt, 1_700_000_000);
  });

  it('refuses anything that is not the backend answering', () => {
    for (const payload of [null, undefined, 42, 'ok', [], {}, { state: 'sunny' }]) {
      assert.equal(browserWeather(payload), null, String(payload));
    }
  });

  it('keeps every honest state and never invents a reading for one', () => {
    for (const state of ['no_location', 'unavailable']) {
      const weather = browserWeather({
        ...WEATHER,
        state,
        observation: null,
        location: state === 'no_location' ? null : WEATHER.location,
        conditions: null,
        advice: null,
        cached_at: null,
        expires_at: null,
        reason: state === 'unavailable' ? 'transport' : null,
      });
      assert.ok(weather, state);
      assert.equal(weather.state, state);
      assert.equal(weather.observation, null);
      assert.equal(weather.stale, false);
    }
    const stale = browserWeather({ ...WEATHER, state: 'stale', stale: true, reason: 'transport' });
    assert.equal(stale?.state, 'stale');
    assert.equal(stale?.stale, true);
    assert.equal(stale?.reason, 'transport');
  });

  it('drops a reading the backend could not have meant', () => {
    const weather = browserWeather({
      ...WEATHER,
      observation: {
        ...OBSERVATION,
        temperature_c: 'warm',
        relative_humidity_pct: 180,
        wind_speed_kmh: -3,
        wind_direction_deg: 900,
        cloud_cover_pct: null,
        is_day: 1,
      },
    });
    assert.ok(weather);
    assert.equal(weather.observation?.temperatureC, null);
    assert.equal(weather.observation?.relativeHumidityPct, null);
    assert.equal(weather.observation?.windSpeedKmh, null);
    assert.equal(weather.observation?.windDirectionDeg, null);
    assert.equal(weather.observation?.cloudCoverPct, null);
    assert.equal(weather.observation?.isDay, null);
  });

  it('never accepts a location source or a label it did not expect', () => {
    assert.equal(
      browserWeather({ ...WEATHER, location: { ...WEATHER.location, source: 'gps' } })?.location,
      null
    );
    const long = browserWeather({
      ...WEATHER,
      location: { ...WEATHER.location, label: 'x'.repeat(400) },
    });
    assert.ok((long?.location?.label.length ?? 0) <= 120);
    const controls = browserWeather({
      ...WEATHER,
      location: { ...WEATHER.location, label: 'Pa ris' + NEWLINE },
    });
    assert.equal(controls?.location?.label, 'Pa ris');
  });
});


describe('city search parser', () => {
  const ROW = {
    name: 'Paris',
    latitude: 48.85,
    longitude: 2.35,
    timezone: 'Europe/Paris',
    country: 'France',
    region: 'Ile-de-France',
  };

  it('keeps the places the backend named and nothing else', () => {
    const search = browserCitySearch({ query: 'Paris', results: [ROW] });
    assert.ok(search);
    assert.equal(search.query, 'Paris');
    assert.deepEqual(search.results, [
      {
        name: 'Paris',
        latitude: 48.85,
        longitude: 2.35,
        timezone: 'Europe/Paris',
        country: 'France',
        region: 'Ile-de-France',
      },
    ]);
  });

  it('drops a place that could not be one', () => {
    const search = browserCitySearch({
      query: 'x',
      results: [
        { ...ROW, latitude: 91 },
        { ...ROW, longitude: 'east' },
        { ...ROW, name: '' },
        null,
        ROW,
      ],
    });
    assert.equal(search?.results.length, 1);
  });

  it('never renders more places than the bound', () => {
    const search = browserCitySearch({
      query: 'p',
      results: Array.from({ length: 50 }, () => ROW),
    });
    assert.equal(search?.results.length, MAX_CITY_RESULTS);
  });

  it('refuses a payload that is not a search answer', () => {
    for (const payload of [null, 7, [], { results: [ROW] }, { query: 'p' }]) {
      assert.equal(browserCitySearch(payload), null, String(payload));
    }
  });
});

describe('weather location settings parser', () => {
  it('reads the saved city and the browser fix without any position', () => {
    const settings = browserWeatherLocation({
      source: 'city',
      label: 'Paris',
      city: { name: 'Paris', region: 'Ile-de-France', country: 'France', timezone: 'Europe/Paris' },
      browser: null,
    });
    assert.ok(settings);
    assert.equal(settings.source, 'city');
    assert.equal(settings.city?.name, 'Paris');
    assert.equal(settings.browser, null);

    const browser = browserWeatherLocation({
      source: 'browser',
      label: 'Your current location',
      city: null,
      browser: { updated_at: 1_700_000_000, expires_at: 1_700_021_600 },
    });
    assert.equal(browser?.browser?.expiresAt, 1_700_021_600);
    assert.equal(browser?.city, null);
  });

  it('reads the empty state', () => {
    const settings = browserWeatherLocation({
      source: null,
      label: null,
      city: null,
      browser: null,
    });
    assert.ok(settings);
    assert.equal(settings.source, null);
  });

  it('refuses anything else', () => {
    for (const payload of [null, 'city', [], { source: 'gps', label: null, city: null, browser: null }]) {
      assert.equal(browserWeatherLocation(payload), null, String(payload));
    }
  });
});

describe('what the browser may send', () => {
  it('bounds a city query before it becomes a backend request', () => {
    assert.equal(citySearchQuery(new URLSearchParams('q=  Paris  ')), 'q=Paris&count=10');
    assert.equal(citySearchQuery(new URLSearchParams('q=Paris&count=3')), 'q=Paris&count=3');
    for (const raw of [
      '',
      'q=',
      'q=%20%20',
      'q=' + 'x'.repeat(MAX_CITY_QUERY_LENGTH + 1),
      'q=Paris&count=0',
      'q=Paris&count=99',
      'q=Paris&count=two',
    ]) {
      assert.equal(citySearchQuery(new URLSearchParams(raw)), null, raw);
    }
  });

  it('accepts a city selection only in the exact shape the backend takes', () => {
    assert.deepEqual(citySelection({ name: 'Paris', latitude: 48.85, longitude: 2.35 }), {
      name: 'Paris',
      latitude: 48.85,
      longitude: 2.35,
    });
    // Anything else the page might have attached is dropped, not forwarded.
    assert.deepEqual(
      citySelection({ name: 'Paris', latitude: 48.85, longitude: 2.35, country: 'Atlantis' }),
      { name: 'Paris', latitude: 48.85, longitude: 2.35 }
    );
    for (const body of [
      null,
      {},
      { name: '', latitude: 1, longitude: 1 },
      { name: 'Paris', latitude: 91, longitude: 1 },
      { name: 'Paris', latitude: 1, longitude: 181 },
      { name: 'Paris', latitude: '1', longitude: 1 },
      { name: 'Paris', latitude: Number.NaN, longitude: 1 },
      { name: 'x'.repeat(MAX_CITY_QUERY_LENGTH + 1), latitude: 1, longitude: 1 },
    ]) {
      assert.equal(citySelection(body), null, JSON.stringify(body));
    }
  });

  it('forwards a position only with an explicit, literal consent', () => {
    assert.deepEqual(consentedPosition({ consent: true, latitude: 51.5, longitude: -0.13 }), {
      consent: true,
      latitude: 51.5,
      longitude: -0.13,
    });
    for (const body of [
      { latitude: 51.5, longitude: -0.13 },
      { consent: false, latitude: 51.5, longitude: -0.13 },
      { consent: 'true', latitude: 51.5, longitude: -0.13 },
      { consent: 1, latitude: 51.5, longitude: -0.13 },
      { consent: true, latitude: 91, longitude: 0 },
      { consent: true, latitude: 0, longitude: 181 },
      { consent: true, latitude: Number.POSITIVE_INFINITY, longitude: 0 },
      { consent: true, latitude: null, longitude: 0 },
    ]) {
      assert.equal(consentedPosition(body), null, JSON.stringify(body));
    }
  });
});

describe('what the widget shows', () => {
  it('reads a temperature as a whole degree, and nothing as nothing', () => {
    assert.equal(temperatureLabel(7.4), '7°');
    assert.equal(temperatureLabel(-0.4), '0°');
    assert.equal(temperatureLabel(null), '--');
  });

  it('says how the conditions read without inventing any of them', () => {
    assert.equal(
      conditionsLine(browserWeather(WEATHER)!.observation),
      'Feels 4° · 81% humidity · wind 14 km/h · gusts 31 km/h'
    );
    const bare = conditionsLine({
      ...browserWeather(WEATHER)!.observation!,
      apparentTemperatureC: null,
      relativeHumidityPct: null,
      windSpeedKmh: null,
      windGustsKmh: null,
    });
    assert.equal(bare, '');
    assert.equal(conditionsLine(null), '');
  });

  it('says truthfully how old a reading is', () => {
    const at = 1_700_000_000;
    const weather = browserWeather({ ...WEATHER, cached_at: at })!;
    assert.equal(freshnessLabel(weather, new Date(at * 1000)), 'Updated just now');
    assert.equal(freshnessLabel(weather, new Date((at + 300) * 1000)), 'Updated 5 min ago');
    assert.equal(freshnessLabel(weather, new Date((at + 7200) * 1000)), 'Updated 2 h ago');
    const stale = browserWeather({ ...WEATHER, state: 'stale', stale: true, cached_at: at })!;
    assert.equal(
      freshnessLabel(stale, new Date((at + 7200) * 1000)),
      'Last reading 2 h ago; the weather service could not be reached'
    );
    const none = browserWeather({ ...WEATHER, state: 'unavailable', cached_at: null })!;
    assert.equal(freshnessLabel(none, new Date(at * 1000)), 'No reading yet');
  });
});

describe('the place a reading came from', () => {
  const named = {
    source: 'browser' as const,
    label: 'London',
    region: 'England',
    country: 'United Kingdom',
    timezone: 'Europe/London',
  };

  it('reads as the nearby place, its region and its country', () => {
    assert.equal(placeLine(named), 'London - England, United Kingdom');
  });

  it('is nothing at all when there is no place', () => {
    assert.equal(placeLine(null), '');
    assert.equal(placeLine(undefined), '');
  });

  it('shows a place that has only a name', () => {
    assert.equal(placeLine({ ...named, region: null, country: null }), 'London');
    assert.equal(placeLine({ ...named, region: null }), 'London, United Kingdom');
  });

  it('never repeats one name as its own region or country', () => {
    assert.equal(
      placeLine({ ...named, region: 'London', country: 'United Kingdom' }),
      'London, United Kingdom'
    );
    assert.equal(
      placeLine({ ...named, region: 'Singapore', country: 'Singapore', label: 'Singapore' }),
      'Singapore'
    );
  });

  it('says a browser place is approximate, so a wrong one can be seen and replaced', () => {
    const hint = placeHint(named);
    assert.ok(hint && hint.toLowerCase().includes('browser'));
    // A place the user chose by hand needs no such warning.
    assert.equal(placeHint({ ...named, source: 'city', label: 'Paris' }), null);
    assert.equal(placeHint(null), null);
  });

  it('says the same of a browser place the backend could not name', () => {
    const generic = { ...named, label: 'Your current location', region: null, country: null };
    assert.equal(placeLine(generic), 'Your current location');
    assert.ok(placeHint(generic));
  });
});
