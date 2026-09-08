/**
 * The browser-facing contract for the dashboard's weather.
 *
 * The BFF proxies the backend's per-user weather routes and reduces every
 * answer through these parsers before anything reaches a page, and checks
 * everything a page sends before it reaches the backend. The rules mirror the
 * backend's: a state is one of a fixed vocabulary, a reading is a number that
 * could be true or it is nothing at all, a place name is plain and bounded,
 * and a position is only ever forwarded with an explicit, literal consent.
 *
 * No coordinate the browser reported ever comes back through here: the
 * backend does not return one, and nothing here asks for one. What does come
 * back for a shared position is the name of a nearby place, resolved by the
 * backend; `placeLine` and `placeHint` present it as the approximate place it
 * is, so a wrong one can be replaced with a city chosen by hand.
 *
 * Free of Next.js and React so it can be unit tested.
 */

export const WEATHER_STATES = ['ok', 'stale', 'unavailable', 'no_location'] as const;
export type WeatherState = (typeof WEATHER_STATES)[number];

export const LOCATION_SOURCES = ['city', 'browser'] as const;
export type LocationSource = (typeof LOCATION_SOURCES)[number];

/** Matches the backend's own bounds. */
export const MAX_CITY_QUERY_LENGTH = 80;
export const MAX_CITY_RESULTS = 10;
const MAX_LABEL = 120;
const SHORT_CODE = /^[a-z_]{1,40}$/;
const MAX_TIMEZONE = 64;
const MINUTE = 60;
const HOUR = 3600;

export interface WeatherLocation {
  source: LocationSource;
  label: string;
  region: string | null;
  country: string | null;
  timezone: string | null;
}

export interface WeatherObservation {
  observedAt: string | null;
  temperatureC: number | null;
  apparentTemperatureC: number | null;
  relativeHumidityPct: number | null;
  isDay: boolean | null;
  precipitationMm: number | null;
  rainMm: number | null;
  showersMm: number | null;
  snowfallCm: number | null;
  weatherCode: number | null;
  cloudCoverPct: number | null;
  windSpeedKmh: number | null;
  windDirectionDeg: number | null;
  windGustsKmh: number | null;
}

export interface CurrentWeather {
  state: WeatherState;
  stale: boolean;
  reason: string | null;
  location: WeatherLocation | null;
  observation: WeatherObservation | null;
  conditions: string | null;
  advice: string | null;
  cachedAt: number | null;
  expiresAt: number | null;
  generatedAt: number | null;
}

export interface CityMatch {
  name: string;
  latitude: number;
  longitude: number;
  timezone: string | null;
  country: string | null;
  region: string | null;
}

export interface CitySearch {
  query: string;
  results: CityMatch[];
}

export interface SavedCity {
  name: string;
  region: string | null;
  country: string | null;
  timezone: string | null;
}

export interface BrowserFix {
  updatedAt: number;
  expiresAt: number;
}

export interface WeatherLocationSettings {
  source: LocationSource | null;
  label: string | null;
  city: SavedCity | null;
  browser: BrowserFix | null;
}

// --- bounded values ---------------------------------------------------------------------

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/** Plain, single-line, bounded text; null for anything else or for nothing. */
function text(value: unknown, limit: number): string | null {
  if (typeof value !== 'string') return null;
  const plain = value
    .replace(/\p{Cc}/gu, ' ')
    .replace(/\p{Cf}/gu, '')
    .trim()
    .replace(/\s+/g, ' ');
  return plain ? plain.slice(0, limit) : null;
}

function num(value: unknown, low: number, high: number): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value >= low && value <= high ? value : null;
}

function int(value: unknown, low: number, high: number): number | null {
  const found = num(value, low, high);
  return found !== null && Number.isInteger(found) ? found : null;
}

function flag(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function epoch(value: unknown): number | null {
  return int(value, 0, 4_102_444_800);
}

function shortCode(value: unknown): string | null {
  return typeof value === 'string' && SHORT_CODE.test(value) ? value : null;
}

function timezone(value: unknown): string | null {
  const zone = text(value, MAX_TIMEZONE);
  return zone && /^[A-Za-z0-9_+/-]+$/.test(zone) ? zone : null;
}

// --- what the backend answers -----------------------------------------------------------

function weatherLocation(value: unknown): WeatherLocation | null {
  const row = record(value);
  const source = row?.source;
  const label = text(row?.label, MAX_LABEL);
  if (!row || !label || !(LOCATION_SOURCES as readonly unknown[]).includes(source)) return null;
  return {
    source: source as LocationSource,
    label,
    region: text(row.region, MAX_LABEL),
    country: text(row.country, MAX_LABEL),
    timezone: timezone(row.timezone),
  };
}

function observation(value: unknown): WeatherObservation | null {
  const row = record(value);
  if (!row) return null;
  return {
    observedAt: text(row.observed_at ?? row.observedAt, 32),
    temperatureC: num(row.temperature_c ?? row.temperatureC, -100, 100),
    apparentTemperatureC: num(row.apparent_temperature_c ?? row.apparentTemperatureC, -150, 150),
    relativeHumidityPct: int(row.relative_humidity_pct ?? row.relativeHumidityPct, 0, 100),
    isDay: flag(row.is_day ?? row.isDay),
    precipitationMm: num(row.precipitation_mm ?? row.precipitationMm, 0, 2000),
    rainMm: num(row.rain_mm ?? row.rainMm, 0, 2000),
    showersMm: num(row.showers_mm ?? row.showersMm, 0, 2000),
    snowfallCm: num(row.snowfall_cm ?? row.snowfallCm, 0, 1000),
    weatherCode: int(row.weather_code ?? row.weatherCode, 0, 99),
    cloudCoverPct: int(row.cloud_cover_pct ?? row.cloudCoverPct, 0, 100),
    windSpeedKmh: num(row.wind_speed_kmh ?? row.windSpeedKmh, 0, 600),
    windDirectionDeg: int(row.wind_direction_deg ?? row.windDirectionDeg, 0, 360),
    windGustsKmh: num(row.wind_gusts_kmh ?? row.windGustsKmh, 0, 900),
  };
}

/** The current weather as a page may render it, or null if this is not that. */
export function browserWeather(data: unknown): CurrentWeather | null {
  const row = record(data);
  const state = row?.state;
  if (!row || !(WEATHER_STATES as readonly unknown[]).includes(state)) return null;
  return {
    state: state as WeatherState,
    // Never trusted from the wire on its own: only the state says it.
    stale: state === 'stale',
    reason: shortCode(row.reason),
    location: weatherLocation(row.location),
    observation: observation(row.observation),
    conditions: text(row.conditions, MAX_LABEL),
    advice: text(row.advice, MAX_LABEL),
    cachedAt: epoch(row.cached_at ?? row.cachedAt),
    expiresAt: epoch(row.expires_at ?? row.expiresAt),
    generatedAt: epoch(row.generated_at ?? row.generatedAt),
  };
}

function cityMatch(value: unknown): CityMatch | null {
  const row = record(value);
  const name = text(row?.name, MAX_LABEL);
  const latitude = num(row?.latitude, -90, 90);
  const longitude = num(row?.longitude, -180, 180);
  if (!row || !name || latitude === null || longitude === null) return null;
  return {
    name,
    latitude,
    longitude,
    timezone: timezone(row.timezone),
    country: text(row.country, MAX_LABEL),
    region: text(row.region, MAX_LABEL),
  };
}

export function browserCitySearch(data: unknown): CitySearch | null {
  const row = record(data);
  const query = text(row?.query, MAX_CITY_QUERY_LENGTH);
  if (!row || !query || !Array.isArray(row.results)) return null;
  const results: CityMatch[] = [];
  for (const entry of row.results.slice(0, MAX_CITY_RESULTS)) {
    const city = cityMatch(entry);
    if (city) results.push(city);
  }
  return { query, results };
}

export function browserWeatherLocation(data: unknown): WeatherLocationSettings | null {
  const row = record(data);
  if (!row) return null;
  const source = row.source;
  if (source !== null && !(LOCATION_SOURCES as readonly unknown[]).includes(source)) return null;
  const city = record(row.city);
  const cityName = text(city?.name, MAX_LABEL);
  const browser = record(row.browser);
  const updatedAt = epoch(browser?.updated_at);
  const expiresAt = epoch(browser?.expires_at);
  return {
    source: (source as LocationSource | null) ?? null,
    label: text(row.label, MAX_LABEL),
    city:
      city && cityName
        ? {
            name: cityName,
            region: text(city.region, MAX_LABEL),
            country: text(city.country, MAX_LABEL),
            timezone: timezone(city.timezone),
          }
        : null,
    browser: updatedAt !== null && expiresAt !== null ? { updatedAt, expiresAt } : null,
  };
}

// --- what the browser may send ----------------------------------------------------------

/**
 * The bounded backend query for a city search, or null when the browser asked
 * for something outside what the backend accepts. Refusing here means a
 * malformed search never becomes an upstream lookup.
 */
export function citySearchQuery(params: URLSearchParams): string | null {
  const typed = (params.get('q') ?? '').trim().replace(/\s+/g, ' ');
  if (!typed || typed.length > MAX_CITY_QUERY_LENGTH) return null;
  const raw = params.get('count');
  let count = MAX_CITY_RESULTS;
  if (raw !== null) {
    if (!/^[0-9]{1,3}$/.test(raw)) return null;
    count = Number(raw);
    if (count < 1 || count > MAX_CITY_RESULTS) return null;
  }
  const out = new URLSearchParams();
  out.set('q', typed);
  out.set('count', String(count));
  return out.toString();
}

function coordinate(value: unknown, limit: number): number | null {
  return typeof value === 'number' && Number.isFinite(value) && Math.abs(value) <= limit
    ? value
    : null;
}

export interface CitySelection {
  name: string;
  latitude: number;
  longitude: number;
}

/**
 * A city choice reduced to exactly what the backend takes. Anything else a
 * page attached -- a country, a label -- is dropped rather than forwarded:
 * the backend takes every label from its own lookup, not from the browser.
 */
export function citySelection(body: unknown): CitySelection | null {
  const row = record(body);
  if (typeof row?.name !== 'string' || row.name.length > MAX_CITY_QUERY_LENGTH) return null;
  const name = text(row.name, MAX_CITY_QUERY_LENGTH);
  const latitude = coordinate(row.latitude, 90);
  const longitude = coordinate(row.longitude, 180);
  if (!name || latitude === null || longitude === null) return null;
  return { name, latitude, longitude };
}

export interface ConsentedPosition {
  consent: true;
  latitude: number;
  longitude: number;
}

/**
 * A browser position, forwarded only when the body carries a literal true
 * consent flag.
 *
 * That consent is the user's own decision in the page -- they pressed a
 * button and the browser then asked them -- and nothing weaker stands in for
 * it here.
 */
export function consentedPosition(body: unknown): ConsentedPosition | null {
  const row = record(body);
  if (row?.consent !== true) return null;
  const latitude = coordinate(row.latitude, 90);
  const longitude = coordinate(row.longitude, 180);
  if (latitude === null || longitude === null) return null;
  return { consent: true, latitude, longitude };
}

// --- what the widget shows --------------------------------------------------------------

const DEGREE = String.fromCharCode(176);
const DOT = ' ' + String.fromCharCode(183) + ' ';

/** A whole degree, or two dashes. Never a rounded nothing. */
export function temperatureLabel(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '--';
  // -0.4 reads as 0, never as minus zero.
  const whole = Math.round(value);
  return (whole === 0 ? 0 : whole) + DEGREE;
}

/** The practical detail line: only the parts that were actually measured. */
export function conditionsLine(reading: WeatherObservation | null | undefined): string {
  if (!reading) return '';
  const parts: string[] = [];
  if (reading.apparentTemperatureC !== null) {
    parts.push('Feels ' + temperatureLabel(reading.apparentTemperatureC));
  }
  if (reading.relativeHumidityPct !== null) {
    parts.push(reading.relativeHumidityPct + '% humidity');
  }
  if (reading.windSpeedKmh !== null) {
    parts.push('wind ' + Math.round(reading.windSpeedKmh) + ' km/h');
  }
  if (reading.windGustsKmh !== null) {
    parts.push('gusts ' + Math.round(reading.windGustsKmh) + ' km/h');
  }
  return parts.join(DOT);
}

/**
 * The place a reading came from, as one line: the nearby place, then its
 * region and country when they say something the name did not.
 *
 * For a shared browser position the backend names the nearest place it could
 * resolve, and falls back to a generic label when it could not; either way the
 * line is exactly what the backend said, never something assembled here from a
 * coordinate.
 */
export function placeLine(location: WeatherLocation | null | undefined): string {
  if (!location) return '';
  const label = location.label;
  const parts = [label];
  if (location.region && location.region !== label) parts.push(' - ' + location.region);
  if (location.country && location.country !== label && location.country !== location.region) {
    parts.push(', ' + location.country);
  }
  return parts.join('');
}

/**
 * Why a place might be wrong, when it might be: a browser position is
 * approximate and can be resolved to the wrong nearby place, so the widget
 * says so and the user can set a city by hand instead. A city the user chose
 * needs no such warning.
 */
export function placeHint(location: WeatherLocation | null | undefined): string | null {
  if (!location || location.source !== 'browser') return null;
  return 'Approximate, from the location this browser shared';
}

function agoLabel(seconds: number): string {
  if (seconds < MINUTE) return 'just now';
  if (seconds < HOUR) return Math.floor(seconds / MINUTE) + ' min ago';
  return Math.floor(seconds / HOUR) + ' h ago';
}

/** How old this reading is, and -- when it is stale -- why it is not newer. */
export function freshnessLabel(weather: CurrentWeather, now: Date): string {
  if (weather.cachedAt === null) return 'No reading yet';
  const seconds = Math.max(0, Math.floor(now.getTime() / 1000) - weather.cachedAt);
  if (weather.stale) {
    return 'Last reading ' + agoLabel(seconds) + '; the weather service could not be reached';
  }
  return 'Updated ' + agoLabel(seconds);
}
