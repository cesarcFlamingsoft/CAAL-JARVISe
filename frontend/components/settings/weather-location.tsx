'use client';

/**
 * Settings -- Integrations -- Weather location.
 *
 * Two ways to say where the weather should come from, and nothing in between:
 * this browser can offer its position, once, after the button below is
 * pressed; or a city can be searched for by name and chosen from what the
 * lookup returned. A chosen city wins while it is set, and clearing it falls
 * back to the shared position.
 *
 * The position is never read on load, on render, or as a side effect: the
 * browser asks the person for permission only because they pressed the
 * button, and the panel says beforehand what it is for and how long it is
 * kept. Typing a city name is debounced, so a search is one lookup rather
 * than one per keystroke.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { CircleNotch, Crosshair, MagnifyingGlass } from '@phosphor-icons/react/dist/ssr';
import { apiRequest, explain } from '@/components/account/api-client';
import { Button } from '@/components/livekit/button';
import type { CityMatch, CitySearch, WeatherLocationSettings } from '@/lib/dashboard/weather';

/** One lookup per pause in typing, never one per keystroke. */
const DEBOUNCE_MS = 400;
const MIN_QUERY = 2;

interface Notice {
  tone: 'error' | 'info';
  text: string;
}

function announce() {
  window.dispatchEvent(new Event('weather-location-updated'));
}

function whenLabel(seconds: number): string {
  try {
    return new Date(seconds * 1000).toLocaleString();
  } catch {
    return 'soon';
  }
}

function placeLabel(city: CityMatch): string {
  return [city.name, city.region, city.country].filter(Boolean).join(', ');
}

export function WeatherLocation() {
  const [settings, setSettings] = useState<WeatherLocationSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<CityMatch[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [locating, setLocating] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const latest = useRef(0);

  const load = useCallback(async () => {
    setLoading(true);
    const result = await apiRequest<WeatherLocationSettings>('/api/dashboard/weather/location');
    setLoading(false);
    if (result.ok) {
      setSettings(result.data);
      setLoadError(null);
    } else {
      setSettings(null);
      setLoadError(explain(result.error));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Search only after typing pauses, and only for a query worth sending.
  useEffect(() => {
    const typed = query.trim();
    if (typed.length < MIN_QUERY) {
      setResults(null);
      setSearchError(null);
      return;
    }
    const ticket = latest.current + 1;
    latest.current = ticket;
    const timer = window.setTimeout(async () => {
      setSearching(true);
      const result = await apiRequest<CitySearch>(
        '/api/dashboard/weather/cities?q=' + encodeURIComponent(typed)
      );
      if (latest.current !== ticket) return;
      setSearching(false);
      if (result.ok) {
        setResults(result.data.results);
        setSearchError(null);
      } else {
        setResults(null);
        setSearchError(explain(result.error));
      }
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [query]);

  async function apply(
    path: string,
    method: 'PUT' | 'DELETE',
    body: unknown,
    done: string
  ): Promise<boolean> {
    setBusy(true);
    setNotice(null);
    const result = await apiRequest<WeatherLocationSettings>(path, { method, body });
    setBusy(false);
    if (!result.ok) {
      setNotice({ tone: 'error', text: explain(result.error) });
      return false;
    }
    setSettings(result.data);
    setNotice({ tone: 'info', text: done });
    announce();
    return true;
  }

  async function chooseCity(city: CityMatch) {
    const saved = await apply(
      '/api/dashboard/weather/city',
      'PUT',
      { name: city.name, latitude: city.latitude, longitude: city.longitude },
      placeLabel(city) + ' is now the place your weather is read for.'
    );
    if (saved) {
      setQuery('');
      setResults(null);
    }
  }

  async function clearCity() {
    await apply(
      '/api/dashboard/weather/city',
      'DELETE',
      undefined,
      'The chosen city was cleared.'
    );
  }

  async function forgetPosition() {
    await apply(
      '/api/dashboard/weather/browser-location',
      'DELETE',
      undefined,
      'The shared location was forgotten.'
    );
  }

  /**
   * Ask this browser for a position, once, because the button was pressed.
   *
   * The browser puts its own permission prompt in front of this; a refusal is
   * reported as a refusal and nothing is sent. Coarse accuracy is asked for
   * deliberately: the backend rounds the position to about a kilometre, so
   * anything finer would be precision nobody uses.
   */
  async function shareLocation() {
    if (locating || busy) return;
    setNotice(null);
    const geo = typeof navigator === 'undefined' ? null : navigator.geolocation;
    if (!geo) {
      setNotice({ tone: 'error', text: 'This browser cannot offer a location.' });
      return;
    }
    setLocating(true);
    const fix = await new Promise<GeolocationPosition | null>((resolve) => {
      geo.getCurrentPosition(
        (position) => resolve(position),
        () => resolve(null),
        { enableHighAccuracy: false, timeout: 10_000, maximumAge: 300_000 }
      );
    });
    setLocating(false);
    if (!fix) {
      setNotice({
        tone: 'error',
        text: 'This browser did not share a location. You can still choose a city by name.',
      });
      return;
    }
    await apply(
      '/api/dashboard/weather/browser-location',
      'PUT',
      {
        consent: true,
        latitude: fix.coords.latitude,
        longitude: fix.coords.longitude,
      },
      'This browser location is now used for your weather.'
    );
  }

  const city = settings?.city ?? null;
  const browserFix = settings?.browser ?? null;
  const working = busy || locating;

  return (
    <div className="overflow-hidden rounded-xl border">
      <div className="bg-muted/50 border-b px-4 py-3">
        <span className="font-semibold">Weather location</span>
        <p className="text-muted-foreground text-xs">
          Where the dashboard reads your weather for. Either this browser can offer its position,
          or you can pick a city anywhere in the world. Nothing is read until you choose one.
        </p>
      </div>

      <div className="space-y-4 p-4">
        {loading && !settings && (
          <p className="text-muted-foreground flex items-center gap-2 text-sm">
            <CircleNotch className="h-4 w-4 animate-spin" aria-hidden />
            Loading your weather location...
          </p>
        )}
        {loadError && !settings && <p className="text-destructive text-sm">{loadError}</p>}

        {settings && (
          <div className="bg-muted/30 rounded-lg border px-3 py-2">
            <p className="text-sm font-medium">
              {city
                ? 'Using the city you chose: ' + city.name
                : browserFix
                  ? 'Using the location this browser shared'
                  : 'No location set yet'}
            </p>
            {city && (
              <p className="text-muted-foreground mt-0.5 text-xs">
                {[city.region, city.country, city.timezone].filter(Boolean).join(' - ')}
              </p>
            )}
            {browserFix && !city && (
              <p className="text-muted-foreground mt-0.5 text-xs">
                Kept only until it expires at {whenLabel(browserFix.expiresAt)}, then the dashboard
                asks again.
              </p>
            )}
            <div className="mt-2 flex flex-wrap gap-2">
              {city && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => void clearCity()}
                  disabled={working}
                >
                  Clear city
                </Button>
              )}
              {browserFix && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => void forgetPosition()}
                  disabled={working}
                >
                  Forget shared location
                </Button>
              )}
            </div>
          </div>
        )}

        <div className="space-y-2 rounded-lg border p-3">
          <p className="text-sm font-medium">Use this browser location</p>
          <p className="text-muted-foreground text-xs">
            Pressing the button asks this browser for your position.
            {' '}
            It is used only to show your weather.
            {' '}
            It is rounded to about a kilometre before it is stored, and it is never shown back to
            you or written to a log: this server looks up the name of a nearby place and the weather
            widget shows that instead, so you can tell when the location is wrong and pick a city
            here instead. It expires within hours unless you share it again.
          </p>
          <Button
            variant="primary"
            size="sm"
            onClick={() => void shareLocation()}
            disabled={working}
          >
            {locating ? (
              <>
                <CircleNotch className="h-4 w-4 animate-spin" aria-hidden />
                Asking this browser...
              </>
            ) : (
              <>
                <Crosshair className="h-4 w-4" aria-hidden />
                Use current location
              </>
            )}
          </Button>
        </div>

        <div className="space-y-2 rounded-lg border p-3">
          <label htmlFor="weather-city" className="text-sm font-medium">
            Or choose a city
          </label>
          <p className="text-muted-foreground text-xs">
            Type a city name and pick one of the matches. A city is remembered until you clear it,
            and it takes precedence over any location this browser shared.
          </p>
          <div className="flex items-center gap-2">
            <MagnifyingGlass className="text-muted-foreground h-4 w-4 shrink-0" aria-hidden />
            <input
              id="weather-city"
              type="search"
              autoComplete="off"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              disabled={busy}
              placeholder="London, Tokyo, Sao Paulo..."
              className="bg-background w-full rounded-lg border px-3 py-2 text-sm"
            />
            {searching && <CircleNotch className="h-4 w-4 shrink-0 animate-spin" aria-hidden />}
          </div>

          {searchError && <p className="text-destructive text-xs">{searchError}</p>}
          {results !== null && results.length === 0 && !searching && (
            <p className="text-muted-foreground text-xs">No place of that name was found.</p>
          )}
          {results !== null && results.length > 0 && (
            <ul className="divide-border/60 divide-y rounded-lg border">
              {results.map((match) => (
                <li key={placeLabel(match) + match.latitude + ':' + match.longitude}>
                  <button
                    type="button"
                    onClick={() => void chooseCity(match)}
                    disabled={working}
                    className="hover:bg-muted/60 flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm disabled:opacity-50"
                  >
                    <span className="min-w-0 truncate">{placeLabel(match)}</span>
                    <span className="text-muted-foreground shrink-0 text-xs">
                      {match.timezone ?? ''}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {notice && (
          <p
            className={
              notice.tone === 'error' ? 'text-destructive text-xs' : 'text-muted-foreground text-xs'
            }
          >
            {notice.text}
          </p>
        )}
      </div>
    </div>
  );
}
