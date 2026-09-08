'use client';

/**
 * The signed-in user's current weather, from the BFF and nowhere else.
 *
 * The backend caps the upstream to one call per place per hour, so polling
 * here is cheap: a refresh that lands inside the hour is answered from the
 * cache. The reading is re-read on a timer while the page is visible, and
 * immediately whenever the weather location setting changes, so choosing a
 * city or sharing a position shows up without a reload. Nothing here reads
 * the voice session: weather is not a property of being in a call.
 */
import { useCallback, useEffect, useState } from 'react';
import { type CurrentWeather, browserWeather } from '@/lib/dashboard/weather';

export type WeatherState =
  | { status: 'loading' }
  | { status: 'ready'; data: CurrentWeather }
  /** Identity is configured and this browser is not a signed-in user. */
  | { status: 'unauthorized'; code: string }
  /** A single-user deployment: there are no per-user places to read. */
  | { status: 'unconfigured' }
  | { status: 'error'; code: string };

export type WeatherController = WeatherState & { reload: () => void };

const WEATHER_PATH = '/api/dashboard/weather';
/** Half the backend cache lifetime: often enough to be current, never wasteful. */
const REFRESH_INTERVAL_MS = 30 * 60_000;

export function useWeather(): WeatherController {
  const [state, setState] = useState<WeatherState>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);

  const reload = useCallback(() => setAttempt((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch(WEATHER_PATH, {
          cache: 'no-store',
          credentials: 'same-origin',
        });
        let body: unknown = null;
        try {
          body = await response.json();
        } catch {
          body = null;
        }
        if (cancelled) return;
        const code = (body as { error?: unknown } | null)?.error;
        const errorCode = typeof code === 'string' ? code : 'http_' + response.status;
        if (response.status === 401 || response.status === 403) {
          setState({ status: 'unauthorized', code: errorCode });
          return;
        }
        if (response.status === 503 && errorCode === 'identity_not_configured') {
          setState({ status: 'unconfigured' });
          return;
        }
        if (!response.ok) {
          setState({ status: 'error', code: errorCode });
          return;
        }
        const weather = browserWeather(body);
        setState(
          weather ? { status: 'ready', data: weather } : { status: 'error', code: 'malformed' }
        );
      } catch {
        if (!cancelled) setState({ status: 'error', code: 'network' });
      }
    };
    void load();
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') reload();
    }, REFRESH_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [attempt, reload]);

  // Choosing a city or sharing a position announces itself.
  useEffect(() => {
    window.addEventListener('weather-location-updated', reload);
    return () => window.removeEventListener('weather-location-updated', reload);
  }, [reload]);

  return { ...state, reload };
}
