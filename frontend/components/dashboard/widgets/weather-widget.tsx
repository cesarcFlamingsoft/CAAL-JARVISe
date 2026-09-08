'use client';

/**
 * Current conditions where this user's weather is read from.
 *
 * Everything shown came from the backend as a bounded reading: a temperature,
 * the words for the sky, one practical sentence, and how old the reading is.
 * When the weather service could not be reached the last reading is shown and
 * said to be stale; when there has never been one, the widget says that
 * instead of drawing an empty box. Nothing here reads the voice session.
 */
import { CloudSun, MapPin, WarningCircle } from '@phosphor-icons/react/dist/ssr';
import { Button } from '@/components/livekit/button';
import type { WeatherController } from '@/hooks/useWeather';
import {
  type CurrentWeather,
  conditionsLine,
  freshnessLabel,
  placeHint,
  placeLine,
  temperatureLabel,
} from '@/lib/dashboard/weather';
import { WidgetEmpty, WidgetError, WidgetLoading, WidgetSignIn } from '../widget-notice';

interface WeatherWidgetProps {
  weather: WeatherController;
  passwordLogin: boolean;
  now: Date | null;
  onOpenSettings: () => void;
}

export function WeatherWidget({
  weather,
  passwordLogin,
  now,
  onOpenSettings,
}: WeatherWidgetProps) {
  if (weather.status === 'loading') {
    return <WidgetLoading label="Reading the weather..." />;
  }
  if (weather.status === 'unauthorized') {
    return <WidgetSignIn passwordLogin={passwordLogin} what="your weather" />;
  }
  if (weather.status === 'unconfigured') {
    return (
      <WidgetEmpty
        title="Weather needs multi-user identity"
        detail="This JARVIS server runs in single-user mode, so there is no per-user place to read the weather for."
      />
    );
  }
  if (weather.status === 'error') {
    return (
      <WidgetError
        title="Weather is unavailable"
        detail="The backend did not answer."
        onRetry={weather.reload}
      />
    );
  }

  const reading = weather.data;
  if (reading.state === 'no_location') {
    return (
      <WidgetEmpty
        title="No location set"
        detail="Choose a city, or share this browser location, under Settings and your weather appears here."
        action={
          <Button variant="outline" size="sm" onClick={onOpenSettings}>
            Set location
          </Button>
        }
      />
    );
  }
  if (reading.state === 'unavailable') {
    return (
      <WidgetError
        title="The weather service could not be reached"
        detail="Nothing is shown rather than a guess. It will be retried on the next refresh."
        onRetry={weather.reload}
      />
    );
  }
  return <Conditions reading={reading} now={now} onOpenSettings={onOpenSettings} />;
}

interface ConditionsProps {
  reading: CurrentWeather;
  now: Date | null;
  onOpenSettings: () => void;
}

/** A reading that exists: current or stale, always dated, never embellished. */
function Conditions({ reading, now, onOpenSettings }: ConditionsProps) {
  const detail = conditionsLine(reading.observation);
  const place = reading.location;
  // A shared browser position is resolved to a nearby place by the backend, so
  // a wrong one can be seen here and replaced with a city chosen by hand.
  const approximate = placeHint(place);
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-3xl font-semibold tabular-nums">
          {temperatureLabel(reading.observation?.temperatureC ?? null)}
        </span>
        <span className="text-sm font-medium">{reading.conditions ?? 'Conditions unknown'}</span>
      </div>

      {detail && <p className="text-muted-foreground text-xs">{detail}</p>}
      {reading.advice && <p className="text-sm">{reading.advice}</p>}

      {place && (
        <div className="text-muted-foreground space-y-0.5 text-xs">
          <p className="flex items-center gap-1.5 truncate">
            <MapPin aria-hidden className="size-3.5 shrink-0" weight="bold" />
            <span className="truncate">{placeLine(place)}</span>
          </p>
          {approximate && <p className="truncate pl-5">{approximate}</p>}
        </div>
      )}

      <p
        className={
          reading.stale
            ? 'text-destructive flex items-start gap-1.5 text-xs'
            : 'text-muted-foreground flex items-start gap-1.5 text-xs'
        }
      >
        {reading.stale ? (
          <WarningCircle aria-hidden className="mt-px size-3.5 shrink-0" weight="bold" />
        ) : (
          <CloudSun aria-hidden className="mt-px size-3.5 shrink-0" weight="bold" />
        )}
        <span>{freshnessLabel(reading, now ?? new Date())}</span>
      </p>

      <Button variant="ghost" size="sm" className="-ml-2" onClick={onOpenSettings}>
        {approximate ? 'Not right? Choose a city' : 'Change location'}
      </Button>
    </div>
  );
}
