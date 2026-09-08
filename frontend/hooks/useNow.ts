'use client';

import { useEffect, useState } from 'react';

/**
 * The current time, ticking. Null until mounted so server and client markup
 * agree; callers render a placeholder for that first paint.
 */
export function useNow(intervalMs = 1000): Date | null {
  const [now, setNow] = useState<Date | null>(null);

  useEffect(() => {
    setNow(new Date());
    const timer = setInterval(() => setNow(new Date()), intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs]);

  return now;
}
