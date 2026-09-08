/**
 * Small in-memory sliding-window rate limiter for the BFF.
 *
 * Keyed by caller (user id or client address), bounded in memory, and reset
 * only by time. A backstop against brute force and abuse of the identity
 * routes; the backend enforces its own limits independently.
 */

export interface RateLimiterOptions {
  readonly limit: number;
  readonly windowMs: number;
  readonly maxKeys?: number;
}

export class RateLimiter {
  private readonly limit: number;
  private readonly windowMs: number;
  private readonly maxKeys: number;
  private readonly events = new Map<string, number[]>();

  constructor(options: RateLimiterOptions) {
    if (options.limit < 1 || options.windowMs <= 0) {
      throw new Error('limit and window must be positive');
    }
    this.limit = Math.trunc(options.limit);
    this.windowMs = options.windowMs;
    this.maxKeys = Math.max(1, Math.trunc(options.maxKeys ?? 10_000));
  }

  get size(): number {
    return this.events.size;
  }

  allow(key: string, now: number = Date.now()): boolean {
    const events = this.touch(key, now);
    if (events.length >= this.limit) {
      return false;
    }
    events.push(now);
    return true;
  }

  retryAfterSeconds(key: string, now: number = Date.now()): number {
    const events = this.touch(key, now);
    if (events.length < this.limit) {
      return 0;
    }
    return Math.max(0, Math.ceil((events[0] + this.windowMs - now) / 1000));
  }

  private touch(key: string, now: number): number[] {
    let events = this.events.get(key);
    if (events === undefined) {
      events = [];
      this.events.set(key, events);
      while (this.events.size > this.maxKeys) {
        const oldest = this.events.keys().next().value;
        if (oldest === undefined) break;
        this.events.delete(oldest);
      }
    } else {
      // Refresh insertion order so eviction drops the least recently seen key.
      this.events.delete(key);
      this.events.set(key, events);
    }
    const cutoff = now - this.windowMs;
    while (events.length > 0 && events[0] <= cutoff) {
      events.shift();
    }
    return events;
  }
}
