import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { RateLimiter } from './rate-limit.ts';

describe('rate limiter', () => {
  it('allows a burst then refuses until the window moves', () => {
    const limiter = new RateLimiter({ limit: 3, windowMs: 60_000 });
    const now = 1_000_000;

    assert.deepEqual(
      [0, 1, 2, 3].map((i) => limiter.allow('ip-1', now + i)),
      [true, true, true, false]
    );
    assert.equal(limiter.allow('ip-2', now), true);
    assert.equal(limiter.allow('ip-1', now + 60_001), true);
    assert.ok(limiter.retryAfterSeconds('ip-1', now + 60_001) >= 0);
  });

  it('bounds its memory', () => {
    const limiter = new RateLimiter({ limit: 1, windowMs: 60_000, maxKeys: 10 });
    for (let i = 0; i < 50; i++) limiter.allow(`k${i}`, 1);
    assert.ok(limiter.size <= 10);
  });
});
