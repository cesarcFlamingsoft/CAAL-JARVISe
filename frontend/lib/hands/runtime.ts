/**
 * The life of the hand-recognition runtime, without React or MediaPipe.
 *
 * A runtime is created to be loaded: it asks the given loader for a provider
 * at once and reports loading, then ready with that provider, or failed with
 * one plain sentence. Disposing it aborts the load; a provider that arrives
 * after that is closed on the spot and never reported. Nothing here knows
 * which model sits behind the loader, so the whole lifecycle is tested with
 * fakes, and a provider only ever exists once its model is actually running.
 */
import type { HandLandmarkProvider } from './landmarks';

export type HandRuntimeStatus = 'loading' | 'ready' | 'failed';

export interface HandRuntimeSnapshot {
  status: HandRuntimeStatus;
  /** Present only while ready. */
  provider: HandLandmarkProvider | null;
  /** One plain sentence when loading failed. */
  detail: string | null;
}

export type HandProviderLoader = (signal: AbortSignal) => Promise<HandLandmarkProvider>;

export interface HandRuntimeOptions {
  load: HandProviderLoader;
  onChange?: (snapshot: HandRuntimeSnapshot) => void;
  /** Give up after this long. The files are local, so this is generous. */
  timeoutMs?: number;
}

export const DEFAULT_RUNTIME_TIMEOUT_MS = 30_000;

const TOO_LONG = 'Hand recognition took too long to load. Try again.';

/** One honest sentence for whatever the runtime or the browser threw. */
export function describeRuntimeError(error: unknown): string {
  const name = error instanceof Error ? error.name : '';
  const message = error instanceof Error ? error.message : typeof error === 'string' ? error : '';
  // A loader script that fails to load rejects with an ErrorEvent, not an Error.
  const isEvent =
    !!error && typeof error === 'object' && (error as { type?: unknown }).type === 'error';
  if (name === 'AbortError') return 'Hand recognition was switched off while loading.';
  if (isEvent || /ModuleFactory|fetch|\(40\d\)|\(5\d\d\)|network|not found/i.test(message)) {
    return 'The hand model files could not be loaded from this server.';
  }
  if (/CompileError|LinkError|RuntimeError/.test(name) || /WebAssembly|wasm|SIMD/i.test(message)) {
    return 'This browser cannot run the hand model.';
  }
  if (/WebGL|GPU|graphics|canvas/i.test(message)) {
    return 'The hand model could not start in this browser.';
  }
  return 'Hand recognition could not start in this browser.';
}

function safeDispose(provider: HandLandmarkProvider): void {
  try {
    provider.dispose();
  } catch {
    // Already closed.
  }
}

export class HandRuntime {
  private readonly load: HandProviderLoader;
  private readonly onChange?: (snapshot: HandRuntimeSnapshot) => void;
  private readonly timeoutMs: number;
  private state: HandRuntimeSnapshot = { status: 'loading', provider: null, detail: null };
  /** Bumped by every dispose, timeout or new attempt; an older attempt settling is discarded. */
  private generation = 0;
  private controller: AbortController | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private inFlight: Promise<void> | null = null;
  private disposed = false;

  constructor(options: HandRuntimeOptions) {
    this.load = options.load;
    this.onChange = options.onChange;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_RUNTIME_TIMEOUT_MS;
    this.inFlight = this.attempt();
  }

  snapshot(): HandRuntimeSnapshot {
    return this.state;
  }

  /** Load again after a failure. Does nothing while loading, ready or disposed. */
  retry(): void {
    if (this.disposed || this.state.status !== 'failed') return;
    this.inFlight = this.attempt();
  }

  /** Abort any load, close the provider and go quiet: for unmount. */
  dispose(): void {
    this.disposed = true;
    this.generation += 1;
    this.abortLoad();
    const provider = this.state.provider;
    this.state = { ...this.state, provider: null };
    if (provider) safeDispose(provider);
  }

  /** Resolves once the latest attempt has been handled. */
  async settled(): Promise<void> {
    let seen: Promise<void> | null = null;
    while (this.inFlight && this.inFlight !== seen) {
      seen = this.inFlight;
      await seen;
    }
  }

  private update(patch: Partial<HandRuntimeSnapshot>): void {
    this.state = { ...this.state, ...patch };
    if (!this.disposed) this.onChange?.(this.state);
  }

  private clearTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private abortLoad(): void {
    this.clearTimer();
    const controller = this.controller;
    this.controller = null;
    if (controller && !controller.signal.aborted) controller.abort();
  }

  private async attempt(): Promise<void> {
    const generation = ++this.generation;
    const controller = new AbortController();
    this.controller = controller;
    this.update({ status: 'loading', provider: null, detail: null });
    this.timer = setTimeout(() => {
      if (generation !== this.generation) return;
      this.generation += 1;
      this.abortLoad();
      this.update({ status: 'failed', provider: null, detail: TOO_LONG });
    }, this.timeoutMs);

    let provider: HandLandmarkProvider;
    try {
      provider = await this.load(controller.signal);
    } catch (error) {
      if (generation !== this.generation) return;
      this.clearTimer();
      this.controller = null;
      this.update({ status: 'failed', provider: null, detail: describeRuntimeError(error) });
      return;
    }
    if (generation !== this.generation) {
      // Disposed, timed out or superseded while loading: nobody can use this.
      safeDispose(provider);
      return;
    }
    this.clearTimer();
    this.controller = null;
    this.update({ status: 'ready', provider, detail: null });
  }
}
