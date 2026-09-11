/**
 * The one place that knows which model runs: the MediaPipe Tasks Vision
 * HandLandmarker, loaded on demand from the runtime and model files this
 * origin serves (see assets.ts). The library is imported only when hand
 * control is switched on, so the rest of the app never pays for it, and it
 * is told exactly where its WebAssembly and model live, so it never looks
 * anywhere else. Frames go in and landmarks come out, all in this browser.
 */
import { HAND_MODEL_PATH, HAND_RUNTIME_PATH } from './assets';
import { type HandLandmarkProvider, createLandmarkerProvider } from './landmarks';

export const MEDIAPIPE_PROVIDER_ID = 'mediapipe-hand-landmarker';

/** A little above the library defaults: a missed frame is cheaper than a phantom hand. */
const CONFIDENCE = 0.6;

function throwIfAborted(signal: AbortSignal): void {
  if (!signal.aborted) return;
  throw signal.reason instanceof Error
    ? signal.reason
    : new DOMException('Hand recognition was switched off.', 'AbortError');
}

/**
 * Load the runtime, then the model, then hand back a provider. Between each
 * step the abort signal is honoured, and a landmarker that finishes after an
 * abort is closed rather than returned.
 */
export async function loadMediaPipeHandProvider(
  signal: AbortSignal
): Promise<HandLandmarkProvider> {
  throwIfAborted(signal);
  const { FilesetResolver, HandLandmarker } = await import('@mediapipe/tasks-vision');
  throwIfAborted(signal);
  const fileset = await FilesetResolver.forVisionTasks(HAND_RUNTIME_PATH);
  throwIfAborted(signal);

  // The GPU delegate is much faster; a browser without usable WebGL still gets the CPU path.
  let lastError: unknown = null;
  for (const delegate of ['GPU', 'CPU'] as const) {
    try {
      const landmarker = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: HAND_MODEL_PATH, delegate },
        runningMode: 'VIDEO',
        numHands: 1,
        minHandDetectionConfidence: CONFIDENCE,
        minHandPresenceConfidence: CONFIDENCE,
        minTrackingConfidence: CONFIDENCE,
      });
      if (signal.aborted) {
        landmarker.close();
        throwIfAborted(signal);
      }
      return createLandmarkerProvider(landmarker, MEDIAPIPE_PROVIDER_ID);
    } catch (error) {
      if (signal.aborted) throw error;
      lastError = error;
    }
  }
  throw lastError;
}
