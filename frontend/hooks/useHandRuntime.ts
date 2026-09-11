'use client';

/**
 * The hand-recognition runtime as React state. The framework-free HandRuntime
 * does the loading; this hook creates one for the life of the component,
 * starts it at once (mounting the dock is the request) and disposes it,
 * model and all, on unmount.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { loadMediaPipeHandProvider } from '@/lib/hands/mediapipe';
import { HandRuntime, type HandRuntimeSnapshot } from '@/lib/hands/runtime';

export interface HandRuntimeController extends HandRuntimeSnapshot {
  /** Load again after a failure. */
  retry: () => void;
}

const BEFORE_MOUNT: HandRuntimeSnapshot = { status: 'loading', provider: null, detail: null };

export function useHandRuntime(): HandRuntimeController {
  const runtimeRef = useRef<HandRuntime | null>(null);
  const [snapshot, setSnapshot] = useState<HandRuntimeSnapshot>(BEFORE_MOUNT);

  useEffect(() => {
    const runtime = new HandRuntime({ load: loadMediaPipeHandProvider, onChange: setSnapshot });
    runtimeRef.current = runtime;
    setSnapshot(runtime.snapshot());
    return () => {
      runtimeRef.current = null;
      runtime.dispose();
    };
  }, []);

  const retry = useCallback(() => runtimeRef.current?.retry(), []);

  return { ...snapshot, retry };
}
