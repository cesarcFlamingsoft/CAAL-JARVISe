'use client';

/**
 * The one camera this workspace may hold, as React state. All media work is
 * done by the framework-free CameraSession; this hook only creates it for the
 * life of the component and tears it down, tracks and all, on unmount.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  type CameraConsumer,
  CameraSession,
  type CameraSnapshot,
  type MediaDevicesLike,
} from '@/lib/hands/camera-session';

export interface CameraController extends CameraSnapshot {
  start: () => Promise<void>;
  acquire: (consumer: CameraConsumer) => Promise<void>;
  release: (consumer: CameraConsumer) => void;
  selectDevice: (deviceId: string) => Promise<void>;
}

const BEFORE_MOUNT: CameraSnapshot = {
  status: 'idle',
  stream: null,
  devices: [],
  selectedDeviceId: null,
  detail: null,
};

function browserMediaDevices(): MediaDevicesLike | null {
  if (typeof navigator === 'undefined') return null;
  const devices = navigator.mediaDevices;
  if (!devices || typeof devices.getUserMedia !== 'function') return null;
  return devices;
}

export function useCamera(): CameraController {
  const sessionRef = useRef<CameraSession | null>(null);
  const [snapshot, setSnapshot] = useState<CameraSnapshot>(BEFORE_MOUNT);

  useEffect(() => {
    const session = new CameraSession({
      mediaDevices: browserMediaDevices(),
      onChange: setSnapshot,
    });
    sessionRef.current = session;
    setSnapshot(session.snapshot());
    return () => {
      sessionRef.current = null;
      session.dispose();
    };
  }, []);

  const start = useCallback(() => sessionRef.current?.start() ?? Promise.resolve(), []);
  const acquire = useCallback(
    (consumer: CameraConsumer) => sessionRef.current?.acquire(consumer) ?? Promise.resolve(),
    []
  );
  const release = useCallback(
    (consumer: CameraConsumer) => sessionRef.current?.release(consumer),
    []
  );
  const selectDevice = useCallback(
    (deviceId: string) => sessionRef.current?.selectDevice(deviceId) ?? Promise.resolve(),
    []
  );

  return { ...snapshot, start, acquire, release, selectDevice };
}
