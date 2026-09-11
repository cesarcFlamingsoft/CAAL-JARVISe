/**
 * The camera lifecycle, without React.
 *
 * Owns exactly one video stream at a time: asks for it with explicit
 * constraints, lists the cameras once permission exists (labels are empty
 * before that), switches between them by stopping the old track first, and
 * names every failure the browser can report instead of a generic error.
 *
 * Frames never leave the stream object handed to the caller. Nothing here
 * logs, stores or sends anything about the camera; the only side effects are
 * getUserMedia, enumerateDevices and stopping tracks.
 */

export type CameraStatus =
  | 'unsupported'
  | 'idle'
  | 'requesting'
  | 'live'
  | 'denied'
  | 'no-camera'
  | 'busy'
  | 'unavailable'
  | 'ended'
  | 'error';

export interface CameraDevice {
  deviceId: string;
  label: string;
}

export interface CameraSnapshot {
  status: CameraStatus;
  stream: MediaStreamLike | null;
  devices: CameraDevice[];
  selectedDeviceId: string | null;
  /** One plain sentence for the surface to show when something went wrong. */
  detail: string | null;
}

/** The slices of the DOM media types this module needs; real objects satisfy them. */
export interface MediaStreamTrackLike {
  kind: string;
  readyState: 'live' | 'ended';
  stop(): void;
  addEventListener(type: 'ended', listener: () => void): void;
  removeEventListener(type: 'ended', listener: () => void): void;
  getSettings(): { deviceId?: string };
}

export interface MediaStreamLike {
  getTracks(): MediaStreamTrackLike[];
}

export interface MediaDevicesLike {
  getUserMedia(constraints: MediaStreamConstraints): Promise<MediaStreamLike>;
  enumerateDevices(): Promise<MediaDeviceInfo[]>;
  addEventListener(type: 'devicechange', listener: () => void): void;
  removeEventListener(type: 'devicechange', listener: () => void): void;
}

/**
 * Landmark inference needs a small, steady picture, not a sharp one; asking
 * for more only heats the machine. A chosen camera is pinned exactly so a
 * switch can fail loudly rather than quietly reopen the previous one.
 */
export function cameraConstraints(deviceId: string | null): MediaStreamConstraints {
  const video: MediaTrackConstraints = {
    width: { ideal: 640 },
    height: { ideal: 480 },
    frameRate: { ideal: 30, max: 30 },
  };
  if (deviceId) video.deviceId = { exact: deviceId };
  else video.facingMode = 'user';
  return { video, audio: false };
}

export interface CameraProblem {
  status: CameraStatus;
  detail: string;
}

export function describeCameraError(error: unknown): CameraProblem {
  const name = error instanceof Error ? error.name : '';
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return { status: 'denied', detail: 'Camera access was declined for this site.' };
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return { status: 'no-camera', detail: 'No camera is connected to this device.' };
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return { status: 'busy', detail: 'The camera is in use by another app or could not start.' };
    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
      return { status: 'unavailable', detail: 'That camera is no longer available.' };
    default:
      return { status: 'error', detail: 'The camera could not be opened.' };
  }
}

export interface CameraSessionOptions {
  /** navigator.mediaDevices, or null where it does not exist (insecure context, old browser). */
  mediaDevices: MediaDevicesLike | null;
  onChange?: (snapshot: CameraSnapshot) => void;
}

function stopAll(stream: MediaStreamLike | null): void {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    try {
      track.stop();
    } catch {
      // A track that is already gone has nothing to stop.
    }
  }
}

export class CameraSession {
  private readonly media: MediaDevicesLike | null;
  private readonly onChange?: (snapshot: CameraSnapshot) => void;
  private state: CameraSnapshot;
  /** Bumped by every stop or new request; a result from an older request is discarded. */
  private generation = 0;
  private inFlight: Promise<void> | null = null;
  private listening = false;
  private endedListener: (() => void) | null = null;
  private readonly onDeviceChange = () => {
    this.inFlight = this.handleDeviceChange();
  };

  constructor(options: CameraSessionOptions) {
    this.media = options.mediaDevices;
    this.onChange = options.onChange;
    this.state = {
      status: this.media ? 'idle' : 'unsupported',
      stream: null,
      devices: [],
      selectedDeviceId: null,
      detail: this.media ? null : 'This browser does not offer camera access here.',
    };
  }

  snapshot(): CameraSnapshot {
    return this.state;
  }

  /** Ask for the camera (a particular one, the last chosen, or any). */
  start(deviceId?: string | null): Promise<void> {
    if (!this.media) return Promise.resolve();
    if (!this.listening) {
      this.media.addEventListener('devicechange', this.onDeviceChange);
      this.listening = true;
    }
    const wanted = deviceId === undefined ? this.state.selectedDeviceId : deviceId;
    this.inFlight = this.open(wanted);
    return this.inFlight;
  }

  selectDevice(deviceId: string): Promise<void> {
    return this.start(deviceId);
  }

  /** Stop every track now. Anything still being requested is dropped on arrival. */
  stop(): void {
    this.generation += 1;
    this.detach();
    this.update({ status: this.media ? 'idle' : 'unsupported', stream: null, detail: null });
  }

  /** Stop and stop listening: for unmount. */
  dispose(): void {
    this.stop();
    if (this.media && this.listening) {
      this.media.removeEventListener('devicechange', this.onDeviceChange);
      this.listening = false;
    }
  }

  /** Resolves once the latest request or device change has been handled. */
  async settled(): Promise<void> {
    let seen: Promise<void> | null = null;
    while (this.inFlight && this.inFlight !== seen) {
      seen = this.inFlight;
      await seen;
    }
  }

  private update(patch: Partial<CameraSnapshot>): void {
    this.state = { ...this.state, ...patch };
    this.onChange?.(this.state);
  }

  private detach(): void {
    const current = this.state.stream;
    if (current && this.endedListener) {
      for (const track of current.getTracks()) {
        track.removeEventListener('ended', this.endedListener);
      }
    }
    this.endedListener = null;
    stopAll(current);
  }

  private async open(deviceId: string | null, fallback = true): Promise<void> {
    if (!this.media) return;
    const generation = ++this.generation;
    // Stop the previous camera first: many devices refuse to open two.
    this.detach();
    this.update({ status: 'requesting', stream: null, selectedDeviceId: deviceId, detail: null });

    let stream: MediaStreamLike;
    try {
      stream = await this.media.getUserMedia(cameraConstraints(deviceId));
    } catch (error) {
      if (generation !== this.generation) return;
      const problem = describeCameraError(error);
      if (deviceId && fallback && problem.status === 'unavailable') {
        // The pinned camera is gone: any camera beats none.
        return this.open(null, false);
      }
      this.update({ status: problem.status, stream: null, detail: problem.detail });
      return;
    }
    if (generation !== this.generation) {
      // Stopped, or superseded, while the browser was asking.
      stopAll(stream);
      return;
    }

    const ended = () => {
      if (generation !== this.generation) return;
      this.generation += 1;
      this.detach();
      this.update({
        status: 'ended',
        stream: null,
        detail: 'The camera stopped. It may have been unplugged or taken by another app.',
      });
    };
    this.endedListener = ended;
    for (const track of stream.getTracks()) track.addEventListener('ended', ended);

    const devices = await this.listCameras();
    if (generation !== this.generation) return;
    const settings = stream
      .getTracks()
      .find((track) => track.kind === 'video')
      ?.getSettings();
    const selectedDeviceId = settings?.deviceId || deviceId || devices[0]?.deviceId || null;
    this.update({ status: 'live', stream, devices, selectedDeviceId, detail: null });
  }

  private async listCameras(): Promise<CameraDevice[]> {
    if (!this.media) return [];
    try {
      const all = await this.media.enumerateDevices();
      const cameras = all.filter((device) => device.kind === 'videoinput' && device.deviceId);
      return cameras.map((device, index) => ({
        deviceId: device.deviceId,
        label: device.label || 'Camera ' + (index + 1),
      }));
    } catch {
      return this.state.devices;
    }
  }

  private async handleDeviceChange(): Promise<void> {
    const devices = await this.listCameras();
    const selected = this.state.selectedDeviceId;
    const stillThere = !selected || devices.some((device) => device.deviceId === selected);
    if (this.state.status === 'live' && !stillThere) {
      // The camera in use was unplugged: move to whatever is left.
      await this.open(null, false);
      return;
    }
    this.update({ devices });
  }
}
