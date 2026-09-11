import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  CameraSession,
  type CameraSnapshot,
  type MediaDevicesLike,
  type MediaStreamLike,
  type MediaStreamTrackLike,
  cameraConstraints,
  describeCameraError,
} from './camera-session.ts';

class FakeTrack implements MediaStreamTrackLike {
  kind = 'video';
  readyState: 'live' | 'ended' = 'live';
  onended: (() => void) | null = null;
  stopped = 0;
  label: string;
  constructor(label: string) {
    this.label = label;
  }
  stop() {
    this.stopped += 1;
    this.readyState = 'ended';
  }
  addEventListener(type: string, listener: () => void) {
    if (type === 'ended') this.onended = listener;
  }
  removeEventListener(type: string) {
    if (type === 'ended') this.onended = null;
  }
  getSettings() {
    return { deviceId: this.label };
  }
}

class FakeStream implements MediaStreamLike {
  tracks: FakeTrack[];
  constructor(tracks: FakeTrack[]) {
    this.tracks = tracks;
  }
  getTracks() {
    return this.tracks;
  }
}

interface FakeDevice {
  deviceId: string;
  kind: 'videoinput' | 'audioinput';
  label: string;
}

class FakeMediaDevices implements MediaDevicesLike {
  granted = false;
  requests: MediaStreamConstraints[] = [];
  failWith: Error | null = null;
  streams: FakeStream[] = [];
  private listeners = new Set<() => void>();
  devices: FakeDevice[];
  constructor(devices: FakeDevice[]) {
    this.devices = devices;
  }
  async getUserMedia(constraints: MediaStreamConstraints) {
    this.requests.push(constraints);
    if (this.failWith) throw this.failWith;
    this.granted = true;
    const video = constraints.video;
    const wanted =
      typeof video === 'object' && video && typeof video.deviceId === 'object' && video.deviceId
        ? String((video.deviceId as { exact?: string }).exact ?? '')
        : '';
    const device =
      this.devices.find((d) => d.kind === 'videoinput' && (!wanted || d.deviceId === wanted)) ??
      null;
    if (!device) throw domError('OverconstrainedError');
    const stream = new FakeStream([new FakeTrack(device.deviceId)]);
    this.streams.push(stream);
    return stream;
  }
  async enumerateDevices() {
    // Labels and stable ids only exist after permission, as in real browsers.
    return this.devices.map((d) => ({
      deviceId: this.granted ? d.deviceId : '',
      kind: d.kind,
      label: this.granted ? d.label : '',
      groupId: '',
    })) as MediaDeviceInfo[];
  }
  addEventListener(type: string, listener: () => void) {
    if (type === 'devicechange') this.listeners.add(listener);
  }
  removeEventListener(type: string, listener: () => void) {
    if (type === 'devicechange') this.listeners.delete(listener);
  }
  fireDeviceChange() {
    for (const listener of this.listeners) listener();
  }
  liveTracks() {
    return this.streams.flatMap((s) => s.tracks).filter((t) => t.readyState === 'live');
  }
}

function domError(name: string): Error {
  const error = new Error(name);
  error.name = name;
  return error;
}

const TWO_CAMERAS: FakeDevice[] = [
  { deviceId: 'cam-front', kind: 'videoinput', label: 'FaceTime HD' },
  { deviceId: 'mic-1', kind: 'audioinput', label: 'Microphone' },
  { deviceId: 'cam-usb', kind: 'videoinput', label: 'USB Capture' },
];

function session(devices: FakeMediaDevices | null) {
  const snapshots: CameraSnapshot[] = [];
  const cam = new CameraSession({ mediaDevices: devices, onChange: (s) => snapshots.push(s) });
  return { cam, snapshots, last: () => snapshots.at(-1) ?? cam.snapshot() };
}

describe('camera constraints', () => {
  it('ask for a modest video track and never audio', () => {
    const constraints = cameraConstraints(null);
    assert.equal(constraints.audio, false);
    const video = constraints.video as MediaTrackConstraints;
    assert.equal(typeof video, 'object');
    assert.ok(video.width && video.height && video.frameRate, 'resolution and rate are explicit');
    assert.equal(video.deviceId, undefined);
  });

  it('pin a chosen camera exactly so a switch cannot silently fall back', () => {
    const video = cameraConstraints('cam-usb').video as MediaTrackConstraints;
    assert.deepEqual(video.deviceId, { exact: 'cam-usb' });
    assert.equal(video.facingMode, undefined);
  });
});

describe('camera error taxonomy', () => {
  it('names each browser failure honestly', () => {
    assert.equal(describeCameraError(domError('NotAllowedError')).status, 'denied');
    assert.equal(describeCameraError(domError('SecurityError')).status, 'denied');
    assert.equal(describeCameraError(domError('NotFoundError')).status, 'no-camera');
    assert.equal(describeCameraError(domError('NotReadableError')).status, 'busy');
    assert.equal(describeCameraError(domError('AbortError')).status, 'busy');
    assert.equal(describeCameraError(domError('OverconstrainedError')).status, 'unavailable');
    assert.equal(describeCameraError(new TypeError('x')).status, 'error');
    assert.equal(describeCameraError('nope').status, 'error');
  });
});

describe('camera session', () => {
  it('is unsupported without getUserMedia, and asks nothing', async () => {
    const { cam, last } = session(null);
    await cam.start();
    assert.equal(last().status, 'unsupported');
    assert.equal(last().stream, null);
  });

  it('requests permission with explicit constraints, then lists cameras with labels', async () => {
    const devices = new FakeMediaDevices(TWO_CAMERAS);
    const { cam, snapshots, last } = session(devices);
    await cam.start();
    assert.deepEqual(snapshots.map((s) => s.status).slice(0, 2), ['requesting', 'live']);
    assert.equal(devices.requests.length, 1);
    assert.equal(devices.requests[0].audio, false);
    assert.deepEqual(
      last().devices.map((d) => d.label),
      ['FaceTime HD', 'USB Capture'],
      'only video inputs, with labels, after permission'
    );
    assert.equal(last().selectedDeviceId, 'cam-front');
    assert.ok(last().stream);
  });

  it('switches cameras by stopping the old track before opening the new one', async () => {
    const devices = new FakeMediaDevices(TWO_CAMERAS);
    const { cam, last } = session(devices);
    await cam.start();
    const first = devices.liveTracks();
    assert.equal(first.length, 1);
    await cam.selectDevice('cam-usb');
    assert.equal(first[0].stopped, 1, 'the first track was stopped');
    assert.equal(devices.liveTracks().length, 1, 'exactly one live track at a time');
    assert.equal(last().selectedDeviceId, 'cam-usb');
    const video = devices.requests.at(-1)?.video as MediaTrackConstraints;
    assert.deepEqual(video.deviceId, { exact: 'cam-usb' });
  });

  it('reports denial and holds no tracks', async () => {
    const devices = new FakeMediaDevices(TWO_CAMERAS);
    devices.failWith = domError('NotAllowedError');
    const { cam, last } = session(devices);
    await cam.start();
    assert.equal(last().status, 'denied');
    assert.equal(last().stream, null);
    assert.equal(devices.liveTracks().length, 0);
  });

  it('reports a camera that is busy elsewhere, and one that does not exist', async () => {
    const busy = new FakeMediaDevices(TWO_CAMERAS);
    busy.failWith = domError('NotReadableError');
    const a = session(busy);
    await a.cam.start();
    assert.equal(a.last().status, 'busy');

    const none = new FakeMediaDevices([]);
    none.failWith = domError('NotFoundError');
    const b = session(none);
    await b.cam.start();
    assert.equal(b.last().status, 'no-camera');
  });

  it('falls back to any camera when the pinned one has vanished', async () => {
    const devices = new FakeMediaDevices(TWO_CAMERAS);
    const { cam, last } = session(devices);
    await cam.start();
    await cam.selectDevice('cam-usb');
    devices.devices = devices.devices.filter((d) => d.deviceId !== 'cam-usb');
    devices.fireDeviceChange();
    await cam.settled();
    assert.equal(last().status, 'live');
    assert.equal(last().selectedDeviceId, 'cam-front');
    assert.equal(devices.liveTracks().length, 1);
  });

  it('notices when the track ends underneath it', async () => {
    const devices = new FakeMediaDevices(TWO_CAMERAS);
    const { cam, last } = session(devices);
    await cam.start();
    const track = devices.liveTracks()[0];
    track.readyState = 'ended';
    track.onended?.();
    assert.equal(last().status, 'ended');
    assert.equal(last().stream, null);
  });

  it('stops every track and forgets the stream on stop', async () => {
    const devices = new FakeMediaDevices(TWO_CAMERAS);
    const { cam, last } = session(devices);
    await cam.start();
    cam.stop();
    assert.equal(devices.liveTracks().length, 0);
    assert.equal(last().status, 'idle');
    assert.equal(last().stream, null);
    // A request that resolves after stop must not leak a live track.
    const late = cam.start();
    cam.stop();
    await late;
    assert.equal(devices.liveTracks().length, 0);
    assert.equal(last().status, 'idle');
  });
});
