import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { type CanvasLike, MAX_FRAME_BYTES, MAX_FRAME_EDGE, captureVideoFrame } from './capture.ts';

function canvasReturning(bytes: number, calls: string[]): CanvasLike {
  return {
    width: 0,
    height: 0,
    getContext() {
      return {
        drawImage(_video, _x, _y, width, height) {
          calls.push(`draw:${width}x${height}`);
        },
      };
    },
    toBlob(callback, type, quality) {
      calls.push(`blob:${type}:${quality}`);
      callback(new Blob([new Uint8Array(bytes)], { type: 'image/jpeg' }));
    },
  };
}

describe('one-shot camera frame capture', () => {
  it('downscales the longest edge and returns only a bounded JPEG', async () => {
    const calls: string[] = [];
    const canvas = canvasReturning(1024, calls);
    const frame = await captureVideoFrame(
      { videoWidth: 1920, videoHeight: 1080, readyState: 4 },
      { createCanvas: () => canvas }
    );

    assert.equal(frame.width, MAX_FRAME_EDGE);
    assert.equal(frame.height, 360);
    assert.equal(frame.bytes, 1024);
    assert.ok(frame.jpegBase64.length > 0);
    assert.deepEqual(calls, ['draw:640x360', 'blob:image/jpeg:0.78']);
    assert.equal(canvas.width, 0, 'pixel buffer is released after encoding');
    assert.equal(canvas.height, 0, 'pixel buffer is released after encoding');
  });

  it('refuses an unavailable video, invalid dimensions, and an oversized JPEG', async () => {
    const createCanvas = () => canvasReturning(1, []);
    await assert.rejects(
      captureVideoFrame({ videoWidth: 640, videoHeight: 480, readyState: 1 }, { createCanvas }),
      /camera_not_ready/
    );
    await assert.rejects(
      captureVideoFrame({ videoWidth: 0, videoHeight: 480, readyState: 4 }, { createCanvas }),
      /camera_not_ready/
    );
    await assert.rejects(
      captureVideoFrame(
        { videoWidth: 640, videoHeight: 480, readyState: 4 },
        { createCanvas: () => canvasReturning(MAX_FRAME_BYTES + 1, []) }
      ),
      /frame_too_large/
    );
  });

  it('lets the caller erase the sole base64 reference after the request', async () => {
    const frame = await captureVideoFrame(
      { videoWidth: 320, videoHeight: 200, readyState: 4 },
      { createCanvas: () => canvasReturning(16, []) }
    );
    frame.clear();
    assert.equal(frame.jpegBase64, '');
    frame.clear();
    assert.equal(frame.jpegBase64, '');
  });
});
