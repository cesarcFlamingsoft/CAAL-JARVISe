/** A deliberately small, one-shot JPEG capture from an already-playing video. */

export const MAX_FRAME_EDGE = 640;
export const MAX_FRAME_PIXELS = MAX_FRAME_EDGE * MAX_FRAME_EDGE;
export const MAX_FRAME_BYTES = 400 * 1024;
const JPEG_QUALITY = 0.78;

export interface VideoLike {
  readonly videoWidth: number;
  readonly videoHeight: number;
  readonly readyState: number;
}

export interface CanvasLike {
  width: number;
  height: number;
  getContext(
    type: '2d',
    options?: { alpha: false }
  ): Pick<CanvasRenderingContext2D, 'drawImage'> | null;
  toBlob(callback: (blob: Blob | null) => void, type?: string, quality?: number): void;
}

export interface CapturedFrame {
  jpegBase64: string;
  readonly width: number;
  readonly height: number;
  readonly bytes: number;
  clear(): void;
}

interface CaptureDependencies {
  createCanvas?: () => CanvasLike;
}

function encoded(canvas: CanvasLike): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('capture_failed'))),
      'image/jpeg',
      JPEG_QUALITY
    );
  });
}

function base64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunk) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunk));
  }
  return btoa(binary);
}

/** Capture one frame without changing, stopping, or cloning the video's stream. */
export async function captureVideoFrame(
  video: VideoLike,
  dependencies: CaptureDependencies = {}
): Promise<CapturedFrame> {
  const sourceWidth = Math.trunc(video.videoWidth);
  const sourceHeight = Math.trunc(video.videoHeight);
  if (
    video.readyState < 2 ||
    sourceWidth < 1 ||
    sourceHeight < 1 ||
    !Number.isSafeInteger(sourceWidth) ||
    !Number.isSafeInteger(sourceHeight)
  ) {
    throw new Error('camera_not_ready');
  }

  const scale = Math.min(1, MAX_FRAME_EDGE / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  if (width * height > MAX_FRAME_PIXELS) throw new Error('frame_too_large');

  const canvas: CanvasLike =
    dependencies.createCanvas?.() ?? (document.createElement('canvas') as unknown as CanvasLike);
  canvas.width = width;
  canvas.height = height;
  try {
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('capture_failed');
    context.drawImage(video as unknown as CanvasImageSource, 0, 0, width, height);
    const blob = await encoded(canvas);
    if (blob.type !== 'image/jpeg' || blob.size < 1 || blob.size > MAX_FRAME_BYTES) {
      throw new Error('frame_too_large');
    }
    const jpegBase64 = base64(new Uint8Array(await blob.arrayBuffer()));
    const frame: CapturedFrame = {
      jpegBase64,
      width,
      height,
      bytes: blob.size,
      clear() {
        this.jpegBase64 = '';
      },
    };
    return frame;
  } finally {
    // Dropping both dimensions releases the canvas backing pixel buffer.
    canvas.width = 0;
    canvas.height = 0;
  }
}
