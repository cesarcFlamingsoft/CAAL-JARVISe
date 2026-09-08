import { headers } from 'next/headers';
import { ImageResponse } from 'next/og';
import getImageSize from 'buffer-image-size';
import mime from 'mime';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { APP_CONFIG_DEFAULTS } from '@/app-config';
import { getAppConfig } from '@/lib/utils';

type Dimensions = {
  width: number;
  height: number;
};

type ImageData = {
  base64: string;
  dimensions: Dimensions;
};

const BRAND_NAME = 'MEXCANTECH';
const BRAND_TAGLINE = 'Learn. Build. Empower.';

// Local brand assets, resolved from the public/ folder.
const BRAND_WORDMARK_PATH = 'public/mexcantech-logo.png';
const BRAND_MARK_PATH = 'public/mexcantech-mark.png';

// Image metadata
export const alt = 'JARVIS by MexcanTech - Learn. Build. Empower.';
export const size = {
  width: 1200,
  height: 628,
};

function isRemoteFile(uri: string) {
  return uri.startsWith('http');
}

function doesLocalFileExist(uri: string) {
  return existsSync(join(process.cwd(), uri));
}

// LOCAL FILES MUST BE IN PUBLIC FOLDER
async function loadFileData(filePath: string): Promise<ArrayBuffer> {
  if (isRemoteFile(filePath)) {
    const response = await fetch(filePath);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${filePath} - ${response.status} ${response.statusText}`);
    }
    return await response.arrayBuffer();
  }

  // Try file system first (works in local development)
  if (doesLocalFileExist(filePath)) {
    const buffer = await readFile(join(process.cwd(), filePath));
    return buffer.buffer.slice(
      buffer.byteOffset,
      buffer.byteOffset + buffer.byteLength
    ) as ArrayBuffer;
  }

  // Fallback to fetching from public URL (works in production)
  const publicFilePath = filePath.replace('public/', '');
  const fontUrl = `https://${process.env.VERCEL_URL}/${publicFilePath}`;

  const response = await fetch(fontUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${fontUrl} - ${response.status} ${response.statusText}`);
  }

  return await response.arrayBuffer();
}

async function getImageData(uri: string, fallbackUri?: string): Promise<ImageData> {
  try {
    const fileData = await loadFileData(uri);
    const buffer = Buffer.from(fileData);
    const mimeType = mime.getType(uri);

    return {
      base64: `data:${mimeType};base64,${buffer.toString('base64')}`,
      dimensions: getImageSize(buffer),
    };
  } catch (e) {
    if (fallbackUri) {
      return getImageData(fallbackUri, fallbackUri);
    }
    throw e;
  }
}

function scaleImageSize(size: { width: number; height: number }, desiredHeight: number) {
  const scale = desiredHeight / size.height;
  return {
    width: size.width * scale,
    height: desiredHeight,
  };
}

/**
 * Config logos may be absolute URLs or public/ paths such as `/mexcantech-mark.png`.
 * Remote URIs are fetched as-is; everything else is read from the public folder.
 */
function resolveImageUri(uri: string) {
  if (isRemoteFile(uri)) {
    return uri;
  }

  return 'public/' + uri.replace(/^[.\/]+/, '');
}

export const contentType = 'image/png';

// Image generation
export default async function Image() {
  const hdrs = await headers();
  const appConfig = await getAppConfig(hdrs);

  const pageTitle = appConfig.pageTitle || APP_CONFIG_DEFAULTS.pageTitle;
  const logoUri = resolveImageUri(appConfig.logoDark || appConfig.logo);

  // Load fonts - use file system in dev, fetch in production
  let commitMonoData: ArrayBuffer | undefined;
  let everettLightData: ArrayBuffer | undefined;

  try {
    commitMonoData = await loadFileData('public/commit-mono-400-regular.woff');
    everettLightData = await loadFileData('public/everett-light.woff');
  } catch (e) {
    console.error('Failed to load fonts:', e);
    // Continue without custom fonts - will fall back to system fonts
  }

  // bg
  const { base64: bgSrcBase64 } = await getImageData('public/opengraph-image-bg.png');

  // full brand logo (hero)
  const { base64: wordmarkSrcBase64, dimensions: wordmarkDimensions } =
    await getImageData(BRAND_WORDMARK_PATH);
  const wordmarkSize = scaleImageSize(wordmarkDimensions, 400);

  // square brand mark (header badge)
  const { base64: logoSrcBase64, dimensions: logoDimensions } = await getImageData(
    logoUri,
    BRAND_MARK_PATH
  );
  const logoSize = scaleImageSize(logoDimensions, 48);

  return new ImageResponse(
    // ImageResponse JSX element
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: size.width,
        height: size.height,
        backgroundImage: `url(${bgSrcBase64})`,
        backgroundSize: '100% 100%',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
      }}
    >
      {/* brand mark + wordmark */}
      <div
        style={{
          position: 'absolute',
          top: 30,
          left: 30,
          display: 'flex',
          alignItems: 'center',
          gap: 14,
        }}
      >
        {/* eslint-disable-next-line jsx-a11y/alt-text */}
        <img
          src={logoSrcBase64}
          width={logoSize.width}
          height={logoSize.height}
          style={{ borderRadius: 8 }}
        />
        <div
          style={{
            fontSize: 20,
            fontFamily: 'CommitMono',
            fontWeight: 600,
            color: 'white',
            letterSpacing: 3,
          }}
        >
          {BRAND_NAME}
        </div>
      </div>
      {/* full brand logo */}
      <div
        style={{
          position: 'absolute',
          top: (size.height - wordmarkSize.height) / 2,
          left: size.width - wordmarkSize.width - 80,
          display: 'flex',
          alignItems: 'center',
        }}
      >
        {/* eslint-disable-next-line jsx-a11y/alt-text */}
        <img src={wordmarkSrcBase64} width={wordmarkSize.width} height={wordmarkSize.height} />
      </div>
      {/* title */}
      <div
        style={{
          position: 'absolute',
          bottom: 100,
          left: 30,
          width: '560px',
          display: 'flex',
          flexDirection: 'column',
          gap: 16,
        }}
      >
        <div
          style={{
            backgroundColor: '#1F1F1F',
            padding: '2px 8px',
            borderRadius: 4,
            fontSize: 12,
            fontFamily: 'CommitMono',
            fontWeight: 600,
            color: '#CCCCCC',
            letterSpacing: 0.8,
            textTransform: 'uppercase',
          }}
        >
          {BRAND_TAGLINE}
        </div>
        <div
          style={{
            fontSize: 48,
            fontWeight: 300,
            fontFamily: 'Everett',
            color: 'white',
            lineHeight: 1,
          }}
        >
          {pageTitle}
        </div>
      </div>
    </div>,
    // ImageResponse options
    {
      // For convenience, we can re-use the exported opengraph-image
      // size config to also set the ImageResponse's width and height.
      ...size,
      fonts: [
        ...(commitMonoData
          ? [
              {
                name: 'CommitMono',
                data: commitMonoData,
                style: 'normal' as const,
                weight: 400 as const,
              },
            ]
          : []),
        ...(everettLightData
          ? [
              {
                name: 'Everett',
                data: everettLightData,
                style: 'normal' as const,
                weight: 300 as const,
              },
            ]
          : []),
      ],
    }
  );
}
