import { type CapturedFrame, captureVideoFrame } from './capture';
import { VISUAL_PROMPT } from './contract';

interface Dependencies {
  expectedUser?: string;
  signal?: AbortSignal;
  capture?: (video: HTMLVideoElement) => Promise<CapturedFrame>;
  csrf?: () => Promise<string>;
  request?: typeof fetch;
}

async function csrfToken(expectedUser?: string): Promise<string> {
  const response = await fetch('/api/auth/me', {
    cache: 'no-store',
    credentials: 'same-origin',
  });
  if (!response.ok) throw new Error('not_signed_in');
  const body = (await response.json()) as {
    csrfToken?: unknown;
    authenticated?: boolean;
    user?: { userId?: string };
  };
  if (expectedUser && (body.authenticated !== true || body.user?.userId !== expectedUser)) {
    throw new Error('not_signed_in');
  }
  if (typeof body.csrfToken !== 'string' || body.csrfToken.length < 16) {
    throw new Error('csrf_unavailable');
  }
  return body.csrfToken;
}

export async function analyzeCameraView(
  video: HTMLVideoElement,
  companyPrivate: boolean,
  dependencies: Dependencies = {}
): Promise<string> {
  // This check deliberately precedes capture: a company-private page creates no frame.
  if (companyPrivate) throw new Error('company_mode_blocked');

  let frame: CapturedFrame | null = null;
  try {
    dependencies.signal?.throwIfAborted();
    const csrf = await (dependencies.csrf ?? (() => csrfToken(dependencies.expectedUser)))();
    // Capture only after prerequisites are ready, keeping frame bytes alive
    // for the single analysis request and no longer.
    dependencies.signal?.throwIfAborted();
    frame = await (dependencies.capture ?? captureVideoFrame)(video);
    dependencies.signal?.throwIfAborted();
    let response: Response;
    try {
      response = await (dependencies.request ?? fetch)('/api/visual/analyze', {
        method: 'POST',
        signal: dependencies.signal,
        cache: 'no-store',
        credentials: 'same-origin',
        headers: {
          'Content-Type': 'application/json',
          'X-CAAL-CSRF': csrf,
          ...(dependencies.expectedUser ? { 'X-CAAL-Visual-User': dependencies.expectedUser } : {}),
        },
        body: JSON.stringify({
          image: frame.jpegBase64,
          prompt: VISUAL_PROMPT,
          company_private: false,
        }),
      });
    } catch {
      throw new Error('analysis_unavailable');
    }
    const body = (await response.json().catch(() => null)) as {
      description?: unknown;
      error?: unknown;
    } | null;
    if (!response.ok) {
      const code = typeof body?.error === 'string' ? body.error : 'analysis_unavailable';
      throw new Error(code);
    }
    if (typeof body?.description !== 'string' || body.description.length > 1200) {
      throw new Error('analysis_unavailable');
    }
    if (dependencies.expectedUser && (await csrfToken(dependencies.expectedUser)) !== csrf) {
      throw new Error('not_signed_in');
    }
    dependencies.signal?.throwIfAborted();
    return body.description;
  } finally {
    frame?.clear();
    frame = null;
  }
}
