/**
 * Centralised API client.
 *
 * Every network call in the application goes through `request()`. That is what
 * makes timeout handling, error normalisation and the backend-unavailable
 * state consistent instead of being reimplemented per component.
 */

import type { ApiErrorBody } from '@/types/api';

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') ?? 'http://localhost:8000';

/** Analysis runs a full inference + calibration pipeline; be generous. */
export const ANALYSIS_TIMEOUT_MS = 120_000;
export const DEFAULT_TIMEOUT_MS = 20_000;

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: Record<string, unknown>;
  /** True when we never reached the server (down, CORS, DNS, offline). */
  readonly isNetworkError: boolean;
  readonly isTimeout: boolean;

  constructor(
    message: string,
    options: {
      status?: number;
      code?: string;
      detail?: Record<string, unknown>;
      isNetworkError?: boolean;
      isTimeout?: boolean;
    } = {},
  ) {
    super(message);
    this.name = 'ApiError';
    this.status = options.status ?? 0;
    this.code = options.code ?? 'unknown_error';
    this.detail = options.detail ?? {};
    this.isNetworkError = options.isNetworkError ?? false;
    this.isTimeout = options.isTimeout ?? false;
  }

  /** A message worth showing a user, with a hint about what to do next. */
  get userMessage(): string {
    if (this.isTimeout) {
      return 'The backend took too long to respond. The first analysis after startup builds the calibration set and can take up to a minute.';
    }
    if (this.isNetworkError) {
      return `Cannot reach the PUMS backend at ${API_BASE_URL}. Start it with "uvicorn app.main:app --reload" in the backend directory.`;
    }
    return this.message;
  }
}

interface RequestOptions extends Omit<RequestInit, 'signal'> {
  timeoutMs?: number;
}

export async function request<T>(
  path: string,
  { timeoutMs = DEFAULT_TIMEOUT_MS, ...init }: RequestOptions = {},
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: 'application/json',
        ...(init.headers ?? {}),
      },
    });
  } catch (error) {
    const aborted = error instanceof DOMException && error.name === 'AbortError';
    throw new ApiError(
      aborted ? 'Request timed out' : 'Network request failed',
      { isNetworkError: !aborted, isTimeout: aborted, code: aborted ? 'timeout' : 'network_error' },
    );
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    // The backend always returns {code, message, detail}; tolerate anything else.
    let body: Partial<ApiErrorBody> = {};
    try {
      body = (await response.json()) as ApiErrorBody;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(body.message ?? `Request failed (HTTP ${response.status})`, {
      status: response.status,
      code: body.code ?? `http_${response.status}`,
      detail: body.detail ?? {},
    });
  }

  if (response.status === 204) return undefined as T;

  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError('Backend returned a malformed response body', {
      status: response.status,
      code: 'invalid_response',
    });
  }
}

/** Absolute URL for a media path the API returned (e.g. an uncertainty PNG). */
export function mediaUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  return path.startsWith('http') ? path : `${API_BASE_URL}${path}`;
}
