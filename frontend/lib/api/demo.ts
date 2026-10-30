/**
 * Client for the live lesion-level CRC demo (backend/app/api/routes/demo.py).
 *
 * JSON responses are validated with zod (lib/api/demo-schemas.ts) like every other endpoint; score and label
 * maps arrive as raw little-endian buffers so the browser holds the real numbers.
 */

import type { z } from 'zod';

import { API_BASE_URL, ApiError, request } from '@/lib/api/client';
import {
  demoRescuesSchema,
  demoRunSchema,
  demoStatusSchema,
  demoValiditySchema,
  type DemoRescues,
  type DemoStatus,
  type DemoRun,
  type DemoValidity,
} from '@/lib/api/demo-schemas';
import { decodeScores } from '@/lib/live-demo';

export type { DemoMethod, DemoRescues, DemoRun, DemoStatus, DemoUnit, DemoValidity, RiskKey } from '@/lib/api/demo-schemas';

function parse<T>(schema: z.ZodType<T>, data: unknown, path: string): T {
  const result = schema.safeParse(data);
  if (!result.success) {
    throw new ApiError(`Unexpected response from ${path}: ${result.error.issues[0]?.message ?? 'invalid'}`, {
      code: 'contract_mismatch',
    });
  }
  return result.data;
}

async function binary(path: string): Promise<ArrayBuffer> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) throw new ApiError(`Failed to load ${path} (HTTP ${response.status})`, { status: response.status });
  return response.arrayBuffer();
}

export const demoApi = {
  async status(): Promise<DemoStatus> {
    // The first call computes every unit's loss curves (cached afterwards), so allow time.
    const data = await request<unknown>('/api/demo/status', { timeoutMs: 300_000 });
    return parse(demoStatusSchema, data, '/api/demo/status');
  },

  async run(alpha: number, seed: number): Promise<DemoRun> {
    const data = await request<unknown>(`/api/demo/run?alpha=${alpha}&seed=${seed}`, { timeoutMs: 120_000 });
    return parse(demoRunSchema, data, '/api/demo/run');
  },

  async validity(alpha: number, reps: number): Promise<DemoValidity> {
    const data = await request<unknown>(`/api/demo/validity?alpha=${alpha}&reps=${reps}`, { timeoutMs: 300_000 });
    return parse(demoValiditySchema, data, '/api/demo/validity');
  },

  async rescues(alpha: number, splits: number): Promise<DemoRescues> {
    const data = await request<unknown>(`/api/demo/rescues?alpha=${alpha}&splits=${splits}`, { timeoutMs: 120_000 });
    return parse(demoRescuesSchema, data, '/api/demo/rescues');
  },

  async scores(uid: number, family: string): Promise<Float32Array> {
    return decodeScores(await binary(`/api/demo/units/${uid}/score.bin?family=${encodeURIComponent(family)}`));
  },

  async labels(uid: number): Promise<Uint8Array> {
    return new Uint8Array(await binary(`/api/demo/units/${uid}/labels.bin`));
  },

  async image(uid: number): Promise<ImageBitmap> {
    // Fetched as a blob so the bitmap is same-origin and never taints a canvas.
    const response = await fetch(`${API_BASE_URL}/api/demo/units/${uid}/image.jpg`);
    if (!response.ok) throw new ApiError(`Failed to load image ${uid}`, { status: response.status });
    return createImageBitmap(await response.blob());
  },

  imageUrl(uid: number): string {
    return `${API_BASE_URL}/api/demo/units/${uid}/image.jpg`;
  },
};
