'use client';

/**
 * Data hooks for the live lesion-level CRC demo (/live-demo).
 *
 * Calibration runs on the backend for a given (alpha, split seed); the per-unit
 * image, label map and score map are fetched as binary so the viewer can
 * re-threshold them in the browser while a slider moves.
 */

import { keepPreviousData, useQuery } from '@tanstack/react-query';

import { demoApi } from '@/lib/api/demo';
import { retryPolicy } from '@/hooks/use-pums';

export function useDemoStatus() {
  return useQuery({
    queryKey: ['live-demo', 'status'],
    queryFn: demoApi.status,
    staleTime: 10 * 60_000,
    retry: retryPolicy,
  });
}

export function useDemoRun(alpha: number, seed: number, enabled: boolean) {
  return useQuery({
    queryKey: ['live-demo', 'run', alpha, seed],
    queryFn: () => demoApi.run(alpha, seed),
    enabled,
    staleTime: Infinity,
    placeholderData: keepPreviousData,
    retry: retryPolicy,
  });
}

export function useDemoValidity(alpha: number, reps: number, enabled: boolean) {
  return useQuery({
    queryKey: ['live-demo', 'validity', alpha, reps],
    queryFn: () => demoApi.validity(alpha, reps),
    enabled,
    staleTime: Infinity,
    retry: retryPolicy,
  });
}

export function useDemoRescues(alpha: number, splits: number, enabled: boolean) {
  return useQuery({
    queryKey: ['live-demo', 'rescues', alpha, splits],
    queryFn: () => demoApi.rescues(alpha, splits),
    enabled,
    staleTime: Infinity,
    retry: retryPolicy,
  });
}

export interface UnitAssets {
  image: ImageBitmap;
  labels: Uint8Array;
  /** Score maps by score family (threshold, dilation, lesion_equalized). */
  scores: Record<string, Float32Array>;
}

export const SCORE_FAMILIES = ['threshold', 'dilation', 'lesion_equalized'] as const;

/**
 * Image, label map and every family's score map for one unit (~0.75 MB at 352 px).
 * No placeholder data: showing the previous unit's pixels under the next unit's
 * lesion table would be exactly the kind of mismatch this page must not make.
 */
export function useUnitAssets(uid: number | null) {
  return useQuery({
    queryKey: ['live-demo', 'unit', uid],
    queryFn: async (): Promise<UnitAssets> => {
      const id = uid as number;
      const [image, labels, ...maps] = await Promise.all([
        demoApi.image(id),
        demoApi.labels(id),
        ...SCORE_FAMILIES.map((f) => demoApi.scores(id, f)),
      ]);
      return { image, labels, scores: Object.fromEntries(SCORE_FAMILIES.map((f, i) => [f, maps[i]])) };
    },
    enabled: uid !== null && typeof window !== 'undefined',
    staleTime: Infinity,
    gcTime: 5 * 60_000,
    retry: 1,
  });
}
