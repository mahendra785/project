'use client';

/**
 * Decode the backend's uncertainty PNG back into a real numeric matrix.
 *
 * The backend transports the dense map as an 8-bit grayscale PNG (~15 kB for a
 * 256x256 map, versus ~1.5 MB as JSON floats). This hook reverses that exactly:
 * channel value / 255 recovers the original [0, 1] uncertainty to quantisation
 * precision, giving the client genuine numbers to threshold and colour-map —
 * not a pre-rendered picture it can only display.
 *
 * The image is fetched via `fetch` into a Blob rather than assigned to an
 * `img.src` cross-origin, because a Blob URL is same-origin and therefore never
 * taints the canvas — `getImageData` would otherwise throw.
 *
 * The pure maths lives in `lib/uncertainty.ts` so it can be unit-tested without
 * a DOM.
 */

import { useQuery } from '@tanstack/react-query';

import { decodeGrayscale, type UncertaintyMatrix } from '@/lib/uncertainty';

export {
  matrixHistogram,
  matrixStats,
  uncertaintyColour,
  type UncertaintyMatrix,
} from '@/lib/uncertainty';

async function decodeMatrix(url: string): Promise<UncertaintyMatrix> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load uncertainty map (HTTP ${response.status})`);
  }
  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);

  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Canvas 2D context unavailable');

  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();

  return decodeGrayscale(data, bitmap.width, bitmap.height);
}

export function useUncertaintyMatrix(url: string | null) {
  return useQuery({
    queryKey: ['uncertainty-matrix', url],
    queryFn: () => decodeMatrix(url as string),
    enabled: Boolean(url) && typeof window !== 'undefined',
    staleTime: Infinity,
    gcTime: 10 * 60_000,
    retry: 1,
  });
}
