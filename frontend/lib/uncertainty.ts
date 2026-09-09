/**
 * Pure uncertainty-map helpers.
 *
 * Deliberately free of React and of any data-fetching dependency so this logic
 * is directly unit-testable — `hooks/use-uncertainty-matrix.ts` re-exports it
 * for components.
 */

export interface UncertaintyMatrix {
  width: number;
  height: number;
  /** Row-major values in [0, 1]. */
  values: Float32Array;
  at: (x: number, y: number) => number;
}

/**
 * Colour ramp mirroring `backend/app/utils/encoding.py::_HEATMAP_STOPS`.
 *
 * Keeping the two in sync matters: it is what makes the browser-rendered
 * heatmap and the backend-rendered PNG the same picture of the same numbers.
 */
export const RAMP: Array<[number, [number, number, number]]> = [
  [0.0, [12, 24, 58]],
  [0.25, [24, 96, 160]],
  [0.5, [46, 168, 152]],
  [0.75, [238, 186, 62]],
  [1.0, [206, 52, 48]],
];

export function uncertaintyColour(value: number): [number, number, number] {
  const v = Math.min(Math.max(value, 0), 1);
  for (let i = 0; i < RAMP.length - 1; i += 1) {
    const [p0, c0] = RAMP[i];
    const [p1, c1] = RAMP[i + 1];
    if (v <= p1) {
      const t = (v - p0) / (p1 - p0 || 1);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * t),
        Math.round(c0[1] + (c1[1] - c0[1]) * t),
        Math.round(c0[2] + (c1[2] - c0[2]) * t),
      ];
    }
  }
  return RAMP[RAMP.length - 1][1];
}

/** Build a matrix from raw values (used by the decoder and by tests). */
export function makeMatrix(
  values: Float32Array,
  width: number,
  height: number,
): UncertaintyMatrix {
  return {
    width,
    height,
    values,
    at: (x, y) => values[y * width + x] ?? 0,
  };
}

/**
 * Decode 8-bit grayscale RGBA pixel data into [0, 1] uncertainty values.
 *
 * This is the exact inverse of the backend's `quantise_map`: value/255 recovers
 * the original float to quantisation precision (±1/510).
 */
export function decodeGrayscale(
  data: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
): UncertaintyMatrix {
  // Fail loudly on a degenerate decode. A 0x0 matrix is not a valid map, and
  // silently returning one pushes the failure into a canvas call several
  // components away where the cause is unrecoverable from the stack trace.
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error(
      `Invalid uncertainty map dimensions: ${width}x${height}. ` +
        'The source image was empty or its dimensions were read after the ' +
        'ImageBitmap had been closed.',
    );
  }
  if (data.length < width * height * 4) {
    throw new Error(
      `Uncertainty map pixel data is too short: got ${data.length} bytes, ` +
        `expected at least ${width * height * 4} for ${width}x${height} RGBA.`,
    );
  }

  const values = new Float32Array(width * height);
  for (let i = 0; i < values.length; i += 1) {
    values[i] = data[i * 4] / 255;
  }
  return makeMatrix(values, width, height);
}

export function matrixHistogram(matrix: UncertaintyMatrix, bins = 20): number[] {
  const counts = new Array<number>(bins).fill(0);
  for (let i = 0; i < matrix.values.length; i += 1) {
    const bin = Math.min(Math.floor(matrix.values[i] * bins), bins - 1);
    counts[bin] += 1;
  }
  return counts;
}

export function matrixStats(matrix: UncertaintyMatrix, threshold: number) {
  let sum = 0;
  let max = 0;
  let above = 0;
  for (let i = 0; i < matrix.values.length; i += 1) {
    const v = matrix.values[i];
    sum += v;
    if (v > max) max = v;
    if (v >= threshold) above += 1;
  }
  const n = matrix.values.length || 1;
  return { mean: sum / n, max, aboveFraction: above / n, count: n };
}
