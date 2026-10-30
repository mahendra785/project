/**
 * Pure helpers for the live lesion-level CRC demo (/live-demo).
 *
 * Free of React and DOM so they are unit-tested directly (tests/live-demo.test.mts).
 * The backend sends each unit's score map as uint16 (score x 65535) and its
 * ground-truth lesions as a uint8 label map; everything drawn in the viewer is
 * computed from those numbers here, so moving the threshold slider re-derives the
 * prediction set and which lesions it finds, exactly as the calibration code does.
 */

import { uncertaintyColour } from './uncertainty.ts';

/** Fraction of a lesion that must lie in the set for it to count as found. */
export const TAU = 0.5;

export function decodeScores(buffer: ArrayBuffer): Float32Array {
  const raw = new Uint16Array(buffer);
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw[i] / 65535;
  return out;
}

/** Posterior cutoff of grid index i on the 1001-point lambda grid: S = {s >= 1 - lambda}. */
export function cutoffOfIndex(i: number, gridSize = 1001): number {
  return 1 - i / (gridSize - 1);
}

/**
 * Per-lesion coverage: the fraction of lesion k's pixels with score >= cutoff (index 0 = lesion 1).
 * The uint16 quantisation is applied to the cutoff too, so a pixel exactly at the
 * cutoff counts as inside, matching the backend's float comparison up to 1/65535.
 */
export function lesionCoverage(scores: Float32Array, labels: Uint8Array, cutoff: number, k: number): number[] {
  const inside = new Array<number>(k).fill(0);
  const total = new Array<number>(k).fill(0);
  const c = Math.round(cutoff * 65535) / 65535 - 1e-9;
  for (let i = 0; i < labels.length; i++) {
    const lab = labels[i];
    if (lab === 0 || lab > k) continue;
    total[lab - 1]++;
    if (scores[i] >= c) inside[lab - 1]++;
  }
  return inside.map((v, j) => (total[j] ? v / total[j] : 0));
}

export function setSize(scores: Float32Array, cutoff: number): number {
  const c = Math.round(cutoff * 65535) / 65535 - 1e-9;
  let n = 0;
  for (let i = 0; i < scores.length; i++) if (scores[i] >= c) n++;
  return n;
}

/** Pixel false-negative rate: share of lesion pixels outside the set. */
export function pixelFnr(scores: Float32Array, labels: Uint8Array, cutoff: number): number {
  const c = Math.round(cutoff * 65535) / 65535 - 1e-9;
  let gt = 0;
  let missed = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === 0) continue;
    gt++;
    if (scores[i] < c) missed++;
  }
  return gt ? missed / gt : 0;
}

/** Label of each boundary pixel (a lesion pixel with a 4-neighbour of another label), else 0. */
export function outline(labels: Uint8Array, width: number, height: number): Uint8Array {
  const out = new Uint8Array(labels.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const lab = labels[i];
      if (!lab) continue;
      const edge =
        x === 0 || y === 0 || x === width - 1 || y === height - 1 ||
        labels[i - 1] !== lab || labels[i + 1] !== lab || labels[i - width] !== lab || labels[i + width] !== lab;
      if (edge) out[i] = lab;
    }
  }
  return out;
}

/** Centroid (x, y) of each lesion label 1..k, for drawing its number on the image. */
export function lesionCentroids(labels: Uint8Array, width: number, k: number): Array<[number, number]> {
  const sx = new Array<number>(k).fill(0);
  const sy = new Array<number>(k).fill(0);
  const n = new Array<number>(k).fill(0);
  for (let i = 0; i < labels.length; i++) {
    const lab = labels[i];
    if (lab === 0 || lab > k) continue;
    sx[lab - 1] += i % width;
    sy[lab - 1] += Math.floor(i / width);
    n[lab - 1]++;
  }
  return n.map((c, j) => (c ? [sx[j] / c, sy[j] / c] : [0, 0]));
}

/**
 * The threshold scrubber is log-scaled in the cutoff, because the interesting
 * cutoffs (where small, faint polyps are found) sit within 0.001-0.05 of zero.
 * Position p in [0, 1] maps to a grid index; both directions snap to the grid.
 */
export const SCRUB_MIN_CUTOFF = 1e-3;

export function cutoffToPosition(cutoff: number): number {
  const c = Math.min(1, Math.max(SCRUB_MIN_CUTOFF, cutoff));
  return (Math.log10(c) - Math.log10(SCRUB_MIN_CUTOFF)) / -Math.log10(SCRUB_MIN_CUTOFF);
}

export function positionToIndex(p: number, gridSize = 1001): number {
  const c = Math.pow(10, Math.log10(SCRUB_MIN_CUTOFF) * (1 - Math.min(1, Math.max(0, p))));
  return Math.min(gridSize - 1, Math.max(0, Math.round((1 - c) * (gridSize - 1))));
}

export type Rgb = [number, number, number];

export const OVERLAY = {
  set: [45, 212, 191] as Rgb,
  found: [34, 197, 94] as Rgb,
  missed: [239, 68, 68] as Rgb,
  truth: [250, 204, 21] as Rgb,
};

/** Thicken an outline by one pixel so it stays visible when the canvas is scaled down. */
function dilateOutline(edge: Uint8Array, width: number, height: number): Uint8Array {
  const out = new Uint8Array(edge);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const lab = edge[y * width + x];
      if (!lab) continue;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const xx = x + dx;
        const yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < width && yy < height && !out[yy * width + xx]) out[yy * width + xx] = lab;
      }
    }
  }
  return out;
}

/**
 * RGBA overlay drawn over the image: the prediction set as a translucent fill and each
 * lesion's outline in green (found) or red (missed). With `found` omitted, outlines are
 * drawn in the neutral truth colour and no set is shown.
 */
export function overlayRgba(
  scores: Float32Array | null,
  labels: Uint8Array,
  width: number,
  height: number,
  cutoff: number | null,
  found: boolean[] | null,
  setAlpha = 110,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(width * height * 4);
  if (scores && cutoff !== null) {
    const c = Math.round(cutoff * 65535) / 65535 - 1e-9;
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] >= c) {
        out[i * 4] = OVERLAY.set[0];
        out[i * 4 + 1] = OVERLAY.set[1];
        out[i * 4 + 2] = OVERLAY.set[2];
        out[i * 4 + 3] = setAlpha;
      }
    }
  }
  const edge = dilateOutline(outline(labels, width, height), width, height);
  for (let i = 0; i < edge.length; i++) {
    const lab = edge[i];
    if (!lab) continue;
    const colour = found ? (found[lab - 1] ? OVERLAY.found : OVERLAY.missed) : OVERLAY.truth;
    out[i * 4] = colour[0];
    out[i * 4 + 1] = colour[1];
    out[i * 4 + 2] = colour[2];
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** Heat-map of a score map with the project's uncertainty colour ramp. */
export function heatmapRgba(scores: Float32Array): Uint8ClampedArray {
  const out = new Uint8ClampedArray(scores.length * 4);
  for (let i = 0; i < scores.length; i++) {
    const [r, g, b] = uncertaintyColour(scores[i]);
    out[i * 4] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** Log-spaced sample of grid indices for plotting curves against the cutoff on a log axis. */
export function logSampleIndices(gridSize = 1001, points = 160): number[] {
  const out = new Set<number>();
  const minCut = 1 / (gridSize - 1);
  for (let j = 0; j <= points; j++) {
    const c = Math.pow(10, Math.log10(minCut) * (1 - j / points));
    out.add(Math.min(gridSize - 2, Math.max(0, Math.round((1 - c) * (gridSize - 1)))));
  }
  return [...out].sort((a, b) => b - a);
}

/** Compact method names for chart axes (the backend labels are written for tables). */
const SHORT_LABELS: Record<string, string> = {
  pixel: 'Pixel CRC',
  joint: 'Joint CRC (ours)',
  joint_multi: 'Joint + multi (ours)',
  ltt: 'Learn-Then-Test',
  dilation: 'Dilation family',
  strata: 'CCRA-S strata',
  lesion_eq: 'Lesion-equalized (ours)',
};

export function shortLabel(id: string, fallback: string): string {
  return SHORT_LABELS[id] ?? fallback;
}

const RISK_NAMES: Record<string, string> = { pix: 'pixel', les: 'polyp', small: 'small-polyp', multi: 'multi-polyp' };

/** "Certifies pixel, polyp and small-polyp miss ≤ α (expectation, each risk)". */
export function certifies(guaranteed: string[], guarantee: string): string {
  const names = guaranteed.map((r) => RISK_NAMES[r] ?? r);
  const list = names.length > 1 ? `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}` : names[0];
  return `Certifies ${list} miss ≤ α (${guarantee})`;
}

export function pct(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined || !Number.isFinite(v) ? '—' : `${(100 * v).toFixed(digits)}%`;
}

export function times(v: number | null | undefined, alpha: number, digits = 2): string {
  return v === null || v === undefined || !Number.isFinite(v) ? '—' : `${(v / alpha).toFixed(digits)}×α`;
}
