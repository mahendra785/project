/**
 * Server-only loader for the showcase.
 *
 * Reads the committed result files under docs/research/results/ directly, so the
 * showcase needs no backend changes and shows exactly what the scripts wrote.
 * Any file that is missing (for example in a slim Docker image) becomes null,
 * and the page says so instead of failing.
 */

import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';

export interface ResultRow {
  alpha: number;
  method: string;
  risk_pix: number | null;
  risk_les: number | null;
  risk_small: number | null;
  risk_les_multi: number | null;
  set_to_truth: number | null;
  set_to_truth_lo: number | null;
  set_to_truth_hi: number | null;
  fp_to_truth: number | null;
  lam: number | null;
  rel?: number | null;
  rel_lo?: number | null;
  rel_hi?: number | null;
  p_smaller?: number | null;
  n_seeds: number;
}

export interface GapRow {
  alpha: number;
  G: number;
  G_multi: number;
  G_multi_lo: number;
  G_multi_hi: number;
  G_small: number;
  G_small_lo: number;
  G_small_hi: number;
  n_cal_multi: number;
  n_cal_small: number;
}

export interface Block {
  dataset: string;
  backbone: string;
  tau: number;
  mean_dice: number | null;
  meta: Array<{ n_units: number; n_multi: number; n_with_small: number; small_cut_px: number; reps: number; seed: number }>;
  rows: ResultRow[];
  gap: GapRow[];
  verdicts: { H1: string; H2: Record<string, string>; validity: Record<string, string> };
  chosen_family: Record<string, number>;
}

export interface SimRow {
  gamma: number;
  background: number;
  alpha: number;
  rel: number;
  lo: number;
  hi: number;
  p_smaller: number;
}

export interface DiagRow {
  gamma: number;
  background: number;
  frac_small_lesions_below_t0: number;
  frac_small_lesions_not_above_background_p99: number;
}

export interface E8Row {
  alpha: number;
  thr_joint3_multi: { mean: number };
  thr_joint4_multi: { mean: number };
  thr_joint3_str: { mean: number };
  thr_joint4_str: { mean: number };
  price_of_multi: { mean: number; lo: number; hi: number };
}

export interface BackboneManifest {
  dataset: string;
  backbone: string;
  seed: number;
  folds: number;
  size: number;
  n_units: number;
  fold_dice: number[];
  mean_dice: number;
  sha256: string;
  seconds: number;
}

export interface ShowcaseData {
  blocks: Block[] | null;
  sim: { verdict: string; rows: SimRow[] } | null;
  diag: DiagRow[] | null;
  e8: Record<string, E8Row[]>;
  backbones: BackboneManifest[];
  source: string;
}

const RESULTS = path.resolve(process.cwd(), '..', 'docs', 'research', 'results');

async function json<T>(file: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(path.join(RESULTS, file), 'utf8')) as T;
  } catch {
    return null;
  }
}

export async function loadShowcaseData(): Promise<ShowcaseData> {
  const [blocks, sim, diag] = await Promise.all([
    json<Block[]>('mgcrc_summary.json'),
    json<{ verdict: string; rows: SimRow[] }>('simulation_E7.json'),
    json<{ rows: DiagRow[] }>('simulation_E7_posthoc_diagnostic.json'),
  ]);
  const e8: Record<string, E8Row[]> = {};
  for (const ds of ['fracatlas', 'kvasir-seg']) {
    const d = await json<{ summary: E8Row[] }>(`posthoc_E8_${ds}_rf.json`);
    if (d) e8[ds] = d.summary;
  }
  let backbones: BackboneManifest[] = [];
  try {
    const files = (await readdir(path.join(RESULTS, 'backbones'))).filter((f) => f.endsWith('.json'));
    backbones = (
      await Promise.all(files.map((f) => json<BackboneManifest>(path.join('backbones', f))))
    ).filter((b): b is BackboneManifest => b !== null);
  } catch {
    backbones = [];
  }
  return { blocks, sim, diag: diag?.rows ?? null, e8, backbones, source: 'docs/research/results/' };
}
