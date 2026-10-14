/**
 * In-browser port of the lesion-level joint CRC pipeline, for the showcase.
 *
 * Mirrors backend/app/calibration/{families,losses,joint}.py on synthetic
 * 64x64 scans: lesion-equalized scores, Lemma-2 lesion statistics, split CRC
 * per risk, union threshold, and family selection on a tuning split. Pure
 * functions only; no DOM access.
 */

export const S = 64;
export const N = S * S;
export const G = 201;
export const TAU = 0.5;
export const LAMS = Float32Array.from({ length: G }, (_, g) => g / (G - 1));

export interface Family {
  rho: number;
  t0: number;
}

export const FAMILIES: Family[] = [{ rho: 0, t0: 0 }].concat(
  [0.05, 0.1, 0.2, 0.3].flatMap((t0) => [0.5, 1].map((rho) => ({ rho, t0 }))),
);

export const familyLabel = (f: Family) => (f.rho === 0 ? 'thresholding' : `ρ=${f.rho}, t₀=${f.t0}`);

export interface Unit {
  p: Float32Array;
  lab: Int32Array;
  K: number;
  sizes: Int32Array;
}

export interface UnitStats {
  pix: Float32Array;
  les: Float32Array;
  small: Float32Array | null;
  size: Float32Array;
  nGT: number;
  multi: boolean;
}

export interface Evaluation {
  pix: number;
  les: number;
  small: number;
  multi: number;
  str: number;
}

export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const normal = (r: () => number) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());

function poisson(r: () => number, l: number) {
  const L = Math.exp(-l);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= r();
  } while (p > L);
  return k - 1;
}

/** 8-connected component labelling. */
export function label8(mask: Uint8Array): { lab: Int32Array; n: number } {
  const lab = new Int32Array(N);
  const stack = new Int32Array(N);
  let n = 0;
  for (let i = 0; i < N; i++) {
    if (!mask[i] || lab[i]) continue;
    n++;
    let sp = 0;
    stack[sp++] = i;
    lab[i] = n;
    while (sp) {
      const j = stack[--sp];
      const y = (j / S) | 0;
      const x = j - y * S;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy;
          const xx = x + dx;
          if (yy < 0 || xx < 0 || yy >= S || xx >= S) continue;
          const k = yy * S + xx;
          if (mask[k] && !lab[k]) {
            lab[k] = n;
            stack[sp++] = k;
          }
        }
    }
  }
  return { lab, n };
}

/** Same generator as scripts/experiments/simulation.py, scaled to 64x64. */
export function generateUnit(r: () => number, gamma: number, b: number): Unit {
  const p = new Float32Array(N);
  const gt = new Uint8Array(N);
  const RMIN = 1.5;
  const RMAX = 6;
  for (let i = 0; i < N; i++) p[i] = r() * b;
  const k = 1 + poisson(r, 1.5);
  for (let q = 0; q < k; q++) {
    const cy = RMAX + r() * (S - 2 * RMAX);
    const cx = RMAX + r() * (S - 2 * RMAX);
    const rad = RMIN + r() * (RMAX - RMIN);
    const a = Math.min(1, Math.max(0.05, 0.9 * Math.pow(rad / RMAX, gamma) + 0.05 * normal(r)));
    for (let y = 0; y < S; y++)
      for (let x = 0; x < S; x++) {
        const d2 = (y - cy) ** 2 + (x - cx) ** 2;
        const i = y * S + x;
        const v = a * Math.exp(-d2 / (2 * rad * rad));
        if (v > p[i]) p[i] = v;
        if (d2 <= rad * rad) gt[i] = 1;
      }
  }
  const nb = poisson(r, 3);
  for (let q = 0; q < nb; q++) {
    const cy = r() * S;
    const cx = r() * S;
    const rad = RMIN + r() * (RMAX / 2 - RMIN);
    const a = 0.1 + r() * 0.25;
    for (let y = 0; y < S; y++)
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        const v = a * Math.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * rad * rad));
        if (v > p[i]) p[i] = v;
      }
  }
  const { lab, n } = label8(gt);
  const sizes = new Int32Array(n + 1);
  for (let i = 0; i < N; i++) if (lab[i]) sizes[lab[i]]++;
  return { p, lab, K: n, sizes };
}

/** Lesion-equalized score: p / peak^rho inside each candidate of {p >= t0}. */
export function equalize(p: Float32Array, f: Family): Float32Array {
  if (f.rho === 0) return p;
  const m = new Uint8Array(N);
  for (let i = 0; i < N; i++) m[i] = p[i] >= f.t0 ? 1 : 0;
  const { lab, n } = label8(m);
  const peak = new Float32Array(n + 1);
  for (let i = 0; i < N; i++) if (lab[i] && p[i] > peak[lab[i]]) peak[lab[i]] = p[i];
  const s = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const l = lab[i];
    s[i] = l ? Math.min(1, p[i] / Math.pow(Math.max(peak[l], f.t0), f.rho)) : p[i];
  }
  return s;
}

function lowerBound(arr: ArrayLike<number>, v: number) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (arr[m] < v) lo = m + 1;
    else hi = m;
  }
  return lo;
}

/** Grid-reduced losses for one unit (Lemma 2 for the lesion terms). */
export function unitStats(s: Float32Array, u: Unit, smallCut: number): UnitStats {
  const all = Float32Array.from(s).sort();
  const gtv: number[] = [];
  const comp: number[][] = Array.from({ length: u.K + 1 }, () => []);
  for (let i = 0; i < N; i++) {
    const l = u.lab[i];
    if (l) {
      gtv.push(s[i]);
      comp[l].push(s[i]);
    }
  }
  const gts = Float32Array.from(gtv).sort();
  const q: number[] = [];
  const qs: number[] = [];
  for (let l = 1; l <= u.K; l++) {
    const c = comp[l].sort((a, b) => b - a);
    const v = c[Math.ceil(TAU * c.length) - 1];
    q.push(v);
    if (u.sizes[l] < smallCut) qs.push(v);
  }
  q.sort((a, b) => a - b);
  qs.sort((a, b) => a - b);
  const pix = new Float32Array(G);
  const les = new Float32Array(G);
  const small = qs.length ? new Float32Array(G) : null;
  const size = new Float32Array(G);
  for (let g = 0; g < G; g++) {
    const c = 1 - LAMS[g];
    size[g] = N - lowerBound(all, c);
    pix[g] = lowerBound(gts, c) / gts.length;
    les[g] = lowerBound(q, c) / q.length;
    if (small) small[g] = lowerBound(qs, c) / qs.length;
  }
  return { pix, les, small, size, nGT: gts.length, multi: u.K >= 2 };
}

type RiskKey = 'pix' | 'les' | 'small';

/** Split CRC: first grid index with (n R + 1) / (n + 1) <= alpha. */
export function crcIndex(units: UnitStats[], key: RiskKey, alpha: number) {
  const us = units.filter((u) => u[key]);
  const n = us.length;
  if (!n) return 0;
  for (let g = 0; g < G; g++) {
    let s = 0;
    for (const u of us) s += (u[key] as Float32Array)[g];
    if ((s + 1) / (n + 1) <= alpha) return g;
  }
  return G - 1;
}

export const unionIndex = (us: UnitStats[], a: number) =>
  Math.max(crcIndex(us, 'pix', a), crcIndex(us, 'les', a), crcIndex(us, 'small', a));

export function evaluate(us: UnitStats[], g: number): Evaluation {
  let p = 0;
  let l = 0;
  let sm = 0;
  let ns = 0;
  let lm = 0;
  let nm = 0;
  let sz = 0;
  let gt = 0;
  for (const u of us) {
    p += u.pix[g];
    l += u.les[g];
    if (u.small) {
      sm += u.small[g];
      ns++;
    }
    if (u.multi) {
      lm += u.les[g];
      nm++;
    }
    sz += u.size[g];
    gt += u.nGT;
  }
  return { pix: p / us.length, les: l / us.length, small: ns ? sm / ns : NaN, multi: nm ? lm / nm : NaN, str: sz / gt };
}

export interface LabResult {
  units: Unit[];
  scores: Float32Array[][];
  stats: UnitStats[][];
  tune: number[];
  cal: number[];
  test: number[];
  smallCut: number;
  byAlpha: Record<string, {
    gPix: number;
    gThr: number;
    gOurs: number;
    family: number;
    pixel: Evaluation;
    thr: Evaluation;
    ours: Evaluation;
  }>;
}

export const LAB_ALPHAS = [0.05, 0.1, 0.15];

/** The whole workflow: generate → split → score → reduce → calibrate → select → evaluate. */
export function runLab(gamma: number, b: number, seed: number, count = 300): LabResult {
  const r = rng(seed * 1009 + Math.round(gamma * 100) + Math.round(b * 1000) * 7);
  const units = Array.from({ length: count }, () => generateUnit(r, gamma, b)).filter((u) => u.K > 0);
  const sizes = units.flatMap((u) => Array.from(u.sizes.slice(1))).sort((x, y) => x - y);
  const smallCut = sizes[Math.floor(0.33 * sizes.length)];
  const order = units.map((_, i) => i);
  const r2 = rng(seed * 31 + 5);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(r2() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const nt = Math.round(order.length * 0.2);
  const nc = Math.round(order.length * 0.4);
  const tune = order.slice(0, nt);
  const cal = order.slice(nt, nt + nc);
  const test = order.slice(nt + nc);
  const scores = FAMILIES.map((f) => units.map((u) => equalize(u.p, f)));
  const stats = FAMILIES.map((_, fi) => units.map((u, i) => unitStats(scores[fi][i], u, smallCut)));
  const pick = (fi: number, ids: number[]) => ids.map((i) => stats[fi][i]);
  const byAlpha: LabResult['byAlpha'] = {};
  for (const a of LAB_ALPHAS) {
    const tc = tune.concat(cal);
    const gPix = crcIndex(pick(0, tc), 'pix', a);
    const gThr = unionIndex(pick(0, tc), a);
    let family = 0;
    let best = Infinity;
    FAMILIES.forEach((_, fi) => {
      const tu = pick(fi, tune);
      const e = evaluate(tu, unionIndex(tu, a)).str;
      if (e < best - 1e-12) {
        best = e;
        family = fi;
      }
    });
    const gOurs = unionIndex(pick(family, cal), a);
    byAlpha[String(a)] = {
      gPix,
      gThr,
      gOurs,
      family,
      pixel: evaluate(pick(0, test), gPix),
      thr: evaluate(pick(0, test), gThr),
      ours: evaluate(pick(family, test), gOurs),
    };
  }
  return { units, scores, stats, tune, cal, test, smallCut, byAlpha };
}
