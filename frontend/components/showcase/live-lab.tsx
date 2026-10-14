'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

import {
  crcIndex,
  evaluate,
  FAMILIES,
  familyLabel,
  LAB_ALPHAS,
  LAMS,
  N,
  runLab,
  S,
  TAU,
  unionIndex,
  type LabResult,
  type Unit,
} from './lab-engine';

const STEPS = [
  { id: 'data', title: 'Data', what: 'Generate scans with known lesions. Lesion confidence scales with radius^γ, and false-candidate noise bumps are added.', file: 'scripts/experiments/simulation.py · datasets/loaders.py' },
  { id: 'split', title: 'Split', what: 'Partition scans into tune / calibrate / test (20 / 40 / 40%). The tuning split is the only data used to pick the family.', file: 'scripts/experiments/mgcrc.py' },
  { id: 'posterior', title: 'Posterior', what: 'The segmenter outputs a per-pixel foreground probability. The real runs use a 5-fold cross-fitted backbone; here the posterior is simulated.', file: 'scripts/experiments/train_backbone.py' },
  { id: 'family', title: 'Score family', what: 'Transform the posterior into a score s. Thresholding keeps s = p. Lesion-equalized divides each candidate component by its peak^ρ.', file: 'app/calibration/families.py' },
  { id: 'losses', title: 'Losses', what: 'For every λ, compute pixel FNR, lesion miss and small-lesion miss per scan. A lesion is found when at least 50% of it is in the set (Lemma 2: one order statistic per lesion).', file: 'app/calibration/losses.py' },
  { id: 'calibrate', title: 'Joint threshold', what: 'Split CRC per risk, with the first λ where (n·R̂ + 1)/(n + 1) ≤ α. Take the largest λ across the three risks, so all three hold at once.', file: 'app/calibration/joint.py · union_index' },
  { id: 'select', title: 'Family selection', what: 'On the tuning split, run the joint threshold for each of the 9 families and keep the one with the smallest set. Calibration data is never used here.', file: 'app/calibration/joint.py · select_family' },
  { id: 'evaluate', title: 'Held-out test', what: 'Apply the calibrated thresholds to unseen scans and measure every risk and the set size.', file: 'app/calibration/joint.py · evaluate' },
] as const;

type StepId = (typeof STEPS)[number]['id'];

function colour(name: string): [number, number, number] {
  const c = document.createElement('canvas');
  c.width = c.height = 1;
  const ctx = c.getContext('2d');
  if (!ctx) return [128, 128, 128];
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
  ctx.fillRect(0, 0, 1, 1);
  const d = ctx.getImageData(0, 0, 1, 1).data;
  return [d[0], d[1], d[2]];
}

function paint(canvas: HTMLCanvasElement | null, u: Unit, s: Float32Array | null, cut: number, setVar: string | null) {
  if (!canvas) return 0;
  const ctx = canvas.getContext('2d');
  if (!ctx) return 0;
  const img = ctx.createImageData(S, S);
  const lo = [8, 12, 16];
  const hi = [232, 238, 242];
  const sc = setVar ? colour(setVar) : null;
  const bad = colour('--status-danger');
  const found = new Uint8Array(u.K + 1);
  if (s) {
    const cov = new Int32Array(u.K + 1);
    for (let i = 0; i < N; i++) if (u.lab[i] && s[i] >= cut) cov[u.lab[i]]++;
    for (let l = 1; l <= u.K; l++) found[l] = cov[l] >= TAU * u.sizes[l] ? 1 : 0;
  }
  for (let i = 0; i < N; i++) {
    const v = u.p[i];
    let c = [0, 1, 2].map((k) => lo[k] + (hi[k] - lo[k]) * v * (s ? 0.55 : 1));
    if (s && sc && s[i] >= cut) c = c.map((x, k) => x * 0.25 + sc[k] * 0.75);
    const l = u.lab[i];
    if (l) {
      const y = (i / S) | 0;
      const x = i - y * S;
      let edge = false;
      for (const [dy, dx] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const yy = y + dy;
        const xx = x + dx;
        if (yy < 0 || xx < 0 || yy >= S || xx >= S || u.lab[yy * S + xx] !== l) {
          edge = true;
          break;
        }
      }
      if (edge) c = s ? (found[l] ? hi : bad) : hi;
    }
    img.data[i * 4] = c[0];
    img.data[i * 4 + 1] = c[1];
    img.data[i * 4 + 2] = c[2];
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  let miss = 0;
  if (s) for (let l = 1; l <= u.K; l++) if (!found[l]) miss++;
  return miss;
}

const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '–');

function RiskCurves({ res, family, alpha }: { res: LabResult; family: number; alpha: number }) {
  const cal = res.cal.map((i) => res.stats[family][i]);
  const mean = (key: 'pix' | 'les' | 'small') => {
    const us = cal.filter((u) => u[key]);
    return Array.from(LAMS, (_, g) => us.reduce((s, u) => s + (u[key] as Float32Array)[g], 0) / Math.max(us.length, 1));
  };
  const curves: Array<[string, number[], string, number]> = [
    ['pixel FNR', mean('pix'), 'var(--status-info)', crcIndex(cal, 'pix', alpha)],
    ['lesion miss', mean('les'), 'var(--status-warn)', crcIndex(cal, 'les', alpha)],
    ['small miss', mean('small'), 'var(--status-danger)', crcIndex(cal, 'small', alpha)],
  ];
  const gU = unionIndex(cal, alpha);
  const W = 520;
  const H = 200;
  const L = 34;
  const R = 10;
  const T = 10;
  const B = 28;
  const x = (l: number) => L + (W - L - R) * l;
  const ymax = 0.6;
  const y = (v: number) => T + (H - T - B) * (1 - Math.min(v, ymax) / ymax);
  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full min-w-[420px]" role="img" aria-label="Calibration risk curves">
        {[0, 0.2, 0.4, 0.6].map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--border)" />
            <text x={L - 5} y={y(v) + 3} textAnchor="end" fontSize="9" fill="var(--muted-foreground)">{v}</text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <text key={v} x={x(v)} y={H - 10} textAnchor="middle" fontSize="9" fill="var(--muted-foreground)">{v}</text>
        ))}
        <text x={W - R} y={H - 1} textAnchor="end" fontSize="9" fill="var(--muted-foreground)">λ (larger → bigger set)</text>
        <line x1={L} x2={W - R} y1={y(alpha)} y2={y(alpha)} stroke="var(--foreground)" strokeDasharray="4 3" />
        <text x={W - R} y={y(alpha) - 4} textAnchor="end" fontSize="9" fill="var(--foreground)">α = {alpha}</text>
        {curves.map(([name, c, col, g]) => (
          <g key={name}>
            <path d={c.map((v, i) => `${i ? 'L' : 'M'}${x(LAMS[i]).toFixed(1)},${y(v).toFixed(1)}`).join('')} fill="none" stroke={col} strokeWidth={1.8} />
            <line x1={x(LAMS[g])} x2={x(LAMS[g])} y1={T} y2={H - B} stroke={col} strokeOpacity={0.5} />
          </g>
        ))}
        <line x1={x(LAMS[gU])} x2={x(LAMS[gU])} y1={T} y2={H - B} stroke="var(--foreground)" strokeWidth={2} />
        <text x={x(LAMS[gU]) + 4} y={T + 10} fontSize="9" fill="var(--foreground)">λ̂ joint = {LAMS[gU].toFixed(3)}</text>
      </svg>
      <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
        {curves.map(([name, , col, g]) => (
          <span key={name}><i className="mr-1 inline-block size-2.5 rounded-sm align-[-1px]" style={{ background: col }} />{name}: λ̂ = {LAMS[g].toFixed(3)}</span>
        ))}
      </div>
    </div>
  );
}

export function LiveLab() {
  const [gamma, setGamma] = useState(0.5);
  const [bg, setBg] = useState(0.05);
  const [seed, setSeed] = useState(7);
  const [alpha, setAlpha] = useState(0.1);
  const [res, setRes] = useState<LabResult | null>(null);
  const [ms, setMs] = useState(0);
  const [idx, setIdx] = useState(0);
  const [step, setStep] = useState<StepId>('data');
  const c0 = useRef<HTMLCanvasElement>(null);
  const c1 = useRef<HTMLCanvasElement>(null);
  const c2 = useRef<HTMLCanvasElement>(null);
  const [caps, setCaps] = useState<[string, string, string]>(['', '', '']);

  const key = `${gamma}|${bg}|${seed}`;
  const [resKey, setResKey] = useState('');
  const busy = resKey !== key;

  useEffect(() => {
    const id = window.setTimeout(() => {
      const t = performance.now();
      const r = runLab(gamma, bg, seed);
      setMs(performance.now() - t);
      setRes(r);
      setIdx((i) => Math.min(i, r.test.length - 1));
      setResKey(`${gamma}|${bg}|${seed}`);
    }, 30);
    return () => window.clearTimeout(id);
  }, [gamma, bg, seed]);

  const A = res?.byAlpha[String(alpha)];

  useEffect(() => {
    if (!res || !A) return;
    const i = res.test[idx];
    const u = res.units[i];
    paint(c0.current, u, null, 0, null);
    const m1 = paint(c1.current, u, res.scores[0][i], 1 - LAMS[A.gThr], '--status-warn');
    const m2 = paint(c2.current, u, res.scores[A.family][i], 1 - LAMS[A.gOurs], '--primary');
    setCaps([
      `${u.K} lesion${u.K > 1 ? 's' : ''}`,
      `${res.stats[0][i].size[A.gThr]} px · ${m1} missed`,
      `${res.stats[A.family][i].size[A.gOurs]} px · ${m2} missed`,
    ]);
  }, [res, A, idx]);

  const tuneTable = useMemo(() => {
    if (!res) return [];
    return FAMILIES.map((f, fi) => {
      const tu = res.tune.map((i) => res.stats[fi][i]);
      const g = unionIndex(tu, alpha);
      return { f, fi, g, str: evaluate(tu, g).str };
    });
  }, [res, alpha]);

  const jumpToDifference = () => {
    if (!res || !A) return;
    let best = idx;
    let bd = -1;
    res.test.forEach((i, k) => {
      const a = res.stats[0][i].size[A.gThr];
      const b = res.stats[A.family][i].size[A.gOurs];
      const d = Math.abs(a - b) / Math.max(res.stats[0][i].nGT, 1) + (res.units[i].K >= 2 ? 0.5 : 0);
      if (d > bd && k !== idx) {
        bd = d;
        best = k;
      }
    });
    setIdx(best);
  };

  const kHist = useMemo(() => {
    if (!res) return [];
    const h = new Map<number, number>();
    res.units.forEach((u) => h.set(u.K, (h.get(u.K) ?? 0) + 1));
    return [...h.entries()].sort((a, b) => a[0] - b[0]);
  }, [res]);

  const delta = A ? A.ours.str / A.thr.str - 1 : 0;

  const detail = () => {
    if (!res || !A) return null;
    switch (step) {
      case 'data':
        return (
          <div className="grid gap-1 text-sm">
            <p>{res.units.length} scans with at least one lesion, 64×64 px each, {res.units.reduce((s, u) => s + u.K, 0)} lesions in total.</p>
            <p>Lesions per scan: {kHist.map(([k, c]) => `${k}: ${c}`).join(' · ')}. Multi-lesion share: {((res.units.filter((u) => u.K >= 2).length / res.units.length) * 100).toFixed(0)}%.</p>
            <p>Small lesion = area below {res.smallCut} px (33rd percentile of lesion areas, from labels only).</p>
          </div>
        );
      case 'split':
        return <p className="text-sm">Tune {res.tune.length} · calibrate {res.cal.length} · test {res.test.length}. The baselines calibrate on tune + calibrate ({res.tune.length + res.cal.length}); our method calibrates on calibrate only, so the comparison is biased against it.</p>;
      case 'posterior':
        return <p className="text-sm">The left panel of the viewer shows the posterior for the selected held-out scan. White outlines are the ground-truth lesions. Faint blobs without an outline are false-candidate noise.</p>;
      case 'family':
        return <p className="text-sm">Family chosen for α = {alpha}: <b>{familyLabel(FAMILIES[A.family])}</b>. The middle panel uses thresholding and the right panel the chosen family. Both are nested sets {'{'}s ≥ 1 − λ{'}'}, so every loss is monotone in λ (Lemma 1).</p>;
      case 'losses':
      case 'calibrate':
        return <RiskCurves res={res} family={A.family} alpha={alpha} />;
      case 'select':
        return (
          <div className="overflow-x-auto">
            <table className="w-full text-xs tabular-nums">
              <thead className="text-muted-foreground"><tr className="border-b border-border"><th className="py-1 text-left">Family (tuning split)</th><th className="text-right">joint λ̂</th><th className="text-right">set / truth</th></tr></thead>
              <tbody>
                {tuneTable.map((t) => (
                  <tr key={t.fi} className={`border-b border-border/50 ${t.fi === A.family ? 'bg-primary/10 font-medium' : ''}`}>
                    <td className="py-1">{familyLabel(t.f)}{t.fi === A.family ? ' ← chosen' : ''}</td>
                    <td className="text-right">{LAMS[t.g].toFixed(3)}</td>
                    <td className="text-right">{t.str.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      case 'evaluate':
        return <p className="text-sm">Held-out set size changed by <b className={delta < -0.005 ? 'text-status-ok' : delta > 0.005 ? 'text-status-danger' : ''}>{(delta * 100).toFixed(1)}%</b> versus joint thresholding, with all three risks at or below α (table on the right). Change γ to see where the gain appears and where it vanishes.</p>;
    }
  };

  const row = (name: string, e: { pix: number; les: number; small: number; multi: number; str: number }, claims: string[], tone = '') => (
    <tr className={`border-b border-border/50 ${tone}`}>
      <td className="py-1">{name}</td>
      {(['pix', 'les', 'small'] as const).map((k) => (
        <td key={k} className={`text-right ${claims.includes(k) && e[k] > alpha + 0.005 ? 'font-semibold text-status-danger' : ''}`}>{f3(e[k])}</td>
      ))}
      <td className={`text-right ${e.multi > alpha ? 'text-status-danger' : ''}`}>{f3(e.multi)}</td>
      <td className="text-right">{e.str.toFixed(2)}</td>
    </tr>
  );

  return (
    <div className="grid gap-4">
      <div className="grid gap-4 rounded-lg border border-border bg-card p-4 md:grid-cols-4">
        <label className="grid gap-1 text-sm" htmlFor="lab-gamma">
          <span className="flex justify-between font-medium">Small-lesion faintness γ <span className="font-mono text-primary">{gamma.toFixed(2)}</span></span>
          <input id="lab-gamma" type="range" min={0} max={2} step={0.25} value={gamma} onChange={(e) => setGamma(+e.target.value)} className="accent-[var(--primary)]" />
        </label>
        <label className="grid gap-1 text-sm" htmlFor="lab-bg">
          <span className="flex justify-between font-medium">Background noise b <span className="font-mono text-primary">{bg.toFixed(2)}</span></span>
          <input id="lab-bg" type="range" min={0.02} max={0.2} step={0.01} value={bg} onChange={(e) => setBg(+e.target.value)} className="accent-[var(--primary)]" />
        </label>
        <label className="grid gap-1 text-sm" htmlFor="lab-seed">
          <span className="flex justify-between font-medium">Seed <span className="font-mono text-primary">{seed}</span></span>
          <input id="lab-seed" type="range" min={1} max={50} step={1} value={seed} onChange={(e) => setSeed(+e.target.value)} className="accent-[var(--primary)]" />
        </label>
        <div className="grid gap-1 text-sm">
          <span className="font-medium">Target risk α</span>
          <div className="inline-flex w-max overflow-hidden rounded-md border border-border" role="group" aria-label="Target risk">
            {LAB_ALPHAS.map((a) => (
              <button key={a} type="button" aria-pressed={a === alpha} onClick={() => setAlpha(a)} className={`px-3 py-1 text-xs ${a === alpha ? 'bg-primary text-primary-foreground' : 'bg-card'}`}>{a.toFixed(2)}</button>
            ))}
          </div>
          <span className="font-mono text-[11px] text-muted-foreground" aria-live="polite">{busy ? 'calibrating…' : `${res?.units.length ?? 0} scans · ${ms.toFixed(0)} ms`}</span>
        </div>
      </div>

      <ol className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-8" aria-label="Workflow steps">
        {STEPS.map((s, i) => (
          <li key={s.id}>
            <button type="button" onClick={() => setStep(s.id)} aria-pressed={step === s.id}
              className={`w-full rounded-md border px-2 py-2 text-left text-xs ${step === s.id ? 'border-primary bg-primary/10' : 'border-border bg-card hover:border-primary/50'}`}>
              <span className="block font-mono text-[10px] text-muted-foreground">step {i + 1}</span>
              <span className="font-medium">{s.title}</span>
            </button>
          </li>
        ))}
      </ol>

      <div className="grid gap-4 rounded-lg border border-border bg-card p-4">
        {(() => {
          const s = STEPS.find((x) => x.id === step)!;
          return (
            <div className="grid gap-1">
              <p className="text-sm font-semibold">{s.title}</p>
              <p className="text-sm text-muted-foreground">{s.what}</p>
              <p className="font-mono text-[11px] text-muted-foreground">{s.file}</p>
            </div>
          );
        })()}
        <div className="min-w-0">{detail()}</div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="grid gap-3 rounded-lg border border-border bg-card p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm font-semibold">Held-out scan {res ? idx + 1 : 0} of {res?.test.length ?? 0}</p>
            <div className="flex gap-2">
              <button type="button" aria-label="Previous scan" className="rounded-md border border-border px-2 py-1 text-xs" onClick={() => res && setIdx((idx - 1 + res.test.length) % res.test.length)}>←</button>
              <button type="button" aria-label="Next scan" className="rounded-md border border-border px-2 py-1 text-xs" onClick={() => res && setIdx((idx + 1) % res.test.length)}>→</button>
              <button type="button" className="rounded-md border border-border px-2 py-1 text-xs" onClick={jumpToDifference}>Show a scan where they differ</button>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {[['Posterior', c0], ['Joint thresholding', c1], ['Lesion-equalized', c2]].map(([name, ref], k) => (
              <figure key={name as string} className="grid gap-1">
                <canvas ref={ref as React.RefObject<HTMLCanvasElement>} width={S} height={S} className="aspect-square w-full rounded bg-black [image-rendering:pixelated]" aria-label={name as string} />
                <figcaption className="flex flex-wrap justify-between gap-1 text-[11px] text-muted-foreground"><span className="font-medium text-foreground">{name as string}</span><span>{caps[k]}</span></figcaption>
              </figure>
            ))}
          </div>
          <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
            <span><i className="mr-1 inline-block size-2.5 rounded-sm bg-status-warn align-[-1px]" />thresholding set</span>
            <span><i className="mr-1 inline-block size-2.5 rounded-sm bg-primary align-[-1px]" />lesion-equalized set</span>
            <span><i className="mr-1 inline-block size-2.5 rounded-sm bg-white align-[-1px] ring-1 ring-border" />lesion found</span>
            <span><i className="mr-1 inline-block size-2.5 rounded-sm bg-status-danger align-[-1px]" />lesion missed</span>
          </div>
        </div>

        <div className="grid content-start gap-3 rounded-lg border border-border bg-card p-4">
          <p className="text-sm font-semibold">Held-out results, this run</p>
          <div className="rounded-md bg-muted p-3">
            <p className={`font-mono text-2xl font-semibold ${delta < -0.005 ? 'text-status-ok' : delta > 0.005 ? 'text-status-danger' : ''}`}>{A ? `${delta > 0 ? '+' : ''}${(delta * 100).toFixed(1)}%` : '–'}</p>
            <p className="text-xs text-muted-foreground">Set size, lesion-equalized vs joint thresholding, same guarantee.</p>
          </div>
          {A && (
            <div className="overflow-x-auto">
              <table className="w-full text-xs tabular-nums">
                <thead className="text-muted-foreground"><tr className="border-b border-border"><th className="py-1 text-left">Method</th><th className="text-right">pixel</th><th className="text-right">lesion</th><th className="text-right">small</th><th className="text-right">K≥2</th><th className="text-right">set/GT</th></tr></thead>
                <tbody>
                  {row('Pixel CRC only', A.pixel, ['pix'])}
                  {row('Joint thresholding', A.thr, ['pix', 'les', 'small'], 'bg-status-warn/10')}
                  {row('Lesion-equalized', A.ours, ['pix', 'les', 'small'], 'bg-primary/10')}
                </tbody>
              </table>
            </div>
          )}
          <p className="text-[11px] text-muted-foreground">Miss rates on the held-out scans of this run. Red: above α in this single run. The guarantee is an average over calibration draws, so one draw with only about 40 small-lesion test scans can land above α; the recorded results below average 3000 draws. No method here guarantees the K≥2 column; the post-hoc conditional risk below addresses it.</p>
        </div>
      </div>
    </div>
  );
}
