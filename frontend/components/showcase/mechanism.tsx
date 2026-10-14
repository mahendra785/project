'use client';

import { useMemo, useState } from 'react';

import { rng, TAU } from './lab-engine';

const X = Array.from({ length: 281 }, (_, i) => i / 280);
const bump = (x: number, c: number, w: number, a: number) => a * Math.exp(-((x - c) ** 2) / (2 * w * w));
const NOISE = (() => {
  const r = rng(3);
  return X.map(() => 0.06 * r());
})();
const P = X.map((x, i) =>
  Math.max(NOISE[i], bump(x, 0.3, 0.07, 0.92), bump(x, 0.74, 0.025, 0.2), bump(x, 0.55, 0.012, 0.19)),
);
const LESIONS: Array<[number, number, string]> = [
  [0.23, 0.37, 'large, confident'],
  [0.715, 0.765, 'small, faint'],
];
const T0 = 0.1;

function equalized(rho: number) {
  const s = P.slice();
  let start = -1;
  for (let i = 0; i <= P.length; i++) {
    const on = i < P.length && P[i] >= T0;
    if (on && start < 0) start = i;
    if (!on && start >= 0) {
      let peak = 0;
      for (let j = start; j < i; j++) peak = Math.max(peak, P[j]);
      for (let j = start; j < i; j++) s[j] = Math.min(1, P[j] / Math.pow(peak, rho));
      start = -1;
    }
  }
  return s;
}

function found(arr: number[], cut: number) {
  return LESIONS.map(([a, b]) => {
    const idx = X.map((_, i) => i).filter((i) => X[i] >= a && X[i] <= b);
    return idx.filter((i) => arr[i] >= cut).length >= TAU * idx.length;
  });
}

export function Mechanism() {
  const [rho, setRho] = useState(1);
  const [cut, setCut] = useState(0.5);
  const s = useMemo(() => equalized(rho), [rho]);
  const W = 640;
  const H = 260;
  const L = 34;
  const R = 10;
  const T = 14;
  const B = 74;
  const x = (v: number) => L + (W - L - R) * v;
  const y = (v: number) => T + (H - T - B) * (1 - v);
  const path = (a: number[]) => a.map((v, i) => `${i ? 'L' : 'M'}${x(X[i]).toFixed(1)},${y(v).toFixed(1)}`).join('');
  const strips = (arr: number[]) => {
    const out: Array<[number, number]> = [];
    let st = -1;
    arr.forEach((v, i) => {
      const on = v >= cut;
      if (on && st < 0) st = i;
      if ((!on || i === arr.length - 1) && st >= 0) {
        out.push([st, on ? i : i - 1]);
        st = -1;
      }
    });
    return out;
  };
  const ft = found(P, cut);
  const fe = found(s, cut);
  const nt = P.filter((v) => v >= cut).length;
  const need = useMemo(() => {
    const idx = X.map((_, i) => i).filter((i) => X[i] >= LESIONS[1][0] && X[i] <= LESIONS[1][1]);
    const v = idx.map((i) => P[i]).sort((a, b) => b - a);
    return v[Math.ceil(TAU * v.length) - 1];
  }, []);
  const ne = s.filter((v) => v >= cut).length;

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px]">
      <div className="min-w-0 overflow-x-auto rounded-md border border-border bg-background p-2">
        <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full min-w-[520px]" role="img" aria-label="Posterior profile under thresholding and lesion equalization">
          {LESIONS.map(([a, b, name]) => (
            <g key={name}>
              <rect x={x(a)} y={T} width={x(b) - x(a)} height={H - T - B} fill="var(--muted)" />
              <text x={x((a + b) / 2)} y={T + 12} textAnchor="middle" fontSize="11" fill="var(--muted-foreground)">{name}</text>
            </g>
          ))}
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--border)" />
              <text x={L - 6} y={y(v) + 4} textAnchor="end" fontSize="10" fill="var(--muted-foreground)">{v}</text>
            </g>
          ))}
          <text x={x(0.55)} y={y(0.19) - 8} textAnchor="middle" fontSize="10" fill="var(--muted-foreground)">noise</text>
          <path d={path(P)} fill="none" stroke="var(--muted-foreground)" strokeWidth={1.4} />
          <path d={path(s)} fill="none" stroke="var(--primary)" strokeWidth={2.2} />
          <line x1={L} x2={W - R} y1={y(cut)} y2={y(cut)} stroke="var(--foreground)" strokeDasharray="4 3" />
          <text x={L} y={H - B + 14} fontSize="11" fill="var(--muted-foreground)">thresholding set (posterior ≥ cutoff)</text>
          {strips(P).map(([a, b]) => (
            <rect key={`t${a}`} x={x(X[a])} y={H - B + 19} width={Math.max(1, x(X[b]) - x(X[a]))} height={10} rx={1} fill="var(--status-warn)" />
          ))}
          <text x={L} y={H - B + 46} fontSize="11" fill="var(--muted-foreground)">lesion-equalized set (score ≥ cutoff)</text>
          {strips(s).map(([a, b]) => (
            <rect key={`e${a}`} x={x(X[a])} y={H - B + 51} width={Math.max(1, x(X[b]) - x(X[a]))} height={10} rx={1} fill="var(--primary)" />
          ))}
        </svg>
      </div>
      <div className="grid content-start gap-4 text-sm">
        <label className="grid gap-1" htmlFor="mech-rho">
          <span className="flex justify-between font-medium">Equalization ρ <span className="font-mono text-primary">{rho.toFixed(2)}</span></span>
          <input id="mech-rho" type="range" min={0} max={1} step={0.05} value={rho} onChange={(e) => setRho(+e.target.value)} className="accent-[var(--primary)]" />
        </label>
        <label className="grid gap-1" htmlFor="mech-cut">
          <span className="flex justify-between font-medium">Cutoff <span className="font-mono text-primary">{cut.toFixed(2)}</span></span>
          <input id="mech-cut" type="range" min={0.05} max={0.95} step={0.01} value={cut} onChange={(e) => setCut(+e.target.value)} className="accent-[var(--primary)]" />
        </label>
        <div className="grid gap-2 rounded-md bg-muted p-3 text-xs">
          <p><span className="font-semibold">Thresholding:</span> finds {ft.filter(Boolean).length} of 2 lesions, {nt} samples in the set.</p>
          <p><span className="font-semibold">Equalized:</span> finds {fe.filter(Boolean).length} of 2 lesions, {ne} samples in the set.</p>
          <p className="text-muted-foreground">To find the faint lesion, thresholding needs a cutoff of {need.toFixed(2)} or lower. That widens the halo around the large lesion and admits the noise bump. Equalization rescales each candidate (posterior ≥ t₀ = 0.1) by its own peak, so a higher cutoff finds both lesions. The cost: the noise bump is a candidate too, so it is rescaled as well. On real data this trade-off is decided on a tuning split.</p>
        </div>
      </div>
    </div>
  );
}
