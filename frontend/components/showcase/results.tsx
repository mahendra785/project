'use client';

import { Fragment, useState } from 'react';

import type { BackboneManifest, Block, DiagRow, E8Row, ResultRow, SimRow } from './showcase-data';

export const DS_NAME: Record<string, string> = { fracatlas: 'FracAtlas', 'kvasir-seg': 'Kvasir-SEG' };

export const METHODS: Array<{ id: string; name: string; claims: Array<keyof ResultRow>; note: string }> = [
  { id: 'pixel_crc', name: 'Pixel CRC', claims: ['risk_pix'], note: 'Standard CRC, pixel FNR only (ICLR 2024)' },
  { id: 'union_threshold', name: 'Joint CRC · thresholding', claims: ['risk_pix', 'risk_les', 'risk_small'], note: 'Our formulation with the standard family; the reference' },
  { id: 'max_threshold', name: 'Joint max-form · thresholding', claims: ['risk_pix', 'risk_les', 'risk_small'], note: 'Stronger per-image-worst guarantee' },
  { id: 'ltt_threshold', name: 'Learn-Then-Test', claims: ['risk_pix', 'risk_les', 'risk_small'], note: 'High-probability multi-risk control, δ = 0.1' },
  { id: 'union_dilation', name: 'Joint · morphological sets', claims: ['risk_pix', 'risk_les', 'risk_small'], note: 'Dilation family (MICCAI 2025)' },
  { id: 'union_stratified', name: 'Joint · CCRA-S strata', claims: ['risk_pix', 'risk_les', 'risk_small'], note: 'Per-stratum thresholds (arXiv 2504.07611)' },
  { id: 'ours_union', name: 'Lesion-equalized · joint', claims: ['risk_pix', 'risk_les', 'risk_small'], note: 'This work' },
  { id: 'ours_max', name: 'Lesion-equalized · max-form', claims: ['risk_pix', 'risk_les', 'risk_small'], note: 'This work, stronger guarantee' },
];

const f = (v: number | null | undefined, d = 3) => (v === null || v === undefined || !Number.isFinite(v) ? '–' : v.toFixed(d));
const pct = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? '–' : `${v > 0 ? '+' : ''}${(100 * v).toFixed(1)}%`);

export function Verdict({ value }: { value: string }) {
  const tone =
    value === 'PROCEED' || value === 'SUPPORTED' || value === 'ok'
      ? 'bg-status-ok/15 text-status-ok'
      : value === 'FAILED' || value === 'KILL'
        ? 'bg-status-danger/15 text-status-danger'
        : 'bg-status-warn/15 text-status-warn';
  return <span className={`inline-block rounded px-1.5 py-0.5 font-mono text-[10.5px] font-semibold uppercase tracking-wide ${tone}`}>{value}</span>;
}

function Table({ children }: { children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full text-xs tabular-nums [&_td]:whitespace-nowrap [&_td]:border-b [&_td]:border-border/60 [&_td]:px-2 [&_td]:py-1.5 [&_th]:whitespace-nowrap [&_th]:border-b [&_th]:border-border [&_th]:bg-muted [&_th]:px-2 [&_th]:py-1.5 [&_th]:font-mono [&_th]:text-[10px] [&_th]:font-medium [&_th]:uppercase [&_th]:tracking-wide [&_th]:text-muted-foreground">
        {children}
      </table>
    </div>
  );
}

export function EveryCase({ blocks }: { blocks: Block[] }) {
  const [ds, setDs] = useState(blocks.find((b) => b.dataset === 'kvasir-seg')?.dataset ?? blocks[0].dataset);
  const b = blocks.find((x) => x.dataset === ds) ?? blocks[0];
  const alphas = [...new Set(b.rows.map((r) => r.alpha))].sort();
  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <div className="inline-flex overflow-hidden rounded-md border border-border" role="group" aria-label="Dataset">
          {blocks.map((x) => (
            <button key={x.dataset} type="button" aria-pressed={x.dataset === ds} onClick={() => setDs(x.dataset)} className={`px-3 py-1 text-xs ${x.dataset === ds ? 'bg-primary text-primary-foreground' : 'bg-card'}`}>{DS_NAME[x.dataset] ?? x.dataset}</button>
          ))}
        </div>
        <span className="text-xs text-muted-foreground">
          {b.meta[0].n_units} images · {b.meta[0].n_multi} multi-lesion · {b.meta[0].n_with_small} with a small lesion (&lt; {b.meta[0].small_cut_px.toFixed(0)} px) · backbone Dice {f(b.mean_dice)} · {b.meta.length} seeds × {b.meta[0].reps} splits
        </span>
      </div>
      <div className="flex flex-wrap gap-2 text-xs">
        <span>Gap (H1): <Verdict value={b.verdicts.H1} /></span>
        {Object.entries(b.verdicts.H2).map(([a, v]) => (
          <span key={a}>Equalized beats thresholding (H2), α={a}: <Verdict value={v} /></span>
        ))}
      </div>
      {alphas.map((a) => (
        <div key={a} className="grid gap-1.5">
          <p className="text-sm font-semibold">α = {a.toFixed(2)}</p>
          <Table>
            <thead>
              <tr><th className="text-left">Method</th><th>pixel FNR</th><th>lesion miss</th><th>small miss</th><th>miss | K≥2</th><th>set / truth</th><th>95% interval</th><th>Δ vs joint thr.</th><th>P(smaller)</th><th>validity</th></tr>
            </thead>
            <tbody>
              {METHODS.map((m) => {
                const r = b.rows.find((x) => x.alpha === a && x.method === m.id);
                if (!r) return null;
                const v = b.verdicts.validity[`${a}|${m.id}`] ?? '–';
                const cell = (k: keyof ResultRow) => {
                  const val = r[k] as number | null;
                  const over = m.claims.includes(k) && val !== null && val > a + 0.005;
                  return <td className={`text-right ${over ? 'font-semibold text-status-danger' : ''}`}>{f(val)}</td>;
                };
                return (
                  <tr key={m.id} className={m.id.startsWith('ours') ? 'bg-primary/10' : m.id === 'union_threshold' ? 'bg-status-warn/10' : ''}>
                    <td className="text-left" title={m.note}>{m.name}</td>
                    {cell('risk_pix')}{cell('risk_les')}{cell('risk_small')}
                    <td className={`text-right ${(r.risk_les_multi ?? 0) > a ? 'text-status-danger' : ''}`}>{f(r.risk_les_multi)}</td>
                    <td className="text-right">{f(r.set_to_truth, 2)}</td>
                    <td className="text-right text-muted-foreground">{f(r.set_to_truth_lo, 2)}–{f(r.set_to_truth_hi, 2)}</td>
                    <td className="text-right">{m.id === 'union_threshold' ? 'reference' : pct(r.rel)}</td>
                    <td className="text-right">{m.id === 'union_threshold' ? '–' : f(r.p_smaller, 2)}</td>
                    <td className="text-right"><Verdict value={v === 'ok' ? 'ok' : 'violated'} /></td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        </div>
      ))}
      <p className="text-[11px] text-muted-foreground">Red risk: a risk the method claims to control is above α + 0.005. Red K≥2: lesion miss on multi-lesion images above α; no pre-registered method controls that. Hover a method name for its source.</p>
    </div>
  );
}

export function GapTable({ blocks }: { blocks: Block[] }) {
  return (
    <Table>
      <thead><tr><th className="text-left">Dataset</th><th>α</th><th>G all images</th><th>G | K≥2</th><th>95% interval</th><th>G small</th><th>95% interval</th><th>n multi / cal split</th><th>verdict</th></tr></thead>
      <tbody>
        {blocks.flatMap((b) =>
          b.gap.map((g) => (
            <tr key={`${b.dataset}${g.alpha}`}>
              <td className="text-left">{DS_NAME[b.dataset]}</td>
              <td className="text-right">{g.alpha.toFixed(2)}</td>
              <td className="text-right">{f(g.G, 2)}</td>
              <td className={`text-right ${g.G_multi > 1 ? 'font-semibold text-status-danger' : ''}`}>{f(g.G_multi, 2)}</td>
              <td className="text-right text-muted-foreground">{f(g.G_multi_lo, 2)}–{f(g.G_multi_hi, 2)}</td>
              <td className="text-right">{f(g.G_small, 2)}</td>
              <td className="text-right text-muted-foreground">{f(g.G_small_lo, 2)}–{f(g.G_small_hi, 2)}</td>
              <td className="text-right">{f(g.n_cal_multi, 0)}</td>
              <td className="text-right"><Verdict value={b.verdicts.H1} /></td>
            </tr>
          )),
        )}
      </tbody>
    </Table>
  );
}

export function Backbones({ items }: { items: BackboneManifest[] }) {
  if (!items.length) return <p className="text-sm text-muted-foreground">No backbone manifests found.</p>;
  const sorted = [...items].sort((a, b) => a.dataset.localeCompare(b.dataset) || a.seed - b.seed);
  return (
    <Table>
      <thead><tr><th className="text-left">Dataset</th><th>Backbone</th><th>Seed</th><th>Units</th><th>Res.</th><th>Dice per fold</th><th>Mean Dice</th><th>Train time</th><th className="text-left">sha256</th></tr></thead>
      <tbody>
        {sorted.map((m) => (
          <tr key={`${m.dataset}${m.seed}`}>
            <td className="text-left">{DS_NAME[m.dataset] ?? m.dataset}</td>
            <td className="text-right">{m.backbone}</td>
            <td className="text-right">{m.seed}</td>
            <td className="text-right">{m.n_units}</td>
            <td className="text-right">{m.size}²</td>
            <td className="text-right">{m.fold_dice.map((d) => d.toFixed(3)).join(' · ')}</td>
            <td className="text-right font-medium">{m.mean_dice.toFixed(3)}</td>
            <td className="text-right">{m.seconds}s</td>
            <td className="text-left font-mono text-[10px] text-muted-foreground">{m.sha256.slice(0, 12)}…</td>
          </tr>
        ))}
      </tbody>
    </Table>
  );
}

export function SimGrid({ rows, diag, verdict }: { rows: SimRow[]; diag: DiagRow[] | null; verdict: string }) {
  const gammas = [...new Set(rows.map((r) => r.gamma))].sort((a, b) => a - b);
  const bgs = [...new Set(rows.map((r) => r.background))].sort((a, b) => a - b);
  const alphas = [...new Set(rows.map((r) => r.alpha))].sort((a, b) => a - b);
  const shade = (v: number) => {
    const t = Math.min(1, Math.abs(v) / 0.5);
    return v < 0 ? `color-mix(in oklch, var(--status-ok) ${Math.round(t * 55)}%, transparent)` : `color-mix(in oklch, var(--status-danger) ${Math.round(Math.min(1, v / 0.1) * 35)}%, transparent)`;
  };
  return (
    <div className="grid gap-2">
      <Table>
        <thead><tr><th className="text-left">Background · α</th>{gammas.map((g) => <th key={g}>γ = {g}</th>)}</tr></thead>
        <tbody>
          {bgs.map((b) => (
            <Fragment key={b}>
              {alphas.map((a) => (
                <tr key={`${b}${a}`}>
                  <td className="text-left">b = {b} · α = {a.toFixed(2)}</td>
                  {gammas.map((g) => {
                    const r = rows.find((x) => x.gamma === g && x.background === b && x.alpha === a);
                    if (!r) return <td key={g}>–</td>;
                    return (
                      <td key={g} className={`text-right ${r.p_smaller >= 0.95 ? 'font-semibold' : ''}`} style={{ background: shade(r.rel) }} title={`95% split interval ${pct(r.lo)} to ${pct(r.hi)} · P(smaller) ${r.p_smaller.toFixed(2)}`}>
                        {pct(r.rel)}
                      </td>
                    );
                  })}
                </tr>
              ))}
              {diag && (
                <tr key={`d${b}`} className="text-muted-foreground">
                  <td className="text-left">b = {b} · small lesions below background</td>
                  {gammas.map((g) => {
                    const d = diag.find((x) => x.gamma === g && x.background === b);
                    return <td key={g} className="text-right">{d ? `${(100 * d.frac_small_lesions_not_above_background_p99).toFixed(0)}%` : '–'}</td>;
                  })}
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </Table>
      <p className="text-[11px] text-muted-foreground">Change in set size, lesion-equalized vs joint thresholding (green = smaller). Bold: smaller in at least 95% of 500 splits. Hover a cell for its interval. Pre-registered rule verdict: <Verdict value={verdict} />. The grey rows are a post-hoc diagnostic.</p>
    </div>
  );
}

export function PosthocTable({ e8 }: { e8: Record<string, E8Row[]> }) {
  return (
    <Table>
      <thead><tr><th className="text-left">Dataset</th><th>α</th><th>miss | K≥2, joint</th><th>miss | K≥2, + conditional risk</th><th>set / truth, joint</th><th>set / truth, + conditional</th><th>price</th></tr></thead>
      <tbody>
        {Object.entries(e8).flatMap(([ds, rs]) =>
          rs.map((r) => (
            <tr key={`${ds}${r.alpha}`}>
              <td className="text-left">{DS_NAME[ds]}</td>
              <td className="text-right">{r.alpha.toFixed(2)}</td>
              <td className={`text-right ${r.thr_joint3_multi.mean > r.alpha ? 'text-status-danger' : ''}`}>{f(r.thr_joint3_multi.mean)}</td>
              <td className={`text-right ${r.thr_joint4_multi.mean > r.alpha ? 'text-status-danger' : 'text-status-ok'}`}>{f(r.thr_joint4_multi.mean)}</td>
              <td className="text-right">{f(r.thr_joint3_str.mean, 2)}</td>
              <td className="text-right">{f(r.thr_joint4_str.mean, 2)}</td>
              <td className="text-right">{pct(r.price_of_multi.mean)}</td>
            </tr>
          )),
        )}
      </tbody>
    </Table>
  );
}
