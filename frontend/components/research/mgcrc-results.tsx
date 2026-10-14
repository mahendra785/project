'use client';

/**
 * Lesion-aware joint CRC results.
 *
 * Renders docs/research/results/mgcrc_summary.json exactly as served by
 * GET /api/research/mgcrc. Verdict strings are computed by aggregate.py from
 * the pre-registered rules; this component only displays them.
 */

import { useMemo, useState } from 'react';
import { Bar, BarChart, Cell, ErrorBar, ReferenceLine } from 'recharts';

import { ChartFrame, ChartGrid, ChartTooltip, XAxis, YAxis, CHART_COLOURS, axisProps } from '@/components/charts/chart-primitives';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useMgcrc } from '@/hooks/use-pums';
import type { MgcrcBlock, MgcrcRow } from '@/types/api';

const METHOD_LABEL: Record<string, string> = {
  pixel_crc: 'Pixel CRC',
  union_threshold: 'Joint · threshold',
  max_threshold: 'Joint max · threshold',
  ltt_threshold: 'LTT · threshold',
  union_dilation: 'Joint · morphological',
  union_stratified: 'Joint · CCRA-S',
  ours_union: 'Ours · joint',
  ours_max: 'Ours · max',
};
const ORDER = Object.keys(METHOD_LABEL);

function fmt(v: number | null | undefined, digits = 3) {
  return v === null || v === undefined || !Number.isFinite(v) ? '—' : v.toFixed(digits);
}

function verdictTone(v: string) {
  if (v === 'SUPPORTED' || v === 'PROCEED') return 'bg-status-ok/15 text-status-ok';
  if (v === 'FAILED' || v === 'KILL') return 'bg-status-danger/15 text-status-danger';
  return 'bg-status-warn/15 text-status-warn';
}

function riskCell(value: number | null, alpha: number, claimed: boolean) {
  const over = claimed && value !== null && value > alpha + 0.005;
  return <td className={`px-2 py-1 text-right tabular-nums ${over ? 'text-status-danger font-medium' : ''}`}>{fmt(value)}</td>;
}

function BlockView({ block }: { block: MgcrcBlock }) {
  const alphas = useMemo(() => [...new Set(block.rows.map((r) => r.alpha))].sort(), [block]);
  const [alpha, setAlpha] = useState(alphas.includes(0.1) ? 0.1 : alphas[0]);
  const rows = ORDER.map((m) => block.rows.find((r) => r.alpha === alpha && r.method === m)).filter(
    (r): r is MgcrcRow => Boolean(r),
  );
  const chart = rows.map((r) => ({
    name: METHOD_LABEL[r.method],
    method: r.method,
    value: r.set_to_truth ?? 0,
    err: [Math.max((r.set_to_truth ?? 0) - (r.set_to_truth_lo ?? 0), 0), Math.max((r.set_to_truth_hi ?? 0) - (r.set_to_truth ?? 0), 0)],
  }));
  const ref = rows.find((r) => r.method === 'union_threshold')?.set_to_truth ?? undefined;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-muted-foreground">
          {block.n_units} units with lesions · {block.n_multi} multi-lesion · {block.n_with_small} with a small lesion ·
          backbone <code>{block.backbone}</code>
          {block.mean_dice !== null ? ` (out-of-fold Dice ${block.mean_dice.toFixed(3)})` : ''} · {block.n_seeds} seed(s) ×{' '}
          {block.reps_per_seed} splits · τ = {block.tau}
        </span>
        <span className={`rounded px-1.5 py-0.5 font-medium ${verdictTone(block.verdicts.H1)}`}>H1 gap: {block.verdicts.H1}</span>
        {Object.entries(block.verdicts.H2).map(([a, v]) => (
          <span key={a} className={`rounded px-1.5 py-0.5 font-medium ${verdictTone(v)}`}>
            H2 α={a}: {v}
          </span>
        ))}
        <Badge variant="outline">computed</Badge>
      </div>

      <div className="flex items-center gap-1 text-xs">
        <span className="text-muted-foreground mr-1">α</span>
        {alphas.map((a) => (
          <button
            key={a}
            onClick={() => setAlpha(a)}
            className={`rounded border px-2 py-0.5 ${a === alpha ? 'border-primary bg-primary/10' : 'border-border'}`}
          >
            {a.toFixed(2)}
          </button>
        ))}
      </div>

      <div>
        <p className="mb-1 text-xs text-muted-foreground">
          Set-to-truth ratio Σ|S| / Σ|Y| on test units (lower is better; bars = 95% split interval). Dashed line:
          joint CRC with thresholding, the reference every guarantee-matched method is compared to.
        </p>
        <ChartFrame height={240}>
          <BarChart data={chart} margin={{ top: 8, right: 8, bottom: 40, left: 0 }}>
            <ChartGrid />
            <XAxis dataKey="name" {...axisProps} angle={-25} textAnchor="end" interval={0} height={50} />
            <YAxis {...axisProps} />
            <ChartTooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))} />
            {ref !== undefined && <ReferenceLine y={ref} stroke={CHART_COLOURS.reference} strokeDasharray="4 3" />}
            <Bar dataKey="value" isAnimationActive={false}>
              {chart.map((d) => (
                <Cell key={d.method} fill={d.method.startsWith('ours') ? CHART_COLOURS.primary : CHART_COLOURS.muted} />
              ))}
              <ErrorBar dataKey="err" width={4} stroke="var(--foreground)" />
            </Bar>
          </BarChart>
        </ChartFrame>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-muted-foreground">
            <tr className="border-b border-border">
              <th className="px-2 py-1 text-left">Method</th>
              <th className="px-2 py-1 text-right">pixel FNR</th>
              <th className="px-2 py-1 text-right">lesion miss</th>
              <th className="px-2 py-1 text-right">small miss</th>
              <th className="px-2 py-1 text-right">miss | K≥2</th>
              <th className="px-2 py-1 text-right">set / truth</th>
              <th className="px-2 py-1 text-right">Δ vs joint-threshold</th>
              <th className="px-2 py-1 text-right">P(smaller)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const joint = r.method !== 'pixel_crc';
              return (
                <tr key={r.method} className={`border-b border-border/50 ${r.method.startsWith('ours') ? 'bg-primary/5' : ''}`}>
                  <td className="px-2 py-1">{METHOD_LABEL[r.method]}</td>
                  {riskCell(r.risk_pix, alpha, true)}
                  {riskCell(r.risk_les, alpha, joint)}
                  {riskCell(r.risk_small, alpha, joint)}
                  {riskCell(r.risk_les_multi, alpha, false)}
                  <td className="px-2 py-1 text-right tabular-nums">{fmt(r.set_to_truth, 2)}</td>
                  <td className="px-2 py-1 text-right tabular-nums">
                    {r.method === 'union_threshold' || r.rel === null ? 'ref.' : `${(100 * r.rel).toFixed(1)}%`}
                  </td>
                  <td className="px-2 py-1 text-right tabular-nums">
                    {r.method === 'union_threshold' ? '—' : fmt(r.p_smaller, 2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-1 text-[11px] text-muted-foreground">
          Red: a risk the method claims to control exceeds α (+0.005 Monte-Carlo slack). Pixel CRC claims only pixel
          FNR; its lesion columns show the gap the joint methods close.
        </p>
      </div>

      <div className="overflow-x-auto">
        <p className="mb-1 text-xs font-medium">Gap at the pixel-CRC threshold — G = lesion-miss / α</p>
        <table className="w-full text-xs">
          <thead className="text-muted-foreground">
            <tr className="border-b border-border">
              <th className="px-2 py-1 text-left">α</th>
              <th className="px-2 py-1 text-right">G all</th>
              <th className="px-2 py-1 text-right">G | K≥2 [95%]</th>
              <th className="px-2 py-1 text-right">G small [95%]</th>
              <th className="px-2 py-1 text-right">n_cal multi</th>
            </tr>
          </thead>
          <tbody>
            {block.gap.map((g) => (
              <tr key={g.alpha} className="border-b border-border/50">
                <td className="px-2 py-1">{g.alpha.toFixed(2)}</td>
                <td className="px-2 py-1 text-right tabular-nums">{fmt(g.G, 2)}</td>
                <td className="px-2 py-1 text-right tabular-nums">
                  {fmt(g.G_multi, 2)} [{fmt(g.G_multi_lo, 2)}, {fmt(g.G_multi_hi, 2)}]
                </td>
                <td className="px-2 py-1 text-right tabular-nums">
                  {fmt(g.G_small, 2)} [{fmt(g.G_small_lo, 2)}, {fmt(g.G_small_hi, 2)}]
                </td>
                <td className="px-2 py-1 text-right tabular-nums">{fmt(g.n_cal_multi, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function MgcrcResults() {
  const query = useMgcrc();
  const [index, setIndex] = useState(0);
  if (query.isLoading) return <p className="text-sm text-muted-foreground">Loading results…</p>;
  if (query.isError) return <p className="text-sm text-status-danger">Could not load /api/research/mgcrc.</p>;
  const data = query.data;
  if (!data?.available || data.blocks.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No results yet. Run <code>scripts/reproduce</code> to generate {data?.source ?? 'mgcrc_summary.json'}.
      </p>
    );
  }
  const block = data.blocks[Math.min(index, data.blocks.length - 1)];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Lesion-aware joint conformal risk control</CardTitle>
        <div className="flex flex-wrap gap-1 pt-2 text-xs">
          {data.blocks.map((b, i) => (
            <button
              key={`${b.dataset}-${b.backbone}-${b.tau}`}
              onClick={() => setIndex(i)}
              className={`rounded border px-2 py-0.5 ${i === index ? 'border-primary bg-primary/10' : 'border-border'}`}
            >
              {b.dataset} · {b.backbone} · τ={b.tau}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent>
        <BlockView key={index} block={block} />
      </CardContent>
    </Card>
  );
}
