'use client';

import { Bar, BarChart, Cell, LabelList, Line, LineChart, ReferenceLine } from 'recharts';

import { axisProps, CHART_COLOURS, ChartFrame, ChartGrid, ChartTooltip, XAxis, YAxis } from '@/components/charts/chart-primitives';
import type { DemoRun, DemoValidity, RiskKey } from '@/lib/api/demo';
import { cutoffOfIndex, logSampleIndices, pct, SCRUB_MIN_CUTOFF, certifies, shortLabel } from '@/lib/live-demo';
import { cn } from '@/lib/utils';

export const RISKS: Array<{ key: RiskKey; label: string; short: string; colour: string; joint: boolean }> = [
  { key: 'pix', label: 'Pixel miss rate (FNR)', short: 'pixel', colour: 'var(--viz-1)', joint: true },
  { key: 'les', label: 'Polyp miss rate', short: 'polyp', colour: 'var(--viz-2)', joint: true },
  { key: 'small', label: 'Small-polyp miss rate', short: 'small polyp', colour: 'var(--viz-3)', joint: true },
  { key: 'multi', label: 'Miss rate on multi-polyp images', short: 'multi-polyp', colour: 'var(--viz-4)', joint: false },
];

function barFill(value: number | null, alpha: number, ours: boolean) {
  if (value !== null && value > alpha) return CHART_COLOURS.danger;
  return ours ? CHART_COLOURS.primary : CHART_COLOURS.muted;
}

/** One small bar chart per risk: every method's held-out risk on this split against alpha. */
export function GuaranteeBars({ run }: { run: DemoRun }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {RISKS.map((risk) => {
        const data = run.methods.map((m) => ({
          name: shortLabel(m.id, m.label),
          value: m.test[risk.key],
          ours: m.ours,
          guaranteed: m.guaranteed.includes(risk.key),
        }));
        const max = Math.max(run.alpha * 1.6, ...data.map((d) => d.value ?? 0));
        return (
          <figure key={risk.key} className="space-y-1">
            <figcaption className="text-xs font-medium">
              {risk.label}
              {!risk.joint && <span className="ml-1 font-normal text-muted-foreground">(not part of the joint rule)</span>}
            </figcaption>
            <ChartFrame height={232}>
              <BarChart data={data} layout="vertical" margin={{ top: 4, right: 64, bottom: 4, left: 8 }} barCategoryGap={5}>
                <ChartGrid />
                <XAxis type="number" domain={[0, max]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} {...axisProps} />
                <YAxis type="category" dataKey="name" width={136} {...axisProps} />
                <ReferenceLine x={run.alpha} stroke="var(--foreground)" strokeDasharray="4 3" label={{ value: `α = ${run.alpha}`, position: 'top', fontSize: 10, fill: 'var(--muted-foreground)' }} />
                <ChartTooltip
                  cursor={{ fill: 'var(--muted)', opacity: 0.4 }}
                  formatter={(v, _n, item) => [
                    `${pct(Number(v))}${item.payload.guaranteed ? ' · certified ≤ α' : ' · no certificate'}`,
                    risk.label,
                  ]}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                  {data.map((d) => (
                    <Cell key={d.name} fill={barFill(d.value, run.alpha, d.ours)} />
                  ))}
                  <LabelList
                    dataKey="value"
                    position="right"
                    fontSize={10}
                    fill="var(--foreground)"
                    formatter={(v: number | null) => (v === null ? '—' : `${pct(v)}${v > run.alpha ? ' > α' : ''}`)}
                  />
                </Bar>
              </BarChart>
            </ChartFrame>
          </figure>
        );
      })}
    </div>
  );
}

/**
 * Corrected calibration risk (n·R̂ + 1)/(n + 1) as the cutoff falls. Each risk's λ̂ is the
 * first point where its curve drops below alpha; the joint rule takes the lowest cutoff of the three.
 */
export function LambdaCurves({ run }: { run: DemoRun }) {
  const idx = logSampleIndices(run.grid_size, 180);
  const data = idx.map((i) => {
    const row: Record<string, number> = { cutoff: Math.max(SCRUB_MIN_CUTOFF, cutoffOfIndex(i, run.grid_size)) };
    for (const r of RISKS) row[r.key] = run.curves[r.key]?.corrected[i] ?? NaN;
    return row;
  });
  const cut = (r: RiskKey) => Math.max(SCRUB_MIN_CUTOFF, 1 - (run.curves[r]?.lam_hat ?? 1));
  const jointCut = Math.min(cut('pix'), cut('les'), cut('small'));
  const yMax = Math.min(1, Math.max(run.alpha * 3, 0.3));
  return (
    <div className="space-y-3">
      <ChartFrame height={320}>
        <LineChart data={data} margin={{ top: 18, right: 16, bottom: 18, left: 0 }}>
          <ChartGrid />
          <XAxis
            dataKey="cutoff"
            type="number"
            scale="log"
            domain={[SCRUB_MIN_CUTOFF, 1]}
            reversed
            ticks={[1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]}
            tickFormatter={(v: number) => String(v)}
            label={{ value: 'posterior cutoff 1 − λ  (set grows →)', position: 'insideBottom', offset: -8, fontSize: 10, fill: 'var(--muted-foreground)' }}
            {...axisProps}
          />
          <YAxis domain={[0, yMax]} allowDataOverflow tickFormatter={(v: number) => `${Math.round(v * 100)}%`} {...axisProps} />
          <ChartTooltip
            labelFormatter={(v) => `cutoff ${Number(v).toFixed(3)}`}
            formatter={(v, name) => [pct(Number(v)), RISKS.find((r) => r.key === name)?.label ?? String(name)]}
          />
          <ReferenceLine y={run.alpha} stroke="var(--foreground)" strokeDasharray="4 3" label={{ value: `α = ${run.alpha}`, position: 'insideTopLeft', fontSize: 10, fill: 'var(--muted-foreground)' }} />
          {RISKS.filter((r) => r.joint).map((r) => (
            <ReferenceLine key={r.key} x={cut(r.key)} stroke={r.colour} strokeDasharray="2 3" />
          ))}
          <ReferenceLine x={jointCut} stroke="var(--primary)" strokeWidth={2} label={{ value: 'joint λ̂ = max', position: 'insideTopRight', fontSize: 10, fill: 'var(--primary)' }} />
          {RISKS.map((r) => (
            <Line
              key={r.key}
              dataKey={r.key}
              stroke={r.colour}
              strokeWidth={2}
              strokeDasharray={r.joint ? undefined : '5 4'}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ChartFrame>
      <div className="overflow-x-auto">
        <table className="tabular w-full text-xs">
          <thead className="text-[11px] text-muted-foreground">
            <tr className="border-b border-border text-left">
              <th className="py-1 pr-3 font-medium">Risk</th>
              <th className="py-1 pr-3 font-medium">Calibration images used</th>
              <th className="py-1 pr-3 font-medium">Its own cutoff</th>
              <th className="py-1 font-medium">Role</th>
            </tr>
          </thead>
          <tbody>
            {RISKS.map((r) => (
              <tr key={r.key} className="border-b border-border/60">
                <td className="py-1.5 pr-3">
                  <span className="mr-2 inline-block h-0.5 w-4 align-middle" style={{ background: r.colour, borderTop: r.joint ? undefined : `2px dashed ${r.colour}` }} />
                  {r.label}
                </td>
                <td className="py-1.5 pr-3">{run.curves[r.key]?.n ?? 0}</td>
                <td className="py-1.5 pr-3 font-mono">{(1 - (run.curves[r.key]?.lam_hat ?? 1)).toFixed(3)}</td>
                <td className={cn('py-1.5', r.joint && cut(r.key) === jointCut && 'font-medium text-primary')}>
                  {r.key === 'pix'
                    ? 'pixel CRC uses only this one'
                    : r.joint
                      ? cut(r.key) === jointCut
                        ? 'binding: sets the joint threshold'
                        : 'satisfied by the joint threshold'
                      : 'joint + multi adds this one (few eligible images, so it needs a near-zero cutoff)'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const VALIDITY_COLS: Array<{ key: RiskKey; label: string }> = [
  { key: 'pix', label: 'Pixel miss' },
  { key: 'les', label: 'Polyp miss' },
  { key: 'small', label: 'Small-polyp miss' },
  { key: 'multi', label: 'Multi-polyp miss' },
];

/** Mean and 5-95% range of each held-out risk over many random calibration/test splits. */
export function ValidityStrips({ validity }: { validity: DemoValidity }) {
  const { alpha } = validity;
  const domain = Object.fromEntries(
    VALIDITY_COLS.map((c) => [c.key, Math.max(alpha * 2, ...validity.methods.map((m) => m.stats[c.key].p95 ?? 0)) * 1.05]),
  ) as Record<RiskKey, number>;
  return (
    <div className="overflow-x-auto">
      <table className="tabular w-full min-w-[860px] text-xs">
        <thead className="text-[11px] text-muted-foreground">
          <tr className="border-b border-border text-left">
            <th className="py-1.5 pr-3 font-medium">Method</th>
            {VALIDITY_COLS.map((c) => (
              <th key={c.key} className="py-1.5 pr-3 font-medium">{c.label}</th>
            ))}
            <th className="py-1.5 font-medium">Set size / truth</th>
          </tr>
        </thead>
        <tbody>
          {validity.methods.map((m) => (
            <tr key={m.id} className={cn('border-b border-border/60', m.ours && 'bg-primary/5')}>
              <td className="py-2 pr-3 align-top">
                <div className={cn('font-medium', m.ours && 'text-primary')}>{m.label}</div>
                <div className="text-[10px] text-muted-foreground">{certifies(m.guaranteed, m.guarantee)}</div>
              </td>
              {VALIDITY_COLS.map((c) => {
                const s = m.stats[c.key];
                const d = domain[c.key];
                const above = s.mean !== null && s.mean > alpha;
                const certified = m.guaranteed.includes(c.key);
                return (
                  <td key={c.key} className="py-2 pr-3 align-top">
                    <div className="relative h-4 w-36 rounded bg-muted" title={`5-95%: ${pct(s.p05)} to ${pct(s.p95)}`}>
                      <div
                        className={cn('absolute inset-y-1 rounded-sm', above ? 'bg-status-danger/35' : 'bg-foreground/20')}
                        style={{ left: `${(100 * (s.p05 ?? 0)) / d}%`, width: `${(100 * ((s.p95 ?? 0) - (s.p05 ?? 0))) / d}%` }}
                      />
                      <div
                        className={cn('absolute inset-y-0 w-1 -translate-x-1/2 rounded-sm', above ? 'bg-status-danger' : 'bg-foreground')}
                        style={{ left: `${(100 * (s.mean ?? 0)) / d}%` }}
                      />
                      <div className="absolute inset-y-[-2px] w-px border-l border-dashed border-foreground/70" style={{ left: `${(100 * alpha) / d}%` }} />
                    </div>
                    <div className={cn('mt-0.5 text-[11px]', above && 'font-medium text-status-danger')}>
                      mean {pct(s.mean)} {above && '> α'}
                      {certified && <span className="ml-1 text-status-ok" title="This method certifies this risk ≤ α">✓ cert.</span>}
                    </div>
                    <div className="text-[10px] text-muted-foreground">above α on {pct(s.frac_above, 0)} of splits</div>
                  </td>
                );
              })}
              <td className="py-2 align-top">
                <div className="font-medium">{(m.stats.set_to_truth.mean ?? 0).toFixed(2)}×</div>
                <div className="text-[10px] text-muted-foreground">
                  {m.id === 'joint' ? 'reference' : `${(m.stats.rel_vs_joint.mean ?? 0) >= 0 ? '+' : ''}${pct(m.stats.rel_vs_joint.mean, 0)} vs joint (paired)`}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-muted-foreground">
        Bar = 5–95% range across splits, tick = mean, dashed line = α. CRC certifies the <i>expected</i> risk, so a
        certified method can exceed α on individual splits; its mean must not. Means above α are flagged in red.
      </p>
    </div>
  );
}
