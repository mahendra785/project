'use client';

import { useEffect } from 'react';
import { AlertTriangle, Radar, ScanSearch } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend } from 'recharts';

import {
  ChartFrame,
  ChartTooltip,
  XAxis,
  YAxis,
} from '@/components/charts/chart-primitives';
import { PageContainer, PageHeader } from '@/components/layout/page-container';
import { EmptyState, ErrorState, LoadingState } from '@/components/layout/states';
import { MetricCard } from '@/components/research/metric-card';
import {
  ProvenanceBadge,
  ShiftBadge,
  StatusBadge,
} from '@/components/research/status-badges';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/primitives';
import { AnalysisPicker } from '@/features/results/analysis-picker';
import { useAnalyses, useAnalysis, useShiftConfig } from '@/hooks/use-pums';
import { num } from '@/lib/utils';
import { useAnalysisStore } from '@/stores/analysis-store';
import type { DistributionShiftResult } from '@/types/api';

function HistogramComparison({ shift }: { shift: DistributionShiftResult }) {
  const data = shift.current_distribution.histogram.map((value, index) => ({
    intensity: (index / shift.current_distribution.histogram.length).toFixed(2),
    current: value,
    reference: shift.reference_distribution.histogram[index] ?? 0,
  }));

  return (
    <ChartFrame height={240}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -14 }}>
        <CartesianGrid strokeDasharray="2 3" stroke="var(--border)" vertical={false} />
        <XAxis
          dataKey="intensity"
          stroke="var(--muted-foreground)"
          fontSize={9}
          tickLine={false}
          interval={3}
        />
        <YAxis stroke="var(--muted-foreground)" fontSize={9} tickLine={false} />
        <ChartTooltip
          formatter={(value, name) => [
            typeof value === 'number' ? value.toFixed(4) : value,
            name === 'current' ? 'Uploaded image' : 'Reference domain',
          ]}
          labelFormatter={(label) => `Intensity ≈ ${label}`}
        />
        <Legend
          wrapperStyle={{ fontSize: 10 }}
          formatter={(value) => (value === 'current' ? 'Uploaded image' : 'Reference domain')}
        />
        <Bar dataKey="reference" fill="var(--muted-foreground)" fillOpacity={0.4} radius={[2, 2, 0, 0]} />
        <Bar dataKey="current" fill="var(--status-info)" fillOpacity={0.85} radius={[2, 2, 0, 0]} />
      </BarChart>
    </ChartFrame>
  );
}

export default function DistributionShiftPage() {
  const list = useAnalyses(50);
  const config = useShiftConfig();
  const selectedId = useAnalysisStore((s) => s.selectedId);
  const setSelectedId = useAnalysisStore((s) => s.setSelectedId);

  const effectiveId =
    selectedId && list.data?.items.some((i) => i.analysis_id === selectedId)
      ? selectedId
      : (list.data?.items[0]?.analysis_id ?? null);

  const analysis = useAnalysis(effectiveId);
  const shift = analysis.data?.distribution_shift ?? null;

  useEffect(() => {
    if (!selectedId && effectiveId) setSelectedId(effectiveId);
  }, [effectiveId, selectedId, setSelectedId]);

  return (
    <PageContainer>
      <PageHeader
        title="Distribution shift"
        description="A composite input-space detector comparing each image against a reference domain. It catches modality, exposure and scanner-protocol changes; it does not catch semantic shift that leaves intensity statistics intact."
        badges={config.data && <StatusBadge status={config.data.status} />}
      />

      {config.isError && <ErrorState error={config.error} onRetry={() => config.refetch()} />}

      {/* ------------------------------------------------------ detector config */}
      {config.data && (
        <Card>
          <CardHeader className="flex-row items-start justify-between">
            <div>
              <CardTitle className="flex items-center gap-1.5">
                <Radar className="size-4" />
                {config.data.detector_name}
              </CardTitle>
              <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                {config.data.detector}
              </p>
            </div>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Band thresholds (configurable)
              </p>
              <div className="space-y-1">
                {[
                  ['ID', `< ${num(config.data.thresholds.mild, 2)}`],
                  ['Mild shift', `${num(config.data.thresholds.mild, 2)} – ${num(config.data.thresholds.moderate, 2)}`],
                  ['Moderate shift', `${num(config.data.thresholds.moderate, 2)} – ${num(config.data.thresholds.severe, 2)}`],
                  ['Severe shift', `≥ ${num(config.data.thresholds.severe, 2)}`],
                ].map(([label, range]) => (
                  <div key={label} className="flex justify-between text-[11px]">
                    <span className="text-muted-foreground">{label}</span>
                    <span className="tabular font-medium">{range}</span>
                  </div>
                ))}
              </div>
              <p className="text-[10px] leading-relaxed text-muted-foreground">
                Derived by <code>scripts/calibrate_shift_thresholds.py</code> against
                the reference domain — operating points, not experimental findings.
              </p>
            </div>

            <div className="space-y-2">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Indicator weights
              </p>
              {config.data.indicators.map((indicator) => (
                <div key={indicator.id} className="space-y-0.5" title={indicator.description}>
                  <div className="flex justify-between text-[11px]">
                    <span className="truncate text-muted-foreground">{indicator.name}</span>
                    <span className="tabular shrink-0 font-medium">
                      {num(indicator.weight, 2)}
                    </span>
                  </div>
                  <div className="h-1 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full bg-primary/70"
                      style={{ width: `${indicator.weight * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="space-y-2">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Known limitations
              </p>
              <ul className="space-y-1">
                {config.data.limitations.map((limitation) => (
                  <li
                    key={limitation}
                    className="flex gap-1.5 text-[10px] leading-relaxed text-muted-foreground"
                  >
                    <AlertTriangle className="mt-0.5 size-3 shrink-0 text-status-warn" />
                    {limitation}
                  </li>
                ))}
              </ul>
            </div>
          </CardContent>
        </Card>
      )}

      {/* --------------------------------------------------------- per-analysis */}
      {list.isLoading && <LoadingState label="Loading analyses…" />}

      {!list.isLoading && (list.data?.items.length ?? 0) === 0 && (
        <EmptyState
          title="No analyses to evaluate"
          description="Shift detection runs as part of an analysis. Upload an image — the sample set in data/sample/ includes deliberately shifted phantoms (gamma and inverted) to exercise the bands."
          actionLabel="Go to Analyze"
          actionHref="/analyze"
          icon={<ScanSearch className="size-5" />}
        />
      )}

      {(list.data?.items.length ?? 0) > 0 && (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <AnalysisPicker
              items={list.data?.items ?? []}
              value={effectiveId}
              onChange={setSelectedId}
            />
            {shift && (
              <>
                <ShiftBadge status={shift.status} />
                <ProvenanceBadge provenance={shift.provenance} />
              </>
            )}
          </div>

          {analysis.isLoading && <LoadingState />}
          {analysis.isError && (
            <ErrorState error={analysis.error} onRetry={() => analysis.refetch()} />
          )}

          {analysis.data && !shift && (
            <Card>
              <CardContent className="px-6 py-10 text-center">
                <p className="text-sm font-medium">Shift detection was disabled</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  This analysis was run with <code>detect_shift = false</code>.
                </p>
              </CardContent>
            </Card>
          )}

          {shift && (
            <>
              <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="Composite shift score"
                  value={num(shift.shift_score, 4)}
                  icon={Radar}
                  provenance={shift.provenance}
                  tone={
                    shift.status === 'ID'
                      ? 'ok'
                      : shift.status === 'SEVERE_SHIFT'
                        ? 'danger'
                        : 'warn'
                  }
                  footer={shift.status.replace('_', ' ').toLowerCase()}
                />
                <MetricCard
                  label="Predicted confidence drop"
                  value={num(shift.confidence_degradation, 3)}
                  provenance="computed"
                  hint="A heuristic coupling of the shift score, NOT a measured performance drop. Measuring real degradation needs paired ID/OOD labelled data."
                  footer="Heuristic coupling"
                />
                <MetricCard
                  label="Predicted uncertainty rise"
                  value={num(shift.uncertainty_increase, 3)}
                  provenance="computed"
                  hint="Also a heuristic coupling, not a measurement."
                  footer="Heuristic coupling"
                />
                <MetricCard
                  label="Reference domain"
                  value={shift.reference_domain.replace(/_/g, ' ')}
                  provenance="synthetic"
                  footer={shift.current_domain}
                />
              </section>

              <div className="grid gap-4 lg:grid-cols-5">
                <Card className="lg:col-span-3">
                  <CardHeader>
                    <CardTitle>Intensity distribution comparison</CardTitle>
                    <p className="text-[11px] text-muted-foreground">
                      The two histograms the Wasserstein and Jensen–Shannon terms
                      are computed from.
                    </p>
                  </CardHeader>
                  <CardContent>
                    <HistogramComparison shift={shift} />
                  </CardContent>
                </Card>

                <Card className="lg:col-span-2">
                  <CardHeader>
                    <CardTitle>Indicator breakdown</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div>
                      <div className="flex items-baseline justify-between">
                        <span className="text-xs text-muted-foreground">Composite</span>
                        <span className="tabular text-lg font-semibold">
                          {num(shift.shift_score, 4)}
                        </span>
                      </div>
                      <Progress value={shift.shift_score * 100} className="mt-1.5" />
                    </div>

                    {shift.indicators.map((indicator) => (
                      <div key={indicator.id} className="space-y-1">
                        <div className="flex items-baseline justify-between gap-2">
                          <span className="text-[11px] font-medium">{indicator.name}</span>
                          <span className="tabular shrink-0 text-[11px]">
                            {num(indicator.normalised, 3)} × {num(indicator.weight, 2)}
                          </span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                          <div
                            className="h-full rounded-full bg-status-info"
                            style={{ width: `${indicator.normalised * 100}%` }}
                          />
                        </div>
                        <p className="text-[10px] leading-relaxed text-muted-foreground">
                          {indicator.description} (raw {num(indicator.value, 4)})
                        </p>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                {[
                  ['Uploaded image', shift.current_distribution],
                  ['Reference domain', shift.reference_distribution],
                ].map(([title, dist]) => {
                  const d = dist as typeof shift.current_distribution;
                  return (
                    <Card key={String(title)}>
                      <CardHeader>
                        <CardTitle>{String(title)}</CardTitle>
                        <p className="font-mono text-[11px] text-muted-foreground">{d.name}</p>
                      </CardHeader>
                      <CardContent>
                        <table className="w-full text-xs">
                          <tbody className="divide-y divide-border/60">
                            {[
                              ['Mean intensity', num(d.mean, 4)],
                              ['Std. deviation', num(d.std, 4)],
                              ['Skewness', num(d.skewness, 4)],
                              ['Kurtosis', num(d.kurtosis, 4)],
                              ['Histogram entropy', num(d.entropy, 4)],
                            ].map(([label, value]) => (
                              <tr key={label}>
                                <td className="py-1.5 text-muted-foreground">{label}</td>
                                <td className="tabular py-1.5 text-right font-medium">{value}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Detector notes</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-1.5">
                    {shift.notes.map((note) => (
                      <li
                        key={note}
                        className="border-l-2 border-border pl-2 text-[11px] leading-relaxed text-muted-foreground"
                      >
                        {note}
                      </li>
                    ))}
                  </ul>
                  <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
                    No claim of clinical OOD detection performance is made. Validating
                    this detector requires labelled multi-site data.
                  </p>
                </CardContent>
              </Card>
            </>
          )}
        </>
      )}
    </PageContainer>
  );
}
