'use client';

import Link from 'next/link';
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  Boxes,
  GaugeCircle,
  Radar,
  ScanSearch,
  Sigma,
} from 'lucide-react';
import { Bar, BarChart, Cell } from 'recharts';

import {
  ChartFrame,
  ChartGrid,
  ChartTooltip,
  XAxis,
  YAxis,
} from '@/components/charts/chart-primitives';
import { PageContainer, PageHeader } from '@/components/layout/page-container';
import { EmptyState, ErrorState, Skeleton } from '@/components/layout/states';
import { MetricCard } from '@/components/research/metric-card';
import {
  ResearchDisclaimer,
  RiskBadge,
  ShiftBadge,
} from '@/components/research/status-badges';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useAnalyses, useDashboard, useHealth, useModels } from '@/hooks/use-pums';
import { uncertaintyColour } from '@/hooks/use-uncertainty-matrix';
import { num, pct, relativeTime } from '@/lib/utils';
import type { RiskLevel, ShiftStatus } from '@/types/api';

function UncertaintyHistogram({ counts }: { counts: number[] }) {
  const data = counts.map((count, index) => {
    const centre = (index + 0.5) / counts.length;
    const [r, g, b] = uncertaintyColour(centre);
    return {
      bin: `${(index / counts.length).toFixed(1)}`,
      count,
      fill: `rgb(${r}, ${g}, ${b})`,
    };
  });

  return (
    <ChartFrame height={170}>
      <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -22 }}>
        <ChartGrid />
        <XAxis dataKey="bin" stroke="var(--muted-foreground)" fontSize={10} tickLine={false} />
        <YAxis stroke="var(--muted-foreground)" fontSize={10} tickLine={false} allowDecimals={false} />
        <ChartTooltip
          formatter={(value) => [`${value} analyses`, 'Count']}
          labelFormatter={(label) => `Mean uncertainty ≥ ${label}`}
        />
        <Bar dataKey="count" radius={[2, 2, 0, 0]}>
          {data.map((entry) => (
            <Cell key={entry.bin} fill={entry.fill} />
          ))}
        </Bar>
      </BarChart>
    </ChartFrame>
  );
}

export default function DashboardPage() {
  const dashboard = useDashboard();
  const analyses = useAnalyses(8);
  const models = useModels();
  const health = useHealth();

  if (dashboard.isError) {
    return (
      <PageContainer>
        <PageHeader title="Dashboard" />
        <ErrorState error={dashboard.error} onRetry={() => dashboard.refetch()} />
      </PageContainer>
    );
  }

  const stats = dashboard.data;
  const hasData = (stats?.completed_analyses ?? 0) > 0;

  return (
    <PageContainer>
      <PageHeader
        title="Research dashboard"
        description="Aggregates over analyses run on this backend instance. Every figure is computed from your own runs — no seeded, illustrative or benchmark numbers appear on this page."
        actions={
          <Button asChild size="sm">
            <Link href="/analyze">
              <ScanSearch className="size-4" />
              New analysis
            </Link>
          </Button>
        }
      />

      {dashboard.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-[86px]" />
          ))}
        </div>
      ) : !stats ? null : (
        <>
          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="Analyses completed"
              value={String(stats.completed_analyses)}
              icon={Activity}
              provenance="computed"
              footer={
                stats.failed_analyses > 0
                  ? `${stats.failed_analyses} failed`
                  : 'On this instance'
              }
            />
            <MetricCard
              label="Mean uncertainty"
              value={num(stats.mean_uncertainty, 3)}
              icon={BrainCircuit}
              provenance="computed"
              hint="Mean of the per-analysis Level-1 uncertainty (normalised predictive entropy), averaged over completed analyses."
              footer={`Mean confidence ${num(stats.mean_confidence, 3)}`}
            />
            <MetricCard
              label="Cases needing review"
              value={String(stats.review_required_count)}
              icon={AlertTriangle}
              tone={stats.review_required_count > 0 ? 'danger' : 'default'}
              provenance="computed"
              footer={`${stats.high_risk_count} at high risk or above`}
            />
            <MetricCard
              label="Shift-flagged inputs"
              value={String(stats.shift_flagged_count)}
              icon={Radar}
              tone={stats.shift_flagged_count > 0 ? 'warn' : 'default'}
              provenance="computed"
              footer={`Mean shift score ${num(stats.mean_shift_score, 3)}`}
            />
          </section>

          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="Calibration status"
              value={stats.mean_ece === null ? '—' : num(stats.mean_ece, 4)}
              icon={GaugeCircle}
              provenance="synthetic"
              hint="Expected Calibration Error measured on the synthetic phantom calibration set. Characterises phantom behaviour, not clinical performance."
              footer="Mean ECE · conformal risk control"
            />
            <MetricCard
              label="Active model"
              value={
                models.data?.models.find((m) => m.id === stats.active_model_id)
                  ?.model_type ?? '—'
              }
              icon={Boxes}
              hint={stats.active_model_id}
              footer={stats.active_model_name}
            />
            <MetricCard
              label="Model adapters"
              value={String(models.data?.models.length ?? 0)}
              icon={Sigma}
              footer={`${health.data?.available_models.length ?? 0} runnable here`}
            />
            <MetricCard
              label="Uncertainty method"
              value="Entropy"
              icon={BrainCircuit}
              hint="Normalised predictive entropy — a single-forward-pass, post-hoc estimator derived from the model's own probabilities."
              footer="Normalised predictive entropy"
            />
          </section>

          <div className="grid gap-4 lg:grid-cols-5">
            <Card className="lg:col-span-3">
              <CardHeader className="flex-row items-center justify-between">
                <CardTitle>Uncertainty distribution</CardTitle>
                <span className="text-[11px] text-muted-foreground">
                  {stats.completed_analyses} analyses · 10 bins
                </span>
              </CardHeader>
              <CardContent>
                {hasData ? (
                  <UncertaintyHistogram counts={stats.uncertainty_histogram} />
                ) : (
                  <p className="py-12 text-center text-xs text-muted-foreground">
                    No analyses yet. The chart populates from real runs only.
                  </p>
                )}
              </CardContent>
            </Card>

            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Reliability summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-1.5">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Risk bands
                  </p>
                  {(['LOW', 'MODERATE', 'HIGH', 'REVIEW_REQUIRED'] as RiskLevel[]).map(
                    (level) => {
                      const count = stats.risk_breakdown[level] ?? 0;
                      const share = stats.completed_analyses
                        ? count / stats.completed_analyses
                        : 0;
                      return (
                        <div key={level} className="flex items-center gap-2">
                          <RiskBadge level={level} className="w-32 justify-start" />
                          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                            <div
                              className="h-full rounded-full bg-primary"
                              style={{ width: `${share * 100}%` }}
                            />
                          </div>
                          <span className="tabular w-6 text-right text-[11px] text-muted-foreground">
                            {count}
                          </span>
                        </div>
                      );
                    },
                  )}
                </div>

                <div className="space-y-1.5 border-t border-border pt-3">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Distribution shift
                  </p>
                  {(
                    ['ID', 'MILD_SHIFT', 'MODERATE_SHIFT', 'SEVERE_SHIFT'] as ShiftStatus[]
                  ).map((status) => (
                    <div key={status} className="flex items-center justify-between gap-2">
                      <ShiftBadge status={status} />
                      <span className="tabular text-[11px] text-muted-foreground">
                        {stats.shift_breakdown[status] ?? 0}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader className="flex-row items-center justify-between">
              <CardTitle>Recent analyses</CardTitle>
              <Button asChild variant="ghost" size="sm">
                <Link href="/uncertainty">View all</Link>
              </Button>
            </CardHeader>
            <CardContent className="p-0">
              {analyses.isLoading ? (
                <div className="space-y-2 p-4">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <Skeleton key={i} className="h-9" />
                  ))}
                </div>
              ) : (analyses.data?.items.length ?? 0) === 0 ? (
                <EmptyState
                  className="m-4 border-0 shadow-none"
                  title="No analyses yet"
                  description="Upload an image on the Analyze page to run the full pipeline: preprocessing, segmentation, hierarchical uncertainty, calibration, shift detection and risk assessment."
                  actionLabel="Run your first analysis"
                  actionHref="/analyze"
                  icon={<ScanSearch className="size-5" />}
                />
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead className="border-b border-border text-[10px] uppercase tracking-wider text-muted-foreground">
                      <tr>
                        <th className="px-4 py-2 text-left font-semibold">File</th>
                        <th className="px-3 py-2 text-left font-semibold">Model</th>
                        <th className="px-3 py-2 text-right font-semibold">Uncertainty</th>
                        <th className="px-3 py-2 text-right font-semibold">Confidence</th>
                        <th className="px-3 py-2 text-left font-semibold">Shift</th>
                        <th className="px-3 py-2 text-left font-semibold">Risk</th>
                        <th className="px-4 py-2 text-right font-semibold">When</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analyses.data?.items.map((item) => (
                        <tr
                          key={item.analysis_id}
                          className="border-b border-border/60 last:border-0 hover:bg-accent/40"
                        >
                          <td className="max-w-[180px] truncate px-4 py-2 font-medium">
                            <Link
                              href={`/uncertainty?id=${item.analysis_id}`}
                              className="hover:underline"
                            >
                              {item.filename}
                            </Link>
                          </td>
                          <td className="px-3 py-2 text-muted-foreground">
                            <Badge variant="outline">{item.model_id}</Badge>
                          </td>
                          <td className="tabular px-3 py-2 text-right">
                            {num(item.mean_uncertainty, 3)}
                          </td>
                          <td className="tabular px-3 py-2 text-right">
                            {pct(item.confidence, 1)}
                          </td>
                          <td className="px-3 py-2">
                            {item.shift_status ? (
                              <ShiftBadge status={item.shift_status as ShiftStatus} />
                            ) : (
                              <span className="text-muted-foreground">—</span>
                            )}
                          </td>
                          <td className="px-3 py-2">
                            <RiskBadge level={item.risk_level as RiskLevel} />
                          </td>
                          <td className="px-4 py-2 text-right text-muted-foreground">
                            {relativeTime(item.created_at)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}

      <ResearchDisclaimer text="PUMS is a B.Tech research prototype. It ships no clinically validated model and no medical benchmark results. Risk outputs describe model reliability, not patient condition." />
    </PageContainer>
  );
}
