'use client';

import Link from 'next/link';
import { ArrowUpRight, ClipboardList, ScanSearch, Stethoscope } from 'lucide-react';

import { PageContainer, PageHeader } from '@/components/layout/page-container';
import { EmptyState, ErrorState, LoadingState } from '@/components/layout/states';
import { MetricCard } from '@/components/research/metric-card';
import {
  ResearchDisclaimer,
  RiskBadge,
  ShiftBadge,
} from '@/components/research/status-badges';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/primitives';
import { AnalysisPicker } from '@/features/results/analysis-picker';
import { RiskPanel } from '@/features/results/summary-panels';
import { useAnalyses, useAnalysis } from '@/hooks/use-pums';
import { mediaUrl } from '@/lib/api/client';
import { num, pct } from '@/lib/utils';
import { useAnalysisStore } from '@/stores/analysis-store';
import type { ReferralPriority, RiskLevel, ShiftStatus } from '@/types/api';

const PRIORITY_COPY: Record<ReferralPriority, { label: string; tone: 'ok' | 'warn' | 'danger'; note: string }> = {
  routine: {
    label: 'Routine',
    tone: 'ok',
    note: 'Model reliability signals are within the configured normal range for this input.',
  },
  elevated: {
    label: 'Elevated',
    tone: 'warn',
    note: 'Reliability signals suggest this case warrants earlier clinician attention than routine.',
  },
  urgent_review: {
    label: 'Urgent review',
    tone: 'danger',
    note: 'One or more override rules fired. The model output should not be relied on without clinician review.',
  },
};

const SEVERITY_BANDS: Array<[number, string]> = [
  [0.2, 'Minimal'],
  [0.4, 'Low'],
  [0.6, 'Moderate'],
  [0.8, 'High'],
  [1.01, 'Severe'],
];

function severityLabel(score: number): string {
  return SEVERITY_BANDS.find(([bound]) => score < bound)?.[1] ?? 'Severe';
}

export default function DecisionSupportPage() {
  const list = useAnalyses(50);
  const selectedId = useAnalysisStore((s) => s.selectedId);
  const setSelectedId = useAnalysisStore((s) => s.setSelectedId);

  const effectiveId =
    selectedId && list.data?.items.some((i) => i.analysis_id === selectedId)
      ? selectedId
      : (list.data?.items[0]?.analysis_id ?? null);
  const analysis = useAnalysis(effectiveId);

  const queue = [...(list.data?.items ?? [])].sort((a, b) => b.risk_score - a.risk_score);

  return (
    <PageContainer>
      <PageHeader
        title="Decision support"
        description="A research prototype that turns uncertainty into an actionable triage signal. It describes how reliable the MODEL is on a given input — it says nothing about the patient, and it is not a diagnosis."
        badges={<Badge variant="warn">Research prototype</Badge>}
      />

      <ResearchDisclaimer text="Not a diagnostic result. Not clinically validated. Not for patient care. Outputs describe model reliability only and must be interpreted by a qualified clinician alongside the underlying images." />

      {list.isLoading && <LoadingState label="Loading analyses…" />}
      {list.isError && <ErrorState error={list.error} onRetry={() => list.refetch()} />}

      {!list.isLoading && (list.data?.items.length ?? 0) === 0 && (
        <EmptyState
          title="No cases in the queue"
          description="The review queue is built from analyses run on this instance. Run one to see how uncertainty, calibration and distribution shift combine into a triage signal."
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
          </div>

          {analysis.isLoading && <LoadingState />}
          {analysis.isError && (
            <ErrorState error={analysis.error} onRetry={() => analysis.refetch()} />
          )}

          {analysis.data && (
            <>
              {(() => {
                const risk = analysis.data.risk_assessment;
                const uncertainty = analysis.data.uncertainty;
                const priority = PRIORITY_COPY[risk.referral_priority];
                const flagged = uncertainty.objects.filter(
                  (o) => o.risk_level === 'HIGH' || o.risk_level === 'REVIEW_REQUIRED',
                );

                return (
                  <>
                    <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                      <MetricCard
                        label="Uncertainty severity"
                        value={severityLabel(uncertainty.global.score)}
                        icon={Stethoscope}
                        provenance="computed"
                        footer={`Mean ${num(uncertainty.global.score, 3)}`}
                      />
                      <MetricCard
                        label="Risk level"
                        value={risk.risk_level.replace('_', ' ')}
                        tone={
                          risk.risk_level === 'LOW'
                            ? 'ok'
                            : risk.risk_level === 'REVIEW_REQUIRED'
                              ? 'danger'
                              : 'warn'
                        }
                        provenance="computed"
                        footer={`Score ${num(risk.risk_score, 3)}`}
                      />
                      <MetricCard
                        label="Referral priority"
                        value={priority.label}
                        tone={priority.tone}
                        provenance="computed"
                        footer={risk.review_recommended ? 'Review recommended' : 'No review flagged'}
                      />
                      <MetricCard
                        label="Flagged regions"
                        value={String(flagged.length)}
                        icon={ClipboardList}
                        provenance="computed"
                        tone={flagged.length > 0 ? 'warn' : 'default'}
                        footer={`of ${uncertainty.objects.length} identified`}
                      />
                    </section>

                    <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
                      <div className="space-y-4">
                        <Card>
                          <CardHeader className="flex-row items-start justify-between">
                            <div>
                              <CardTitle>Review recommendation</CardTitle>
                              <p className="mt-0.5 text-[11px] text-muted-foreground">
                                {analysis.data.metadata.image.filename}
                              </p>
                            </div>
                            <RiskBadge level={risk.risk_level} />
                          </CardHeader>
                          <CardContent className="space-y-3">
                            <div
                              className={
                                risk.review_recommended
                                  ? 'rounded-md border border-status-warn/40 bg-status-warn/8 p-3'
                                  : 'rounded-md border border-status-ok/40 bg-status-ok/8 p-3'
                              }
                            >
                              <p className="text-sm font-medium">
                                {risk.review_recommended
                                  ? 'Clinician review recommended'
                                  : 'No review flagged by the current rules'}
                              </p>
                              <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                                {priority.note}
                              </p>
                            </div>

                            <div>
                              <div className="flex items-baseline justify-between">
                                <span className="text-xs text-muted-foreground">
                                  Composite risk score
                                </span>
                                <span className="tabular text-lg font-semibold">
                                  {num(risk.risk_score, 4)}
                                </span>
                              </div>
                              <Progress value={risk.risk_score * 100} className="mt-1.5" />
                              <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
                                <span>low {num(risk.thresholds.low, 2)}</span>
                                <span>moderate {num(risk.thresholds.moderate, 2)}</span>
                                <span>high {num(risk.thresholds.high, 2)}</span>
                              </div>
                            </div>
                          </CardContent>
                        </Card>

                        <Card>
                          <CardHeader className="flex-row items-center justify-between">
                            <CardTitle>High-uncertainty regions for review</CardTitle>
                            <Button asChild variant="ghost" size="sm">
                              <Link href={`/uncertainty?id=${analysis.data.analysis_id}`}>
                                Open viewer
                                <ArrowUpRight className="size-3.5" />
                              </Link>
                            </Button>
                          </CardHeader>
                          <CardContent className="p-0">
                            {flagged.length === 0 ? (
                              <p className="px-4 py-8 text-center text-xs text-muted-foreground">
                                No region reached the high-risk band. Regions below that
                                band are still listed on the Uncertainty page.
                              </p>
                            ) : (
                              <table className="w-full text-xs">
                                <thead className="border-b border-border text-[10px] uppercase tracking-wider text-muted-foreground">
                                  <tr>
                                    <th className="px-4 py-2 text-left font-semibold">Region</th>
                                    <th className="px-3 py-2 text-right font-semibold">
                                      Uncertainty
                                    </th>
                                    <th className="px-3 py-2 text-right font-semibold">Peak</th>
                                    <th className="px-3 py-2 text-right font-semibold">Area</th>
                                    <th className="px-3 py-2 text-left font-semibold">Location</th>
                                    <th className="px-4 py-2 text-left font-semibold">Band</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {flagged.map((object) => (
                                    <tr
                                      key={object.id}
                                      className="border-b border-border/60 last:border-0"
                                    >
                                      <td className="max-w-[200px] truncate px-4 py-2 font-medium">
                                        {object.label}
                                      </td>
                                      <td className="tabular px-3 py-2 text-right">
                                        {num(object.uncertainty, 3)}
                                      </td>
                                      <td className="tabular px-3 py-2 text-right text-muted-foreground">
                                        {num(object.peak_uncertainty, 3)}
                                      </td>
                                      <td className="tabular px-3 py-2 text-right text-muted-foreground">
                                        {object.area_px} px
                                      </td>
                                      <td className="tabular px-3 py-2 font-mono text-[10px] text-muted-foreground">
                                        ({object.bbox.x}, {object.bbox.y})
                                      </td>
                                      <td className="px-4 py-2">
                                        <RiskBadge level={object.risk_level} />
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            )}
                          </CardContent>
                        </Card>

                        <Card>
                          <CardHeader className="flex-row items-center justify-between">
                            <CardTitle>Triage queue</CardTitle>
                            <span className="text-[11px] text-muted-foreground">
                              Ordered by risk score
                            </span>
                          </CardHeader>
                          <CardContent className="p-0">
                            <table className="w-full text-xs">
                              <thead className="border-b border-border text-[10px] uppercase tracking-wider text-muted-foreground">
                                <tr>
                                  <th className="px-4 py-2 text-left font-semibold">Case</th>
                                  <th className="px-3 py-2 text-right font-semibold">Risk</th>
                                  <th className="px-3 py-2 text-right font-semibold">
                                    Uncertainty
                                  </th>
                                  <th className="px-3 py-2 text-left font-semibold">Shift</th>
                                  <th className="px-4 py-2 text-left font-semibold">Band</th>
                                </tr>
                              </thead>
                              <tbody>
                                {queue.slice(0, 10).map((item) => (
                                  <tr
                                    key={item.analysis_id}
                                    onClick={() => setSelectedId(item.analysis_id)}
                                    className={
                                      item.analysis_id === effectiveId
                                        ? 'cursor-pointer border-b border-border/60 bg-primary/8 last:border-0'
                                        : 'cursor-pointer border-b border-border/60 last:border-0 hover:bg-accent/40'
                                    }
                                  >
                                    <td className="max-w-[220px] truncate px-4 py-2 font-medium">
                                      {item.filename}
                                    </td>
                                    <td className="tabular px-3 py-2 text-right">
                                      {num(item.risk_score, 3)}
                                    </td>
                                    <td className="tabular px-3 py-2 text-right text-muted-foreground">
                                      {num(item.mean_uncertainty, 3)}
                                    </td>
                                    <td className="px-3 py-2">
                                      {item.shift_status ? (
                                        <ShiftBadge status={item.shift_status as ShiftStatus} />
                                      ) : (
                                        <span className="text-muted-foreground">—</span>
                                      )}
                                    </td>
                                    <td className="px-4 py-2">
                                      <RiskBadge level={item.risk_level as RiskLevel} />
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </CardContent>
                        </Card>
                      </div>

                      <div className="space-y-4">
                        <Card>
                          <CardHeader>
                            <CardTitle>Case preview</CardTitle>
                          </CardHeader>
                          <CardContent className="space-y-2">
                            <div className="grid grid-cols-2 gap-2">
                              {[
                                ['Source', mediaUrl(analysis.data.metadata.image.preview_url)],
                                [
                                  'Uncertainty',
                                  mediaUrl(analysis.data.uncertainty.pixel.heatmap_url),
                                ],
                              ].map(([label, url]) => (
                                <div key={String(label)} className="space-y-1">
                                  <p className="text-[10px] text-muted-foreground">
                                    {String(label)}
                                  </p>
                                  {url ? (
                                    // eslint-disable-next-line @next/next/no-img-element -- backend PNG
                                    <img
                                      src={String(url)}
                                      alt={String(label)}
                                      className="checkerboard aspect-square w-full rounded border border-border object-contain"
                                      style={{ imageRendering: 'pixelated' }}
                                    />
                                  ) : null}
                                </div>
                              ))}
                            </div>
                            <div className="space-y-1 pt-1">
                              <div className="flex justify-between text-[11px]">
                                <span className="text-muted-foreground">Mean confidence</span>
                                <span className="tabular font-medium">
                                  {pct(analysis.data.prediction.confidence, 1)}
                                </span>
                              </div>
                              <div className="flex justify-between text-[11px]">
                                <span className="text-muted-foreground">High-uncertainty area</span>
                                <span className="tabular font-medium">
                                  {pct(uncertainty.global.high_uncertainty_fraction, 2)}
                                </span>
                              </div>
                              {analysis.data.distribution_shift && (
                                <div className="flex items-center justify-between text-[11px]">
                                  <span className="text-muted-foreground">Domain</span>
                                  <ShiftBadge
                                    status={analysis.data.distribution_shift.status}
                                  />
                                </div>
                              )}
                            </div>
                          </CardContent>
                        </Card>

                        <RiskPanel risk={risk} />
                      </div>
                    </div>
                  </>
                );
              })()}
            </>
          )}
        </>
      )}
    </PageContainer>
  );
}
