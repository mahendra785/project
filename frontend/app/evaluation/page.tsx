'use client';

import { CheckCircle2, FlaskConical, XCircle } from 'lucide-react';
import {
  CartesianGrid,
  Line,
  LineChart,
  Scatter,
  ScatterChart,
  ZAxis,
} from 'recharts';

import {
  ChartFrame,
  ChartTooltip,
  XAxis,
  YAxis,
} from '@/components/charts/chart-primitives';
import { PageContainer, PageHeader } from '@/components/layout/page-container';
import { DatasetBanner } from '@/components/research/dataset-banner';
import { ErrorState, LoadingState } from '@/components/layout/states';
import { MetricCard } from '@/components/research/metric-card';
import { ProvenanceBadge, StatusBadge } from '@/components/research/status-badges';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/primitives';
import { useEvaluation, useSelfTest } from '@/hooks/use-pums';
import { humanise, num } from '@/lib/utils';

const CATEGORY_LABEL: Record<string, string> = {
  segmentation: 'Segmentation',
  calibration: 'Calibration',
  uncertainty: 'Uncertainty',
  distribution_shift: 'Distribution shift',
  detection: 'Detection',
};

export default function EvaluationPage() {
  const evaluation = useEvaluation(true);
  const selfTest = useSelfTest();

  const metrics = evaluation.data?.metrics ?? [];
  const available = metrics.filter((m) => m.value !== null);
  const unavailable = metrics.filter((m) => m.value === null);

  const scatterData =
    evaluation.data?.uncertainty_vs_confidence.map((row) => ({
      uncertainty: Number(row.uncertainty),
      confidence: Number(row.confidence),
      risk: Number(row.risk),
      id: String(row.analysis_id),
    })) ?? [];

  return (
    <PageContainer>
      <PageHeader
        title="Evaluation"
        description="PUMS ships no medical benchmark results. This page reports two things only: metrics computed over analyses this instance actually ran, and metrics computed on the synthetic phantom set whose masks are exact by construction. Everything requiring annotated clinical data is reported as unavailable, with the reason."
        badges={
          evaluation.data && (
            <Badge variant="outline">
              {evaluation.data.session_sample_size} session analyses
            </Badge>
          )
        }
      />

      <DatasetBanner />

      {evaluation.isLoading && <LoadingState label="Computing evaluation metrics…" />}
      {evaluation.isError && (
        <ErrorState error={evaluation.error} onRetry={() => evaluation.refetch()} />
      )}

      {evaluation.data && (
        <>
          {evaluation.data.notes.length > 0 && (
            <div className="space-y-1.5 rounded-md border border-border bg-muted/40 p-3">
              {evaluation.data.notes.map((note) => (
                <p key={note} className="text-[11px] leading-relaxed text-muted-foreground">
                  {note}
                </p>
              ))}
            </div>
          )}

          <Tabs defaultValue="measured">
            <TabsList>
              <TabsTrigger value="measured">Measured ({available.length})</TabsTrigger>
              <TabsTrigger value="unavailable">
                Not measured ({unavailable.length})
              </TabsTrigger>
              <TabsTrigger value="definitions">Definitions</TabsTrigger>
              <TabsTrigger value="verification">Code verification</TabsTrigger>
            </TabsList>

            {/* ------------------------------------------------- measured */}
            <TabsContent value="measured" className="space-y-4">
              <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {available.map((metric) => (
                  <MetricCard
                    key={metric.id}
                    label={metric.name}
                    value={num(metric.value, 4)}
                    provenance={metric.provenance}
                    footer={`${CATEGORY_LABEL[metric.category] ?? metric.category} · n = ${metric.sample_size}`}
                  />
                ))}
              </section>

              <div className="grid gap-4 lg:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Risk–coverage curve</CardTitle>
                    <p className="text-[11px] text-muted-foreground">
                      Selective risk over the phantom set as coverage increases.
                      Risk rising with coverage is the signature of an uncertainty
                      map that genuinely ranks errors.
                    </p>
                  </CardHeader>
                  <CardContent>
                    {evaluation.data.risk_coverage_curve.length === 0 ? (
                      <p className="py-10 text-center text-xs text-muted-foreground">
                        Not available.
                      </p>
                    ) : (
                      <ChartFrame height={230}>
                        <LineChart
                          data={evaluation.data.risk_coverage_curve}
                          margin={{ top: 8, right: 8, bottom: 4, left: -14 }}
                        >
                          <CartesianGrid strokeDasharray="2 3" stroke="var(--border)" />
                          <XAxis
                            dataKey="coverage"
                            type="number"
                            domain={[0, 1]}
                            tickFormatter={(v: number) => v.toFixed(1)}
                            stroke="var(--muted-foreground)"
                            fontSize={10}
                            tickLine={false}
                          />
                          <YAxis
                            stroke="var(--muted-foreground)"
                            fontSize={10}
                            tickLine={false}
                          />
                          <ChartTooltip
                            formatter={(value) => [
                              typeof value === 'number' ? value.toFixed(5) : value,
                              'Selective risk',
                            ]}
                            labelFormatter={(label) =>
                              `Coverage ${typeof label === 'number' ? (label * 100).toFixed(0) : label}%`
                            }
                          />
                          <Line
                            type="monotone"
                            dataKey="selective_risk"
                            stroke="var(--primary)"
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ChartFrame>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Session analyses · uncertainty vs confidence</CardTitle>
                    <p className="text-[11px] text-muted-foreground">
                      One point per analysis run on this instance. Point size
                      encodes the risk score.
                    </p>
                  </CardHeader>
                  <CardContent>
                    {scatterData.length === 0 ? (
                      <p className="py-10 text-center text-xs text-muted-foreground">
                        No analyses yet.
                      </p>
                    ) : (
                      <ChartFrame height={230}>
                        <ScatterChart margin={{ top: 8, right: 8, bottom: 4, left: -14 }}>
                          <CartesianGrid strokeDasharray="2 3" stroke="var(--border)" />
                          <XAxis
                            dataKey="uncertainty"
                            type="number"
                            name="Uncertainty"
                            domain={[0, 1]}
                            stroke="var(--muted-foreground)"
                            fontSize={10}
                            tickLine={false}
                          />
                          <YAxis
                            dataKey="confidence"
                            type="number"
                            name="Confidence"
                            domain={[0, 1]}
                            stroke="var(--muted-foreground)"
                            fontSize={10}
                            tickLine={false}
                          />
                          <ZAxis dataKey="risk" range={[40, 260]} />
                          <ChartTooltip
                            cursor={{ strokeDasharray: '3 3' }}
                            formatter={(value, name) => [
                              typeof value === 'number' ? value.toFixed(4) : value,
                              humanise(String(name)),
                            ]}
                          />
                          <Scatter
                            data={scatterData}
                            fill="var(--primary)"
                            fillOpacity={0.6}
                          />
                        </ScatterChart>
                      </ChartFrame>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* ---------------------------------------------- unavailable */}
            <TabsContent value="unavailable">
              <Card>
                <CardHeader>
                  <CardTitle>Metrics deliberately not reported</CardTitle>
                  <p className="text-[11px] leading-relaxed text-muted-foreground">
                    Each of these is either unimplemented or requires ground truth
                    PUMS does not have. Reporting a number here would be fabrication,
                    so the reason is shown instead.
                  </p>
                </CardHeader>
                <CardContent className="p-0">
                  <table className="w-full text-xs">
                    <thead className="border-b border-border text-[10px] uppercase tracking-wider text-muted-foreground">
                      <tr>
                        <th className="px-4 py-2 text-left font-semibold">Metric</th>
                        <th className="px-3 py-2 text-left font-semibold">Category</th>
                        <th className="px-3 py-2 text-left font-semibold">Why not reported</th>
                      </tr>
                    </thead>
                    <tbody>
                      {unavailable.map((metric) => (
                        <tr key={metric.id} className="border-b border-border/60 last:border-0">
                          <td className="px-4 py-2 font-medium">{metric.name}</td>
                          <td className="px-3 py-2 text-muted-foreground">
                            {CATEGORY_LABEL[metric.category] ?? metric.category}
                          </td>
                          <td className="px-3 py-2 leading-relaxed text-muted-foreground">
                            {metric.unavailable_reason}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>
            </TabsContent>

            {/* ---------------------------------------------- definitions */}
            <TabsContent value="definitions">
              <Card>
                <CardHeader>
                  <CardTitle>Metric definitions and implementation status</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <table className="w-full text-xs">
                    <thead className="border-b border-border text-[10px] uppercase tracking-wider text-muted-foreground">
                      <tr>
                        <th className="px-4 py-2 text-left font-semibold">Metric</th>
                        <th className="px-3 py-2 text-left font-semibold">Category</th>
                        <th className="px-3 py-2 text-left font-semibold">Status</th>
                        <th className="px-3 py-2 text-left font-semibold">Needs labels</th>
                        <th className="px-3 py-2 text-left font-semibold">Definition</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evaluation.data.definitions.map((definition) => (
                        <tr key={definition.id} className="border-b border-border/60 last:border-0">
                          <td className="px-4 py-2 font-medium">{definition.name}</td>
                          <td className="px-3 py-2 text-muted-foreground">
                            {CATEGORY_LABEL[definition.category] ?? definition.category}
                          </td>
                          <td className="px-3 py-2">
                            <StatusBadge status={definition.status} />
                          </td>
                          <td className="px-3 py-2 text-muted-foreground">
                            {definition.requires_ground_truth ? 'Yes' : 'No'}
                          </td>
                          <td className="max-w-md px-3 py-2 leading-relaxed text-muted-foreground">
                            {definition.description}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>
            </TabsContent>

            {/* --------------------------------------------- verification */}
            <TabsContent value="verification">
              <Card>
                <CardHeader className="flex-row items-start justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-1.5">
                      <FlaskConical className="size-4" />
                      Metric implementation verification
                    </CardTitle>
                    <p className="mt-1 max-w-2xl text-[11px] leading-relaxed text-muted-foreground">
                      {selfTest.data?.note}
                    </p>
                  </div>
                  {selfTest.data && (
                    <Badge variant={selfTest.data.all_passed ? 'ok' : 'danger'}>
                      {selfTest.data.all_passed ? 'All passed' : 'Failures'}
                    </Badge>
                  )}
                </CardHeader>
                <CardContent className="p-0">
                  {selfTest.isLoading && <LoadingState />}
                  {selfTest.isError && <ErrorState error={selfTest.error} />}
                  {selfTest.data && (
                    <table className="w-full text-xs">
                      <thead className="border-b border-border text-[10px] uppercase tracking-wider text-muted-foreground">
                        <tr>
                          <th className="px-4 py-2 text-left font-semibold">Metric</th>
                          <th className="px-3 py-2 text-left font-semibold">Scenario</th>
                          <th className="px-3 py-2 text-right font-semibold">Expected</th>
                          <th className="px-3 py-2 text-right font-semibold">Actual</th>
                          <th className="px-4 py-2 text-center font-semibold">Result</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selfTest.data.results.map((result, index) => (
                          <tr
                            key={`${result.metric}-${index}`}
                            className="border-b border-border/60 last:border-0"
                          >
                            <td className="px-4 py-2 font-mono">{result.metric}</td>
                            <td className="px-3 py-2 leading-relaxed text-muted-foreground">
                              {result.scenario}
                            </td>
                            <td className="tabular px-3 py-2 text-right">
                              {num(result.expected, 6)}
                            </td>
                            <td className="tabular px-3 py-2 text-right">
                              {num(result.actual, 6)}
                            </td>
                            <td className="px-4 py-2 text-center">
                              {result.passed ? (
                                <CheckCircle2 className="mx-auto size-4 text-status-ok" />
                              ) : (
                                <XCircle className="mx-auto size-4 text-status-danger" />
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          <p className="text-[11px] text-muted-foreground">
            Generated {evaluation.data.generated_at} ·{' '}
            <ProvenanceBadge provenance="synthetic" className="align-middle" /> figures
            describe phantom behaviour, not clinical performance.
          </p>
        </>
      )}
    </PageContainer>
  );
}
