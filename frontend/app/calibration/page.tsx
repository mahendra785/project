'use client';

import { useState } from 'react';
import { GaugeCircle, Info, Play } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
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
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Label,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Slider,
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/primitives';
import { useCalibrationMethods, useCalibrationRun, useModels } from '@/hooks/use-pums';
import { num, pct } from '@/lib/utils';
import type { CalibrationResult } from '@/types/api';

/**
 * Reliability diagram.
 *
 * Bars are per-bin empirical accuracy; the dashed diagonal is perfect
 * calibration. The vertical distance between them, weighted by bin population,
 * *is* the ECE — so the chart and the reported number are the same quantity
 * rather than two independent computations.
 */
function ReliabilityDiagram({ result }: { result: CalibrationResult }) {
  const data = result.reliability_bins
    .filter((bin) => bin.count > 0)
    .map((bin) => ({
      confidence: (bin.bin_lower + bin.bin_upper) / 2,
      accuracy: bin.empirical_accuracy,
      perfect: (bin.bin_lower + bin.bin_upper) / 2,
      count: bin.count,
      gap: bin.empirical_accuracy - bin.mean_confidence,
    }));

  if (data.length === 0) {
    return <p className="py-8 text-center text-xs text-muted-foreground">No populated bins.</p>;
  }

  return (
    <ChartFrame height={230}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -20 }}>
        <CartesianGrid strokeDasharray="2 3" stroke="var(--border)" />
        <XAxis
          dataKey="confidence"
          type="number"
          domain={[0, 1]}
          tickFormatter={(v: number) => v.toFixed(1)}
          stroke="var(--muted-foreground)"
          fontSize={10}
          tickLine={false}
        />
        <YAxis domain={[0, 1]} stroke="var(--muted-foreground)" fontSize={10} tickLine={false} />
        <ChartTooltip
          formatter={(value, name) => [
            typeof value === 'number' ? value.toFixed(4) : value,
            name === 'accuracy' ? 'Empirical accuracy' : String(name),
          ]}
          labelFormatter={(label) =>
            `Confidence bin ≈ ${typeof label === 'number' ? label.toFixed(2) : label}`
          }
        />
        <Bar dataKey="accuracy" fill="var(--primary)" fillOpacity={0.75} radius={[2, 2, 0, 0]} />
        <Line
          type="linear"
          dataKey="perfect"
          stroke="var(--muted-foreground)"
          strokeDasharray="4 3"
          strokeWidth={1.5}
          dot={false}
          legendType="none"
        />
      </BarChart>
    </ChartFrame>
  );
}

function RiskCoverageChart({ result }: { result: CalibrationResult }) {
  const data = result.risk_coverage_curve.map((point) => ({
    coverage: point.coverage,
    risk: point.risk,
  }));

  if (data.length === 0) return null;

  return (
    <ChartFrame height={230}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
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
        <YAxis stroke="var(--muted-foreground)" fontSize={10} tickLine={false} />
        <ChartTooltip
          formatter={(value) => [typeof value === 'number' ? value.toFixed(4) : value, 'Selective risk']}
          labelFormatter={(label) =>
            `Coverage ${typeof label === 'number' ? (label * 100).toFixed(0) : label}%`
          }
        />
        <ReferenceLine
          y={result.target_risk}
          stroke="var(--status-warn)"
          strokeDasharray="4 3"
          label={{ value: `α = ${result.target_risk}`, fontSize: 10, fill: 'var(--status-warn)' }}
        />
        <Line
          type="monotone"
          dataKey="risk"
          stroke="var(--primary)"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ChartFrame>
  );
}

function ConfidenceDistribution({ result }: { result: CalibrationResult }) {
  const data = result.reliability_bins.map((bin) => ({
    bin: ((bin.bin_lower + bin.bin_upper) / 2).toFixed(2),
    count: bin.count,
  }));

  return (
    <ChartFrame height={160}>
      <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
        <CartesianGrid strokeDasharray="2 3" stroke="var(--border)" vertical={false} />
        <XAxis dataKey="bin" stroke="var(--muted-foreground)" fontSize={9} tickLine={false} interval={1} />
        <YAxis stroke="var(--muted-foreground)" fontSize={9} tickLine={false} />
        <ChartTooltip formatter={(value) => [`${value} px`, 'Pixels']} />
        <Bar dataKey="count" fill="var(--status-info)" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
      </BarChart>
    </ChartFrame>
  );
}

export default function CalibrationPage() {
  const models = useModels();
  const methods = useCalibrationMethods();

  const [modelId, setModelId] = useState<string>('');
  const [method, setMethod] = useState<string>('conformal_risk_control');
  const [targetRisk, setTargetRisk] = useState(0.1);
  const [pending, setPending] = useState({ method: 'conformal_risk_control', alpha: 0.1 });

  const activeModelId = modelId || models.data?.default_model_id || '';

  const run = useCalibrationRun({
    modelId: activeModelId,
    method: pending.method,
    targetRisk: pending.alpha,
    enabled: Boolean(activeModelId),
  });

  const selectedMethodInfo = methods.data?.methods.find((m) => m.id === method);
  const isPlanned = selectedMethodInfo?.status === 'planned';
  const result = run.data;

  return (
    <PageContainer>
      <PageHeader
        title="Calibration"
        description="Split Conformal Risk Control and temperature scaling, fitted on the shipped calibration set. Conformal guarantees are marginal and require exchangeability — they do not hold under distribution shift, and PUMS says so rather than continuing to display a number that no longer means what it says."
        badges={
          methods.data && <Badge variant="outline">{methods.data.methods.length} methods</Badge>
        }
      />

      <DatasetBanner />

      <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-1.5">
                <Label>Model</Label>
                <Select value={activeModelId} onValueChange={setModelId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    {models.data?.models.map((m) => (
                      <SelectItem key={m.id} value={m.id} disabled={!m.available}>
                        {m.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <Label>Calibration dataset</Label>
                <div className="rounded-md border border-border bg-muted/40 p-2.5">
                  <div className="flex items-center gap-1.5">
                    <ProvenanceBadge provenance="synthetic" />
                    <span className="text-[11px] font-medium">synthetic_phantom_v1</span>
                  </div>
                  <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground">
                    Procedurally generated phantoms with exact masks. Real model
                    probabilities, synthetic ground truth. Not a medical dataset.
                  </p>
                </div>
              </div>

              <div className="space-y-1.5">
                <Label>Method</Label>
                <Select value={method} onValueChange={setMethod}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {methods.data?.methods.map((m) => (
                      <SelectItem key={m.id} value={m.id}>
                        <span className="flex items-center gap-2">
                          {m.name}
                          {m.status === 'planned' && (
                            <span className="text-[10px] text-muted-foreground">
                              (planned)
                            </span>
                          )}
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label className="flex items-center gap-1">
                    Target risk α
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="size-3 cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>
                        The expected false-negative rate the procedure must
                        control. λ̂ is the smallest threshold whose finite-sample
                        corrected empirical risk meets it.
                      </TooltipContent>
                    </Tooltip>
                  </Label>
                  <span className="tabular text-[11px] text-muted-foreground">
                    {num(targetRisk, 2)}
                  </span>
                </div>
                <Slider
                  value={[targetRisk]}
                  min={0.01}
                  max={0.5}
                  step={0.01}
                  onValueChange={([v]) => setTargetRisk(v)}
                />
              </div>

              <Button
                className="w-full"
                disabled={run.isFetching || !activeModelId}
                onClick={() => setPending({ method, alpha: targetRisk })}
              >
                <Play className="size-4" />
                {run.isFetching ? 'Calibrating…' : 'Run calibration'}
              </Button>
            </CardContent>
          </Card>

          {selectedMethodInfo && (
            <Card>
              <CardHeader className="flex-row items-start justify-between">
                <CardTitle>{selectedMethodInfo.name}</CardTitle>
                <StatusBadge status={selectedMethodInfo.status} />
              </CardHeader>
              <CardContent className="space-y-2">
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  {selectedMethodInfo.description}
                </p>
                {selectedMethodInfo.reference && (
                  <p className="text-[10px] italic leading-relaxed text-muted-foreground">
                    {selectedMethodInfo.reference}
                  </p>
                )}
                {selectedMethodInfo.assumptions.length > 0 && (
                  <div className="space-y-1 border-t border-border pt-2">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      Assumptions
                    </p>
                    {selectedMethodInfo.assumptions.map((a) => (
                      <p
                        key={a}
                        className="border-l-2 border-border pl-2 text-[10px] leading-relaxed text-muted-foreground"
                      >
                        {a}
                      </p>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>

        <div className="space-y-4">
          {isPlanned && (
            <Card className="border-dashed">
              <CardContent className="space-y-2 px-6 py-10 text-center">
                <StatusBadge status="planned" className="mx-auto" />
                <p className="text-sm font-medium">{selectedMethodInfo?.name} is not implemented</p>
                <p className="mx-auto max-w-lg text-xs leading-relaxed text-muted-foreground">
                  {selectedMethodInfo?.description} Running it returns HTTP 501 from
                  the backend — the extension point exists, the algorithm does not.
                </p>
              </CardContent>
            </Card>
          )}

          {run.isFetching && <LoadingState label="Fitting on the calibration set…" />}
          {run.isError && <ErrorState error={run.error} onRetry={() => run.refetch()} />}

          {result && !run.isFetching && (
            <>
              <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="Expected Calibration Error"
                  value={num(result.expected_calibration_error, 4)}
                  icon={GaugeCircle}
                  provenance={result.provenance}
                  tone={result.expected_calibration_error > 0.1 ? 'warn' : 'ok'}
                  footer="Lower is better"
                />
                <MetricCard
                  label={
                    result.method === 'temperature_scaling' ? 'Fitted temperature' : 'Threshold λ̂'
                  }
                  value={num(result.threshold, 4)}
                  provenance={result.provenance}
                  footer={`α = ${num(result.target_risk, 2)}`}
                />
                <MetricCard
                  label="Empirical risk"
                  value={num(result.empirical_risk, 4)}
                  provenance={result.provenance}
                  tone={result.empirical_risk <= result.target_risk ? 'ok' : 'warn'}
                  footer={
                    result.empirical_risk <= result.target_risk
                      ? 'Meets target'
                      : 'Above target'
                  }
                />
                <MetricCard
                  label="Coverage"
                  value={pct(result.coverage, 2)}
                  provenance={result.provenance}
                  footer={`n = ${result.calibration_set_size}`}
                />
              </section>

              {result.validity_warnings.length > 0 && (
                <div className="space-y-1.5 rounded-md border border-status-warn/30 bg-status-warn/8 p-3">
                  {result.validity_warnings.map((w) => (
                    <p key={w} className="text-[11px] leading-relaxed text-muted-foreground">
                      {w}
                    </p>
                  ))}
                </div>
              )}

              <div className="grid gap-4 xl:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Reliability diagram</CardTitle>
                    <p className="text-[11px] text-muted-foreground">
                      Bars: empirical accuracy per confidence bin. Dashed line:
                      perfect calibration. Their population-weighted gap is the ECE.
                    </p>
                  </CardHeader>
                  <CardContent>
                    <ReliabilityDiagram result={result} />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Risk–coverage curve</CardTitle>
                    <p className="text-[11px] text-muted-foreground">
                      Selective risk as coverage grows. A useful uncertainty
                      estimator makes risk fall as the most uncertain predictions
                      are rejected.
                    </p>
                  </CardHeader>
                  <CardContent>
                    <RiskCoverageChart result={result} />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Confidence distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ConfidenceDistribution result={result} />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="flex-row items-start justify-between">
                    <CardTitle>Result detail</CardTitle>
                    <div className="flex items-center gap-1.5">
                      <StatusBadge status={result.status} />
                      <ProvenanceBadge provenance={result.provenance} />
                    </div>
                  </CardHeader>
                  <CardContent>
                    <table className="w-full text-xs">
                      <tbody className="divide-y divide-border/60">
                        {[
                          ['Method', result.method_name],
                          ['Target risk α', num(result.target_risk, 3)],
                          ['Selected threshold', num(result.threshold, 5)],
                          ['Empirical risk', num(result.empirical_risk, 5)],
                          ['Coverage', pct(result.coverage, 3)],
                          ['ECE', num(result.expected_calibration_error, 5)],
                          [
                            'Brier score',
                            result.brier_score === null ? '—' : num(result.brier_score, 5),
                          ],
                          ['Calibration set size', String(result.calibration_set_size)],
                          ['Provenance', result.provenance],
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
              </div>
            </>
          )}
        </div>
      </div>
    </PageContainer>
  );
}
