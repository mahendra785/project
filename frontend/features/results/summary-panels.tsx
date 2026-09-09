'use client';

/** Compact result panels reused by /analyze, /uncertainty and /decision-support. */

import {
  AlertTriangle,
  BrainCircuit,
  GaugeCircle,
  Radar,
  Shield,
  Timer,
} from 'lucide-react';

import { DataRow } from '@/components/layout/page-container';
import {
  ProvenanceBadge,
  RiskBadge,
  ShiftBadge,
  StatusBadge,
} from '@/components/research/status-badges';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/primitives';
import { humanise, ms, num, pct } from '@/lib/utils';
import type {
  AnalysisResponse,
  CalibrationResult,
  DistributionShiftResult,
  HierarchicalUncertainty,
  RiskAssessment,
} from '@/types/api';

export function GlobalUncertaintyPanel({
  uncertainty,
}: {
  uncertainty: HierarchicalUncertainty;
}) {
  const g = uncertainty.global;
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-1.5">
          <BrainCircuit className="size-4" />
          Level 1 · Global uncertainty
        </CardTitle>
        <div className="flex items-center gap-1.5">
          <StatusBadge status={uncertainty.status} />
          <ProvenanceBadge provenance={g.provenance} />
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <div className="flex items-baseline justify-between">
            <span className="text-xs text-muted-foreground">Mean uncertainty</span>
            <span className="tabular text-xl font-semibold">{num(g.score, 4)}</span>
          </div>
          <Progress value={g.score * 100} className="mt-1.5" />
        </div>

        <div className="divide-y divide-border/60">
          <DataRow
            label="High-uncertainty area"
            value={pct(g.high_uncertainty_fraction, 2)}
            hint="Fraction of pixels at or above the configured threshold."
          />
          <DataRow
            label="Boundary uncertainty"
            value={num(g.boundary_uncertainty, 4)}
            hint="Mean uncertainty on pixels adjacent to a class boundary."
          />
          <DataRow
            label="Interior uncertainty"
            value={num(g.interior_uncertainty, 4)}
            hint="Mean uncertainty away from class boundaries."
          />
          <DataRow
            label="Aggregate distribution entropy"
            value={num(g.aggregate_distribution_entropy, 4)}
            hint="Entropy of the spatially-averaged class distribution — distinct from the mean of per-pixel entropies."
          />
          <DataRow label="Median · p95 · max" value={
            `${num(g.statistics.median, 3)} · ${num(g.statistics.p95, 3)} · ${num(g.statistics.maximum, 3)}`
          } />
          <DataRow label="Std. deviation" value={num(g.statistics.std, 4)} />
        </div>

        <p className="text-[10px] leading-relaxed text-muted-foreground">
          Method: {humanise(g.method)}. Levels 1 and 2 are aggregations of the same
          Level-3 field, so the hierarchy is consistent by construction.
        </p>
      </CardContent>
    </Card>
  );
}

export function CalibrationPanel({ calibration }: { calibration: CalibrationResult | null }) {
  if (!calibration) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-1.5">
            <GaugeCircle className="size-4" />
            Calibration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">
            Calibration was not applied to this analysis.
          </p>
        </CardContent>
      </Card>
    );
  }

  const isTemperature = calibration.method === 'temperature_scaling';

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between">
        <div>
          <CardTitle className="flex items-center gap-1.5">
            <GaugeCircle className="size-4" />
            Calibration
          </CardTitle>
          <p className="mt-0.5 text-[11px] text-muted-foreground">
            {calibration.method_name}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <StatusBadge status={calibration.status} />
          <ProvenanceBadge provenance={calibration.provenance} />
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="divide-y divide-border/60">
          <DataRow label="Target risk α" value={num(calibration.target_risk, 3)} />
          <DataRow
            label={isTemperature ? 'Fitted temperature T' : 'Threshold λ̂'}
            value={num(calibration.threshold, 4)}
            hint={
              isTemperature
                ? 'Scalar temperature fitted on the calibration set.'
                : 'Smallest λ whose finite-sample corrected risk meets α.'
            }
          />
          <DataRow
            label="Empirical risk"
            value={num(calibration.empirical_risk, 4)}
            hint="Measured on the calibration set at the selected threshold."
          />
          <DataRow label="Coverage" value={pct(calibration.coverage, 2)} />
          <DataRow
            label="Expected Calibration Error"
            value={num(calibration.expected_calibration_error, 4)}
          />
          {calibration.brier_score !== null && (
            <DataRow label="Brier score" value={num(calibration.brier_score, 4)} />
          )}
          {calibration.applied_abstention_rate !== null && (
            <DataRow
              label="Applied ambiguity band"
              value={pct(calibration.applied_abstention_rate, 2)}
              hint="Fraction of THIS image inside the calibrated prediction set but outside the argmax decision."
            />
          )}
          <DataRow
            label="Calibration set"
            value={`${calibration.calibration_set_size} samples`}
          />
        </div>

        {calibration.validity_warnings.length > 0 && (
          <div className="space-y-1.5 rounded-md border border-status-warn/30 bg-status-warn/8 p-2.5">
            {calibration.validity_warnings.map((warning) => (
              <p
                key={warning}
                className="flex gap-1.5 text-[11px] leading-relaxed text-muted-foreground"
              >
                <AlertTriangle className="mt-0.5 size-3 shrink-0 text-status-warn" />
                {warning}
              </p>
            ))}
          </div>
        )}

        <details className="group">
          <summary className="cursor-pointer text-[11px] font-medium text-muted-foreground hover:text-foreground">
            Assumptions ({calibration.assumptions.length})
          </summary>
          <ul className="mt-1.5 space-y-1">
            {calibration.assumptions.map((assumption) => (
              <li
                key={assumption}
                className="border-l-2 border-border pl-2 text-[10px] leading-relaxed text-muted-foreground"
              >
                {assumption}
              </li>
            ))}
          </ul>
        </details>
      </CardContent>
    </Card>
  );
}

export function ShiftPanel({ shift }: { shift: DistributionShiftResult | null }) {
  if (!shift) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-1.5">
            <Radar className="size-4" />
            Distribution shift
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">
            Shift detection was disabled for this analysis.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between">
        <div>
          <CardTitle className="flex items-center gap-1.5">
            <Radar className="size-4" />
            Distribution shift
          </CardTitle>
          <p className="mt-0.5 text-[11px] text-muted-foreground">
            vs {shift.reference_domain}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <StatusBadge status={shift.status_label} />
          <ShiftBadge status={shift.status} />
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <div className="flex items-baseline justify-between">
            <span className="text-xs text-muted-foreground">Composite shift score</span>
            <span className="tabular text-xl font-semibold">
              {num(shift.shift_score, 4)}
            </span>
          </div>
          <Progress value={shift.shift_score * 100} className="mt-1.5" />
          <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
            <span>ID &lt; {num(shift.thresholds.mild, 2)}</span>
            <span>mod. {num(shift.thresholds.moderate, 2)}</span>
            <span>severe ≥ {num(shift.thresholds.severe, 2)}</span>
          </div>
        </div>

        <div className="space-y-1.5">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Indicators
          </p>
          {shift.indicators.map((indicator) => (
            <div key={indicator.id} className="space-y-0.5" title={indicator.description}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="truncate text-[11px] text-muted-foreground">
                  {indicator.name}
                </span>
                <span className="tabular shrink-0 text-[11px]">
                  {num(indicator.normalised, 3)}
                  <span className="ml-1 text-muted-foreground">
                    ×{num(indicator.weight, 2)}
                  </span>
                </span>
              </div>
              <div className="h-1 overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-status-info"
                  style={{ width: `${indicator.normalised * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        <p className="text-[10px] leading-relaxed text-muted-foreground">
          Input-space intensity statistics only. Semantic shift with unchanged
          intensity statistics is not detected. Band thresholds are configuration,
          not experimental findings.
        </p>
      </CardContent>
    </Card>
  );
}

export function RiskPanel({ risk }: { risk: RiskAssessment }) {
  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between">
        <CardTitle className="flex items-center gap-1.5">
          <Shield className="size-4" />
          Risk assessment
        </CardTitle>
        <RiskBadge level={risk.risk_level} />
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-muted-foreground">Risk score</span>
          <span className="tabular text-xl font-semibold">{num(risk.risk_score, 4)}</span>
        </div>
        <Progress value={risk.risk_score * 100} />

        <div className="space-y-1.5">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Contributing factors
          </p>
          {risk.factors.map((factor) => (
            <div key={factor.id} className="space-y-0.5" title={factor.rationale}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="truncate text-[11px]">{factor.name}</span>
                <span className="tabular shrink-0 text-[11px] text-muted-foreground">
                  {num(factor.normalised, 3)} × {num(factor.weight, 2)} ={' '}
                  <strong className="text-foreground">
                    {num(factor.contribution, 3)}
                  </strong>
                </span>
              </div>
              <div className="h-1 overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary"
                  style={{ width: `${(factor.contribution / 0.4) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {risk.triggered_overrides.length > 0 && (
          <div className="space-y-1 rounded-md border border-status-danger/30 bg-status-danger/8 p-2.5">
            <p className="text-[11px] font-medium text-status-danger">
              Overrides triggered
            </p>
            {risk.triggered_overrides.map((override) => (
              <p key={override} className="font-mono text-[10px] text-muted-foreground">
                {override}
              </p>
            ))}
          </div>
        )}

        <div className="space-y-1">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Reasoning
          </p>
          <ul className="space-y-1">
            {risk.reasons.map((reason) => (
              <li
                key={reason}
                className="border-l-2 border-border pl-2 text-[11px] leading-relaxed text-muted-foreground"
              >
                {reason}
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

export function PredictionPanel({ analysis }: { analysis: AnalysisResponse }) {
  const { prediction, metadata } = analysis;
  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between">
        <div>
          <CardTitle>Prediction</CardTitle>
          <p className="mt-0.5 text-[11px] text-muted-foreground">
            {metadata.model_name}
          </p>
        </div>
        <Badge variant="outline">{prediction.task}</Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="divide-y divide-border/60">
          <DataRow label="Mean confidence" value={pct(prediction.confidence, 2)} />
          <DataRow
            label="Inference time"
            value={ms(prediction.inference_time_ms)}
          />
          <DataRow label="Pipeline duration" value={ms(metadata.duration_ms)} />
          <DataRow
            label="Working resolution"
            value={`${metadata.image.working_width}×${metadata.image.working_height}`}
            hint="All masks, boxes and uncertainty maps are in these coordinates."
          />
          <DataRow
            label="Source"
            value={`${metadata.image.original_width}×${metadata.image.original_height} ${metadata.image.format}`}
          />
        </div>

        {prediction.segmentation && (
          <div className="space-y-1.5">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Class composition
            </p>
            {prediction.segmentation.class_labels.map((label, index) => (
              <div key={label} className="flex items-center gap-2">
                <span className="w-40 shrink-0 truncate text-[11px] text-muted-foreground">
                  {label}
                </span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-primary/70"
                    style={{
                      width: `${(prediction.segmentation!.class_pixel_fraction[index] ?? 0) * 100}%`,
                    }}
                  />
                </div>
                <span className="tabular w-12 shrink-0 text-right text-[11px]">
                  {pct(prediction.segmentation!.class_pixel_fraction[index] ?? 0, 1)}
                </span>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
          <Timer className="size-3" />
          <span className="font-mono">{metadata.pipeline_version}</span>
          <span>· seed {metadata.seed}</span>
        </div>
      </CardContent>
    </Card>
  );
}
