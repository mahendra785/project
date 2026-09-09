'use client';

/**
 * Research summary — what the project can currently claim, with its evidence.
 *
 * Verdicts come from the committed experiment registry; numbers come from the
 * raw result files each experiment names. "By rule" and "recommendation" are
 * shown side by side on purpose: when they disagree, that disagreement is the
 * finding, and hiding either would misrepresent the week.
 */

import { Beaker, Database, FlaskConical, Gavel, Target } from 'lucide-react';

import { ProvenanceBadge, StatusBadge } from '@/components/research/status-badges';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useResearchSummary } from '@/hooks/use-pums';
import { cn, num } from '@/lib/utils';
import type { Experiment, ImplementationStatus } from '@/types/api';

const VERDICT_TONE: Record<string, 'ok' | 'warn' | 'danger' | 'outline'> = {
  PROCEED: 'ok',
  INCONCLUSIVE: 'warn',
  KILL: 'danger',
  'N/A': 'outline',
};

function VerdictChip({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-center gap-1.5 text-[11px]">
      <span className="text-muted-foreground">{label}</span>
      <Badge variant={VERDICT_TONE[value] ?? 'outline'}>{value}</Badge>
    </span>
  );
}

function ExperimentCard({ experiment }: { experiment: Experiment }) {
  const disagree =
    experiment.verdict_by_rule !== experiment.recommendation && experiment.verdict_by_rule !== 'N/A';
  return (
    <div className="rounded-md border border-border bg-card p-3.5">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Week {experiment.week} · {experiment.dataset} · {experiment.role}
          </p>
          <h4 className="mt-0.5 text-sm font-semibold leading-snug">{experiment.title}</h4>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <VerdictChip label="by rule" value={experiment.verdict_by_rule} />
          <VerdictChip label="recommend" value={experiment.recommendation} />
        </div>
      </div>

      {experiment.rows.length > 0 && (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead className="text-[10px] uppercase tracking-wider text-muted-foreground">
              <tr>
                <th className="py-1 text-left font-semibold">α</th>
                <th className="py-1 text-right font-semibold">n</th>
                <th className="py-1 text-right font-semibold">λ̂ pix</th>
                <th className="py-1 text-right font-semibold">R pix</th>
                <th className="py-1 text-right font-semibold">R les</th>
                <th className="py-1 text-right font-semibold">G</th>
                <th className="py-1 text-right font-semibold">R les test</th>
                <th className="py-1 text-right font-semibold">λ̂ les</th>
                <th className="py-1 text-right font-semibold">infl</th>
              </tr>
            </thead>
            <tbody className="tabular">
              {experiment.rows.map((r) => (
                <tr key={r.alpha} className="border-t border-border/60">
                  <td className="py-1">{num(r.alpha, 2)}</td>
                  <td className="py-1 text-right text-muted-foreground">{r.n_cal}</td>
                  <td className="py-1 text-right">{num(r.lambda_pix, 3)}</td>
                  <td className="py-1 text-right">{num(r.R_pix, 3)}</td>
                  <td className="py-1 text-right">{num(r.R_les_at_lambda_pix, 3)}</td>
                  <td
                    className={cn(
                      'py-1 text-right font-semibold',
                      r.G >= 1.5 ? 'text-status-ok' : r.G > 1.1 ? 'text-status-warn' : 'text-status-danger',
                    )}
                  >
                    {num(r.G, 2)}
                    {r.G_sd !== null && <span className="font-normal text-muted-foreground"> ±{num(r.G_sd, 2)}</span>}
                  </td>
                  <td className="py-1 text-right text-muted-foreground">{num(r.R_les_test, 3)}</td>
                  <td className="py-1 text-right">{num(r.lambda_les, 3)}</td>
                  <td className="py-1 text-right text-muted-foreground">{num(r.inflation_union, 2)}×</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1 text-[10px] text-muted-foreground">
            τ = {String(experiment.protocol.tau)} · {String(experiment.protocol.connectivity)} · m ={' '}
            {String(experiment.protocol.min_lesion_px)} px · seeds {experiment.seeds.join(', ')} · G colour: red
            ≤ 1.1 (KILL band), amber (1.1, 1.5), green ≥ 1.5
          </p>
        </div>
      )}

      <div
        className={cn(
          'mt-3 rounded-md border p-2.5 text-[11px] leading-relaxed text-muted-foreground',
          disagree ? 'border-status-warn/35 bg-status-warn/8' : 'border-border bg-muted/40',
        )}
      >
        {disagree && (
          <p className="mb-1 flex items-center gap-1.5 font-medium text-foreground">
            <Gavel className="size-3.5 text-status-warn" />
            Rule and recommendation disagree — the disagreement is the finding
          </p>
        )}
        <p>{experiment.recommendation_reason}</p>
        <p className="mt-1.5">
          <span className="font-medium text-foreground">Next:</span> {experiment.next}
        </p>
        {(experiment.report || experiment.memo) && (
          <p className="mt-1.5 font-mono text-[10px]">
            {experiment.report && <span>{experiment.report}</span>}
            {experiment.report && experiment.memo && <span> · </span>}
            {experiment.memo && <span>{experiment.memo}</span>}
          </p>
        )}
      </div>
    </div>
  );
}

export function ResearchSummaryPanel({ className }: { className?: string }) {
  const summary = useResearchSummary();
  const d = summary.data;
  const report = d?.model.training_report as Record<string, number | string> | null | undefined;

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5">
          <FlaskConical className="size-4" />
          Research summary
        </CardTitle>
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          What the project can claim right now, and the evidence behind each claim.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {summary.isLoading && (
          <p className="py-6 text-center text-xs text-muted-foreground">Loading…</p>
        )}
        {summary.isError && (
          <p className="py-6 text-center text-xs text-status-danger">Could not load the research summary.</p>
        )}

        {d && (
          <>
            <div className="grid gap-3 md:grid-cols-3">
              <div className="rounded-md border border-border bg-muted/40 p-3">
                <p className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  <Database className="size-3" /> Data
                </p>
                <p className="mt-1 text-sm font-medium">{d.dataset.dataset_name ?? 'Synthetic phantoms'}</p>
                <div className="mt-1.5">
                  <ProvenanceBadge provenance={d.dataset.provenance} />
                </div>
              </div>

              <div className="rounded-md border border-border bg-muted/40 p-3">
                <p className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  <Beaker className="size-3" /> Model
                </p>
                <p className="mt-1 text-sm font-medium">{d.model.name}</p>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  <Badge variant={d.model.is_medical ? 'ok' : 'outline'}>
                    {d.model.is_medical ? 'trained on medical data' : 'not medical'}
                  </Badge>
                  <Badge variant="danger">not clinically validated</Badge>
                </div>
                {report && (
                  <p className="tabular mt-1.5 text-[10px] text-muted-foreground">
                    {String(report.n_train_cases)} cases · {String(report.n_train_slices)} slices · OOB acc{' '}
                    {num(Number(report.oob_accuracy), 3)}
                  </p>
                )}
              </div>

              <div className="rounded-md border border-border bg-muted/40 p-3">
                <p className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  <Target className="size-3" /> Calibration methods
                </p>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {(Object.entries(d.method_status_counts) as [ImplementationStatus, number][])
                    .filter(([, n]) => n > 0)
                    .map(([status, n]) => (
                      <span key={status} className="flex items-center gap-1 text-[11px]">
                        <StatusBadge status={status} />
                        <span className="tabular text-muted-foreground">×{n}</span>
                      </span>
                    ))}
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Experiments
              </p>
              {d.experiments.length === 0 && (
                <p className="text-xs text-muted-foreground">No experiments registered yet.</p>
              )}
              {d.experiments.map((e) => (
                <ExperimentCard key={e.id} experiment={e} />
              ))}
            </div>

            <p className="text-[10px] leading-relaxed text-muted-foreground">{d.disclaimer}</p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
