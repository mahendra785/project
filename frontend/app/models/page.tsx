'use client';

import { AlertTriangle, Boxes, CheckCircle2, Cpu, XCircle } from 'lucide-react';

import { PageContainer, PageHeader } from '@/components/layout/page-container';
import { ErrorState, LoadingState } from '@/components/layout/states';
import { StatusBadge } from '@/components/research/status-badges';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useHealth, useModels } from '@/hooks/use-pums';
import { humanise } from '@/lib/utils';

/** Adapter families the abstraction is designed to accept. */
const ROADMAP = [
  {
    family: 'CNN segmentation',
    status: 'Partially available',
    note: 'DeepLabV3-ResNet50 adapter ships and runs when torch/torchvision are installed. A medical CNN (U-Net / nnU-Net) drops into the same interface.',
  },
  {
    family: 'Transformer segmentation',
    status: 'Planned',
    note: 'SegFormer / Swin-UNETR style adapters need only implement predict_segmentation returning (C, H, W) probabilities.',
  },
  {
    family: 'Object detector',
    status: 'Baseline only',
    note: 'Segmentation-derived region proposals ship today. A trained detector (Faster R-CNN, DETR, nnDetection) replaces it via predict_detection.',
  },
  {
    family: 'SAM / SAM2 foundation models',
    status: 'Planned',
    note: 'Foundation-model uncertainty is explicitly out of scope for this phase; the model-agnostic interface is the groundwork for it.',
  },
  {
    family: 'Custom medical vision models',
    status: 'Supported by design',
    note: 'Subclass BaseMedicalModel, register a factory, and the uncertainty, calibration, shift and risk layers work unchanged.',
  },
];

export default function ModelsPage() {
  const models = useModels();
  const health = useHealth();

  return (
    <PageContainer>
      <PageHeader
        title="Model registry"
        description="Every adapter declares what it is, what it was trained on, and which research components it is wired to. No adapter shipped in this phase is a validated medical model, and each says so."
        badges={
          models.data && (
            <Badge variant="outline">
              default: {models.data.default_model_id}
            </Badge>
          )
        }
      />

      {models.isLoading && <LoadingState label="Loading model registry…" />}
      {models.isError && <ErrorState error={models.error} onRetry={() => models.refetch()} />}

      {health.data && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-1.5">
              <Cpu className="size-4" />
              Runtime capabilities
            </CardTitle>
            <p className="text-[11px] text-muted-foreground">
              What is actually installed in the backend environment. Optional
              adapters are reported unavailable rather than failing at request time.
            </p>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {Object.entries(health.data.capabilities).map(([name, present]) => (
              <span
                key={name}
                className="flex items-center gap-1.5 rounded border border-border px-2 py-1 text-[11px]"
              >
                {present ? (
                  <CheckCircle2 className="size-3.5 text-status-ok" />
                ) : (
                  <XCircle className="size-3.5 text-muted-foreground" />
                )}
                <span className={present ? '' : 'text-muted-foreground'}>{name}</span>
              </span>
            ))}
          </CardContent>
        </Card>
      )}

      {models.data && (
        <div className="grid gap-4 xl:grid-cols-2">
          {models.data.models.map((model) => (
            <Card key={model.id} className={model.available ? '' : 'opacity-75'}>
              <CardHeader className="flex-row items-start justify-between gap-3">
                <div className="min-w-0">
                  <CardTitle className="flex items-center gap-1.5">
                    <Boxes className="size-4 shrink-0" />
                    <span className="truncate">{model.name}</span>
                  </CardTitle>
                  <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                    {model.id}
                  </p>
                </div>
                <div className="flex shrink-0 flex-col items-end gap-1">
                  <StatusBadge status={model.status} />
                  <Badge variant={model.available ? 'ok' : 'outline'}>
                    {model.available ? 'Runnable' : 'Unavailable'}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  {model.description}
                </p>

                {!model.is_medical && (
                  <div className="flex items-start gap-2 rounded-md border border-status-warn/30 bg-status-warn/8 p-2.5">
                    <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-status-warn" />
                    <div className="space-y-0.5">
                      <p className="text-[11px] font-medium">Not a validated medical model</p>
                      <p className="text-[10px] leading-relaxed text-muted-foreground">
                        Trained on: {model.trained_on}
                      </p>
                    </div>
                  </div>
                )}

                {!model.available && model.unavailable_reason && (
                  <p className="rounded-md border border-border bg-muted/40 p-2 text-[10px] text-muted-foreground">
                    {model.unavailable_reason}
                  </p>
                )}

                <table className="w-full text-xs">
                  <tbody className="divide-y divide-border/60">
                    {[
                      ['Model type', humanise(model.model_type)],
                      ['Tasks', model.tasks.map(humanise).join(', ')],
                      ['Uncertainty method', humanise(model.uncertainty_method)],
                      ['OOD detector', model.ood_detector],
                      ['Input channels', String(model.input_channels)],
                      ['Classes', String(model.class_labels.length)],
                    ].map(([label, value]) => (
                      <tr key={label}>
                        <td className="py-1.5 text-muted-foreground">{label}</td>
                        <td className="py-1.5 text-right font-medium">{value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                <div className="space-y-1">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Calibration methods wired
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {model.calibration_methods.map((method) => (
                      <Badge key={method} variant="secondary">
                        {method}
                      </Badge>
                    ))}
                  </div>
                </div>

                {model.class_labels.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      Class labels
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {model.class_labels.slice(0, 10).map((label) => (
                        <Badge key={label} variant="outline">
                          {label}
                        </Badge>
                      ))}
                      {model.class_labels.length > 10 && (
                        <Badge variant="outline">
                          +{model.class_labels.length - 10} more
                        </Badge>
                      )}
                    </div>
                  </div>
                )}

                {Object.keys(model.parameters).length > 0 && (
                  <details>
                    <summary className="cursor-pointer text-[11px] font-medium text-muted-foreground hover:text-foreground">
                      Parameters
                    </summary>
                    <pre className="mt-1.5 overflow-x-auto rounded-md border border-border bg-muted/50 p-2 font-mono text-[10px] leading-relaxed">
{JSON.stringify(model.parameters, null, 2)}
                    </pre>
                  </details>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Adapter roadmap</CardTitle>
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            The uncertainty stack consumes only a <code>(C, H, W)</code> probability
            array, which is what makes it model-agnostic. Adding a family below
            means implementing one method — nothing downstream changes.
          </p>
        </CardHeader>
        <CardContent className="p-0">
          <table className="w-full text-xs">
            <thead className="border-b border-border text-[10px] uppercase tracking-wider text-muted-foreground">
              <tr>
                <th className="px-4 py-2 text-left font-semibold">Family</th>
                <th className="px-3 py-2 text-left font-semibold">Status</th>
                <th className="px-4 py-2 text-left font-semibold">Notes</th>
              </tr>
            </thead>
            <tbody>
              {ROADMAP.map((row) => (
                <tr key={row.family} className="border-b border-border/60 last:border-0">
                  <td className="px-4 py-2 font-medium">{row.family}</td>
                  <td className="px-3 py-2">
                    <Badge
                      variant={
                        row.status === 'Planned'
                          ? 'outline'
                          : row.status === 'Supported by design'
                            ? 'ok'
                            : 'info'
                      }
                    >
                      {row.status}
                    </Badge>
                  </td>
                  <td className="px-4 py-2 leading-relaxed text-muted-foreground">
                    {row.note}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </PageContainer>
  );
}
