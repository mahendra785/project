'use client';

/**
 * Analysis configuration.
 *
 * Everything offered here is gated on what the backend actually reports:
 * tasks come from the selected model's declared `tasks`, calibration methods
 * are filtered to those whose status is implemented or baseline, and
 * unavailable adapters are disabled with their reason shown. The UI therefore
 * cannot advertise a capability the backend does not have.
 */

import { AlertTriangle, Info } from 'lucide-react';

import {
  StatusBadge,
} from '@/components/research/status-badges';
import { Badge } from '@/components/ui/badge';
import {
  Label,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Separator,
  Slider,
  Switch,
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/primitives';
import { num } from '@/lib/utils';
import type { AnalysisConfig } from '@/stores/analysis-store';
import type {
  CalibrationMethodsResponse,
  ModelListResponse,
  TaskType,
} from '@/types/api';

const TASK_LABELS: Record<TaskType, string> = {
  segmentation: 'Segmentation',
  detection: 'Detection',
  classification: 'Classification',
};

export function AnalysisConfigForm({
  config,
  onChange,
  models,
  calibration,
  disabled,
}: {
  config: AnalysisConfig;
  onChange: (patch: Partial<AnalysisConfig>) => void;
  models?: ModelListResponse;
  calibration?: CalibrationMethodsResponse;
  disabled?: boolean;
}) {
  const activeModelId = config.modelId ?? models?.default_model_id ?? '';
  const activeModel = models?.models.find((m) => m.id === activeModelId);
  const supportedTasks = activeModel?.tasks ?? ['segmentation'];

  // Only methods the backend has actually implemented can be selected.
  const runnableMethods =
    calibration?.methods.filter(
      (m) => m.status === 'implemented' || m.status === 'baseline',
    ) ?? [];
  const plannedMethods =
    calibration?.methods.filter((m) => m.status === 'planned') ?? [];
  const selectedMethod = calibration?.methods.find(
    (m) => m.id === config.calibrationMethod,
  );

  return (
    <div className="space-y-4">
      {/* ------------------------------------------------------------ model */}
      <div className="space-y-1.5">
        <Label htmlFor="model">Model adapter</Label>
        <Select
          value={activeModelId}
          onValueChange={(value) => {
            const model = models?.models.find((m) => m.id === value);
            const nextTask: TaskType = model?.tasks.includes(config.task)
              ? config.task
              : (model?.tasks[0] ?? 'segmentation');
            onChange({ modelId: value, task: nextTask });
          }}
          disabled={disabled || !models}
        >
          <SelectTrigger id="model">
            <SelectValue placeholder="Select a model" />
          </SelectTrigger>
          <SelectContent>
            {models?.models.map((model) => (
              <SelectItem key={model.id} value={model.id} disabled={!model.available}>
                <span className="flex items-center gap-2">
                  {model.name}
                  {!model.available && (
                    <span className="text-[10px] text-muted-foreground">
                      (unavailable)
                    </span>
                  )}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {activeModel && (
          <div className="space-y-1.5 rounded-md border border-border bg-muted/40 p-2.5">
            <div className="flex flex-wrap items-center gap-1.5">
              <StatusBadge status={activeModel.status} />
              <Badge variant="outline">{activeModel.model_type}</Badge>
              {!activeModel.is_medical && (
                <Badge variant="warn">
                  <AlertTriangle className="size-3" />
                  Not a medical model
                </Badge>
              )}
            </div>
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              {activeModel.description}
            </p>
            <p className="text-[10px] leading-relaxed text-muted-foreground">
              <span className="font-medium">Trained on:</span> {activeModel.trained_on}
            </p>
            {!activeModel.available && activeModel.unavailable_reason && (
              <p className="text-[10px] text-status-danger">
                {activeModel.unavailable_reason}
              </p>
            )}
          </div>
        )}
      </div>

      {/* ------------------------------------------------------------- task */}
      <div className="space-y-1.5">
        <Label htmlFor="task">Task</Label>
        <Select
          value={config.task}
          onValueChange={(value) => onChange({ task: value as TaskType })}
          disabled={disabled}
        >
          <SelectTrigger id="task">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {(['segmentation', 'detection', 'classification'] as TaskType[]).map((task) => {
              const supported = supportedTasks.includes(task);
              return (
                <SelectItem key={task} value={task} disabled={!supported}>
                  <span className="flex items-center gap-2">
                    {TASK_LABELS[task]}
                    {!supported && (
                      <span className="text-[10px] text-muted-foreground">
                        (adapter not configured)
                      </span>
                    )}
                  </span>
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
        {!supportedTasks.includes('classification') && (
          <p className="text-[10px] leading-relaxed text-muted-foreground">
            Classification is disabled: no adapter implements it. An image-level
            label from an unsupervised segmenter would be a fabricated diagnosis.
          </p>
        )}
      </div>

      <Separator />

      {/* ------------------------------------------------------ calibration */}
      <div className="space-y-2.5">
        <div className="flex items-center justify-between">
          <Label htmlFor="apply-calibration">Apply calibration</Label>
          <Switch
            id="apply-calibration"
            checked={config.applyCalibration}
            onCheckedChange={(v) => onChange({ applyCalibration: v })}
            disabled={disabled}
          />
        </div>

        {config.applyCalibration && (
          <>
            <Select
              value={config.calibrationMethod}
              onValueChange={(value) => onChange({ calibrationMethod: value })}
              disabled={disabled}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {runnableMethods.map((method) => (
                  <SelectItem key={method.id} value={method.id}>
                    {method.name}
                  </SelectItem>
                ))}
                {plannedMethods.map((method) => (
                  <SelectItem key={method.id} value={method.id} disabled>
                    <span className="flex items-center gap-2">
                      {method.name}
                      <span className="text-[10px] text-muted-foreground">(planned)</span>
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {selectedMethod && (
              <div className="space-y-1 rounded-md border border-border bg-muted/40 p-2.5">
                <StatusBadge status={selectedMethod.status} />
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  {selectedMethod.description}
                </p>
                {selectedMethod.reference && (
                  <p className="text-[10px] italic text-muted-foreground">
                    {selectedMethod.reference}
                  </p>
                )}
              </div>
            )}

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label className="flex items-center gap-1">
                  Target risk α
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="size-3 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      The expected false-negative rate the conformal procedure is
                      asked to control. Smaller α forces a more inclusive (more
                      conservative) prediction set.
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <span className="tabular text-[11px] text-muted-foreground">
                  {num(config.targetRisk, 2)}
                </span>
              </div>
              <Slider
                value={[config.targetRisk]}
                min={0.01}
                max={0.5}
                step={0.01}
                onValueChange={([v]) => onChange({ targetRisk: v })}
                disabled={disabled}
              />
            </div>
          </>
        )}
      </div>

      <Separator />

      {/* ------------------------------------------------------------ shift */}
      <div className="space-y-2.5">
        <div className="flex items-center justify-between">
          <Label htmlFor="detect-shift">Detect distribution shift</Label>
          <Switch
            id="detect-shift"
            checked={config.detectShift}
            onCheckedChange={(v) => onChange({ detectShift: v })}
            disabled={disabled}
          />
        </div>
        {config.detectShift && (
          <p className="text-[10px] leading-relaxed text-muted-foreground">
            Compared against reference domain{' '}
            <code className="text-foreground">{config.referenceDomain}</code>, which is
            synthetic — &ldquo;in-distribution&rdquo; here means phantom-like.
          </p>
        )}
      </div>

      <Separator />

      {/* ------------------------------------------------------ uncertainty */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <Label>High-uncertainty threshold</Label>
          <span className="tabular text-[11px] text-muted-foreground">
            {num(config.uncertaintyThreshold, 2)}
          </span>
        </div>
        <Slider
          value={[config.uncertaintyThreshold]}
          min={0.1}
          max={0.95}
          step={0.01}
          onValueChange={([v]) => onChange({ uncertaintyThreshold: v })}
          disabled={disabled}
        />
        <p className="text-[10px] leading-relaxed text-muted-foreground">
          Pixels above this are counted in the high-uncertainty mask and seed the
          Level-2 regions.
        </p>
      </div>
    </div>
  );
}
