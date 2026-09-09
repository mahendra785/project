'use client';

import { useState } from 'react';
import Link from 'next/link';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Loader2,
  Play,
  RotateCcw,
} from 'lucide-react';

import { PageContainer, PageHeader } from '@/components/layout/page-container';
import { ErrorState } from '@/components/layout/states';
import { ResearchDisclaimer } from '@/components/research/status-badges';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress, Separator } from '@/components/ui/primitives';
import { ImageViewer } from '@/components/viewer/image-viewer';
import { LayerControls } from '@/components/viewer/layer-controls';
import { UncertaintyLegend } from '@/components/viewer/uncertainty-legend';
import { AnalysisConfigForm } from '@/features/analyze/analysis-config-form';
import { UploadDropzone } from '@/features/analyze/upload-dropzone';
import {
  DetectionNotConfigured,
  ObjectUncertaintyPanel,
} from '@/features/results/object-uncertainty-panel';
import {
  CalibrationPanel,
  GlobalUncertaintyPanel,
  PredictionPanel,
  RiskPanel,
  ShiftPanel,
} from '@/features/results/summary-panels';
import { useCalibrationMethods, useModels, useRunAnalysis } from '@/hooks/use-pums';
import { mediaUrl } from '@/lib/api/client';
import { useAnalysisStore } from '@/stores/analysis-store';

const PIPELINE_STAGES = [
  'Preprocessing',
  'Model inference',
  'Pixel uncertainty',
  'Hierarchy',
  'Calibration',
  'Shift detection',
  'Risk engine',
];

function RunningState() {
  return (
    <Card>
      <CardContent className="space-y-4 px-6 py-10">
        <div className="flex items-center gap-3">
          <Loader2 className="size-5 animate-spin text-primary" />
          <div>
            <p className="text-sm font-medium">Running the PUMS pipeline…</p>
            <p className="text-xs text-muted-foreground">
              The first run after backend startup also builds the synthetic
              calibration set and can take up to a minute.
            </p>
          </div>
        </div>
        <Progress value={65} />
        <div className="flex flex-wrap gap-1.5">
          {PIPELINE_STAGES.map((stage) => (
            <span
              key={stage}
              className="rounded border border-border bg-muted/50 px-2 py-0.5 text-[10px] text-muted-foreground"
            >
              {stage}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);

  const models = useModels();
  const calibrationMethods = useCalibrationMethods();
  const runAnalysis = useRunAnalysis();

  const config = useAnalysisStore((s) => s.config);
  const updateConfig = useAnalysisStore((s) => s.updateConfig);
  const viewer = useAnalysisStore((s) => s.viewer);
  const updateViewer = useAnalysisStore((s) => s.updateViewer);
  const resetViewer = useAnalysisStore((s) => s.resetViewer);
  const selectedObjectId = useAnalysisStore((s) => s.selectedObjectId);
  const setSelectedObject = useAnalysisStore((s) => s.setSelectedObject);

  const analysis = runAnalysis.data ?? null;

  const handleRun = () => {
    if (!file) return;
    runAnalysis.mutate({
      file,
      options: {
        model_id: config.modelId,
        task: config.task,
        calibration_method: config.calibrationMethod,
        target_risk: config.targetRisk,
        uncertainty_threshold: config.uncertaintyThreshold,
        reference_domain: config.referenceDomain,
        apply_calibration: config.applyCalibration,
        detect_shift: config.detectShift,
      },
    });
  };

  const detectionRequested = analysis?.metadata.task === 'detection';
  const hasDetections = (analysis?.prediction.detections?.length ?? 0) > 0;

  return (
    <PageContainer>
      <PageHeader
        title="Analyze"
        description="Upload an image and run the complete pipeline: preprocessing → model → pixel uncertainty → hierarchical aggregation → calibration → distribution shift → risk assessment."
        actions={
          analysis && (
            <Button asChild variant="outline" size="sm">
              <Link href={`/uncertainty?id=${analysis.analysis_id}`}>
                Open in uncertainty viewer
                <ArrowRight className="size-3.5" />
              </Link>
            </Button>
          )
        }
      />

      <div className="grid gap-4 lg:grid-cols-[340px_1fr]">
        {/* ------------------------------------------------------- controls */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>1 · Image</CardTitle>
            </CardHeader>
            <CardContent>
              <UploadDropzone
                file={file}
                onFileChange={(next) => {
                  setFile(next);
                  runAnalysis.reset();
                }}
                disabled={runAnalysis.isPending}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>2 · Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <AnalysisConfigForm
                config={config}
                onChange={updateConfig}
                models={models.data}
                calibration={calibrationMethods.data}
                disabled={runAnalysis.isPending}
              />
            </CardContent>
          </Card>

          <div className="flex gap-2">
            <Button
              className="flex-1"
              onClick={handleRun}
              disabled={!file || runAnalysis.isPending}
            >
              {runAnalysis.isPending ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  Analysing…
                </>
              ) : (
                <>
                  <Play className="size-4" />
                  Run analysis
                </>
              )}
            </Button>
            {analysis && (
              <Button
                variant="outline"
                size="icon"
                aria-label="Clear result"
                onClick={() => {
                  runAnalysis.reset();
                  setFile(null);
                }}
              >
                <RotateCcw className="size-4" />
              </Button>
            )}
          </div>

          {analysis && (
            <Card>
              <CardHeader>
                <CardTitle>Viewer layers</CardTitle>
              </CardHeader>
              <CardContent>
                <LayerControls
                  settings={viewer}
                  onChange={updateViewer}
                  onReset={resetViewer}
                  hasMask={Boolean(analysis.prediction.segmentation?.png_url)}
                  hasRegions={analysis.uncertainty.objects.length > 0}
                />
              </CardContent>
            </Card>
          )}
        </div>

        {/* -------------------------------------------------------- results */}
        <div className="space-y-4">
          {runAnalysis.isPending && <RunningState />}

          {runAnalysis.isError && (
            <ErrorState error={runAnalysis.error} onRetry={handleRun} />
          )}

          {!analysis && !runAnalysis.isPending && !runAnalysis.isError && (
            <Card className="border-dashed">
              <CardContent className="flex min-h-[420px] flex-col items-center justify-center gap-3 text-center">
                <p className="text-sm font-medium">No analysis yet</p>
                <p className="max-w-md text-xs leading-relaxed text-muted-foreground">
                  Select an image and press <strong>Run analysis</strong>. Results
                  shown here are computed from your image by the Python backend —
                  this page never falls back to mock data.
                </p>
                <div className="mt-2 flex flex-wrap justify-center gap-1.5">
                  {PIPELINE_STAGES.map((stage) => (
                    <span
                      key={stage}
                      className="rounded border border-border px-2 py-0.5 text-[10px] text-muted-foreground"
                    >
                      {stage}
                    </span>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {analysis && (
            <>
              <Card>
                <CardHeader className="flex-row items-center justify-between">
                  <CardTitle className="flex items-center gap-1.5">
                    <CheckCircle2 className="size-4 text-status-ok" />
                    {analysis.metadata.image.filename}
                  </CardTitle>
                  <span className="font-mono text-[11px] text-muted-foreground">
                    {analysis.analysis_id}
                  </span>
                </CardHeader>
                <CardContent className="space-y-3">
                  <ImageViewer
                    sourceUrl={mediaUrl(analysis.metadata.image.preview_url)}
                    maskUrl={mediaUrl(analysis.prediction.segmentation?.png_url)}
                    uncertaintyUrl={mediaUrl(analysis.uncertainty.pixel.raw_url)}
                    objects={analysis.uncertainty.objects}
                    settings={viewer}
                    onSettingsChange={updateViewer}
                    selectedObjectId={selectedObjectId}
                    onSelectObject={setSelectedObject}
                    width={analysis.uncertainty.pixel.width}
                    height={analysis.uncertainty.pixel.height}
                  />
                  <Separator />
                  <UncertaintyLegend threshold={viewer.uncertaintyThreshold} />
                </CardContent>
              </Card>

              {analysis.warnings.length > 0 && (
                <div className="space-y-1.5 rounded-md border border-status-warn/30 bg-status-warn/8 p-3">
                  {analysis.warnings.map((warning) => (
                    <p
                      key={warning}
                      className="flex gap-2 text-[11px] leading-relaxed text-muted-foreground"
                    >
                      <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-status-warn" />
                      {warning}
                    </p>
                  ))}
                </div>
              )}

              <div className="grid gap-4 xl:grid-cols-2">
                <PredictionPanel analysis={analysis} />
                <GlobalUncertaintyPanel uncertainty={analysis.uncertainty} />
                <CalibrationPanel calibration={analysis.calibration} />
                <ShiftPanel shift={analysis.distribution_shift} />
              </div>

              {detectionRequested && !hasDetections ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Level 2 · Objects</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <DetectionNotConfigured task={analysis.metadata.task} />
                  </CardContent>
                </Card>
              ) : (
                <ObjectUncertaintyPanel
                  objects={analysis.uncertainty.objects}
                  selectedId={selectedObjectId}
                  onSelect={setSelectedObject}
                />
              )}

              <RiskPanel risk={analysis.risk_assessment} />
              <ResearchDisclaimer text={analysis.risk_assessment.disclaimer} />
            </>
          )}
        </div>
      </div>
    </PageContainer>
  );
}
