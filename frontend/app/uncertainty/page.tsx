'use client';

import { Suspense, useEffect, useMemo } from 'react';
import { useSearchParams } from 'next/navigation';
import { BrainCircuit, ScanSearch } from 'lucide-react';

import { Area, AreaChart, Cell, Bar, BarChart } from 'recharts';

import {
  ChartFrame,
  ChartGrid,
  ChartTooltip,
  XAxis,
  YAxis,
} from '@/components/charts/chart-primitives';
import { PageContainer, PageHeader } from '@/components/layout/page-container';
import { EmptyState, ErrorState, LoadingState } from '@/components/layout/states';
import { ProvenanceBadge, StatusBadge } from '@/components/research/status-badges';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/primitives';
import { ImageViewer } from '@/components/viewer/image-viewer';
import { LayerControls } from '@/components/viewer/layer-controls';
import { UncertaintyLegend } from '@/components/viewer/uncertainty-legend';
import { AnalysisPicker } from '@/features/results/analysis-picker';
import { ObjectUncertaintyPanel } from '@/features/results/object-uncertainty-panel';
import { GlobalUncertaintyPanel } from '@/features/results/summary-panels';
import { useAnalyses, useAnalysis } from '@/hooks/use-pums';
import {
  matrixHistogram,
  matrixStats,
  uncertaintyColour,
  useUncertaintyMatrix,
} from '@/hooks/use-uncertainty-matrix';
import { mediaUrl } from '@/lib/api/client';
import { num, pct } from '@/lib/utils';
import { useAnalysisStore } from '@/stores/analysis-store';

/**
 * Histogram computed in the browser from the decoded matrix.
 *
 * This is deliberately not a value the backend sends: computing it client-side
 * from the decoded PNG proves the transported map really is numeric data and
 * keeps the threshold slider instantaneous.
 */
function LiveHistogram({ url, threshold }: { url: string | null; threshold: number }) {
  const { data: matrix, isLoading } = useUncertaintyMatrix(url);

  const data = useMemo(() => {
    if (!matrix) return [];
    const counts = matrixHistogram(matrix, 20);
    return counts.map((count, index) => {
      const centre = (index + 0.5) / counts.length;
      const [r, g, b] = uncertaintyColour(centre);
      return {
        bin: (index / counts.length).toFixed(2),
        centre,
        count,
        fill: `rgb(${r}, ${g}, ${b})`,
      };
    });
  }, [matrix]);

  const stats = matrix ? matrixStats(matrix, threshold) : null;

  if (isLoading) return <LoadingState label="Decoding uncertainty map…" />;
  if (!matrix) return null;

  return (
    <div className="space-y-2">
      <ChartFrame height={180}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -18 }}>
          <ChartGrid />
          <XAxis dataKey="bin" stroke="var(--muted-foreground)" fontSize={9} tickLine={false} interval={3} />
          <YAxis stroke="var(--muted-foreground)" fontSize={9} tickLine={false} />
          <ChartTooltip
            formatter={(value) => [`${value} px`, 'Pixels']}
            labelFormatter={(label) => `Uncertainty ≥ ${label}`}
          />
          <Bar dataKey="count">
            {data.map((entry) => (
              <Cell
                key={entry.bin}
                fill={entry.fill}
                fillOpacity={entry.centre >= threshold ? 1 : 0.35}
              />
            ))}
          </Bar>
        </BarChart>
      </ChartFrame>
      {stats && (
        <div className="grid grid-cols-3 gap-2 text-center">
          {[
            ['Mean', num(stats.mean, 4)],
            ['Max', num(stats.max, 4)],
            ['Above threshold', pct(stats.aboveFraction, 2)],
          ].map(([label, value]) => (
            <div key={label} className="rounded-md border border-border bg-muted/40 p-2">
              <p className="text-[10px] text-muted-foreground">{label}</p>
              <p className="tabular text-sm font-semibold">{value}</p>
            </div>
          ))}
        </div>
      )}
      <p className="text-[10px] leading-relaxed text-muted-foreground">
        Computed in the browser from {stats?.count.toLocaleString()} decoded pixel
        values — the transported PNG is real numeric data, not a rendered picture.
      </p>
    </div>
  );
}

function PreviewProfile({ grid }: { grid: number[][] }) {
  // Row-mean profile of the coarse preview grid: a cheap read on where
  // uncertainty concentrates vertically without decoding the full map.
  const data = grid.map((row, index) => ({
    row: index,
    mean: row.reduce((a, b) => a + b, 0) / (row.length || 1),
    max: Math.max(...row),
  }));

  return (
    <ChartFrame height={170}>
      <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -22 }}>
        <ChartGrid />
        <XAxis dataKey="row" stroke="var(--muted-foreground)" fontSize={9} tickLine={false} />
        <YAxis domain={[0, 1]} stroke="var(--muted-foreground)" fontSize={9} tickLine={false} />
        <ChartTooltip />
        <Area
          type="monotone"
          dataKey="max"
          stroke="var(--status-warn)"
          fill="var(--status-warn)"
          fillOpacity={0.12}
          strokeWidth={1}
        />
        <Area
          type="monotone"
          dataKey="mean"
          stroke="var(--primary)"
          fill="var(--primary)"
          fillOpacity={0.2}
          strokeWidth={1.5}
        />
      </AreaChart>
    </ChartFrame>
  );
}

function UncertaintyContent() {
  const searchParams = useSearchParams();
  const idFromUrl = searchParams.get('id');

  const selectedId = useAnalysisStore((s) => s.selectedId);
  const setSelectedId = useAnalysisStore((s) => s.setSelectedId);
  const viewer = useAnalysisStore((s) => s.viewer);
  const updateViewer = useAnalysisStore((s) => s.updateViewer);
  const resetViewer = useAnalysisStore((s) => s.resetViewer);
  const selectedObjectId = useAnalysisStore((s) => s.selectedObjectId);
  const setSelectedObject = useAnalysisStore((s) => s.setSelectedObject);

  const list = useAnalyses(50);

  useEffect(() => {
    if (idFromUrl && idFromUrl !== selectedId) setSelectedId(idFromUrl);
  }, [idFromUrl, selectedId, setSelectedId]);

  // Fall back to the most recent analysis so the page is never blank when data
  // exists but nothing has been explicitly selected.
  const effectiveId =
    selectedId && list.data?.items.some((i) => i.analysis_id === selectedId)
      ? selectedId
      : (list.data?.items[0]?.analysis_id ?? null);

  const analysis = useAnalysis(effectiveId);

  if (list.isError) {
    return <ErrorState error={list.error} onRetry={() => list.refetch()} />;
  }
  if (list.isLoading) return <LoadingState label="Loading analyses…" />;

  if ((list.data?.items.length ?? 0) === 0) {
    return (
      <EmptyState
        title="No analyses to inspect"
        description="The hierarchical uncertainty viewer works on a completed analysis. Run one first — its pixel map, regions and global summary will appear here."
        actionLabel="Go to Analyze"
        actionHref="/analyze"
        icon={<ScanSearch className="size-5" />}
      />
    );
  }

  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        <AnalysisPicker
          items={list.data?.items ?? []}
          value={effectiveId}
          onChange={setSelectedId}
        />
        {analysis.data && (
          <>
            <StatusBadge status={analysis.data.uncertainty.status} />
            <ProvenanceBadge provenance={analysis.data.uncertainty.global.provenance} />
          </>
        )}
      </div>

      {analysis.isLoading && <LoadingState label="Loading analysis…" />}
      {analysis.isError && (
        <ErrorState error={analysis.error} onRetry={() => analysis.refetch()} />
      )}

      {analysis.data && (
        <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
          <div className="space-y-4">
            <Card>
              <CardHeader className="flex-row items-center justify-between">
                <CardTitle className="flex items-center gap-1.5">
                  <BrainCircuit className="size-4" />
                  Level 3 · Pixel uncertainty
                </CardTitle>
                <span className="text-[11px] text-muted-foreground">
                  {analysis.data.uncertainty.pixel.width}×
                  {analysis.data.uncertainty.pixel.height} ·{' '}
                  {analysis.data.uncertainty.pixel.encoding}
                </span>
              </CardHeader>
              <CardContent className="space-y-3">
                <ImageViewer
                  sourceUrl={mediaUrl(analysis.data.metadata.image.preview_url)}
                  maskUrl={mediaUrl(analysis.data.prediction.segmentation?.png_url)}
                  uncertaintyUrl={mediaUrl(analysis.data.uncertainty.pixel.raw_url)}
                  objects={analysis.data.uncertainty.objects}
                  settings={viewer}
                  onSettingsChange={updateViewer}
                  selectedObjectId={selectedObjectId}
                  onSelectObject={setSelectedObject}
                  width={analysis.data.uncertainty.pixel.width}
                  height={analysis.data.uncertainty.pixel.height}
                />
                <Separator />
                <UncertaintyLegend threshold={viewer.uncertaintyThreshold} />
              </CardContent>
            </Card>

            <div className="grid gap-4 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Pixel value distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <LiveHistogram
                    url={mediaUrl(analysis.data.uncertainty.pixel.raw_url)}
                    threshold={viewer.uncertaintyThreshold}
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Vertical uncertainty profile</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <PreviewProfile grid={analysis.data.uncertainty.pixel.preview_grid} />
                  <p className="text-[10px] leading-relaxed text-muted-foreground">
                    Row-wise mean (blue) and max (amber) of the 32×32 preview grid
                    the backend sends alongside the full map.
                  </p>
                </CardContent>
              </Card>
            </div>

            <ObjectUncertaintyPanel
              objects={analysis.data.uncertainty.objects}
              selectedId={selectedObjectId}
              onSelect={setSelectedObject}
            />
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Viewer layers</CardTitle>
              </CardHeader>
              <CardContent>
                <LayerControls
                  settings={viewer}
                  onChange={updateViewer}
                  onReset={resetViewer}
                  hasMask={Boolean(analysis.data.prediction.segmentation?.png_url)}
                  hasRegions={analysis.data.uncertainty.objects.length > 0}
                />
              </CardContent>
            </Card>

            <GlobalUncertaintyPanel uncertainty={analysis.data.uncertainty} />

            <Card>
              <CardHeader>
                <CardTitle>Method notes</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-1.5">
                  {analysis.data.uncertainty.notes.map((note) => (
                    <li
                      key={note}
                      className="border-l-2 border-border pl-2 text-[11px] leading-relaxed text-muted-foreground"
                    >
                      {note}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </>
  );
}

export default function UncertaintyPage() {
  return (
    <PageContainer>
      <PageHeader
        title="Hierarchical uncertainty"
        description="One dense field, three levels. Level 3 is the per-pixel normalised predictive entropy; Level 2 aggregates it over connected regions; Level 1 aggregates it over the whole image. Because the levels are aggregations of the same field, they are consistent by construction."
      />
      <Suspense fallback={<LoadingState />}>
        <UncertaintyContent />
      </Suspense>
    </PageContainer>
  );
}
