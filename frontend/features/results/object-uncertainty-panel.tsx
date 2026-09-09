'use client';

/**
 * Level-2 (object / region) uncertainty table.
 *
 * When the analysis ran a detection task, real detections appear. Otherwise the
 * rows are connected components of the high-uncertainty mask — genuinely
 * computed, and labelled as such via the `source` column so they are never
 * mistaken for model detections. If a detection task was requested but the
 * adapter has no detection head, the caller renders the explicit
 * "Detection adapter not configured" state below.
 */

import { Boxes, ScanEye } from 'lucide-react';

import { RiskBadge } from '@/components/research/status-badges';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/primitives';
import { cn, num } from '@/lib/utils';
import type { ObjectUncertainty, TaskType } from '@/types/api';

export function DetectionNotConfigured({ task }: { task: TaskType }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-md border border-dashed border-border px-4 py-8 text-center">
      <Boxes className="size-5 text-muted-foreground" />
      <p className="text-sm font-medium">Detection adapter not configured</p>
      <p className="max-w-md text-xs leading-relaxed text-muted-foreground">
        The selected model exposes no detection head, so no object detections
        were produced for this {task} run. Select the
        <code className="mx-1">region-proposal-det-v1</code> adapter to get
        segmentation-derived proposals, or register a trained detector.
      </p>
    </div>
  );
}

export function ObjectUncertaintyPanel({
  objects,
  selectedId,
  onSelect,
  className,
}: {
  objects: ObjectUncertainty[];
  selectedId?: string | null;
  onSelect?: (id: string | null) => void;
  className?: string;
}) {
  return (
    <Card className={className}>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-1.5">
          <ScanEye className="size-4" />
          Level 2 · Object &amp; region uncertainty
        </CardTitle>
        <span className="text-[11px] text-muted-foreground">
          {objects.length} region{objects.length === 1 ? '' : 's'}
        </span>
      </CardHeader>
      <CardContent className="p-0">
        {objects.length === 0 ? (
          <p className="px-4 py-8 text-center text-xs text-muted-foreground">
            No connected region exceeded the high-uncertainty threshold. Lower the
            threshold to surface smaller ambiguous areas.
          </p>
        ) : (
          <div className="max-h-[420px] overflow-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 border-b border-border bg-card text-[10px] uppercase tracking-wider text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold">ID</th>
                  <th className="px-3 py-2 text-left font-semibold">Class</th>
                  <th className="px-2 py-2 text-right font-semibold">Conf.</th>
                  <th className="px-2 py-2 text-right font-semibold">Unc.</th>
                  <th className="px-2 py-2 text-right font-semibold">Peak</th>
                  <th className="px-2 py-2 text-right font-semibold">Risk</th>
                  <th className="px-2 py-2 text-left font-semibold">BBox</th>
                  <th className="px-3 py-2 text-left font-semibold">Level</th>
                  <th className="px-3 py-2 text-left font-semibold">Review</th>
                </tr>
              </thead>
              <tbody>
                {objects.map((object) => {
                  const selected = object.id === selectedId;
                  return (
                    <tr
                      key={object.id}
                      onClick={() => onSelect?.(selected ? null : object.id)}
                      className={cn(
                        'cursor-pointer border-b border-border/60 last:border-0 transition-colors',
                        selected ? 'bg-primary/8' : 'hover:bg-accent/40',
                      )}
                    >
                      <td className="px-3 py-1.5 font-mono text-[11px]">{object.id}</td>
                      <td className="max-w-[190px] truncate px-3 py-1.5">{object.label}</td>
                      <td className="tabular px-2 py-1.5 text-right">
                        {num(object.confidence, 3)}
                      </td>
                      <td className="tabular px-2 py-1.5 text-right font-medium">
                        {num(object.uncertainty, 3)}
                      </td>
                      <td className="tabular px-2 py-1.5 text-right text-muted-foreground">
                        {num(object.peak_uncertainty, 3)}
                      </td>
                      <td className="tabular px-2 py-1.5 text-right">
                        {object.calibrated_risk === null ? (
                          <span className="text-muted-foreground">—</span>
                        ) : (
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className="cursor-help underline decoration-dotted">
                                {num(object.calibrated_risk, 3)}
                              </span>
                            </TooltipTrigger>
                            <TooltipContent>
                              Calibrated empirical risk combined with this region&apos;s
                              uncertainty. Not a region-conditional conformal
                              guarantee — localized risk control is PLANNED.
                            </TooltipContent>
                          </Tooltip>
                        )}
                      </td>
                      <td className="tabular px-2 py-1.5 font-mono text-[10px] text-muted-foreground">
                        {object.bbox.x},{object.bbox.y} · {object.bbox.width}×
                        {object.bbox.height}
                      </td>
                      <td className="px-3 py-1.5">
                        <RiskBadge level={object.risk_level} />
                      </td>
                      <td className="px-3 py-1.5">
                        <Badge variant={object.source === 'detection' ? 'info' : 'outline'}>
                          {object.source === 'detection' ? 'detection' : 'region'}
                        </Badge>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
