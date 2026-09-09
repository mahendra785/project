'use client';

/**
 * Composite image viewer: source image, mask, uncertainty heatmap and region
 * boxes, with zoom/pan, side-by-side comparison, and a live value readout.
 *
 * The readout is the important part — hovering reports the *actual decoded
 * uncertainty value* at that pixel, which is what turns the heatmap from a
 * picture into an inspectable measurement.
 */

import { useCallback, useRef, useState } from 'react';
import { Crosshair, Loader2 } from 'lucide-react';

import {
  PredictionOverlay,
  RegionOverlay,
} from '@/components/viewer/prediction-overlay';
import { UncertaintyHeatmap } from '@/components/viewer/uncertainty-heatmap';
import {
  useUncertaintyMatrix,
  type UncertaintyMatrix,
} from '@/hooks/use-uncertainty-matrix';
import { cn, num } from '@/lib/utils';
import type { ViewerSettings } from '@/stores/analysis-store';
import type { ObjectUncertainty } from '@/types/api';

interface ImageViewerProps {
  sourceUrl: string | null;
  maskUrl: string | null;
  uncertaintyUrl: string | null;
  objects: ObjectUncertainty[];
  settings: ViewerSettings;
  onSettingsChange: (patch: Partial<ViewerSettings>) => void;
  selectedObjectId?: string | null;
  onSelectObject?: (id: string | null) => void;
  width: number;
  height: number;
  className?: string;
}

interface HoverInfo {
  x: number;
  y: number;
  value: number;
}

function Stage({
  children,
  settings,
  onSettingsChange,
  onHover,
  matrix,
  width,
  height,
}: {
  children: React.ReactNode;
  settings: ViewerSettings;
  onSettingsChange: (patch: Partial<ViewerSettings>) => void;
  onHover?: (info: HoverInfo | null) => void;
  matrix?: UncertaintyMatrix;
  width: number;
  height: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef<{ x: number; y: number } | null>(null);

  const handleMove = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (dragging.current) {
        onSettingsChange({
          panX: settings.panX + (event.clientX - dragging.current.x),
          panY: settings.panY + (event.clientY - dragging.current.y),
        });
        dragging.current = { x: event.clientX, y: event.clientY };
        return;
      }
      if (!onHover || !matrix || !ref.current) return;

      const rect = ref.current.getBoundingClientRect();
      // Undo the CSS transform to recover image-space coordinates.
      const localX = (event.clientX - rect.left - settings.panX) / settings.zoom;
      const localY = (event.clientY - rect.top - settings.panY) / settings.zoom;
      const px = Math.floor((localX / rect.width) * settings.zoom * matrix.width);
      const py = Math.floor((localY / rect.height) * settings.zoom * matrix.height);

      if (px < 0 || py < 0 || px >= matrix.width || py >= matrix.height) {
        onHover(null);
        return;
      }
      onHover({ x: px, y: py, value: matrix.at(px, py) });
    },
    [matrix, onHover, onSettingsChange, settings.panX, settings.panY, settings.zoom],
  );

  return (
    <div
      ref={ref}
      className="checkerboard relative aspect-square w-full cursor-crosshair overflow-hidden rounded-md border border-border bg-muted/40"
      onMouseMove={handleMove}
      onMouseDown={(e) => {
        dragging.current = { x: e.clientX, y: e.clientY };
      }}
      onMouseUp={() => {
        dragging.current = null;
      }}
      onMouseLeave={() => {
        dragging.current = null;
        onHover?.(null);
      }}
      onWheel={(e) => {
        if (!e.ctrlKey && !e.metaKey) return;
        e.preventDefault();
        const next = settings.zoom * (e.deltaY < 0 ? 1.12 : 0.89);
        onSettingsChange({ zoom: Math.min(6, Math.max(0.5, next)) });
      }}
    >
      <div
        className="absolute inset-0 origin-top-left"
        style={{
          transform: `translate(${settings.panX}px, ${settings.panY}px) scale(${settings.zoom})`,
          width: '100%',
          height: '100%',
        }}
      >
        <div className="relative size-full" style={{ aspectRatio: `${width}/${height}` }}>
          {children}
        </div>
      </div>
    </div>
  );
}

export function ImageViewer({
  sourceUrl,
  maskUrl,
  uncertaintyUrl,
  objects,
  settings,
  onSettingsChange,
  selectedObjectId,
  onSelectObject,
  width,
  height,
  className,
}: ImageViewerProps) {
  const [hover, setHover] = useState<HoverInfo | null>(null);
  const { data: matrix, isLoading, isError } = useUncertaintyMatrix(uncertaintyUrl);

  const source = sourceUrl ? (
    // eslint-disable-next-line @next/next/no-img-element -- backend PNG served raw
    <img
      src={sourceUrl}
      alt="Source image"
      className="absolute inset-0 size-full object-contain"
      style={{ imageRendering: 'pixelated' }}
      draggable={false}
    />
  ) : null;

  const layers = (
    <>
      {settings.showOriginal && source}
      {settings.showMask && (
        <PredictionOverlay maskUrl={maskUrl} opacity={settings.maskOpacity} />
      )}
      {settings.showHeatmap && matrix && (
        <UncertaintyHeatmap
          matrix={matrix}
          threshold={settings.uncertaintyThreshold}
          opacity={settings.heatmapOpacity}
        />
      )}
      {settings.showRegions && objects.length > 0 && (
        <RegionOverlay
          objects={objects}
          width={width}
          height={height}
          selectedId={selectedObjectId}
          onSelect={onSelectObject}
        />
      )}
    </>
  );

  return (
    <div className={cn('space-y-2', className)}>
      {settings.sideBySide ? (
        <div className="grid gap-2 sm:grid-cols-2">
          <Stage
            settings={settings}
            onSettingsChange={onSettingsChange}
            width={width}
            height={height}
          >
            {source}
          </Stage>
          <Stage
            settings={settings}
            onSettingsChange={onSettingsChange}
            onHover={setHover}
            matrix={matrix}
            width={width}
            height={height}
          >
            {layers}
          </Stage>
        </div>
      ) : (
        <Stage
          settings={settings}
          onSettingsChange={onSettingsChange}
          onHover={setHover}
          matrix={matrix}
          width={width}
          height={height}
        >
          {layers}
        </Stage>
      )}

      <div className="flex min-h-5 items-center justify-between gap-3 text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          {isLoading && (
            <>
              <Loader2 className="size-3 animate-spin" />
              Decoding uncertainty map…
            </>
          )}
          {isError && (
            <span className="text-status-danger">
              Uncertainty map could not be decoded.
            </span>
          )}
          {matrix && !isLoading && (
            <>
              <Crosshair className="size-3" />
              {hover ? (
                <span className="tabular">
                  ({hover.x}, {hover.y}) · uncertainty{' '}
                  <strong className="text-foreground">{num(hover.value, 3)}</strong>
                  {hover.value >= settings.uncertaintyThreshold && (
                    <span className="ml-1 text-status-warn">above threshold</span>
                  )}
                </span>
              ) : (
                <span>Hover to read the decoded value at a pixel</span>
              )}
            </>
          )}
        </span>
        <span className="tabular shrink-0">
          {matrix ? `${matrix.width}×${matrix.height}` : `${width}×${height}`} · ctrl+scroll
          to zoom · drag to pan
        </span>
      </div>
    </div>
  );
}
