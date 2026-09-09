'use client';

/** Segmentation mask PNG plus region/detection boxes, drawn in image space. */

import { cn } from '@/lib/utils';
import type { ObjectUncertainty } from '@/types/api';

const RISK_STROKE: Record<string, string> = {
  LOW: 'var(--status-ok)',
  MODERATE: 'var(--status-info)',
  HIGH: 'var(--status-warn)',
  REVIEW_REQUIRED: 'var(--status-danger)',
};

export function PredictionOverlay({
  maskUrl,
  opacity,
  className,
}: {
  maskUrl: string | null;
  opacity: number;
  className?: string;
}) {
  if (!maskUrl) return null;
  return (
    // eslint-disable-next-line @next/next/no-img-element -- backend-served PNG, no optimisation wanted
    <img
      src={maskUrl}
      alt="Predicted segmentation mask"
      className={cn('absolute inset-0 size-full', className)}
      style={{ opacity, imageRendering: 'pixelated' }}
    />
  );
}

export function RegionOverlay({
  objects,
  width,
  height,
  selectedId,
  onSelect,
  className,
}: {
  objects: ObjectUncertainty[];
  width: number;
  height: number;
  selectedId?: string | null;
  onSelect?: (id: string | null) => void;
  className?: string;
}) {
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className={cn('absolute inset-0 size-full', className)}
      role="group"
      aria-label="High-uncertainty regions"
    >
      {objects.map((object) => {
        const selected = object.id === selectedId;
        const stroke = RISK_STROKE[object.risk_level] ?? 'var(--status-info)';
        return (
          <g key={object.id}>
            <rect
              x={object.bbox.x}
              y={object.bbox.y}
              width={object.bbox.width}
              height={object.bbox.height}
              fill={selected ? stroke : 'transparent'}
              fillOpacity={selected ? 0.14 : 0}
              stroke={stroke}
              strokeWidth={selected ? 2.5 : 1.25}
              strokeDasharray={object.source === 'detection' ? undefined : '4 2'}
              vectorEffect="non-scaling-stroke"
              className="cursor-pointer transition-all"
              onClick={() => onSelect?.(selected ? null : object.id)}
            >
              <title>
                {object.label} — uncertainty {object.uncertainty.toFixed(3)}
              </title>
            </rect>
          </g>
        );
      })}
    </svg>
  );
}
