'use client';

import { cn, num } from '@/lib/utils';

/**
 * Colour key for the uncertainty heatmap.
 *
 * The ramp mirrors the backend's PNG lookup table exactly (see
 * `backend/app/utils/encoding.py`), so the on-screen colour of a pixel is a
 * faithful readout of its numeric value rather than an independent styling
 * choice.
 */
export function UncertaintyLegend({
  threshold,
  className,
  compact = false,
}: {
  threshold?: number;
  className?: string;
  compact?: boolean;
}) {
  return (
    <div className={cn('space-y-1.5', className)}>
      {!compact && (
        <div className="flex items-baseline justify-between">
          <span className="text-[11px] font-medium text-muted-foreground">
            Normalised predictive entropy
          </span>
          {threshold !== undefined && (
            <span className="tabular text-[11px] text-muted-foreground">
              threshold {num(threshold, 2)}
            </span>
          )}
        </div>
      )}

      <div className="relative">
        <div className="uncertainty-ramp h-2.5 w-full rounded-sm" />
        {threshold !== undefined && (
          <div
            className="absolute -top-0.5 h-3.5 w-px bg-foreground"
            style={{ left: `${Math.min(Math.max(threshold, 0), 1) * 100}%` }}
            aria-hidden
          />
        )}
      </div>

      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>0.00 — confident</span>
        <span>0.50</span>
        <span>1.00 — maximally uncertain</span>
      </div>
    </div>
  );
}
