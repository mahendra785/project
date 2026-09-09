'use client';

/**
 * Renders the decoded uncertainty matrix to a canvas.
 *
 * Colour and transparency are computed here from the real numeric values, so
 * the threshold slider genuinely re-partitions the data rather than adjusting a
 * CSS filter over a fixed picture. Pixels below the threshold are dimmed rather
 * than hidden, which keeps the spatial context of the uncertain regions.
 */

import { useEffect, useRef } from 'react';

import {
  uncertaintyColour,
  type UncertaintyMatrix,
} from '@/hooks/use-uncertainty-matrix';
import { cn } from '@/lib/utils';

export function UncertaintyHeatmap({
  matrix,
  threshold,
  opacity,
  showOnlyAboveThreshold = false,
  className,
}: {
  matrix: UncertaintyMatrix;
  threshold: number;
  opacity: number;
  showOnlyAboveThreshold?: boolean;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = matrix.width;
    canvas.height = matrix.height;
    const context = canvas.getContext('2d');
    if (!context) return;

    const image = context.createImageData(matrix.width, matrix.height);
    for (let i = 0; i < matrix.values.length; i += 1) {
      const value = matrix.values[i];
      const [r, g, b] = uncertaintyColour(value);
      const above = value >= threshold;

      image.data[i * 4] = r;
      image.data[i * 4 + 1] = g;
      image.data[i * 4 + 2] = b;

      if (showOnlyAboveThreshold && !above) {
        image.data[i * 4 + 3] = 0;
      } else {
        // Below-threshold pixels stay visible but recede, so the eye is drawn
        // to flagged regions without losing the surrounding field.
        const alpha = above ? 0.45 + 0.55 * value : 0.18 * value;
        image.data[i * 4 + 3] = Math.round(alpha * 255);
      }
    }
    context.putImageData(image, 0, 0);
  }, [matrix, threshold, showOnlyAboveThreshold]);

  return (
    <canvas
      ref={canvasRef}
      aria-label="Uncertainty heatmap"
      className={cn('absolute inset-0 size-full', className)}
      style={{ opacity, imageRendering: 'pixelated' }}
    />
  );
}
