'use client';

import { useEffect, useRef } from 'react';

import { cn } from '@/lib/utils';

/**
 * One viewer panel: an optional base image, an RGBA overlay computed from the real
 * score/label arrays, and numbered tags at each lesion centroid. The overlay goes
 * through an offscreen canvas because putImageData replaces pixels instead of blending.
 */
export function LayerCanvas({
  size,
  image,
  overlay,
  tags,
  className,
  label,
}: {
  size: number;
  image?: ImageBitmap | null;
  overlay?: Uint8ClampedArray | null;
  tags?: Array<{ x: number; y: number; text: string; colour: string }>;
  className?: string;
  label: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    canvas.width = size;
    canvas.height = size;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, size, size);
    if (image) ctx.drawImage(image, 0, 0, size, size);
    if (overlay && overlay.length === size * size * 4) {
      const off = document.createElement('canvas');
      off.width = size;
      off.height = size;
      off.getContext('2d')?.putImageData(new ImageData(new Uint8ClampedArray(overlay), size, size), 0, 0);
      ctx.drawImage(off, 0, 0);
    }
    for (const tag of tags ?? []) {
      const r = Math.max(9, size / 30);
      ctx.beginPath();
      ctx.arc(tag.x, tag.y, r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.72)';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = tag.colour;
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.font = `600 ${Math.round(r * 1.15)}px ui-sans-serif, system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(tag.text, tag.x, tag.y + 0.5);
    }
  }, [size, image, overlay, tags]);

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={label}
      className={cn('aspect-square w-full rounded-md border border-border bg-black [image-rendering:auto]', className)}
    />
  );
}
