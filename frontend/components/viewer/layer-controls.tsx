'use client';

import {
  Columns2,
  Eye,
  Grid3x3,
  Layers2,
  RotateCcw,
  Square,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Label, Separator, Slider, Switch } from '@/components/ui/primitives';
import { num } from '@/lib/utils';
import type { ViewerSettings } from '@/stores/analysis-store';

function Toggle({
  id,
  label,
  icon: Icon,
  checked,
  onChange,
  disabled,
}: {
  id: string;
  label: string;
  icon: typeof Eye;
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <Label htmlFor={id} className="flex items-center gap-1.5 text-xs">
        <Icon className="size-3.5" />
        {label}
      </Label>
      <Switch id={id} checked={checked} onCheckedChange={onChange} disabled={disabled} />
    </div>
  );
}

export function LayerControls({
  settings,
  onChange,
  onReset,
  hasMask,
  hasRegions,
}: {
  settings: ViewerSettings;
  onChange: (patch: Partial<ViewerSettings>) => void;
  onReset: () => void;
  hasMask: boolean;
  hasRegions: boolean;
}) {
  return (
    <div className="space-y-4">
      <div className="space-y-2.5">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Layers
        </p>
        <Toggle
          id="layer-original"
          label="Source image"
          icon={Square}
          checked={settings.showOriginal}
          onChange={(v) => onChange({ showOriginal: v })}
        />
        <Toggle
          id="layer-mask"
          label="Segmentation mask"
          icon={Layers2}
          checked={settings.showMask}
          onChange={(v) => onChange({ showMask: v })}
          disabled={!hasMask}
        />
        <Toggle
          id="layer-heatmap"
          label="Uncertainty heatmap"
          icon={Grid3x3}
          checked={settings.showHeatmap}
          onChange={(v) => onChange({ showHeatmap: v })}
        />
        <Toggle
          id="layer-regions"
          label="Region boxes"
          icon={Eye}
          checked={settings.showRegions}
          onChange={(v) => onChange({ showRegions: v })}
          disabled={!hasRegions}
        />
      </div>

      <Separator />

      <div className="space-y-3.5">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Opacity
        </p>
        <div className="space-y-1.5">
          <div className="flex justify-between">
            <Label>Heatmap</Label>
            <span className="tabular text-[11px] text-muted-foreground">
              {num(settings.heatmapOpacity, 2)}
            </span>
          </div>
          <Slider
            value={[settings.heatmapOpacity]}
            min={0}
            max={1}
            step={0.05}
            onValueChange={([v]) => onChange({ heatmapOpacity: v })}
            disabled={!settings.showHeatmap}
          />
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between">
            <Label>Mask</Label>
            <span className="tabular text-[11px] text-muted-foreground">
              {num(settings.maskOpacity, 2)}
            </span>
          </div>
          <Slider
            value={[settings.maskOpacity]}
            min={0}
            max={1}
            step={0.05}
            onValueChange={([v]) => onChange({ maskOpacity: v })}
            disabled={!settings.showMask || !hasMask}
          />
        </div>
      </div>

      <Separator />

      <div className="space-y-1.5">
        <div className="flex justify-between">
          <Label>Uncertainty threshold</Label>
          <span className="tabular text-[11px] text-muted-foreground">
            {num(settings.uncertaintyThreshold, 2)}
          </span>
        </div>
        <Slider
          value={[settings.uncertaintyThreshold]}
          min={0}
          max={1}
          step={0.01}
          onValueChange={([v]) => onChange({ uncertaintyThreshold: v })}
        />
        <p className="text-[10px] leading-relaxed text-muted-foreground">
          Applied client-side to the decoded map. Changing it re-partitions the
          real values; it does not re-run the model.
        </p>
      </div>

      <Separator />

      <div className="space-y-2.5">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          View
        </p>
        <Toggle
          id="layer-side-by-side"
          label="Side by side"
          icon={Columns2}
          checked={settings.sideBySide}
          onChange={(v) => onChange({ sideBySide: v })}
        />
        <div className="flex items-center gap-1.5">
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Zoom out"
            onClick={() => onChange({ zoom: Math.max(0.5, settings.zoom - 0.25) })}
          >
            <ZoomOut className="size-3.5" />
          </Button>
          <span className="tabular min-w-11 text-center text-[11px] text-muted-foreground">
            {(settings.zoom * 100).toFixed(0)}%
          </span>
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Zoom in"
            onClick={() => onChange({ zoom: Math.min(6, settings.zoom + 0.25) })}
          >
            <ZoomIn className="size-3.5" />
          </Button>
          <Button variant="ghost" size="sm" className="ml-auto" onClick={onReset}>
            <RotateCcw className="size-3.5" />
            Reset
          </Button>
        </div>
      </div>
    </div>
  );
}
