'use client';

import { useMemo, useState } from 'react';
import { CircleCheck, CircleX, Crosshair, Loader2 } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue, Slider } from '@/components/ui/primitives';
import { useUnitAssets, type UnitAssets } from '@/hooks/use-live-demo';
import type { DemoMethod, DemoUnit } from '@/lib/api/demo';
import {
  cutoffOfIndex,
  cutoffToPosition,
  heatmapRgba,
  lesionCentroids,
  lesionCoverage,
  overlayRgba,
  pixelFnr,
  positionToIndex,
  setSize,
  TAU,
  certifies,
  shortLabel,
} from '@/lib/live-demo';
import { cn } from '@/lib/utils';

import { LayerCanvas } from './layer-canvas';

const TAG = { truth: '#facc15', found: '#22c55e', missed: '#ef4444' };
const SCRUB_STEPS = 1000;

function indexOfCutoff(cutoff: number, gridSize: number) {
  return Math.round((1 - cutoff) * (gridSize - 1));
}

/** Slider position runs left to right as the set grows (cutoff falls). */
function scrubLeft(cutoff: number) {
  return (1 - cutoffToPosition(cutoff)) * 100;
}

type Panel = { cutoff: number; found: boolean[]; overlay: Uint8ClampedArray; setPx: number; tags: Tag[] };
type Tag = { x: number; y: number; text: string; colour: string };

/** A method's prediction set on this unit at the cutoff it was calibrated to. */
function computePanel(
  assets: UnitAssets | undefined,
  base: { centroids: Array<[number, number]> } | null,
  m: DemoMethod,
  unit: DemoUnit,
  size: number,
): Panel | null {
  if (!assets || !base) return null;
  const cutoff = unit.cutoffs[m.id];
  const scores = assets.scores[m.family ?? 'threshold'];
  const found = unit.lesions.map((l) => l.found[m.id]);
  return {
    cutoff,
    found,
    overlay: overlayRgba(scores, assets.labels, size, size, cutoff, found),
    setPx: setSize(scores, cutoff),
    tags: base.centroids.map(([x, y], j) => ({ x, y, text: String(j + 1), colour: found[j] ? TAG.found : TAG.missed })),
  };
}

/** Keep a marker label inside the track: anchor it left/right near the ends instead of centring. */
function edgeShift(left: number) {
  if (left > 80) return '-translate-x-full';
  if (left < 20) return '';
  return '-translate-x-1/2';
}

function Found({ found, label }: { found: boolean; label: string }) {
  return found ? (
    <span className="inline-flex items-center gap-1 text-status-ok" title={`${label}: found`}>
      <CircleCheck className="size-3.5" aria-hidden /> found
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-status-danger" title={`${label}: missed`}>
      <CircleX className="size-3.5" aria-hidden /> missed
    </span>
  );
}

function MethodPicker({ value, onChange, methods }: { value: string; onChange: (v: string) => void; methods: DemoMethod[] }) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="h-7 w-full text-xs">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {methods.map((m) => (
          <SelectItem key={m.id} value={m.id} className="text-xs">
            {m.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export function UnitViewer({
  unit,
  methods,
  gridSize,
  size,
  methodA,
  methodB,
  onMethodA,
  onMethodB,
}: {
  unit: DemoUnit;
  methods: DemoMethod[];
  gridSize: number;
  size: number;
  methodA: string;
  methodB: string;
  onMethodA: (id: string) => void;
  onMethodB: (id: string) => void;
}) {
  const assets = useUnitAssets(unit.uid);
  const mA = methods.find((m) => m.id === methodA) ?? methods[0];
  const mB = methods.find((m) => m.id === methodB) ?? methods[1];
  const [scrubIdx, setScrubIdx] = useState(() => indexOfCutoff(unit.cutoffs[mA.id] ?? 0.5, gridSize));
  const scrubCutoff = cutoffOfIndex(scrubIdx, gridSize);
  const k = unit.k;

  const base = useMemo(() => {
    if (!assets.data) return null;
    const { labels, scores } = assets.data;
    const centroids = lesionCentroids(labels, size, k);
    const truth = overlayRgba(null, labels, size, size, null, null);
    const heat = heatmapRgba(scores.threshold);
    for (let i = 3; i < truth.length; i += 4) {
      if (truth[i] === 255) heat.set(truth.subarray(i - 3, i + 1), i - 3);
    }
    return { centroids, truth, heat };
  }, [assets.data, size, k]);

  const pA = useMemo(() => computePanel(assets.data, base, mA, unit, size), [assets.data, base, mA, unit, size]);
  const pB = useMemo(() => computePanel(assets.data, base, mB, unit, size), [assets.data, base, mB, unit, size]);

  const explorer = useMemo(() => {
    if (!assets.data || !base) return null;
    const { labels, scores } = assets.data;
    const coverage = lesionCoverage(scores.threshold, labels, scrubCutoff, k);
    const found = coverage.map((c) => c >= TAU);
    return {
      coverage,
      found,
      overlay: overlayRgba(scores.threshold, labels, size, size, scrubCutoff, found),
      setPx: setSize(scores.threshold, scrubCutoff),
      fnr: pixelFnr(scores.threshold, labels, scrubCutoff),
      tags: base.centroids.map(([x, y], j) => ({ x, y, text: String(j + 1), colour: found[j] ? TAG.found : TAG.missed })),
    };
  }, [assets.data, base, scrubCutoff, k, size]);

  const truthTags = base?.centroids.map(([x, y], j) => ({ x, y, text: String(j + 1), colour: TAG.truth }));
  const ratio = (px: number) => `${(px / Math.max(1, unit.gt_pixels)).toFixed(2)}× truth`;
  const nFound = (f: boolean[]) => f.filter(Boolean).length;

  const panelCaption = (m: DemoMethod, p: Panel | null) =>
    p && (
      <div className="space-y-0.5 text-[11px] leading-snug">
        <div className="tabular text-muted-foreground">
          cutoff {p.cutoff.toFixed(3)} · set {p.setPx.toLocaleString()} px ({ratio(p.setPx)})
        </div>
        <div className={cn('tabular font-medium', nFound(p.found) < k ? 'text-status-danger' : 'text-status-ok')}>
          polyps found {nFound(p.found)} / {k}
        </div>
        <div className="text-muted-foreground">{certifies(m.guaranteed, m.guarantee)}</div>
      </div>
    );

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="font-mono text-xs">{unit.id}</CardTitle>
          <Badge variant="outline">{k} {k === 1 ? 'polyp' : 'polyps'}</Badge>
          {unit.multi && <Badge variant="info">multi-polyp</Badge>}
          {unit.lesions.some((l) => l.small) && <Badge variant="info">small polyp</Badge>}
          {unit.rescued > 0 && <Badge variant="ok">joint rescues {unit.rescued}</Badge>}
          {unit.missed_joint > 0 && <Badge variant="danger">{unit.missed_joint} missed by joint too</Badge>}
        </div>
        <CardDescription>
          Every overlay below is computed in your browser from the model&apos;s real posterior (uint16) and the
          ground-truth label map. Teal = prediction set; outline green = polyp found (≥ {TAU * 100}% of its pixels in the
          set), red = missed.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {assets.isError && <p className="text-xs text-status-danger">Could not load this image&apos;s maps.</p>}
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <figure className="space-y-1.5">
            <figcaption className="h-7 text-xs font-medium leading-7">1 · Colonoscopy frame + ground truth</figcaption>
            <LayerCanvas size={size} image={assets.data?.image} overlay={base?.truth} tags={truthTags} label="Image with ground-truth outlines" />
            <p className="tabular text-[11px] text-muted-foreground">{unit.gt_pixels.toLocaleString()} polyp pixels</p>
          </figure>
          <figure className="space-y-1.5">
            <figcaption className="h-7 text-xs font-medium leading-7">2 · Model posterior p(polyp)</figcaption>
            <LayerCanvas size={size} overlay={base?.heat} tags={truthTags} label="Posterior heat-map" />
            <div className="space-y-0.5">
              <div className="h-1.5 rounded-full" style={{ background: 'linear-gradient(90deg, var(--unc-1), var(--unc-2), var(--unc-3), var(--unc-4))' }} />
              <div className="flex justify-between text-[10px] text-muted-foreground"><span>0</span><span>0.5</span><span>1</span></div>
            </div>
          </figure>
          {[{ m: mA, p: pA, set: onMethodA }, { m: mB, p: pB, set: onMethodB }].map(({ m, p, set }, j) => (
            <figure key={j} className="space-y-1.5">
              <figcaption className="flex items-center gap-2 text-xs font-medium">
                <span className="shrink-0">{j + 3} ·</span>
                <MethodPicker value={m.id} onChange={set} methods={methods} />
              </figcaption>
              <LayerCanvas size={size} image={assets.data?.image} overlay={p?.overlay} tags={p?.tags} label={`${m.label} prediction set`} />
              {panelCaption(m, p)}
            </figure>
          ))}
        </div>

        {assets.isLoading && (
          <p className="flex items-center gap-2 text-xs text-muted-foreground"><Loader2 className="size-3.5 animate-spin" /> Loading maps…</p>
        )}

        {explorer && (
          <div className="grid gap-4 rounded-lg border border-border p-3 lg:grid-cols-[minmax(0,280px)_1fr]">
            <div className="space-y-1.5">
              <div className="text-xs font-medium">Threshold explorer</div>
              <LayerCanvas size={size} image={assets.data?.image} overlay={explorer.overlay} tags={explorer.tags} label="Prediction set at the explorer cutoff" />
            </div>
            <div className="min-w-0 space-y-3">
              <p className="text-xs leading-relaxed text-muted-foreground">
                Drag the cutoff. The set is every pixel with posterior ≥ cutoff. Lemma 2: polyp <i>k</i> is found exactly
                when the cutoff drops below its τ-quantile score q<sup>k</sup> (the numbered ticks), so one number per polyp
                decides detection. CRC picks the cutoff from calibration images; the joint rule takes the lowest cutoff
                that satisfies all three risks.
              </p>
              <div className="tabular flex flex-wrap gap-x-4 gap-y-1 text-xs">
                <span>cutoff <b>{scrubCutoff.toFixed(3)}</b></span>
                <span>set {explorer.setPx.toLocaleString()} px ({ratio(explorer.setPx)})</span>
                <span>pixel miss {(100 * explorer.fnr).toFixed(1)}%</span>
                <span className={nFound(explorer.found) < k ? 'text-status-danger' : 'text-status-ok'}>
                  polyps found {nFound(explorer.found)} / {k}
                </span>
              </div>
              <div className="relative pt-7 pb-1">
                {[mA, mB].map((m, j) => (
                  <button
                    key={m.id + j}
                    type="button"
                    onClick={() => setScrubIdx(indexOfCutoff(unit.cutoffs[m.id], gridSize))}
                    className={cn(
                      'absolute whitespace-nowrap rounded border border-border bg-card px-1 text-[10px] hover:bg-accent',
                      edgeShift(scrubLeft(unit.cutoffs[m.id])),
                    )}
                    style={{ left: `${scrubLeft(unit.cutoffs[m.id])}%`, top: j === 0 ? 0 : 14 }}
                    title={`Jump to ${m.label}'s cutoff`}
                  >
                    ▼ {shortLabel(m.id, m.label)}
                  </button>
                ))}
                <Slider
                  min={0}
                  max={SCRUB_STEPS}
                  step={1}
                  value={[Math.round((1 - cutoffToPosition(scrubCutoff)) * SCRUB_STEPS)]}
                  onValueChange={([v]) => setScrubIdx(positionToIndex(1 - v / SCRUB_STEPS, gridSize))}
                  aria-label="Posterior cutoff"
                />
                <div className="relative mt-1 h-5">
                  {unit.lesions.map((l) => (
                    <span
                      key={l.index}
                      className={cn(
                        'absolute -translate-x-1/2 rounded px-1 text-[10px] font-semibold tabular',
                        explorer.found[l.index - 1] ? 'bg-status-ok/15 text-status-ok' : 'bg-status-danger/12 text-status-danger',
                      )}
                      style={{ left: `${scrubLeft(l.stat)}%` }}
                      title={`Polyp ${l.index}: q = ${l.stat.toFixed(4)}`}
                    >
                      |{l.index}
                    </span>
                  ))}
                </div>
                <div className="flex justify-between text-[10px] text-muted-foreground">
                  <span>cutoff 1 · empty set</span>
                  <span>log scale</span>
                  <span>0.001 · nearly whole frame</span>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="tabular w-full text-xs">
                  <thead className="text-[11px] text-muted-foreground">
                    <tr className="border-b border-border text-left">
                      <th className="py-1 pr-2 font-medium">Polyp</th>
                      <th className="py-1 pr-2 font-medium">Size</th>
                      <th className="py-1 pr-2 font-medium" title="τ-quantile of the posterior inside the polyp (τ = 0.5)">q<sup>k</sup></th>
                      <th className="py-1 pr-2 font-medium">Peak p</th>
                      <th className="py-1 pr-2 font-medium">In set at explorer cutoff</th>
                      <th className="py-1 pr-2 font-medium">{mA.label}</th>
                      <th className="py-1 font-medium">{mB.label}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {unit.lesions.map((l) => {
                      const cov = explorer.coverage[l.index - 1] ?? 0;
                      return (
                        <tr key={l.index} className="border-b border-border/60">
                          <td className="py-1.5 pr-2 font-semibold">{l.index}</td>
                          <td className="py-1.5 pr-2">
                            {l.size.toLocaleString()} px {l.small && <Badge variant="info" className="ml-1">small</Badge>}
                          </td>
                          <td className="py-1.5 pr-2 font-mono">{l.stat.toFixed(4)}</td>
                          <td className="py-1.5 pr-2 font-mono">{l.peak.toFixed(3)}</td>
                          <td className="py-1.5 pr-2">
                            <div className="flex items-center gap-2">
                              <div className="relative h-2 w-28 rounded-full bg-muted">
                                <div
                                  className={cn('absolute inset-y-0 left-0 rounded-full', cov >= TAU ? 'bg-status-ok' : 'bg-status-danger')}
                                  style={{ width: `${100 * cov}%` }}
                                />
                                <div className="absolute inset-y-[-3px] w-px bg-foreground" style={{ left: `${100 * TAU}%` }} title="τ = 50%" />
                              </div>
                              <span>{(100 * cov).toFixed(0)}%</span>
                            </div>
                          </td>
                          <td className="py-1.5 pr-2"><Found found={l.found[mA.id]} label={mA.label} /></td>
                          <td className="py-1.5"><Found found={l.found[mB.id]} label={mB.label} /></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="flex flex-wrap gap-2">
                {[mA, mB].map((m) => (
                  <Button key={m.id} size="sm" variant="outline" onClick={() => setScrubIdx(indexOfCutoff(unit.cutoffs[m.id], gridSize))}>
                    <Crosshair /> {m.label} cutoff
                  </Button>
                ))}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
