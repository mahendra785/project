'use client';

import { useMemo, useState } from 'react';
import { AlertTriangle, Dices, Loader2, Radio, ShieldCheck } from 'lucide-react';

import { PageContainer, PageHeader } from '@/components/layout/page-container';
import { EmptyState, ErrorState, LoadingState } from '@/components/layout/states';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider, Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/primitives';
import { useDemoRescues, useDemoRun, useDemoStatus, useDemoValidity } from '@/hooks/use-live-demo';
import { demoApi, type DemoRescues, type DemoRun, type DemoUnit, type DemoValidity } from '@/lib/api/demo';
import { certifies, pct } from '@/lib/live-demo';
import { cn } from '@/lib/utils';

import { GuaranteeBars, LambdaCurves, ValidityStrips } from './demo-charts';
import { UnitViewer } from './unit-viewer';

const VALIDITY_REPS = 200;
const RESCUE_SPLITS = 100;
const ALPHA_PRESETS = [0.05, 0.1, 0.15];

type Filter = 'rescued' | 'multi' | 'small' | 'missed' | 'all';

const FILTERS: Array<{ id: Filter; label: string; test: (u: DemoUnit) => boolean }> = [
  { id: 'rescued', label: 'Rescued by joint', test: (u) => u.rescued > 0 },
  { id: 'multi', label: 'Multi-polyp', test: (u) => u.multi },
  { id: 'small', label: 'Small polyp missed by pixel CRC', test: (u) => u.lesions.some((l) => l.small && !l.found.pixel) },
  { id: 'missed', label: 'Missed by both', test: (u) => u.missed_joint > 0 },
  { id: 'all', label: 'All test images', test: () => true },
];

/** Polyp counts on this split's test images, for the KPI tiles. */
function splitCounts(units: DemoUnit[]) {
  const all = units.flatMap((u) => u.lesions.map((l) => ({ ...l, multi: u.multi })));
  const count = (pred: (l: (typeof all)[number]) => boolean, m: string) => all.filter((l) => pred(l) && !l.found[m]).length;
  const scope = {
    all: () => true,
    small: (l: (typeof all)[number]) => l.small,
    multi: (l: (typeof all)[number]) => l.multi,
  };
  return Object.fromEntries(
    (Object.keys(scope) as Array<keyof typeof scope>).map((k) => [
      k,
      {
        total: all.filter(scope[k]).length,
        pixel: count(scope[k], 'pixel'),
        joint: count(scope[k], 'joint'),
        joint_multi: count(scope[k], 'joint_multi'),
      },
    ]),
  ) as Record<keyof typeof scope, { total: number; pixel: number; joint: number; joint_multi: number }>;
}

function Verdict({ validity, alpha, onAlpha }: { validity: DemoValidity | undefined; alpha: number; onAlpha: (a: number) => void }) {
  if (!validity) {
    return (
      <Card>
        <CardContent className="flex items-center gap-2 py-5 text-xs text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Re-calibrating every method on {VALIDITY_REPS} random splits at α = {alpha}…
        </CardContent>
      </Card>
    );
  }
  const by = Object.fromEntries(validity.methods.map((m) => [m.id, m.stats]));
  const p = by.pixel;
  const j = by.joint;
  const jm = by.joint_multi;
  const broken = (['les', 'small'] as const).filter((r) => (p[r].mean ?? 0) > alpha);
  const sizeRatio = (j.set_to_truth.mean ?? 0) / Math.max(1e-9, p.set_to_truth.mean ?? 1);
  const name = { les: 'polyp', small: 'small-polyp' } as const;
  return (
    <Card className={cn(broken.length ? 'border-status-danger/40' : 'border-status-ok/40')}>
      <CardContent className="space-y-3 py-4">
        <div className="flex items-start gap-3">
          <div className={cn('rounded-full p-2', broken.length ? 'bg-status-danger/12 text-status-danger' : 'bg-status-ok/12 text-status-ok')}>
            {broken.length ? <AlertTriangle className="size-5" /> : <ShieldCheck className="size-5" />}
          </div>
          <div className="space-y-1.5 text-sm leading-relaxed">
            {broken.length ? (
              <p>
                <b>At α = {alpha}, pixel CRC breaks the {broken.map((r) => name[r]).join(' and ')} target.</b> Across{' '}
                {validity.reps} random calibration/test splits its mean{' '}
                {broken.map((r) => `${name[r]} miss rate is ${pct(p[r].mean)} (above α on ${pct(p[r].frac_above, 0)} of splits)`).join('; ')}.
                The joint rule holds {broken.map((r) => `${name[r]} miss at ${pct(j[r].mean)}`).join(' and ')}, certified, with
                prediction sets {sizeRatio.toFixed(1)}× as large as pixel CRC&apos;s.
              </p>
            ) : (
              <p>
                <b>At α = {alpha}, pixel CRC happens to meet the polyp and small-polyp targets on this model too</b> (means{' '}
                {pct(p.les.mean)} and {pct(p.small.mean)}), but nothing certifies that. The joint rule certifies all three
                risks for {sizeRatio >= 1 ? `${pct(sizeRatio - 1, 0)} larger` : 'no larger'} sets.{' '}
                {alpha > 0.05 && (
                  <>
                    Lower α to see where pixel CRC breaks:{' '}
                    <button type="button" className="font-medium text-primary underline-offset-2 hover:underline" onClick={() => onAlpha(0.05)}>
                      try α = 0.05
                    </button>
                    .
                  </>
                )}
              </p>
            )}
            <p className="text-xs text-muted-foreground">
              Neither rule controls misses on multi-polyp images (pixel {pct(p.multi.mean)}, joint {pct(j.multi.mean)}); adding that
              risk (joint + multi) brings it to {pct(jm.multi.mean)} but needs sets {(jm.set_to_truth.mean ?? 0).toFixed(1)}× the polyp
              area, because only a few calibration images have more than one polyp.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function Kpi({
  title,
  total,
  pixel,
  joint,
  rate,
  alpha,
  note,
}: {
  title: string;
  total: number;
  pixel: number;
  joint: number;
  rate: { pixel: number | null; joint: number | null };
  alpha: number;
  note?: string;
}) {
  const row = (label: string, n: number, r: number | null, ours: boolean) => (
    <div className="flex items-baseline justify-between gap-2">
      <span className={cn('text-xs', ours ? 'font-medium text-primary' : 'text-muted-foreground')}>{label}</span>
      <span className={cn('tabular text-sm font-semibold', r !== null && r > alpha && 'text-status-danger')}>
        {n} <span className="text-[11px] font-normal text-muted-foreground">({pct(r)})</span>
        {r !== null && r > alpha && <span className="ml-1 text-[10px] font-medium">&gt; α</span>}
      </span>
    </div>
  );
  return (
    <Card>
      <CardContent className="space-y-1.5 py-3">
        <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{title}</div>
        <div className="tabular text-[11px] text-muted-foreground">of {total} on held-out images</div>
        {row('Pixel CRC', pixel, rate.pixel, false)}
        {row('Joint (ours)', joint, rate.joint, true)}
        {note && <div className="pt-1 text-[10px] leading-snug text-muted-foreground">{note}</div>}
      </CardContent>
    </Card>
  );
}

function Gallery({
  units,
  filter,
  onFilter,
  selected,
  onSelect,
}: {
  units: DemoUnit[];
  filter: Filter;
  onFilter: (f: Filter) => void;
  selected: number | null;
  onSelect: (uid: number) => void;
}) {
  const [limit, setLimit] = useState(12);
  const active = FILTERS.find((f) => f.id === filter) ?? FILTERS[0];
  const shown = units.filter(active.test);
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle>Held-out images, ranked by significance</CardTitle>
        <CardDescription>
          Ranked by what each image shows about the method: a polyp the joint rule rescues counts most, then multi-polyp images,
          then polyps pixel CRC misses. Click an image to inspect it.
        </CardDescription>
        <div className="flex flex-wrap gap-1.5 pt-1">
          {FILTERS.map((f) => (
            <Button key={f.id} size="sm" variant={f.id === filter ? 'default' : 'outline'} onClick={() => onFilter(f.id)}>
              {f.label} <span className="tabular opacity-70">{units.filter(f.test).length}</span>
            </Button>
          ))}
        </div>
      </CardHeader>
      <CardContent>
        {shown.length === 0 ? (
          <p className="py-6 text-center text-xs text-muted-foreground">No held-out image in this split matches. Try another α or split.</p>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-6">
              {shown.slice(0, limit).map((u) => (
                <button
                  key={u.uid}
                  type="button"
                  onClick={() => onSelect(u.uid)}
                  className={cn(
                    'group overflow-hidden rounded-md border text-left transition-colors',
                    u.uid === selected ? 'border-primary ring-2 ring-primary/40' : 'border-border hover:border-primary/60',
                  )}
                >
                  {/* CORS mode so the cached copy can be reused by the viewer's fetch() of the same URL. */}
                  {/* eslint-disable-next-line @next/next/no-img-element -- backend JPEG served raw */}
                  <img src={demoApi.imageUrl(u.uid)} alt={`Test image ${u.id}`} crossOrigin="anonymous" loading="lazy" className="aspect-square w-full bg-black object-cover" />
                  <div className="flex flex-wrap gap-1 p-1.5">
                    <Badge variant="outline">{u.k}×</Badge>
                    {u.rescued > 0 && <Badge variant="ok">+{u.rescued} rescued</Badge>}
                    {u.missed_pixel > 0 && <Badge variant="danger">pixel −{u.missed_pixel}</Badge>}
                    {u.missed_joint > 0 && <Badge variant="warn">joint −{u.missed_joint}</Badge>}
                  </div>
                </button>
              ))}
            </div>
            {shown.length > limit && (
              <Button size="sm" variant="ghost" className="mt-2" onClick={() => setLimit((l) => l + 12)}>
                Show more ({shown.length - limit} left)
              </Button>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Every image where pixel CRC's set disagrees with the ground truth (a polyp missed) and the
 * joint rule's set finds it, collected over many splits. Opening one switches the page to a
 * split where that rescue happens, so the viewer below shows it with the real cutoffs.
 */
function RescueGallery({
  rescues,
  alpha,
  selected,
  onOpen,
}: {
  rescues: DemoRescues | undefined;
  alpha: number;
  selected: number | null;
  onOpen: (uid: number, seed: number) => void;
}) {
  return (
    <Card className="border-status-ok/40">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2">
          <ShieldCheck className="size-4 text-status-ok" /> Pixel CRC disagrees with the ground truth, the joint rule fixes it
        </CardTitle>
        <CardDescription>
          Every image where, at α = {alpha}, pixel CRC&apos;s prediction set misses a ground-truth polyp and the joint rule&apos;s set
          finds it, collected over {RESCUE_SPLITS} random calibration splits (an image is held out in roughly 40% of them).
          Click one to open it in the viewer on a split where the rescue happens.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!rescues ? (
          <LoadingState className="min-h-24" label={`Scanning ${RESCUE_SPLITS} splits…`} />
        ) : rescues.units.length === 0 ? (
          <p className="py-6 text-center text-xs text-muted-foreground">
            At α = {alpha} pixel CRC and the joint rule find the same polyps on every held-out image. Lower α to see rescues.
          </p>
        ) : (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-6">
            {rescues.units.map((u) => (
              <button
                key={u.uid}
                type="button"
                onClick={() => onOpen(u.uid, u.seed)}
                className={cn(
                  'overflow-hidden rounded-md border text-left transition-colors',
                  u.uid === selected ? 'border-primary ring-2 ring-primary/40' : 'border-border hover:border-primary/60',
                )}
              >
                {/* eslint-disable-next-line @next/next/no-img-element -- backend JPEG served raw */}
                <img src={demoApi.imageUrl(u.uid)} alt={`Image ${u.id}`} crossOrigin="anonymous" loading="lazy" className="aspect-square w-full bg-black object-cover" />
                <div className="space-y-1 p-1.5">
                  <div className="flex flex-wrap gap-1">
                    <Badge variant="ok">polyp {u.lesions.join(', ')} rescued</Badge>
                    {u.small && <Badge variant="info">small</Badge>}
                    {u.multi && <Badge variant="outline">{u.k} polyps</Badge>}
                  </div>
                  <div className="tabular text-[10px] text-muted-foreground">
                    rescued in {u.n_rescued} of {u.n_test} splits it was held out
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function MethodsTable({ run }: { run: DemoRun }) {
  return (
    <div className="overflow-x-auto">
      <table className="tabular w-full min-w-[760px] text-xs">
        <thead className="text-[11px] text-muted-foreground">
          <tr className="border-b border-border text-left">
            <th className="py-1.5 pr-3 font-medium">Method</th>
            <th className="py-1.5 pr-3 font-medium">Score family</th>
            <th className="py-1.5 pr-3 font-medium">What it certifies</th>
            <th className="py-1.5 pr-3 font-medium">Cutoff on this split</th>
            <th className="py-1.5 font-medium">Set size / truth</th>
          </tr>
        </thead>
        <tbody>
          {run.methods.map((m) => (
            <tr key={m.id} className={cn('border-b border-border/60', m.ours && 'bg-primary/5')}>
              <td className={cn('py-1.5 pr-3 font-medium', m.ours && 'text-primary')}>{m.label}</td>
              <td className="py-1.5 pr-3 text-muted-foreground">{m.family?.replace('_', '-')}</td>
              <td className="py-1.5 pr-3">{certifies(m.guaranteed, m.guarantee)}</td>
              <td className="py-1.5 pr-3 font-mono">
                {m.cutoff.map((c) => c.toFixed(3)).join(' / ')}
                {m.cutoff.length > 1 && <span className="ml-1 font-sans text-[10px] text-muted-foreground">(per mass tercile)</span>}
                {!m.feasible && <Badge variant="warn" className="ml-1">infeasible → full frame</Badge>}
              </td>
              <td className="py-1.5">{(m.test.set_to_truth ?? 0).toFixed(2)}×</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function LiveDemo() {
  const status = useDemoStatus();
  const ready = status.data?.available === true;
  const [alphaDraft, setAlphaDraft] = useState(0.05);
  const [alpha, setAlpha] = useState(0.05);
  const [seed, setSeed] = useState(0);
  const [filter, setFilter] = useState<Filter>('rescued');
  const [picked, setPicked] = useState<number | null>(null);
  const [methodA, setMethodA] = useState('pixel');
  const [methodB, setMethodB] = useState('joint');

  const run = useDemoRun(alpha, seed, ready);
  const validity = useDemoValidity(alpha, VALIDITY_REPS, ready);
  const rescues = useDemoRescues(alpha, RESCUE_SPLITS, ready);
  const data = run.data;

  const counts = useMemo(() => (data ? splitCounts(data.units) : null), [data]);
  // Default to the most significant image of the split; keep a user's pick while it stays in the test set.
  const selectedUnit = data ? (data.units.find((u) => u.uid === picked) ?? data.units[0]) : undefined;

  const chooseAlpha = (a: number) => {
    setAlphaDraft(a);
    setAlpha(a);
  };

  const header = (
    <PageHeader
      title="Live demo · lesion-level joint risk control"
      description="Pixel-level conformal risk control promises that at most α of polyp pixels are missed, which says nothing about whole polyps; a large polyp can hide a small one entirely. The joint rule calibrates one threshold that bounds pixel, polyp and small-polyp miss rates together. Everything below is recomputed live from a frozen U-Net's real posteriors on Kvasir-SEG."
      badges={
        ready && status.data?.available ? (
          <>
            <Badge variant="ok"><Radio className="size-3" /> live</Badge>
            <Badge variant="outline">{status.data.dataset} · {status.data.arch} · seed {status.data.seed}</Badge>
            <Badge variant="outline">pool Dice {status.data.pool_metrics.dice?.toFixed(3)}</Badge>
          </>
        ) : undefined
      }
    />
  );

  if (status.isLoading) {
    return (
      <PageContainer>
        {header}
        <LoadingState label="Loading the frozen model's posteriors. The first load computes every image's loss curves (about 10 s)." />
      </PageContainer>
    );
  }
  if (status.isError) {
    return (
      <PageContainer>
        {header}
        <ErrorState error={status.error} onRetry={() => status.refetch()} />
      </PageContainer>
    );
  }
  if (status.data && !status.data.available) {
    return (
      <PageContainer>
        {header}
        <EmptyState
          title="Demo data not built"
          description={`${status.data.reason} From backend/: python scripts/experiments/kaggle_train.py --dataset kvasir-seg --arch unet-r34 --seed 0 --data-root var/demo_data, then python scripts/build_demo_bundle.py --data-root var/demo_data.`}
        />
      </PageContainer>
    );
  }
  const st = status.data!;

  return (
    <PageContainer>
      {header}

      <Card>
        <CardContent className="grid gap-4 py-4 lg:grid-cols-[1.4fr_1fr_1.2fr]">
          <div className="space-y-2">
            <div className="flex items-baseline justify-between">
              <span className="text-xs font-medium">Target miss rate α</span>
              <span className="tabular text-sm font-semibold">{alphaDraft.toFixed(2)}</span>
            </div>
            <Slider
              min={0.03}
              max={0.25}
              step={0.01}
              value={[alphaDraft]}
              onValueChange={([v]) => setAlphaDraft(Math.round(v * 100) / 100)}
              onValueCommit={([v]) => setAlpha(Math.round(v * 100) / 100)}
              aria-label="Target miss rate alpha"
            />
            <div className="flex gap-1.5">
              {ALPHA_PRESETS.map((a) => (
                <Button key={a} size="sm" variant={alpha === a ? 'secondary' : 'ghost'} className="h-6 px-2" onClick={() => chooseAlpha(a)}>
                  α = {a}
                </Button>
              ))}
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-xs font-medium">Calibration split</div>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline" onClick={() => setSeed((s) => s + 1)}>
                <Dices /> New random split
              </Button>
              <span className="tabular text-xs text-muted-foreground">seed {seed}</span>
              {run.isFetching && <Loader2 className="size-3.5 animate-spin text-muted-foreground" />}
            </div>
            <p className="text-[11px] leading-snug text-muted-foreground">
              Each split re-draws which images calibrate and which are held out, then recalibrates all {st.methods.length} methods.
            </p>
          </div>
          {data && (
            <div className="tabular grid grid-cols-3 gap-2 text-center">
              {[
                ['tune', data.n_tune],
                ['calibrate', data.n_cal],
                ['test (held out)', data.n_test],
              ].map(([l, n]) => (
                <div key={l} className="rounded-md bg-muted/60 p-2">
                  <div className="text-base font-semibold">{n}</div>
                  <div className="text-[10px] text-muted-foreground">{l}</div>
                </div>
              ))}
              <p className="col-span-3 text-[10px] text-muted-foreground">
                of {st.n_units} Kvasir-SEG images · {st.n_lesions} polyps · small = under {Math.round(st.small_cut).toLocaleString()} px at {st.size} px ·{' '}
                {data.eligible_cal.small} calibration images with a small polyp, {data.eligible_cal.multi} with several
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Verdict validity={validity.data} alpha={alpha} onAlpha={chooseAlpha} />
      {validity.isError && <ErrorState error={validity.error} onRetry={() => validity.refetch()} />}
      {run.isError && <ErrorState error={run.error} onRetry={() => run.refetch()} />}

      {data && counts && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {(() => {
              const t = (id: string) => data.methods.find((m) => m.id === id)?.test;
              const p = t('pixel');
              const j = t('joint');
              const sizeP = p?.set_to_truth ?? 0;
              const sizeJ = j?.set_to_truth ?? 0;
              return (
                <>
                  <Kpi title="Polyps missed" total={counts.all.total} pixel={counts.all.pixel} joint={counts.all.joint}
                    rate={{ pixel: p?.les ?? null, joint: j?.les ?? null }} alpha={alpha}
                    note="Rate = mean per-image share of polyps missed (the calibrated risk)." />
                  <Kpi title="Small polyps missed" total={counts.small.total} pixel={counts.small.pixel} joint={counts.small.joint}
                    rate={{ pixel: p?.small ?? null, joint: j?.small ?? null }} alpha={alpha}
                    note="The risk pixel CRC ignores: small polyps contribute few pixels." />
                  <Kpi title="Polyps missed on multi-polyp images" total={counts.multi.total} pixel={counts.multi.pixel} joint={counts.multi.joint}
                    rate={{ pixel: p?.multi ?? null, joint: j?.multi ?? null }} alpha={alpha}
                    note={`Not controlled by either rule. Joint + multi misses ${counts.multi.joint_multi}.`} />
                  <Card>
                    <CardContent className="space-y-1.5 py-3">
                      <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Cost: prediction-set size</div>
                      <div className="text-[11px] text-muted-foreground">set pixels / polyp pixels, held-out images</div>
                      <div className="flex items-baseline justify-between"><span className="text-xs text-muted-foreground">Pixel CRC</span><span className="tabular text-sm font-semibold">{sizeP.toFixed(2)}×</span></div>
                      <div className="flex items-baseline justify-between"><span className="text-xs font-medium text-primary">Joint (ours)</span><span className="tabular text-sm font-semibold">{sizeJ.toFixed(2)}×</span></div>
                      <div className="pt-1 text-[10px] text-muted-foreground">
                        Joint sets are {(sizeJ / Math.max(1e-9, sizeP)).toFixed(2)}× pixel CRC&apos;s on this split. Pixel miss: {pct(p?.pix)} vs {pct(j?.pix)}.
                      </div>
                    </CardContent>
                  </Card>
                </>
              );
            })()}
          </div>

          <RescueGallery
            rescues={rescues.data}
            alpha={alpha}
            selected={selectedUnit?.uid ?? null}
            onOpen={(uid, s) => {
              setSeed(s);
              setPicked(uid);
              setMethodA('pixel');
              setMethodB('joint');
              document.getElementById('unit-viewer')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }}
          />

          <Gallery units={data.units} filter={filter} onFilter={setFilter} selected={selectedUnit?.uid ?? null} onSelect={setPicked} />

          {selectedUnit && (
            <div id="unit-viewer" className="scroll-mt-4">
              <UnitViewer
                key={`${selectedUnit.uid}-${alpha}-${seed}`}
                unit={selectedUnit}
                methods={data.methods}
                gridSize={data.grid_size}
                size={st.size}
                methodA={methodA}
                methodB={methodB}
                onMethodA={setMethodA}
                onMethodB={setMethodB}
              />
            </div>
          )}

          <Card>
            <CardContent className="pt-4">
              <Tabs defaultValue="guarantees">
                <TabsList className="flex-wrap">
                  <TabsTrigger value="guarantees">Risks on this split</TabsTrigger>
                  <TabsTrigger value="lambda">How the threshold is chosen</TabsTrigger>
                  <TabsTrigger value="validity">Across {VALIDITY_REPS} splits</TabsTrigger>
                  <TabsTrigger value="methods">Methods</TabsTrigger>
                </TabsList>
                <TabsContent value="guarantees" className="space-y-2">
                  <p className="text-xs text-muted-foreground">
                    Held-out risk of every method on the {data.n_test} test images of this split. Red bars exceed α. One split is
                    noisy; the &ldquo;Across {VALIDITY_REPS} splits&rdquo; tab shows whether a method meets α on average, which is what CRC promises.
                  </p>
                  <GuaranteeBars run={data} />
                </TabsContent>
                <TabsContent value="lambda" className="space-y-2">
                  <p className="text-xs text-muted-foreground">
                    For each risk, the corrected calibration risk (n·R̂ + 1)/(n + 1) on the {data.n_cal} calibration images as the
                    cutoff falls. CRC picks the first cutoff where the curve is at or below α. Pixel CRC stops at the pixel curve; the joint rule
                    keeps going until all three curves are under α.
                  </p>
                  <LambdaCurves run={data} />
                </TabsContent>
                <TabsContent value="validity">
                  {validity.data ? (
                    <ValidityStrips validity={validity.data} />
                  ) : (
                    <LoadingState label={`Calibrating every method on ${VALIDITY_REPS} random splits…`} />
                  )}
                </TabsContent>
                <TabsContent value="methods">
                  <MethodsTable run={data} />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </>
      )}

      <Card className="border-status-warn/40">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2"><AlertTriangle className="size-4 text-status-warn" /> What this demo does not show</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc space-y-1 pl-5 text-xs leading-relaxed text-muted-foreground">
            <li>The joint rule does not fix misses on multi-polyp images; it does not control that risk, and controlling it (joint + multi) needs near-full-frame sets here.</li>
            <li>Its cost grows as α falls: sets roughly 2× pixel CRC&apos;s at α = 0.05 on this model, and close to equal at α ≥ 0.1.</li>
            <li>The lesion-equalized score family gave no real-data gain over plain thresholding; it is shown for completeness.</li>
            <li>A single split is noisy. CRC bounds the expected risk over splits, so individual splits can exceed α.</li>
            <li>One model (U-Net/ResNet-34, seed {st.seed}, retrained locally, best validation Dice {st.best_val_dice.toFixed(3)}) on one dataset. Research prototype, not a medical device.</li>
          </ul>
        </CardContent>
      </Card>
    </PageContainer>
  );
}
