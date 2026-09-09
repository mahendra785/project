'use client';

/**
 * Bundled sample cases — real test-split slices the pipeline can run on with
 * one click, so the dashboard is never empty on a fresh install.
 */

import Link from 'next/link';
import { useState } from 'react';
import { CheckCircle2, Loader2, Play, PlayCircle } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useRunSample, useSamples } from '@/hooks/use-pums';
import { API_BASE_URL } from '@/lib/api/client';
import { cn, pct } from '@/lib/utils';
import type { SampleCase } from '@/types/api';

function fileUrl(name: string) {
  return `${API_BASE_URL}/api/samples/files/${encodeURIComponent(name)}`;
}

function SampleTile({
  sample,
  running,
  onRun,
}: {
  sample: SampleCase;
  running: string | null;
  onRun: (file: string) => void;
}) {
  const latest = sample.analysis_ids[0];
  const latestShifted = sample.shifted_analysis_ids[0];
  return (
    <div className="flex flex-col gap-2 rounded-md border border-border bg-card p-2.5">
      <div className="grid grid-cols-2 gap-1.5">
        <figure className="space-y-1">
          {/* eslint-disable-next-line @next/next/no-img-element -- backend PNG */}
          <img
            src={fileUrl(sample.image)}
            alt={`${sample.case_id} slice`}
            className="checkerboard aspect-square w-full rounded border border-border object-contain"
            style={{ imageRendering: 'pixelated' }}
          />
          <figcaption className="text-[10px] text-muted-foreground">source</figcaption>
        </figure>
        <figure className="space-y-1">
          {sample.ground_truth && (
            // eslint-disable-next-line @next/next/no-img-element -- backend PNG
            <img
              src={fileUrl(sample.ground_truth)}
              alt={`${sample.case_id} expert mask`}
              className="checkerboard aspect-square w-full rounded border border-border object-contain"
              style={{ imageRendering: 'pixelated' }}
            />
          )}
          <figcaption className="text-[10px] text-muted-foreground">expert mask</figcaption>
        </figure>
      </div>

      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate font-mono text-[11px]">{sample.case_id}</span>
        <span className="tabular shrink-0 text-[10px] text-muted-foreground">
          fg {pct(sample.foreground_fraction, 1)}
        </span>
      </div>

      <div className="flex flex-col gap-1">
        <RunRow
          label="Original"
          file={sample.image}
          count={sample.analysis_ids.length}
          latest={latest}
          running={running}
          onRun={onRun}
        />
        {sample.shifted_variant && (
          <RunRow
            label="Simulated shift"
            file={sample.shifted_variant}
            count={sample.shifted_analysis_ids.length}
            latest={latestShifted}
            running={running}
            onRun={onRun}
          />
        )}
      </div>
    </div>
  );
}

function RunRow({
  label,
  file,
  count,
  latest,
  running,
  onRun,
}: {
  label: string;
  file: string;
  count: number;
  latest: string | undefined;
  running: string | null;
  onRun: (file: string) => void;
}) {
  const isRunning = running === file;
  return (
    <div className="flex items-center justify-between gap-2 text-[11px]">
      <span className="flex items-center gap-1.5">
        {count > 0 ? (
          <CheckCircle2 className="size-3 text-status-ok" />
        ) : (
          <span className="size-3 rounded-full border border-border" />
        )}
        {label}
        {count > 0 && <span className="text-muted-foreground">×{count}</span>}
      </span>
      <span className="flex items-center gap-1">
        {latest && (
          <Button asChild variant="ghost" size="sm" className="h-6 px-1.5 text-[11px]">
            <Link href={`/uncertainty?id=${latest}`}>Open</Link>
          </Button>
        )}
        <Button
          variant="outline"
          size="sm"
          className="h-6 px-1.5 text-[11px]"
          disabled={running !== null}
          onClick={() => onRun(file)}
        >
          {isRunning ? <Loader2 className="size-3 animate-spin" /> : <Play className="size-3" />}
          Run
        </Button>
      </span>
    </div>
  );
}

export function SampleCases({ className }: { className?: string }) {
  const samples = useSamples();
  const run = useRunSample();
  const [running, setRunning] = useState<string | null>(null);
  const [queue, setQueue] = useState<string[]>([]);

  const runOne = async (file: string) => {
    setRunning(file);
    try {
      await run.mutateAsync(file);
    } finally {
      setRunning(null);
    }
  };

  // Run every not-yet-analysed file sequentially — the backend is single-process.
  const runAll = async () => {
    const files: string[] = [];
    for (const s of samples.data?.samples ?? []) {
      if (s.analysis_ids.length === 0) files.push(s.image);
      if (s.shifted_variant && s.shifted_analysis_ids.length === 0) files.push(s.shifted_variant);
    }
    setQueue(files);
    for (const f of files) {
      await runOne(f);
      setQueue((q) => q.filter((x) => x !== f));
    }
  };

  const total = samples.data?.samples.length ?? 0;
  const pending = (samples.data?.samples ?? []).reduce(
    (n, s) =>
      n + (s.analysis_ids.length === 0 ? 1 : 0) +
      (s.shifted_variant && s.shifted_analysis_ids.length === 0 ? 1 : 0),
    0,
  );

  return (
    <Card className={className}>
      <CardHeader className="flex-row items-start justify-between gap-3">
        <div>
          <CardTitle className="flex items-center gap-1.5">
            <PlayCircle className="size-4" />
            Sample cases
          </CardTitle>
          <p className="mt-0.5 text-[11px] leading-relaxed text-muted-foreground">
            Real slices from the held-out test split — patient-disjoint from training.
            &ldquo;Simulated shift&rdquo; is gamma + bias field + noise, not a second site.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {samples.data && <Badge variant="outline">{samples.data.samples[0]?.dataset_name ?? '—'}</Badge>}
          <Button size="sm" onClick={runAll} disabled={running !== null || pending === 0}>
            {running !== null && queue.length > 0 ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Play className="size-3.5" />
            )}
            {pending === 0 ? 'All run' : `Run remaining (${pending})`}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {samples.isLoading && (
          <p className="py-6 text-center text-xs text-muted-foreground">Loading samples…</p>
        )}
        {samples.isError && (
          <p className="py-6 text-center text-xs text-status-danger">Could not load samples.</p>
        )}
        {samples.data && total === 0 && (
          <p className="py-6 text-center text-xs text-muted-foreground">
            No samples exported. Run <code>python scripts/export_samples.py</code>.
          </p>
        )}
        {samples.data && total > 0 && (
          <div className={cn('grid gap-3 sm:grid-cols-2 xl:grid-cols-4')}>
            {samples.data.samples.map((s) => (
              <SampleTile key={s.id} sample={s} running={running} onRun={runOne} />
            ))}
          </div>
        )}
        {run.isError && (
          <p className="mt-3 text-[11px] text-status-danger">{run.error.userMessage}</p>
        )}
        {samples.data && (
          <p className="mt-3 text-[10px] leading-relaxed text-muted-foreground">
            {samples.data.licence} · {samples.data.citation}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
