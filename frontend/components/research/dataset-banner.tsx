'use client';

/**
 * States which dataset is backing the numbers on screen.
 *
 * This is the single most important honesty surface in the app: the same
 * conformal threshold means something completely different depending on
 * whether it was fitted on expert-annotated MRI or on procedural phantoms.
 * Showing the source next to the results removes the ambiguity.
 */

import { Database, TriangleAlert } from 'lucide-react';

import { ProvenanceBadge } from '@/components/research/status-badges';
import { useDatasets } from '@/hooks/use-pums';
import { cn } from '@/lib/utils';

export function DatasetBanner({ className }: { className?: string }) {
  const { data } = useDatasets();
  if (!data) return null;

  const { active } = data;
  const isReal = active.available;

  return (
    <div
      className={cn(
        'flex items-start gap-2.5 rounded-md border px-3 py-2.5 text-xs',
        isReal
          ? 'border-status-ok/30 bg-status-ok/6'
          : 'border-status-warn/35 bg-status-warn/8',
        className,
      )}
    >
      {isReal ? (
        <Database className="mt-0.5 size-3.5 shrink-0 text-status-ok" />
      ) : (
        <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-status-warn" />
      )}
      <div className="min-w-0 space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-medium">
            {isReal ? active.dataset_name : 'No real dataset — using synthetic phantoms'}
          </span>
          <ProvenanceBadge provenance={active.provenance} />
        </div>
        <p className="leading-relaxed text-muted-foreground">{active.summary}</p>
        {active.download_command && (
          <p className="pt-0.5">
            <code className="rounded bg-muted px-1.5 py-0.5 text-[11px]">
              {active.download_command}
            </code>
          </p>
        )}
      </div>
    </div>
  );
}
