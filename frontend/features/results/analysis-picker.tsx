'use client';

/** Shared selector for pages that operate on one previously-run analysis. */

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/primitives';
import { num, relativeTime } from '@/lib/utils';
import type { AnalysisSummary } from '@/types/api';

export function AnalysisPicker({
  items,
  value,
  onChange,
  className,
}: {
  items: AnalysisSummary[];
  value: string | null;
  onChange: (id: string) => void;
  className?: string;
}) {
  return (
    <Select value={value ?? undefined} onValueChange={onChange}>
      <SelectTrigger className={className ?? 'w-[340px]'}>
        <SelectValue placeholder="Select an analysis" />
      </SelectTrigger>
      <SelectContent>
        {items.map((item) => (
          <SelectItem key={item.analysis_id} value={item.analysis_id}>
            <span className="flex w-full items-center gap-2">
              <span className="max-w-[150px] truncate">{item.filename}</span>
              <span className="tabular text-[10px] text-muted-foreground">
                u={num(item.mean_uncertainty, 2)} · {relativeTime(item.created_at)}
              </span>
            </span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
