'use client';

import type { LucideIcon } from 'lucide-react';

import { ProvenanceBadge } from '@/components/research/status-badges';
import { Card, CardContent } from '@/components/ui/card';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/primitives';
import { cn } from '@/lib/utils';
import type { DataProvenance } from '@/types/api';

/**
 * A single headline figure.
 *
 * `value` is deliberately `string` — the caller formats it, and passing an em
 * dash for a missing value is the intended way to render "not measured". The
 * card never invents a placeholder number.
 */
export function MetricCard({
  label,
  value,
  hint,
  icon: Icon,
  provenance,
  tone = 'default',
  footer,
  className,
}: {
  label: string;
  value: string;
  hint?: string;
  icon?: LucideIcon;
  provenance?: DataProvenance;
  tone?: 'default' | 'ok' | 'warn' | 'danger' | 'info';
  footer?: React.ReactNode;
  className?: string;
}) {
  const toneClass = {
    default: 'text-foreground',
    ok: 'text-status-ok',
    info: 'text-status-info',
    warn: 'text-status-warn',
    danger: 'text-status-danger',
  }[tone];

  const body = (
    <Card className={cn('h-full', className)}>
      <CardContent className="space-y-1.5 p-3.5">
        <div className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
            {Icon && <Icon className="size-3.5" />}
            {label}
          </span>
          {provenance && <ProvenanceBadge provenance={provenance} />}
        </div>
        <p className={cn('tabular text-2xl font-semibold leading-none', toneClass)}>
          {value}
        </p>
        {footer && <div className="pt-0.5 text-[11px] text-muted-foreground">{footer}</div>}
      </CardContent>
    </Card>
  );

  if (!hint) return body;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="cursor-help">{body}</div>
      </TooltipTrigger>
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  );
}
