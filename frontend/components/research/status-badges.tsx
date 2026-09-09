'use client';

/**
 * Research-integrity badges.
 *
 * These are not decoration. Every figure PUMS shows must be accompanied by
 * (a) what maturity the producing component has and (b) where the number came
 * from. Centralising them here means a new page cannot accidentally render a
 * synthetic number as if it were measured.
 */

import { AlertTriangle, Beaker, CircleDot, FlaskConical, Info, ShieldCheck } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/primitives';
import { cn } from '@/lib/utils';
import type {
  DataProvenance,
  ImplementationStatus,
  RiskLevel,
  ShiftStatus,
} from '@/types/api';

const STATUS_META: Record<
  ImplementationStatus,
  { label: string; variant: 'ok' | 'info' | 'warn' | 'outline'; help: string }
> = {
  implemented: {
    label: 'IMPLEMENTED',
    variant: 'ok',
    help: 'Code exists in this repository, runs in the pipeline, and is covered by tests.',
  },
  baseline: {
    label: 'BASELINE',
    variant: 'info',
    help: 'Implemented and running, but deliberately a simple reference method rather than the research contribution.',
  },
  experimental: {
    label: 'EXPERIMENTAL',
    variant: 'warn',
    help: 'Implemented but not validated. Outputs are not research claims.',
  },
  planned: {
    label: 'PLANNED',
    variant: 'outline',
    help: 'Not implemented. Present only as an extension point; invoking it returns HTTP 501.',
  },
};

const PROVENANCE_META: Record<
  DataProvenance,
  { label: string; variant: 'ok' | 'warn' | 'danger'; help: string }
> = {
  computed: {
    label: 'COMPUTED',
    variant: 'ok',
    help: 'Computed from your uploaded image by code in this repository.',
  },
  synthetic: {
    label: 'SYNTHETIC',
    variant: 'warn',
    help: 'Derived from the procedural phantom set that ships with PUMS. A real measurement of behaviour on phantoms — NOT clinical performance.',
  },
  mock: {
    label: 'MOCK',
    variant: 'danger',
    help: 'Static placeholder. The backend for this surface is not implemented.',
  },
};

export function StatusBadge({
  status,
  className,
}: {
  status: ImplementationStatus;
  className?: string;
}) {
  const meta = STATUS_META[status];
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge variant={meta.variant} className={cn('cursor-help', className)}>
          {status === 'implemented' ? (
            <ShieldCheck className="size-3" />
          ) : status === 'planned' ? (
            <CircleDot className="size-3" />
          ) : (
            <Beaker className="size-3" />
          )}
          {meta.label}
        </Badge>
      </TooltipTrigger>
      <TooltipContent>{meta.help}</TooltipContent>
    </Tooltip>
  );
}

export function ProvenanceBadge({
  provenance,
  className,
}: {
  provenance: DataProvenance;
  className?: string;
}) {
  const meta = PROVENANCE_META[provenance];
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge variant={meta.variant} className={cn('cursor-help', className)}>
          <FlaskConical className="size-3" />
          {meta.label}
        </Badge>
      </TooltipTrigger>
      <TooltipContent>{meta.help}</TooltipContent>
    </Tooltip>
  );
}

const RISK_META: Record<RiskLevel, { label: string; variant: 'ok' | 'info' | 'warn' | 'danger' }> = {
  LOW: { label: 'Low', variant: 'ok' },
  MODERATE: { label: 'Moderate', variant: 'info' },
  HIGH: { label: 'High', variant: 'warn' },
  REVIEW_REQUIRED: { label: 'Review required', variant: 'danger' },
};

export function RiskBadge({ level, className }: { level: RiskLevel; className?: string }) {
  const meta = RISK_META[level] ?? RISK_META.LOW;
  return (
    <Badge variant={meta.variant} className={className}>
      {(level === 'HIGH' || level === 'REVIEW_REQUIRED') && (
        <AlertTriangle className="size-3" />
      )}
      {meta.label}
    </Badge>
  );
}

const SHIFT_META: Record<
  ShiftStatus,
  { label: string; variant: 'ok' | 'info' | 'warn' | 'danger' }
> = {
  ID: { label: 'In-distribution', variant: 'ok' },
  MILD_SHIFT: { label: 'Mild shift', variant: 'info' },
  MODERATE_SHIFT: { label: 'Moderate shift', variant: 'warn' },
  SEVERE_SHIFT: { label: 'Severe shift', variant: 'danger' },
};

export function ShiftBadge({ status, className }: { status: ShiftStatus; className?: string }) {
  const meta = SHIFT_META[status] ?? SHIFT_META.ID;
  return (
    <Badge variant={meta.variant} className={className}>
      {meta.label}
    </Badge>
  );
}

/** A short, always-visible note that this is not a clinical tool. */
export function ResearchDisclaimer({
  text,
  className,
}: {
  text?: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        'flex items-start gap-2 rounded-md border border-status-warn/30 bg-status-warn/8 px-3 py-2 text-xs text-muted-foreground',
        className,
      )}
    >
      <Info className="mt-0.5 size-3.5 shrink-0 text-status-warn" />
      <p className="leading-relaxed">
        {text ??
          'Research decision-support signal about model reliability. Not a diagnosis, not clinically validated, and not for patient care.'}
      </p>
    </div>
  );
}
