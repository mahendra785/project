import { cn } from '@/lib/utils';

export function PageContainer({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('mx-auto w-full max-w-[1500px] space-y-5 p-5', className)}>
      {children}
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
  badges,
}: {
  title: string;
  description?: string;
  actions?: React.ReactNode;
  badges?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0 space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
          {badges}
        </div>
        {description && (
          <p className="max-w-3xl text-xs leading-relaxed text-muted-foreground">
            {description}
          </p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

/** Dense label/value row used throughout the metric panels. */
export function DataRow({
  label,
  value,
  hint,
  className,
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
  className?: string;
}) {
  return (
    <div className={cn('flex items-baseline justify-between gap-4 py-1', className)}>
      <span className="text-xs text-muted-foreground" title={hint}>
        {label}
      </span>
      <span className="tabular text-xs font-medium">{value}</span>
    </div>
  );
}
