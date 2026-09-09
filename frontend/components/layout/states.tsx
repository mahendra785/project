'use client';

/** Shared loading / empty / error states so every page fails the same way. */

import Link from 'next/link';
import { AlertCircle, Inbox, Loader2, PlugZap, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { ApiError } from '@/lib/api/client';
import { cn } from '@/lib/utils';

export function LoadingState({
  label = 'Loading…',
  className,
}: {
  label?: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        'flex min-h-40 flex-col items-center justify-center gap-3 text-sm text-muted-foreground',
        className,
      )}
    >
      <Loader2 className="size-5 animate-spin" />
      <span>{label}</span>
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-md bg-muted', className)} />;
}

export function EmptyState({
  title,
  description,
  actionLabel,
  actionHref,
  icon,
  className,
}: {
  title: string;
  description: string;
  actionLabel?: string;
  actionHref?: string;
  icon?: React.ReactNode;
  className?: string;
}) {
  return (
    <Card className={cn('border-dashed', className)}>
      <CardContent className="flex flex-col items-center gap-3 px-6 py-12 text-center">
        <div className="rounded-full bg-muted p-3 text-muted-foreground">
          {icon ?? <Inbox className="size-5" />}
        </div>
        <div className="space-y-1">
          <p className="text-sm font-medium">{title}</p>
          <p className="mx-auto max-w-md text-xs leading-relaxed text-muted-foreground">
            {description}
          </p>
        </div>
        {actionLabel && actionHref && (
          <Button asChild size="sm" className="mt-1">
            <Link href={actionHref}>{actionLabel}</Link>
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

export function ErrorState({
  error,
  onRetry,
  className,
}: {
  error: unknown;
  onRetry?: () => void;
  className?: string;
}) {
  const apiError = error instanceof ApiError ? error : null;
  const isBackendDown = apiError?.isNetworkError ?? false;
  const message =
    apiError?.userMessage ??
    (error instanceof Error ? error.message : 'An unexpected error occurred.');

  return (
    <Card className={cn('border-destructive/40 bg-destructive/4', className)}>
      <CardContent className="flex flex-col gap-3 px-6 py-8">
        <div className="flex items-start gap-3">
          <div className="rounded-full bg-destructive/10 p-2 text-destructive">
            {isBackendDown ? <PlugZap className="size-4" /> : <AlertCircle className="size-4" />}
          </div>
          <div className="space-y-1">
            <p className="text-sm font-medium">
              {isBackendDown ? 'Backend unavailable' : 'Request failed'}
            </p>
            <p className="max-w-2xl text-xs leading-relaxed text-muted-foreground">{message}</p>
            {apiError?.code && apiError.code !== 'network_error' && (
              <p className="font-mono text-[11px] text-muted-foreground/70">
                {apiError.code}
                {apiError.status ? ` · HTTP ${apiError.status}` : ''}
              </p>
            )}
          </div>
        </div>

        {isBackendDown && (
          <pre className="overflow-x-auto rounded-md border border-border bg-muted/60 px-3 py-2 font-mono text-[11px] leading-relaxed text-muted-foreground">
{`cd backend
.venv\\Scripts\\activate      # Windows
uvicorn app.main:app --reload`}
          </pre>
        )}

        {onRetry && (
          <div>
            <Button size="sm" variant="outline" onClick={onRetry}>
              <RefreshCw className="size-3.5" />
              Retry
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
