'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useTheme } from 'next-themes';
import { ChevronRight, Loader2, Moon, Sun } from 'lucide-react';

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/primitives';
import { Button } from '@/components/ui/button';
import { useHealth } from '@/hooks/use-pums';
import { findNavItem } from '@/lib/navigation';
import { cn } from '@/lib/utils';

/** Live backend indicator — makes "is the API up?" answerable at a glance. */
function BackendStatus() {
  const { data, isLoading, isError } = useHealth();

  if (isLoading) {
    return (
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Loader2 className="size-3 animate-spin" />
        Connecting…
      </span>
    );
  }

  if (isError || !data) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="flex cursor-help items-center gap-1.5 text-xs text-status-danger">
            <span className="size-1.5 rounded-full bg-status-danger" />
            Backend offline
          </span>
        </TooltipTrigger>
        <TooltipContent>
          Start it with <code>uvicorn app.main:app --reload</code> in{' '}
          <code>backend/</code>. Pages will show an explicit error rather than
          silently switching to mock data.
        </TooltipContent>
      </Tooltip>
    );
  }

  const warming = data.warmup.status === 'warming' || data.warmup.status === 'queued';

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className={cn(
            'flex cursor-help items-center gap-1.5 text-xs',
            warming ? 'text-status-warn' : 'text-status-ok',
          )}
        >
          <span
            className={cn(
              'size-1.5 rounded-full',
              warming ? 'animate-pulse bg-status-warn' : 'bg-status-ok',
            )}
          />
          {warming ? 'Warming up' : 'Backend online'}
          <span className="text-muted-foreground">v{data.version}</span>
        </span>
      </TooltipTrigger>
      <TooltipContent>
        {warming ? (
          <>
            Building the synthetic calibration set and reference profile. The
            first analysis will be slower until this finishes.
          </>
        ) : (
          <>
            Python {data.python} · {data.available_models.length} model adapter(s)
            available
            {data.warmup.duration_s !== null && (
              <> · warmed in {data.warmup.duration_s}s</>
            )}
          </>
        )}
      </TooltipContent>
    </Tooltip>
  );
}

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();

  // Both icons are rendered and swapped by the `dark` class next-themes puts on
  // <html>. That avoids the usual mounted-flag effect: the server and client
  // emit identical markup, so there is no hydration mismatch to guard against.
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label="Toggle theme"
      onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
    >
      <Moon className="size-4 dark:hidden" />
      <Sun className="hidden size-4 dark:block" />
    </Button>
  );
}

export function Header() {
  const pathname = usePathname();
  const item = findNavItem(pathname);

  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-border bg-card px-5">
      <div className="min-w-0">
        <nav aria-label="Breadcrumb" className="flex items-center gap-1 text-xs text-muted-foreground">
          <Link href="/dashboard" className="hover:text-foreground">
            PUMS
          </Link>
          {item && (
            <>
              <ChevronRight className="size-3" />
              <span className="text-foreground">{item.label}</span>
            </>
          )}
        </nav>
        {item && (
          <p className="mt-0.5 truncate text-[11px] text-muted-foreground">
            {item.description}
          </p>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-4">
        <BackendStatus />
        <ThemeToggle />
      </div>
    </header>
  );
}
