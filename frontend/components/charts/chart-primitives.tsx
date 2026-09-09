'use client';

/**
 * Shared Recharts configuration.
 *
 * Charts read their colours from the CSS custom properties defined in
 * globals.css, so light/dark switching needs no chart-level logic and the
 * semantic colours stay consistent with the badges.
 */

import {
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts';

export const CHART_COLOURS = {
  primary: 'var(--primary)',
  ok: 'var(--status-ok)',
  info: 'var(--status-info)',
  warn: 'var(--status-warn)',
  danger: 'var(--status-danger)',
  muted: 'var(--muted-foreground)',
  reference: 'var(--muted-foreground)',
};

export const axisProps = {
  stroke: 'var(--muted-foreground)',
  fontSize: 10,
  tickLine: false,
  axisLine: { stroke: 'var(--border)' },
} as const;

export function ChartFrame({
  children,
  height = 220,
}: {
  children: React.ReactElement;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      {children}
    </ResponsiveContainer>
  );
}

export function ChartGrid() {
  return <CartesianGrid strokeDasharray="2 3" stroke="var(--border)" vertical={false} />;
}

export function ChartTooltip(props: React.ComponentProps<typeof RechartsTooltip>) {
  return (
    <RechartsTooltip
      contentStyle={{
        background: 'var(--popover)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        fontSize: 11,
        padding: '6px 10px',
        color: 'var(--popover-foreground)',
      }}
      labelStyle={{ color: 'var(--muted-foreground)', marginBottom: 2 }}
      cursor={{ stroke: 'var(--border)' }}
      {...props}
    />
  );
}

export { XAxis, YAxis };
