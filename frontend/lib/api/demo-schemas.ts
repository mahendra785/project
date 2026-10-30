/**
 * Zod schemas for the live lesion-level CRC demo (backend/app/api/routes/demo.py).
 *
 * Kept free of path aliases, like schemas.ts, so tests/live-demo-contract.test.mts can
 * validate real captured backend responses against them.
 */

import { z } from 'zod';

const num = z.number().nullable();

const methodInfo = z.object({
  id: z.string(),
  label: z.string(),
  family: z.string().optional(),
  guaranteed: z.array(z.string()),
  guarantee: z.string(),
  ours: z.boolean(),
});

export const demoStatusSchema = z.union([
  z.object({ available: z.literal(false), reason: z.string() }),
  z.object({
    available: z.literal(true),
    dataset: z.string(),
    arch: z.string(),
    seed: z.number(),
    size: z.number(),
    n_units: z.number(),
    n_multi: z.number(),
    n_small: z.number(),
    n_lesions: z.number(),
    small_cut: z.number(),
    tau: z.number(),
    split: z.array(z.number()),
    pool_metrics: z.record(z.string(), z.number()),
    best_val_dice: z.number(),
    epochs: z.number(),
    env: z.record(z.string(), z.unknown()),
    methods: z.array(methodInfo),
  }),
]);

const lesionSchema = z.object({
  index: z.number(),
  size: z.number(),
  small: z.boolean(),
  stat: z.number(),
  peak: z.number(),
  found: z.record(z.string(), z.boolean()),
});

const unitSchema = z.object({
  uid: z.number(),
  id: z.string(),
  k: z.number(),
  multi: z.boolean(),
  lesions: z.array(lesionSchema),
  cutoffs: z.record(z.string(), z.number()),
  rescued: z.number(),
  missed_pixel: z.number(),
  missed_joint: z.number(),
  gt_pixels: z.number(),
  set_pixels: z.record(z.string(), z.number()),
  bg_q99: z.number(),
  significance: z.number(),
});

const risks = z.object({ pix: num, les: num, small: num, multi: num, set_to_truth: num, bg_to_truth: num });

export const demoRunSchema = z.object({
  alpha: z.number(),
  seed: z.number(),
  n_tune: z.number(),
  n_cal: z.number(),
  n_test: z.number(),
  eligible_cal: z.record(z.string(), z.number()),
  grid_size: z.number(),
  methods: z.array(
    methodInfo.extend({
      feasible: z.boolean(),
      lam: z.array(z.number()),
      cutoff: z.array(z.number()),
      test: risks,
    }),
  ),
  curves: z.record(z.string(), z.object({ n: z.number(), corrected: z.array(z.number()), lam_hat: z.number() })),
  units: z.array(unitSchema),
});

const stat = z.object({ mean: num, p05: num, p95: num, frac_above: num });

export const demoValiditySchema = z.object({
  alpha: z.number(),
  reps: z.number(),
  methods: z.array(
    methodInfo.extend({
      stats: z.object({
        pix: stat,
        les: stat,
        small: stat,
        multi: stat,
        set_to_truth: stat,
        rel_vs_joint: stat,
      }),
    }),
  ),
});

export const demoRescuesSchema = z.object({
  alpha: z.number(),
  splits: z.number(),
  units: z.array(
    z.object({
      uid: z.number(),
      id: z.string(),
      k: z.number(),
      multi: z.boolean(),
      lesions: z.array(z.number()),
      small: z.boolean(),
      stat: z.number(),
      n_test: z.number(),
      n_rescued: z.number(),
      seed: z.number(),
    }),
  ),
});

export type DemoRescues = z.infer<typeof demoRescuesSchema>;
export type DemoStatus = z.infer<typeof demoStatusSchema>;
export type DemoRun = z.infer<typeof demoRunSchema>;
export type DemoUnit = z.infer<typeof unitSchema>;
export type DemoMethod = DemoRun['methods'][number];
export type DemoValidity = z.infer<typeof demoValiditySchema>;
export type RiskKey = 'pix' | 'les' | 'small' | 'multi';
