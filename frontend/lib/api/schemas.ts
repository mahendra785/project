/**
 * Runtime validation of API responses.
 *
 * TypeScript types disappear at runtime, so a contract drift between the
 * Pydantic schemas and this app would surface as a confusing render error deep
 * in a component. Validating at the boundary turns that into one clear message
 * naming the offending field.
 *
 * Schemas are intentionally permissive about *extra* fields (Zod strips them by
 * default) and strict about the ones the UI actually renders — especially the
 * research-integrity fields (`provenance`, `status`, `assumptions`), which must
 * never silently go missing.
 */

import { z } from 'zod';

export const implementationStatusSchema = z.enum([
  'implemented',
  'baseline',
  'experimental',
  'planned',
]);
export const provenanceSchema = z.enum(['computed', 'synthetic', 'mock']);
export const taskTypeSchema = z.enum(['segmentation', 'detection', 'classification']);
export const riskLevelSchema = z.enum(['LOW', 'MODERATE', 'HIGH', 'REVIEW_REQUIRED']);
export const shiftStatusSchema = z.enum([
  'ID',
  'MILD_SHIFT',
  'MODERATE_SHIFT',
  'SEVERE_SHIFT',
]);

const unitInterval = z.number().min(0).max(1);

export const boundingBoxSchema = z.object({
  x: z.number(),
  y: z.number(),
  width: z.number(),
  height: z.number(),
});

export const healthSchema = z.object({
  status: z.string(),
  version: z.string(),
  environment: z.string(),
  python: z.string(),
  capabilities: z.record(z.boolean()),
  available_models: z.array(z.string()),
  unavailable_models: z.record(z.string()),
  warmup: z.object({
    status: z.enum(['cold', 'queued', 'warming', 'ready', 'failed']),
    duration_s: z.number().nullable(),
    error: z.string().nullable(),
  }),
});

export const modelInfoSchema = z.object({
  id: z.string(),
  name: z.string(),
  model_type: z.string(),
  tasks: z.array(taskTypeSchema),
  status: implementationStatusSchema,
  available: z.boolean(),
  unavailable_reason: z.string().nullable(),
  uncertainty_method: z.string(),
  calibration_methods: z.array(z.string()),
  ood_detector: z.string(),
  trained_on: z.string(),
  is_medical: z.boolean(),
  clinically_validated: z.boolean(),
  trained_dataset: z.string().nullable(),
  description: z.string(),
  parameters: z.record(z.unknown()),
  input_channels: z.number(),
  class_labels: z.array(z.string()),
});

export const modelListSchema = z.object({
  models: z.array(modelInfoSchema),
  default_model_id: z.string(),
});

export const predictionSchema = z.object({
  task: taskTypeSchema,
  model_id: z.string(),
  confidence: unitInterval,
  classes: z.array(
    z.object({ class_id: z.number(), label: z.string(), probability: z.number() }),
  ),
  segmentation: z
    .object({
      height: z.number(),
      width: z.number(),
      num_classes: z.number(),
      class_labels: z.array(z.string()),
      rle: z.array(z.number()),
      class_pixel_fraction: z.array(z.number()),
      png_url: z.string().nullable(),
    })
    .nullable(),
  detections: z
    .array(
      z.object({
        id: z.string(),
        class_id: z.number(),
        label: z.string(),
        bbox: boundingBoxSchema,
        confidence: unitInterval,
      }),
    )
    .nullable(),
  inference_time_ms: z.number(),
  metadata: z.record(z.unknown()),
});

export const uncertaintySchema = z.object({
  method: z.string(),
  status: implementationStatusSchema,
  global: z.object({
    score: unitInterval,
    statistics: z.object({
      mean: z.number(),
      std: z.number(),
      minimum: z.number(),
      maximum: z.number(),
      median: z.number(),
      p90: z.number(),
      p95: z.number(),
      p99: z.number(),
    }),
    high_uncertainty_fraction: unitInterval,
    aggregate_distribution_entropy: z.number(),
    boundary_uncertainty: z.number(),
    interior_uncertainty: z.number(),
    method: z.string(),
    provenance: provenanceSchema,
  }),
  objects: z.array(
    z.object({
      id: z.string(),
      label: z.string(),
      class_id: z.number().nullable(),
      bbox: boundingBoxSchema,
      area_px: z.number(),
      confidence: z.number(),
      uncertainty: unitInterval,
      peak_uncertainty: unitInterval,
      calibrated_risk: z.number().nullable(),
      risk_level: riskLevelSchema,
      source: z.string(),
      review_status: z.string(),
    }),
  ),
  pixel: z.object({
    height: z.number(),
    width: z.number(),
    encoding: z.string(),
    data: z.string().nullable(),
    heatmap_url: z.string().nullable(),
    raw_url: z.string().nullable(),
    threshold: z.number(),
    high_uncertainty_mask_fraction: z.number(),
    preview_grid: z.array(z.array(z.number())),
  }),
  notes: z.array(z.string()),
});

export const calibrationSchema = z.object({
  method: z.string(),
  method_name: z.string(),
  status: implementationStatusSchema,
  target_risk: z.number(),
  threshold: z.number(),
  empirical_risk: z.number(),
  coverage: z.number(),
  expected_calibration_error: z.number(),
  brier_score: z.number().nullable(),
  applied_abstention_rate: z.number().nullable(),
  calibrated_confidence: z.number().nullable(),
  applied: z.boolean(),
  reliability_bins: z.array(
    z.object({
      bin_lower: z.number(),
      bin_upper: z.number(),
      count: z.number(),
      mean_confidence: z.number(),
      empirical_accuracy: z.number(),
    }),
  ),
  risk_coverage_curve: z.array(
    z.object({ coverage: z.number(), risk: z.number(), threshold: z.number() }),
  ),
  calibration_set_size: z.number(),
  provenance: provenanceSchema,
  assumptions: z.array(z.string()),
  validity_warnings: z.array(z.string()),
});

export const shiftSchema = z.object({
  method: z.string(),
  status_label: implementationStatusSchema,
  shift_score: unitInterval,
  status: shiftStatusSchema,
  reference_domain: z.string(),
  current_domain: z.string(),
  thresholds: z.record(z.number()),
  indicators: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      value: z.number(),
      normalised: z.number(),
      weight: z.number(),
      description: z.string(),
    }),
  ),
  reference_distribution: z.object({
    name: z.string(),
    mean: z.number(),
    std: z.number(),
    skewness: z.number(),
    kurtosis: z.number(),
    entropy: z.number(),
    histogram: z.array(z.number()),
    bin_edges: z.array(z.number()),
  }),
  current_distribution: z.object({
    name: z.string(),
    mean: z.number(),
    std: z.number(),
    skewness: z.number(),
    kurtosis: z.number(),
    entropy: z.number(),
    histogram: z.array(z.number()),
    bin_edges: z.array(z.number()),
  }),
  confidence_degradation: z.number(),
  uncertainty_increase: z.number(),
  provenance: provenanceSchema,
  notes: z.array(z.string()),
});

export const riskSchema = z.object({
  risk_score: unitInterval,
  risk_level: riskLevelSchema,
  referral_priority: z.enum(['routine', 'elevated', 'urgent_review']),
  review_recommended: z.boolean(),
  reasons: z.array(z.string()),
  factors: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      raw_value: z.number(),
      normalised: z.number(),
      weight: z.number(),
      contribution: z.number(),
      rationale: z.string(),
    }),
  ),
  weights: z.record(z.number()),
  thresholds: z.record(z.number()),
  triggered_overrides: z.array(z.string()),
  disclaimer: z.string(),
});

export const analysisSchema = z.object({
  analysis_id: z.string(),
  status: z.enum(['pending', 'running', 'completed', 'failed']),
  metadata: z.object({
    created_at: z.string(),
    duration_ms: z.number(),
    model_id: z.string(),
    model_name: z.string(),
    task: taskTypeSchema,
    options: z.record(z.unknown()),
    image: z.object({
      filename: z.string(),
      content_type: z.string(),
      size_bytes: z.number(),
      original_width: z.number(),
      original_height: z.number(),
      original_channels: z.number(),
      working_width: z.number(),
      working_height: z.number(),
      format: z.string(),
      is_dicom: z.boolean(),
      checksum: z.string(),
      tags: z.record(z.string()),
      preview_url: z.string().nullable(),
    }),
    pipeline_version: z.string(),
    seed: z.number(),
  }),
  prediction: predictionSchema,
  uncertainty: uncertaintySchema,
  calibration: calibrationSchema.nullable(),
  distribution_shift: shiftSchema.nullable(),
  risk_assessment: riskSchema,
  warnings: z.array(z.string()),
});

export const analysisSummarySchema = z.object({
  analysis_id: z.string(),
  created_at: z.string(),
  filename: z.string(),
  model_id: z.string(),
  task: taskTypeSchema,
  status: z.string(),
  mean_uncertainty: z.number(),
  confidence: z.number(),
  shift_score: z.number().nullable(),
  shift_status: z.string().nullable(),
  risk_level: z.string(),
  risk_score: z.number(),
});

export const analysisListSchema = z.object({
  items: z.array(analysisSummarySchema),
  total: z.number(),
});

export const dashboardSchema = z.object({
  total_analyses: z.number(),
  completed_analyses: z.number(),
  failed_analyses: z.number(),
  active_model_id: z.string(),
  active_model_name: z.string(),
  mean_uncertainty: z.number().nullable(),
  mean_confidence: z.number().nullable(),
  high_risk_count: z.number(),
  review_required_count: z.number(),
  shift_flagged_count: z.number(),
  mean_shift_score: z.number().nullable(),
  mean_ece: z.number().nullable(),
  uncertainty_histogram: z.array(z.number()),
  risk_breakdown: z.record(z.number()),
  shift_breakdown: z.record(z.number()),
  recent: z.array(analysisSummarySchema),
  backend_available: z.boolean(),
});

export const calibrationMethodsSchema = z.object({
  methods: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      status: implementationStatusSchema,
      description: z.string(),
      reference: z.string().nullable(),
      assumptions: z.array(z.string()),
    }),
  ),
  default_method: z.string(),
  default_target_risk: z.number(),
});

export const shiftConfigSchema = z.object({
  detector: z.string(),
  detector_name: z.string(),
  status: implementationStatusSchema,
  thresholds: z.record(z.number()),
  indicators: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      weight: z.number(),
      description: z.string(),
    }),
  ),
  scales: z.record(z.number()),
  reference_domains: z.array(
    z.object({
      name: z.string(),
      description: z.string(),
      n_samples: z.number(),
      mean: z.number(),
      std: z.number(),
      entropy: z.number(),
      histogram: z.array(z.number()),
    }),
  ),
  limitations: z.array(z.string()),
});

export const evaluationSchema = z.object({
  generated_at: z.string(),
  session_sample_size: z.number(),
  metrics: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      category: z.string(),
      value: z.number().nullable(),
      unit: z.string().nullable(),
      sample_size: z.number(),
      provenance: provenanceSchema,
      unavailable_reason: z.string().nullable(),
    }),
  ),
  definitions: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      category: z.string(),
      status: implementationStatusSchema,
      description: z.string(),
      requires_ground_truth: z.boolean(),
      higher_is_better: z.boolean(),
    }),
  ),
  risk_coverage_curve: z.array(
    z.object({
      coverage: z.number(),
      selective_risk: z.number(),
      threshold: z.number(),
    }),
  ),
  uncertainty_vs_confidence: z.array(z.record(z.unknown())),
  notes: z.array(z.string()),
});

export const selfTestSchema = z.object({
  results: z.array(
    z.object({
      metric: z.string(),
      scenario: z.string(),
      expected: z.number(),
      actual: z.number(),
      passed: z.boolean(),
    }),
  ),
  all_passed: z.boolean(),
  note: z.string(),
});

export const datasetsSchema = z.object({
  active: z.object({
    dataset_id: z.string().nullable(),
    dataset_name: z.string().nullable(),
    available: z.boolean(),
    provenance: provenanceSchema,
    summary: z.string(),
    download_command: z.string().nullable(),
  }),
  datasets: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      modality: z.string(),
      target: z.string(),
      use_case: z.string(),
      licence: z.string(),
      licence_url: z.string(),
      citation: z.string(),
      homepage: z.string(),
      approx_mb: z.number(),
      downloadable: z.boolean(),
      requires_registration: z.boolean(),
      available: z.boolean(),
      notes: z.string(),
    }),
  ),
});
