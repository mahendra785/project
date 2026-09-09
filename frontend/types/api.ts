/**
 * TypeScript mirror of the backend Pydantic schemas (`backend/app/schemas/`).
 *
 * Wire format is snake_case on both sides — there is deliberately no casing
 * transformation layer, so a field name here can be grepped for in the Python
 * source and vice versa.
 *
 * See `shared/api-contracts/README.md` for the full contract.
 */

/** Honest maturity label attached to every research component. */
export type ImplementationStatus =
  | 'implemented'
  | 'baseline'
  | 'experimental'
  | 'planned';

/** Where a displayed number came from. Never render a figure without it. */
export type DataProvenance = 'computed' | 'synthetic' | 'mock';

export type TaskType = 'segmentation' | 'detection' | 'classification';
export type AnalysisStatus = 'pending' | 'running' | 'completed' | 'failed';
export type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH' | 'REVIEW_REQUIRED';
export type ShiftStatus =
  | 'ID'
  | 'MILD_SHIFT'
  | 'MODERATE_SHIFT'
  | 'SEVERE_SHIFT';
export type ReferralPriority = 'routine' | 'elevated' | 'urgent_review';

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/* -------------------------------------------------------------- system -- */

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  python: string;
  capabilities: Record<string, boolean>;
  available_models: string[];
  unavailable_models: Record<string, string>;
  /** Background warmup of the calibration/reference sets. */
  warmup: {
    status: 'cold' | 'queued' | 'warming' | 'ready' | 'failed';
    duration_s: number | null;
    error: string | null;
  };
}

/* -------------------------------------------------------------- models -- */

export interface ModelInfo {
  id: string;
  name: string;
  model_type: string;
  tasks: TaskType[];
  status: ImplementationStatus;
  available: boolean;
  unavailable_reason: string | null;
  uncertainty_method: string;
  calibration_methods: string[];
  ood_detector: string;
  /** What the weights were actually trained on. Never omit in the UI. */
  trained_on: string;
  /** Whether the weights were fitted on real medical imaging data. */
  is_medical: boolean;
  /** Always false. Trained-on-medical-data is NOT validated-for-clinical-use. */
  clinically_validated: boolean;
  /** Dataset id the adapter was trained on, when it is a trained model. */
  trained_dataset: string | null;
  description: string;
  parameters: Record<string, unknown>;
  input_channels: number;
  class_labels: string[];
}

export interface ModelListResponse {
  models: ModelInfo[];
  default_model_id: string;
}

/* ---------------------------------------------------------- prediction -- */

export interface ClassScore {
  class_id: number;
  label: string;
  probability: number;
}

export interface SegmentationMask {
  height: number;
  width: number;
  num_classes: number;
  class_labels: string[];
  /** Flattened (value, run_length) pairs, row-major. */
  rle: number[];
  class_pixel_fraction: number[];
  png_url: string | null;
}

export interface Detection {
  id: string;
  class_id: number;
  label: string;
  bbox: BoundingBox;
  confidence: number;
}

export interface PredictionResult {
  task: TaskType;
  model_id: string;
  confidence: number;
  classes: ClassScore[];
  segmentation: SegmentationMask | null;
  detections: Detection[] | null;
  inference_time_ms: number;
  metadata: Record<string, unknown>;
}

/* --------------------------------------------------------- uncertainty -- */

export interface UncertaintyStatistics {
  mean: number;
  std: number;
  minimum: number;
  maximum: number;
  median: number;
  p90: number;
  p95: number;
  p99: number;
}

/** Level 1 — whole image. */
export interface GlobalUncertainty {
  score: number;
  statistics: UncertaintyStatistics;
  high_uncertainty_fraction: number;
  aggregate_distribution_entropy: number;
  boundary_uncertainty: number;
  interior_uncertainty: number;
  method: string;
  provenance: DataProvenance;
}

/** Level 2 — regions and detections. */
export interface ObjectUncertainty {
  id: string;
  label: string;
  class_id: number | null;
  bbox: BoundingBox;
  area_px: number;
  confidence: number;
  uncertainty: number;
  peak_uncertainty: number;
  calibrated_risk: number | null;
  risk_level: RiskLevel;
  source: 'high_uncertainty_region' | 'detection';
  review_status: string;
}

/** Level 3 — the dense map, transported as PNG rather than JSON floats. */
export interface PixelUncertainty {
  height: number;
  width: number;
  encoding: string;
  data: string | null;
  heatmap_url: string | null;
  raw_url: string | null;
  threshold: number;
  high_uncertainty_mask_fraction: number;
  /** 32x32 block means, for charts that should not decode an image. */
  preview_grid: number[][];
}

export interface HierarchicalUncertainty {
  method: string;
  status: ImplementationStatus;
  global: GlobalUncertainty;
  objects: ObjectUncertainty[];
  pixel: PixelUncertainty;
  notes: string[];
}

/* --------------------------------------------------------- calibration -- */

export interface CalibrationMethodInfo {
  id: string;
  name: string;
  status: ImplementationStatus;
  description: string;
  reference: string | null;
  assumptions: string[];
}

export interface ReliabilityBin {
  bin_lower: number;
  bin_upper: number;
  count: number;
  mean_confidence: number;
  empirical_accuracy: number;
}

export interface RiskCoveragePoint {
  coverage: number;
  risk: number;
  threshold: number;
}

export interface CalibrationResult {
  method: string;
  method_name: string;
  status: ImplementationStatus;
  target_risk: number;
  /** lambda_hat for conformal methods; the fitted T for temperature scaling. */
  threshold: number;
  empirical_risk: number;
  coverage: number;
  expected_calibration_error: number;
  brier_score: number | null;
  applied_abstention_rate: number | null;
  calibrated_confidence: number | null;
  applied: boolean;
  reliability_bins: ReliabilityBin[];
  risk_coverage_curve: RiskCoveragePoint[];
  calibration_set_size: number;
  provenance: DataProvenance;
  assumptions: string[];
  validity_warnings: string[];
}

export interface CalibrationMethodsResponse {
  methods: CalibrationMethodInfo[];
  default_method: string;
  default_target_risk: number;
}

/* ------------------------------------------------------ distribution shift */

export interface ShiftIndicator {
  id: string;
  name: string;
  value: number;
  normalised: number;
  weight: number;
  description: string;
}

export interface DistributionSummary {
  name: string;
  mean: number;
  std: number;
  skewness: number;
  kurtosis: number;
  entropy: number;
  histogram: number[];
  bin_edges: number[];
}

export interface DistributionShiftResult {
  method: string;
  status_label: ImplementationStatus;
  shift_score: number;
  status: ShiftStatus;
  reference_domain: string;
  current_domain: string;
  thresholds: Record<string, number>;
  indicators: ShiftIndicator[];
  reference_distribution: DistributionSummary;
  current_distribution: DistributionSummary;
  /** Heuristic couplings of the shift score — NOT measured degradation. */
  confidence_degradation: number;
  uncertainty_increase: number;
  provenance: DataProvenance;
  notes: string[];
}

export interface ShiftConfigResponse {
  detector: string;
  detector_name: string;
  status: ImplementationStatus;
  thresholds: Record<string, number>;
  indicators: Array<{
    id: string;
    name: string;
    weight: number;
    description: string;
  }>;
  scales: Record<string, number>;
  reference_domains: Array<{
    name: string;
    description: string;
    n_samples: number;
    mean: number;
    std: number;
    entropy: number;
    histogram: number[];
  }>;
  limitations: string[];
}

/* ---------------------------------------------------------------- risk -- */

export interface RiskFactor {
  id: string;
  name: string;
  raw_value: number;
  normalised: number;
  weight: number;
  contribution: number;
  rationale: string;
}

export interface RiskAssessment {
  risk_score: number;
  risk_level: RiskLevel;
  referral_priority: ReferralPriority;
  review_recommended: boolean;
  reasons: string[];
  factors: RiskFactor[];
  weights: Record<string, number>;
  thresholds: Record<string, number>;
  triggered_overrides: string[];
  disclaimer: string;
}

/* ------------------------------------------------------------ analysis -- */

export interface AnalysisOptions {
  model_id?: string | null;
  task: TaskType;
  calibration_method: string;
  target_risk: number;
  uncertainty_threshold: number;
  reference_domain: string;
  apply_calibration: boolean;
  detect_shift: boolean;
  notes?: string | null;
}

export interface ImageMetadata {
  filename: string;
  content_type: string;
  size_bytes: number;
  original_width: number;
  original_height: number;
  original_channels: number;
  working_width: number;
  working_height: number;
  format: string;
  is_dicom: boolean;
  checksum: string;
  tags: Record<string, string>;
  preview_url: string | null;
}

export interface AnalysisMetadata {
  created_at: string;
  duration_ms: number;
  model_id: string;
  model_name: string;
  task: TaskType;
  options: AnalysisOptions;
  image: ImageMetadata;
  pipeline_version: string;
  seed: number;
}

export interface AnalysisResponse {
  analysis_id: string;
  status: AnalysisStatus;
  metadata: AnalysisMetadata;
  prediction: PredictionResult;
  uncertainty: HierarchicalUncertainty;
  calibration: CalibrationResult | null;
  distribution_shift: DistributionShiftResult | null;
  risk_assessment: RiskAssessment;
  warnings: string[];
}

export interface AnalysisSummary {
  analysis_id: string;
  created_at: string;
  filename: string;
  model_id: string;
  task: TaskType;
  status: AnalysisStatus;
  mean_uncertainty: number;
  confidence: number;
  shift_score: number | null;
  shift_status: string | null;
  risk_level: string;
  risk_score: number;
}

export interface AnalysisListResponse {
  items: AnalysisSummary[];
  total: number;
}

export interface DashboardStats {
  total_analyses: number;
  completed_analyses: number;
  failed_analyses: number;
  active_model_id: string;
  active_model_name: string;
  mean_uncertainty: number | null;
  mean_confidence: number | null;
  high_risk_count: number;
  review_required_count: number;
  shift_flagged_count: number;
  mean_shift_score: number | null;
  mean_ece: number | null;
  uncertainty_histogram: number[];
  risk_breakdown: Record<string, number>;
  shift_breakdown: Record<string, number>;
  recent: AnalysisSummary[];
  backend_available: boolean;
}

/* ---------------------------------------------------------- evaluation -- */

export interface MetricDefinition {
  id: string;
  name: string;
  category: string;
  status: ImplementationStatus;
  description: string;
  requires_ground_truth: boolean;
  higher_is_better: boolean;
}

export interface MetricValue {
  id: string;
  name: string;
  category: string;
  value: number | null;
  unit: string | null;
  sample_size: number;
  provenance: DataProvenance;
  unavailable_reason: string | null;
}

export interface EvaluationReport {
  generated_at: string;
  session_sample_size: number;
  metrics: MetricValue[];
  definitions: MetricDefinition[];
  risk_coverage_curve: Array<{
    coverage: number;
    selective_risk: number;
    threshold: number;
  }>;
  uncertainty_vs_confidence: Array<{
    analysis_id: string;
    uncertainty: number;
    confidence: number;
    shift: number | null;
    risk: number;
    risk_level: string;
  }>;
  notes: string[];
}

export interface SelfTestResponse {
  results: Array<{
    metric: string;
    scenario: string;
    expected: number;
    actual: number;
    passed: boolean;
  }>;
  all_passed: boolean;
  note: string;
}

/* --------------------------------------------------------------- error -- */

export interface ApiErrorBody {
  code: string;
  message: string;
  detail: Record<string, unknown>;
}

/* ------------------------------------------------------------- datasets -- */

export interface DatasetInfo {
  id: string;
  name: string;
  modality: string;
  target: string;
  use_case: string;
  licence: string;
  licence_url: string;
  citation: string;
  homepage: string;
  approx_mb: number;
  downloadable: boolean;
  requires_registration: boolean;
  /** Downloaded and extracted on the backend machine. */
  available: boolean;
  notes: string;
}

/** What is actually backing calibration and the reference domain right now. */
export interface ActiveDatasetInfo {
  dataset_id: string | null;
  dataset_name: string | null;
  available: boolean;
  provenance: DataProvenance;
  summary: string;
  download_command: string | null;
}

export interface DatasetsResponse {
  active: ActiveDatasetInfo;
  datasets: DatasetInfo[];
}

/* -------------------------------------------------------------- samples -- */

export interface SampleCase {
  id: string;
  image: string;
  ground_truth: string | null;
  shifted_variant: string | null;
  dataset: string;
  dataset_name: string;
  modality: string;
  target: string;
  case_id: string;
  split: string;
  foreground_fraction: number;
  licence: string;
  analysis_ids: string[];
  shifted_analysis_ids: string[];
}

export interface SamplesResponse {
  note: string;
  citation: string;
  licence: string;
  samples: SampleCase[];
}

/* ------------------------------------------------------------- research -- */

export interface ExperimentRow {
  alpha: number;
  n_cal: number;
  multi_lesion_fraction: number;
  lambda_pix: number;
  R_pix: number;
  R_les_at_lambda_pix: number;
  G: number;
  G_sd: number | null;
  R_les_test: number | null;
  lambda_les: number;
  lambda_union: number;
  inflation_union: number;
}

export interface Experiment {
  id: string;
  week: number;
  title: string;
  dataset: string;
  role: string;
  verdict_by_rule: string;
  recommendation: string;
  recommendation_reason: string;
  next: string;
  pre_registered_statistic: string;
  protocol: Record<string, unknown>;
  report: string | null;
  memo: string | null;
  rows: ExperimentRow[];
  seeds: number[];
  m_justification: string | null;
}

export interface ResearchSummary {
  dataset: ActiveDatasetInfo;
  model: {
    id: string;
    name: string;
    trained_on: string;
    trained_dataset: string | null;
    is_medical: boolean;
    clinically_validated: boolean;
    training_report: Record<string, unknown> | null;
  };
  method_status_counts: Record<string, number>;
  experiments: Experiment[];
  disclaimer: string;
}
