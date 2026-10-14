/**
 * Typed endpoint functions. Components never call `fetch` or build URLs.
 *
 * Each function validates its response with the matching Zod schema, so a
 * contract drift fails loudly at the boundary with a field-level message.
 */

import { ANALYSIS_TIMEOUT_MS, ApiError, request } from './client';
import * as s from './schemas';
import type {
  MgcrcSummary,
  AnalysisListResponse,
  DatasetsResponse,
  ResearchSummary,
  SamplesResponse,
  AnalysisOptions,
  AnalysisResponse,
  CalibrationMethodsResponse,
  CalibrationResult,
  DashboardStats,
  DistributionShiftResult,
  EvaluationReport,
  HealthResponse,
  HierarchicalUncertainty,
  ModelListResponse,
  RiskAssessment,
  SelfTestResponse,
  ShiftConfigResponse,
} from '@/types/api';
import type { z } from 'zod';

function parse<T extends z.ZodTypeAny>(schema: T, data: unknown, endpoint: string) {
  const result = schema.safeParse(data);
  if (!result.success) {
    const first = result.error.issues[0];
    throw new ApiError(
      `Unexpected response shape from ${endpoint}: ${first.path.join('.') || '(root)'} — ${first.message}`,
      { code: 'contract_mismatch' },
    );
  }
  return result.data;
}

export const api = {
  async health(): Promise<HealthResponse> {
    const data = await request<unknown>('/health', { timeoutMs: 5000 });
    return parse(s.healthSchema, data, '/health') as HealthResponse;
  },

  async models(): Promise<ModelListResponse> {
    const data = await request<unknown>('/api/models');
    return parse(s.modelListSchema, data, '/api/models') as ModelListResponse;
  },

  async analyze(file: File, options: Partial<AnalysisOptions>): Promise<AnalysisResponse> {
    const form = new FormData();
    form.append('file', file);
    form.append('options', JSON.stringify(options));

    const data = await request<unknown>('/api/analyze', {
      method: 'POST',
      body: form,
      timeoutMs: ANALYSIS_TIMEOUT_MS,
    });
    return parse(s.analysisSchema, data, '/api/analyze') as unknown as AnalysisResponse;
  },

  async analysis(id: string): Promise<AnalysisResponse> {
    const data = await request<unknown>(`/api/analyses/${id}`);
    return parse(s.analysisSchema, data, '/api/analyses/{id}') as unknown as AnalysisResponse;
  },

  async analyses(limit = 25, offset = 0): Promise<AnalysisListResponse> {
    const data = await request<unknown>(`/api/analyses?limit=${limit}&offset=${offset}`);
    return parse(s.analysisListSchema, data, '/api/analyses') as unknown as AnalysisListResponse;
  },

  async deleteAnalysis(id: string): Promise<void> {
    await request<void>(`/api/analyses/${id}`, { method: 'DELETE' });
  },

  async uncertainty(id: string): Promise<HierarchicalUncertainty> {
    const data = await request<unknown>(`/api/analyses/${id}/uncertainty`);
    return parse(s.uncertaintySchema, data, 'uncertainty') as unknown as HierarchicalUncertainty;
  },

  async calibrationFor(id: string): Promise<CalibrationResult | null> {
    const data = await request<unknown>(`/api/analyses/${id}/calibration`);
    return data === null
      ? null
      : (parse(s.calibrationSchema, data, 'calibration') as unknown as CalibrationResult);
  },

  async shiftFor(id: string): Promise<DistributionShiftResult | null> {
    const data = await request<unknown>(`/api/analyses/${id}/shift`);
    return data === null
      ? null
      : (parse(s.shiftSchema, data, 'shift') as unknown as DistributionShiftResult);
  },

  async riskFor(id: string): Promise<RiskAssessment> {
    const data = await request<unknown>(`/api/analyses/${id}/risk`);
    return parse(s.riskSchema, data, 'risk') as unknown as RiskAssessment;
  },

  async dashboard(): Promise<DashboardStats> {
    const data = await request<unknown>('/api/dashboard');
    return parse(s.dashboardSchema, data, '/api/dashboard') as unknown as DashboardStats;
  },

  async calibrationMethods(): Promise<CalibrationMethodsResponse> {
    const data = await request<unknown>('/api/calibration/methods');
    return parse(
      s.calibrationMethodsSchema,
      data,
      '/api/calibration/methods',
    ) as CalibrationMethodsResponse;
  },

  async runCalibration(params: {
    modelId?: string;
    method: string;
    targetRisk: number;
  }): Promise<CalibrationResult> {
    const query = new URLSearchParams({
      method: params.method,
      target_risk: String(params.targetRisk),
    });
    if (params.modelId) query.set('model_id', params.modelId);

    const data = await request<unknown>(`/api/calibration/run?${query}`, {
      method: 'POST',
      timeoutMs: ANALYSIS_TIMEOUT_MS,
    });
    return parse(s.calibrationSchema, data, '/api/calibration/run') as unknown as CalibrationResult;
  },

  async shiftConfig(): Promise<ShiftConfigResponse> {
    const data = await request<unknown>('/api/shift/config');
    return parse(s.shiftConfigSchema, data, '/api/shift/config') as ShiftConfigResponse;
  },

  async evaluation(includeCalibrationRun = true): Promise<EvaluationReport> {
    const data = await request<unknown>(
      `/api/evaluation?include_calibration_run=${includeCalibrationRun}`,
      { timeoutMs: ANALYSIS_TIMEOUT_MS },
    );
    return parse(s.evaluationSchema, data, '/api/evaluation') as unknown as EvaluationReport;
  },

  async datasets(): Promise<DatasetsResponse> {
    const data = await request<unknown>('/api/datasets');
    return parse(s.datasetsSchema, data, '/api/datasets') as unknown as DatasetsResponse;
  },

  async samples(): Promise<SamplesResponse> {
    const data = await request<unknown>('/api/samples');
    return parse(s.samplesSchema, data, '/api/samples') as SamplesResponse;
  },

  /** Run the full pipeline on a bundled sample, server-side (no upload). */
  async analyzeSample(filename: string): Promise<AnalysisResponse> {
    const data = await request<unknown>(`/api/samples/${encodeURIComponent(filename)}/analyze`, {
      method: 'POST',
      timeoutMs: ANALYSIS_TIMEOUT_MS,
    });
    return parse(s.analysisSchema, data, '/api/samples/{f}/analyze') as unknown as AnalysisResponse;
  },

  async researchSummary(): Promise<ResearchSummary> {
    const data = await request<unknown>('/api/research/summary', { timeoutMs: 60_000 });
    return parse(s.researchSummarySchema, data, '/api/research/summary') as unknown as ResearchSummary;
  },

  async mgcrc(): Promise<MgcrcSummary> {
    const data = await request<unknown>('/api/research/mgcrc', { timeoutMs: 60_000 });
    return parse(s.mgcrcSummarySchema, data, '/api/research/mgcrc') as MgcrcSummary;
  },

  async selfTest(): Promise<SelfTestResponse> {
    const data = await request<unknown>('/api/evaluation/self-test');
    return parse(s.selfTestSchema, data, '/api/evaluation/self-test') as SelfTestResponse;
  },
};
