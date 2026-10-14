'use client';

/** TanStack Query hooks. All server state flows through these. */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { ApiError } from '@/lib/api/client';
import { api } from '@/lib/api/endpoints';
import { useAnalysisStore } from '@/stores/analysis-store';
import type { AnalysisOptions, AnalysisResponse } from '@/types/api';

export const queryKeys = {
  health: ['health'] as const,
  models: ['models'] as const,
  dashboard: ['dashboard'] as const,
  analyses: (limit: number) => ['analyses', limit] as const,
  analysis: (id: string) => ['analysis', id] as const,
  calibrationMethods: ['calibration', 'methods'] as const,
  calibrationRun: (model: string, method: string, alpha: number) =>
    ['calibration', 'run', model, method, alpha] as const,
  shiftConfig: ['shift', 'config'] as const,
  evaluation: (withCalibration: boolean) => ['evaluation', withCalibration] as const,
  selfTest: ['evaluation', 'self-test'] as const,
  datasets: ['datasets'] as const,
  samples: ['samples'] as const,
  research: ['research', 'summary'] as const,
  mgcrc: ['research', 'mgcrc'] as const,
};

/** Do not retry contract mismatches or 4xx — only transient failures. */
function retryPolicy(failureCount: number, error: unknown) {
  if (error instanceof ApiError) {
    if (error.code === 'contract_mismatch') return false;
    if (error.status >= 400 && error.status < 500) return false;
  }
  return failureCount < 2;
}

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: api.health,
    // Backend warmup state changes on its own; poll while it settles.
    refetchInterval: (query) =>
      query.state.data?.warmup.status === 'ready' ? 60_000 : 5_000,
    retry: retryPolicy,
    staleTime: 3_000,
  });
}

export function useModels() {
  return useQuery({
    queryKey: queryKeys.models,
    queryFn: api.models,
    staleTime: 5 * 60_000,
    retry: retryPolicy,
  });
}

export function useDashboard() {
  return useQuery({
    queryKey: queryKeys.dashboard,
    queryFn: api.dashboard,
    staleTime: 10_000,
    retry: retryPolicy,
  });
}

export function useAnalyses(limit = 25) {
  return useQuery({
    queryKey: queryKeys.analyses(limit),
    queryFn: () => api.analyses(limit),
    staleTime: 10_000,
    retry: retryPolicy,
  });
}

export function useAnalysis(id: string | null) {
  return useQuery({
    queryKey: queryKeys.analysis(id ?? 'none'),
    queryFn: () => api.analysis(id as string),
    enabled: Boolean(id),
    staleTime: 5 * 60_000,
    retry: retryPolicy,
  });
}

export function useCalibrationMethods() {
  return useQuery({
    queryKey: queryKeys.calibrationMethods,
    queryFn: api.calibrationMethods,
    staleTime: 10 * 60_000,
    retry: retryPolicy,
  });
}

export function useCalibrationRun(params: {
  modelId: string;
  method: string;
  targetRisk: number;
  enabled: boolean;
}) {
  return useQuery({
    queryKey: queryKeys.calibrationRun(params.modelId, params.method, params.targetRisk),
    queryFn: () =>
      api.runCalibration({
        modelId: params.modelId,
        method: params.method,
        targetRisk: params.targetRisk,
      }),
    enabled: params.enabled,
    // The calibration set is cached server-side, so repeats are cheap; still,
    // there is no reason to refetch an already-computed deterministic result.
    staleTime: Infinity,
    retry: retryPolicy,
  });
}

export function useShiftConfig() {
  return useQuery({
    queryKey: queryKeys.shiftConfig,
    queryFn: api.shiftConfig,
    staleTime: 10 * 60_000,
    retry: retryPolicy,
  });
}

export function useEvaluation(includeCalibrationRun = true) {
  return useQuery({
    queryKey: queryKeys.evaluation(includeCalibrationRun),
    queryFn: () => api.evaluation(includeCalibrationRun),
    staleTime: 60_000,
    retry: retryPolicy,
  });
}

export function useDatasets() {
  return useQuery({
    queryKey: queryKeys.datasets,
    queryFn: api.datasets,
    staleTime: 5 * 60_000,
    retry: retryPolicy,
  });
}

export function useSamples() {
  return useQuery({
    queryKey: queryKeys.samples,
    queryFn: api.samples,
    staleTime: 30_000,
    retry: retryPolicy,
  });
}

export function useResearchSummary() {
  return useQuery({
    queryKey: queryKeys.research,
    queryFn: api.researchSummary,
    staleTime: 5 * 60_000,
    retry: retryPolicy,
  });
}

export function useMgcrc() {
  return useQuery({
    queryKey: queryKeys.mgcrc,
    queryFn: api.mgcrc,
    staleTime: 5 * 60_000,
    retry: retryPolicy,
  });
}

/** Run a bundled sample server-side; refreshes everything that lists analyses. */
export function useRunSample() {
  const queryClient = useQueryClient();
  const setCurrent = useAnalysisStore((s) => s.setCurrent);
  return useMutation<AnalysisResponse, ApiError, string>({
    mutationFn: (filename) => api.analyzeSample(filename),
    onSuccess: (analysis) => {
      setCurrent(analysis);
      queryClient.setQueryData(queryKeys.analysis(analysis.analysis_id), analysis);
      queryClient.invalidateQueries({ queryKey: queryKeys.samples });
      queryClient.invalidateQueries({ queryKey: queryKeys.dashboard });
      queryClient.invalidateQueries({ queryKey: ['analyses'] });
      queryClient.invalidateQueries({ queryKey: ['evaluation'] });
    },
  });
}

export function useSelfTest() {
  return useQuery({
    queryKey: queryKeys.selfTest,
    queryFn: api.selfTest,
    staleTime: 10 * 60_000,
    retry: retryPolicy,
  });
}

export function useRunAnalysis() {
  const queryClient = useQueryClient();
  const setCurrent = useAnalysisStore((s) => s.setCurrent);

  return useMutation<AnalysisResponse, ApiError, { file: File; options: Partial<AnalysisOptions> }>(
    {
      mutationFn: ({ file, options }) => api.analyze(file, options),
      onSuccess: (analysis) => {
        setCurrent(analysis);
        queryClient.setQueryData(queryKeys.analysis(analysis.analysis_id), analysis);
        queryClient.invalidateQueries({ queryKey: queryKeys.dashboard });
        queryClient.invalidateQueries({ queryKey: ['analyses'] });
        queryClient.invalidateQueries({ queryKey: ['evaluation'] });
      },
    },
  );
}

export function useDeleteAnalysis() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteAnalysis(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['analyses'] });
      queryClient.invalidateQueries({ queryKey: queryKeys.dashboard });
    },
  });
}
