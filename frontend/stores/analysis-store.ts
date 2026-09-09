'use client';

/**
 * Client state for the analysis workflow.
 *
 * Server data (analyses, models, metrics) lives in TanStack Query, not here.
 * This store holds only what the *user* is doing: the current selection, the
 * viewer's layer settings, and the analysis they most recently ran — so the
 * Uncertainty and Decision-support pages can pick up where /analyze left off.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import type { AnalysisResponse, TaskType } from '@/types/api';

export interface ViewerSettings {
  showOriginal: boolean;
  showMask: boolean;
  showHeatmap: boolean;
  showRegions: boolean;
  heatmapOpacity: number;
  maskOpacity: number;
  /** Client-side threshold applied to the decoded uncertainty PNG. */
  uncertaintyThreshold: number;
  sideBySide: boolean;
  zoom: number;
  panX: number;
  panY: number;
}

export const DEFAULT_VIEWER: ViewerSettings = {
  showOriginal: true,
  showMask: false,
  showHeatmap: true,
  showRegions: true,
  heatmapOpacity: 0.65,
  maskOpacity: 0.5,
  uncertaintyThreshold: 0.55,
  sideBySide: false,
  zoom: 1,
  panX: 0,
  panY: 0,
};

export interface AnalysisConfig {
  modelId: string | null;
  task: TaskType;
  calibrationMethod: string;
  targetRisk: number;
  uncertaintyThreshold: number;
  referenceDomain: string;
  applyCalibration: boolean;
  detectShift: boolean;
}

export const DEFAULT_CONFIG: AnalysisConfig = {
  modelId: null,
  task: 'segmentation',
  calibrationMethod: 'conformal_risk_control',
  targetRisk: 0.1,
  uncertaintyThreshold: 0.55,
  referenceDomain: 'synthetic_reference_v1',
  applyCalibration: true,
  detectShift: true,
};

interface AnalysisState {
  /** The most recent completed analysis, shared across pages. */
  current: AnalysisResponse | null;
  /** Id selected on the Uncertainty / Decision-support pages. */
  selectedId: string | null;
  config: AnalysisConfig;
  viewer: ViewerSettings;
  selectedObjectId: string | null;

  setCurrent: (analysis: AnalysisResponse | null) => void;
  setSelectedId: (id: string | null) => void;
  updateConfig: (patch: Partial<AnalysisConfig>) => void;
  resetConfig: () => void;
  updateViewer: (patch: Partial<ViewerSettings>) => void;
  resetViewer: () => void;
  setSelectedObject: (id: string | null) => void;
}

export const useAnalysisStore = create<AnalysisState>()(
  persist(
    (set) => ({
      current: null,
      selectedId: null,
      config: DEFAULT_CONFIG,
      viewer: DEFAULT_VIEWER,
      selectedObjectId: null,

      setCurrent: (analysis) =>
        set({
          current: analysis,
          selectedId: analysis?.analysis_id ?? null,
          selectedObjectId: null,
        }),
      setSelectedId: (id) => set({ selectedId: id, selectedObjectId: null }),
      updateConfig: (patch) => set((s) => ({ config: { ...s.config, ...patch } })),
      resetConfig: () => set({ config: DEFAULT_CONFIG }),
      updateViewer: (patch) => set((s) => ({ viewer: { ...s.viewer, ...patch } })),
      resetViewer: () => set({ viewer: DEFAULT_VIEWER }),
      setSelectedObject: (id) => set({ selectedObjectId: id }),
    }),
    {
      name: 'pums-analysis',
      // Only persist user preferences. The analysis payload is server data and
      // is refetched by id, so persisting it would risk showing a stale result.
      partialize: (state) => ({
        config: state.config,
        viewer: state.viewer,
        selectedId: state.selectedId,
      }),
    },
  ),
);
