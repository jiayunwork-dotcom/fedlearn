import { create } from "zustand";
import {
  ExperimentConfig,
  ExperimentStatus,
  RoundMetrics,
  AttackLogEntry,
  ComparisonItem,
  listExperiments,
  getStatus,
  getMetrics,
  getContributions,
  getLabelDistribution,
  getAttackLog,
  batchCompareExperiments,
} from "@/api";

interface AppState {
  experiments: ExperimentConfig[];
  currentExperimentId: string | null;
  status: ExperimentStatus | null;
  metrics: RoundMetrics[];
  contributions: Record<string, number>;
  labelDistribution: number[][];
  attackLog: AttackLogEntry[];
  pollingTimer: ReturnType<typeof setInterval> | null;

  comparisons: ComparisonItem[];
  comparisonsLoading: boolean;
  selectedCompareIds: string[];

  fetchExperiments: () => Promise<void>;
  selectExperiment: (id: string) => void;
  startPolling: (id: string) => void;
  stopPolling: () => void;
  refreshData: (id: string) => Promise<void>;
  setCurrentExperimentId: (id: string | null) => void;

  fetchComparisons: (ids: string[]) => Promise<void>;
  toggleCompareSelection: (id: string) => void;
  clearCompareSelection: () => void;
  setSelectedCompareIds: (ids: string[]) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  experiments: [],
  currentExperimentId: null,
  status: null,
  metrics: [],
  contributions: {},
  labelDistribution: [],
  attackLog: [],
  pollingTimer: null,

  comparisons: [],
  comparisonsLoading: false,
  selectedCompareIds: [],

  fetchExperiments: async () => {
    try {
      const data = await listExperiments();
      set({ experiments: data.experiments });
    } catch (e) {
      console.error("Failed to fetch experiments:", e);
    }
  },

  selectExperiment: (id: string) => {
    set({ currentExperimentId: id });
    get().refreshData(id);
  },

  startPolling: (id: string) => {
    get().stopPolling();
    const timer = setInterval(() => {
      get().refreshData(id);
    }, 3000);
    set({ pollingTimer: timer });
  },

  stopPolling: () => {
    const { pollingTimer } = get();
    if (pollingTimer) {
      clearInterval(pollingTimer);
      set({ pollingTimer: null });
    }
  },

  refreshData: async (id: string) => {
    try {
      const [statusData, metricsData, contribData, labelData, attackData] = await Promise.allSettled([
        getStatus(id),
        getMetrics(id),
        getContributions(id),
        getLabelDistribution(id),
        getAttackLog(id),
      ]);

      const updates: Partial<AppState> = {};
      if (statusData.status === "fulfilled") updates.status = statusData.value;
      if (metricsData.status === "fulfilled") updates.metrics = metricsData.value.metrics;
      if (contribData.status === "fulfilled") updates.contributions = contribData.value.contributions;
      if (labelData.status === "fulfilled") updates.labelDistribution = labelData.value.distribution;
      if (attackData.status === "fulfilled") updates.attackLog = attackData.value.attack_log;

      set(updates);

      if (statusData.status === "fulfilled" && !statusData.value.is_running) {
        get().stopPolling();
      }
    } catch (e) {
      console.error("Failed to refresh data:", e);
    }
  },

  setCurrentExperimentId: (id) => set({ currentExperimentId: id }),

  fetchComparisons: async (ids: string[]) => {
    set({ comparisonsLoading: true });
    try {
      const data = await batchCompareExperiments(ids);
      set({ comparisons: data.comparisons, comparisonsLoading: false });
    } catch (e) {
      console.error("Failed to fetch comparisons:", e);
      set({ comparisonsLoading: false });
    }
  },

  toggleCompareSelection: (id: string) => {
    const { selectedCompareIds } = get();
    if (selectedCompareIds.includes(id)) {
      set({ selectedCompareIds: selectedCompareIds.filter((x) => x !== id) });
    } else if (selectedCompareIds.length < 4) {
      set({ selectedCompareIds: [...selectedCompareIds, id] });
    }
  },

  clearCompareSelection: () => set({ selectedCompareIds: [] }),

  setSelectedCompareIds: (ids: string[]) => set({ selectedCompareIds: ids }),
}));
