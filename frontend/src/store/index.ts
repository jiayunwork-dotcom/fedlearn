import { create } from "zustand";
import {
  ExperimentConfig,
  ExperimentStatus,
  RoundMetrics,
  AttackLogEntry,
  ComparisonItem,
  AnalysisMethod,
  InterpretabilityResult,
  InterpretabilityLogEntry,
  listExperiments,
  getStatus,
  getMetrics,
  getContributions,
  getLabelDistribution,
  getAttackLog,
  batchCompareExperiments,
  startInterpretabilityAnalysis,
  cancelInterpretabilityAnalysis,
  getInterpretabilityResult,
  getInterpretabilityWebSocketUrl,
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

  interpretabilityPanelOpen: boolean;
  interpretabilityMethod: AnalysisMethod;
  interpretabilityNumSamples: number;
  interpretabilityStatus: "idle" | "running" | "completed" | "cancelled" | "error";
  interpretabilityProgress: number;
  interpretabilityCurrentSample: number;
  interpretabilityLogs: InterpretabilityLogEntry[];
  interpretabilityResult: InterpretabilityResult | null;
  interpretabilityError: string | null;
  interpretabilityWs: WebSocket | null;
  selectedClassForDetail: number | null;

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

  openInterpretabilityPanel: () => void;
  closeInterpretabilityPanel: () => void;
  setInterpretabilityMethod: (method: AnalysisMethod) => void;
  setInterpretabilityNumSamples: (num: number) => void;
  startInterpretability: (experimentId: string) => Promise<void>;
  cancelInterpretability: (experimentId: string) => Promise<void>;
  connectInterpretabilityWs: (experimentId: string) => void;
  disconnectInterpretabilityWs: () => void;
  setSelectedClassForDetail: (classIdx: number | null) => void;
  resetInterpretability: () => void;
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

  interpretabilityPanelOpen: false,
  interpretabilityMethod: "gradient",
  interpretabilityNumSamples: 50,
  interpretabilityStatus: "idle",
  interpretabilityProgress: 0,
  interpretabilityCurrentSample: 0,
  interpretabilityLogs: [],
  interpretabilityResult: null,
  interpretabilityError: null,
  interpretabilityWs: null,
  selectedClassForDetail: null,

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

  openInterpretabilityPanel: () => set({ interpretabilityPanelOpen: true }),
  closeInterpretabilityPanel: () => {
    get().disconnectInterpretabilityWs();
    set({
      interpretabilityPanelOpen: false,
      selectedClassForDetail: null,
    });
  },

  setInterpretabilityMethod: (method: AnalysisMethod) => set({ interpretabilityMethod: method }),
  setInterpretabilityNumSamples: (num: number) => set({ interpretabilityNumSamples: num }),
  setSelectedClassForDetail: (classIdx: number | null) => set({ selectedClassForDetail: classIdx }),

  resetInterpretability: () => set({
    interpretabilityStatus: "idle",
    interpretabilityProgress: 0,
    interpretabilityCurrentSample: 0,
    interpretabilityLogs: [],
    interpretabilityResult: null,
    interpretabilityError: null,
    selectedClassForDetail: null,
  }),

  connectInterpretabilityWs: (experimentId: string) => {
    const { interpretabilityWs } = get();
    if (interpretabilityWs && interpretabilityWs.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const wsUrl = getInterpretabilityWebSocketUrl(experimentId);
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log("Interpretability WebSocket connected");
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);

          if (message.type === "progress") {
            const updates: Partial<AppState> = {
              interpretabilityProgress: message.progress,
              interpretabilityCurrentSample: message.current_sample,
            };
            if (message.log) {
              updates.interpretabilityLogs = [...get().interpretabilityLogs, message.log];
            }
            set(updates);
          } else if (message.type === "complete") {
            const status = message.status;
            if (status === "completed") {
              set({
                interpretabilityStatus: "completed",
                interpretabilityProgress: 100,
                interpretabilityResult: message.result,
              });
            } else if (status === "cancelled") {
              set({ interpretabilityStatus: "cancelled" });
            } else if (status === "failed") {
              set({
                interpretabilityStatus: "error",
                interpretabilityError: message.result?.error || "Analysis failed",
              });
            }
            get().disconnectInterpretabilityWs();
          }
        } catch (e) {
          console.error("Failed to parse WebSocket message:", e);
        }
      };

      ws.onerror = (error) => {
        console.error("Interpretability WebSocket error:", error);
        set({
          interpretabilityStatus: "error",
          interpretabilityError: "WebSocket connection error",
        });
      };

      ws.onclose = () => {
        console.log("Interpretability WebSocket disconnected");
        set({ interpretabilityWs: null });
      };

      set({ interpretabilityWs: ws });
    } catch (e) {
      console.error("Failed to connect WebSocket:", e);
      set({
        interpretabilityStatus: "error",
        interpretabilityError: "Failed to connect to server",
      });
    }
  },

  disconnectInterpretabilityWs: () => {
    const { interpretabilityWs } = get();
    if (interpretabilityWs) {
      interpretabilityWs.close();
      set({ interpretabilityWs: null });
    }
  },

  startInterpretability: async (experimentId: string) => {
    const { interpretabilityMethod, interpretabilityNumSamples } = get();

    set({
      interpretabilityStatus: "running",
      interpretabilityProgress: 0,
      interpretabilityCurrentSample: 0,
      interpretabilityLogs: [],
      interpretabilityResult: null,
      interpretabilityError: null,
      selectedClassForDetail: null,
    });

    try {
      get().connectInterpretabilityWs(experimentId);

      const response = await startInterpretabilityAnalysis(
        experimentId,
        interpretabilityMethod,
        interpretabilityNumSamples
      );

      if (response.cached && response.result) {
        set({
          interpretabilityStatus: "completed",
          interpretabilityProgress: 100,
          interpretabilityResult: response.result,
        });
        get().disconnectInterpretabilityWs();
      }
    } catch (e: any) {
      console.error("Failed to start interpretability analysis:", e);
      set({
        interpretabilityStatus: "error",
        interpretabilityError: e.message || "Failed to start analysis",
      });
      get().disconnectInterpretabilityWs();
    }
  },

  cancelInterpretability: async (experimentId: string) => {
    const { interpretabilityMethod, interpretabilityNumSamples } = get();

    try {
      await cancelInterpretabilityAnalysis(
        experimentId,
        interpretabilityMethod,
        interpretabilityNumSamples
      );
      set({ interpretabilityStatus: "cancelled" });
      get().disconnectInterpretabilityWs();
    } catch (e: any) {
      console.error("Failed to cancel interpretability analysis:", e);
      set({
        interpretabilityStatus: "error",
        interpretabilityError: e.message || "Failed to cancel analysis",
      });
    }
  },
}));
