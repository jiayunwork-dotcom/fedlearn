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
  InterpretabilityHistoryEntry,
  InterpretabilityCompareItem,
  listExperiments,
  getStatus,
  getMetrics,
  getContributions,
  getLabelDistribution,
  getAttackLog,
  batchCompareExperiments,
  startInterpretabilityAnalysis,
  cancelInterpretabilityAnalysis,
  getInterpretabilityWebSocketUrl,
  getInterpretabilityHistory,
  batchCompareInterpretability,
  exportInterpretabilityReport,
  resumeInterpretability,
  getInterpretabilityStatus,
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
  interpretabilityCustomSamples: string;
  interpretabilityCustomSamplesError: string | null;
  interpretabilityStatus: "idle" | "running" | "completed" | "cancelled" | "error";
  interpretabilityProgress: number;
  interpretabilityCurrentSample: number;
  interpretabilityLogs: InterpretabilityLogEntry[];
  interpretabilityResult: InterpretabilityResult | null;
  interpretabilityError: string | null;
  interpretabilityWs: WebSocket | null;
  interpretabilityAnalysisTimestamp: number | null;
  selectedClassForDetail: number | null;

  interpretabilityHistory: InterpretabilityHistoryEntry[];
  interpretabilityHistoryLoading: boolean;
  selectedHistoryForCompare: Array<{ method: AnalysisMethod; num_samples: number }>;
  interpretabilityCompareData: InterpretabilityCompareItem[] | null;
  interpretabilityCompareLoading: boolean;

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
  setInterpretabilityCustomSamples: (value: string) => void;
  validateCustomSamples: () => boolean;
  getEffectiveNumSamples: () => number;
  startInterpretability: (experimentId: string) => Promise<void>;
  cancelInterpretability: (experimentId: string) => Promise<void>;
  connectInterpretabilityWs: (experimentId: string) => void;
  disconnectInterpretabilityWs: () => void;
  setSelectedClassForDetail: (classIdx: number | null) => void;
  resetInterpretability: () => void;
  loadInterpretabilityHistory: (experimentId: string) => Promise<void>;
  toggleHistoryCompareSelection: (item: { method: AnalysisMethod; num_samples: number }) => void;
  clearHistoryCompareSelection: () => void;
  performInterpretabilityCompare: (experimentId: string) => Promise<void>;
  clearInterpretabilityCompare: () => void;
  exportReport: (experimentId: string) => Promise<void>;
  checkAndResumeAnalysis: (experimentId: string) => Promise<void>;
  requestWsResume: (experimentId: string, method: AnalysisMethod, numSamples: number) => void;
}

const SAMPLE_OPTIONS = [10, 50, 100, 500];

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
  interpretabilityCustomSamples: "",
  interpretabilityCustomSamplesError: null,
  interpretabilityStatus: "idle",
  interpretabilityProgress: 0,
  interpretabilityCurrentSample: 0,
  interpretabilityLogs: [],
  interpretabilityResult: null,
  interpretabilityError: null,
  interpretabilityWs: null,
  interpretabilityAnalysisTimestamp: null,
  selectedClassForDetail: null,

  interpretabilityHistory: [],
  interpretabilityHistoryLoading: false,
  selectedHistoryForCompare: [],
  interpretabilityCompareData: null,
  interpretabilityCompareLoading: false,

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
      interpretabilityCompareData: null,
      selectedHistoryForCompare: [],
    });
  },

  setInterpretabilityMethod: (method: AnalysisMethod) => set({ interpretabilityMethod: method }),
  setInterpretabilityNumSamples: (num: number) => set({
    interpretabilityNumSamples: num,
    interpretabilityCustomSamples: "",
    interpretabilityCustomSamplesError: null,
  }),

  setInterpretabilityCustomSamples: (value: string) => {
    let error: string | null = null;
    if (value !== "") {
      if (!/^\d+$/.test(value)) {
        error = "请输入整数";
      } else {
        const n = parseInt(value, 10);
        if (n < 10) {
          error = "最少10个样本(需覆盖所有类别)";
        } else if (n > 1000) {
          error = "最多1000个样本";
        }
      }
    }
    if (error === null && value !== "") {
      set({
        interpretabilityCustomSamples: value,
        interpretabilityCustomSamplesError: null,
      });
    } else {
      set({
        interpretabilityCustomSamples: value,
        interpretabilityCustomSamplesError: error,
      });
    }
  },

  validateCustomSamples: () => {
    const { interpretabilityCustomSamples } = get();
    if (interpretabilityCustomSamples === "") return true;
    if (!/^\d+$/.test(interpretabilityCustomSamples)) return false;
    const n = parseInt(interpretabilityCustomSamples, 10);
    return n >= 10 && n <= 1000;
  },

  getEffectiveNumSamples: () => {
    const { interpretabilityCustomSamples, interpretabilityNumSamples } = get();
    if (interpretabilityCustomSamples !== "" && /^\d+$/.test(interpretabilityCustomSamples)) {
      const n = parseInt(interpretabilityCustomSamples, 10);
      if (n >= 10 && n <= 1000) return n;
    }
    return interpretabilityNumSamples;
  },

  setSelectedClassForDetail: (classIdx: number | null) => set({ selectedClassForDetail: classIdx }),

  resetInterpretability: () => set({
    interpretabilityStatus: "idle",
    interpretabilityProgress: 0,
    interpretabilityCurrentSample: 0,
    interpretabilityLogs: [],
    interpretabilityResult: null,
    interpretabilityError: null,
    interpretabilityAnalysisTimestamp: null,
    selectedClassForDetail: null,
    interpretabilityCompareData: null,
    selectedHistoryForCompare: [],
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
        const currentMethod = get().interpretabilityMethod;
        const currentSamples = get().getEffectiveNumSamples();
        try {
          ws.send(JSON.stringify({
            type: "request_resume",
            method: currentMethod,
            num_samples: currentSamples,
          }));
        } catch (e) {
          console.warn("Failed to send resume request:", e);
        }
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          const currentMethod = get().interpretabilityMethod;
          const currentSamples = get().getEffectiveNumSamples();

          if (message.type === "resume_state") {
            if (message.method === currentMethod && message.num_samples === currentSamples) {
              const updates: Partial<AppState> = {
                interpretabilityStatus: message.status === "running" ? "running" : message.status,
                interpretabilityProgress: message.progress || 0,
                interpretabilityCurrentSample: message.current_sample || 0,
                interpretabilityLogs: message.logs || [],
              };
              set(updates);
            }
          } else if (message.type === "progress") {
            if (message.method === currentMethod && message.num_samples === currentSamples) {
              const updates: Partial<AppState> = {
                interpretabilityProgress: message.progress,
                interpretabilityCurrentSample: message.current_sample,
              };
              if (message.log) {
                updates.interpretabilityLogs = [...get().interpretabilityLogs, message.log];
              }
              set(updates);
            }
          } else if (message.type === "complete") {
            if (message.method === currentMethod && message.num_samples === currentSamples) {
              const status = message.status;
              if (status === "completed") {
                set({
                  interpretabilityStatus: "completed",
                  interpretabilityProgress: 100,
                  interpretabilityAnalysisTimestamp: message.analysis_timestamp || null,
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
              get().loadInterpretabilityHistory(experimentId);
            }
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

  requestWsResume: (_experimentId: string, method: AnalysisMethod, numSamples: number) => {
    const { interpretabilityWs } = get();
    if (interpretabilityWs && interpretabilityWs.readyState === WebSocket.OPEN) {
      try {
        interpretabilityWs.send(JSON.stringify({
          type: "request_resume",
          method,
          num_samples: numSamples,
        }));
      } catch (e) {
        console.warn("Failed to send resume request:", e);
      }
    }
  },

  startInterpretability: async (experimentId: string) => {
    const { interpretabilityMethod } = get();
    const effectiveSamples = get().getEffectiveNumSamples();

    if (!get().validateCustomSamples()) {
      set({
        interpretabilityStatus: "error",
        interpretabilityError: get().interpretabilityCustomSamplesError || "无效的样本数量",
      });
      return;
    }

    set({
      interpretabilityStatus: "running",
      interpretabilityProgress: 0,
      interpretabilityCurrentSample: 0,
      interpretabilityLogs: [],
      interpretabilityResult: null,
      interpretabilityError: null,
      interpretabilityAnalysisTimestamp: null,
      selectedClassForDetail: null,
      interpretabilityCompareData: null,
      selectedHistoryForCompare: [],
    });

    try {
      get().connectInterpretabilityWs(experimentId);

      const response = await startInterpretabilityAnalysis(
        experimentId,
        interpretabilityMethod,
        effectiveSamples
      );

      if (response.analysis_timestamp) {
        set({ interpretabilityAnalysisTimestamp: response.analysis_timestamp });
      }

      if (response.cached && response.result) {
        set({
          interpretabilityStatus: "completed",
          interpretabilityProgress: 100,
          interpretabilityAnalysisTimestamp: response.analysis_timestamp || null,
          interpretabilityResult: response.result,
        });
        get().disconnectInterpretabilityWs();
        get().loadInterpretabilityHistory(experimentId);
      }
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : "Failed to start analysis";
      console.error("Failed to start interpretability analysis:", e);
      set({
        interpretabilityStatus: "error",
        interpretabilityError: message,
      });
      get().disconnectInterpretabilityWs();
    }
  },

  cancelInterpretability: async (experimentId: string) => {
    const { interpretabilityMethod } = get();
    const effectiveSamples = get().getEffectiveNumSamples();

    try {
      await cancelInterpretabilityAnalysis(
        experimentId,
        interpretabilityMethod,
        effectiveSamples
      );
      set({ interpretabilityStatus: "cancelled" });
      get().disconnectInterpretabilityWs();
      get().loadInterpretabilityHistory(experimentId);
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : "Failed to cancel analysis";
      console.error("Failed to cancel interpretability analysis:", e);
      set({
        interpretabilityStatus: "error",
        interpretabilityError: message,
      });
    }
  },

  loadInterpretabilityHistory: async (experimentId: string) => {
    set({ interpretabilityHistoryLoading: true });
    try {
      const data = await getInterpretabilityHistory(experimentId);
      set({
        interpretabilityHistory: data.analyses,
        interpretabilityHistoryLoading: false,
      });
    } catch (e) {
      console.error("Failed to load interpretability history:", e);
      set({ interpretabilityHistoryLoading: false });
    }
  },

  toggleHistoryCompareSelection: (item: { method: AnalysisMethod; num_samples: number }) => {
    const { selectedHistoryForCompare } = get();
    const idx = selectedHistoryForCompare.findIndex(
      (x) => x.method === item.method && x.num_samples === item.num_samples
    );
    if (idx >= 0) {
      set({ selectedHistoryForCompare: selectedHistoryForCompare.filter((_, i) => i !== idx) });
    } else if (selectedHistoryForCompare.length < 3) {
      set({ selectedHistoryForCompare: [...selectedHistoryForCompare, item] });
    }
  },

  clearHistoryCompareSelection: () => set({
    selectedHistoryForCompare: [],
    interpretabilityCompareData: null,
  }),

  performInterpretabilityCompare: async (experimentId: string) => {
    const { selectedHistoryForCompare, interpretabilityHistory } = get();
    if (selectedHistoryForCompare.length < 2 || selectedHistoryForCompare.length > 3) return;

    set({ interpretabilityCompareLoading: true });
    try {
      const items = selectedHistoryForCompare.map((sel) => {
        const hist = interpretabilityHistory.find(
          (h) => h.method === sel.method && h.num_samples === sel.num_samples
        );
        return {
          method: sel.method,
          num_samples: sel.num_samples,
          status: hist?.status || "completed",
          analysis_timestamp: hist?.analysis_timestamp,
        };
      });
      const data = await batchCompareInterpretability(experimentId, items);
      set({
        interpretabilityCompareData: data.comparisons,
        interpretabilityCompareLoading: false,
      });
    } catch (e) {
      console.error("Failed to perform interpretability compare:", e);
      set({ interpretabilityCompareLoading: false });
    }
  },

  clearInterpretabilityCompare: () => set({
    interpretabilityCompareData: null,
  }),

  exportReport: async (experimentId: string) => {
    const { interpretabilityMethod, interpretabilityResult } = get();
    const effectiveSamples = get().getEffectiveNumSamples();
    try {
      await exportInterpretabilityReport(
        experimentId,
        interpretabilityMethod,
        effectiveSamples
      );
    } catch (e) {
      console.error("Failed to export report:", e);
      if (interpretabilityResult) {
        try {
          const ts = Date.now();
          const blob = new Blob([JSON.stringify({
            experiment_id: experimentId,
            analysis_method: interpretabilityMethod,
            num_samples: effectiveSamples,
            analysis_timestamp: ts / 1000,
            global_feature_importance: interpretabilityResult.overall_attribution,
            client_contributions: interpretabilityResult.client_contributions,
          }, null, 2)], { type: "application/json" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `${experimentId}_${interpretabilityMethod}_${effectiveSamples}_${Math.floor(ts/1000)}.json`;
          a.click();
          URL.revokeObjectURL(url);
        } catch (e2) {
          console.error("Fallback export also failed:", e2);
        }
      }
    }
  },

  checkAndResumeAnalysis: async (experimentId: string) => {
    try {
      const data = await resumeInterpretability(experimentId);
      if (data.running_analyses && data.running_analyses.length > 0) {
        const currentMethod = get().interpretabilityMethod;
        const currentSamples = get().getEffectiveNumSamples();
        const match = data.running_analyses.find(
          (r) => r.method === currentMethod && r.num_samples === currentSamples
        );
        if (match) {
          set({
            interpretabilityStatus: match.status === "running" ? "running" : "idle",
            interpretabilityProgress: match.progress || 0,
            interpretabilityCurrentSample: match.current_sample || 0,
            interpretabilityLogs: match.logs || [],
          });
          get().connectInterpretabilityWs(experimentId);
          setTimeout(() => {
            get().requestWsResume(experimentId, currentMethod, currentSamples);
          }, 300);
          return;
        }
        const first = data.running_analyses[0];
        set({
          interpretabilityMethod: first.method,
          interpretabilityNumSamples: SAMPLE_OPTIONS.includes(first.num_samples) ? first.num_samples : get().interpretabilityNumSamples,
          interpretabilityCustomSamples: SAMPLE_OPTIONS.includes(first.num_samples) ? "" : String(first.num_samples),
          interpretabilityCustomSamplesError: null,
          interpretabilityStatus: first.status === "running" ? "running" : "idle",
          interpretabilityProgress: first.progress || 0,
          interpretabilityCurrentSample: first.current_sample || 0,
          interpretabilityLogs: first.logs || [],
        });
        get().connectInterpretabilityWs(experimentId);
        setTimeout(() => {
          get().requestWsResume(experimentId, first.method, first.num_samples);
        }, 300);
      } else {
        try {
          const currentMethod = get().interpretabilityMethod;
          const currentSamples = get().getEffectiveNumSamples();
          const statusData = await getInterpretabilityStatus(
            experimentId, currentMethod, currentSamples
          ) as any;
          if (statusData && statusData.status === "completed" && !statusData.cached) {
          } else if (statusData && statusData.status === "completed" && statusData.cached) {
          }
        } catch (_) {
        }
      }
    } catch (e) {
      console.error("Failed to check and resume analysis:", e);
    }
  },
}));
