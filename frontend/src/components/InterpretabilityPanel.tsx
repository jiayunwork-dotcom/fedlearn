import { useEffect, useMemo, useRef, type ComponentType } from "react";
import {
  X, Play, Square, BarChart3, Brain, Zap, Target, Download,
  GitCompare, Loader2, type LucideProps
} from "lucide-react";
import { useAppStore } from "@/store";
import type { AnalysisMethod, InterpretabilityLogEntry } from "@/api";
import { FeatureImportanceHeatmap } from "./FeatureImportanceHeatmap";
import { ClassAttributionComparison } from "./ClassAttributionComparison";
import { ClientContributionAttribution } from "./ClientContributionAttribution";

interface Props {
  experimentId: string;
  onClose: () => void;
}

const METHOD_OPTIONS: Array<{ value: AnalysisMethod; label: string; description: string; icon: ComponentType<LucideProps> }> = [
  {
    value: "gradient",
    label: "梯度归因",
    description: "通过反向传播计算输入梯度衡量特征重要性",
    icon: Brain,
  },
  {
    value: "permutation",
    label: "置换重要性",
    description: "随机置换单特征后观察精度下降衡量重要性",
    icon: Shuffle,
  },
  {
    value: "shap",
    label: "SHAP近似值",
    description: "通过采样近似计算Shapley值进行特征归因",
    icon: Target,
  },
];

const SAMPLE_OPTIONS = [10, 50, 100, 500];

function Shuffle(props: LucideProps) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M16 3h5v5" />
      <path d="M4 20 21 3" />
      <path d="M21 16v5h-5" />
      <path d="m15 15 6 6" />
      <path d="M4 4l5 5" />
    </svg>
  );
}

function getTopKFeatures(attribution: number[][][], k: number): Array<{ idx: number; x: number; y: number; value: number }> {
  if (!attribution || !attribution[0]) return [];
  const grid = attribution[0];
  const flat: Array<{ idx: number; x: number; y: number; value: number }> = [];
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) {
      flat.push({ idx: y * 28 + x, x, y, value: grid[y][x] });
    }
  }
  flat.sort((a, b) => b.value - a.value);
  return flat.slice(0, k);
}

export function InterpretabilityPanel({ experimentId, onClose }: Props) {
  const logContainerRef = useRef<HTMLDivElement>(null);

  const interpretabilityMethod = useAppStore((s) => s.interpretabilityMethod);
  const interpretabilityNumSamples = useAppStore((s) => s.interpretabilityNumSamples);
  const interpretabilityCustomSamples = useAppStore((s) => s.interpretabilityCustomSamples);
  const interpretabilityCustomSamplesError = useAppStore((s) => s.interpretabilityCustomSamplesError);
  const interpretabilityStatus = useAppStore((s) => s.interpretabilityStatus);
  const interpretabilityProgress = useAppStore((s) => s.interpretabilityProgress);
  const interpretabilityCurrentSample = useAppStore((s) => s.interpretabilityCurrentSample);
  const interpretabilityLogs = useAppStore((s) => s.interpretabilityLogs);
  const interpretabilityResult = useAppStore((s) => s.interpretabilityResult);
  const interpretabilityError = useAppStore((s) => s.interpretabilityError);
  const selectedClassForDetail = useAppStore((s) => s.selectedClassForDetail);
  const interpretabilityHistory = useAppStore((s) => s.interpretabilityHistory);
  const interpretabilityHistoryLoading = useAppStore((s) => s.interpretabilityHistoryLoading);
  const selectedHistoryForCompare = useAppStore((s) => s.selectedHistoryForCompare);
  const interpretabilityCompareData = useAppStore((s) => s.interpretabilityCompareData);
  const interpretabilityCompareLoading = useAppStore((s) => s.interpretabilityCompareLoading);

  const setInterpretabilityMethod = useAppStore((s) => s.setInterpretabilityMethod);
  const setInterpretabilityNumSamples = useAppStore((s) => s.setInterpretabilityNumSamples);
  const setInterpretabilityCustomSamples = useAppStore((s) => s.setInterpretabilityCustomSamples);
  const startInterpretability = useAppStore((s) => s.startInterpretability);
  const cancelInterpretability = useAppStore((s) => s.cancelInterpretability);
  const setSelectedClassForDetail = useAppStore((s) => s.setSelectedClassForDetail);
  const resetInterpretability = useAppStore((s) => s.resetInterpretability);
  const loadInterpretabilityHistory = useAppStore((s) => s.loadInterpretabilityHistory);
  const toggleHistoryCompareSelection = useAppStore((s) => s.toggleHistoryCompareSelection);
  const clearHistoryCompareSelection = useAppStore((s) => s.clearHistoryCompareSelection);
  const performInterpretabilityCompare = useAppStore((s) => s.performInterpretabilityCompare);
  const clearInterpretabilityCompare = useAppStore((s) => s.clearInterpretabilityCompare);
  const exportReport = useAppStore((s) => s.exportReport);
  const checkAndResumeAnalysis = useAppStore((s) => s.checkAndResumeAnalysis);
  const getEffectiveNumSamples = useAppStore((s) => s.getEffectiveNumSamples);

  const effectiveNumSamples = getEffectiveNumSamples();

  useEffect(() => {
    loadInterpretabilityHistory(experimentId);
    checkAndResumeAnalysis(experimentId);
  }, [experimentId]);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [interpretabilityLogs]);

  const isRunning = interpretabilityStatus === "running";
  const showProgress = isRunning;
  const showResults = interpretabilityStatus === "completed" && interpretabilityResult;

  const handleStart = () => {
    startInterpretability(experimentId);
  };

  const handleCancel = () => {
    cancelInterpretability(experimentId);
  };

  const handleReset = () => {
    resetInterpretability();
  };

  const handleExport = () => {
    exportReport(experimentId);
  };

  const handleCompare = () => {
    performInterpretabilityCompare(experimentId);
  };

  const formatLogEntry = (log: InterpretabilityLogEntry) => {
    const time = new Date(log.timestamp * 1000).toLocaleTimeString();
    const topFeatures = log.top_features.map((f) => {
      const x = Math.floor(f / 28);
      const y = f % 28;
      return `(${x},${y})`;
    }).join(", ");

    if ("sample_start" in log && log.sample_start !== undefined) {
      return `[${time}] 批次 ${log.batch}/${log.total_batches} | 样本 ${log.sample_start}-${log.sample_end} | 耗时 ${log.batch_time_ms}ms | Top-3特征: ${topFeatures}`;
    } else if ("feature_index" in log && log.feature_index !== undefined) {
      return `[${time}] 批次 ${log.batch}/${log.total_batches} | 特征 ${log.feature_index} (${log.feature_coord?.join(",")}) | 精度下降 ${(log.accuracy_drop || 0).toFixed(4)} | 耗时 ${log.batch_time_ms}ms | Top-3: ${topFeatures}`;
    }
    return `[${time}] 批次 ${log.batch}/${log.total_batches} | 耗时 ${log.batch_time_ms}ms | Top-3特征: ${topFeatures}`;
  };

  const getMethodLabel = (method: string) => {
    return METHOD_OPTIONS.find((m) => m.value === method)?.label || method;
  };

  const formatTimestamp = (ts?: number) => {
    if (!ts) return "—";
    return new Date(ts * 1000).toLocaleString();
  };

  const historyCompleted = useMemo(
    () => interpretabilityHistory.filter((h) => h.status === "completed" || h.cached),
    [interpretabilityHistory]
  );

  const isCustomMode = interpretabilityCustomSamples !== "";

  const top10CompareRows = useMemo(() => {
    if (!interpretabilityCompareData || interpretabilityCompareData.length < 2) return null;
    const validData = interpretabilityCompareData.filter((c) => c.overall_attribution && !c.error);
    if (validData.length < 2) return null;

    const topFeaturesSet = new Set<number>();
    validData.forEach((c) => {
      const top10 = getTopKFeatures(c.overall_attribution!, 10);
      top10.forEach((f) => topFeaturesSet.add(f.idx));
    });

    const rows = Array.from(topFeaturesSet).map((idx) => {
      const y = Math.floor(idx / 28);
      const x = idx % 28;
      const values = validData.map((c) => {
        if (!c.overall_attribution || !c.overall_attribution[0]) return 0;
        return c.overall_attribution[0][y][x] || 0;
      });
      const max = Math.max(...values);
      const min = Math.min(...values);
      const diff = max - min;
      return { idx, x, y, values, maxDiff: diff, maxIdx: values.indexOf(max) };
    });

    rows.sort((a, b) => b.maxDiff - a.maxDiff);
    return {
      validData,
      rows: rows.slice(0, 10),
    };
  }, [interpretabilityCompareData]);

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-6">
      <div className="w-full max-w-7xl max-h-[90vh] bg-[#0a0e1a] rounded-2xl border border-gray-800 shadow-2xl flex flex-col overflow-hidden">
        <div className="flex items-center justify-between p-5 border-b border-gray-800 bg-gradient-to-r from-cyan-500/10 to-blue-600/10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
              <BarChart3 size={20} className="text-white" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-100">模型可解释性分析</h2>
              <p className="text-xs text-gray-500">实验 {experimentId}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {showResults && (
              <button
                onClick={handleExport}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-600 text-white text-sm font-medium hover:from-emerald-600 hover:to-teal-700 transition-all shadow-lg shadow-emerald-500/20"
              >
                <Download size={16} />
                导出报告
              </button>
            )}
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-gray-800 transition-colors text-gray-400 hover:text-gray-300"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
            <div className="text-sm font-semibold text-gray-300 mb-4 flex items-center gap-2">
              <Zap size={16} className="text-cyan-400" />
              控制区
            </div>

            <div className="grid grid-cols-2 gap-6 mb-5">
              <div>
                <label className="text-xs text-gray-400 mb-2 block">分析方法</label>
                <div className="space-y-2">
                  {METHOD_OPTIONS.map((option) => {
                    const Icon = option.icon;
                    const isSelected = interpretabilityMethod === option.value;
                    return (
                      <button
                        key={option.value}
                        onClick={() => !isRunning && setInterpretabilityMethod(option.value)}
                        disabled={isRunning}
                        className={`w-full p-3 rounded-lg border text-left transition-all flex items-start gap-3 ${
                          isSelected
                            ? "border-cyan-500/50 bg-cyan-500/10"
                            : "border-gray-700 bg-[#0a0e1a] hover:border-gray-600"
                        } ${isRunning ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
                      >
                        <div
                          className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
                            isSelected ? "bg-cyan-500/20 text-cyan-400" : "bg-gray-700 text-gray-400"
                          }`}
                        >
                          <Icon size={16} />
                        </div>
                        <div>
                          <div
                            className={`text-sm font-medium ${
                              isSelected ? "text-cyan-400" : "text-gray-300"
                            }`}
                          >
                            {option.label}
                          </div>
                          <div className="text-xs text-gray-500 mt-0.5">{option.description}</div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <label className="text-xs text-gray-400 mb-2 block">分析样本数量</label>
                <div className="space-y-2">
                  {SAMPLE_OPTIONS.map((num) => {
                    const isSelected = !isCustomMode && interpretabilityNumSamples === num;
                    return (
                      <button
                        key={num}
                        onClick={() => !isRunning && setInterpretabilityNumSamples(num)}
                        disabled={isRunning}
                        className={`w-full p-3 rounded-lg border text-left transition-all ${
                          isSelected
                            ? "border-cyan-500/50 bg-cyan-500/10"
                            : "border-gray-700 bg-[#0a0e1a] hover:border-gray-600"
                        } ${isRunning ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
                      >
                        <div
                          className={`text-sm font-medium ${
                            isSelected ? "text-cyan-400" : "text-gray-300"
                          }`}
                        >
                          {num} 个样本
                        </div>
                        <div className="text-xs text-gray-500 mt-0.5">
                          每类别至少 {Math.max(1, Math.floor(num / 10))} 个样本（分层采样）
                        </div>
                      </button>
                    );
                  })}

                  <div
                    className={`p-3 rounded-lg border transition-all ${
                      isCustomMode
                        ? interpretabilityCustomSamplesError
                          ? "border-red-500/50 bg-red-500/10"
                          : "border-cyan-500/50 bg-cyan-500/10"
                        : "border-gray-700 bg-[#0a0e1a] hover:border-gray-600"
                    } ${isRunning ? "opacity-50" : ""}`}
                  >
                    <label className="block">
                      <span className={`text-sm font-medium ${
                        isCustomMode
                          ? interpretabilityCustomSamplesError
                            ? "text-red-400"
                            : "text-cyan-400"
                          : "text-gray-300"
                      }`}>
                        自定义样本数量
                      </span>
                      <div className="mt-2">
                        <input
                          type="text"
                          inputMode="numeric"
                          pattern="[0-9]*"
                          value={interpretabilityCustomSamples}
                          onChange={(e) => {
                            const v = e.target.value.replace(/[^\d]/g, "");
                            !isRunning && setInterpretabilityCustomSamples(v);
                          }}
                          placeholder="输入 10-1000 的整数"
                          disabled={isRunning}
                          className={`w-full px-3 py-2 rounded-lg bg-[#0a0e1a] border text-sm text-gray-200 outline-none transition-all ${
                            interpretabilityCustomSamplesError
                              ? "border-red-500/70 focus:border-red-500"
                              : isCustomMode
                                ? "border-cyan-500/50 focus:border-cyan-500"
                                : "border-gray-600 focus:border-gray-500"
                          } disabled:opacity-50 disabled:cursor-not-allowed`}
                        />
                      </div>
                      {interpretabilityCustomSamplesError ? (
                        <div className="mt-1.5 text-xs text-red-400 flex items-start gap-1">
                          <span>⚠ {interpretabilityCustomSamplesError}</span>
                        </div>
                      ) : (
                        <div className="mt-1.5 text-xs text-gray-500">
                          10-1000 之间，每类别至少 1 个样本
                        </div>
                      )}
                    </label>
                  </div>
                </div>

                <div className="mt-4 p-3 rounded-lg bg-[#0a0e1a] border border-gray-700">
                  <div className="text-xs text-gray-500">
                    当前选择：<span className="text-cyan-400 font-medium">{getMethodLabel(interpretabilityMethod)}</span> × <span className="text-cyan-400 font-medium">{effectiveNumSamples} 样本</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3 pt-4 border-t border-gray-700">
              {!isRunning && interpretabilityStatus !== "completed" && (
                <button
                  onClick={handleStart}
                  disabled={interpretabilityCustomSamplesError !== null}
                  className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white text-sm font-medium hover:from-cyan-600 hover:to-blue-700 transition-all shadow-lg shadow-cyan-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Play size={16} />
                  开始分析
                </button>
              )}

              {isRunning && (
                <button
                  onClick={handleCancel}
                  className="flex items-center gap-2 px-6 py-2.5 rounded-lg border border-red-500/30 text-red-400 text-sm font-medium hover:bg-red-500/10 transition-colors"
                >
                  <Square size={16} />
                  取消分析
                </button>
              )}

              {interpretabilityStatus === "completed" && (
                <button
                  onClick={handleReset}
                  className="flex items-center gap-2 px-6 py-2.5 rounded-lg border border-gray-600 text-gray-300 text-sm font-medium hover:bg-gray-700/50 transition-colors"
                >
                  <Zap size={16} />
                  重新分析
                </button>
              )}

              {interpretabilityStatus === "cancelled" && (
                <button
                  onClick={handleReset}
                  className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white text-sm font-medium hover:from-cyan-600 hover:to-blue-700 transition-all"
                >
                  <Play size={16} />
                  重新开始
                </button>
              )}

              {interpretabilityStatus === "error" && (
                <>
                  <div className="text-sm text-red-400 flex-1">{interpretabilityError}</div>
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white text-sm font-medium hover:from-cyan-600 hover:to-blue-700 transition-all"
                  >
                    <Play size={16} />
                    重试
                  </button>
                </>
              )}

              {interpretabilityStatus === "cancelled" && (
                <span className="text-sm text-yellow-500">分析已取消</span>
              )}
            </div>
          </div>

          {showProgress && (
            <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
              <div className="text-sm font-semibold text-gray-300 mb-4 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                进度区 - {getMethodLabel(interpretabilityMethod)}
              </div>

              <div className="mb-4">
                <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
                  <span>分析进度</span>
                  <span className="font-mono text-cyan-400">{interpretabilityProgress.toFixed(1)}%</span>
                </div>
                <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-blue-600 transition-all duration-300 rounded-full"
                    style={{ width: `${interpretabilityProgress}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500 mt-2">
                  已处理 {interpretabilityCurrentSample} 个样本/特征
                </div>
              </div>

              <div>
                <div className="text-xs text-gray-400 mb-2">实时日志</div>
                <div
                  ref={logContainerRef}
                  className="h-48 overflow-y-auto bg-[#0a0e1a] rounded-lg border border-gray-700 p-3 font-mono text-xs"
                >
                  {interpretabilityLogs.length === 0 ? (
                    <div className="text-gray-600">等待分析开始...</div>
                  ) : (
                    interpretabilityLogs.map((log, idx) => (
                      <div key={idx} className="text-gray-400 py-0.5">
                        {formatLogEntry(log)}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          )}

          {showResults && (
            <div className="space-y-5">
              <FeatureImportanceHeatmap
                attribution={interpretabilityResult!.overall_attribution}
                title="特征重要性热力图（总体）"
                showLegend={true}
              />

              <ClassAttributionComparison
                classAttributions={interpretabilityResult!.class_attributions}
                classSampleCounts={interpretabilityResult!.class_sample_counts}
                selectedClass={selectedClassForDetail}
                onSelectClass={setSelectedClassForDetail}
              />

              <ClientContributionAttribution
                clientContributions={interpretabilityResult!.client_contributions}
              />

              <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
                <div className="text-sm font-semibold text-gray-300 mb-4 flex items-center gap-2">
                  <GitCompare size={16} className="text-purple-400" />
                  历史分析对比
                  {interpretabilityHistoryLoading && (
                    <Loader2 size={14} className="animate-spin text-gray-500" />
                  )}
                </div>

                {historyCompleted.length === 0 ? (
                  <div className="text-sm text-gray-500 py-8 text-center">
                    暂无历史分析记录，完成至少一次分析后可进行对比
                  </div>
                ) : (
                  <>
                    <div className="text-xs text-gray-400 mb-3">
                      勾选 2-3 条已完成的记录进行横向对比（已选 {selectedHistoryForCompare.length}/3）
                    </div>

                    <div className="grid gap-2 mb-4 max-h-56 overflow-y-auto pr-2">
                      {historyCompleted.map((h, idx) => {
                        const key = `${h.method}-${h.num_samples}`;
                        const isSelected = selectedHistoryForCompare.some(
                          (x) => x.method === h.method && x.num_samples === h.num_samples
                        );
                        const isCurrent =
                          h.method === interpretabilityMethod &&
                          h.num_samples === effectiveNumSamples &&
                          interpretabilityStatus === "completed";
                        return (
                          <label
                            key={`${key}-${idx}`}
                            className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                              isSelected
                                ? "border-purple-500/50 bg-purple-500/10"
                                : "border-gray-700 bg-[#0a0e1a] hover:border-gray-600"
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => toggleHistoryCompareSelection({
                                method: h.method,
                                num_samples: h.num_samples,
                              })}
                              disabled={!isSelected && selectedHistoryForCompare.length >= 3}
                              className="w-4 h-4 rounded accent-purple-500"
                            />
                            <div className="flex-1 grid grid-cols-3 gap-3 text-xs">
                              <div>
                                <span className="text-gray-500">方法：</span>
                                <span className={`font-medium ${
                                  isCurrent ? "text-cyan-400" : "text-gray-200"
                                }`}>
                                  {getMethodLabel(h.method)}
                                  {isCurrent && (
                                    <span className="ml-1 px-1.5 py-0.5 text-[10px] rounded bg-cyan-500/20 text-cyan-300">
                                      当前
                                    </span>
                                  )}
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-500">样本：</span>
                                <span className="font-medium text-gray-200">{h.num_samples}</span>
                              </div>
                              <div>
                                <span className="text-gray-500">时间：</span>
                                <span className="font-medium text-gray-200">
                                  {formatTimestamp(h.analysis_timestamp)}
                                </span>
                              </div>
                            </div>
                          </label>
                        );
                      })}
                    </div>

                    <div className="flex items-center gap-3 pt-3 border-t border-gray-700">
                      <button
                        onClick={handleCompare}
                        disabled={
                          selectedHistoryForCompare.length < 2 ||
                          interpretabilityCompareLoading
                        }
                        className="flex items-center gap-2 px-5 py-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-600 text-white text-sm font-medium hover:from-purple-600 hover:to-pink-700 transition-all shadow-lg shadow-purple-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {interpretabilityCompareLoading ? (
                          <Loader2 size={16} className="animate-spin" />
                        ) : (
                          <GitCompare size={16} />
                        )}
                        开始对比
                      </button>
                      {interpretabilityCompareData && (
                        <button
                          onClick={clearHistoryCompareSelection}
                          className="flex items-center gap-2 px-4 py-2 rounded-lg border border-gray-600 text-gray-300 text-sm hover:bg-gray-700/50 transition-colors"
                        >
                          清除对比
                        </button>
                      )}
                      <button
                        onClick={() => clearInterpretabilityCompare()}
                        className="text-xs text-gray-500 hover:text-gray-400 transition-colors"
                      >
                        关闭对比视图
                      </button>
                    </div>
                  </>
                )}

                {interpretabilityCompareData && top10CompareRows && (
                  <div className="mt-5 pt-5 border-t border-gray-700 space-y-5">
                    <div className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
                      对比视图
                    </div>

                    <div className={`grid gap-4 ${
                      top10CompareRows.validData.length === 2 ? "grid-cols-2" : "grid-cols-3"
                    }`}>
                      {top10CompareRows.validData.map((c, i) => (
                        <div key={i}>
                          <div className="text-xs text-gray-400 mb-2 flex items-center gap-2">
                            <span className="px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 font-medium">
                              {String.fromCharCode(65 + i)}
                            </span>
                            {getMethodLabel(c.method)} × {c.num_samples}样本
                          </div>
                          {c.overall_attribution && (
                            <FeatureImportanceHeatmap
                              attribution={c.overall_attribution}
                              showLegend={false}
                              thumbnail={false}
                            />
                          )}
                        </div>
                      ))}
                    </div>

                    <div>
                      <div className="text-xs text-gray-400 mb-3">
                        Top-10 特征维度贡献值差异（贡献百分比，差异最大高亮标红）
                      </div>
                      <div className="overflow-x-auto rounded-lg border border-gray-700">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="bg-[#0a0e1a] text-gray-400">
                              <th className="px-3 py-2 text-left font-medium border-b border-gray-700">
                                特征维度
                              </th>
                              {top10CompareRows.validData.map((c, i) => (
                                <th
                                  key={i}
                                  className="px-3 py-2 text-left font-medium border-b border-gray-700"
                                >
                                  <span className="px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 mr-1">
                                    {String.fromCharCode(65 + i)}
                                  </span>
                                  {getMethodLabel(c.method)}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {top10CompareRows.rows.map((row) => {
                              const maxVal = Math.max(...row.values);
                              return (
                                <tr
                                  key={row.idx}
                                  className="border-b border-gray-800 last:border-0 hover:bg-gray-800/30"
                                >
                                  <td className="px-3 py-2 font-mono text-gray-300">
                                    ({row.x}, {row.y})
                                    <span className="ml-2 text-gray-500">#{row.idx}</span>
                                  </td>
                                  {row.values.map((v, i) => {
                                    const isMax = v === maxVal && row.maxDiff > 10;
                                    return (
                                      <td
                                        key={i}
                                        className={`px-3 py-2 font-mono ${
                                          isMax
                                            ? "text-red-300 bg-red-500/15 font-semibold"
                                            : "text-gray-200"
                                        }`}
                                      >
                                        {v.toFixed(2)}%
                                      </td>
                                    );
                                  })}
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                      <div className="mt-2 text-[11px] text-gray-500 flex items-center gap-2">
                        <span className="inline-flex items-center gap-1">
                          <span className="w-2 h-2 rounded bg-red-500/60" />
                          = 该行差异最大的值
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
