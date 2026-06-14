import { useEffect, useRef, type ComponentType } from "react";
import { X, Play, Square, BarChart3, Brain, Zap, Target, type LucideProps } from "lucide-react";
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

export function InterpretabilityPanel({ experimentId, onClose }: Props) {
  const logContainerRef = useRef<HTMLDivElement>(null);

  const interpretabilityMethod = useAppStore((s) => s.interpretabilityMethod);
  const interpretabilityNumSamples = useAppStore((s) => s.interpretabilityNumSamples);
  const interpretabilityStatus = useAppStore((s) => s.interpretabilityStatus);
  const interpretabilityProgress = useAppStore((s) => s.interpretabilityProgress);
  const interpretabilityCurrentSample = useAppStore((s) => s.interpretabilityCurrentSample);
  const interpretabilityLogs = useAppStore((s) => s.interpretabilityLogs);
  const interpretabilityResult = useAppStore((s) => s.interpretabilityResult);
  const interpretabilityError = useAppStore((s) => s.interpretabilityError);
  const selectedClassForDetail = useAppStore((s) => s.selectedClassForDetail);

  const setInterpretabilityMethod = useAppStore((s) => s.setInterpretabilityMethod);
  const setInterpretabilityNumSamples = useAppStore((s) => s.setInterpretabilityNumSamples);
  const startInterpretability = useAppStore((s) => s.startInterpretability);
  const cancelInterpretability = useAppStore((s) => s.cancelInterpretability);
  const setSelectedClassForDetail = useAppStore((s) => s.setSelectedClassForDetail);
  const resetInterpretability = useAppStore((s) => s.resetInterpretability);

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
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-800 transition-colors text-gray-400 hover:text-gray-300"
          >
            <X size={20} />
          </button>
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
                    const isSelected = interpretabilityNumSamples === num;
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
                </div>

                <div className="mt-4 p-3 rounded-lg bg-[#0a0e1a] border border-gray-700">
                  <div className="text-xs text-gray-500">
                    当前选择：<span className="text-cyan-400 font-medium">{getMethodLabel(interpretabilityMethod)}</span> × <span className="text-cyan-400 font-medium">{interpretabilityNumSamples} 样本</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3 pt-4 border-t border-gray-700">
              {!isRunning && interpretabilityStatus !== "completed" && (
                <button
                  onClick={handleStart}
                  className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white text-sm font-medium hover:from-cyan-600 hover:to-blue-700 transition-all shadow-lg shadow-cyan-500/20"
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
                attribution={interpretabilityResult.overall_attribution}
                title="特征重要性热力图（总体）"
                showLegend={true}
              />

              <ClassAttributionComparison
                classAttributions={interpretabilityResult.class_attributions}
                classSampleCounts={interpretabilityResult.class_sample_counts}
                selectedClass={selectedClassForDetail}
                onSelectClass={setSelectedClassForDetail}
              />

              <ClientContributionAttribution
                clientContributions={interpretabilityResult.client_contributions}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
