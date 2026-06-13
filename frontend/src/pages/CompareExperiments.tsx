import { useEffect, useMemo } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { useAppStore } from "@/store";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
} from "recharts";
import { ArrowLeft, GitCompareArrows, Loader2 } from "lucide-react";
import type { ComparisonItem } from "@/api";

const COLORS = ["#22d3ee", "#f59e0b", "#a855f7", "#10b981"];

function formatTime(secs: number): string {
  if (secs < 60) return `${Math.round(secs)}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

function ExperimentSummaryCards({ items }: { items: ComparisonItem[] }) {
  return (
    <div className="flex gap-4 overflow-x-auto pb-2">
      {items.map((item, i) => {
        const cfg = item.config;
        const color = COLORS[i % COLORS.length];
        return (
          <div
            key={item.experiment_id}
            className="flex-shrink-0 min-w-[220px] rounded-xl border border-gray-800 bg-[#111827] p-4"
            style={{ borderTop: `2px solid ${color}` }}
          >
            <div className="font-semibold text-sm truncate" style={{ color }}>
              {item.experiment_id}
            </div>
            <div className="mt-3 space-y-1.5 text-xs text-gray-400">
              <div className="flex justify-between">
                <span>聚合策略</span>
                <span className="text-gray-200">{cfg.aggregation_strategy}</span>
              </div>
              <div className="flex justify-between">
                <span>数据集</span>
                <span className="text-gray-200">{cfg.dataset}</span>
              </div>
              <div className="flex justify-between">
                <span>客户端数</span>
                <span className="text-gray-200">{cfg.num_clients}</span>
              </div>
              <div className="flex justify-between">
                <span>Non-IID类型</span>
                <span className="text-gray-200">{cfg.non_iid_type}</span>
              </div>
              <div className="flex justify-between">
                <span>DP</span>
                <span className={cfg.dp_enabled ? "text-cyan-400" : "text-gray-200"}>
                  {cfg.dp_enabled ? "启用" : "关闭"}
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function AccuracyComparisonChart({ items }: { items: ComparisonItem[] }) {
  const maxRounds = Math.max(...items.map((it) => it.metrics.length), 1);
  const rounds = Array.from({ length: maxRounds }, (_, i) => i + 1);

  const data = rounds.map((r) => {
    const point: Record<string, number> = { round: r };
    items.forEach((item) => {
      const m = item.metrics.find((m) => m.round === r);
      if (m) {
        point[item.experiment_id] = parseFloat((m.global_accuracy * 100).toFixed(2));
      }
    });
    return point;
  });

  return (
    <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
      <div className="text-sm font-semibold text-gray-300 mb-4">精度对比 - 收敛曲线</div>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="round" stroke="#4b5563" tick={{ fontSize: 11 }} label={{ value: "通信轮次", position: "insideBottomRight", offset: -5, fontSize: 11, fill: "#6b7280" }} />
          <YAxis stroke="#4b5563" tick={{ fontSize: 11 }} domain={[0, 100]} label={{ value: "全局验证精度 (%)", angle: -90, position: "insideLeft", offset: 10, fontSize: 11, fill: "#6b7280" }} />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
            formatter={(value: number, name: string) => {
              const item = items.find((it) => it.experiment_id === name);
              const label = item ? `${name} (${item.config.aggregation_strategy})` : name;
              return [`${value.toFixed(2)}%`, label];
            }}
          />
          <Legend
            formatter={(value: string) => {
              const item = items.find((it) => it.experiment_id === value);
              return item ? `${value} (${item.config.aggregation_strategy})` : value;
            }}
            wrapperStyle={{ fontSize: 11 }}
          />
          {items.map((item, i) => (
            <Line
              key={item.experiment_id}
              type="monotone"
              dataKey={item.experiment_id}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              dot={false}
              name={item.experiment_id}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function PrivacyComparisonChart({ items }: { items: ComparisonItem[] }) {
  const maxRounds = Math.max(...items.map((it) => it.metrics.length), 1);
  const rounds = Array.from({ length: maxRounds }, (_, i) => i + 1);

  const data = rounds.map((r) => {
    const point: Record<string, number> = { round: r };
    items.forEach((item) => {
      const m = item.metrics.find((m) => m.round === r);
      if (m) {
        point[item.experiment_id] = parseFloat(m.epsilon.toFixed(4));
      }
    });
    return point;
  });

  return (
    <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
      <div className="text-sm font-semibold text-gray-300 mb-4">隐私消耗对比 - Epsilon 累计变化</div>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="round" stroke="#4b5563" tick={{ fontSize: 11 }} label={{ value: "通信轮次", position: "insideBottomRight", offset: -5, fontSize: 11, fill: "#6b7280" }} />
          <YAxis stroke="#4b5563" tick={{ fontSize: 11 }} label={{ value: "ε 累计消耗", angle: -90, position: "insideLeft", offset: 10, fontSize: 11, fill: "#6b7280" }} />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
            formatter={(value: number, name: string) => {
              const item = items.find((it) => it.experiment_id === name);
              const label = item ? `${name} (${item.config.aggregation_strategy})` : name;
              return [value.toFixed(4), label];
            }}
          />
          <Legend
            formatter={(value: string) => {
              const item = items.find((it) => it.experiment_id === value);
              return item ? `${value} (${item.config.aggregation_strategy})` : value;
            }}
            wrapperStyle={{ fontSize: 11 }}
          />
          {items.map((item, i) => (
            <Line
              key={item.experiment_id}
              type="monotone"
              dataKey={item.experiment_id}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              dot={false}
              name={item.experiment_id}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function SummaryComparisonTable({ items }: { items: ComparisonItem[] }) {
  const rows = useMemo(() => {
    const finalAccs = items.map((it) => it.summary.final_accuracy);
    const bestAcc = Math.max(...finalAccs);

    const round90s = items.map((it) => it.summary.round_to_90_percent).filter((v): v is number => v !== null);
    const bestRound90 = round90s.length > 0 ? Math.min(...round90s) : null;

    const epsilons = items.map((it) => it.summary.final_epsilon);
    const bestEps = Math.min(...epsilons.filter((e) => e > 0));
    const hasEps = epsilons.some((e) => e > 0);

    const totalTimes = items.map((it) => it.summary.total_elapsed_seconds);
    const bestTime = Math.min(...totalTimes.filter((t) => t > 0));
    const hasTime = totalTimes.some((t) => t > 0);

    const avgTimes = items.map((it) => it.summary.avg_round_seconds);
    const bestAvgTime = Math.min(...avgTimes.filter((t) => t > 0));
    const hasAvgTime = avgTimes.some((t) => t > 0);

    return [
      {
        label: "最终精度",
        values: items.map((it) => `${(it.summary.final_accuracy * 100).toFixed(2)}%`),
        bestIdx: items.findIndex((it) => it.summary.final_accuracy === bestAcc),
      },
      {
        label: "达到90%精度所需轮次",
        values: items.map((it) =>
          it.summary.round_to_90_percent !== null ? `${it.summary.round_to_90_percent}` : "未达到"
        ),
        bestIdx: bestRound90 !== null ? items.findIndex((it) => it.summary.round_to_90_percent === bestRound90) : -1,
      },
      {
        label: "最终ε消耗",
        values: items.map((it) => (it.summary.final_epsilon > 0 ? it.summary.final_epsilon.toFixed(4) : "N/A")),
        bestIdx: hasEps ? items.findIndex((it) => it.summary.final_epsilon === bestEps) : -1,
      },
      {
        label: "总训练时长",
        values: items.map((it) =>
          it.summary.total_elapsed_seconds > 0 ? formatTime(it.summary.total_elapsed_seconds) : "N/A"
        ),
        bestIdx: hasTime ? items.findIndex((it) => it.summary.total_elapsed_seconds === bestTime) : -1,
      },
      {
        label: "每轮平均耗时",
        values: items.map((it) =>
          it.summary.avg_round_seconds > 0 ? `${it.summary.avg_round_seconds.toFixed(1)}s` : "N/A"
        ),
        bestIdx: hasAvgTime ? items.findIndex((it) => it.summary.avg_round_seconds === bestAvgTime) : -1,
      },
    ];
  }, [items]);

  return (
    <div className="rounded-xl border border-gray-800 bg-[#111827] p-5 overflow-x-auto">
      <div className="text-sm font-semibold text-gray-300 mb-4">最终指标对比表</div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="text-left py-2 px-3 text-gray-500 font-medium w-48">对比维度</th>
            {items.map((item, i) => (
              <th key={item.experiment_id} className="text-center py-2 px-3 font-medium" style={{ color: COLORS[i % COLORS.length] }}>
                {item.experiment_id}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label} className="border-b border-gray-800">
              <td className="py-2.5 px-3 text-gray-400">{row.label}</td>
              {row.values.map((val, i) => (
                <td
                  key={i}
                  className={`text-center py-2.5 px-3 ${
                    i === row.bestIdx ? "text-emerald-400 font-semibold" : "text-gray-200"
                  }`}
                >
                  {val}
                  {i === row.bestIdx && <span className="ml-1 text-emerald-500 text-xs">★</span>}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ContributionComparisonChart({ items }: { items: ComparisonItem[] }) {
  const allClientIds = useMemo(() => {
    const ids = new Set<string>();
    items.forEach((item) => Object.keys(item.contributions).forEach((id) => ids.add(id)));
    return Array.from(ids).sort((a, b) => parseInt(a) - parseInt(b)).slice(0, 15);
  }, [items]);

  const data = useMemo(() => {
    return allClientIds.map((cid) => {
      const point: Record<string, string | number> = { client: `C${cid}` };
      items.forEach((item) => {
        point[item.experiment_id] = item.contributions[cid] ? parseFloat(item.contributions[cid].toFixed(6)) : 0;
      });
      return point;
    });
  }, [allClientIds, items]);

  if (data.length === 0) {
    return (
      <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
        <div className="text-sm font-semibold text-gray-300 mb-4">客户端贡献度对比</div>
        <div className="h-[300px] flex items-center justify-center text-gray-500 text-sm">
          暂无贡献度数据
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
      <div className="text-sm font-semibold text-gray-300 mb-4">客户端贡献度对比</div>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="client" stroke="#4b5563" tick={{ fontSize: 10 }} />
          <YAxis stroke="#4b5563" tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
            formatter={(value: number, name: string) => {
              const item = items.find((it) => it.experiment_id === name);
              const label = item ? `${name} (${item.config.aggregation_strategy})` : name;
              return [value.toFixed(6), label];
            }}
          />
          <Legend
            formatter={(value: string) => {
              const item = items.find((it) => it.experiment_id === value);
              return item ? `${value} (${item.config.aggregation_strategy})` : value;
            }}
            wrapperStyle={{ fontSize: 11 }}
          />
          {items.map((item, i) => (
            <Bar
              key={item.experiment_id}
              dataKey={item.experiment_id}
              fill={COLORS[i % COLORS.length]}
              radius={[3, 3, 0, 0]}
              name={item.experiment_id}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function CompareExperiments() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const comparisons = useAppStore((s) => s.comparisons);
  const comparisonsLoading = useAppStore((s) => s.comparisonsLoading);
  const fetchComparisons = useAppStore((s) => s.fetchComparisons);

  const idsParam = searchParams.get("ids") || "";
  const ids = idsParam.split(",").filter(Boolean);

  useEffect(() => {
    if (ids.length >= 2) {
      fetchComparisons(ids);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsParam]);

  if (ids.length < 2) {
    return (
      <div className="min-h-screen bg-[#0a0e1a] text-gray-100 flex items-center justify-center">
        <div className="text-gray-500 text-sm">请至少选择2个实验进行对比</div>
      </div>
    );
  }

  if (comparisonsLoading) {
    return (
      <div className="min-h-screen bg-[#0a0e1a] text-gray-100 flex items-center justify-center">
        <div className="flex items-center gap-3 text-gray-400">
          <Loader2 size={20} className="animate-spin" />
          <span className="text-sm">加载对比数据...</span>
        </div>
      </div>
    );
  }

  if (comparisons.length === 0) {
    return (
      <div className="min-h-screen bg-[#0a0e1a] text-gray-100 flex items-center justify-center">
        <div className="text-gray-500 text-sm">未找到对比数据</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0e1a] text-gray-100">
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex items-center gap-3 mb-6">
          <button onClick={() => navigate("/")} className="p-2 rounded-lg hover:bg-gray-800 transition-colors">
            <ArrowLeft size={20} />
          </button>
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
            <GitCompareArrows size={16} />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">实验对比</h1>
            <div className="text-xs text-gray-500">
              对比 {comparisons.length} 个实验的指标差异
            </div>
          </div>
        </div>

        <ExperimentSummaryCards items={comparisons} />

        <div className="grid grid-cols-2 gap-4 mt-6">
          <AccuracyComparisonChart items={comparisons} />
          <PrivacyComparisonChart items={comparisons} />
        </div>

        <div className="mt-4">
          <SummaryComparisonTable items={comparisons} />
        </div>

        <div className="mt-4">
          <ContributionComparisonChart items={comparisons} />
        </div>
      </div>
    </div>
  );
}
