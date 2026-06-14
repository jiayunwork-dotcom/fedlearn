import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  type TooltipProps,
} from "recharts";
import type { ClientContributions } from "@/api";

interface Props {
  clientContributions: ClientContributions | null | undefined;
}

interface ChartDataPoint {
  feature: string;
  importance: number;
  [key: `client_${number}`]: number;
}

interface TooltipPayloadItem {
  dataKey: string;
  value: number;
  color: string;
  payload: ChartDataPoint;
}

const CLIENT_COLORS = [
  "#22d3ee",
  "#3b82f6",
  "#8b5cf6",
  "#ec4899",
  "#f97316",
  "#eab308",
  "#22c55e",
  "#06b6d4",
  "#6366f1",
  "#ef4444",
];

function isClientPayload(item: unknown): item is TooltipPayloadItem {
  return (
    typeof item === "object" &&
    item !== null &&
    "dataKey" in item &&
    typeof (item as { dataKey: unknown }).dataKey === "string" &&
    (item as { dataKey: string }).dataKey.startsWith("client_")
  );
}

export function ClientContributionAttribution({ clientContributions }: Props) {
  const chartData = useMemo((): ChartDataPoint[] => {
    if (!clientContributions?.top_features) return [];

    return clientContributions.top_features.map((feature) => {
      const dataPoint: ChartDataPoint = {
        feature: `(${feature.coord[0]},${feature.coord[1]})`,
        importance: feature.importance,
      };

      feature.client_values.forEach((value, idx) => {
        (dataPoint as unknown as Record<string, number>)[`client_${idx}`] = value;
      });

      return dataPoint;
    });
  }, [clientContributions]);

  const clientNames = useMemo((): string[] => {
    if (!clientContributions?.client_names) return [];
    return clientContributions.client_names;
  }, [clientContributions]);

  if (!clientContributions || chartData.length === 0) {
    return (
      <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
        <div className="text-sm font-semibold text-gray-300 mb-4">客户端贡献度归因</div>
        <div className="h-[360px] flex items-center justify-center text-gray-500 text-sm">
          No data available
        </div>
      </div>
    );
  }

  const CustomTooltip = ({ active, payload, label }: TooltipProps<number, string>) => {
    if (active && payload && payload.length) {
      const clientPayloads = payload.filter(isClientPayload);
      const total = clientPayloads.reduce((sum, p) => sum + p.value, 0);
      const dataPayload = payload[0]?.payload as ChartDataPoint | undefined;

      return (
        <div className="bg-[#111827] border border-gray-700 rounded-lg p-3 shadow-xl">
          <div className="text-xs text-gray-400 mb-2">特征坐标 {label}</div>
          <div className="text-xs text-cyan-400 mb-2">
            总重要性: {dataPayload?.importance?.toFixed(2) || 0}%
          </div>
          <div className="space-y-1">
            {clientPayloads.map((p) => {
              const clientIdx = parseInt(p.dataKey.split("_")[1]);
              const percentage = total > 0 ? ((p.value / total) * 100).toFixed(1) : "0.0";
              return (
                <div key={p.dataKey} className="flex items-center justify-between gap-4 text-xs">
                  <div className="flex items-center gap-2">
                    <div
                      style={{
                        width: "10px",
                        height: "10px",
                        backgroundColor: p.color,
                        borderRadius: "2px",
                      }}
                    />
                    <span className="text-gray-300">{clientNames[clientIdx] || `Client ${clientIdx}`}</span>
                  </div>
                  <span className="text-cyan-400 font-mono">
                    {p.value.toFixed(2)}% ({percentage}%)
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
      <div className="text-sm font-semibold text-gray-300 mb-4">客户端贡献度归因</div>
      <p className="text-xs text-gray-500 mb-4">
        Top-10 特征维度在各客户端的贡献权重分布（柱高代表该客户端对该特征的贡献占比）
      </p>

      <div className="h-[360px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="feature"
              stroke="#4b5563"
              tick={{ fontSize: 10 }}
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis stroke="#4b5563" tick={{ fontSize: 10 }} unit="%" />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(31, 41, 55, 0.5)" }} />
            <Legend
              wrapperStyle={{
                fontSize: "10px",
                color: "#9ca3af",
              }}
              formatter={(value) => {
                const clientIdx = parseInt(value.split("_")[1]);
                return clientNames[clientIdx] || `Client ${clientIdx}`;
              }}
            />
            {clientNames.map((_, idx) => (
              <Bar
                key={`client_${idx}`}
                dataKey={`client_${idx}`}
                stackId="a"
                fill={CLIENT_COLORS[idx % CLIENT_COLORS.length]}
                radius={[0, 0, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="text-xs font-semibold text-gray-400 mb-3">客户端总权重</div>
        <div className="grid grid-cols-5 gap-2">
          {clientContributions.client_weights.map((weight, idx) => (
            <div
              key={idx}
              className="p-2 rounded-lg bg-[#0a0e1a] border border-gray-700 text-center"
            >
              <div
                className="w-3 h-3 rounded-full mx-auto mb-1"
                style={{ backgroundColor: CLIENT_COLORS[idx % CLIENT_COLORS.length] }}
              />
              <div className="text-xs text-gray-400">{clientNames[idx]}</div>
              <div className="text-sm font-semibold text-cyan-400 font-mono">
                {(weight * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
