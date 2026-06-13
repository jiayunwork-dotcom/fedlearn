import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts";
import type { RoundMetrics } from "@/api";

interface Props {
  metrics: RoundMetrics[];
}

export function PrivacyChart({ metrics }: Props) {
  const data = metrics.map((m) => ({
    round: m.round,
    epsilon: parseFloat(m.epsilon.toFixed(4)),
  }));

  const maxEpsilon = data.length > 0 ? Math.max(...data.map((d) => d.epsilon), 1) : 10;

  return (
    <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
      <div className="text-sm font-semibold text-gray-300 mb-4">Privacy Budget (ε) Consumption</div>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="epsGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="round" stroke="#4b5563" tick={{ fontSize: 11 }} />
          <YAxis stroke="#a855f7" tick={{ fontSize: 11 }} domain={[0, maxEpsilon * 1.1]} />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
          />
          <Area type="monotone" dataKey="epsilon" stroke="#a855f7" strokeWidth={2} fill="url(#epsGrad)" name="ε" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
