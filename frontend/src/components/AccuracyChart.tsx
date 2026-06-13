import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import type { RoundMetrics } from "@/api";

interface Props {
  metrics: RoundMetrics[];
}

export function AccuracyChart({ metrics }: Props) {
  const data = metrics.map((m) => ({
    round: m.round,
    accuracy: parseFloat((m.global_accuracy * 100).toFixed(2)),
    loss: parseFloat(m.global_loss.toFixed(4)),
    similarity: parseFloat((m.client_similarity * 100).toFixed(2)),
  }));

  return (
    <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
      <div className="text-sm font-semibold text-gray-300 mb-4">Global Accuracy & Loss</div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="round" stroke="#4b5563" tick={{ fontSize: 11 }} />
          <YAxis yAxisId="acc" stroke="#22d3ee" tick={{ fontSize: 11 }} domain={[0, 100]} />
          <YAxis yAxisId="loss" orientation="right" stroke="#f59e0b" tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
          />
          <Line yAxisId="acc" type="monotone" dataKey="accuracy" stroke="#22d3ee" strokeWidth={2} dot={false} name="Accuracy %" />
          <Line yAxisId="loss" type="monotone" dataKey="loss" stroke="#f59e0b" strokeWidth={2} dot={false} name="Loss" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
