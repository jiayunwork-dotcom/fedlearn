import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

interface Props {
  contributions: Record<string, number>;
}

export function ContributionChart({ contributions }: Props) {
  const entries = Object.entries(contributions)
    .map(([id, val]) => ({ id: `Client ${id}`, value: parseFloat(val.toFixed(6)) }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 20);

  const maxVal = entries.length > 0 ? entries[0].value : 1;

  if (entries.length === 0) {
    return (
      <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
        <div className="text-sm font-semibold text-gray-300 mb-4">Client Contributions</div>
        <div className="h-[260px] flex items-center justify-center text-gray-500 text-sm">
          Waiting for training data...
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
      <div className="text-sm font-semibold text-gray-300 mb-4">Client Contribution Ranking</div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={entries} layout="vertical" margin={{ left: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" stroke="#4b5563" tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="id" stroke="#4b5563" tick={{ fontSize: 10 }} width={55} />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {entries.map((entry, i) => {
              const intensity = 0.3 + (0.7 * (entry.value / maxVal));
              return <Cell key={i} fill={`rgba(34, 211, 238, ${intensity})`} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
