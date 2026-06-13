import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const LABEL_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

interface Props {
  distribution: number[][];
}

export function LabelDistribution({ distribution }: Props) {
  if (distribution.length === 0) {
    return (
      <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
        <div className="text-sm font-semibold text-gray-300 mb-4">Client Data Distribution</div>
        <div className="h-[260px] flex items-center justify-center text-gray-500 text-sm">
          Waiting for data...
        </div>
      </div>
    );
  }

  const displayClients = distribution.slice(0, 10);

  const data = LABEL_NAMES.map((label, labelIdx) => {
    const entry: Record<string, any> = { label };
    displayClients.forEach((clientData, clientIdx) => {
      entry[`c${clientIdx}`] = clientData[labelIdx] || 0;
    });
    return entry;
  });

  const colors = [
    "#22d3ee", "#a855f7", "#f59e0b", "#10b981", "#ef4444",
    "#3b82f6", "#ec4899", "#8b5cf6", "#14b8a6", "#f97316",
  ];

  return (
    <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
      <div className="text-sm font-semibold text-gray-300 mb-4">Client Data Distribution</div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="label" stroke="#4b5563" tick={{ fontSize: 11 }} />
          <YAxis stroke="#4b5563" tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
          />
          {displayClients.map((_, i) => (
            <Bar key={i} dataKey={`c${i}`} stackId="a" fill={colors[i % colors.length]} name={`Client ${i}`} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
