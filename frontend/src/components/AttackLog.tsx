import { AlertTriangle, ShieldAlert, ShieldOff } from "lucide-react";
import type { AttackLogEntry } from "@/api";

interface Props {
  logs: AttackLogEntry[];
}

const iconMap: Record<string, React.ReactNode> = {
  anomaly_detected: <AlertTriangle size={14} className="text-amber-400" />,
  anomaly_detected_low: <AlertTriangle size={14} className="text-yellow-400" />,
  anomaly_detected_medium: <AlertTriangle size={14} className="text-orange-400" />,
  anomaly_detected_high: <AlertTriangle size={14} className="text-red-500" />,
  privacy_budget_exhausted: <ShieldAlert size={14} className="text-red-400" />,
  byzantine_clients_active: <ShieldOff size={14} className="text-rose-400" />,
  secure_aggregation: <ShieldAlert size={14} className="text-cyan-400" />,
};

export function AttackLog({ logs }: Props) {
  return (
    <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
      <div className="flex items-center gap-2 mb-4 text-sm font-semibold text-gray-300">
        <ShieldOff size={16} className="text-red-400" />
        Attack Detection Log
      </div>
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {logs.map((log, i) => (
          <div key={i} className="flex items-start gap-3 px-3 py-2 rounded-lg bg-[#0a0e1a] border border-gray-800">
            <div className="mt-0.5">{iconMap[log.type] || <AlertTriangle size={14} className="text-amber-400" />}</div>
            <div>
              <div className="text-xs font-medium text-gray-300">Round {log.round}</div>
              <div className="text-xs text-gray-500 mt-0.5">{log.message}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
