import type { ExperimentStatus, RoundMetrics } from "@/api";
import { Clock, Target, Zap, Shield } from "lucide-react";

interface Props {
  status: ExperimentStatus;
  metrics: RoundMetrics[];
}

export function TrainingProgress({ status, metrics }: Props) {
  const progress = status.global_rounds > 0 ? (status.current_round / status.global_rounds) * 100 : 0;
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const estRemaining = status.estimated_remaining_seconds;

  const formatTime = (secs: number) => {
    if (secs < 60) return `${Math.round(secs)}s`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
    return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
  };

  const stats = [
    {
      icon: <Target size={16} />,
      label: "Best Accuracy",
      value: `${(status.best_accuracy * 100).toFixed(2)}%`,
      color: "text-emerald-400",
    },
    {
      icon: <Zap size={16} />,
      label: "Current Loss",
      value: latest ? latest.global_loss.toFixed(4) : "—",
      color: "text-amber-400",
    },
    {
      icon: <Shield size={16} />,
      label: "Privacy ε",
      value: latest ? latest.epsilon.toFixed(4) : "—",
      color: "text-cyan-400",
    },
    {
      icon: <Clock size={16} />,
      label: "Est. Remaining",
      value: estRemaining ? formatTime(estRemaining) : "—",
      color: "text-purple-400",
    },
  ];

  return (
    <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="text-sm font-semibold text-gray-300">Training Progress</div>
        <div className="text-xs text-gray-500">
          Round {status.current_round} / {status.global_rounds}
        </div>
      </div>

      <div className="w-full bg-gray-800 rounded-full h-2 mb-5">
        <div
          className="h-2 rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-500"
          style={{ width: `${Math.min(progress, 100)}%` }}
        />
      </div>

      <div className="grid grid-cols-4 gap-4">
        {stats.map((s) => (
          <div key={s.label} className="flex items-center gap-2">
            <div className={`${s.color}`}>{s.icon}</div>
            <div>
              <div className="text-xs text-gray-500">{s.label}</div>
              <div className={`text-sm font-semibold ${s.color}`}>{s.value}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
