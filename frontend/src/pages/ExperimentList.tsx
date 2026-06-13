import { useEffect } from "react";
import { useAppStore } from "@/store";
import { Link } from "react-router-dom";
import { Plus, FlaskConical, Clock, CheckCircle, XCircle } from "lucide-react";

export default function ExperimentList() {
  const experiments = useAppStore((s) => s.experiments);
  const fetchExperiments = useAppStore((s) => s.fetchExperiments);

  useEffect(() => {
    fetchExperiments();
    const timer = setInterval(fetchExperiments, 5000);
    return () => clearInterval(timer);
  }, [fetchExperiments]);

  const statusIcon = (status?: string) => {
    if (status === "running") return <Clock size={14} className="text-cyan-400 animate-pulse" />;
    if (status === "completed") return <CheckCircle size={14} className="text-emerald-400" />;
    if (status === "stopped") return <XCircle size={14} className="text-red-400" />;
    return <Clock size={14} className="text-gray-500" />;
  };

  const statusBadge = (status?: string) => {
    const colors: Record<string, string> = {
      running: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
      completed: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
      stopped: "bg-red-500/10 text-red-400 border-red-500/20",
      created: "bg-gray-500/10 text-gray-400 border-gray-500/20",
    };
    return colors[status || "created"] || colors.created;
  };

  return (
    <div className="min-h-screen bg-[#0a0e1a] text-gray-100">
      <div className="max-w-5xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
              <FlaskConical size={20} />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Federated Experiments</h1>
              <p className="text-sm text-gray-500">{experiments.length} experiment{experiments.length !== 1 ? "s" : ""}</p>
            </div>
          </div>
          <Link
            to="/create"
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold text-sm hover:from-cyan-400 hover:to-blue-500 transition-all"
          >
            <Plus size={16} />
            New Experiment
          </Link>
        </div>

        {experiments.length === 0 ? (
          <div className="text-center py-20 text-gray-500">
            <FlaskConical size={48} className="mx-auto mb-4 opacity-30" />
            <p className="text-lg">No experiments yet</p>
            <p className="text-sm mt-1">Create your first federated learning experiment</p>
          </div>
        ) : (
          <div className="space-y-3">
            {experiments.map((exp) => (
              <Link
                key={exp.experiment_id}
                to={`/experiment/${exp.experiment_id}`}
                className="block rounded-xl border border-gray-800 bg-[#111827] p-5 hover:border-cyan-500/30 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {statusIcon(exp.status)}
                    <div>
                      <div className="font-semibold text-sm">{exp.experiment_id}</div>
                      <div className="text-xs text-gray-500 mt-0.5">
                        {exp.dataset} · {exp.aggregation_strategy} · {exp.num_clients} clients · {exp.global_rounds} rounds
                      </div>
                    </div>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs border ${statusBadge(exp.status)}`}>
                    {exp.status || "created"}
                  </span>
                </div>
                <div className="mt-3 flex gap-4 text-xs text-gray-500">
                  <span>Non-IID: {exp.non_iid_type}</span>
                  <span>Selection: {exp.client_selection}</span>
                  {exp.dp_enabled && <span className="text-cyan-400">DP enabled</span>}
                  {exp.attack_type !== "none" && <span className="text-red-400">Attack: {exp.attack_type}</span>}
                  {exp.defense_type !== "none" && <span className="text-emerald-400">Defense: {exp.defense_type}</span>}
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
