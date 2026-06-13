import { useEffect } from "react";
import { useAppStore } from "@/store";
import { Link, useNavigate } from "react-router-dom";
import { Plus, FlaskConical, Clock, CheckCircle, XCircle, GitCompareArrows, X } from "lucide-react";

export default function ExperimentList() {
  const experiments = useAppStore((s) => s.experiments);
  const fetchExperiments = useAppStore((s) => s.fetchExperiments);
  const selectedCompareIds = useAppStore((s) => s.selectedCompareIds);
  const toggleCompareSelection = useAppStore((s) => s.toggleCompareSelection);
  const clearCompareSelection = useAppStore((s) => s.clearCompareSelection);
  const navigate = useNavigate();

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

  const handleCompare = () => {
    if (selectedCompareIds.length >= 2) {
      navigate(`/compare?ids=${selectedCompareIds.join(",")}`);
    }
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
            {experiments.map((exp) => {
              const isSelected = selectedCompareIds.includes(exp.experiment_id || "");
              return (
                <div
                  key={exp.experiment_id}
                  className={`block rounded-xl border bg-[#111827] p-5 transition-colors ${
                    isSelected ? "border-cyan-500/60 ring-1 ring-cyan-500/20" : "border-gray-800 hover:border-cyan-500/30"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <button
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          toggleCompareSelection(exp.experiment_id || "");
                        }}
                        className={`flex-shrink-0 w-5 h-5 rounded border-2 flex items-center justify-center transition-all ${
                          isSelected
                            ? "bg-cyan-500 border-cyan-500"
                            : "border-gray-600 hover:border-cyan-400"
                        }`}
                      >
                        {isSelected && (
                          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                            <path d="M2 6L5 9L10 3" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        )}
                      </button>
                      <Link to={`/experiment/${exp.experiment_id}`} className="flex items-center gap-3 flex-1 min-w-0">
                        {statusIcon(exp.status)}
                        <div className="min-w-0">
                          <div className="font-semibold text-sm truncate">{exp.experiment_id}</div>
                          <div className="text-xs text-gray-500 mt-0.5">
                            {exp.dataset} · {exp.aggregation_strategy} · {exp.num_clients} clients · {exp.global_rounds} rounds
                          </div>
                        </div>
                      </Link>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs border ${statusBadge(exp.status)}`}>
                      {exp.status || "created"}
                    </span>
                  </div>
                  <div className="mt-3 flex gap-4 text-xs text-gray-500 ml-8">
                    <span>Non-IID: {exp.non_iid_type}</span>
                    <span>Selection: {exp.client_selection}</span>
                    {exp.dp_enabled && <span className="text-cyan-400">DP enabled</span>}
                    {exp.attack_type !== "none" && <span className="text-red-400">Attack: {exp.attack_type}</span>}
                    {exp.defense_type !== "none" && <span className="text-emerald-400">Defense: {exp.defense_type}</span>}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {selectedCompareIds.length >= 2 && (
        <div className="fixed bottom-0 left-0 right-0 bg-[#111827]/95 backdrop-blur-sm border-t border-cyan-500/20 px-6 py-4 z-50">
          <div className="max-w-5xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
              <GitCompareArrows size={20} className="text-cyan-400" />
              <span className="text-sm text-gray-300">
                已选择 <span className="text-cyan-400 font-semibold">{selectedCompareIds.length}</span> 个实验
                <span className="text-gray-500 ml-1">(最多4个)</span>
              </span>
              <div className="flex gap-1.5 ml-2">
                {selectedCompareIds.map((id) => (
                  <span key={id} className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20 text-xs text-cyan-400">
                    {id}
                    <button
                      onClick={() => toggleCompareSelection(id)}
                      className="hover:text-white transition-colors"
                    >
                      <X size={10} />
                    </button>
                  </span>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={clearCompareSelection}
                className="px-4 py-2 rounded-lg border border-gray-700 text-gray-400 text-sm hover:border-gray-500 hover:text-gray-300 transition-colors"
              >
                取消选择
              </button>
              <button
                onClick={handleCompare}
                className="flex items-center gap-2 px-5 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold text-sm hover:from-cyan-400 hover:to-blue-500 transition-all"
              >
                <GitCompareArrows size={16} />
                对比选中实验
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
