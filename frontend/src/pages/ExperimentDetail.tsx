import { useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAppStore } from "@/store";
import { stopTraining } from "@/api";
import { TrainingProgress } from "@/components/TrainingProgress";
import { AccuracyChart } from "@/components/AccuracyChart";
import { ContributionChart } from "@/components/ContributionChart";
import { PrivacyChart } from "@/components/PrivacyChart";
import { LabelDistribution } from "@/components/LabelDistribution";
import { AttackLog } from "@/components/AttackLog";
import { ArrowLeft, Square, Activity } from "lucide-react";

export default function ExperimentDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const selectExperiment = useAppStore((s) => s.selectExperiment);
  const startPolling = useAppStore((s) => s.startPolling);
  const stopPolling = useAppStore((s) => s.stopPolling);
  const status = useAppStore((s) => s.status);
  const metrics = useAppStore((s) => s.metrics);
  const contributions = useAppStore((s) => s.contributions);
  const labelDistribution = useAppStore((s) => s.labelDistribution);
  const attackLog = useAppStore((s) => s.attackLog);

  useEffect(() => {
    if (!id) return;
    selectExperiment(id);
    startPolling(id);
    return () => stopPolling();
  }, [id]);

  const handleStop = async () => {
    if (!id) return;
    try {
      await stopTraining(id);
    } catch (e) {
      console.error(e);
    }
  };

  if (!status) {
    return (
      <div className="min-h-screen bg-[#0a0e1a] text-gray-100 flex items-center justify-center">
        <div className="text-gray-500 text-sm">Loading experiment data...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0e1a] text-gray-100">
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <button onClick={() => navigate("/")} className="p-2 rounded-lg hover:bg-gray-800 transition-colors">
              <ArrowLeft size={20} />
            </button>
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
              <Activity size={16} />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">Experiment {id}</h1>
              <div className="text-xs text-gray-500">
                Round {status.current_round} / {status.global_rounds}
                {status.is_running && <span className="ml-2 text-cyan-400 animate-pulse">● Running</span>}
                {!status.is_running && status.current_round > 0 && <span className="ml-2 text-emerald-400">● Completed</span>}
              </div>
            </div>
          </div>
          {status.is_running && (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 px-4 py-2 rounded-lg border border-red-500/30 text-red-400 text-sm hover:bg-red-500/10 transition-colors"
            >
              <Square size={14} />
              Stop Training
            </button>
          )}
        </div>

        <TrainingProgress status={status} metrics={metrics} />

        <div className="grid grid-cols-2 gap-4 mt-4">
          <AccuracyChart metrics={metrics} />
          <PrivacyChart metrics={metrics} />
        </div>

        <div className="grid grid-cols-2 gap-4 mt-4">
          <ContributionChart contributions={contributions} />
          <LabelDistribution distribution={labelDistribution} />
        </div>

        {attackLog.length > 0 && (
          <div className="mt-4">
            <AttackLog logs={attackLog} />
          </div>
        )}
      </div>
    </div>
  );
}
