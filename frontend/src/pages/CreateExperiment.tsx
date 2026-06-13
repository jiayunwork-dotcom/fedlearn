import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createExperiment, startTraining } from "@/api";
import type { ExperimentConfig } from "@/api";
import { useAppStore } from "@/store";
import {
  Play,
  Server,
  Shield,
  Swords,
  Cpu,
  Database,
  Settings2,
} from "lucide-react";

const defaultConfig: ExperimentConfig = {
  dataset: "mnist",
  num_clients: 10,
  client_fraction: 0.4,
  local_epochs: 5,
  global_rounds: 50,
  early_stop_patience: 10,
  aggregation_strategy: "fedavg",
  fedprox_mu: 0.01,
  non_iid_type: "label_skew",
  non_iid_param: 2,
  learning_rate: 0.01,
  batch_size: 32,
  client_selection: "random",
  dp_enabled: false,
  dp_clip_bound: 1.0,
  dp_noise_multiplier: 1.0,
  dp_delta: 1e-5,
  dp_epsilon_budget: 10.0,
  secure_aggregation: false,
  secure_agg_threshold: null,
  drop_rate: 0.0,
  attack_type: "none",
  attack_ratio: 0.0,
  defense_type: "none",
  defense_beta: 0.2,
};

export default function CreateExperiment() {
  const [config, setConfig] = useState<ExperimentConfig>(defaultConfig);
  const [creating, setCreating] = useState(false);
  const navigate = useNavigate();
  const fetchExperiments = useAppStore((s) => s.fetchExperiments);

  const update = (key: keyof ExperimentConfig, value: any) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async () => {
    setCreating(true);
    try {
      const result = await createExperiment(config);
      await startTraining(result.experiment_id);
      await fetchExperiments();
      navigate(`/experiment/${result.experiment_id}`);
    } catch (e) {
      console.error("Failed to create experiment:", e);
      alert("Failed to create experiment. Check console for details.");
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0e1a] text-gray-100">
      <div className="max-w-5xl mx-auto px-6 py-8">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
            <Server size={20} />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Create Federated Experiment</h1>
            <p className="text-sm text-gray-500">Configure all parameters for a new training run</p>
          </div>
        </div>

        <div className="space-y-6">
          <Section icon={<Database size={16} />} title="Dataset & Data Distribution">
            <div className="grid grid-cols-2 gap-4">
              <Field label="Dataset">
                <select value={config.dataset} onChange={(e) => update("dataset", e.target.value)} className={sel}>
                  <option value="mnist">MNIST</option>
                  <option value="fashion_mnist">Fashion-MNIST</option>
                  <option value="cifar10">CIFAR-10</option>
                </select>
              </Field>
              <Field label="Non-IID Type">
                <select value={config.non_iid_type} onChange={(e) => update("non_iid_type", e.target.value)} className={sel}>
                  <option value="label_skew">Label Skew (k classes)</option>
                  <option value="dirichlet">Dirichlet (quantity skew)</option>
                  <option value="feature_skew">Feature Skew (rotation/noise)</option>
                </select>
              </Field>
              <Field label="Non-IID Parameter">
                <input type="number" step="0.1" value={config.non_iid_param} onChange={(e) => update("non_iid_param", parseFloat(e.target.value))} className={inp} />
              </Field>
              <Field label="Num Clients">
                <input type="number" min={5} max={100} value={config.num_clients} onChange={(e) => update("num_clients", parseInt(e.target.value))} className={inp} />
              </Field>
            </div>
          </Section>

          <Section icon={<Settings2 size={16} />} title="Training Configuration">
            <div className="grid grid-cols-2 gap-4">
              <Field label="Client Fraction (per round)">
                <input type="number" step={0.05} min={0.1} max={1} value={config.client_fraction} onChange={(e) => update("client_fraction", parseFloat(e.target.value))} className={inp} />
              </Field>
              <Field label="Local Epochs">
                <input type="number" min={1} value={config.local_epochs} onChange={(e) => update("local_epochs", parseInt(e.target.value))} className={inp} />
              </Field>
              <Field label="Global Rounds">
                <input type="number" min={1} value={config.global_rounds} onChange={(e) => update("global_rounds", parseInt(e.target.value))} className={inp} />
              </Field>
              <Field label="Early Stop Patience">
                <input type="number" min={1} value={config.early_stop_patience} onChange={(e) => update("early_stop_patience", parseInt(e.target.value))} className={inp} />
              </Field>
              <Field label="Learning Rate">
                <input type="number" step={0.001} value={config.learning_rate} onChange={(e) => update("learning_rate", parseFloat(e.target.value))} className={inp} />
              </Field>
              <Field label="Batch Size">
                <input type="number" min={8} value={config.batch_size} onChange={(e) => update("batch_size", parseInt(e.target.value))} className={inp} />
              </Field>
            </div>
          </Section>

          <Section icon={<Cpu size={16} />} title="Aggregation Strategy">
            <div className="grid grid-cols-2 gap-4">
              <Field label="Strategy">
                <select value={config.aggregation_strategy} onChange={(e) => update("aggregation_strategy", e.target.value)} className={sel}>
                  <option value="fedavg">FedAvg</option>
                  <option value="fedprox">FedProx</option>
                  <option value="fednova">FedNova</option>
                  <option value="scaffold">Scaffold</option>
                </select>
              </Field>
              {config.aggregation_strategy === "fedprox" && (
                <Field label="FedProx mu">
                  <input type="number" step={0.001} value={config.fedprox_mu} onChange={(e) => update("fedprox_mu", parseFloat(e.target.value))} className={inp} />
                </Field>
              )}
              <Field label="Client Selection">
                <select value={config.client_selection} onChange={(e) => update("client_selection", e.target.value)} className={sel}>
                  <option value="random">Random</option>
                  <option value="resource">By Resource</option>
                  <option value="contribution">By Contribution</option>
                </select>
              </Field>
            </div>
          </Section>

          <Section icon={<Shield size={16} />} title="Privacy Protection">
            <div className="space-y-4">
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={config.dp_enabled} onChange={(e) => update("dp_enabled", e.target.checked)} className="accent-cyan-500" />
                Enable Differential Privacy (DP-SGD)
              </label>
              {config.dp_enabled && (
                <div className="grid grid-cols-2 gap-4 pl-6">
                  <Field label="Clip Bound (C)">
                    <input type="number" step={0.1} value={config.dp_clip_bound} onChange={(e) => update("dp_clip_bound", parseFloat(e.target.value))} className={inp} />
                  </Field>
                  <Field label="Noise Multiplier (sigma)">
                    <input type="number" step={0.1} value={config.dp_noise_multiplier} onChange={(e) => update("dp_noise_multiplier", parseFloat(e.target.value))} className={inp} />
                  </Field>
                  <Field label="Delta">
                    <input type="text" value={config.dp_delta} onChange={(e) => update("dp_delta", parseFloat(e.target.value))} className={inp} />
                  </Field>
                  <Field label="Epsilon Budget">
                    <input type="number" step={0.5} value={config.dp_epsilon_budget} onChange={(e) => update("dp_epsilon_budget", parseFloat(e.target.value))} className={inp} />
                  </Field>
                </div>
              )}
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={config.secure_aggregation} onChange={(e) => update("secure_aggregation", e.target.checked)} className="accent-cyan-500" />
                Enable Secure Aggregation (Shamir Secret Sharing)
              </label>
              {config.secure_aggregation && (
                <div className="grid grid-cols-2 gap-4 pl-6">
                  <Field label="Threshold (t)">
                    <input type="number" value={config.secure_agg_threshold || ""} placeholder="Auto (N/2+1)" onChange={(e) => update("secure_agg_threshold", e.target.value ? parseInt(e.target.value) : null)} className={inp} />
                  </Field>
                  <Field label="Client Drop Rate">
                    <input type="number" step={0.05} min={0} max={0.8} value={config.drop_rate} onChange={(e) => update("drop_rate", parseFloat(e.target.value))} className={inp} />
                  </Field>
                </div>
              )}
            </div>
          </Section>

          <Section icon={<Swords size={16} />} title="Attack & Defense">
            <div className="grid grid-cols-2 gap-4">
              <Field label="Attack Type">
                <select value={config.attack_type} onChange={(e) => update("attack_type", e.target.value)} className={sel}>
                  <option value="none">None</option>
                  <option value="data_poisoning">Data Poisoning</option>
                  <option value="model_poisoning">Model Poisoning</option>
                  <option value="backdoor">Backdoor Attack</option>
                </select>
              </Field>
              {config.attack_type !== "none" && (
                <Field label="Attack Ratio">
                  <input type="number" step={0.05} min={0} max={0.5} value={config.attack_ratio} onChange={(e) => update("attack_ratio", parseFloat(e.target.value))} className={inp} />
                </Field>
              )}
              <Field label="Defense Type">
                <select value={config.defense_type} onChange={(e) => update("defense_type", e.target.value)} className={sel}>
                  <option value="none">None</option>
                  <option value="krum">Krum</option>
                  <option value="trimmed_mean">Trimmed Mean</option>
                  <option value="median">Median</option>
                </select>
              </Field>
              {config.defense_type === "trimmed_mean" && (
                <Field label="Trim Beta">
                  <input type="number" step={0.05} min={0} max={0.5} value={config.defense_beta} onChange={(e) => update("defense_beta", parseFloat(e.target.value))} className={inp} />
                </Field>
              )}
            </div>
          </Section>

          <div className="flex justify-end pt-4">
            <button
              onClick={handleSubmit}
              disabled={creating}
              className="flex items-center gap-2 px-6 py-3 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold hover:from-cyan-400 hover:to-blue-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Play size={18} />
              {creating ? "Creating & Starting..." : "Create & Start Training"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Section({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
      <div className="flex items-center gap-2 mb-4 text-cyan-400 font-semibold text-sm uppercase tracking-wider">
        {icon}
        {title}
      </div>
      {children}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <label className="text-xs text-gray-400 font-medium">{label}</label>
      {children}
    </div>
  );
}

const inp =
  "w-full px-3 py-2 rounded-lg bg-[#0a0e1a] border border-gray-700 text-gray-200 text-sm focus:outline-none focus:border-cyan-500 transition-colors";
const sel =
  "w-full px-3 py-2 rounded-lg bg-[#0a0e1a] border border-gray-700 text-gray-200 text-sm focus:outline-none focus:border-cyan-500 transition-colors";
