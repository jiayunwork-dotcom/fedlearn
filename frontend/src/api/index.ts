const API_BASE = import.meta.env.VITE_API_URL || "/api";

export interface ExperimentConfig {
  experiment_id?: string;
  status?: string;
  dataset: string;
  num_clients: number;
  client_fraction: number;
  local_epochs: number;
  global_rounds: number;
  early_stop_patience: number;
  aggregation_strategy: string;
  fedprox_mu: number;
  non_iid_type: string;
  non_iid_param: number;
  learning_rate: number;
  batch_size: number;
  client_selection: string;
  dp_enabled: boolean;
  dp_clip_bound: number;
  dp_noise_multiplier: number;
  dp_delta: number;
  dp_epsilon_budget: number;
  secure_aggregation: boolean;
  secure_agg_threshold: number | null;
  drop_rate: number;
  attack_type: string;
  attack_ratio: number;
  defense_type: string;
  defense_beta: number;
}

export interface ExperimentStatus {
  experiment_id: string;
  current_round: number;
  global_rounds: number;
  best_accuracy: number;
  is_running: boolean;
  current_accuracy?: number;
  current_loss?: number;
  epsilon?: number;
  client_similarity?: number;
  estimated_remaining_seconds?: number;
}

export interface RoundMetrics {
  round: number;
  global_accuracy: number;
  global_loss: number;
  client_accuracies: Record<string, number>;
  client_similarity: number;
  epsilon: number;
  num_active_clients: number;
  elapsed_seconds: number;
  round_seconds: number;
}

export interface AttackLogEntry {
  round: number;
  type: string;
  message: string;
}

export async function createExperiment(config: ExperimentConfig): Promise<{ experiment_id: string; config: ExperimentConfig }> {
  const res = await fetch(`${API_BASE}/experiments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`Create experiment failed: ${res.statusText}`);
  return res.json();
}

export async function startTraining(experimentId: string): Promise<{ experiment_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/start`, { method: "POST" });
  if (!res.ok) throw new Error(`Start training failed: ${res.statusText}`);
  return res.json();
}

export async function getStatus(experimentId: string): Promise<ExperimentStatus> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/status`);
  if (!res.ok) throw new Error(`Get status failed: ${res.statusText}`);
  return res.json();
}

export async function getMetrics(experimentId: string): Promise<{ experiment_id: string; metrics: RoundMetrics[] }> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/metrics`);
  if (!res.ok) throw new Error(`Get metrics failed: ${res.statusText}`);
  return res.json();
}

export async function stopTraining(experimentId: string): Promise<{ experiment_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/stop`, { method: "POST" });
  if (!res.ok) throw new Error(`Stop training failed: ${res.statusText}`);
  return res.json();
}

export async function listExperiments(): Promise<{ experiments: ExperimentConfig[] }> {
  const res = await fetch(`${API_BASE}/experiments`);
  if (!res.ok) throw new Error(`List experiments failed: ${res.statusText}`);
  return res.json();
}

export async function getContributions(experimentId: string): Promise<{ experiment_id: string; contributions: Record<string, number> }> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/contributions`);
  if (!res.ok) throw new Error(`Get contributions failed: ${res.statusText}`);
  return res.json();
}

export async function getLabelDistribution(experimentId: string): Promise<{ experiment_id: string; distribution: number[][] }> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/label-distribution`);
  if (!res.ok) throw new Error(`Get label distribution failed: ${res.statusText}`);
  return res.json();
}

export async function getAttackLog(experimentId: string): Promise<{ experiment_id: string; attack_log: AttackLogEntry[] }> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/attack-log`);
  if (!res.ok) throw new Error(`Get attack log failed: ${res.statusText}`);
  return res.json();
}

export async function compareExperiments(experimentId: string, otherId: string) {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/compare?other_id=${otherId}`);
  if (!res.ok) throw new Error(`Compare experiments failed: ${res.statusText}`);
  return res.json();
}
