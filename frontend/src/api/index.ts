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

export interface ExperimentSummary {
  final_accuracy: number;
  round_to_90_percent: number | null;
  final_epsilon: number;
  total_elapsed_seconds: number;
  avg_round_seconds: number;
  best_accuracy: number;
  current_round: number;
  global_rounds: number;
}

export interface ComparisonItem {
  experiment_id: string;
  config: ExperimentConfig;
  metrics: RoundMetrics[];
  contributions: Record<string, number>;
  summary: ExperimentSummary;
}

export async function batchCompareExperiments(ids: string[]): Promise<{ comparisons: ComparisonItem[] }> {
  const res = await fetch(`${API_BASE}/experiments/batch-compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ experiment_ids: ids }),
  });
  if (!res.ok) throw new Error(`Batch compare failed: ${res.statusText}`);
  return res.json();
}

export async function compareExperiments(experimentId: string, otherId: string) {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/compare?other_id=${otherId}`);
  if (!res.ok) throw new Error(`Compare experiments failed: ${res.statusText}`);
  return res.json();
}

export type AnalysisMethod = "gradient" | "permutation" | "shap";

export interface InterpretabilityLogEntry {
  batch: number;
  total_batches: number;
  sample_start?: number;
  sample_end?: number;
  feature_index?: number;
  feature_coord?: number[];
  accuracy_drop?: number;
  batch_time_ms: number;
  top_features: number[];
  timestamp: number;
}

export interface ClientContributionFeature {
  index: number;
  coord: number[];
  importance: number;
  client_values: number[];
}

export interface ClientContributions {
  client_names: string[];
  client_weights: number[];
  top_features: ClientContributionFeature[];
}

export interface InterpretabilityResult {
  method: AnalysisMethod;
  num_samples: number;
  overall_attribution: number[][][];
  class_attributions: number[][][][];
  class_sample_counts: number[];
  client_contributions: ClientContributions;
  logs: InterpretabilityLogEntry[];
  status: string;
  error?: string;
}

export interface InterpretabilityProgress {
  type: "progress";
  progress: number;
  current_sample: number;
  log?: InterpretabilityLogEntry;
}

export interface InterpretabilityComplete {
  type: "complete";
  status: string;
  result: InterpretabilityResult;
}

export type InterpretabilityWebSocketMessage = InterpretabilityProgress | InterpretabilityComplete;

export interface InterpretabilitySingleStatus {
  experiment_id: string;
  method: AnalysisMethod;
  num_samples: number;
  status: string;
  progress: number;
  current_sample?: number;
  logs?: InterpretabilityLogEntry[];
  cached?: boolean;
}

export interface InterpretabilityListStatus {
  experiment_id: string;
  analyses: Array<{
    method: AnalysisMethod;
    num_samples: number;
    status: string;
    progress: number;
    cached?: boolean;
    current_sample?: number;
    logs?: InterpretabilityLogEntry[];
  }>;
}

export async function startInterpretabilityAnalysis(
  experimentId: string,
  method: AnalysisMethod,
  numSamples: number
): Promise<{
  experiment_id: string; method: AnalysisMethod; num_samples: number; status: string;
  cached: boolean; result?: InterpretabilityResult; analysis_timestamp?: number;
}> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/interpretability/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ method, num_samples: numSamples }),
  });
  if (!res.ok) throw new Error(`Start interpretability analysis failed: ${res.statusText}`);
  return res.json();
}

export async function cancelInterpretabilityAnalysis(
  experimentId: string,
  method?: AnalysisMethod,
  numSamples?: number
): Promise<{ experiment_id: string; status: string; method?: AnalysisMethod; num_samples?: number; cancelled_tasks?: Array<{ method: AnalysisMethod; num_samples: number }> }> {
  const params = new URLSearchParams();
  if (method) params.append("method", method);
  if (numSamples !== undefined) params.append("num_samples", numSamples.toString());
  const query = params.toString() ? `?${params.toString()}` : "";

  const res = await fetch(`${API_BASE}/experiments/${experimentId}/interpretability/cancel${query}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`Cancel interpretability analysis failed: ${res.statusText}`);
  return res.json();
}

export async function getInterpretabilityStatus(
  experimentId: string,
  method?: AnalysisMethod,
  numSamples?: number
): Promise<InterpretabilitySingleStatus | InterpretabilityListStatus> {
  const params = new URLSearchParams();
  if (method) params.append("method", method);
  if (numSamples !== undefined) params.append("num_samples", numSamples.toString());
  const query = params.toString() ? `?${params.toString()}` : "";

  const res = await fetch(`${API_BASE}/experiments/${experimentId}/interpretability/status${query}`);
  if (!res.ok) throw new Error(`Get interpretability status failed: ${res.statusText}`);
  return res.json();
}

export async function getInterpretabilityResult(
  experimentId: string,
  method: AnalysisMethod,
  numSamples: number
): Promise<{
  experiment_id: string; method: AnalysisMethod; num_samples: number; status: string;
  result?: InterpretabilityResult; progress?: number; current_sample?: number;
  logs?: InterpretabilityLogEntry[]; analysis_timestamp?: number; resumed?: boolean;
}> {
  const res = await fetch(
    `${API_BASE}/experiments/${experimentId}/interpretability/result?method=${method}&num_samples=${numSamples}`
  );
  if (!res.ok) throw new Error(`Get interpretability result failed: ${res.statusText}`);
  return res.json();
}

export interface InterpretabilityHistoryEntry {
  method: AnalysisMethod;
  num_samples: number;
  status: string;
  progress: number;
  analysis_timestamp?: number;
  cached?: boolean;
  resumed?: boolean;
  current_sample?: number;
}

export async function getInterpretabilityHistory(
  experimentId: string
): Promise<{ experiment_id: string; analyses: InterpretabilityHistoryEntry[] }> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/interpretability/history`);
  if (!res.ok) throw new Error(`Get interpretability history failed: ${res.statusText}`);
  return res.json();
}

export interface ResumeAnalysisEntry {
  method: AnalysisMethod;
  num_samples: number;
  status: string;
  progress: number;
  current_sample: number;
  logs?: InterpretabilityLogEntry[];
  resumed?: boolean;
}

export async function resumeInterpretability(
  experimentId: string
): Promise<{ experiment_id: string; running_analyses: ResumeAnalysisEntry[] }> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/interpretability/resume`);
  if (!res.ok) throw new Error(`Resume interpretability failed: ${res.statusText}`);
  return res.json();
}

export interface InterpretabilityCompareItem {
  method: AnalysisMethod;
  num_samples: number;
  analysis_timestamp?: number;
  overall_attribution?: number[][][];
  class_attributions?: number[][][][];
  client_contributions?: ClientContributions;
  error?: string;
}

export async function batchCompareInterpretability(
  experimentId: string,
  analyses: Array<{ method: AnalysisMethod; num_samples: number; status: string; analysis_timestamp?: number }>
): Promise<{ experiment_id: string; comparisons: InterpretabilityCompareItem[] }> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/interpretability/batch-compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analyses }),
  });
  if (!res.ok) throw new Error(`Batch compare interpretability failed: ${res.statusText}`);
  return res.json();
}

export async function exportInterpretabilityReport(
  experimentId: string,
  method: AnalysisMethod,
  numSamples: number
): Promise<void> {
  const url = `${API_BASE}/experiments/${experimentId}/interpretability/export?method=${method}&num_samples=${numSamples}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Export interpretability report failed: ${res.statusText}`);
  const blob = await res.blob();
  const disposition = res.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename="?([^"]+)"?/);
  const filename = match ? match[1] : `${experimentId}_${method}_${numSamples}_${Date.now()}.json`;
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(link.href);
}

export function getInterpretabilityWebSocketUrl(experimentId: string): string {
  const wsBase = import.meta.env.VITE_WS_URL ||
    (window.location.protocol === "https:" ? "wss:" : "ws:") + "//" + window.location.host;
  return `${wsBase}/ws/experiments/${experimentId}/interpretability`;
}
