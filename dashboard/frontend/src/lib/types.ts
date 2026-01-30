// TypeScript types matching Rust models

export interface Experiment {
  id: number;
  name: string;
  log_path: string;
  started_at: string;
  ended_at: string | null;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  config: ExperimentConfig;
}

export interface ExperimentConfig {
  ga_generations?: number;
  ts_iterations?: number;
  population_size?: number;
  neighbor_count?: number;
  patience?: number;
  context_size?: number;
  tier_config?: string;
  tier0_only?: boolean;
}

export interface Phase {
  id: number;
  experiment_id: number;
  name: string;
  phase_type: PhaseType;
  started_at: string;
  ended_at: string | null;
  status: 'running' | 'completed' | 'skipped';
}

export type PhaseType =
  | 'ga_neurons'
  | 'ts_neurons'
  | 'ga_bits'
  | 'ts_bits'
  | 'ga_connections'
  | 'ts_connections';

export interface Iteration {
  id: number;
  phase_id: number;
  iteration_num: number;
  best_ce: number;
  avg_ce: number | null;
  best_accuracy: number | null;
  elapsed_secs: number;
  timestamp: string;
}

export interface HealthCheck {
  id: number;
  phase_id: number;
  top_k_ce: number;
  top_k_accuracy: number;
  best_ce: number;
  best_ce_accuracy: number;
  best_acc_ce: number;
  best_acc_accuracy: number;
  k: number;
  timestamp: string;
}

export interface PhaseResult {
  id: number;
  phase_id: number;
  metric_type: 'top_k_mean' | 'best_ce' | 'best_acc';
  ce: number;
  accuracy: number;
  memory_bytes: number;
  improvement_pct: number;
}

export interface PhaseSummaryRow {
  phase_name: string;
  metric_type: string;
  ce: number;
  ppl: number;
  accuracy: number;
}

export interface PhaseSummary {
  rows: PhaseSummaryRow[];
  timestamp: string;
}

export interface DashboardSnapshot {
  phases: Phase[];
  current_phase: Phase | null;
  iterations: Iteration[];
  best_ce: number;
  best_ce_acc: number;
  best_acc: number;
  best_acc_ce: number;
}

export type WsMessage =
  | { type: 'Snapshot'; data: DashboardSnapshot }
  | { type: 'IterationUpdate'; data: Iteration }
  | { type: 'PhaseStarted'; data: Phase }
  | { type: 'PhaseCompleted'; data: { phase: Phase; result: PhaseResult } }
  | { type: 'HealthCheck'; data: HealthCheck }
  | { type: 'ExperimentCompleted'; data: Experiment }
  | { type: 'FlowStarted'; data: Flow }
  | { type: 'FlowCompleted'; data: Flow }
  | { type: 'FlowFailed'; data: { flow: Flow; error: string } }
  | { type: 'CheckpointCreated'; data: Checkpoint }
  | { type: 'CheckpointDeleted'; data: { id: number } }
  | { type: 'PhaseSummary'; data: PhaseSummary };

// =============================================================================
// Flow types
// =============================================================================

export interface Flow {
  id: number;
  name: string;
  description: string | null;
  config: FlowConfig;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  status: FlowStatus;
  seed_checkpoint_id: number | null;
}

export interface FlowConfig {
  experiments: ExperimentSpec[];
  template: string | null;
  params: Record<string, unknown>;
}

export interface ExperimentSpec {
  name: string;
  experiment_type: 'ga' | 'ts';
  optimize_bits: boolean;
  optimize_neurons: boolean;
  optimize_connections: boolean;
  params: Record<string, unknown>;
}

export type FlowStatus = 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

// =============================================================================
// Checkpoint types
// =============================================================================

export interface Checkpoint {
  id: number;
  experiment_id: number;
  name: string;
  file_path: string;
  file_size_bytes: number | null;
  created_at: string;
  final_fitness: number | null;
  final_accuracy: number | null;
  iterations_run: number | null;
  genome_stats: GenomeStats | null;
  is_final: boolean;
  reference_count: number;
}

export interface GenomeStats {
  num_clusters: number;
  total_neurons: number;
  total_connections: number;
  bits_range: [number, number];
  neurons_range: [number, number];
}

// =============================================================================
// Flow-Experiment mapping
// =============================================================================

export interface FlowExperiment {
  id: number;
  flow_id: number;
  experiment_id: number;
  sequence_order: number;
}
