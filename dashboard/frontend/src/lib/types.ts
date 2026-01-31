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
  | { type: 'FlowQueued'; data: Flow }
  | { type: 'FlowCancelled'; data: Flow }
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

// =============================================================================
// V2 Types: Database as source of truth
// =============================================================================

export type ExperimentStatusV2 =
  | 'pending'
  | 'queued'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type PhaseStatusV2 =
  | 'pending'
  | 'running'
  | 'paused'
  | 'completed'
  | 'skipped'
  | 'failed';

export type FitnessCalculator = 'ce' | 'harmonic_rank' | 'weighted_harmonic';

export interface ExperimentV2 {
  id: number;
  flow_id: number | null;
  sequence_order: number | null;
  name: string;
  status: ExperimentStatusV2;
  fitness_calculator: FitnessCalculator;
  fitness_weight_ce: number;
  fitness_weight_acc: number;
  tier_config: string | null;
  context_size: number;
  population_size: number;
  pid: number | null;
  last_phase_id: number | null;
  last_iteration: number | null;
  resume_checkpoint_id: number | null;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  paused_at: string | null;
}

export interface PhaseV2 {
  id: number;
  experiment_id: number;
  name: string;
  phase_type: string;
  sequence_order: number;
  status: PhaseStatusV2;
  max_iterations: number;
  population_size: number | null;
  current_iteration: number;
  best_ce: number | null;
  best_accuracy: number | null;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
}

export interface IterationV2 {
  id: number;
  phase_id: number;
  iteration_num: number;
  best_ce: number;
  best_accuracy: number | null;
  avg_ce: number | null;
  avg_accuracy: number | null;
  elite_count: number | null;
  offspring_count: number | null;
  offspring_viable: number | null;
  fitness_threshold: number | null;
  elapsed_secs: number | null;
  // New metrics for dashboard display
  baseline_ce: number | null;
  delta_baseline: number | null;
  delta_previous: number | null;
  patience_counter: number | null;
  patience_max: number | null;
  candidates_total: number | null;
  created_at: string;
}

export interface DashboardSnapshotV2 {
  current_experiment: ExperimentV2 | null;
  current_phase: PhaseV2 | null;
  phases: PhaseV2[];
  iterations: IterationV2[];
  best_ce: number;
  best_accuracy: number;
}

export type GenomeRole =
  | 'elite'
  | 'offspring'
  | 'init'
  // TS-specific roles
  | 'top_k'
  | 'neighbor'
  | 'current';

export interface GenomeEvaluationV2 {
  id: number;
  iteration_id: number;
  genome_id: number;
  position: number;
  role: GenomeRole;
  elite_rank: number | null;
  ce: number;
  accuracy: number;
  fitness_score: number | null;
  eval_time_ms: number | null;
  created_at: string;
}

export type WsMessageV2 =
  | { type: 'Snapshot'; data: DashboardSnapshotV2 }
  | { type: 'IterationCompleted'; data: IterationV2 }
  | { type: 'PhaseStarted'; data: PhaseV2 }
  | { type: 'PhaseCompleted'; data: PhaseV2 }
  | { type: 'ExperimentStatusChanged'; data: ExperimentV2 };
