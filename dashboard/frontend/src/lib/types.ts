// TypeScript types matching Rust models (unified schema)

// =============================================================================
// Status types
// =============================================================================

export type FlowStatus = 'pending' | 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';

export type ExperimentStatus =
  | 'pending'
  | 'queued'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type FitnessCalculator = 'ce' | 'harmonic_rank' | 'weighted_harmonic';

export type GenomeRole =
  | 'elite'
  | 'offspring'
  | 'init'
  // TS-specific roles
  | 'top_k'
  | 'neighbor'
  | 'current';

export type CheckpointType = 'auto' | 'user' | 'experiment_end';

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
  pid: number | null;
}

// Flow-level configuration (normalized: experiments stored in experiments table, not here)
export interface FlowConfig {
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

// =============================================================================
// Experiment types
// =============================================================================

export interface Experiment {
  id: number;
  flow_id: number | null;
  sequence_order: number | null;
  name: string;
  status: ExperimentStatus;
  fitness_calculator: FitnessCalculator;
  fitness_weight_ce: number;
  fitness_weight_acc: number;
  tier_config: string | null;
  context_size: number;
  population_size: number;
  pid: number | null;
  last_iteration: number | null;
  resume_checkpoint_id: number | null;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  paused_at: string | null;
  // Experiment type (e.g., "ga_neurons", "ts_bits")
  phase_type: string | null;
  max_iterations: number | null;
  current_iteration: number | null;
  best_ce: number | null;
  best_accuracy: number | null;
}

// =============================================================================
// Iteration types
// =============================================================================

export interface Iteration {
  id: number;
  experiment_id: number;
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
  // Delta and patience tracking
  baseline_ce: number | null;
  delta_baseline: number | null;
  delta_previous: number | null;
  patience_counter: number | null;
  patience_max: number | null;
  candidates_total: number | null;
  created_at: string;
}

// =============================================================================
// Genome types
// =============================================================================

export interface Genome {
  id: number;
  experiment_id: number;
  config_hash: string;
  tiers_json: string;
  total_clusters: number;
  total_neurons: number;
  total_memory_bytes: number;
  created_at: string;
}

export interface TierConfig {
  tier: number;
  clusters: number;
  neurons: number;
  bits: number;
}

export interface GenomeEvaluation {
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

// =============================================================================
// Health check types
// =============================================================================

export interface HealthCheck {
  id: number;
  iteration_id: number;
  k: number;
  top_k_ce: number;
  top_k_accuracy: number;
  best_ce: number | null;
  best_ce_accuracy: number | null;
  best_acc_ce: number | null;
  best_acc_accuracy: number | null;
  patience_remaining: number | null;
  patience_status: string | null;
  created_at: string;
}

// =============================================================================
// Checkpoint types
// =============================================================================

export interface Checkpoint {
  id: number;
  experiment_id: number;
  iteration_id: number | null;
  name: string;
  file_path: string;
  file_size_bytes: number | null;
  checkpoint_type: CheckpointType;
  best_ce: number | null;
  best_accuracy: number | null;
  created_at: string;
}

// =============================================================================
// Dashboard snapshot
// =============================================================================

export interface DashboardSnapshot {
  current_experiment: Experiment | null;
  iterations: Iteration[];
  best_ce: number;
  best_accuracy: number;
}

// =============================================================================
// WebSocket messages
// =============================================================================

export type WsMessage =
  | { type: 'Snapshot'; data: DashboardSnapshot }
  | { type: 'IterationCompleted'; data: Iteration }
  | { type: 'HealthCheck'; data: HealthCheck }
  | { type: 'ExperimentStatusChanged'; data: Experiment }
  | { type: 'FlowStarted'; data: Flow }
  | { type: 'FlowCompleted'; data: Flow }
  | { type: 'FlowFailed'; data: { flow: Flow; error: string } }
  | { type: 'FlowCancelled'; data: Flow }
  | { type: 'FlowQueued'; data: Flow }
  | { type: 'CheckpointCreated'; data: Checkpoint }
  | { type: 'CheckpointDeleted'; data: { id: number } };
