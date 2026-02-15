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

export type PhaseStatus = 'pending' | 'running' | 'completed' | 'failed';

// =============================================================================
// Phase types (used by watcher for multi-phase experiment tracking)
// =============================================================================

export interface Phase {
  id: number;
  experiment_id: number;
  name: string;
  phase_type: string;
  sequence_order: number;
  max_iterations: number;
  status: PhaseStatus;
  started_at: string | null;
  ended_at: string | null;
}

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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  params: Record<string, any>;
  // Legacy: some older flows stored experiments inline in config
  experiments?: ExperimentSpec[];
}

export interface ExperimentSpec {
  name: string;
  experiment_type: 'ga' | 'ts' | 'grid_search';
  optimize_bits: boolean;
  optimize_neurons: boolean;
  optimize_connections: boolean;
  params: Record<string, unknown>;
}

// =============================================================================
// Experiment types
// =============================================================================

export type GatingStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface GatingConfig {
  neurons_per_gate: number;
  bits_per_neuron: number;
  threshold: number;
}

export interface GatingResult {
  genome_type: string;  // 'best_ce', 'best_acc', 'best_fitness'
  ce: number;
  acc: number;
  gated_ce: number;
  gated_acc: number;
  gating_config: GatingConfig;
}

export interface GatingResults {
  completed_at: string | null;
  genomes_tested: number;
  results: GatingResult[];
  error: string | null;
}

export interface GatingRun {
  id: number;
  experiment_id: number;
  status: GatingStatus;
  config: GatingConfig | null;
  genomes_tested: number | null;
  results: GatingResult[] | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

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
  // Architecture type
  architecture_type?: 'tiered' | 'bitwise';
  // Gating analysis
  gating_status: GatingStatus | null;
  gating_results: GatingResults | null;
  // Cluster architecture type (legacy, prefer architecture_type)
  cluster_type?: 'tiered' | 'bitwise';
}

// =============================================================================
// Iteration types
// =============================================================================

export interface Iteration {
  id: number;
  experiment_id: number;
  phase_id?: number;
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

export interface TierStats {
  tier_index: number;
  cluster_count: number;
  start_cluster: number;
  end_cluster: number;
  avg_bits: number;
  avg_neurons: number;
  min_bits: number;
  max_bits: number;
  min_neurons: number;
  max_neurons: number;
  total_neurons: number;
  total_connections: number;
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
// Validation Summary types
// =============================================================================

export type ValidationPoint = 'init' | 'final';
export type GenomeValidationType = 'best_ce' | 'best_acc' | 'best_fitness';

export interface ValidationSummary {
  id: number;
  flow_id: number | null;
  experiment_id: number;
  validation_point: ValidationPoint;
  genome_type: GenomeValidationType;
  genome_hash: string;
  ce: number;
  accuracy: number;
  created_at: string;
}

// =============================================================================
// Checkpoint types
// =============================================================================

export interface BitwiseClusterStat {
  cluster: number;
  bits: number;
  neurons: number;
  connections: number;
  memory_words: number;
}

export interface GenomeStats {
  num_clusters: number;
  total_neurons: number;
  total_connections: number;
  bits_range: [number, number];
  neurons_range: [number, number];
  tier_stats?: TierStats[];
  cluster_stats?: BitwiseClusterStat[];
}

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
  genome_stats: GenomeStats | null;
  created_at: string;
  // Flow info (from joined experiment)
  flow_id?: number | null;
  flow_name?: string | null;
  // Optional fields used by checkpoints page
  is_final?: boolean;
  final_fitness?: number | null;
  final_accuracy?: number | null;
  iterations_run?: number | null;
  reference_count?: number;
}

// =============================================================================
// Dashboard snapshot
// =============================================================================

export interface DashboardSnapshot {
  current_experiment: Experiment | null;
  iterations: Iteration[];
  best_ce: number;
  best_accuracy: number;
  // Phase tracking (sent by watcher when multi-phase experiments are active)
  phases?: Phase[];
  current_phase?: Phase | null;
}

// =============================================================================
// WebSocket messages
// =============================================================================

export type WsMessage =
  | { type: 'Snapshot'; data: DashboardSnapshot }
  | { type: 'IterationCompleted'; data: Iteration }
  | { type: 'HealthCheck'; data: HealthCheck }
  | { type: 'ExperimentStatusChanged'; data: Experiment }
  | { type: 'PhaseStarted'; data: Phase }
  | { type: 'PhaseCompleted'; data: Phase }
  | { type: 'FlowStarted'; data: Flow }
  | { type: 'FlowCompleted'; data: Flow }
  | { type: 'FlowFailed'; data: { flow: Flow; error: string } }
  | { type: 'FlowCancelled'; data: Flow }
  | { type: 'FlowQueued'; data: Flow }
  | { type: 'CheckpointCreated'; data: Checkpoint }
  | { type: 'CheckpointDeleted'; data: { id: number } }
  | { type: 'GatingRunCreated'; data: GatingRun }
  | { type: 'GatingRunUpdated'; data: GatingRun };
