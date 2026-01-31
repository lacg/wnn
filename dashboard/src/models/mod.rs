//! Data models for experiment tracking

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An experiment run (one invocation of phased_search)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: i64,
    pub name: String,
    pub log_path: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub status: ExperimentStatus,
    pub config: ExperimentConfig,
    /// Process ID of the running experiment (for signal handling)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<i32>,
    /// Last completed checkpoint phase (e.g., "1a", "2b")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_phase: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExperimentConfig {
    pub ga_generations: Option<i32>,
    pub ts_iterations: Option<i32>,
    pub population_size: Option<i32>,
    pub neighbor_count: Option<i32>,
    pub patience: Option<i32>,
    pub tier_config: Option<String>,
}

/// A phase within an experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase {
    pub id: i64,
    pub experiment_id: i64,
    pub name: String,
    pub phase_type: PhaseType,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub status: PhaseStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PhaseType {
    GaNeurons,
    TsNeurons,
    GaBits,
    TsBits,
    GaConnections,
    TsConnections,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PhaseStatus {
    Running,
    Completed,
    Skipped,
}

/// A single generation/iteration within a phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Iteration {
    pub id: i64,
    pub phase_id: i64,
    pub iteration_num: i32,
    pub best_ce: f64,
    pub avg_ce: Option<f64>,
    pub best_accuracy: Option<f64>,
    pub elapsed_secs: f64,
    pub timestamp: DateTime<Utc>,
}

/// Metrics snapshot from health check (full validation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub id: i64,
    pub phase_id: i64,
    pub top_k_ce: f64,
    pub top_k_accuracy: f64,
    pub best_ce: f64,
    pub best_ce_accuracy: f64,
    pub best_acc_ce: f64,
    pub best_acc_accuracy: f64,
    pub k: i32,
    pub timestamp: DateTime<Utc>,
}

/// Final phase result for comparison table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseResult {
    pub id: i64,
    pub phase_id: i64,
    pub metric_type: MetricType,
    pub ce: f64,
    pub accuracy: f64,
    pub memory_bytes: i64,
    pub improvement_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MetricType {
    TopKMean,
    BestCe,
    BestAcc,
}

/// Row from final phase comparison summary table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseSummaryRow {
    pub phase_name: String,
    pub metric_type: String,
    pub ce: f64,
    pub ppl: f64,
    pub accuracy: f64,
}

/// Complete phase comparison summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseSummary {
    pub rows: Vec<PhaseSummaryRow>,
    pub timestamp: DateTime<Utc>,
}

/// Real-time update message for WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WsMessage {
    /// Initial snapshot sent to new clients
    Snapshot(DashboardSnapshot),
    IterationUpdate(Iteration),
    PhaseStarted(Phase),
    PhaseCompleted { phase: Phase, result: PhaseResult },
    HealthCheck(HealthCheck),
    ExperimentCompleted(Experiment),
    /// Flow lifecycle events
    FlowStarted(Flow),
    FlowCompleted(Flow),
    FlowFailed { flow: Flow, error: String },
    FlowCancelled(Flow),
    FlowQueued(Flow),
    /// Checkpoint events
    CheckpointCreated(Checkpoint),
    CheckpointDeleted { id: i64 },
    /// Final phase comparison summary
    PhaseSummary(PhaseSummary),
}

/// Full dashboard state snapshot for new clients
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardSnapshot {
    pub phases: Vec<Phase>,
    pub current_phase: Option<Phase>,
    pub iterations: Vec<Iteration>,  // Last N iterations of current phase
    pub best_ce: f64,
    pub best_ce_acc: f64,
    pub best_acc: f64,
    pub best_acc_ce: f64,
}

// =============================================================================
// Flow and Checkpoint models
// =============================================================================

/// A flow is a sequence of experiments (like a multi-pass search)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flow {
    pub id: i64,
    pub name: String,
    pub description: Option<String>,
    pub config: FlowConfig,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub status: FlowStatus,
    pub seed_checkpoint_id: Option<i64>,
    pub pid: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowConfig {
    /// List of experiment configurations in sequence order
    pub experiments: Vec<ExperimentSpec>,
    /// Template name (e.g., "standard-6-phase")
    pub template: Option<String>,
    /// Additional parameters
    #[serde(default)]
    pub params: HashMap<String, serde_json::Value>,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            experiments: vec![],
            template: None,
            params: HashMap::new(),
        }
    }
}

/// Specification for a single experiment within a flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSpec {
    pub name: String,
    pub experiment_type: ExperimentType,
    #[serde(default)]
    pub optimize_bits: bool,
    #[serde(default)]
    pub optimize_neurons: bool,
    #[serde(default)]
    pub optimize_connections: bool,
    #[serde(default)]
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentType {
    Ga,
    Ts,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FlowStatus {
    Pending,
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl Default for FlowStatus {
    fn default() -> Self {
        FlowStatus::Pending
    }
}

/// A checkpoint is a saved state from an experiment phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: i64,
    pub experiment_id: i64,
    pub name: String,
    pub file_path: String,
    pub file_size_bytes: Option<i64>,
    pub created_at: DateTime<Utc>,
    pub final_fitness: Option<f64>,
    pub final_accuracy: Option<f64>,
    pub iterations_run: Option<i32>,
    pub genome_stats: Option<GenomeStats>,
    pub is_final: bool,
    pub reference_count: i32,
}

/// Summary statistics for a genome (for display without loading full file)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeStats {
    pub num_clusters: i32,
    pub total_neurons: i64,
    pub total_connections: i64,
    pub bits_range: (i32, i32),
    pub neurons_range: (i32, i32),
}

/// Mapping between flows and experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowExperiment {
    pub id: i64,
    pub flow_id: i64,
    pub experiment_id: i64,
    pub sequence_order: i32,
}

/// Reference tracking for safe checkpoint deletion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointReference {
    pub id: i64,
    pub checkpoint_id: i64,
    pub referencing_experiment_id: i64,
    pub reference_type: String,
}

// =============================================================================
// V2 Models: New data model with DB as source of truth
// =============================================================================

/// Flow status (v2)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FlowStatusV2 {
    Pending,
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

impl Default for FlowStatusV2 {
    fn default() -> Self {
        FlowStatusV2::Pending
    }
}

/// Experiment status (v2)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStatusV2 {
    Pending,
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

impl Default for ExperimentStatusV2 {
    fn default() -> Self {
        ExperimentStatusV2::Pending
    }
}

/// Phase status (v2)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PhaseStatusV2 {
    Pending,
    Running,
    Paused,
    Completed,
    Skipped,
    Failed,
}

impl Default for PhaseStatusV2 {
    fn default() -> Self {
        PhaseStatusV2::Pending
    }
}

/// Fitness calculator type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FitnessCalculator {
    Ce,
    HarmonicRank,
    WeightedHarmonic,
}

impl Default for FitnessCalculator {
    fn default() -> Self {
        FitnessCalculator::HarmonicRank
    }
}

/// Genome evaluation role
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GenomeRole {
    Elite,
    Offspring,
    Init,
}

/// Checkpoint type (v2)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointTypeV2 {
    Auto,
    User,
    PhaseEnd,
    ExperimentEnd,
}

/// Flow (v2) - A sequence of experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowV2 {
    pub id: i64,
    pub name: String,
    pub description: Option<String>,
    pub status: FlowStatusV2,
    pub config_json: String,
    pub seed_checkpoint_id: Option<i64>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Experiment (v2) - A single optimization run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentV2 {
    pub id: i64,
    pub flow_id: Option<i64>,
    pub sequence_order: Option<i32>,
    pub name: String,
    pub status: ExperimentStatusV2,
    pub fitness_calculator: FitnessCalculator,
    pub fitness_weight_ce: f64,
    pub fitness_weight_acc: f64,
    pub tier_config: Option<String>,
    pub context_size: i32,
    pub population_size: i32,
    pub pid: Option<i32>,
    pub last_phase_id: Option<i64>,
    pub last_iteration: Option<i32>,
    pub resume_checkpoint_id: Option<i64>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub paused_at: Option<DateTime<Utc>>,
}

/// Phase (v2) - A phase within an experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseV2 {
    pub id: i64,
    pub experiment_id: i64,
    pub name: String,
    pub phase_type: String,
    pub sequence_order: i32,
    pub status: PhaseStatusV2,
    pub max_iterations: i32,
    pub population_size: Option<i32>,
    pub current_iteration: i32,
    pub best_ce: Option<f64>,
    pub best_accuracy: Option<f64>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
}

/// Iteration (v2) - A generation within a phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationV2 {
    pub id: i64,
    pub phase_id: i64,
    pub iteration_num: i32,
    pub best_ce: f64,
    pub best_accuracy: Option<f64>,
    pub avg_ce: Option<f64>,
    pub avg_accuracy: Option<f64>,
    pub elite_count: Option<i32>,
    pub offspring_count: Option<i32>,
    pub offspring_viable: Option<i32>,
    pub fitness_threshold: Option<f64>,
    pub elapsed_secs: Option<f64>,
    pub created_at: DateTime<Utc>,
}

/// Genome (v2) - Unique genome configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeV2 {
    pub id: i64,
    pub experiment_id: i64,
    pub config_hash: String,
    pub tiers_json: String,
    pub total_clusters: i32,
    pub total_neurons: i32,
    pub total_memory_bytes: i64,
    pub created_at: DateTime<Utc>,
}

/// Tier configuration for a genome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    pub tier: i32,
    pub clusters: i32,
    pub neurons: i32,
    pub bits: i32,
}

/// Genome evaluation (v2) - Per-iteration evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeEvaluationV2 {
    pub id: i64,
    pub iteration_id: i64,
    pub genome_id: i64,
    pub position: i32,
    pub role: GenomeRole,
    pub elite_rank: Option<i32>,
    pub ce: f64,
    pub accuracy: f64,
    pub fitness_score: Option<f64>,
    pub eval_time_ms: Option<i32>,
    pub created_at: DateTime<Utc>,
}

/// Health check (v2) - Periodic full validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckV2 {
    pub id: i64,
    pub iteration_id: i64,
    pub k: i32,
    pub top_k_ce: f64,
    pub top_k_accuracy: f64,
    pub best_ce: Option<f64>,
    pub best_ce_accuracy: Option<f64>,
    pub best_acc_ce: Option<f64>,
    pub best_acc_accuracy: Option<f64>,
    pub patience_remaining: Option<i32>,
    pub patience_status: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Checkpoint (v2) - Saved state for resume
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointV2 {
    pub id: i64,
    pub experiment_id: i64,
    pub phase_id: Option<i64>,
    pub iteration_id: Option<i64>,
    pub name: String,
    pub file_path: String,
    pub file_size_bytes: Option<i64>,
    pub checkpoint_type: CheckpointTypeV2,
    pub best_ce: Option<f64>,
    pub best_accuracy: Option<f64>,
    pub created_at: DateTime<Utc>,
}

// =============================================================================
// V2 WebSocket Messages
// =============================================================================

/// Real-time update message for WebSocket (v2)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WsMessageV2 {
    /// Full state snapshot for new clients
    Snapshot(DashboardSnapshotV2),
    /// New iteration completed
    IterationCompleted(IterationV2),
    /// Genome evaluations for an iteration
    GenomeEvaluations { iteration_id: i64, evaluations: Vec<GenomeEvaluationV2> },
    /// Phase started
    PhaseStarted(PhaseV2),
    /// Phase completed
    PhaseCompleted(PhaseV2),
    /// Health check result
    HealthCheck(HealthCheckV2),
    /// Experiment status changed
    ExperimentStatusChanged(ExperimentV2),
    /// Flow status changed
    FlowStatusChanged(FlowV2),
    /// Checkpoint created
    CheckpointCreated(CheckpointV2),
}

/// Full dashboard state snapshot (v2)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardSnapshotV2 {
    pub current_experiment: Option<ExperimentV2>,
    pub current_phase: Option<PhaseV2>,
    pub phases: Vec<PhaseV2>,
    pub iterations: Vec<IterationV2>,
    pub best_ce: f64,
    pub best_accuracy: f64,
}
