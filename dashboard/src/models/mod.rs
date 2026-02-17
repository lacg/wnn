//! Data models for experiment tracking
//!
//! Unified models (no V1/V2 distinction)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Status Enums
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FlowStatus {
    Pending,
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

impl Default for FlowStatus {
    fn default() -> Self {
        FlowStatus::Pending
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStatus {
    Pending,
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

impl Default for ExperimentStatus {
    fn default() -> Self {
        ExperimentStatus::Pending
    }
}

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GenomeRole {
    Elite,
    Offspring,
    Init,
    // TS-specific roles
    TopK,
    Neighbor,
    Current,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointType {
    Auto,
    User,
    ExperimentEnd,
}

impl Default for CheckpointType {
    fn default() -> Self {
        CheckpointType::Auto
    }
}

// =============================================================================
// Architecture Type
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ArchitectureType {
    Tiered,
    Bitwise,
}

impl Default for ArchitectureType {
    fn default() -> Self {
        ArchitectureType::Tiered
    }
}

// =============================================================================
// Flow Models
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
    /// Last heartbeat from worker (for detecting stale running flows)
    pub last_heartbeat: Option<DateTime<Utc>>,
}

/// Flow configuration - flow-level settings only
/// Experiments are stored in the experiments table, not here (normalized design)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowConfig {
    /// Template name (e.g., "standard-6-phase")
    pub template: Option<String>,
    /// Additional parameters (tier_config, generations, patience, etc)
    #[serde(default)]
    pub params: HashMap<String, serde_json::Value>,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
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
    GridSearch,
    Neurogenesis,
    Synaptogenesis,
    Axonogenesis,
}

// =============================================================================
// Experiment Models
// =============================================================================

/// An experiment is a single optimization run (e.g., "GA Neurons", "TS Bits")
/// In the simplified model, each config spec becomes its own experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: i64,
    pub flow_id: Option<i64>,
    pub sequence_order: Option<i32>,
    pub name: String,
    pub status: ExperimentStatus,
    pub fitness_calculator: FitnessCalculator,
    pub fitness_weight_ce: f64,
    pub fitness_weight_acc: f64,
    pub tier_config: Option<String>,
    pub context_size: i32,
    pub population_size: i32,
    pub pid: Option<i32>,
    pub last_iteration: Option<i32>,
    pub resume_checkpoint_id: Option<i64>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub paused_at: Option<DateTime<Utc>>,
    /// Experiment type (e.g., "ga_neurons", "ts_bits")
    pub phase_type: Option<String>,
    pub max_iterations: Option<i32>,
    pub current_iteration: Option<i32>,
    pub best_ce: Option<f64>,
    pub best_accuracy: Option<f64>,
    /// Architecture type: tiered or bitwise
    #[serde(default)]
    pub architecture_type: ArchitectureType,
    /// Gating analysis status: NULL (not run), 'pending', 'running', 'completed', 'failed'
    pub gating_status: Option<GatingStatus>,
    /// Gating analysis results (JSON blob)
    pub gating_results: Option<GatingResults>,
}

/// Status of gating analysis for an experiment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GatingStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

/// Results of gating analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingResults {
    pub completed_at: Option<DateTime<Utc>>,
    pub genomes_tested: usize,
    pub results: Vec<GatingResult>,
    pub error: Option<String>,
}

/// A single gating analysis run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingRun {
    pub id: i64,
    pub experiment_id: i64,
    pub status: GatingStatus,
    pub config: Option<GatingConfig>,
    pub genomes_tested: Option<i32>,
    pub results: Option<Vec<GatingResult>>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Result for a single genome in gating analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingResult {
    pub genome_type: String,  // "best_ce", "best_acc", "best_fitness"
    pub ce: f64,
    pub acc: f64,
    pub gated_ce: f64,
    pub gated_acc: f64,
    pub gating_config: GatingConfig,
}

/// Configuration used for gating
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingConfig {
    pub neurons_per_gate: usize,
    pub bits_per_neuron: usize,
    pub threshold: f64,
}

// =============================================================================
// Iteration Models
// =============================================================================

/// A generation/iteration within an experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Iteration {
    pub id: i64,
    pub experiment_id: i64,
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
    // Delta and patience tracking
    pub baseline_ce: Option<f64>,
    pub delta_baseline: Option<f64>,
    pub delta_previous: Option<f64>,
    pub patience_counter: Option<i32>,
    pub patience_max: Option<i32>,
    pub candidates_total: Option<i32>,
    pub created_at: DateTime<Utc>,
}

// =============================================================================
// Genome Models
// =============================================================================

/// Unique genome configuration (deduplicated by config hash)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Genome {
    pub id: i64,
    pub experiment_id: i64,
    pub config_hash: String,
    pub tiers_json: String,
    pub total_clusters: i32,
    pub total_neurons: i32,
    pub total_memory_bytes: i64,
    /// Architecture type: tiered or bitwise
    #[serde(default)]
    pub architecture_type: ArchitectureType,
    /// Serialized connections for HF export (compressed JSON)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connections_json: Option<String>,
    /// Full WNNConfig as JSON (for direct HF export)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_config_json: Option<String>,
    /// Path to exported HF model directory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_export_path: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Tier configuration for a genome
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct TierConfig {
    pub tier: i32,
    pub clusters: i32,
    pub neurons: i32,
    pub bits: i32,
}

/// Per-iteration evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeEvaluation {
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

// =============================================================================
// Health Check Models
// =============================================================================

/// Periodic full validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
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

// =============================================================================
// Validation Summary Models
// =============================================================================

/// Validation point in the experiment flow
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationPoint {
    Init,
    Final,
}

/// Type of genome being validated
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GenomeValidationType {
    BestCe,
    BestAcc,
    BestFitness,
}

/// Full-dataset validation result for a single genome at a checkpoint
/// Deduplication: genome_hash is used to skip re-validation of already-validated genomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub id: i64,
    pub flow_id: Option<i64>,
    pub experiment_id: i64,
    pub validation_point: ValidationPoint,
    pub genome_type: GenomeValidationType,
    pub genome_hash: String,
    pub ce: f64,
    pub accuracy: f64,
    pub created_at: DateTime<Utc>,
}

// =============================================================================
// Checkpoint Models
// =============================================================================

/// Saved state for resume
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: i64,
    pub experiment_id: i64,
    pub iteration_id: Option<i64>,
    pub name: String,
    pub file_path: String,
    pub file_size_bytes: Option<i64>,
    pub checkpoint_type: CheckpointType,
    pub best_ce: Option<f64>,
    pub best_accuracy: Option<f64>,
    /// Genome statistics including per-tier stats
    pub genome_stats: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    /// Flow info (from joined experiment)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flow_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flow_name: Option<String>,
}

// =============================================================================
// WebSocket Messages
// =============================================================================

/// Real-time update message for WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WsMessage {
    /// Full state snapshot for new clients
    Snapshot(DashboardSnapshot),
    /// New iteration completed
    IterationCompleted(Iteration),
    /// Genome evaluations for an iteration
    GenomeEvaluations { iteration_id: i64, evaluations: Vec<GenomeEvaluation> },
    /// Health check result
    HealthCheck(HealthCheck),
    /// Experiment status changed
    ExperimentStatusChanged(Experiment),
    /// Flow lifecycle events
    FlowStarted(Flow),
    FlowCompleted(Flow),
    FlowFailed { flow: Flow, error: String },
    FlowCancelled(Flow),
    FlowQueued(Flow),
    /// Checkpoint events
    CheckpointCreated(Checkpoint),
    CheckpointDeleted { id: i64 },
    /// Gating run events
    GatingRunCreated(GatingRun),
    GatingRunUpdated(GatingRun),
}

/// Full dashboard state snapshot for new clients
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardSnapshot {
    pub current_experiment: Option<Experiment>,
    pub iterations: Vec<Iteration>,
    pub best_ce: f64,
    pub best_accuracy: f64,
}
