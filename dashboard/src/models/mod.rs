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
