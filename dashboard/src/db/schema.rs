//! Database schema DDL (split from db/mod.rs).

pub(crate) const SCHEMA: &str = r#"
-- ============================================================================
-- FLOWS: A sequence of experiments (multi-pass search)
-- ============================================================================
CREATE TABLE IF NOT EXISTS flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    -- pending, queued, running, paused, completed, failed, cancelled

    -- Configuration
    config_json TEXT NOT NULL DEFAULT '{}',

    -- Seed checkpoint (optional starting point)
    seed_checkpoint_id INTEGER REFERENCES checkpoints(id),

    -- Timing
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    started_at TEXT,
    completed_at TEXT
);

-- ============================================================================
-- EXPERIMENTS: A single optimization run (GA/TS search)
-- Each experiment in a flow represents one optimization stage (e.g., GA-Neurons, TS-Bits)
-- ============================================================================
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id INTEGER REFERENCES flows(id),
    sequence_order INTEGER,

    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    -- pending, queued, running, paused, completed, failed, cancelled

    -- Experiment type (was phase_type before simplification)
    phase_type TEXT,
    -- ga_neurons, ts_neurons, ga_bits, ts_bits, ga_connections, ts_connections

    -- Configuration
    fitness_calculator TEXT NOT NULL DEFAULT 'harmonic_rank',
    fitness_weight_ce REAL DEFAULT 1.0,
    fitness_weight_acc REAL DEFAULT 1.0,
    tier_config TEXT,
    context_size INTEGER DEFAULT 4,
    population_size INTEGER DEFAULT 50,
    max_iterations INTEGER DEFAULT 250,

    -- Process tracking
    pid INTEGER,

    -- Progress
    current_iteration INTEGER DEFAULT 0,
    best_ce REAL,
    best_accuracy REAL,

    -- Resume state
    last_iteration INTEGER,
    resume_checkpoint_id INTEGER REFERENCES checkpoints(id),

    -- Timing
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    started_at TEXT,
    ended_at TEXT,
    paused_at TEXT
);

-- ============================================================================
-- ITERATIONS: A generation/iteration within an experiment
-- ============================================================================
CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    iteration_num INTEGER NOT NULL,

    -- Summary metrics (best of this iteration)
    best_ce REAL NOT NULL,
    best_accuracy REAL,
    avg_ce REAL,
    avg_accuracy REAL,

    -- IDS metrics (NULL for LM experiments)
    best_f1 REAL,
    best_fpr REAL,
    -- Controller metric (NULL for IDS/LM): mean attitude error in degrees
    mean_attitude_error_deg REAL,

    -- Population info
    elite_count INTEGER,
    offspring_count INTEGER,
    offspring_viable INTEGER,

    -- Fitness threshold (progressive filtering)
    fitness_threshold REAL,

    -- Delta tracking
    baseline_ce REAL,
    delta_baseline REAL,
    delta_previous REAL,

    -- Patience tracking
    patience_counter INTEGER,
    patience_max INTEGER,
    candidates_total INTEGER,

    -- Timing
    elapsed_secs REAL,

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    UNIQUE(experiment_id, iteration_num)
);

-- ============================================================================
-- GENOMES: Unique genome configurations (deduplicated by config hash)
-- ============================================================================
CREATE TABLE IF NOT EXISTS genomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),

    -- Configuration identity (for deduplication)
    config_hash TEXT NOT NULL,

    -- Per-tier configuration as JSON
    tiers_json TEXT NOT NULL,
    -- Example: [{"tier": 0, "clusters": 100, "neurons": 15, "bits": 20}, ...]

    -- Connection-inclusive hash (NULL for architecture-only rows, populated for leaderboard genomes)
    genome_hash TEXT,

    -- Aggregates (computed from tiers)
    total_clusters INTEGER NOT NULL,
    total_neurons INTEGER NOT NULL,
    total_memory_bytes INTEGER NOT NULL,

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    UNIQUE(experiment_id, config_hash)
);

-- ============================================================================
-- GENOME_EVALUATIONS: Per-iteration evaluation results
-- ============================================================================
CREATE TABLE IF NOT EXISTS genome_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id INTEGER NOT NULL REFERENCES iterations(id),
    genome_id INTEGER NOT NULL REFERENCES genomes(id),

    -- Position in generation
    position INTEGER NOT NULL,

    -- Role in this iteration
    role TEXT NOT NULL,
    -- 'elite', 'offspring', 'init'
    elite_rank INTEGER,

    -- Evaluation results
    ce REAL NOT NULL,
    accuracy REAL NOT NULL,
    fitness_score REAL,
    f1_macro REAL,        -- IDS: F1-macro score (NULL for LM experiments)
    fpr REAL,             -- IDS: False positive rate (NULL for LM experiments)

    -- Timing
    eval_time_ms INTEGER,

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    -- Prevent duplicate evaluations for same iteration+position
    UNIQUE(iteration_id, position)
);

-- ============================================================================
-- HEALTH_CHECKS: Periodic full validation
-- ============================================================================
CREATE TABLE IF NOT EXISTS health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id INTEGER NOT NULL REFERENCES iterations(id),

    -- Top-K ensemble metrics
    k INTEGER NOT NULL,
    top_k_ce REAL NOT NULL,
    top_k_accuracy REAL NOT NULL,

    -- Best individual metrics
    best_ce REAL,
    best_ce_accuracy REAL,
    best_acc_ce REAL,
    best_acc_accuracy REAL,

    -- Patience tracking
    patience_remaining INTEGER,
    patience_status TEXT,

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ============================================================================
-- VALIDATION_SUMMARIES: Full-dataset validation results per genome
-- Each record = one genome validated at a checkpoint (init/final of an experiment)
-- Deduplication: if genome_hash already exists, skip expensive validation and reuse cached values
-- ============================================================================
CREATE TABLE IF NOT EXISTS validation_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id INTEGER REFERENCES flows(id),
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    validation_point TEXT NOT NULL,   -- 'init' or 'final'
    genome_type TEXT NOT NULL,        -- 'best_ce', 'best_acc', 'best_fitness'
    genome_hash TEXT NOT NULL,        -- Config hash for deduplication
    ce REAL NOT NULL,
    accuracy REAL NOT NULL,
    f1_macro REAL,                    -- IDS: F1-macro score (NULL for LM experiments)
    fpr REAL,                         -- IDS: False positive rate (NULL for LM experiments)
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    -- One record per genome type per checkpoint
    UNIQUE(experiment_id, validation_point, genome_type)
);

-- ============================================================================
-- CHECKPOINTS: Saved state for resume
-- ============================================================================
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    iteration_id INTEGER REFERENCES iterations(id),

    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER,

    checkpoint_type TEXT NOT NULL,
    -- 'auto', 'user', 'experiment_end'

    -- Metrics snapshot
    best_ce REAL,
    best_accuracy REAL,

    -- Genome statistics (includes per-tier stats)
    genome_stats_json TEXT,

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ============================================================================
-- GATING_RUNS: Gating analysis runs per experiment
-- ============================================================================
CREATE TABLE IF NOT EXISTS gating_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),

    status TEXT NOT NULL DEFAULT 'pending',
    -- pending, running, completed, failed

    -- Configuration used for this run
    config_json TEXT,
    -- { neurons_per_gate, bits_per_neuron, threshold, ... }

    -- Results
    genomes_tested INTEGER,
    results_json TEXT,
    -- Array of { genome_type, ce, acc, gated_ce, gated_acc, gating_config }
    error TEXT,

    -- Timing
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    started_at TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_gating_runs_experiment ON gating_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_gating_runs_status ON gating_runs(status);

-- ============================================================================
-- COMBINED_VALIDATIONS: End-to-end metrics for multi-stage flows
-- Pairs each genome type (best_ce, best_acc, best_fitness) across stages
-- ============================================================================
CREATE TABLE IF NOT EXISTS combined_validations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id INTEGER NOT NULL REFERENCES flows(id),
    genome_type TEXT NOT NULL,        -- 'best_ce', 'best_acc', 'best_fitness'
    combined_ce REAL NOT NULL,
    combined_accuracy REAL NOT NULL,
    per_stage_ce_json TEXT,           -- JSON array e.g. [1.23, 3.45]
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(flow_id, genome_type)
);

CREATE INDEX IF NOT EXISTS idx_combined_validations_flow ON combined_validations(flow_id);

-- ============================================================================
-- BEST_GENOMES: Leaderboard of best genomes across experiments
-- ============================================================================
CREATE TABLE IF NOT EXISTS best_genomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT NOT NULL,          -- 'lm', 'ids'
    stage TEXT NOT NULL,              -- 'stage_0', 'stage_1', 'combined'
    metric TEXT NOT NULL,             -- 'ce', 'accuracy', 'f1_macro'
    genome_id INTEGER NOT NULL REFERENCES genomes(id),
    genome_hash TEXT NOT NULL,        -- connection-inclusive hash
    rank INTEGER,                     -- computed async (NULL until ranked)
    ce REAL NOT NULL,
    accuracy REAL NOT NULL,
    f1_macro REAL,
    fpr REAL,
    flow_id INTEGER,
    experiment_id INTEGER,
    threshold_mode TEXT NOT NULL DEFAULT 'train_cal',
    hf_repo_id TEXT,
    hf_exported_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(task_type, stage, metric, genome_hash, threshold_mode)
);

CREATE INDEX IF NOT EXISTS idx_best_genomes_ranking
    ON best_genomes(task_type, stage, metric, rank);

-- ============================================================================
-- INDEXES for efficient queries
-- ============================================================================

-- For polling new records (change detection)
CREATE INDEX IF NOT EXISTS idx_iterations_created ON iterations(created_at);
CREATE INDEX IF NOT EXISTS idx_genome_evals_created ON genome_evaluations(created_at);
CREATE INDEX IF NOT EXISTS idx_health_checks_created ON health_checks(created_at);

-- For lookups
CREATE INDEX IF NOT EXISTS idx_experiments_flow ON experiments(flow_id);
CREATE INDEX IF NOT EXISTS idx_iterations_experiment ON iterations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_genome_evals_iteration ON genome_evaluations(iteration_id);
CREATE INDEX IF NOT EXISTS idx_genomes_experiment ON genomes(experiment_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment ON checkpoints(experiment_id);
CREATE INDEX IF NOT EXISTS idx_validation_summaries_experiment ON validation_summaries(experiment_id);
CREATE INDEX IF NOT EXISTS idx_validation_summaries_genome ON validation_summaries(genome_hash);
CREATE INDEX IF NOT EXISTS idx_validation_summaries_flow ON validation_summaries(flow_id);

-- For finding latest records per entity
CREATE INDEX IF NOT EXISTS idx_iterations_exp_num ON iterations(experiment_id, iteration_num DESC);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_flows_status ON flows(status);
"#;

