"""
Flow-parameter registry + ingestion validation (task 3.2,
docs/ARCHITECTURE_REVIEW_2026-06.md).

The dashboard sends flows.config_json.params as a free-form dict; before this
module, ~50 scattered `.get(key, default)` reads meant a typo'd key silently
fell back to a default — invisible, and defaults DIFFERED between read sites
(ids_k_folds 5 vs 1; min_bits 10 vs 4 per architecture branch).

`validate_flow_params()` is called once at worker ingestion:
- WARNS LOUDLY on unknown keys (typos no longer vanish). Warning, not error —
  rejecting would break live dashboards on a registry gap; the log line makes
  the gap visible so it gets added.
- logs the subset of params that were explicitly provided (the run's
  effective config, for reproducibility).

KNOWN_PARAMS is the single registry. When adding a `.get()` read anywhere in
worker/flow, add the key here — unknown-key warnings in worker logs are the
reminder.
"""

# Every key legitimately read from flows.config_json.params (worker.py,
# flow.py, experiment.py). Grouped by domain for maintainability.
KNOWN_PARAMS: frozenset[str] = frozenset({
	# Core architecture/search
	"architecture_type", "num_clusters", "context_size", "memory_mode",
	"min_bits", "max_bits", "min_neurons", "max_neurons", "max_bit_delta",
	"neuron_sample_rate", "sparse_threshold", "seed",
	# GA / TS / SA / grid
	"population_size", "ga_generations", "ts_iterations", "neighbors_per_iter",
	"patience", "check_interval", "mutation_rate", "magnitude_aware_patience",
	"random_search",
	"neurons_grid", "bits_grid", "grid_top_k", "grid_source", "top_m",
	"cluster_crossover_ratio", "pool_shuffle_ratio", "assortative_mating_ratio",
	"adaptation_iterations", "genesis_mode", "start_from_experiment",
	# Fitness
	"fitness_calculator", "fitness_percentile", "min_accuracy_floor",
	"fitness_weight_ce", "fitness_weight_acc", "fitness_weight_f1", "fitness_weight_fpr",
	"threshold_start", "threshold_step",
	# Data partitioning
	"train_parts", "eval_parts", "full_train_parts", "full_eval_parts",
	# Tiers / stages (LM multistage + tiered)
	"tier_config", "tier0_only", "optimize_tier0_only",
	"num_stages", "stage_k", "stage_mode", "stage_context_size",
	"stage_cluster_type", "stage_min_bits", "stage_max_bits",
	"stage_min_neurons", "stage_max_neurons", "stage_neurons_grid", "stage_bits_grid",
	"tiered_neurons_grid", "tiered_bits_grid",
	"bigram_lambda", "unigram_lambda", "label_smoothing",
	"invalid_mode", "flip_labels",
	# IDS
	"ids_dataset", "ids_split", "ids_arch_type", "ids_classification",
	"ids_n_bits", "ids_rest_bits", "ids_auto_max_bits", "ids_raw",
	"ids_num_parts", "ids_k_folds", "ids_kfold_per_gen", "ids_val_fraction",
	"ids_single_cluster", "ids_feature_selection", "ids_invalid_encoding",
	"ids_fitness_weight_f1", "ids_fitness_weight_fpr",
	"ids_streaming", "ids_streaming_chunk_size",
	# Multiclass (docs/MULTICLASS_DESIGN.md §3): decode rule for K-class flows —
	# "argmax" (baseline) | "benign_margin" (2-stage decode with τ threshold).
	# Consumed by the Rust K-class eval stage; registered here so flows can set
	# it without tripping the unknown-param warning.
	"decode_mode",
	"ids_encoded_storage", "ids_encoded_storage_dir", "ids_memmap_prefetch",
	"balance_classes", "class_weight_multiplier", "undersample_majority",
	"reweight_rounds", "reweight_max_boost",
	# Leaderboard seeding
	"seed_from_leaderboard", "seed_leaderboard_count", "seed_leaderboard_metric",
	"seed_leaderboard_stage", "seed_leaderboard_task_type",
	# Env-forwarded knobs (worker maps these to WNN_* env vars)
	"wnn_batch_size", "wnn_grid_search_parallel", "wnn_hybrid_speed_ratio",
	"wnn_no_metal", "wnn_num_threads", "wnn_order_independent_train",
	# Controller (drone) flows
	"controller_disturbance_level",
	"controller_memory_mode",  # ABI 12 granularity ablation (QUAD_WEIGHTED/TERNARY/BINARY)
	# Scope C stage 1 (13/08/2026): the vertical channel. Registered here so a
	# dashboard-launched controller flow can carry them; the phased_ga CLI takes
	# the same knobs as --translation / --obs-* / --fit-weight-alt.
	"controller_translation",         # integrate vertical dynamics in the sim
	"controller_obs_collective_cmd",  # feature: commanded collective from the outer loop
	"controller_obs_alt_err",         # feature: altitude error (target − z)
	"controller_obs_vz",              # feature: vertical velocity
	"controller_fit_weight_alt",      # reward weight on altitude error (sweep, never guessed)
	# Scope C stage 2 (14/08/2026): the horizontal channel, same rationale.
	"controller_obs_pos_err_xy",      # feature: horizontal position error (target − p), world frame
	"controller_obs_vel_xy",          # feature: horizontal velocity
	"controller_xy_offset",           # plant: max |initial xy offset| in m (0 ⇒ axis disarmed)
	"controller_fit_weight_pos",      # reward weight on radial position error (sweep, never guessed)
	# Specialist programme (14/08/2026)
	"controller_conn_policy",         # connection-creation policy: spread | min2 | min3
	"controller_output_full_window",  # arm D: output layer samples ALL k frames, not just t-0
})


def validate_flow_params(params: dict, log=print, flow_id=None) -> list[str]:
	"""Validate a flow's params dict at ingestion. Returns unknown keys.

	- Unknown keys produce a LOUD warning naming the likely intent (a typo'd
	  key previously fell back to a default silently — the Rule-6 bug class).
	- The provided (non-default) config is logged once for reproducibility.
	"""
	prefix = f"[PARAMS]{f' flow={flow_id}' if flow_id is not None else ''}"
	unknown = sorted(k for k in params if k not in KNOWN_PARAMS)
	if unknown:
		import difflib
		for k in unknown:
			suggestion = difflib.get_close_matches(k, KNOWN_PARAMS, n=1)
			hint = f" — did you mean '{suggestion[0]}'?" if suggestion else ""
			log(
				f"{prefix} ⚠️  UNKNOWN PARAM '{k}'{hint} It will be IGNORED and "
				f"defaults will apply. If legitimate, add it to "
				f"wnn/ram/experiments/params.py KNOWN_PARAMS."
			)
	if params:
		provided = ", ".join(f"{k}={params[k]!r}" for k in sorted(params) if k in KNOWN_PARAMS)
		log(f"{prefix} resolved config ({len(params)} keys): {provided}")
	return unknown
