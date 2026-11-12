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
	"ids_coverage_aware", "ids_tier_bits_rule", "ids_suppress_negatives",
	# ids_write_rule (16/10/2026, worker ABI 19): "lattice" (default — BINARY
	# one-shot Set / TERNARY-PLN TRUE-wins lattice, bit-identical) | "vote"
	# (MAJORITY VOTE: signed class-weighted tally per cell, committed by sign).
	# BINARY/TERNARY/PLN only — "vote" on a QUAD-family mode is rejected.
	"ids_write_rule",
	# ids_vote_share (09/11/2026, IDS-56, worker ABI 28): float theta in (0, 0.5]
	# on a 1/100 grid, ONLY with ids_write_rule="vote" (raises otherwise). The
	# MIN-SHARE vote: an own-class visit weighs (1 - theta), any other theta
	# (times the class weight); TRUE iff the tally is > 0. Absent/None/0.5 = the
	# plain vote (bit-identical). docs/ids55_family_fix_decomposition.md Lever B.
	"ids_vote_share",
	# ids_two_cluster_fitness_f1 (16/10/2026, worker ABI 20): "macro" (default —
	# the 2-class macro F1, bit-identical) | "attack" (the attack class's F1)
	# feeds the GA fitness + patience of a binary TWO-cluster search. Rejected
	# (non-default) on single-cluster, multiclass and hierarchical flows.
	"ids_two_cluster_fitness_f1",
	# ids_offspring_folds (16/10/2026, worker ABI 20): "all" (default — every GA /
	# TS genome, offspring and immigrants included, is scored as the K-fold CV
	# mean, the paper's protocol; needs ids_kfold_per_gen == ids_k_folds) |
	# "phase" (legacy, bit-identical: offspring/immigrants/neighbours on ONE
	# phase-fixed fold; hierarchical stages single-fold).
	"ids_offspring_folds",
	# ids_calib_sample (02/11/2026, worker ABI 24): int rows. The single-cluster
	# binary SEARCH fitness calibrates each genome x fold decision threshold on a
	# FIXED stratified subsample of the fold's train rows (by target and, where
	# the dataset has it, attack category; keyed on the flow seed + fold; drawn
	# in Rust, adaptive/calib_sample.rs) instead of every row — ~72% of cicids
	# scoring time. Absent / 0 / >= rows = every row (bit-identical). Validation
	# thresholds (train_cal, val_cal, platt, beta, empirical*) never subsample.
	# Rejected (non-zero) on two-cluster, multiclass and hierarchical flows.
	"ids_calib_sample",
	# max_bit_delta: wired 08/11/2026 (IDS-54, worker ABI 26) — before, only the
	# Python mutate read it; the Rust GA/TS search derived its own delta. Now
	# FlowConfig -> evaluator._max_bit_delta -> search_offspring/_neighbors ->
	# ram_core MutationConfig.max_bit_delta. 0 = auto (10% of min+max).
	"min_bits", "max_bits", "min_neurons", "max_neurons", "max_bit_delta",
	"neuron_sample_rate", "sparse_threshold", "seed",
	# GA / TS / SA / grid
	"population_size", "ga_generations", "ts_iterations", "neighbors_per_iter",
	# mutation_rate: wired 08/11/2026 (IDS-54) -> ExperimentConfig -> GAConfig /
	# TSConfig.mutation_rate (default 0.1 = the OptimizationConfig default).
	"patience", "check_interval", "mutation_rate", "magnitude_aware_patience",
	"random_search",
	"neurons_grid", "bits_grid", "grid_top_k", "grid_source", "top_m",
	"cluster_crossover_ratio", "pool_shuffle_ratio", "assortative_mating_ratio",
	# IDS-26 GA knobs (11/10/2026), all default OFF (absent = bit-identical GA):
	# ga_dedupe_neurons (bool), immigrant_fraction (float), ga_joint_mutation (bool).
	"ga_dedupe_neurons", "immigrant_fraction", "ga_joint_mutation",
	# ga_dedupe_fill (bool, default True): False = deduped children keep only their
	# distinct neurons (may shrink, floored at max(min_neurons, 1)). Needs ga_dedupe_neurons.
	"ga_dedupe_fill",
	# ga_crossover_bits (str, default "inherit"): crossover child width rule —
	# "blend_neuron" / "blend_cluster" draw the width U[min, max] of the parents'.
	"ga_crossover_bits",
	# max_neurons_total (int, default None): cap on the genome's SUM of neurons over
	# all clusters (max_neurons caps each cluster). Over-budget grid cells are excluded;
	# GA/TS genomes shrink every cluster in proportion (Rust neuron_budget.rs).
	"max_neurons_total",
	"adaptation_iterations", "genesis_mode", "start_from_experiment",
	# Fitness
	"fitness_calculator", "fitness_percentile", "min_accuracy_floor",
	"fitness_weight_ce", "fitness_weight_acc", "fitness_weight_f1", "fitness_weight_fpr",
	# Rank-combine step (20/08/2026): harmonic (the banked WHM) | arithmetic | zscore.
	# Absent means harmonic, so every flow queued before this key stays comparable.
	"fitness_aggregation", "fitness_zrank_clamp",
	# DEPRECATED 27/08/2026: the CE anchor is now derived automatically per task
	# (ram_core NORMALIZED_CE_ANCHOR x the task's own base-rate entropy). Kept
	# registered ONLY so the 875 flows that carry it keep ingesting; nothing
	# reads it — ingestion logs "DEPRECATED and not applied" (inert_params.py).
	"fitness_ce_anchor_normalized",
	"fitness_f1_anchor", "fitness_acc_anchor",
	"threshold_start", "threshold_step",
	# Data partitioning
	"train_parts", "eval_parts", "full_train_parts", "full_eval_parts",
	# Tiers / stages (LM multistage + tiered)
	"tier_config", "tier0_only", "optimize_tier0_only",
	# Read at cli.py:353 (params.get("phase_order", "neurons_first")) and carried
	# by 5,670 flows (all default). Applied at flow CREATION (it orders the
	# experiment rows); ingestion refuses a non-default value whose rows do not
	# follow it (inert_params.enforce_phase_order, IDS-54).
	"phase_order",
	"num_stages", "stage_k", "stage_mode", "stage_context_size",
	"stage_cluster_type", "stage_min_bits", "stage_max_bits",
	"stage_min_neurons", "stage_max_neurons", "stage_neurons_grid", "stage_bits_grid",
	"tiered_neurons_grid", "tiered_bits_grid",
	"bigram_lambda", "unigram_lambda", "label_smoothing",
	"invalid_mode", "flip_labels",
	# IDS
	# ids_arch_type: REMOVED 08/11/2026 (IDS-54) — copied into an unread field;
	# "tiered" accepted with a note, "bitwise" REFUSED (inert_params.py).
	"ids_dataset", "ids_split", "ids_classification",
	"ids_n_bits", "ids_rest_bits", "ids_auto_max_bits", "ids_raw",
	"ids_num_parts", "ids_k_folds", "ids_kfold_per_gen", "ids_val_fraction",
	"ids_single_cluster", "ids_feature_selection", "ids_invalid_encoding",
	# MCST support tiering (27/08/2026): per-class neuron/bits centres from
	# train supports; classnorm decode modes are UNCONDITIONAL (no param —
	# same reasoning as the automatic CE anchor: a choice is a thing that
	# gets chosen wrong).
	"ids_tier_sizing", "ids_tier_neuron_cap", "ids_tier_bits_min", "ids_tier_bits_max",
	"ids_fitness_weight_f1", "ids_fitness_weight_fpr",
	"ids_streaming", "ids_streaming_chunk_size",
	# ids_encoding_dedupe (08/11/2026, IDS-52): "off" (default — bit-identical
	# encoding) | "on": the thermometer keeps only thresholds whose bit is
	# non-constant and distinct on the TRAIN partition (per-feature width shrinks
	# to what the feature's values support), drops features left with no
	# informative bit, and drops exact-duplicate feature columns (train only).
	# Pinned in the test-cache key. Refused with ids_streaming.
	"ids_encoding_dedupe",
	# ids_feature_ranking (08/11/2026, IDS-53): "legacy" (default — the
	# TOP20_RF_FEATURES constants, byte-identical) | "train_only" (the top-N
	# re-derived on the flow's OWN train split; wnn/ids/feature_ranking.py,
	# versioned files under wnn/ids/feature_rankings/). The legacy lists were
	# ranked on partitions holding 76% (cicids) / 84% (ciciot) of today's
	# random_3way test/val. Pinned in the test-cache key ("fr=").
	"ids_feature_ranking",
	# ids_receptive_fields (08/11/2026, worker ABI 27): "off" (default — no
	# projection, no RNG draw, bit-identical) | "family": every neuron holds at
	# most floor(ids_rf_cross x bits) connections outside its feature FAMILY
	# (versioned maps wnn/ids/feature_families/v2/<dataset>.json; a neuron's family
	# = the family holding the majority of its connections, a fresh neuron draws
	# one by its connections' families). Enforced in Rust after every GA/TS/genesis
	# operator and in Python on every grid / init / immigrant genome. IDS port of
	# the controller's CTRL-45. Pinned in the test-cache key ("rf=").
	# ids_rf_cross (float in [0, 0.5), default 0.2): the out-of-family share;
	# refused when set with receptive fields off.
	"ids_receptive_fields", "ids_rf_cross",
	# decode_mode: REMOVED 08/11/2026 (IDS-54) — never read; "argmax" accepted
	# with a note, anything else REFUSED (inert_params.py).
	"ids_encoded_storage", "ids_encoded_storage_dir", "ids_memmap_prefetch",
	"balance_classes", "class_weight_multiplier", "undersample_majority",
	"reweight_rounds", "reweight_max_boost",
	# Leaderboard seeding
	"seed_from_leaderboard", "seed_leaderboard_count", "seed_leaderboard_metric",
	"seed_leaderboard_stage", "seed_leaderboard_task_type",
	# Env-forwarded knobs (worker maps these to WNN_* env vars)
	"wnn_batch_size", "wnn_grid_search_parallel", "wnn_hybrid_speed_ratio",
	"wnn_no_metal", "wnn_num_threads", "wnn_order_independent_train",
	# Controller (drone) flows — the shape/episode keys wnn.control.flow_adapter
	# builds (ADAPTER_KEYS; registered 08/11/2026, they tripped the UNKNOWN warning).
	"controller_num_motors", "controller_levels_per_motor", "controller_bits_per_feature",
	"controller_input_window_k", "controller_state_neurons", "controller_state_bits",
	"controller_output_bits", "controller_delta_control",
	"controller_eval_episodes", "controller_steps", "controller_tilt_deg",
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
	# ABI 32 (31/10/2026): the real-life plant + G5, same knobs as phased_ga
	# --imu-specific-force/--imu-inclinometer, --rotor-drag/--no-rotor-drag,
	# --obs-att-estimator, --att-err-ref.
	"controller_imu_specific_force",  # plant: accelerometer reads specific force (default with translation)
	"controller_rotor_drag",          # plant: linear rotor drag k_d (1/s)
	"controller_obs_att_estimator",   # feature source: Mahony-estimated attitude (default with specific force)
	"controller_att_err_ref",         # scoring: attitude error vs level | tilt (position-loop reference)
	# ABI 38 (CTRL-27): the cascade controller, same knobs as phased_ga --output-mode /
	# --ctbr-rate-max / --ctbr-coll-max / --ctbr-gamma. The flow adapter reads
	# controller_output_mode and REFUSES "ctbr" (its EpisodeConfig carries no airframe
	# firmware cascade to be the inner loop); the CLI is the cascade's entry point.
	"controller_output_mode",         # "pwm" (motor-level) | "ctbr" (cascade)
	"controller_ctbr_rate_max",       # [p, q, r] body-rate range (rad/s)
	"controller_ctbr_coll_max",       # collective-deviation range (pwm)
	"controller_ctbr_gamma",          # [T, p, q, r] per-bank alphabet gamma
	"controller_ctbr_delta_law",      # ablation: delta-law decode instead of absolute
	# CTRL-38 (ABI 41): the cascade input blocks, same knobs as phased_ga --obs-rate-i /
	# --obs-cascade-dhat. Cascade-only, so the flow adapter REFUSES them (it refuses ctbr).
	"controller_obs_rate_i",          # cascade feature: inner rate PID integrator I/i_limit (+3)
	"controller_obs_cascade_dhat",    # cascade feature: the mpcof teacher's d̂ observer (+3)
	# Specialist programme (14/08/2026)
	"controller_conn_policy",         # connection-creation policy: spread | min1 | min2 | min3 (flow adapter; framed1 = CLI only)
	"controller_output_full_window",  # arm D: output layer samples ALL k frames, not just t-0
	"controller_frame_stride",        # K-window shifts every N pushes (spans N*K steps; 1 = legacy)
	# CTRL-45 (ABI 44): phased_ga --receptive-fields / --rf-cross. CTBR/decouple-only, so
	# the flow adapter REFUSES them (it builds motor-level PWM controllers only).
	"controller_receptive_fields",    # off | axis — output neurons keep to their bank's axis family
	"controller_rf_cross",            # cross-family connection budget, fraction of width
	# CTRL-48 (ABI 45): phased_ga --calib-episodes/--calib-regime/--calib-ladder,
	# --drop-features/--drop-constant-features, --obs-cascade-dhat-residual. The flow
	# adapter builds none of them (no training-cascade sampler, motor-level only): REFUSED.
	"controller_calib_episodes",      # thermometer calibration episodes (10 = ABI 44)
	"controller_calib_regime",        # flown | train_mix
	"controller_calib_ladder",        # quantile | centred
	"controller_drop_features",       # comma-separated layout names no connection may land on
	"controller_drop_constant_features",  # exclude features measured constant at calibration
	"controller_obs_cascade_dhat_residual",  # cascade feature: unabsorbed d̂ (+3)
	# CTRL-46 (ABI 46): phased_ga --write-rule/--vote-share, --dagger-label-attitude,
	# --obs-att-bias-pred. The flow adapter builds none of them (no DAgger write rule,
	# no estimator plant): REFUSED by _refuse_unbuilt_keys.
	"controller_write_rule",          # last | vote (the IDS ids_write_rule vocabulary)
	"controller_vote_share",          # vote min-share θ in (0, 0.5], 1/100 grid
	"controller_dagger_label_attitude",  # true | estimate — the DAgger label's attitude
	"controller_obs_att_bias_pred",   # +2 features: onboard Mahony-bias prediction
	"controller_yaw_centred_ladder",  # ARM 2: yaw-error inputs on a 0-edged centred ladder
	"controller_yaw_integral_leak",   # ARM 3: separate leak for the yaw integral inputs
	# Trainer tilt curriculum (27/09/2026): phased_ga --rg-easy-tilt-deg / --rg-full-tilt-deg.
	"controller_rg_easy_tilt_deg",    # DAgger round-0 initial-tilt bound (deg; default 8)
	"controller_rg_full_tilt_deg",    # DAgger last-round + checkpoint-eval tilt (deg; default 30)
	# 09/11/2026 (docs/ctrl_dres_failures.md #3/#4): phased_ga --select-tail/--select-tail-deg
	# and --train-xy-offset/--train-init-vxy (ABI 48). Not built by the flow adapter:
	# REFUSED by _refuse_unbuilt_keys like the other phased_ga-only keys.
	"controller_select_tail",         # off | near_fail | p99 — stage-select tail term
	"controller_select_tail_deg",     # near_fail bound (deg, in (0, 5); default 4)
	# 10/11/2026 (docs/ctrl_select_tail_prereg.md): phased_ga --val-units. Phased_ga-only,
	# REFUSED by _refuse_unbuilt_keys like the select_tail keys.
	"controller_val_units",           # stage-select val units per candidate (default 5)
	"controller_train_xy_offset",     # DAgger trainer xy start bound (m; None = xy_offset)
	"controller_train_init_vxy",      # DAgger trainer initial v_xy bound (m/s; default 0)
})


def validate_flow_params(params: dict, log=print, flow_id=None) -> list[str]:
	"""Validate a flow's params dict at ingestion. Returns unknown keys.

	- Unknown keys produce a LOUD warning naming the likely intent (a typo'd
	  key previously fell back to a default silently — the Rule-6 bug class).
	- The provided (non-default) config is logged once for reproducibility.
	- IDS-54: inert keys are noted at their default and REFUSED (ValueError) at
	  any other value; keys with no effect for the flow's architecture warn.
	"""
	from wnn.ram.experiments.inert_params import INERT_KEYS, enforce_inert_params, warn_inapplicable_params
	prefix = f"[PARAMS]{f' flow={flow_id}' if flow_id is not None else ''}"
	enforce_inert_params(params, log, prefix)
	warn_inapplicable_params(params, log, prefix)
	unknown = sorted(k for k in params if k not in KNOWN_PARAMS and k not in INERT_KEYS)
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
