"""
Flow Orchestration

A Flow is a sequence of experiments (like the current 6-phase pass).
This module provides:
- FlowConfig for defining flows
- Flow class for executing flows
- Factory methods for common patterns (standard-6-phase)
"""

import gzip
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from wnn.ram.fitness import FitnessCalculatorType
from wnn.ram.experiments.experiment import Experiment, ClusterType, ExperimentConfig, ExperimentResult, ExperimentType, StageMode
from wnn.ram.experiments.dashboard_client import DashboardClient, FlowConfig as APIFlowConfig
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


class FlowStoppedError(Exception):
	"""Raised when a flow is stopped due to shutdown request."""
	pass


@dataclass
class FlowConfig:
	"""
	Configuration for a flow (sequence of experiments).

	A flow defines:
	- A name and description
	- A list of experiment configurations in sequence
	- Optional seed checkpoint to start from
	- Shared parameters (tier config, etc.)
	"""

	name: str
	experiments: list[ExperimentConfig]
	description: Optional[str] = None
	seed_checkpoint_path: Optional[str] = None

	# Shared architecture config
	tier_config: Optional[list[tuple[Optional[int], int, int]]] = None
	optimize_tier0_only: bool = False
	context_size: int = 4

	# Shared optimization params
	patience: int = 3
	check_interval: int = 10
	threshold_delta: float = 0.01
	threshold_reference: int = 1000
	fitness_percentile: Optional[float] = None

	# Fitness calculator settings
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.NORMALIZED
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0

	# Random seed
	seed: Optional[int] = None

	@classmethod
	def standard_6_phase(
		cls,
		name: str,
		ga_generations: int = 250,
		ts_iterations: int = 250,
		population_size: int = 50,
		neighbors_per_iter: int = 50,
		patience: int = 3,
		phase_order: Literal["neurons_first", "bits_first"] = "neurons_first",
		tier_config: Optional[list[tuple[Optional[int], int, int]]] = None,
		optimize_tier0_only: bool = False,
		context_size: int = 4,
		fitness_percentile: Optional[float] = None,
		fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.NORMALIZED,
		fitness_weight_ce: float = 1.0,
		fitness_weight_acc: float = 1.0,
		seed: Optional[int] = None,
		description: Optional[str] = None,
		seed_checkpoint_path: Optional[str] = None,
	) -> "FlowConfig":
		"""
		Create a standard 6-phase flow configuration.

		This matches the existing PhasedSearchRunner's behavior:
		- Phase 1a/1b: GA/TS for first dimension (neurons or bits)
		- Phase 2a/2b: GA/TS for second dimension
		- Phase 3a/3b: GA/TS for connections

		Args:
			name: Flow name
			ga_generations: GA generations per phase
			ts_iterations: TS iterations per phase
			population_size: GA population / TS neighbor cache size
			neighbors_per_iter: TS neighbors per iteration
			patience: Early stopping patience
			phase_order: "neurons_first" or "bits_first"
			tier_config: Tiered architecture config
			optimize_tier0_only: Only mutate tier0 clusters
			fitness_percentile: Fitness percentile filter
			fitness_calculator_type: Fitness calculator type (NORMALIZED, HARMONIC_RANK, etc.)
			fitness_weight_ce: Weight for CE in fitness calculation
			fitness_weight_acc: Weight for accuracy in fitness calculation
			seed: Random seed
			description: Optional description
			seed_checkpoint_path: Optional checkpoint to seed from

		Returns:
			FlowConfig for standard 6-phase search
		"""
		if phase_order == "bits_first":
			phases = [
				("Phase 1a: GA Bits", "ga", True, False, False),
				("Phase 1b: TS Bits", "ts", True, False, False),
				("Phase 2a: GA Neurons", "ga", False, True, False),
				("Phase 2b: TS Neurons", "ts", False, True, False),
				("Phase 3a: GA Connections", "ga", False, False, True),
				("Phase 3b: TS Connections", "ts", False, False, True),
			]
		else:
			phases = [
				("Phase 1a: GA Neurons", ExperimentType.GA, False, True, False),
				("Phase 1b: TS Neurons", ExperimentType.TS, False, True, False),
				("Phase 2a: GA Bits", ExperimentType.GA, True, False, False),
				("Phase 2b: TS Bits", ExperimentType.TS, True, False, False),
				("Phase 3a: GA Connections", ExperimentType.GA, False, False, True),
				("Phase 3b: TS Connections", ExperimentType.TS, False, False, True),
			]

		experiments = []
		for phase_name, exp_type, opt_bits, opt_neurons, opt_conns in phases:
			config = ExperimentConfig(
				name=phase_name,
				experiment_type=exp_type,
				optimize_bits=opt_bits,
				optimize_neurons=opt_neurons,
				optimize_connections=opt_conns,
				generations=ga_generations,
				population_size=population_size,
				iterations=ts_iterations,
				neighbors_per_iter=neighbors_per_iter,
				patience=patience,
				tier_config=tier_config,
				optimize_tier0_only=optimize_tier0_only,
				fitness_percentile=fitness_percentile,
				fitness_calculator_type=fitness_calculator_type,
				fitness_weight_ce=fitness_weight_ce,
				fitness_weight_acc=fitness_weight_acc,
				seed=seed,
			)
			experiments.append(config)

		return cls(
			name=name,
			experiments=experiments,
			description=description or f"Standard 6-phase search ({phase_order})",
			seed_checkpoint_path=seed_checkpoint_path,
			tier_config=tier_config,
			optimize_tier0_only=optimize_tier0_only,
			context_size=context_size,
			patience=patience,
			fitness_percentile=fitness_percentile,
			fitness_calculator_type=fitness_calculator_type,
			fitness_weight_ce=fitness_weight_ce,
			fitness_weight_acc=fitness_weight_acc,
			seed=seed,
		)

	# Architecture type
	architecture_type: str = "tiered"

	# Bitwise-specific config
	num_clusters: int = 16
	memory_mode: str = "QUAD_WEIGHTED"
	neuron_sample_rate: float = 0.25
	min_bits: int = 10
	max_bits: int = 24
	min_neurons: int = 10
	max_neurons: int = 300
	max_bit_delta: int = 0  # Max bits change per mutation (0 = auto)
	sparse_threshold: Optional[int] = None

	# Multi-stage specific config
	num_stages: int = 1
	stage_k: Optional[list[int]] = None
	stage_cluster_type: Optional[list[str]] = None
	stage_mode: Optional[list[int]] = None
	# Invalid token mode: S1 groups reject wrong-group inputs via filtered gating
	invalid_mode: bool = False
	top_m: int = 5
	# Label smoothing: CE_smooth = -log[(1-ε) × P_hierarchical + ε/vocab_size]
	label_smoothing: float = 0.0
	# Unigram interpolation (Jelinek-Mercer): P = (1-λ_u)×P_hier + λ_u×P_unigram
	unigram_lambda: float = 0.0
	# KN bigram interpolation: P = w×P_hier + λ_u×P_unigram + λ_b×P_KN_bigram
	bigram_lambda: float = 0.0
	# Per-stage bounds override (indexed by stage)
	stage_min_bits_list: Optional[list[int]] = None
	stage_max_bits_list: Optional[list[int]] = None
	stage_min_neurons_list: Optional[list[int]] = None
	stage_max_neurons_list: Optional[list[int]] = None

	# IDS-specific config (architecture_type="ids")
	ids_classification: str = "binary"  # "binary", "multi", or "hierarchical"
	ids_n_bits: int = 8  # thermometer encoding bits per feature
	ids_val_fraction: float = 0.25  # validation holdout fraction
	ids_num_parts: int = 3  # training data rotation parts
	ids_fitness_weight_f1: float = 0.0  # F1-macro weight in fitness
	ids_split: str = "standard"  # "standard" or "random"
	balance_classes: bool = False  # Class-balanced training (upweight minority class)

	@classmethod
	def bitwise_7_phase(
		cls,
		name: str,
		ga_generations: int = 250,
		ts_iterations: int = 250,
		population_size: int = 50,
		neighbors_per_iter: int = 50,
		patience: int = 3,
		context_size: int = 4,
		num_clusters: int = 16,
		memory_mode: str = "QUAD_WEIGHTED",
		neuron_sample_rate: float = 0.25,
		min_bits: int = 10,
		max_bits: int = 24,
		min_neurons: int = 10,
		max_neurons: int = 300,
		phase_order: str = "neurons_first",
		fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK,
		fitness_weight_ce: float = 1.0,
		fitness_weight_acc: float = 1.0,
		sparse_threshold: Optional[int] = None,
		seed: Optional[int] = None,
		description: Optional[str] = None,
		seed_checkpoint_path: Optional[str] = None,
	) -> "FlowConfig":
		"""
		Create a bitwise 7-phase flow configuration.

		Phase 1 is always Grid Search. Phases 2-5 order depends on phase_order:
		  neurons_first: grid → neurons → bits → connections
		  bits_first: grid → bits → neurons → connections

		The grid search phase uses a special ExperimentType and evaluates
		combinations of [50,100,150,200] neurons × [14,16,18,20] bits.
		"""
		# Grid search is always first
		grid_phase = ("Phase 1: Grid Search", ExperimentType.GRID_SEARCH, False, False, False, {"phase_type": "grid_search"})

		neurons_phases = [
			("GA Neurons", ExperimentType.GA, False, True, False, {}),
			("TS Neurons", ExperimentType.TS, False, True, False, {}),
		]
		bits_phases = [
			("GA Bits", ExperimentType.GA, True, False, False, {}),
			("TS Bits", ExperimentType.TS, True, False, False, {}),
		]
		connections_phases = [
			("GA Connections", ExperimentType.GA, False, False, True, {}),
			("TS Connections", ExperimentType.TS, False, False, True, {}),
		]

		if phase_order == "bits_first":
			ordered = bits_phases + neurons_phases
		else:
			ordered = neurons_phases + bits_phases

		# Number phases 2-7
		numbered = []
		for i, (pname, *rest) in enumerate(ordered + connections_phases, start=2):
			numbered.append((f"Phase {i}: {pname}", *rest))

		phases = [grid_phase] + numbered

		# Default grid for grid search
		default_neurons_grid = [50, 100, 150, 200]
		default_bits_grid = [14, 16, 18, 20]

		experiments = []
		for phase_name, exp_type, opt_bits, opt_neurons, opt_conns, extra_params in phases:
			config = ExperimentConfig(
				name=phase_name,
				experiment_type=exp_type,
				optimize_bits=opt_bits,
				optimize_neurons=opt_neurons,
				optimize_connections=opt_conns,
				generations=ga_generations,
				population_size=population_size,
				iterations=ts_iterations,
				neighbors_per_iter=neighbors_per_iter,
				patience=patience,
				fitness_calculator_type=fitness_calculator_type,
				fitness_weight_ce=fitness_weight_ce,
				fitness_weight_acc=fitness_weight_acc,
				seed=seed,
				cluster_type=ClusterType.BITWISE,
				# Bitwise-specific bounds
				bitwise_min_bits=min_bits,
				bitwise_max_bits=max_bits,
				bitwise_min_neurons=min_neurons,
				bitwise_max_neurons=max_neurons,
			)
			# Grid search: set grid params (grid search is 1 step, no iterations)
			if exp_type == ExperimentType.GRID_SEARCH:
				config.neurons_grid = default_neurons_grid
				config.bits_grid = default_bits_grid
				config.generations = 1
			# Store extra params for phase_type override
			config._extra_params = extra_params
			experiments.append(config)

		return cls(
			name=name,
			experiments=experiments,
			description=description or "Bitwise 7-phase optimization (grid → GA/TS neurons → bits → connections)",
			seed_checkpoint_path=seed_checkpoint_path,
			context_size=context_size,
			patience=patience,
			fitness_calculator_type=fitness_calculator_type,
			fitness_weight_ce=fitness_weight_ce,
			fitness_weight_acc=fitness_weight_acc,
			seed=seed,
			architecture_type="bitwise",
			num_clusters=num_clusters,
			memory_mode=memory_mode,
			neuron_sample_rate=neuron_sample_rate,
			min_bits=min_bits,
			max_bits=max_bits,
			min_neurons=min_neurons,
			max_neurons=max_neurons,
			sparse_threshold=sparse_threshold,
		)

	@classmethod
	def bitwise_10_phase(
		cls,
		name: str,
		ga_generations: int = 250,
		ts_iterations: int = 250,
		adaptation_iterations: int = 50,
		population_size: int = 50,
		neighbors_per_iter: int = 50,
		patience: int = 3,
		context_size: int = 4,
		num_clusters: int = 16,
		memory_mode: str = "QUAD_WEIGHTED",
		neuron_sample_rate: float = 0.25,
		min_bits: int = 10,
		max_bits: int = 24,
		min_neurons: int = 10,
		max_neurons: int = 300,
		fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK,
		fitness_weight_ce: float = 1.0,
		fitness_weight_acc: float = 1.0,
		sparse_threshold: Optional[int] = None,
		seed: Optional[int] = None,
		description: Optional[str] = None,
		seed_checkpoint_path: Optional[str] = None,
	) -> "FlowConfig":
		"""
		Create a bitwise 10-phase flow with adaptation phases.

		Pattern: explore (GA) → exploit stats (adaptation) → refine (TS) at each level.

		Phase 1:  Grid Search (neurons × bits)
		Phase 2:  GA Neurons
		Phase 3:  Neurogenesis
		Phase 4:  TS Neurons (refine)
		Phase 5:  GA Bits
		Phase 6:  Synaptogenesis
		Phase 7:  TS Bits (refine)
		Phase 8:  GA Connections
		Phase 9:  Axonogenesis
		Phase 10: TS Connections (refine)
		"""
		phases = [
			("Phase 1: Grid Search", ExperimentType.GRID_SEARCH, False, False, False),
			("Phase 2: GA Neurons", ExperimentType.GA, False, True, False),
			("Phase 3: Neurogenesis", ExperimentType.NEUROGENESIS, False, False, False),
			("Phase 4: TS Neurons", ExperimentType.TS, False, True, False),
			("Phase 5: GA Bits", ExperimentType.GA, True, False, False),
			("Phase 6: Synaptogenesis", ExperimentType.SYNAPTOGENESIS, False, False, False),
			("Phase 7: TS Bits", ExperimentType.TS, True, False, False),
			("Phase 8: GA Connections", ExperimentType.GA, False, False, True),
			("Phase 9: Axonogenesis", ExperimentType.AXONOGENESIS, False, False, False),
			("Phase 10: TS Connections", ExperimentType.TS, False, False, True),
		]

		default_neurons_grid = [50, 100, 150, 200]
		default_bits_grid = [14, 16, 18, 20]

		adaptation_types = {ExperimentType.NEUROGENESIS, ExperimentType.SYNAPTOGENESIS, ExperimentType.AXONOGENESIS}

		experiments = []
		for phase_name, exp_type, opt_bits, opt_neurons, opt_conns in phases:
			# Adaptation phases use adaptation_iterations, GA/TS use their own
			if exp_type in adaptation_types:
				iters = adaptation_iterations
				gens = adaptation_iterations
			else:
				iters = ts_iterations
				gens = ga_generations

			config = ExperimentConfig(
				name=phase_name,
				experiment_type=exp_type,
				optimize_bits=opt_bits,
				optimize_neurons=opt_neurons,
				optimize_connections=opt_conns,
				generations=gens,
				population_size=population_size,
				iterations=iters,
				neighbors_per_iter=neighbors_per_iter,
				patience=patience,
				fitness_calculator_type=fitness_calculator_type,
				fitness_weight_ce=fitness_weight_ce,
				fitness_weight_acc=fitness_weight_acc,
				seed=seed,
				cluster_type=ClusterType.BITWISE,
				bitwise_min_bits=min_bits,
				bitwise_max_bits=max_bits,
				bitwise_min_neurons=min_neurons,
				bitwise_max_neurons=max_neurons,
			)
			if exp_type == ExperimentType.GRID_SEARCH:
				config.neurons_grid = default_neurons_grid
				config.bits_grid = default_bits_grid
				config.generations = 1
			experiments.append(config)

		return cls(
			name=name,
			experiments=experiments,
			description=description or "Bitwise 10-phase optimization (grid → GA/adapt/TS for neurons → bits → connections)",
			seed_checkpoint_path=seed_checkpoint_path,
			context_size=context_size,
			patience=patience,
			fitness_calculator_type=fitness_calculator_type,
			fitness_weight_ce=fitness_weight_ce,
			fitness_weight_acc=fitness_weight_acc,
			seed=seed,
			architecture_type="bitwise",
			num_clusters=num_clusters,
			memory_mode=memory_mode,
			neuron_sample_rate=neuron_sample_rate,
			min_bits=min_bits,
			max_bits=max_bits,
			min_neurons=min_neurons,
			max_neurons=max_neurons,
			sparse_threshold=sparse_threshold,
		)

	@classmethod
	def multi_stage_flow(
		cls,
		name: str,
		num_stages: int = 2,
		stage_k: Optional[list[int]] = None,
		stage_cluster_type: Optional[list[str]] = None,
		stage_mode: Optional[list[int]] = None,
		ga_generations: int = 250,
		ts_iterations: int = 250,
		adaptation_iterations: int = 50,
		population_size: int = 50,
		neighbors_per_iter: int = 50,
		patience: int = 3,
		context_size: int = 4,
		memory_mode: str = "QUAD_WEIGHTED",
		neuron_sample_rate: float = 0.25,
		min_bits: int = 4,
		max_bits: int = 24,
		min_neurons: int = 5,
		max_neurons: int = 300,
		neurons_grid: Optional[list[int]] = None,
		bits_grid: Optional[list[int]] = None,
		tiered_neurons_grid: Optional[list[int]] = None,
		tiered_bits_grid: Optional[list[int]] = None,
		# Per-stage bounds override (indexed by stage, None = use global/auto)
		stage_min_bits_list: Optional[list[int]] = None,
		stage_max_bits_list: Optional[list[int]] = None,
		stage_min_neurons_list: Optional[list[int]] = None,
		stage_max_neurons_list: Optional[list[int]] = None,
		fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK,
		fitness_weight_ce: float = 1.0,
		fitness_weight_acc: float = 1.0,
		sparse_threshold: Optional[int] = None,
		seed: Optional[int] = None,
		description: Optional[str] = None,
		seed_checkpoint_path: Optional[str] = None,
		template: str = "full",
	) -> "FlowConfig":
		"""Create a multi-stage flow with experiments per stage.

		Templates:
		  "full" (default): 10 phases per stage (same as bitwise-10-phase):
		    Grid → GA Neurons → Neurogenesis → TS Neurons →
		    GA Bits → Synaptogenesis → TS Bits →
		    GA Connections → Axonogenesis → TS Connections
		  "fast": 2 phases per stage for rapid K experiments:
		    Grid Search (expanded grid) → GA Neurons (250 gens, patience 5)

		Names prefixed: "S0: Grid Search", "S1: GA Neurons", etc.

		Args:
			num_stages: Number of stages (currently must be 2)
			stage_k: K per stage; uses compute_default_k() if None
			stage_cluster_type: Per-stage architecture type (default: ["bitwise", "bitwise"])
			stage_mode: StageMode values between stages (default: [INPUT_CONCAT])
			adaptation_iterations: Iterations for neurogenesis/synaptogenesis/axonogenesis phases
			neurons_grid: Grid of neuron counts for bitwise stages (default: [5,10,25,50,100,200,300])
			bits_grid: Grid of bit counts for bitwise stages (default: [4,6,8,10,12,16,20,24])
			tiered_neurons_grid: Grid of neuron counts for tiered stages (default: [20,30,40,50])
			tiered_bits_grid: Grid of bit counts for tiered stages (default: [18,19,20,21,22,23])
			template: "full" (10 phases/stage) or "fast" (2 phases/stage)
		"""
		assert num_stages == 2, "Only 2 stages supported (for now)"

		if stage_cluster_type is None:
			stage_cluster_type = ["bitwise"] * num_stages
		if stage_k is None:
			from wnn.ram.architecture.multistage_evaluator import compute_default_k
			stage_k = compute_default_k(num_stages, stage_cluster_type)
		if stage_mode is None:
			stage_mode = [StageMode.INPUT_CONCAT]

		# Default grids for bitwise stages
		if neurons_grid is None:
			neurons_grid = [5, 10, 25, 50]
		if bits_grid is None:
			bits_grid = [4, 6, 8, 10, 12, 16, 20, 24]

		# Default grids for tiered stages (smaller ranges — tiered has sparse
		# address spaces, so large neurons/bits cause degenerate scores)
		if tiered_neurons_grid is None:
			tiered_neurons_grid = [20, 30, 40, 50]
		if tiered_bits_grid is None:
			tiered_bits_grid = [18, 19, 20, 21, 22, 23]

		adaptation_types = {ExperimentType.NEUROGENESIS, ExperimentType.SYNAPTOGENESIS, ExperimentType.AXONOGENESIS}

		experiments = []
		is_fast = template == "fast"

		for stage in range(num_stages):
			prefix = f"S{stage}"
			is_tiered = stage_cluster_type[stage] == "tiered"
			is_selector = (
				stage > 0
				and stage_mode is not None
				and len(stage_mode) >= stage
				and (stage_mode[stage - 1] if isinstance(stage_mode[stage - 1], int) else stage_mode[stage - 1].value) == StageMode.SELECTOR
			)

			# Select grid and bounds based on stage type
			if is_selector:
				if is_fast:
					# Fast template: expanded grid for rapid K experiments
					stage_neurons_grid = [5, 10, 25, 50, 75, 100, 150]
					stage_bits_grid = [5, 6, 7, 8, 9, 10]
				else:
					# Full template: smaller grid (SELECTOR sub-models have 225× less data)
					stage_neurons_grid = [5, 10, 15]
					stage_bits_grid = [5, 6, 7, 8, 9, 10]
			elif is_tiered:
				stage_neurons_grid = tiered_neurons_grid
				stage_bits_grid = tiered_bits_grid
			else:
				stage_neurons_grid = neurons_grid
				stage_bits_grid = bits_grid

			# Per-stage overrides from dashboard params (take precedence)
			if stage_min_bits_list and stage < len(stage_min_bits_list):
				stage_min_bits = stage_min_bits_list[stage]
			else:
				stage_min_bits = min(stage_bits_grid)
			if stage_max_bits_list and stage < len(stage_max_bits_list):
				stage_max_bits = stage_max_bits_list[stage]
			else:
				stage_max_bits = max(stage_bits_grid)
			if stage_min_neurons_list and stage < len(stage_min_neurons_list):
				stage_min_neurons = stage_min_neurons_list[stage]
			else:
				stage_min_neurons = min(stage_neurons_grid)
			if stage_max_neurons_list and stage < len(stage_max_neurons_list):
				stage_max_neurons = stage_max_neurons_list[stage]
			else:
				stage_max_neurons = max(stage_neurons_grid)

			if is_fast:
				# Fast template: 2 phases per stage (Grid + GA Neurons)
				stage_phases = [
					(f"{prefix}: Grid Search", ExperimentType.GRID_SEARCH, False, False, False),
					(f"{prefix}: GA Neurons", ExperimentType.GA, False, True, False),
				]
			else:
				# Full template: 10 phases per stage (mirrors bitwise-10-phase)
				stage_phases = [
					(f"{prefix}: Grid Search", ExperimentType.GRID_SEARCH, False, False, False),
					(f"{prefix}: GA Neurons", ExperimentType.GA, False, True, False),
					(f"{prefix}: Neurogenesis", ExperimentType.NEUROGENESIS, False, False, False),
					(f"{prefix}: TS Neurons", ExperimentType.TS, False, True, False),
					(f"{prefix}: GA Bits", ExperimentType.GA, True, False, False),
					(f"{prefix}: Synaptogenesis", ExperimentType.SYNAPTOGENESIS, False, False, False),
					(f"{prefix}: TS Bits", ExperimentType.TS, True, False, False),
					(f"{prefix}: GA Connections", ExperimentType.GA, False, False, True),
					(f"{prefix}: Axonogenesis", ExperimentType.AXONOGENESIS, False, False, False),
					(f"{prefix}: TS Connections", ExperimentType.TS, False, False, True),
				]

			for phase_name, exp_type, opt_bits, opt_neurons, opt_conns in stage_phases:
				if exp_type in adaptation_types:
					iters = adaptation_iterations
					gens = adaptation_iterations
				else:
					iters = ts_iterations
					gens = ga_generations

				# Fast template: GA neurons gets higher patience for deeper search
				phase_patience = max(patience, 5) if is_fast and exp_type == ExperimentType.GA else patience

				config = ExperimentConfig(
					name=phase_name,
					experiment_type=exp_type,
					optimize_bits=opt_bits,
					optimize_neurons=opt_neurons,
					optimize_connections=opt_conns,
					generations=gens,
					population_size=population_size,
					iterations=iters,
					neighbors_per_iter=neighbors_per_iter,
					patience=phase_patience,
					fitness_calculator_type=fitness_calculator_type,
					fitness_weight_ce=fitness_weight_ce,
					fitness_weight_acc=fitness_weight_acc,
					seed=seed,
					cluster_type=ClusterType.MULTI_STAGE,
					# Multi-stage config
					num_stages=num_stages,
					stage_k=stage_k,
					stage_cluster_type=stage_cluster_type,
					stage_mode=[m if isinstance(m, int) else m.value for m in stage_mode],
					target_stage=stage,
					# Stage-specific bounds (tiered uses smaller ranges)
					bitwise_min_bits=stage_min_bits,
					bitwise_max_bits=stage_max_bits,
					bitwise_min_neurons=stage_min_neurons,
					bitwise_max_neurons=stage_max_neurons,
				)
				if exp_type == ExperimentType.GRID_SEARCH:
					config.neurons_grid = stage_neurons_grid
					config.bits_grid = stage_bits_grid
					config.generations = 1
				experiments.append(config)

		return cls(
			name=name,
			experiments=experiments,
			description=description or (
				f"Multi-stage fast {num_stages * 2}-phase optimization ({num_stages} stages × 2 phases)"
				if is_fast else
				f"Multi-stage {num_stages * 10}-phase optimization ({num_stages} stages × 10 phases)"
			),
			seed_checkpoint_path=seed_checkpoint_path,
			context_size=context_size,
			patience=patience,
			fitness_calculator_type=fitness_calculator_type,
			fitness_weight_ce=fitness_weight_ce,
			fitness_weight_acc=fitness_weight_acc,
			seed=seed,
			architecture_type="multi_stage",
			memory_mode=memory_mode,
			neuron_sample_rate=neuron_sample_rate,
			min_bits=min_bits,
			max_bits=max_bits,
			min_neurons=min_neurons,
			max_neurons=max_neurons,
			sparse_threshold=sparse_threshold,
			num_stages=num_stages,
			stage_k=stage_k,
			stage_cluster_type=stage_cluster_type,
			stage_mode=[m if isinstance(m, int) else m.value for m in stage_mode],
		)

	def to_api_config(self) -> APIFlowConfig:
		"""Convert to API FlowConfig for dashboard registration."""
		# Convert tier_config to string format for API compatibility
		tier_config_str = None
		if self.tier_config is not None:
			tier_parts = []
			for tier in self.tier_config:
				if tier[0] is None:
					tier_parts.append(f"rest,{tier[1]},{tier[2]}")
				else:
					tier_parts.append(f"{tier[0]},{tier[1]},{tier[2]}")
			tier_config_str = ";".join(tier_parts)

		params = {
			"tier_config": tier_config_str,
			"optimize_tier0_only": self.optimize_tier0_only,
			"context_size": self.context_size,
			"patience": self.patience,
			"fitness_percentile": self.fitness_percentile,
			"fitness_calculator": self.fitness_calculator_type.name.lower(),
			"fitness_weight_ce": self.fitness_weight_ce,
			"fitness_weight_acc": self.fitness_weight_acc,
			"seed": self.seed,
			"architecture_type": self.architecture_type,
		}

		# Add bitwise-specific params
		if self.architecture_type == "bitwise":
			params.update({
				"num_clusters": self.num_clusters,
				"memory_mode": self.memory_mode,
				"neuron_sample_rate": self.neuron_sample_rate,
				"min_bits": self.min_bits,
				"max_bits": self.max_bits,
				"min_neurons": self.min_neurons,
				"max_neurons": self.max_neurons,
				"sparse_threshold": self.sparse_threshold,
			})

		# Add multi-stage params
		if self.architecture_type == "multi_stage":
			params.update({
				"num_stages": self.num_stages,
				"stage_k": self.stage_k,
				"stage_cluster_type": self.stage_cluster_type,
				"stage_mode": self.stage_mode,
				"memory_mode": self.memory_mode,
				"neuron_sample_rate": self.neuron_sample_rate,
				"min_bits": self.min_bits,
				"max_bits": self.max_bits,
				"min_neurons": self.min_neurons,
				"max_neurons": self.max_neurons,
				"sparse_threshold": self.sparse_threshold,
				"invalid_mode": self.invalid_mode,
				"top_m": self.top_m,
				"label_smoothing": self.label_smoothing,
			"unigram_lambda": self.unigram_lambda,
			"bigram_lambda": self.bigram_lambda,
			})

		return APIFlowConfig(
			name=self.name,
			experiments=[exp.to_dict() for exp in self.experiments],
			description=self.description,
			params=params,
		)


@dataclass
class FlowResult:
	"""Result from running a complete flow."""

	flow_name: str
	experiment_results: list[ExperimentResult]
	final_genome: ClusterGenome
	final_fitness: float
	final_accuracy: Optional[float]
	total_elapsed_seconds: float
	flow_id: Optional[int] = None

	# Multi-stage fields
	stage_genomes: Optional[list[ClusterGenome]] = None
	combined_ce: Optional[float] = None
	combined_accuracy: Optional[float] = None
	per_stage_ce: Optional[list[float]] = None

	def get_best_by_accuracy(self) -> Optional[ExperimentResult]:
		"""Get the experiment result with best accuracy."""
		best = None
		best_acc = -1.0
		for result in self.experiment_results:
			if result.final_accuracy and result.final_accuracy > best_acc:
				best = result
				best_acc = result.final_accuracy
		return best


class Flow:
	"""
	Flow executor for running a sequence of experiments.

	Example usage:
		config = FlowConfig.standard_6_phase(
			name="Pass 1",
			patience=3,
			tier_config=[(100, 15, 20), (400, 10, 12), (None, 5, 8)],
		)

		flow = Flow(
			config=config,
			evaluator=cached_evaluator,
			logger=log_fn,
			checkpoint_dir=Path("checkpoints/pass1"),
		)

		result = flow.run()
		print(f"Final CE: {result.final_fitness:.4f}")
	"""

	def __init__(
		self,
		config: FlowConfig,
		evaluator: Any,  # TieredEvaluator or MultiStageEvaluator
		logger: Callable[[str], None],
		checkpoint_dir: Optional[Path] = None,
		dashboard_client: Optional[DashboardClient] = None,
		flow_id: Optional[int] = None,
		tracker: Optional[Any] = None,  # ExperimentTracker for V2 tracking
		shutdown_check: Optional[Callable[[], bool]] = None,  # Callable returning True if shutdown requested
		full_evaluator: Optional[Any] = None,  # Separate evaluator for validation (validation set)
		s1_evaluator: Optional[Any] = None,  # Hierarchical IDS: S1 optimizer evaluator
		s1_full_evaluator: Optional[Any] = None,  # Hierarchical IDS: S1 test evaluator
	):
		"""
		Initialize flow.

		Args:
			config: Flow configuration
			evaluator: BaseEvaluator instance
			logger: Logging function
			checkpoint_dir: Directory for checkpoints
			dashboard_client: Optional dashboard client for API integration
			flow_id: Existing flow ID (skip creating new flow if provided)
			tracker: Optional V2 tracker for direct database writes
			shutdown_check: Optional callable that returns True if shutdown requested
			full_evaluator: Separate evaluator for validation (trains full, evals validation set)
			s1_evaluator: S1 evaluator for hierarchical IDS (attack-only, 9 classes)
			s1_full_evaluator: S1 test evaluator for hierarchical IDS
		"""
		self.config = config
		self.evaluator = evaluator
		self.full_evaluator = full_evaluator
		self.log = logger
		self.checkpoint_dir = checkpoint_dir
		self.dashboard_client = dashboard_client
		self.tracker = tracker
		self.shutdown_check = shutdown_check

		# Hierarchical IDS: store all evaluators for combined validation
		self._s1_evaluator = s1_evaluator
		self._s1_full_evaluator = s1_full_evaluator
		# Keep references to S0 evaluators (self.evaluator/full_evaluator get swapped at boundary)
		self._s0_evaluator = evaluator if s1_evaluator else None
		self._s0_full_evaluator = full_evaluator if s1_evaluator else None

		self._flow_id: Optional[int] = flow_id
		self._experiment_ids: dict[int, int] = {}  # idx -> experiment_id
		self._results: list[ExperimentResult] = []

	def _update_flow_status(self, message: str) -> None:
		"""Update flow status message on dashboard."""
		if self.dashboard_client and self._flow_id:
			try:
				self.dashboard_client.update_flow(self._flow_id, status_message=message)
			except Exception:
				pass  # Non-critical, don't fail the flow

	def _handle_stage_boundary(
		self,
		prev_stage: int,
		next_stage: int,
		current_genome,
		current_population,
		current_evals,
		frozen_genomes: list,
		frozen_populations: dict,
		verbose: bool = True,
	) -> None:
		"""Handle stage boundary transition (shared by multi-stage and IDS hierarchical).

		Freezes previous stage genome/population, transitions evaluators for the
		next stage, and resets state. Modifies frozen_genomes/frozen_populations in-place.
		"""
		cfg = self.config
		is_ids_hierarchical = cfg.architecture_type == "ids" and self._s1_evaluator is not None

		if verbose:
			self.log("")
			self.log("=" * 70)
			self.log(f"  STAGE BOUNDARY: S{prev_stage} → S{next_stage}")
			self.log("=" * 70)

		# Freeze previous stage genome + population
		if prev_stage < len(frozen_genomes):
			frozen_genomes[prev_stage] = current_genome
		if current_population is not None and current_evals is not None:
			frozen_populations[prev_stage] = (list(current_population), list(current_evals))
		self.log(f"  Frozen S{prev_stage} genome: {current_genome}")

		if is_ids_hierarchical:
			# IDS hierarchical: swap to S1 evaluators (entirely different dataset/classes)
			self.evaluator = self._s1_evaluator
			self.full_evaluator = self._s1_full_evaluator
			self.log(f"  Swapped evaluators: S1 (attack-only, {self.evaluator.vocab_size} classes)")
		else:
			# Multi-stage: re-encode data with previous stage's predictions and switch target
			if hasattr(self.evaluator, 'recompute_stage_with_predictions'):
				self.log(f"  Recomputing S{next_stage} data with S{prev_stage} predictions...")
				train_acc, eval_acc = self.evaluator.recompute_stage_with_predictions(
					frozen_stage=prev_stage,
					target_stage=next_stage,
					frozen_genome=current_genome,
				)
				self.log(f"  S{prev_stage} prediction accuracy: train={train_acc:.2%}, eval={eval_acc:.2%}")
				if verbose:
					self.log(f"  S{next_stage} data now uses realistic inputs from S{prev_stage}")

			self.evaluator.target_stage = next_stage
			self.log(f"  Evaluator target_stage → {next_stage}")

		self.log(f"  State reset for S{next_stage}")
		if verbose:
			self.log("")

	def _has_stage_boundaries(self) -> bool:
		"""Check if this flow has stage boundaries (multi-stage or IDS hierarchical)."""
		cfg = self.config
		if cfg.architecture_type == "multi_stage" and cfg.num_stages > 1:
			return True
		if cfg.architecture_type == "ids" and self._s1_evaluator is not None:
			return True
		return False

	def run(
		self,
		resume_from: Optional[int] = None,
		seed_genome: Optional[ClusterGenome] = None,
		seed_population: Optional[list[ClusterGenome]] = None,
		seed_threshold: Optional[float] = None,
	) -> FlowResult:
		"""
		Run the flow.

		Args:
			resume_from: Experiment index to resume from (0-indexed)
			seed_genome: Initial genome to seed first experiment
			seed_population: Initial population to seed first experiment
			seed_threshold: Initial threshold to continue from

		Returns:
			FlowResult with all experiment results
		"""
		cfg = self.config
		start_time = time.time()

		# Propagate max_bit_delta to evaluators (controls bit mutation step size)
		if cfg.max_bit_delta > 0:
			self.evaluator._max_bit_delta = cfg.max_bit_delta
			if self.full_evaluator:
				self.full_evaluator._max_bit_delta = cfg.max_bit_delta

		# Handle empty flow gracefully — complete immediately
		if not cfg.experiments:
			self.log("Flow has no experiments — completing immediately.")
			from wnn.ram.architecture.cluster_genome import ClusterGenome
			empty_genome = seed_genome or ClusterGenome(
				bits_per_neuron=[], neurons_per_cluster=[], connections=[],
			)
			return FlowResult(
				flow_name=cfg.name,
				experiment_results=[],
				final_genome=empty_genome,
				final_fitness=seed_threshold or 0.0,
				final_accuracy=None,
				total_elapsed_seconds=0.0,
				flow_id=self._flow_id,
			)

		self.log("")
		self.log("=" * 70)
		self.log(f"  FLOW: {cfg.name}")
		self.log("=" * 70)
		if cfg.description:
			self.log(f"  {cfg.description}")
		self.log(f"  Experiments: {len(cfg.experiments)}")
		if resume_from:
			self.log(f"  Resuming from experiment {resume_from}")
		self.log("")

		# Register with dashboard if client available (skip if flow_id already set)
		if self.dashboard_client and self._flow_id is None:
			try:
				seed_checkpoint_id = None
				if cfg.seed_checkpoint_path:
					# Look up checkpoint ID from path
					seed_checkpoint_id = self.dashboard_client.find_checkpoint_by_path(
						cfg.seed_checkpoint_path
					)
					if seed_checkpoint_id:
						self.log(f"Found seed checkpoint ID: {seed_checkpoint_id}")

				self._flow_id = self.dashboard_client.create_flow(
					cfg.to_api_config(),
					seed_checkpoint_id=seed_checkpoint_id,
				)
				self.dashboard_client.flow_started(self._flow_id)
				self.log(f"Registered flow {self._flow_id} with dashboard")
			except Exception as e:
				self.log(f"Warning: Failed to register flow with dashboard: {e}")
		elif self._flow_id is not None:
			self.log(f"Using existing flow {self._flow_id}")

		# Create all experiments upfront with pending status (for both new and existing flows)
		# This ensures experiments exist in DB before we start running them
		if self.tracker and self._flow_id:
			adaptation_phase_types = {
				ExperimentType.NEUROGENESIS: "neurogenesis",
				ExperimentType.SYNAPTOGENESIS: "synaptogenesis",
				ExperimentType.AXONOGENESIS: "axonogenesis",
			}
			for idx, exp_config in enumerate(cfg.experiments):
				# Compute correct max_iters from current config
				if exp_config.experiment_type == ExperimentType.GRID_SEARCH:
					phase_type = "grid_search"
					max_iters = 1
				elif exp_config.experiment_type in adaptation_phase_types:
					phase_type = adaptation_phase_types[exp_config.experiment_type]
					max_iters = exp_config.iterations
				else:
					opt_target = "bits" if exp_config.optimize_bits else "neurons" if exp_config.optimize_neurons else "connections"
					phase_type = f"{'ga' if exp_config.experiment_type == ExperimentType.GA else 'ts'}_{opt_target}"
					max_iters = exp_config.generations if exp_config.experiment_type == ExperimentType.GA else exp_config.iterations

				# Check if experiment already exists for this flow/sequence
				existing_exp = self.tracker.get_experiment_by_flow_sequence(self._flow_id, idx)
				if existing_exp:
					self._experiment_ids[idx] = existing_exp["id"]
					# Update max_iterations to match current config (may have changed since creation)
					try:
						self.tracker.update_experiment_max_iterations(existing_exp["id"], max_iters)
					except Exception:
						pass  # Best-effort update
					self.log(f"Found existing experiment {existing_exp['id']}: {exp_config.name} (sequence_order={idx}, max_iterations={max_iters})")
				else:
					exp_id = self.tracker.create_pending_experiment(
						name=exp_config.name,
						flow_id=self._flow_id,
						sequence_order=idx,
						phase_type=phase_type,
						max_iterations=max_iters,
					)
					self._experiment_ids[idx] = exp_id
					self.log(f"Created pending experiment {exp_id}: {exp_config.name} (sequence_order={idx})")

		# Load seed from checkpoint if specified
		self._update_flow_status("Initializing flow...")
		if cfg.seed_checkpoint_path and not seed_genome:
			seed_genome, seed_population, seed_threshold = self._load_seed_checkpoint(
				cfg.seed_checkpoint_path
			)

		# Create initial genome from tier config if not seeded
		if seed_genome is None and cfg.tier_config:
			seed_genome = self._create_tiered_genome()

		# Stage tracking: frozen genomes per completed stage (multi-stage LM or IDS hierarchical)
		has_stages = self._has_stage_boundaries()
		is_multi_stage = cfg.architecture_type == "multi_stage" and cfg.num_stages > 1
		num_stages = cfg.num_stages if is_multi_stage else (2 if has_stages else 0)
		frozen_genomes: list[Optional[ClusterGenome]] = [None] * num_stages if has_stages else []
		# Track populations + metrics per stage for combined validation (best_ce, best_acc, best_fitness)
		frozen_populations: dict[int, tuple[list[ClusterGenome], list[tuple[float, float]]]] = {}

		# Run experiments
		start_idx = resume_from or 0
		current_genome = seed_genome
		current_population = seed_population
		current_threshold = seed_threshold
		current_fitness: Optional[float] = None
		current_evals: Optional[list[tuple[float, float]]] = None  # Cached metrics from previous phase
		stopped_at_idx: Optional[int] = None  # Track where we stopped for checkpoint

		# Auto-detect resume point from completed experiments in the database.
		# This handles the case where a flow crashed (no graceful stop checkpoint)
		# and the dashboard re-queues it without setting start_from_experiment.
		# We only determine start_idx here — the existing skip loop below
		# handles loading checkpoints and setting current_genome/population/etc.
		if resume_from is None and self.tracker and self._flow_id and self.checkpoint_dir:
			auto_resume_idx = 0
			for idx in range(len(cfg.experiments)):
				exp_id = self._experiment_ids.get(idx)
				if not exp_id:
					break
				existing_exp = self.tracker.get_experiment_by_flow_sequence(self._flow_id, idx)
				if not existing_exp or existing_exp.get("status") != "completed":
					break
				# Verify checkpoint file exists before committing to skip
				exp_dir = self.checkpoint_dir / f"exp_{idx:02d}"
				if not exp_dir.exists() or not list(exp_dir.glob("*.json.gz")):
					self.log(f"Warning: experiment {idx} ({cfg.experiments[idx].name}) completed in DB but no checkpoint file — cannot skip")
					break
				auto_resume_idx = idx + 1

			if auto_resume_idx > 0:
				start_idx = auto_resume_idx
				self.log(f"Auto-resuming from experiment {start_idx}/{len(cfg.experiments)} (skipping {start_idx} completed)")

		prev_train_idx = None  # Track previous phase's train subset to avoid collision
		try:
			for idx, exp_config in enumerate(cfg.experiments):
				if idx < start_idx:
					# Load checkpoint for skipped experiments
					self._update_flow_status(f"Loading checkpoint for experiment {idx + 1}/{len(cfg.experiments)}: {exp_config.name}")
					result = self._load_experiment_checkpoint(idx)
					if result:
						self._results.append(result)
						current_genome = result.best_genome
						current_population = result.final_population
						current_threshold = result.final_threshold
						current_fitness = result.final_fitness
						current_evals = result.population_metrics
						# Self-heal old checkpoints: evaluate population and re-save with metrics
						if current_evals is None and current_population:
							self._update_flow_status(f"Backfilling metrics for experiment {idx + 1}: {exp_config.name} ({len(current_population)} genomes)")
							self.log(f"  Backfilling population_metrics for experiment {idx} ({len(current_population)} genomes)...")
							eval_results = self.evaluator.evaluate_batch(current_population)
							current_evals = [(r.ce, r.accuracy) for r in eval_results]
							result.population_metrics = current_evals
							self._resave_checkpoint(idx, result)
						self.log(f"Loaded checkpoint for experiment {idx}: CE={current_fitness:.4f}")
					else:
						# Checkpoint not found - try to query database for completed phase results
						self.log(f"Warning: No checkpoint found for experiment {idx}, querying database...")
						db_result = self._load_from_database(idx, exp_config)
						if db_result:
							current_genome, current_population, current_threshold, current_fitness = db_result
							self.log(f"Loaded from database for experiment {idx}: CE={current_fitness:.4f}")
						else:
							# Cannot skip this experiment without its results
							raise ValueError(
								f"Cannot resume from experiment {start_idx}: "
								f"No checkpoint or database results found for experiment {idx} ({exp_config.name}). "
								f"Either run from the beginning or provide a valid checkpoint."
							)
					# Stage boundary detection during skip
					if has_stages and idx + 1 < len(cfg.experiments):
						next_config = cfg.experiments[idx + 1]
						if hasattr(next_config, 'target_stage') and next_config.target_stage != exp_config.target_stage:
							self._handle_stage_boundary(
								prev_stage=exp_config.target_stage,
								next_stage=next_config.target_stage,
								current_genome=current_genome,
								current_population=current_population,
								current_evals=current_evals,
								frozen_genomes=frozen_genomes,
								frozen_populations=frozen_populations,
								verbose=False,
							)
							current_genome = None
							current_population = None
							current_fitness = None
							current_threshold = None
							current_evals = None

					continue

				# Update flow status for dashboard visibility
				self._update_flow_status(f"Starting experiment {idx + 1}/{len(cfg.experiments)}: {exp_config.name}")

				# Create experiment checkpoint directory
				exp_checkpoint_dir = None
				if self.checkpoint_dir:
					exp_checkpoint_dir = self.checkpoint_dir / f"exp_{idx:02d}"

				# Create experiment in database for this config spec
				# Each config spec becomes its own experiment with proper name and sequence_order
				experiment_id = None
				tracker_experiment_id = None

				# Convert tier_config to string format for DB
				tier_config_str = None
				if cfg.tier_config is not None:
					tier_parts = []
					for tier in cfg.tier_config:
						if tier[0] is None:
							tier_parts.append(f"rest,{tier[1]},{tier[2]}")
						else:
							tier_parts.append(f"{tier[0]},{tier[1]},{tier[2]}")
					tier_config_str = ";".join(tier_parts)

				# Determine phase type string for tracking
				adaptation_phase_types = {
					ExperimentType.NEUROGENESIS: "neurogenesis",
					ExperimentType.SYNAPTOGENESIS: "synaptogenesis",
					ExperimentType.AXONOGENESIS: "axonogenesis",
				}
				if exp_config.experiment_type == ExperimentType.GRID_SEARCH:
					phase_type = "grid_search"
				elif exp_config.experiment_type == ExperimentType.LAMBDA_SWEEP:
					phase_type = "lambda_sweep"
				elif exp_config.experiment_type in adaptation_phase_types:
					phase_type = adaptation_phase_types[exp_config.experiment_type]
				else:
					opt_target = "bits" if exp_config.optimize_bits else "neurons" if exp_config.optimize_neurons else "connections"
					phase_type = f"{'ga' if exp_config.experiment_type == ExperimentType.GA else 'ts'}_{opt_target}"

				# Check if experiment already exists (created when flow was queued from dashboard)
				existing_experiment_id = self._experiment_ids.get(idx)
				if existing_experiment_id:
					experiment_id = existing_experiment_id
					tracker_experiment_id = existing_experiment_id
					# Update existing experiment status to running
					if self.dashboard_client:
						try:
							self.dashboard_client.experiment_started(experiment_id)
							self.log(f"Started experiment {experiment_id}: {exp_config.name} (existing)")
						except Exception as e:
							self.log(f"Warning: Failed to update experiment status: {e}")
					elif self.tracker:
						try:
							self.tracker.update_experiment_status(experiment_id, "running")
							self.log(f"Started experiment {experiment_id}: {exp_config.name} (existing)")
						except Exception as e:
							self.log(f"Warning: Failed to update experiment status via tracker: {e}")
				elif self.tracker:
					try:
						# Create experiment with config spec name and sequence_order
						tracker_experiment_id = self.tracker.start_experiment(
							name=exp_config.name,  # Use config spec name, NOT flow name
							flow_id=self._flow_id,
							sequence_order=idx,
							tier_config=tier_config_str,
							context_size=cfg.context_size,
							population_size=exp_config.population_size,
							phase_type=phase_type,
							max_iterations=1 if exp_config.experiment_type in (ExperimentType.GRID_SEARCH, ExperimentType.LAMBDA_SWEEP) else (exp_config.generations if exp_config.experiment_type == ExperimentType.GA else exp_config.iterations),
						)
						experiment_id = tracker_experiment_id
						self._experiment_ids[idx] = experiment_id
						self.log(f"Created experiment {experiment_id}: {exp_config.name} (sequence_order={idx})")
					except Exception as e:
						self.log(f"Warning: Failed to create experiment via tracker: {e}")

				# Fallback to dashboard_client if tracker not available
				if not experiment_id and self.dashboard_client:
					try:
						experiment_id = self.dashboard_client.create_experiment(
							name=exp_config.name,
							log_path=str(exp_checkpoint_dir) if exp_checkpoint_dir else "",
							config=exp_config.to_dict(),
						)
						self._experiment_ids[idx] = experiment_id

						# Link experiment to flow
						if self._flow_id:
							self.dashboard_client.link_experiment_to_flow(
								flow_id=self._flow_id,
								experiment_id=experiment_id,
								sequence_order=idx,
							)
						self.log(f"Created experiment {experiment_id}: {exp_config.name} (via dashboard)")
					except Exception as e:
						self.log(f"Warning: Failed to create experiment in dashboard: {e}")

				# Check for shutdown before starting experiment
				if self.shutdown_check and self.shutdown_check():
					self.log(f"Shutdown requested, stopping flow before experiment {idx}")
					stopped_at_idx = idx
					raise FlowStoppedError("Shutdown requested")

				# ── Lambda sweep: eval-only, no training ──
				if exp_config.experiment_type == ExperimentType.LAMBDA_SWEEP:
					result = self._run_lambda_sweep(
						exp_config, experiment_id, exp_checkpoint_dir,
					)
					self._results.append(result)
					# Lambda sweep doesn't update genome state
					continue

				# Create and run experiment
				# Run init validation on first experiment only (Phase 1a)
				experiment = Experiment(
					config=exp_config,
					evaluator=self.evaluator,
					logger=self.log,
					checkpoint_dir=exp_checkpoint_dir,
					dashboard_client=self.dashboard_client,
					experiment_id=experiment_id,
					tracker=self.tracker,
					flow_id=self._flow_id,
					shutdown_check=self.shutdown_check,
					full_evaluator=self.full_evaluator,
				)

				# Random train subset, avoiding previous phase's subset
				n = self.evaluator.num_parts
				if prev_train_idx is None:
					exp_train_idx = random.randint(0, n - 1)
				else:
					candidates = [i for i in range(n) if i != prev_train_idx]
					exp_train_idx = random.choice(candidates)
				prev_train_idx = exp_train_idx

				result = experiment.run(
					initial_genome=current_genome,
					initial_fitness=current_fitness if exp_config.experiment_type == ExperimentType.TS else None,
					initial_population=current_population,
					initial_threshold=current_threshold,
					tracker_experiment_id=tracker_experiment_id,  # Pass this experiment's ID
					initial_evals=current_evals,
					train_subset_idx=exp_train_idx,
				)

				self._results.append(result)

				# Check if experiment was stopped due to shutdown
				if result.was_shutdown:
					self.log(f"Experiment {idx} stopped due to shutdown, stopping flow")
					stopped_at_idx = idx
					raise FlowStoppedError("Shutdown requested during experiment")

				# Also check shutdown_check after experiment completes
				# (in case shutdown was requested while experiment was finishing)
				if self.shutdown_check and self.shutdown_check():
					self.log(f"Shutdown detected after experiment {idx}, stopping flow")
					stopped_at_idx = idx
					raise FlowStoppedError("Shutdown requested after experiment")

				# Update state for next experiment
				current_genome = result.best_genome
				current_population = result.final_population
				current_threshold = result.final_threshold
				current_fitness = result.final_fitness
				current_evals = result.population_metrics

				# IDS: compute and store extra metrics (F1, FPR, per-class) on test set
				if cfg.architecture_type == "ids" and self.full_evaluator and current_genome and tracker_experiment_id:
					try:
						self._store_ids_metrics(current_genome, tracker_experiment_id, result)
					except Exception as e:
						self.log(f"  Warning: Failed to compute IDS metrics: {e}")

				# Detect stage boundary (multi-stage LM or IDS hierarchical)
				if has_stages and idx + 1 < len(cfg.experiments):
					next_config = cfg.experiments[idx + 1]
					if hasattr(next_config, 'target_stage') and next_config.target_stage != exp_config.target_stage:
						self._handle_stage_boundary(
							prev_stage=exp_config.target_stage,
							next_stage=next_config.target_stage,
							current_genome=current_genome,
							current_population=current_population,
							current_evals=current_evals,
							frozen_genomes=frozen_genomes,
							frozen_populations=frozen_populations,
						)
						current_genome = None
						current_population = None
						current_fitness = None
						current_threshold = None
						current_evals = None

			# Flow completed successfully
			if self.dashboard_client and self._flow_id:
				try:
					self.dashboard_client.flow_completed(self._flow_id)
				except Exception as e:
					self.log(f"Warning: Failed to mark flow completed: {e}")

		except FlowStoppedError:
			# Flow was stopped gracefully (shutdown requested)
			self.log("Flow stopped due to shutdown request")

			# Save checkpoint to database so we can resume later
			if self.checkpoint_dir and current_genome and self.dashboard_client and self._flow_id:
				try:
					checkpoint_id = self._save_stop_checkpoint_to_db(
						stopped_at_idx=stopped_at_idx or len(self._results),
						current_genome=current_genome,
						current_fitness=current_fitness,
						current_population=current_population,
						current_threshold=current_threshold,
					)
					if checkpoint_id:
						# Update flow's seed_checkpoint_id so it resumes from here
						self.dashboard_client.set_flow_checkpoint(self._flow_id, checkpoint_id)
						self.log(f"Checkpoint saved to database (id={checkpoint_id})")
				except Exception as e:
					self.log(f"Warning: Failed to save stop checkpoint: {e}")

			if self.dashboard_client and self._flow_id:
				try:
					self.dashboard_client.update_flow(self._flow_id, status="cancelled")
				except Exception:
					pass
			raise

		except Exception as e:
			# Flow failed
			if self.dashboard_client and self._flow_id:
				try:
					self.dashboard_client.flow_failed(self._flow_id, str(e))
				except Exception:
					pass
			raise

		elapsed = time.time() - start_time

		# Get final result (handle edge case of no results)
		if not self._results:
			raise ValueError("Flow completed but no experiment results were recorded.")
		final_result = self._results[-1]

		# Multi-stage: compute combined CE for all 3 genome types
		stage_genomes_list = None
		combined_ce = None
		combined_accuracy = None
		per_stage_ce = None

		if is_multi_stage and current_genome is not None:
			# Last stage genome = current_genome (best_fitness)
			frozen_genomes[cfg.num_stages - 1] = current_genome
			stage_genomes_list = list(frozen_genomes)

			# Save last stage population for combined validation
			if current_population is not None and current_evals is not None:
				frozen_populations[cfg.num_stages - 1] = (list(current_population), list(current_evals))

			# Check all stages have genomes
			if all(g is not None for g in stage_genomes_list):
				self.log("")
				self.log("Computing combined metrics across all stages for all genome types...")

				# Compute combined for all 3 genome types (best_ce, best_acc, best_fitness)
				genome_types_to_compute = self._build_combined_genome_pairs(
					frozen_genomes, frozen_populations, cfg.num_stages
				)

				for genome_type, genome_pair in genome_types_to_compute.items():
					try:
						result = self.evaluator.compute_combined_metrics(
							genome_pair,
							label_smoothing=cfg.label_smoothing,
							invalid_mode=cfg.invalid_mode,
							top_m=cfg.top_m,
							unigram_lambda=cfg.unigram_lambda,
						bigram_lambda=cfg.bigram_lambda,
						)
						ce = result.ce
						acc = result.accuracy
						stage_ces = [result.cluster_ce, result.within_ce]
						stage_accs = [result.cluster_accuracy, result.within_accuracy]
						self.log(f"  {genome_type}: CE={ce:.4f}, ACC={acc:.2%}, S0 CE={stage_ces[0]:.4f} ACC={stage_accs[0]:.2%}, S1 CE={stage_ces[1]:.4f} ACC={stage_accs[1]:.2%}")

						# Use best_fitness as the primary combined result
						if genome_type == "best_fitness":
							combined_ce = ce
							combined_accuracy = acc
							per_stage_ce = stage_ces

						# Store in dashboard
						if self.dashboard_client and self._flow_id:
							try:
								self.dashboard_client.create_combined_validation(
									flow_id=self._flow_id,
									genome_type=genome_type,
									combined_ce=ce,
									combined_accuracy=acc,
									per_stage_ce=stage_ces,
									per_stage_acc=stage_accs,
								)
							except Exception as e:
								self.log(f"  Warning: Failed to store combined validation: {e}")

					except Exception as e:
						self.log(f"  Warning: Failed to compute combined metrics for {genome_type}: {e}")

		# IDS hierarchical: combined S0→S1 validation on test set
		is_ids_hierarchical = cfg.architecture_type == "ids" and self._s0_full_evaluator is not None
		if is_ids_hierarchical and len(frozen_genomes) == 2 and all(g is not None for g in frozen_genomes):
			# Freeze S1 genome (current_genome at this point is the last S1 result)
			frozen_genomes[1] = current_genome
			stage_genomes_list = list(frozen_genomes)
			try:
				ids_combined = self._compute_ids_hierarchical_combined(
					frozen_genomes[0], frozen_genomes[1]
				)
				if ids_combined:
					combined_accuracy = ids_combined.get("accuracy")
					self.log("")
					self.log(f"  Combined 10-class: Acc={ids_combined['accuracy']:.2%}, "
							  f"F1={ids_combined['f1_macro']:.4f}, FPR={ids_combined['fpr']:.4f}")
			except Exception as e:
				self.log(f"  Warning: Failed to compute combined IDS metrics: {e}")

		self.log("")
		self.log("=" * 70)
		self.log(f"  FLOW COMPLETE: {cfg.name}")
		self.log("=" * 70)
		if combined_ce is not None:
			self.log(f"  Combined CE: {combined_ce:.4f}")
			self.log(f"  Combined Accuracy: {combined_accuracy:.2%}")
		elif combined_accuracy is not None:
			self.log(f"  Combined Accuracy: {combined_accuracy:.2%}")
		else:
			self.log(f"  Final CE: {final_result.final_fitness:.4f}")
			if final_result.final_accuracy:
				self.log(f"  Final Accuracy: {final_result.final_accuracy:.2%}")
		self.log(f"  Total Duration: {elapsed:.1f}s")
		self.log("")

		return FlowResult(
			flow_name=cfg.name,
			experiment_results=self._results,
			final_genome=final_result.best_genome,
			final_fitness=combined_ce if combined_ce is not None else final_result.final_fitness,
			final_accuracy=combined_accuracy if combined_accuracy is not None else final_result.final_accuracy,
			total_elapsed_seconds=elapsed,
			flow_id=self._flow_id,
			stage_genomes=stage_genomes_list,
			combined_ce=combined_ce,
			combined_accuracy=combined_accuracy,
			per_stage_ce=per_stage_ce,
		)

	def _store_ids_metrics(self, genome: ClusterGenome, experiment_id: int, exp_result=None):
		"""Store IDS metrics on the test set via tracker.

		Uses Rust-computed F1/FPR/accuracy from the experiment's validation pass
		(same training run as validation progression) for top-level metrics.
		Only calls predict_classes() for the per-class breakdown (confusion matrix,
		per-class F1/recall) which the Rust path doesn't provide.
		"""
		from wnn.ids.metrics import compute_ids_metrics

		evaluator = self.full_evaluator
		if not hasattr(evaluator, 'predict_classes'):
			return

		self.log(f"  Computing IDS per-class metrics on test set...")
		y_pred = evaluator.predict_classes(genome)
		y_true = evaluator.y_test
		num_classes = evaluator.num_classes
		class_names = getattr(evaluator, 'class_names', None)

		metrics = compute_ids_metrics(y_true, y_pred, num_classes)
		if class_names:
			metrics['class_names'] = class_names

		# Override top-level metrics with Rust validation values (same training run
		# as validation progression) so both dashboard views show identical numbers.
		if exp_result and exp_result.validation_f1 is not None:
			metrics['f1_macro'] = exp_result.validation_f1
			metrics['fpr'] = exp_result.validation_fpr
			metrics['accuracy'] = exp_result.validation_acc

		self.log(f"  Test: acc={metrics.get('accuracy', 0):.4f}, "
				  f"F1={metrics.get('f1_macro', 0):.4f}, "
				  f"FPR={metrics.get('fpr', 0):.4f}")

		if self.tracker:
			self.tracker.update_experiment_extra_metrics(experiment_id, metrics)

	def _compute_ids_hierarchical_combined(
		self,
		s0_genome: ClusterGenome,
		s1_genome: ClusterGenome,
	) -> Optional[dict]:
		"""Compute combined 10-class metrics for hierarchical IDS (S0→S1 pipeline).

		1. S0 classifies all test flows as Normal (0) or Attack (1)
		2. For predicted-Attack flows, S1 classifies into attack types (0-8)
		3. Remap S1 predictions back to 10-class labels (attack types → 1-9)
		4. Compute overall accuracy, F1-macro (10 classes), and FPR
		"""
		import numpy as np
		from wnn.ids.metrics import compute_ids_metrics

		s0_eval = self._s0_full_evaluator
		s1_eval = self._s1_full_evaluator

		self.log("Computing combined S0→S1 hierarchical metrics on test set...")

		# S0: predict Normal/Attack on full test set
		s0_preds = s0_eval.predict_classes(s0_genome)
		s0_true = s0_eval.y_test
		s0_preds = np.array(s0_preds)
		s0_true = np.array(s0_true)

		s0_attack_mask = s0_preds == 1
		n_predicted_attacks = int(s0_attack_mask.sum())
		self.log(f"  S0: {n_predicted_attacks}/{len(s0_preds)} predicted as Attack")

		# S1: predict attack types on the FULL attack test set (not filtered by S0)
		# S1 test evaluator has all attack flows; we need predictions for flows S0 called Attack
		s1_preds = s1_eval.predict_classes(s1_genome)
		s1_true = s1_eval.y_test

		# Build combined 10-class predictions
		# Original labels: 0=Normal, 1-9=attack types (matching y_test_multi)
		# S0 Normal → predict 0
		# S0 Attack → S1 classifies into attack type (0-8), remap to (1-9)
		combined_pred = np.zeros(len(s0_preds), dtype=np.int32)
		combined_true = np.array(s0_eval.y_test)  # Binary: 0/1

		# We need the multi-class ground truth for the full test set
		# The S0 test evaluator was created from the full dataset with classification="binary",
		# so it only has binary labels. We need the multi-class labels from the original dataset.
		# For now, compute S0 binary metrics + S1 multi-class metrics separately.
		# True combined validation requires access to the full multi-class labels.

		# Report S0 metrics
		s0_metrics = compute_ids_metrics(list(s0_true), list(s0_preds), 2)
		self.log(f"  S0 (binary): Acc={s0_metrics['accuracy']:.2%}, F1={s0_metrics['f1_macro']:.4f}, "
				  f"FPR={s0_metrics['fpr']:.4f}")

		# Report S1 metrics
		s1_metrics = compute_ids_metrics(list(s1_true), list(s1_preds), len(s1_eval.class_names or []) or 9)
		self.log(f"  S1 (9-class): Acc={s1_metrics['accuracy']:.2%}, F1={s1_metrics['f1_macro']:.4f}")

		# Store combined results in dashboard
		combined_result = {
			"accuracy": s0_metrics["accuracy"],  # For now, S0 accuracy as combined (TODO: true 10-class)
			"f1_macro": s0_metrics["f1_macro"],
			"fpr": s0_metrics["fpr"],
			"s0_accuracy": s0_metrics["accuracy"],
			"s0_f1_macro": s0_metrics["f1_macro"],
			"s0_fpr": s0_metrics["fpr"],
			"s1_accuracy": s1_metrics["accuracy"],
			"s1_f1_macro": s1_metrics["f1_macro"],
		}

		if self.dashboard_client and self._flow_id:
			try:
				self.dashboard_client.create_combined_validation(
					flow_id=self._flow_id,
					genome_type="hierarchical_combined",
					combined_ce=0.0,  # Not meaningful for IDS hierarchical
					combined_accuracy=s0_metrics["accuracy"],
					per_stage_ce=[0.0, 0.0],
					per_stage_acc=[s0_metrics["accuracy"], s1_metrics["accuracy"]],
				)
			except Exception as e:
				self.log(f"  Warning: Failed to store combined validation: {e}")

		return combined_result

	def _run_lambda_sweep(
		self,
		exp_config: ExperimentConfig,
		experiment_id: Optional[int],
		checkpoint_dir: Optional[Path],
	) -> ExperimentResult:
		"""Run a lambda sweep experiment (eval-only, no training).

		Loads genomes from specified checkpoints, sweeps unigram_lambda values,
		and stores each result as a combined validation in the dashboard.
		"""
		import time as _time
		cfg = self.config
		start = _time.time()

		self.log("")
		self.log("=" * 70)
		self.log(f"  LAMBDA SWEEP: {exp_config.name}")
		self.log("=" * 70)

		lambda_values = exp_config.lambda_values or [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
		bigram_lambda_values = exp_config.bigram_lambda_values
		self.log(f"  Lambda values: {lambda_values}")
		if bigram_lambda_values:
			self.log(f"  Bigram lambda values: {bigram_lambda_values}")
		self.log(f"  Genome type: {exp_config.genome_type}")

		# Load genomes from checkpoints
		checkpoint_ids = [exp_config.s0_checkpoint_id, exp_config.s1_checkpoint_id]
		stage_genomes: list[ClusterGenome] = []

		for stage, ckpt_id in enumerate(checkpoint_ids):
			if ckpt_id is None:
				raise ValueError(f"Missing checkpoint ID for stage {stage} (s{stage}_checkpoint_id)")

			self.log(f"  Loading S{stage} genome from checkpoint {ckpt_id}...")
			ckpt = self.dashboard_client.get_checkpoint(ckpt_id)
			if not ckpt or not ckpt.get("file_path"):
				raise ValueError(f"Checkpoint {ckpt_id} not found or has no file_path")

			ckpt_path = Path(ckpt["file_path"])
			if not ckpt_path.exists():
				raise ValueError(f"Checkpoint file not found: {ckpt_path}")

			if ckpt_path.suffix == '.gz':
				with gzip.open(ckpt_path, 'rt', encoding='utf-8') as f:
					data = json.load(f)
			else:
				with open(ckpt_path, 'r') as f:
					data = json.load(f)

			# Extract genome by type
			genome = self._extract_genome_from_checkpoint(data, exp_config.genome_type)
			if genome is None:
				raise ValueError(f"Could not find {exp_config.genome_type} genome in checkpoint {ckpt_id}")

			self.log(f"    S{stage}: {len(genome.neurons_per_cluster)} clusters, "
					 f"bits={genome.bits_per_neuron[:3]}..., neurons={genome.neurons_per_cluster[:3]}...")
			stage_genomes.append(genome)

		# Build sweep points: (unigram_lambda, bigram_lambda) pairs
		if bigram_lambda_values:
			# 2D grid sweep
			sweep_points = [(ul, bl) for ul in lambda_values for bl in bigram_lambda_values
							if ul + bl <= 1.0 + 1e-9]
			self.log("")
			self.log(f"  {'λ_uni':>8}  {'λ_bi':>8}  {'Combined CE':>12}  {'Combined Acc':>12}  {'S0 CE':>8}  {'S1 CE':>8}")
			self.log(f"  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}")
		else:
			sweep_points = [(lam, 0.0) for lam in lambda_values]
			self.log("")
			self.log(f"  {'Lambda':>8}  {'Combined CE':>12}  {'Combined Acc':>12}  {'S0 CE':>8}  {'S1 CE':>8}")
			self.log(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}")

		best_ce = float('inf')
		best_point = (0.0, 0.0)
		results_table = []

		for uni_lam, bi_lam in sweep_points:
			result = self.evaluator.compute_combined_metrics(
				stage_genomes,
				label_smoothing=cfg.label_smoothing,
				invalid_mode=cfg.invalid_mode,
				top_m=cfg.top_m,
				unigram_lambda=uni_lam,
				bigram_lambda=bi_lam,
			)
			ce = result.ce
			acc = result.accuracy
			s0_ce = result.cluster_ce
			s1_ce = result.within_ce

			if bigram_lambda_values:
				self.log(f"  {uni_lam:>8.3f}  {bi_lam:>8.3f}  {ce:>12.4f}  {acc:>11.2%}  {s0_ce:>8.4f}  {s1_ce:>8.4f}")
			else:
				self.log(f"  {uni_lam:>8.3f}  {ce:>12.4f}  {acc:>11.2%}  {s0_ce:>8.4f}  {s1_ce:>8.4f}")

			if ce < best_ce:
				best_ce = ce
				best_point = (uni_lam, bi_lam)

			results_table.append({
				"unigram_lambda": uni_lam, "bigram_lambda": bi_lam,
				"ce": ce, "accuracy": acc,
				"s0_ce": s0_ce, "s1_ce": s1_ce,
			})

			# Store each result as a combined validation
			if self.dashboard_client and self._flow_id:
				try:
					genome_label = f"uni{uni_lam:.3f}_bi{bi_lam:.3f}" if bi_lam > 0 else f"unigram_l{uni_lam:.3f}"
					self.dashboard_client.create_combined_validation(
						flow_id=self._flow_id,
						genome_type=genome_label,
						combined_ce=ce,
						combined_accuracy=acc,
						per_stage_ce=[s0_ce, s1_ce],
						unigram_lambda=uni_lam,
					)
				except Exception as e:
					self.log(f"    Warning: Failed to store validation: {e}")

		self.log("")
		if bigram_lambda_values:
			self.log(f"  Best: λ_uni={best_point[0]:.3f}, λ_bi={best_point[1]:.3f} → CE={best_ce:.4f}")
		else:
			self.log(f"  Best: λ={best_point[0]:.3f} → CE={best_ce:.4f}")

		elapsed = _time.time() - start
		self.log(f"  Completed in {elapsed:.1f}s")

		# Mark experiment as completed
		if self.dashboard_client and experiment_id:
			try:
				self.dashboard_client.experiment_completed(
					experiment_id,
					best_ce=best_ce,
					best_accuracy=max(r["accuracy"] for r in results_table),
				)
			except Exception:
				pass
		if self.tracker and experiment_id:
			try:
				self.tracker.update_experiment_status(experiment_id, "completed")
			except Exception:
				pass

		# Return a minimal ExperimentResult
		return ExperimentResult(
			experiment_name=exp_config.name,
			strategy_type="lambda_sweep",
			initial_fitness=None,
			final_fitness=best_ce,
			final_accuracy=max(r["accuracy"] for r in results_table),
			improvement_percent=0.0,
			iterations_run=len(sweep_points),
			best_genome=stage_genomes[0],  # S0 genome as placeholder
			final_population=None,
			final_threshold=None,
			elapsed_seconds=elapsed,
		)

	def _extract_genome_from_checkpoint(
		self, data: dict, genome_type: str,
	) -> Optional[ClusterGenome]:
		"""Extract a specific genome type from checkpoint data."""
		# Try pre-saved best_ce/best_acc genomes first
		if genome_type == "best_ce" and "best_ce_genome" in data:
			return ClusterGenome.deserialize(data["best_ce_genome"])
		if genome_type == "best_acc" and "best_acc_genome" in data:
			return ClusterGenome.deserialize(data["best_acc_genome"])

		# Fall back to phase_result
		if "phase_result" in data:
			pr = data["phase_result"]
			if genome_type == "best_fitness" and "best_genome" in pr:
				return ClusterGenome.deserialize(pr["best_genome"])

			# For best_ce/best_acc, search population_metrics
			pop = pr.get("final_population")
			metrics = data.get("population_metrics")
			if pop and metrics and len(pop) == len(metrics):
				if genome_type == "best_ce":
					best_idx = min(range(len(metrics)), key=lambda i: metrics[i][0])
				elif genome_type == "best_acc":
					best_idx = max(range(len(metrics)), key=lambda i: metrics[i][1])
				else:
					best_idx = 0
				return ClusterGenome.deserialize(pop[best_idx])

			# Last resort: use best_genome regardless of type
			if "best_genome" in pr:
				return ClusterGenome.deserialize(pr["best_genome"])

		return None

	def _build_combined_genome_pairs(
		self,
		frozen_genomes: list[Optional[ClusterGenome]],
		frozen_populations: dict[int, tuple[list[ClusterGenome], list[tuple[float, float]]]],
		num_stages: int,
	) -> dict[str, list[ClusterGenome]]:
		"""Build genome pairs for combined validation across stages.

		For each genome type (best_ce, best_acc, best_fitness), pick the corresponding
		genome from each stage's population and pair them together.

		Returns:
			Dict mapping genome_type -> list of genomes (one per stage)
		"""
		pairs: dict[str, list[ClusterGenome]] = {}

		# best_fitness = the frozen genomes (what was already tracked)
		if all(g is not None for g in frozen_genomes):
			pairs["best_fitness"] = [g for g in frozen_genomes if g is not None]

		# best_ce and best_acc from population metrics
		for genome_type in ("best_ce", "best_acc"):
			stage_picks: list[Optional[ClusterGenome]] = []
			for stage in range(num_stages):
				if stage in frozen_populations:
					pop, metrics = frozen_populations[stage]
					if not pop or not metrics or len(pop) != len(metrics):
						stage_picks.append(frozen_genomes[stage])
						continue

					if genome_type == "best_ce":
						# Minimum CE
						best_idx = min(range(len(metrics)), key=lambda i: metrics[i][0])
					else:
						# Maximum accuracy
						best_idx = max(range(len(metrics)), key=lambda i: metrics[i][1])
					stage_picks.append(pop[best_idx])
				else:
					# Fall back to frozen genome (best_fitness) if no population tracked
					stage_picks.append(frozen_genomes[stage])

			if all(g is not None for g in stage_picks):
				pairs[genome_type] = [g for g in stage_picks if g is not None]

		return pairs

	def _load_seed_checkpoint(
		self,
		checkpoint_path: str,
	) -> tuple[Optional[ClusterGenome], Optional[list[ClusterGenome]], Optional[float]]:
		"""Load seed data from checkpoint file."""
		try:
			path = Path(checkpoint_path)

			if path.suffix == '.gz':
				with gzip.open(path, 'rt', encoding='utf-8') as f:
					data = json.load(f)
			else:
				with open(path, 'r') as f:
					data = json.load(f)

			genome = None
			population = None
			threshold = None

			# Check for phase_result format
			if 'phase_result' in data:
				pr = data['phase_result']
				if 'best_genome' in pr:
					genome = ClusterGenome.deserialize(pr['best_genome'])
				if 'final_population' in pr and pr['final_population']:
					population = [ClusterGenome.deserialize(g) for g in pr['final_population']]
				if 'final_threshold' in pr:
					threshold = pr['final_threshold']

			# Check for final format
			elif 'final' in data:
				final = data['final']
				if 'genome' in final:
					genome = ClusterGenome.deserialize(final['genome'])
				if 'final_population' in final and final['final_population']:
					population = [ClusterGenome.deserialize(g) for g in final['final_population']]
				if 'final_threshold' in final:
					threshold = final['final_threshold']

			if genome:
				self.log(f"Loaded seed from checkpoint: {checkpoint_path}")
				if population:
					self.log(f"  Population: {len(population)} genomes")
				if threshold:
					self.log(f"  Threshold: {threshold:.4%}")

			return genome, population, threshold

		except Exception as e:
			self.log(f"Warning: Failed to load seed checkpoint: {e}")
			return None, None, None

	def _save_stop_checkpoint_to_db(
		self,
		stopped_at_idx: int,
		current_genome: ClusterGenome,
		current_fitness: Optional[float],
		current_population: Optional[list[ClusterGenome]],
		current_threshold: Optional[float],
	) -> Optional[int]:
		"""Save a checkpoint to the database when the flow is stopped.

		Returns:
			Checkpoint ID if successful, None otherwise.
		"""
		# Get the experiment_id for the stopped experiment (or the last completed one)
		experiment_id = self._experiment_ids.get(stopped_at_idx) or (
			self._experiment_ids.get(stopped_at_idx - 1) if stopped_at_idx > 0 else None
		)
		if not self.dashboard_client or not experiment_id:
			self.log("Warning: Cannot save stop checkpoint - no dashboard client or experiment_id")
			return None

		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

		# Save checkpoint file
		checkpoint_name = f"stopped_at_exp_{stopped_at_idx}"
		checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json.gz"

		data = {
			"stopped_at_experiment": stopped_at_idx,
			"total_experiments": len(self.config.experiments),
			"completed_experiments": len(self._results),
			"current_genome": current_genome.serialize() if current_genome else None,
			"current_fitness": current_fitness,
			"current_population": [g.serialize() for g in current_population] if current_population else None,
			"current_threshold": current_threshold,
			"flow_name": self.config.name,
		}

		with gzip.open(checkpoint_path, 'wt', encoding='utf-8') as f:
			json.dump(data, f, separators=(',', ':'))

		self.log(f"Checkpoint file saved: {checkpoint_path}")

		# Register in database
		try:
			checkpoint_id = self.dashboard_client.checkpoint_created(
				experiment_id=experiment_id,
				file_path=str(checkpoint_path),
				name=checkpoint_name,
				final_fitness=current_fitness,
				final_accuracy=None,
				iterations_run=stopped_at_idx,
				genome_stats={"stopped": True, "resume_from": stopped_at_idx},
				is_final=False,
				checkpoint_type="user",  # Mark as user-initiated stop checkpoint
			)
			self.log(f"  Registered checkpoint {checkpoint_id} in database")
			self.log(f"  Resume from experiment {stopped_at_idx}")
			return checkpoint_id
		except Exception as e:
			self.log(f"Warning: Failed to register checkpoint in database: {e}")
			return None

	def _load_experiment_checkpoint(self, idx: int) -> Optional[ExperimentResult]:
		"""Load checkpoint for a specific experiment."""
		if not self.checkpoint_dir:
			return None

		exp_dir = self.checkpoint_dir / f"exp_{idx:02d}"
		if not exp_dir.exists():
			return None

		# Find the checkpoint file
		checkpoints = list(exp_dir.glob("*.json.gz"))
		if not checkpoints:
			return None

		checkpoint_path = checkpoints[0]  # Take first (should be only one)

		try:
			with gzip.open(checkpoint_path, 'rt', encoding='utf-8') as f:
				data = json.load(f)

			pr = data.get('phase_result', {})
			metadata = data.get('_metadata', {})

			# Load population metrics (saved as list of [ce, acc, ...] tuples)
			pop_metrics_raw = data.get('population_metrics')
			pop_metrics = None
			if pop_metrics_raw is not None:
				pop_metrics = [tuple(m) for m in pop_metrics_raw]

			return ExperimentResult(
				experiment_name=pr.get('phase_name', f"Experiment {idx}"),
				strategy_type=pr.get('strategy_type', 'unknown'),
				initial_fitness=pr.get('initial_fitness'),
				final_fitness=pr.get('final_fitness', 0.0),
				final_accuracy=pr.get('final_accuracy'),
				improvement_percent=metadata.get('improvement_percent', 0.0),
				iterations_run=pr.get('iterations_run', 0),
				best_genome=ClusterGenome.deserialize(pr['best_genome']),
				final_population=[ClusterGenome.deserialize(g) for g in pr.get('final_population', [])] if pr.get('final_population') else None,
				final_threshold=pr.get('final_threshold'),
				elapsed_seconds=metadata.get('elapsed_seconds', 0.0),
				checkpoint_path=str(checkpoint_path),
				population_metrics=pop_metrics,
			)

		except Exception as e:
			self.log(f"Warning: Failed to load checkpoint for experiment {idx}: {e}")
			return None

	def _resave_checkpoint(self, idx: int, result: ExperimentResult) -> None:
		"""Re-save a checkpoint with updated fields (e.g. backfilled population_metrics)."""
		if not self.checkpoint_dir:
			return
		exp_dir = self.checkpoint_dir / f"exp_{idx:02d}"
		checkpoints = list(exp_dir.glob("*.json.gz"))
		if not checkpoints:
			return
		checkpoint_path = checkpoints[0]
		try:
			with gzip.open(checkpoint_path, 'rt', encoding='utf-8') as f:
				data = json.load(f)
			if result.population_metrics is not None:
				data["population_metrics"] = result.population_metrics
			with gzip.open(checkpoint_path, 'wt', encoding='utf-8') as f:
				json.dump(data, f, separators=(',', ':'))
			self.log(f"  Re-saved checkpoint with population_metrics: {checkpoint_path}")
		except Exception as e:
			self.log(f"  Warning: Failed to re-save checkpoint: {e}")

	def _load_from_database(
		self,
		idx: int,
		exp_config: 'ExperimentConfig',
	) -> Optional[tuple[ClusterGenome, Optional[list[ClusterGenome]], Optional[float], float]]:
		"""
		Query database for completed phase results when checkpoint is missing.

		Returns tuple of (genome, population, threshold, fitness) or None if not found.
		"""
		if not self.dashboard_client or not self._flow_id:
			return None

		try:
			# Get experiments for this flow
			flow = self.dashboard_client.get_flow(self._flow_id)
			if not flow:
				return None

			# Find completed phases for this flow's experiments
			# We need to match by phase type and sequence
			experiments = self.dashboard_client.list_experiments(flow_id=self._flow_id)
			if not experiments:
				return None

			# Look for a completed phase matching this experiment's type
			phase_type = "ga" if exp_config.experiment_type == ExperimentType.GA else "ts"
			optimize_what = "neurons" if exp_config.optimize_neurons else ("bits" if exp_config.optimize_bits else "connections")
			expected_phase_type = f"{phase_type}_{optimize_what}"

			for exp in experiments:
				phases = self.dashboard_client.get_phases(exp['id'])
				for phase in phases:
					if phase.get('status') == 'completed' and phase.get('phase_type') == expected_phase_type:
						best_ce = phase.get('best_ce')
						best_acc = phase.get('best_accuracy')

						if best_ce is not None:
							self.log(f"Found completed phase in database: {phase.get('name')} CE={best_ce:.4f}")

							# We have the fitness, but we need the genome too
							# Try to find a checkpoint registered in the database
							checkpoints = self.dashboard_client.list_checkpoints(experiment_id=exp['id'])
							for ckpt in checkpoints:
								if ckpt.get('file_path'):
									try:
										genome, population, threshold = self._load_seed_checkpoint(ckpt['file_path'])
										if genome:
											return (genome, population, threshold, best_ce)
									except Exception:
										pass

							# If no checkpoint with genome found, we can't proceed
							self.log(f"Warning: Found completed phase but no checkpoint with genome data")
							return None

			return None

		except Exception as e:
			self.log(f"Warning: Failed to query database for experiment {idx}: {e}")
			return None

	def _create_tiered_genome(self) -> ClusterGenome:
		"""Create a genome with tiered configuration."""
		if not self.config.tier_config:
			return ClusterGenome.create_uniform(
				num_clusters=self.evaluator.vocab_size,
				bits=8,
				neurons=5,
			)

		cluster_bits = []
		neurons_per_cluster = []
		cluster_idx = 0

		for tier_spec in self.config.tier_config:
			# tier_spec can be (count, neurons, bits) or (count, neurons, bits, optimize)
			num_clusters, neurons, bits = tier_spec[0], tier_spec[1], tier_spec[2]
			if num_clusters is None:
				count = self.evaluator.vocab_size - cluster_idx
			else:
				count = min(num_clusters, self.evaluator.vocab_size - cluster_idx)

			cluster_bits.extend([bits] * count)
			neurons_per_cluster.extend([neurons] * count)
			cluster_idx += count

			if cluster_idx >= self.evaluator.vocab_size:
				break

		# Pad if needed
		while len(cluster_bits) < self.evaluator.vocab_size:
			cluster_bits.append(8)
			neurons_per_cluster.append(5)

		# Expand per-cluster bits to per-neuron
		cluster_bits = cluster_bits[:self.evaluator.vocab_size]
		neurons_per_cluster = neurons_per_cluster[:self.evaluator.vocab_size]
		bits_per_neuron = []
		for b, n in zip(cluster_bits, neurons_per_cluster):
			bits_per_neuron.extend([b] * n)

		genome = ClusterGenome(
			bits_per_neuron=bits_per_neuron,
			neurons_per_cluster=neurons_per_cluster,
		)

		# Initialize connections
		if not genome.has_connections():
			genome.initialize_connections(self.evaluator.total_input_bits)

		return genome
