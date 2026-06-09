"""ControllerEvaluator — mirror of IDSEvaluator for the controller pipeline.

Same role: given a "genome" (architecture + connectivity + learned cells),
evaluate it on a held-out set of test episodes and return metrics. The
worker dispatches `architecture_type='controller'` flows here instead of
IDSEvaluator.

A "genome" for the controller is:
  - (state_neurons, state_bits_per_neuron, output_bits_per_neuron):
    shape parameters that come from the grid_search / ga_neurons phases.
  - state_connections, output_connections: which input bits each neuron
    addresses. Same role as IDS connections — evolved by the GA.
  - thresholds: per-feature thermometer thresholds (NUM_FEATURES *
    bits_per_feature floats). Either fitted from sim rollouts at
    evaluator init (preferred) or supplied externally.
  - state_cells, output_cells: trained cell values. The trainer
    (BPTT/EDRA) writes these. For an untrained genome the cells are
    empty (default WEAK_FALSE) and the controller emits its default
    PWM=0.75 from Strategy 5.

The evaluator is reproducible via a seed: every episode uses an RNG
derived deterministically from (seed, episode_idx).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ram_accelerator import AttitudeSim, WnnController

from .pid import AttitudePID, AttitudePIDConfig
from .training import (
	EpisodeConfig,
	fitness_function,
	make_pid_action_fn,
	make_wnn_action_fn,
)


# Layout constants matching controller.rs NUM_FEATURES = 9
NUM_FEATURES = 9


def _rust_dagger_enabled() -> bool:
	"""Native Rust reward-gated DAGGER training is the DEFAULT (opt-OUT).

	It runs the whole rollout+train loop in Rust (Rayon-parallel across genomes,
	~3× faster with cores) AND computes the motor-jerk + monotonicity-violation
	fitness metrics that the Python reference path drops. The Python path
	(reward_gated_train) is a slow, single-genome reference/parity fallback — it
	is NOT what you want by default. Set WNN_RUST_DAGGER=0/off/false/no to force
	it for parity tests or debugging.

	Flipped from opt-in → opt-out 01/06/2026: the path was gated for validation
	when added (29/05), parity was confirmed (test_controller_gpu_parity /
	_solver_parity), but the default was never flipped — so every default run was
	slow and silently ignored weight_jerk/weight_mono."""
	import os
	return os.environ.get("WNN_RUST_DAGGER", "1").strip().lower() not in ("0", "false", "off", "no")


@dataclass
class ControllerSpec:
	"""Shape/architecture of a controller — the equivalent of IDS's
	(neurons, bits) genome but extended for the controller-specific
	layers + window."""
	num_motors: int = 4
	levels_per_motor: int = 256
	bits_per_feature: int = 8
	input_window_k: int = 4

	# State layer
	state_neurons: int = 200
	state_bits_per_neuron: int = 18

	# Output layer (one neuron per motor × level)
	output_bits_per_neuron: int = 18

	# Delta-control: output decodes to a per-step PWM delta (accumulated),
	# not an absolute throttle. Untrained → hold (stable bootstrap). See
	# project_controller_state. delta_max is the per-step clamp.
	delta_control: bool = False
	delta_max: float = 0.1
	# Leaky integrator (delta mode): accumulator deviation from hover decays by
	# delta_leak each step. 1.0 = pure integrator (can run away); <1.0 bounds the
	# steady-state offset to delta/(1-leak).
	delta_leak: float = 1.0


@dataclass
class AdaptationStats:
	"""Per-genome stats surfaced by evaluate_for_adaptation to guide genesis.
	cell counts = distinct addresses written per neuron (the controller-side
	'fill' signal); reward/stable_rate are the closed-loop fitness components."""
	reward: float
	stable_rate: float
	state_cell_counts: list   # len = state_neurons
	output_cell_counts: list  # len = num_motors * levels_per_motor


@dataclass
class ControllerGenome:
	"""A specific instantiation of a controller — connectivity + thresholds
	+ trained cells. Built by the GA + trainer; consumed by the evaluator."""
	spec: ControllerSpec
	# Per-feature thermometer thresholds, flat: NUM_FEATURES * bits_per_feature.
	thresholds: list[float]
	# State layer connectivity, flat: state_neurons * state_bits_per_neuron.
	state_connections: list[int]
	# Output layer connectivity, flat: (num_motors * levels_per_motor) *
	# output_bits_per_neuron.
	output_connections: list[int]
	# Trained cell values per layer. Each is a list of (neuron_idx, address,
	# value) tuples. Defaults empty → all-EMPTY controller.
	state_cells: list[tuple[int, int, int]] = field(default_factory=list)
	output_cells: list[tuple[int, int, int]] = field(default_factory=list)


def fit_thresholds_from_pid_rollouts(
	spec: ControllerSpec,
	num_episodes: int = 20,
	seed: int = 0,
	method: str = "quantile",
) -> list[float]:
	"""Fit per-feature thermometer thresholds by running PID rollouts and
	collecting the empirical sensor distributions.

	Args:
		spec:         ControllerSpec for the controller architecture.
		num_episodes: PID rollouts used to gather sensor distribution data.
		seed:         RNG seed for reproducibility.
		method:       'quantile' (uniformly spaced quantiles → distributive
		              thermometer) or 'linear' (min/max linear spacing).

	Returns:
		thresholds: flat list of length NUM_FEATURES * bits_per_feature.
	"""
	rng = np.random.default_rng(seed)
	sim = AttitudeSim()
	pid = AttitudePID(AttitudePIDConfig())
	cfg = EpisodeConfig(
		dt=0.001, steps_per_episode=2000,
		max_initial_tilt_rad=math.radians(30.0),
		max_initial_yaw_rad=math.radians(30.0),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
	)

	# Collect (gyro_xyz, accel_xyz, target_rpy) samples across rollouts.
	# target_rpy is the setpoint at each step (constant per episode here).
	samples_per_feature: list[list[float]] = [[] for _ in range(NUM_FEATURES)]

	for ep_idx in range(num_episodes):
		ep_seed = int(rng.integers(0, 2**32 - 1))
		ep_rng = np.random.default_rng(ep_seed)
		# Run one PID episode while recording sensor values at every step.
		# Re-do the inner loop directly (rather than via run_episode) so we
		# can capture every sample.
		from .training import _sample_initial_state, _euler_to_quat_xyz  # type: ignore
		init_q, init_omega = _sample_initial_state(
			ep_rng,
			cfg.max_initial_tilt_rad,
			cfg.max_initial_yaw_rad,
			cfg.max_initial_body_rate,
			cfg.max_initial_yaw_rate,
		)
		sim.reset(q=list(init_q), omega=list(init_omega))
		pid.reset()
		target = (0.0, 0.0, 0.0)
		for _ in range(cfg.steps_per_episode):
			gyro, accel = sim.read_imu()
			q = sim.quaternion
			pwm = pid.step(q, gyro, target)
			# Record sensor samples
			samples_per_feature[0].append(float(gyro[0]))
			samples_per_feature[1].append(float(gyro[1]))
			samples_per_feature[2].append(float(gyro[2]))
			samples_per_feature[3].append(float(accel[0]))
			samples_per_feature[4].append(float(accel[1]))
			samples_per_feature[5].append(float(accel[2]))
			samples_per_feature[6].append(float(target[0]))
			samples_per_feature[7].append(float(target[1]))
			samples_per_feature[8].append(float(target[2]))
			sim.step(list(pwm))
			if sim.is_unstable():
				break

	# Now derive thresholds per feature.
	bpf = spec.bits_per_feature
	thresholds = []
	for f in range(NUM_FEATURES):
		arr = np.array(samples_per_feature[f], dtype=float)
		if arr.size == 0:
			# Feature never observed (constant target?). Fall back to [-1, 1] linear.
			arr = np.array([-1.0, 1.0])
		if method == "quantile":
			# Uniform percentiles 1/(bpf+1)..bpf/(bpf+1)
			qs = np.linspace(1.0 / (bpf + 1), bpf / (bpf + 1), bpf)
			ts = np.quantile(arr, qs)
		elif method == "linear":
			lo, hi = float(arr.min()), float(arr.max())
			# If lo == hi (constant feature), spread by ±1 around it
			if hi - lo < 1e-9:
				lo, hi = lo - 1.0, hi + 1.0
			ts = np.linspace(lo, hi, bpf, endpoint=False)
		else:
			raise ValueError(f"unknown method: {method!r}")
		thresholds.extend(float(t) for t in ts)
	return thresholds


def random_connectivity(spec: ControllerSpec, seed: int = 0) -> tuple[list[int], list[int]]:
	"""Structured connectivity for a coherent recurrent FSM.

	For the network to behave as one automaton (not N disjoint mini-automata),
	every state neuron and every output neuron must observe the FULL state
	(see DFA argument: the next-state/output of any neuron depends on which
	GLOBAL state we are in). So we FORCE all state bits into each neuron's
	connections, and only the INPUT connections are sampled (the legitimate
	feature-selection / generalization knob the GA later optimizes).

	State layer input space:  [sensor window (K*F*b) | prev_state (2*n_state)].
	    Each neuron: all 2*n_state state bits + (state_bits_per_neuron - 2*n_state)
	    sampled sensor-window bits.
	Output layer input space (Mealy): [current frame (F*b) | new_state (2*n_state)].
	    Each neuron: all 2*n_state state bits + (output_bits_per_neuron - 2*n_state)
	    sampled current-frame bits.
	"""
	rng = np.random.default_rng(seed)
	n_state = spec.state_neurons
	state_bits = n_state  # forced prefix = 1 bit (MSB)/state neuron (08/06/2026; was 2·n_state)
	sensor_window = spec.input_window_k * NUM_FEATURES * spec.bits_per_feature
	sensor_frame = NUM_FEATURES * spec.bits_per_feature

	n_state_sampled = spec.state_bits_per_neuron - state_bits
	n_out_sampled = spec.output_bits_per_neuron - state_bits
	if n_state_sampled < 0 or n_out_sampled < 0:
		raise ValueError(
			f"bits_per_neuron must be >= state_neurons ({state_bits}) for full-state "
			f"connectivity: state={spec.state_bits_per_neuron}, output={spec.output_bits_per_neuron}"
		)

	# State layer: state bits live at [sensor_window, sensor_window+state_bits).
	state_state_idx = list(range(sensor_window, sensor_window + state_bits))
	state_conn: list[int] = []
	for _ in range(n_state):
		sampled = (rng.choice(sensor_window, size=min(n_state_sampled, sensor_window), replace=False).tolist()
		           if n_state_sampled > 0 else [])
		state_conn.extend(state_state_idx + [int(x) for x in sampled])

	# Output layer (Mealy): state bits live at [sensor_frame, sensor_frame+state_bits).
	out_state_idx = list(range(sensor_frame, sensor_frame + state_bits))
	num_output_neurons = spec.num_motors * spec.levels_per_motor
	output_conn: list[int] = []
	for _ in range(num_output_neurons):
		sampled = (rng.choice(sensor_frame, size=min(n_out_sampled, sensor_frame), replace=False).tolist()
		           if n_out_sampled > 0 else [])
		output_conn.extend(out_state_idx + [int(x) for x in sampled])

	return [int(x) for x in state_conn], [int(x) for x in output_conn]


def build_controller(genome: ControllerGenome) -> WnnController:
	"""Instantiate a Rust WnnController from a ControllerGenome and apply
	all learned cells. The Rust controller takes connectivity at
	construction time and cell writes via the per-layer write methods."""
	spec = genome.spec
	c = WnnController(
		num_motors=spec.num_motors,
		levels_per_motor=spec.levels_per_motor,
		bits_per_feature=spec.bits_per_feature,
		input_window_k=spec.input_window_k,
		state_neurons=spec.state_neurons,
		state_bits_per_neuron=spec.state_bits_per_neuron,
		output_bits_per_neuron=spec.output_bits_per_neuron,
		thresholds=genome.thresholds,
		state_connections=genome.state_connections,
		output_connections=genome.output_connections,
		delta_control=spec.delta_control,
		delta_max=spec.delta_max,
		delta_leak=spec.delta_leak,
	)
	for (n, addr, v) in genome.state_cells:
		c.write_state_cell(n, addr, v)
	for (n, addr, v) in genome.output_cells:
		c.write_output_cell(n, addr, v)
	return c


# ---------------------------------------------------------------------------
# Drone adapter: the ONLY place where drone vocabulary (motors, levels, sensor
# window) meets the domain-free RecurrentArchGenome. Keeps the generic genome
# reusable by any two-layer recurrent RAM arch (see recurrent_genome.py).
# ---------------------------------------------------------------------------

def arch_shape_from_spec(spec: ControllerSpec) -> "RecurrentArchShape":
	"""Project the drone ControllerSpec onto the genome's fixed structural
	constants: motors/levels → output count granularity, K·F·b → input spaces."""
	from .recurrent_genome import RecurrentArchShape
	return RecurrentArchShape(
		# 08/06/2026: recurrent state output is now 1 bit/neuron (the QSR MSB =
		# fired/not), NOT 2 (the LSB was training-confidence, semantically wrong to
		# feed back). Halves the forced prefix (sn, not 2·sn). Must match the Rust
		# controller's state_bits_in = state_neurons.
		prefix_factor=1,  # state output = 1 bit (MSB) per state neuron
		state_input_space=spec.input_window_k * NUM_FEATURES * spec.bits_per_feature,
		output_input_space=NUM_FEATURES * spec.bits_per_feature,
		output_quantum=spec.num_motors,  # one PWM level = num_motors output neurons
	)


def spec_from_arch(genome: "RecurrentArchGenome", base: ControllerSpec) -> ControllerSpec:
	"""Rebuild a concrete ControllerSpec from a genome's evolved shape, inheriting
	the fixed environment params (motors, sensor encoding, delta config) from
	`base`. `levels_per_motor` is DERIVED from output_neurons / num_motors."""
	return ControllerSpec(
		num_motors=base.num_motors,
		levels_per_motor=genome.output_neurons // base.num_motors,
		bits_per_feature=base.bits_per_feature,
		input_window_k=base.input_window_k,
		state_neurons=genome.state_neurons,
		state_bits_per_neuron=genome.state_bits_per_neuron,
		output_bits_per_neuron=genome.output_bits_per_neuron,
		delta_control=base.delta_control,
		delta_max=base.delta_max,
		delta_leak=base.delta_leak,
	)


def controller_genome_from_arch(
	genome: "RecurrentArchGenome", base: ControllerSpec, thresholds: list[float],
	state_cells: list | None = None, output_cells: list | None = None,
) -> ControllerGenome:
	"""Materialize a generic arch genome into a concrete, buildable ControllerGenome
	(connectivity + thresholds + optional cells). Explicit cell args win; otherwise
	the genome's own MemoryPayload (if any) is used — so a unified genome carrying
	evolved/Lamarckian cells builds directly."""
	sc, oc = genome.to_connections()
	if state_cells is None and output_cells is None and genome.cells is not None:
		state_cells, output_cells = genome.cells.to_triples()
	return ControllerGenome(
		spec=spec_from_arch(genome, base),
		thresholds=thresholds,
		state_connections=sc,
		output_connections=oc,
		state_cells=state_cells or [],
		output_cells=output_cells or [],
	)


class ControllerEvaluator:
	"""Evaluate a controller genome over a held-out episode set.

	Mirrors the IDSEvaluator interface used by the GA/grid-search phases:
	  - `__init__(spec, num_episodes, seed)`: prepare the evaluator with
	    the architecture spec and the held-out episode plan.
	  - `evaluate(genome)`: returns (fitness_scalar, metrics_dict).
	  - `validate(genome)`: same as evaluate but for the final validation
	    checkpoint — uses a larger episode count for tighter statistics.

	The fitness scalar is `mean_reward`, which is negative (since reward =
	-attitude_error² typically). The GA should maximize fitness, so the
	convention matches: higher = better.
	"""

	def __init__(
		self,
		spec: ControllerSpec,
		num_eval_episodes: int = 30,
		num_validate_episodes: int = 100,
		seed: int = 0,
		episode_config: Optional[EpisodeConfig] = None,
		thresholds: Optional[list[float]] = None,
		rg_config=None,
		max_train_workers: int = 1,
		max_eval_workers_gpu: bool = True,
		fitness_seeds: int = 1,
		num_eval_folds: int = 1,
	):
		self.spec = spec
		self.num_eval = num_eval_episodes
		self.num_validate = num_validate_episodes
		self.seed = seed
		# K-fold scoring (added 30/05/2026 post-Plan-A-v1-overfit diagnosis).
		# num_eval_folds=1 reproduces legacy single-pool behavior. With K>1, the
		# evaluator pre-generates K deterministic episode-pool seeds (from `seed`)
		# and rotates through them per evaluate_batch call. _active_score_seed is
		# what the GPU/CPU scoring paths read for episode IC sampling — defaults
		# to self.seed, gets overwritten to fold_seeds[k] when a fold is active.
		self.num_eval_folds = max(1, num_eval_folds)
		self._fold_seeds = [
			(seed * 0x9E3779B97F4A7C15 + k * 0xBF58476D1CE4E5B9) & 0xFFFFFFFF
			for k in range(self.num_eval_folds)
		]
		self._fold_counter = 0
		self._active_score_seed = seed
		self.episode_config = episode_config or EpisodeConfig(
			dt=0.001, steps_per_episode=2000,
			max_initial_tilt_rad=math.radians(30.0),
			max_initial_yaw_rad=math.radians(30.0),
			max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
		)
		# Each evaluator owns its own AttitudeSim (cheap to construct;
		# stateless across episodes after reset).
		self._sim = AttitudeSim()
		# GA-path config: shared (PID-fit) thresholds held across all genomes,
		# and the inner-loop trainer config (carries target_source = C1 "pid" /
		# C2 "student"). Lazily filled if not supplied.
		self.thresholds = thresholds
		self.rg_config = rg_config
		# CPU across-genome parallelism for the inner training step. Each genome
		# has its own WnnController (independent), so threads parallelise the
		# GIL-releasing Rust solver. Capped to coexist with the IDS worker.
		self.max_train_workers = max_train_workers
		# Use the GPU-batched Metal kernel for the closed-loop SCORING step
		# (training stays CPU). Falls back to CPU if Metal is unavailable.
		self.max_eval_workers_gpu = max_eval_workers_gpu
		# Multi-seed genome fitness (A): the inner loop is chaotic, so the SAME
		# connectivity yields different controllers per training seed. Averaging
		# the closed-loop score over K independent train+score seeds gives the GA
		# a stable estimate to climb (variance ÷√K) instead of selecting noise.
		self.fitness_seeds = fitness_seeds

	def evaluate(self, genome: ControllerGenome) -> tuple[float, dict]:
		"""Returns (fitness, metrics) over num_eval episodes."""
		controller = build_controller(genome)
		action_fn = make_wnn_action_fn(controller)
		mean_reward, metrics = fitness_function(
			action_fn, self._sim, self.episode_config,
			num_episodes=self.num_eval, seed=self.seed,
		)
		return mean_reward, metrics

	def validate(self, genome: ControllerGenome) -> tuple[float, dict]:
		"""Higher-episode-count validation pass for the final checkpoint."""
		controller = build_controller(genome)
		action_fn = make_wnn_action_fn(controller)
		mean_reward, metrics = fitness_function(
			action_fn, self._sim, self.episode_config,
			num_episodes=self.num_validate, seed=self.seed + 1_000_000,
		)
		return mean_reward, metrics

	def train_and_evaluate(
		self,
		thresholds: list[float],
		state_connections: list[int],
		output_connections: list[int],
		dagger_config=None,
	) -> tuple[float, dict]:
		"""DAGGER-train a controller with the given connectivity, then score it.

		The controller analog of training+evaluating an IDS genome: the GA
		supplies connectivity (+ PID-fit thresholds); DAGGER fills the QSR
		cells by rolling out the student and labelling with the PID teacher;
		we return the closed-loop fitness. This is what
		`architecture_type='controller'` grid_search/ga flows call per genome.

		The final score is recomputed with the evaluator's own seed so it is
		directly comparable to `evaluate()` / `evaluate_pid_baseline()` (same
		held-out episode set), independent of DAGGER's internal eval seed.
		"""
		from .dagger import DaggerConfig, train_dagger

		cfg = dagger_config or DaggerConfig(
			seed=self.seed,
			eval_episodes=self.num_eval,
			episode_config=self.episode_config,
		)
		controller, dagger_stats = train_dagger(
			self.spec, thresholds, state_connections, output_connections, cfg,
		)
		# Score the trained controller on the evaluator's episode set (fresh
		# recurrent state, matching build_controller's fresh-state convention).
		controller.reset()
		action_fn = make_wnn_action_fn(controller)
		fitness, metrics = fitness_function(
			action_fn, self._sim, self.episode_config,
			num_episodes=self.num_eval, seed=self.seed,
		)
		metrics["dagger"] = {
			k: dagger_stats[k] for k in (
				"iter_fitness", "iter_mean_err_deg", "iter_stable_rate",
				"iter_beta", "best_iter", "best_fitness", "final_fitness",
				"train_steps",
			)
		}
		return fitness, metrics

	def evaluate_pid_baseline(self) -> tuple[float, dict]:
		"""Run the PID baseline over the same episode set for direct comparison.
		Returns (mean_reward, metrics) — same shape as evaluate()."""
		pid = AttitudePID(AttitudePIDConfig())
		action_fn = make_pid_action_fn(pid)
		return fitness_function(
			action_fn, self._sim, self.episode_config,
			num_episodes=self.num_eval, seed=self.seed,
		)

	# ------------------------------------------------------------------
	# GA-facing batch interface (consumed by ControllerGAStrategy.optimize via
	# evaluate_fn / batch_evaluate_fn). A "genome" here is duck-typed: anything
	# with .state_connections / .output_connections (a FiniteStateGenome). The
	# shared self.thresholds + self.rg_config make the genome carry ONLY the
	# evolvable connectivity — cells are produced per-genome by the inner loop.
	# ------------------------------------------------------------------

	# ------------------------------------------------------------------
	# Compatibility shims so the shared Experiment orchestration (built around
	# the IDS ClusterGenome model) can construct + run a controller flow as a
	# first-class peer. These are only read by the orchestration's bookkeeping;
	# the controller's actual sizing comes from self.spec. vocab_size = the
	# output-neuron count (a benign positive int); there are no clusters/parts.
	# ------------------------------------------------------------------

	@property
	def vocab_size(self) -> int:
		return self.spec.num_motors * self.spec.levels_per_motor

	@property
	def num_clusters(self) -> int:
		return self.vocab_size

	@property
	def num_parts(self) -> int:
		return 1

	@property
	def total_input_bits(self) -> int:
		# State-layer input size: sensor window + recurrent-state bits. Benign for
		# the orchestration's bookkeeping; the controller's real wiring is in the genome.
		return (self.spec.input_window_k * NUM_FEATURES * self.spec.bits_per_feature
		        + self.spec.state_neurons)  # 1 bit (MSB)/state neuron (was 2·)

	def _ensure_ga_ready(self):
		from .reward_gated import RewardGatedConfig
		if self.thresholds is None:
			self.thresholds = fit_thresholds_from_pid_rollouts(self.spec, num_episodes=10, seed=self.seed)
		if self.rg_config is None:
			self.rg_config = RewardGatedConfig(seed=self.seed, episode_config=self.episode_config)

	def _materialize(self, genome) -> tuple:
		"""(spec, state_connections, output_connections) for either genome type.

		Fixed-shape FiniteStateGenome carries .state_connections + shares self.spec.
		Variable-shape RecurrentArchGenome rebuilds its own spec (via spec_from_arch)
		and connectivity (via to_connections) — so one evaluator drives both."""
		if hasattr(genome, "to_connections") and hasattr(genome, "shape"):
			sc, oc = genome.to_connections()
			return spec_from_arch(genome, self.spec), sc, oc
		return self.spec, genome.state_connections, genome.output_connections

	def _shape_key(self, genome) -> tuple:
		"""GPU-batch grouping key — the dims score_controllers_metal applies
		uniformly. Fixed-shape genomes all collapse to one key (= self.spec)."""
		spec, _sc, _oc = self._materialize(genome)
		return (spec.num_motors, spec.levels_per_motor, spec.state_neurons,
		        spec.state_bits_per_neuron, spec.output_bits_per_neuron)

	def _train_genome(self, genome, seed: int):
		"""Inner-train one genome's cells (C1 or C2 per rg_config.target_source)
		with the given training seed. Returns (WnnController, stats).

		Lamarckian warm-start: if the genome carries inherited cells (genome.cells),
		training starts FROM them (write_back stamps them; 4a remap keeps them valid
		across arch genesis). GA/TS genomes carry no cells → empty start, unchanged.

		Rust fast-path (29/05/2026): if `WNN_RUST_DAGGER=1` and the accelerator
		exposes `dagger_train_inplace`, run the full reward-gated training loop
		natively in Rust (one Python↔Rust crossing per genome, vs ~288K per-step
		crossings in the Python path). 30-100× speedup on the M4 Max for the GA
		pop-build hot path. Algorithmically equivalent to the Python reference;
		RNG values differ bit-for-bit from numpy (Rust uses SmallRng) but produce
		statistically equivalent training distributions. Falls through to Python
		on any error to preserve correctness."""
		spec, sc, oc = self._materialize(genome)
		init_s = init_o = None
		cells = getattr(genome, "cells", None)
		if cells is not None:
			init_s, init_o = cells.to_triples()
		return self._train_core(spec, sc, oc, init_s, init_o, seed)

	def _train_core(self, spec, sc, oc, init_s, init_o, seed):
		"""Train ONE controller from explicit init cells (the shared core of
		_train_genome and the K-fold accumulate loop). init_s/init_o are cell
		triples to warm-start from (None = empty memory). Rust fast-path with the
		Python reward_gated_train reference as fallback."""
		if _rust_dagger_enabled():
			try:
				return self._train_genome_rust(spec, sc, oc, init_s, init_o, seed)
			except Exception as e:
				# Loud (once-per-process): the Python fallback is ~3× slower AND
				# silently drops jerk/mono unless those are plumbed — so a silent
				# degrade here is exactly the trap we just fixed. Make it visible.
				if not getattr(self, "_rust_dagger_warned", False):
					import sys
					print(f"[ControllerEvaluator] ⚠️ Rust DAGGER FELL BACK to the slow Python "
					      f"reference path: {e}. (Set WNN_RUST_DAGGER=0 to silence if intentional.)",
					      file=sys.stderr, flush=True)
					self._rust_dagger_warned = True

		from .reward_gated import reward_gated_train
		import copy
		rg = copy.copy(self.rg_config)
		rg.seed = seed
		rg.progress = False
		return reward_gated_train(spec, self.thresholds, sc, oc, rg,
		                          init_state_cells=init_s, init_output_cells=init_o)

	def _train_genomes_rust_batched(self, genomes, tasks, init_override=None):
		"""B.5-var batched fast-path. ONE Rust call for the whole task list;
		per-genome shape vectors let Rayon parallelize across genomes regardless
		of whether their shapes match.

		Run-level dims (num_motors / levels_per_motor / bits_per_feature /
		input_window_k) are shared scalars — they don't change across genomes
		in any GA dimension. Per-genome shape (state_neurons,
		state_bits_per_neuron, output_bits_per_neuron) flows in as Vec<usize>;
		each Rayon task constructs its own WnnController with its own shape.

		Pre-29/05/2026 this method grouped by shape and called Rust once per
		group, which for variable-shape GAs (Neurons, Bits) produced many
		small calls (~30-50 groups of 4-7 candidates each), defeating the
		batching win. Now a single call handles the entire task list.

		Returns list[(controller, stats_dict)] in task-order.
		"""
		import ram_accelerator as ra

		# Materialize per-task: (spec, sc, oc, init_s, init_o, seed).
		mats = []
		for ti, (gi, seed) in enumerate(tasks):
			spec, sc, oc = self._materialize(genomes[gi])
			if init_override is not None:
				init_s, init_o = init_override[ti]
			else:
				cells = getattr(genomes[gi], "cells", None)
				init_s, init_o = (cells.to_triples() if cells is not None else (None, None))
			mats.append((spec, sc, oc, init_s or [], init_o or [], int(seed)))

		# Sanity: TRULY run-level dims (num_motors / bits_per_feature /
		# input_window_k) must agree. levels_per_motor IS per-genome because
		# spec_from_arch derives it from output_neurons // num_motors, and the
		# Neurons GA mutates output_neurons (29/05/2026 fix).
		first_spec = mats[0][0]
		for (s, *_) in mats[1:]:
			assert (s.num_motors, s.bits_per_feature, s.input_window_k) == (
				first_spec.num_motors, first_spec.bits_per_feature,
				first_spec.input_window_k), (
				"num_motors / bits_per_feature / input_window_k must be uniform "
				"across the batch; levels_per_motor / state_neurons / state_bits "
				"/ output_bits may vary (and are passed per-genome).")

		# Build packed config ONCE.
		rg = self.rg_config
		cfg = ra.RewardGatedConfigPacked(
			num_rounds=rg.num_rounds, episodes_per_round=rg.episodes_per_round,
			steps_per_episode=rg.steps_per_episode, bptt_window=rg.bptt_window,
			topk_per_neuron=rg.topk_per_neuron, protect_learned=rg.protect_learned,
			gate_mode=0 if rg.gate_mode == "improvement" else 1,
			gate_use_best=rg.gate_use_best, gate_window=rg.gate_window,
			gate_quantile=rg.gate_quantile, gate_running=rg.gate_running,
			target_source=0 if rg.target_source == "pid" else 1,
			keep_best_checkpoint=rg.keep_best_checkpoint,
			explore_eps=rg.explore_eps, explore_scale=rg.explore_scale,
			curriculum=rg.curriculum, easy_tilt_deg=rg.easy_tilt_deg,
			full_tilt_deg=rg.full_tilt_deg,
			dt=rg.episode_config.dt,
			max_initial_yaw_rad=rg.episode_config.max_initial_yaw_rad,
			max_initial_body_rate=rg.episode_config.max_initial_body_rate,
			max_initial_yaw_rate=rg.episode_config.max_initial_yaw_rate,
			eval_episodes=rg.eval_episodes,
		)
		target_rpy = list(rg.target_rpy) if rg.target_rpy is not None else [0.0, 0.0, 0.0]

		# ONE call — Rayon par_iter across the whole batch inside Rust.
		results = ra.dagger_train_batch_inplace(
			num_motors=first_spec.num_motors,
			bits_per_feature=first_spec.bits_per_feature,
			input_window_k=first_spec.input_window_k,
			levels_per_motor_per_genome=       [m[0].levels_per_motor for m in mats],
			state_neurons_per_genome=          [m[0].state_neurons for m in mats],
			state_bits_per_neuron_per_genome=  [m[0].state_bits_per_neuron for m in mats],
			output_bits_per_neuron_per_genome= [m[0].output_bits_per_neuron for m in mats],
			thresholds=self.thresholds,
			delta_control=first_spec.delta_control,
			delta_max=first_spec.delta_max,
			delta_leak=first_spec.delta_leak,
			state_connections_per_genome= [m[1] for m in mats],
			output_connections_per_genome=[m[2] for m in mats],
			init_state_cells_per_genome=  [[(int(n), int(a), int(v)) for (n, a, v) in m[3] if 0 <= int(a) < (1 << 64)] for m in mats],
			init_output_cells_per_genome= [[(int(n), int(a), int(v)) for (n, a, v) in m[4] if 0 <= int(a) < (1 << 64)] for m in mats],
			cfg=cfg, target_rpy=target_rpy,
			seeds=[m[5] for m in mats],
		)
		trained = []
		for (controller, ts) in results:
			stats = {
				"iter_fitness":             list(ts.iter_fitness),
				"iter_mean_err_deg":        list(ts.iter_mean_err_deg),
				"iter_stable_rate":         list(ts.iter_stable_rate),
				"iter_tilt_deg":            list(ts.iter_tilt_deg),
				"iter_n_trained":           list(ts.iter_n_trained),
				"iter_cells_written":       list(ts.iter_cells_written),
				"iter_mean_episode_reward": list(ts.iter_mean_episode_reward),
				"iter_motor_jerk_mean":      list(ts.iter_motor_jerk_mean),
				"iter_mono_violations":      list(ts.iter_mono_violations),
				"train_steps":              int(ts.train_steps),
			}
			trained.append((controller, stats))
		return trained

	def _train_genome_rust(self, spec, state_conns, output_conns, init_s, init_o, seed):
		"""Rust dagger_train_inplace fast-path. Returns (WnnController, stats_dict)
		matching the Python reward_gated_train return shape."""
		import ram_accelerator as ra
		# Map the Python RewardGatedConfig → RewardGatedConfigPacked. String
		# enums become integers (0=improvement/pid, 1=quantile/student).
		rg = self.rg_config
		cfg = ra.RewardGatedConfigPacked(
			num_rounds=rg.num_rounds,
			episodes_per_round=rg.episodes_per_round,
			steps_per_episode=rg.steps_per_episode,
			bptt_window=rg.bptt_window,
			topk_per_neuron=rg.topk_per_neuron,
			protect_learned=rg.protect_learned,
			gate_mode=0 if rg.gate_mode == "improvement" else 1,
			gate_use_best=rg.gate_use_best,
			gate_window=rg.gate_window,
			gate_quantile=rg.gate_quantile,
			gate_running=rg.gate_running,
			target_source=0 if rg.target_source == "pid" else 1,
			keep_best_checkpoint=rg.keep_best_checkpoint,
			explore_eps=rg.explore_eps,
			explore_scale=rg.explore_scale,
			curriculum=rg.curriculum,
			easy_tilt_deg=rg.easy_tilt_deg,
			full_tilt_deg=rg.full_tilt_deg,
			dt=rg.episode_config.dt,
			max_initial_yaw_rad=rg.episode_config.max_initial_yaw_rad,
			max_initial_body_rate=rg.episode_config.max_initial_body_rate,
			max_initial_yaw_rate=rg.episode_config.max_initial_yaw_rate,
			eval_episodes=rg.eval_episodes,
		)
		controller = ra.WnnController(
			num_motors=spec.num_motors, levels_per_motor=spec.levels_per_motor,
			bits_per_feature=spec.bits_per_feature, input_window_k=spec.input_window_k,
			state_neurons=spec.state_neurons,
			state_bits_per_neuron=spec.state_bits_per_neuron,
			output_bits_per_neuron=spec.output_bits_per_neuron,
			thresholds=self.thresholds,
			state_connections=state_conns, output_connections=output_conns,
			delta_control=spec.delta_control, delta_max=spec.delta_max,
			delta_leak=spec.delta_leak,
		)
		for (n, addr, v) in (init_s or []):
			controller.write_state_cell(int(n), int(addr), int(v))
		for (n, addr, v) in (init_o or []):
			controller.write_output_cell(int(n), int(addr), int(v))
		target_rpy = list(rg.target_rpy) if rg.target_rpy is not None else [0.0, 0.0, 0.0]
		ts = ra.dagger_train_inplace(controller, cfg, target_rpy, int(seed))
		# Re-pack stats to match Python reward_gated_train's dict shape (the
		# fields ControllerEvaluator + downstream actually read).
		stats = {
			"iter_fitness":             list(ts.iter_fitness),
			"iter_mean_err_deg":        list(ts.iter_mean_err_deg),
			"iter_stable_rate":         list(ts.iter_stable_rate),
			"iter_tilt_deg":            list(ts.iter_tilt_deg),
			"iter_n_trained":           list(ts.iter_n_trained),
			"iter_cells_written":       list(ts.iter_cells_written),
			"iter_mean_episode_reward": list(ts.iter_mean_episode_reward),
				"iter_motor_jerk_mean":      list(ts.iter_motor_jerk_mean),
				"iter_mono_violations":      list(ts.iter_mono_violations),
			"train_steps":              int(ts.train_steps),
		}
		return controller, stats

	def score_population(self, controllers: list) -> list[tuple[float, dict]]:
		"""Closed-loop score each trained controller on the evaluator's fixed
		episode set (fresh recurrent state per episode, comparable to PID).

		Every controller has IDENTICAL shape (same state_neurons/bits/levels —
		they differ only in connectivity + cells), so the GPU path steps all
		(controllers × episodes) rollouts in ONE uniform Metal kernel. The
		closed-loop eval (forward rollout, no solver) is GPU-friendly; the inner
		training (branchy QSR beam-search) stays on CPU. GPU↔CPU parity is
		verified (tests/test_controller_gpu_parity.py) — the GA fitness is the
		same whichever path runs. Falls back to CPU if Metal is unavailable.
		"""
		if self.max_eval_workers_gpu and controllers:
			gpu = self._score_population_gpu(controllers)
			if gpu is not None:
				return gpu
		from .dagger import eval_closed_loop_reset
		out = []
		for c in controllers:
			c.reset()
			fit, m = eval_closed_loop_reset(
				make_wnn_action_fn(c), c.reset,
				self.episode_config, self.num_eval, self._active_score_seed,
			)
			out.append((fit, m))
		return out

	def _score_population_gpu(self, controllers: list):
		"""GPU-batched closed-loop scoring. Samples the SAME per-episode ICs as
		the CPU eval_closed_loop_reset plan (default_rng(seed) → per-episode
		sub-RNG → _sample_initial_state), so results are interchangeable with the
		CPU path. Returns the same list[(mean_reward, metrics)] or None on failure.
		"""
		try:
			from ram_accelerator import score_controllers_metal
		except Exception:
			return None
		from .training import _sample_initial_state
		ec = self.episode_config
		rng = np.random.default_rng(self._active_score_seed)
		q0: list[float] = []
		omega0: list[float] = []
		for _ in range(self.num_eval):
			ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
			q, om = _sample_initial_state(
				ep_rng, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
				ec.max_initial_body_rate, ec.max_initial_yaw_rate,
			)
			q0 += [float(x) for x in q]
			omega0 += [float(x) for x in om]
		try:
			agg = score_controllers_metal(
				controllers, q0, omega0, self.num_eval, ec.steps_per_episode)
		except Exception:
			return None
		out = []
		for (mean_reward, mean_err_rad, stable_rate) in agg:
			out.append((float(mean_reward), {
				"mean_reward": float(mean_reward),
				"mean_attitude_error_rad": float(mean_err_rad),
				"mean_attitude_error_deg": math.degrees(mean_err_rad),
				"stable_rate": float(stable_rate),
			}))
		return out

	def _advance_fold(self) -> int:
		"""Rotate to the next fold's episode-pool seed for K-fold scoring.

		Called at the start of every evaluate_batch / score_genomes invocation.
		With K=1 (default) this no-ops: _active_score_seed stays at self.seed,
		legacy behavior preserved. With K>1, _active_score_seed cycles through
		_fold_seeds — all genomes in this batch share the SAME pool (fair
		within-gen ranking); next batch rotates to the next pool (prevents
		single-pool overfit across gens).

		Returns the active fold index (0..K-1) for logging.
		"""
		if self.num_eval_folds <= 1:
			self._active_score_seed = self.seed
			return 0
		fold_idx = self._fold_counter % self.num_eval_folds
		self._active_score_seed = self._fold_seeds[fold_idx]
		self._fold_counter += 1
		return fold_idx

	def _evaluate_core(self, genomes: list, *, write_back: bool = False,
	                   return_stats: bool = False, seed_offset: int = 0,
	                   generation=None) -> list:
		"""Unified train+score core behind BOTH controller eval entry points.

		Each genome trains by ACCUMULATING across K=num_eval_folds folds into ONE
		canonical controller — fold k+1 warm-starts from fold k's exported cells, so
		writes compound (RAM evidence accumulation, no weight-averaging problem) — then
		the final controller is closed-loop scored once. The "two arms" differ only by
		two booleans; the train/score/cancel-guard machinery is shared (no duplicate
		path):
		  * evaluate_batch          → write_back=False, return_stats=False
		  * evaluate_for_adaptation → write_back=?,     return_stats=True

		Per-fold training uses the Rust-batched trainer (ONE call across all genomes,
		~5x over per-genome), so the Lamarckian path gets the same speedup as the GA
		path; falls back to the per-genome Python core on Rust error. K=num_eval_folds
		is the project-wide "kfold always 5" for controllers (folds are random
		episode-pool seeds → accumulating is "more rollouts", NOT a CV leak;
		generalization is judged by the held-out --report-seed). Subsumes the old
		`fitness_seeds` averaging knob.

		Returns list[Metrics] (return_stats=False) or list[(Metrics, AdaptationStats)]
		(return_stats=True). ONE cancel-guard: PROPER cancel → sentinels and SKIP
		write-back (stamping untrained cells overflows the next warm-start); SPURIOUS →
		reset + retry up to _CANCEL_RETRIES, then raise loudly."""
		from wnn.ram.metrics import Metrics
		from .recurrent_genome import MemoryPayload
		from wnn.control import cancel_state
		self._ensure_ga_ready()
		self._advance_fold()
		try:
			import ram_accelerator
		except Exception:
			ram_accelerator = None

		N = len(genomes)
		K = self.num_eval_folds
		shape_keys = [self._shape_key(g) for g in genomes]
		base_seeds = [self.seed * 100 + seed_offset + gi * K for gi in range(N)]

		_CANCEL_RETRIES = 3
		_cancel_attempt = 0
		while True:
			# Fold 0 inits from genome.cells (Lamarckian warm-start) or empty.
			cur_inits = []
			for g in genomes:
				cells = getattr(g, "cells", None)
				cur_inits.append(cells.to_triples() if cells is not None else (None, None))
			controllers = None
			last_stats = [None] * N
			for k in range(K):
				fold_tasks = [(gi, base_seeds[gi] + k) for gi in range(N)]
				trained_k = None
				if _rust_dagger_enabled() and N >= 1:
					try:
						trained_k = self._train_genomes_rust_batched(
							genomes, fold_tasks, init_override=cur_inits)
					except Exception as e:
						if not getattr(self, "_rust_dagger_batch_warned", False):
							import sys
							print(f"[ControllerEvaluator] ⚠️ batched Rust DAGGER FELL BACK to the "
							      f"per-genome Python path (accumulate): {e}",
							      file=sys.stderr, flush=True)
							self._rust_dagger_batch_warned = True
						trained_k = None
				if trained_k is None:
					trained_k = []
					for gi in range(N):
						spec, sc, oc = self._materialize(genomes[gi])
						is_, io_ = cur_inits[gi]
						trained_k.append(self._train_core(spec, sc, oc, is_, io_, fold_tasks[gi][1]))
				controllers = [c for (c, _s) in trained_k]
				last_stats = [s for (_c, s) in trained_k]
				if k < K - 1:
					# Chain: next fold warm-starts from this fold's accumulated cells.
					cur_inits = [c.export_cells() for c in controllers]

			scored = self._score_grouped(controllers, shape_keys)

			try:
				_cancelled = bool(ram_accelerator.is_cancelled()) if ram_accelerator else False
			except Exception:
				_cancelled = False
			if not _cancelled:
				break
			proper = cancel_state.sigterm_received()
			gp = f"gen {generation}" if generation is not None else "gen ?"
			print(f"[ControllerEvaluator] {gp} _evaluate_core [CANCEL-GUARD] cancel flag SET "
			      f"during eval of {N} genomes — "
			      f"{'PROPER (signum=' + str(cancel_state.last_signum()) + ')' if proper else 'SPURIOUS'}. "
			      f"write_back={write_back} → skipping write-back; "
			      f"{'returning sentinels' if proper else 'reset+retry'}.", flush=True)
			if proper:
				sentinel = Metrics(ce=float('inf'), acc=0.0, fitness=float('-inf'),
				                   mean_attitude_error_deg=180.0)
				if return_stats:
					return [(sentinel, AdaptationStats(reward=float('-inf'), stable_rate=0.0,
					         state_cell_counts=[], output_cell_counts=[])) for _ in genomes]
				return [sentinel for _ in genomes]
			_cancel_attempt += 1
			try:
				if ram_accelerator:
					ram_accelerator.reset_cancel_flag()
			except Exception as _e:
				print(f"[ControllerEvaluator] [CANCEL-GUARD] reset_cancel_flag failed: {_e}", flush=True)
			if _cancel_attempt >= _CANCEL_RETRIES:
				raise RuntimeError(
					f"[ControllerEvaluator] _evaluate_core cancelled {_CANCEL_RETRIES}x consecutively "
					f"(SPURIOUS — flag re-set each time) — refusing to return UNTRAINED controllers.")

		# Build per-genome results from the FINAL accumulated controller.
		out = []
		for gi, g in enumerate(genomes):
			reward, m = scored[gi]
			stable = float(m.get("stable_rate", 0.0))
			err = float(m.get("mean_attitude_error_deg", 0.0))
			st = last_stats[gi]
			# Phase 5c: stamp the splitting trainer's GA-handshake pressure onto the
			# genome so its offspring's mutation consumes it (grow state on
			# saturation, route connections to wished bits). Eval metadata; only the
			# arch genomes carry a `pressure` field.
			if isinstance(st, dict) and hasattr(g, "pressure"):
				g.pressure = (int(st.get("split_saturation", 0)),
				              tuple(st.get("split_wish_bits", ()) or ()))
			jl = st.get("iter_motor_jerk_mean") if isinstance(st, dict) else None
			ml = st.get("iter_mono_violations") if isinstance(st, dict) else None
			metrics = Metrics(
				ce=-float(reward), acc=stable, fitness=float(reward),
				mean_attitude_error_deg=err,
				motor_jerk_mean=(float(jl[-1]) if jl else None),
				mono_violations_total=(float(ml[-1]) if ml else None),
			)
			if write_back or return_stats:
				spec = self._materialize(g)[0]
				s_counts, o_counts, s_cells, o_cells = self._cell_stats(controllers[gi], spec)
				if write_back and hasattr(g, "cells"):
					g.cells = MemoryPayload(
						[(n, a) for (n, a, _v) in s_cells], [(n, a) for (n, a, _v) in o_cells],
						[v for (_n, _a, v) in s_cells], [v for (_n, _a, v) in o_cells])
				if return_stats:
					out.append((metrics, AdaptationStats(
						reward=float(reward), stable_rate=stable,
						state_cell_counts=s_counts, output_cell_counts=o_counts)))
					continue
			out.append(metrics)
		return out

	def evaluate_batch(self, genomes: list, *, generation: Optional[int] = None,
	                   total_generations: Optional[int] = None,
	                   min_accuracy: Optional[float] = None) -> list:
		"""Train + closed-loop-score a batch → list[Metrics]. Thin wrapper over
		_evaluate_core (write_back=False, return_stats=False): each genome trains by
		accumulating K=num_eval_folds folds into one controller, then is scored.
		(total_generations / min_accuracy are accepted for interface compatibility —
		the GA viability check applies min_accuracy itself.)"""
		return self._evaluate_core(genomes, write_back=False, return_stats=False,
		                           generation=generation)

	def score_genomes(self, genomes: list, **_kw) -> list:
		"""Paradigm-B / MEMORY-dimension scorer: build each controller directly
		from the genome's OWN cells (NO training) and closed-loop score it. The
		cells ARE the genome here, so there's nothing to train — this is the
		batch_evaluate_fn a MEMORY-dimension strategy passes to optimize().

		Memory stage K-fold: same fold rotation as evaluate_batch. Genomes are
		cell-values overfit-prone in their own right (no arch search space to
		distract the GA), so K>1 here matters too."""
		from wnn.ram.metrics import Metrics
		from wnn.control import cancel_state
		import os
		self._ensure_ga_ready()
		self._advance_fold()
		try:
			import ram_accelerator
		except Exception:
			ram_accelerator = None
		# CANCEL GUARD (01/06/2026 — same proper-vs-spurious logic as
		# evaluate_batch). No training here (cells ARE the genome), but the GPU
		# score still polls the cancel flag, so a cancel mid-score yields a
		# degenerate batch. Proper cancel → sentinels (GA unwinds + dump/resume);
		# spurious → reset + retry up to 3×, then raise loudly.
		_CANCEL_RETRIES = 3
		_cancel_attempt = 0
		while True:
			controllers = [build_controller(controller_genome_from_arch(g, self.spec, self.thresholds))
			               for g in genomes]
			scored = self._score_grouped(controllers, [self._shape_key(g) for g in genomes])
			try:
				_cancelled = bool(ram_accelerator.is_cancelled()) if ram_accelerator else False
			except Exception:
				_cancelled = False
			if not _cancelled:
				break
			proper = cancel_state.sigterm_received()
			print(
				f"[ControllerEvaluator] score_genomes [CANCEL-GUARD] cancel flag SET during "
				f"score of {len(genomes)} genomes — "
				f"{'PROPER (signum=' + str(cancel_state.last_signum()) + ')' if proper else 'SPURIOUS'}.",
				flush=True,
			)
			if proper:
				return [
					Metrics(ce=float("inf"), acc=0.0, fitness=float("-inf"),
					        mean_attitude_error_deg=180.0)
					for _ in genomes
				]
			_cancel_attempt += 1
			try:
				if ram_accelerator:
					ram_accelerator.reset_cancel_flag()
			except Exception as _e:
				print(f"[ControllerEvaluator] [CANCEL-GUARD] reset_cancel_flag failed: {_e}", flush=True)
			if _cancel_attempt >= _CANCEL_RETRIES:
				raise RuntimeError(
					f"[ControllerEvaluator] score_genomes cancelled {_CANCEL_RETRIES}x consecutively "
					f"(SPURIOUS — no SIGTERM to this process) — refusing to return a degenerate batch."
				)
		return [Metrics(ce=-float(r), acc=float(m.get("stable_rate", 0.0)), fitness=float(r),
		                mean_attitude_error_deg=float(m.get("mean_attitude_error_deg", 0.0)))
		        for (r, m) in scored]

	def _score_grouped(self, controllers: list, shape_keys: list) -> list:
		"""Score controllers in shape-uniform groups, reassembled in input order.
		Each group goes through score_population (GPU-batched, uniform shape)."""
		from collections import defaultdict
		groups: dict = defaultdict(list)
		for i, key in enumerate(shape_keys):
			groups[key].append(i)
		scored: list = [None] * len(controllers)
		for key, idxs in groups.items():
			sub = self.score_population([controllers[i] for i in idxs])
			for j, i in enumerate(idxs):
				scored[i] = sub[j]
		return scored

	def evaluate_single(self, genome) -> float:
		"""Single-genome fitness (CE = -reward, lower=better). Fallback path."""
		return self.evaluate_batch([genome])[0].ce

	# ------------------------------------------------------------------
	# Lamarckian / adaptive-eval interface (Phase B step 4b). Trains + scores
	# AND surfaces per-neuron fill stats from the trained controller (export_cells)
	# so ControllerAdaptationStrategy can drive stats-guided genesis. Optionally
	# stamps the trained cells back into genome.cells (Lamarckian write-back) —
	# the acquired memory is then inherited (and remapped by 4a under arch genesis).
	# ------------------------------------------------------------------

	def _cell_stats(self, controller, spec) -> tuple:
		"""Per-neuron distinct-address counts (the controller-side 'fill' signal)
		+ the raw cell triples, from the trained controller's exported memory."""
		s_cells, o_cells = controller.export_cells()
		s_counts = [0] * spec.state_neurons
		for (n, _a, _v) in s_cells:
			if 0 <= n < len(s_counts):
				s_counts[n] += 1
		o_counts = [0] * (spec.num_motors * spec.levels_per_motor)
		for (n, _a, _v) in o_cells:
			if 0 <= n < len(o_counts):
				o_counts[n] += 1
		return s_counts, o_counts, s_cells, o_cells

	def evaluate_for_adaptation(self, genomes: list, *, write_back: bool = False,
	                            seed_offset: int = 0) -> list:
		"""Train (Lamarckian warm-start) + score + optionally write the accumulated
		cells back, returning per genome (Metrics, AdaptationStats). Thin wrapper over
		_evaluate_core (return_stats=True). See _evaluate_core for the K-fold accumulate
		semantics; with write_back the final accumulated state is stamped into
		genome.cells so offspring inherit it."""
		return self._evaluate_core(genomes, write_back=write_back, return_stats=True,
		                           seed_offset=seed_offset)

__all__ = [
	"ControllerSpec",
	"ControllerGenome",
	"AdaptationStats",
	"ControllerEvaluator",
	"fit_thresholds_from_pid_rollouts",
	"random_connectivity",
	"build_controller",
	"NUM_FEATURES",
]
