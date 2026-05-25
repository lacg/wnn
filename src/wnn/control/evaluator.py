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
	state_bits = 2 * n_state
	sensor_window = spec.input_window_k * NUM_FEATURES * spec.bits_per_feature
	sensor_frame = NUM_FEATURES * spec.bits_per_feature

	n_state_sampled = spec.state_bits_per_neuron - state_bits
	n_out_sampled = spec.output_bits_per_neuron - state_bits
	if n_state_sampled < 0 or n_out_sampled < 0:
		raise ValueError(
			f"bits_per_neuron must be >= 2*state_neurons ({state_bits}) for full-state "
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
		prefix_factor=2,  # QSR state output = 2 bits per state neuron
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
	(connectivity + thresholds + optional trained cells)."""
	sc, oc = genome.to_connections()
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
	):
		self.spec = spec
		self.num_eval = num_eval_episodes
		self.num_validate = num_validate_episodes
		self.seed = seed
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

	def _ensure_ga_ready(self):
		from .reward_gated import RewardGatedConfig
		if self.thresholds is None:
			self.thresholds = fit_thresholds_from_pid_rollouts(self.spec, num_episodes=10, seed=self.seed)
		if self.rg_config is None:
			self.rg_config = RewardGatedConfig(seed=self.seed, episode_config=self.episode_config)

	def _train_genome(self, genome, seed: int):
		"""Inner-train one genome's cells (C1 or C2 per rg_config.target_source)
		with the given training seed. Returns (WnnController, stats)."""
		from .reward_gated import reward_gated_train
		import copy
		rg = copy.copy(self.rg_config)
		rg.seed = seed
		rg.progress = False
		return reward_gated_train(
			self.spec, self.thresholds,
			genome.state_connections, genome.output_connections, rg,
		)

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
				self.episode_config, self.num_eval, self.seed,
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
		rng = np.random.default_rng(self.seed)
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

	def evaluate_batch(self, genomes: list, *, generation: Optional[int] = None,
	                   total_generations: Optional[int] = None,
	                   min_accuracy: Optional[float] = None) -> list:
		"""Train + closed-loop-score a batch of genomes → list[Metrics].

		Multi-seed (A): each genome is trained+scored over K=fitness_seeds
		independent seeds; the genome's fitness is the MEAN closed-loop reward
		(de-noises the chaotic inner loop so the GA climbs signal, not seed luck).
		All K×pop controllers are scored in ONE GPU batch.

		Fitness mapping (loop convention: lower CE = better):
		  ce      = -mean_reward   (so the GA minimises → maximises reward)
		  acc     = mean_stable_rate
		  fitness = mean_reward    (raw, for FitnessCalculatorController + reports)
		"""
		from wnn.ram.metrics import Metrics
		self._ensure_ga_ready()
		K = max(1, self.fitness_seeds)

		# 1. Inner-train each genome over K seeds (gi-major, k inner). Distinct
		#    seed per (genome, k) so the K trains are independent draws.
		tasks = [(gi, self.seed * 100 + gi * K + k)
		         for gi in range(len(genomes)) for k in range(K)]
		if self.max_train_workers > 1 and len(tasks) > 1:
			from concurrent.futures import ThreadPoolExecutor
			with ThreadPoolExecutor(max_workers=min(self.max_train_workers, len(tasks))) as pool:
				trained = list(pool.map(lambda t: self._train_genome(genomes[t[0]], t[1]), tasks))
		else:
			trained = [self._train_genome(genomes[gi], seed) for (gi, seed) in tasks]
		controllers = [c for (c, _st) in trained]

		# 2. Closed-loop score all K×pop controllers in one GPU batch.
		scored = self.score_population(controllers)

		# 3. Aggregate per genome: mean reward / mean stable_rate over its K seeds.
		results = []
		for gi in range(len(genomes)):
			block = scored[gi * K:(gi + 1) * K]
			rewards = [r for (r, _m) in block]
			stables = [m.get("stable_rate", 0.0) for (_r, m) in block]
			mean_reward = float(np.mean(rewards))
			results.append(Metrics(
				ce=-mean_reward,
				acc=float(np.mean(stables)),
				fitness=mean_reward,
			))
		return results

	def evaluate_single(self, genome) -> float:
		"""Single-genome fitness (CE = -reward, lower=better). Fallback path."""
		return self.evaluate_batch([genome])[0].ce


__all__ = [
	"ControllerSpec",
	"ControllerGenome",
	"ControllerEvaluator",
	"fit_thresholds_from_pid_rollouts",
	"random_connectivity",
	"build_controller",
	"NUM_FEATURES",
]
