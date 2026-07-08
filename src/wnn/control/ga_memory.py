"""GA-Memory (paradigm B) — neuroevolve the QSR cell values directly.

The profile (project_controller_state) showed C1/C2's cost is ~98%
`bptt_train_window` — per-genome EDRA TRAINING. GA-Memory removes training
entirely: a genome IS the cell values over a fixed address universe, so
evaluation is just "write cells → closed-loop GPU rollout score". No solver, no
Python per-step loop — the whole population scores in one Metal batch, at
IDS-like speed. This is the only paradigm that scales to 50×250.

  genome  = QSR values (0..3) at a fixed set of (neuron, address) cells
  cells   = build a WnnController, write those cells (rest stay EMPTY=hover)
  fitness = closed-loop reward (GPU), evolved directly — no imitation

ADDRESS UNIVERSE
----------------
We can't evolve 2^24 cells/neuron. We evolve only the cells the controller
actually VISITS along reference rollouts (recorded via WnnController's
last_state/last_output_addresses getters). Unvisited addresses read EMPTY (2 →
QSR 0.75 → hover), so an all-default genome holds hover (safe baseline).

LIMITATION (v1) + the v2 lever: the universe is recorded with EMPTY cells, so
the recurrent-state bits sit at their EMPTY baseline → the state-layer universe
covers mostly the constant-state slice, and v1 effectively neuroevolves a
near-FEEDFORWARD output mapping (frame→PWM). To exploit RECURRENCE (the integral
term), the universe must cover varied state bits — grow it DAGGER-style from
elite closed-loop rollouts (record addresses the evolving policy visits, union
in, re-evolve). v1 ships the scaffold + the base universe; growth is the v2 knob.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from wnn.control._accel import AttitudeSim, WnnController

from wnn.ram.strategies.connectivity.generic_strategies import GenericGAStrategy
from wnn.ram.fitness import FitnessCalculatorType

from .evaluator import ControllerSpec
from .ga_strategy import default_controller_ga_config
from .pid import AttitudePID, AttitudePIDConfig
from .training import EpisodeConfig, _sample_initial_state


# ----------------------------------------------------------------------------
# Address-universe recording
# ----------------------------------------------------------------------------

def record_address_universe(
	spec: ControllerSpec,
	thresholds: list[float],
	state_connections: list[int],
	output_connections: list[int],
	num_episodes: int = 12,
	steps: int = 1500,
	tilt_deg: float = 15.0,
	seed: int = 0,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
	"""Record the (neuron, address) cells the controller visits along PID-driven
	rollouts. Returns (state_universe, output_universe), each sorted-unique.

	PID drives the sim (good operating region); the controller runs forward at
	each visited state (advancing its recurrent state) and we read off the
	addresses its neurons accessed via the last_*_addresses getters.
	"""
	c = WnnController(
		num_motors=spec.num_motors, levels_per_motor=spec.levels_per_motor,
		bits_per_feature=spec.bits_per_feature, input_window_k=spec.input_window_k,
		state_neurons=spec.state_neurons, state_bits_per_neuron=spec.state_bits_per_neuron,
		output_bits_per_neuron=spec.output_bits_per_neuron, thresholds=thresholds,
		state_connections=state_connections, output_connections=output_connections,
		delta_control=spec.delta_control, delta_max=spec.delta_max, delta_leak=spec.delta_leak,
		obs_tilt_p=spec.obs_tilt_p, obs_tilt_i=spec.obs_tilt_i, obs_peraxis_p=spec.obs_peraxis_p, obs_peraxis_i=spec.obs_peraxis_i, obs_peraxis_yaw=spec.obs_peraxis_yaw, obs_pwm=spec.obs_pwm, obs_yaw_err=spec.obs_yaw_err, obs_yaw_err_i=spec.obs_yaw_err_i, dt=spec.dt, integral_leak=spec.integral_leak, integral_scale=spec.integral_scale, decouple_outputs=spec.decouple_outputs,
		action_repeat=spec.action_repeat,
	)
	pid = AttitudePID(AttitudePIDConfig())
	sim = AttitudeSim()
	rng = np.random.default_rng(seed)
	tilt = math.radians(tilt_deg)
	target = (0.0, 0.0, 0.0)
	state_set: set[tuple[int, int]] = set()
	out_set: set[tuple[int, int]] = set()

	for _ in range(num_episodes):
		ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		q0, om0 = _sample_initial_state(ep_rng, tilt, tilt, 0.5, 0.3)
		sim.reset(q=list(q0), omega=list(om0))
		pid.reset()
		c.reset()
		for _t in range(steps):
			if sim.is_unstable():
				break
			gyro, accel = sim.read_imu()
			q = sim.quaternion
			c.step(list(gyro), list(accel), list(target))  # advances + caches inputs
			for na in c.last_state_addresses():
				state_set.add((int(na[0]), int(na[1])))
			for na in c.last_output_addresses():
				out_set.add((int(na[0]), int(na[1])))
			pwm = pid.step(q, gyro, target)   # PID drives the sim
			sim.step(list(pwm))

	return sorted(state_set), sorted(out_set)


# ----------------------------------------------------------------------------
# MemoryGenome — the evolvable cell values
# ----------------------------------------------------------------------------

@dataclass
class MemoryGenome:
	"""QSR cell values over the fixed address universe. Connectivity + universe
	are shared/fixed; only state_values / output_values evolve."""
	spec: ControllerSpec
	state_connections: list[int]
	output_connections: list[int]
	state_universe: list[tuple[int, int]]
	output_universe: list[tuple[int, int]]
	state_values: list[int]    # QSR 0..3, aligned to state_universe
	output_values: list[int]   # QSR 0..3, aligned to output_universe

	@classmethod
	def random(cls, spec, state_conns, output_conns, state_universe, output_universe,
	           rng: np.random.Generator) -> "MemoryGenome":
		return cls(
			spec=spec, state_connections=state_conns, output_connections=output_conns,
			state_universe=state_universe, output_universe=output_universe,
			state_values=[int(v) for v in rng.integers(0, 4, size=len(state_universe))],
			output_values=[int(v) for v in rng.integers(0, 4, size=len(output_universe))],
		)

	def clone(self) -> "MemoryGenome":
		g = MemoryGenome(self.spec, self.state_connections, self.output_connections,
		                 self.state_universe, self.output_universe,
		                 list(self.state_values), list(self.output_values))
		return g

	def mutate(self, rng: np.random.Generator, rate: float) -> "MemoryGenome":
		"""Nudge ~rate fraction of cells one QSR step (±1, clamped 0..3)."""
		g = self.clone()
		for vals in (g.state_values, g.output_values):
			for i in range(len(vals)):
				if rng.random() < rate:
					vals[i] = int(np.clip(vals[i] + (1 if rng.random() < 0.5 else -1), 0, 3))
		return g

	@staticmethod
	def crossover(a: "MemoryGenome", b: "MemoryGenome", rng: np.random.Generator) -> "MemoryGenome":
		"""Uniform per-cell crossover (universe + connectivity shared)."""
		sv = [a.state_values[i] if rng.random() < 0.5 else b.state_values[i]
		      for i in range(len(a.state_values))]
		ov = [a.output_values[i] if rng.random() < 0.5 else b.output_values[i]
		      for i in range(len(a.output_values))]
		return MemoryGenome(a.spec, a.state_connections, a.output_connections,
		                    a.state_universe, a.output_universe, sv, ov)


def build_controller_from_memory(genome: MemoryGenome, thresholds: list[float]) -> WnnController:
	"""Instantiate a controller and write the genome's universe cells (rest EMPTY)."""
	spec = genome.spec
	c = WnnController(
		num_motors=spec.num_motors, levels_per_motor=spec.levels_per_motor,
		bits_per_feature=spec.bits_per_feature, input_window_k=spec.input_window_k,
		state_neurons=spec.state_neurons, state_bits_per_neuron=spec.state_bits_per_neuron,
		output_bits_per_neuron=spec.output_bits_per_neuron, thresholds=thresholds,
		state_connections=genome.state_connections, output_connections=genome.output_connections,
		delta_control=spec.delta_control, delta_max=spec.delta_max, delta_leak=spec.delta_leak,
		obs_tilt_p=spec.obs_tilt_p, obs_tilt_i=spec.obs_tilt_i, obs_peraxis_p=spec.obs_peraxis_p, obs_peraxis_i=spec.obs_peraxis_i, obs_peraxis_yaw=spec.obs_peraxis_yaw, obs_pwm=spec.obs_pwm, obs_yaw_err=spec.obs_yaw_err, obs_yaw_err_i=spec.obs_yaw_err_i, dt=spec.dt, integral_leak=spec.integral_leak, integral_scale=spec.integral_scale, decouple_outputs=spec.decouple_outputs,
		action_repeat=spec.action_repeat,
	)
	for (n, addr), v in zip(genome.state_universe, genome.state_values):
		c.write_state_cell(n, addr, int(v))
	for (n, addr), v in zip(genome.output_universe, genome.output_values):
		c.write_output_cell(n, addr, int(v))
	return c


# ----------------------------------------------------------------------------
# Evaluator — build cells + GPU closed-loop score (NO training)
# ----------------------------------------------------------------------------

class ControllerMemoryEvaluator:
	"""Score MemoryGenomes by closed-loop reward — pure GPU forward rollout, no
	training. evaluate_batch builds all controllers then scores them in ONE
	score_controllers_metal call."""

	def __init__(self, spec: ControllerSpec, thresholds: list[float],
	             num_eval_episodes: int = 20, seed: int = 0,
	             episode_config: Optional[EpisodeConfig] = None):
		self.spec = spec
		self.thresholds = thresholds
		self.num_eval = num_eval_episodes
		self.seed = seed
		self.episode_config = episode_config or EpisodeConfig(
			dt=0.001, steps_per_episode=1500,
			max_initial_tilt_rad=math.radians(15.0), max_initial_yaw_rad=math.radians(15.0),
			max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)

	def _ics(self):
		from .training import sample_ics_flat
		return sample_ics_flat(self.seed, self.num_eval, self.episode_config)

	def evaluate_batch(self, genomes: list, **kwargs) -> list:
		from wnn.ram.metrics import Metrics
		from wnn.control._accel import score_controllers_metal
		controllers = [build_controller_from_memory(g, self.thresholds) for g in genomes]
		q0, omega0 = self._ics()
		agg = score_controllers_metal(controllers, q0, omega0, self.num_eval,
		                               self.episode_config.steps_per_episode)
		out = []
		# 12-metric rows (Vec<Vec<f64>>): the trailing 6 are transient-speed
		# metrics (rise/settle/ITAE) — carried but not yet part of fitness.
		for row in agg:
			(mean_reward, mean_err_rad, stable_rate, jerk, mono, steady_rad,
			 _rise, _settle_abs, _settle_rel, _itae, _iae, _ise) = row
			out.append(Metrics(ce=-float(mean_reward), acc=float(stable_rate),
			                   fitness=float(mean_reward),
			                   mean_attitude_error_deg=math.degrees(float(mean_err_rad)),
			                   motor_jerk_mean=float(jerk),
			                   mono_violations_total=float(mono),
			                   mean_steady_error_deg=math.degrees(float(steady_rad))))
		return out

	def evaluate_single(self, genome) -> float:
		return self.evaluate_batch([genome])[0].ce


# ----------------------------------------------------------------------------
# GA strategy — reuse the canonical loop, MemoryGenome ops
# ----------------------------------------------------------------------------

class ControllerMemoryGAStrategy(GenericGAStrategy):
	"""Connectivity-fixed, cell-evolving GA. Thin GenericGAStrategy subclass."""

	def __init__(self, spec, state_conns, output_conns, state_universe, output_universe,
	             ga_config=None, seed=None, logger=None, batch_evaluator=None,
	             shutdown_check=None):
		super().__init__(config=ga_config or default_controller_ga_config(),
		                 seed=seed, logger=logger)
		self._spec = spec
		self._sc = state_conns
		self._oc = output_conns
		self._su = state_universe
		self._ou = output_universe
		self._batch_evaluator = batch_evaluator
		self._cached_evaluator = None
		self._checkpoint_config = None
		self._shutdown_check = shutdown_check
		self._np_rng = np.random.default_rng(0 if seed is None else seed)

	@property
	def name(self) -> str:
		return "ControllerMemoryGA"

	def clone_genome(self, genome): return genome.clone()
	def mutate_genome(self, genome, mutation_rate): return genome.mutate(self._np_rng, mutation_rate)
	def crossover_genomes(self, p1, p2): return MemoryGenome.crossover(p1, p2, self._np_rng)

	def create_random_genome(self):
		return MemoryGenome.random(self._spec, self._sc, self._oc, self._su, self._ou, self._np_rng)


__all__ = [
	"record_address_universe", "MemoryGenome", "build_controller_from_memory",
	"ControllerMemoryEvaluator", "ControllerMemoryGAStrategy",
]
