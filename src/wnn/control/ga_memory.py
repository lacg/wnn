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

from wnn.control._accel import WnnController
from wnn.control import _accel as ra   # memory_* cell operators (Rust, counter RNG)

from wnn.ram.strategies.connectivity.generic_strategies import GenericGAStrategy
from wnn.ram.fitness import FitnessCalculatorType

from .evaluator import ControllerSpec
from .ga_strategy import default_controller_ga_config
from .training import EpisodeConfig, _sample_initial_state


# ----------------------------------------------------------------------------
# Address-universe recording
# ----------------------------------------------------------------------------

def _recorder_plant_kwargs(ec, num_episodes: int, seed: int) -> dict:
	"""The aircraft the reference rollout flies, as recorder kwargs.

	DEFAULT-INERT BY DESIGN. With translation off this returns {} — the recorder
	keeps flying the legacy synthetic plant and every universe recorded before
	13/08/2026 reproduces bit-for-bit, so no banked attitude result moves.

	With translation ON (scope C stage 1) it passes the airframe AND the
	per-episode vertical draws, because without them the three vertical features
	never move: the recorded universe would cover one degenerate slice and the
	MEMORY phase could not reach the cells that decide stage-1 behaviour.

	KNOWN GAP, deliberately left alone (13/08/2026): attitude-only runs still
	record on the synthetic plant even when --airframe is set. That is
	sound-but-incomplete (a narrower universe, not a wrong one) and changing it
	is a LINEAGE BREAK, so it is a decision to take with the calib-airframe A/B,
	not a silent side effect of this fix.
	"""
	if ec is None or not getattr(ec, "translation", False):
		return {}
	from .training import sample_vertical_ics_flat
	af = getattr(ec, "airframe", None)
	z0, vz0, coll, mass = sample_vertical_ics_flat(seed, num_episodes, ec)
	af_mass = float(af.mass) if af is not None else 1.0
	plant = dict(
		s1_target_altitude=float(getattr(ec, "target_altitude", 0.0)),
		s1_init_z=[float(v) for v in z0],
		s1_init_vz=[float(v) for v in vz0],
		# mass_scale × the airframe's nominal mass — the PLANT draw, never a feature.
		s1_mass=[af_mass * float(m) for m in mass],
		s1_collective_frac=[float(c) for c in coll],
	)
	if af is not None:
		# REUSE EpisodeConfig.airframe_kwargs — the single source of these values.
		# Hand-rolling them is what produced `AttributeError: 'Airframe' object has
		# no attribute 'dt'` (dt lives on the EpisodeConfig, not the Airframe) and
		# cost a 3h run. The recorder takes the PLANT fields only, not the
		# af_pid_* cascade, so select the keys rather than splat the dict.
		af_kw = ec.airframe_kwargs()
		plant.update({k: af_kw[k] for k in (
			"af_dt", "af_arm_length", "af_k_thrust", "af_k_drag",
			"af_inertia", "af_gravity") if k in af_kw})
	return plant


def record_address_universe(
	spec: ControllerSpec,
	thresholds: list[float],
	state_connections: list[int],
	output_connections: list[int],
	num_episodes: int = 12,
	steps: int = 1500,
	tilt_deg: float = 15.0,
	seed: int = 0,
	geometry=None,        # Optional[GeometryConfig] — N-rotor TRUE table (sim side)
	alloc=None,           # Optional[AllocResidualConfig] — baseline driver gains
	episode_config=None,  # Optional[EpisodeConfig] — airframe + stage-1 vertical draws
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
	"""Record the (neuron, address) cells the controller visits along
	reference-driven rollouts. Returns (state_universe, output_universe),
	each sorted-unique.

	The reference driver holds the sim in the good operating region while the
	controller runs forward at each visited state (advancing its recurrent
	state); we read off the addresses via the last_*_addresses getters.
	Quad (default): PID drives sim.step. Overactuated (geometry set): the
	allocator-LQR baseline (AllocLqrRs on the NOMINAL rows) drives step_n on
	the TRUE table — the operating region the residual-composed policy lives in.
	"""
	c = WnnController(
		num_motors=spec.num_motors, levels_per_motor=spec.levels_per_motor,
		bits_per_feature=spec.bits_per_feature, input_window_k=spec.input_window_k,
		state_neurons=spec.state_neurons, state_bits_per_neuron=spec.state_bits_per_neuron,
		output_bits_per_neuron=spec.output_bits_per_neuron, thresholds=thresholds,
		state_connections=state_connections, output_connections=output_connections,
		delta_control=spec.delta_control, delta_max=spec.delta_max, delta_leak=spec.delta_leak, delta_gamma=getattr(spec, 'delta_gamma', 1.0),
		obs_tilt_p=spec.obs_tilt_p, obs_tilt_i=spec.obs_tilt_i, obs_peraxis_p=spec.obs_peraxis_p, obs_peraxis_i=spec.obs_peraxis_i, obs_peraxis_yaw=spec.obs_peraxis_yaw, obs_pwm=spec.obs_pwm, obs_yaw_err=spec.obs_yaw_err, obs_yaw_err_i=spec.obs_yaw_err_i,
		# SCOPE C STAGE 1 (13/08/2026). Omitting these builds a 15-feature
		# controller while stage-1 thresholds carry 18, so the constructor refuses
		# it with "thresholds length 144 != 120" — which killed every stage-1 run
		# at the MEMORY phase after ~3h of grid+neurons. EVERY obs_* flag the spec
		# owns must be forwarded here, or this controller is not the one flying.
		obs_collective_cmd=getattr(spec, 'obs_collective_cmd', False),
		obs_alt_err=getattr(spec, 'obs_alt_err', False),
		obs_vz=getattr(spec, 'obs_vz', False),
		dhat_b=(list(spec.dhat_b) if spec.dhat_b is not None else None), dhat_l_gain=spec.dhat_l_gain, dhat_ff=getattr(spec, 'dhat_ff', False), dhat_ff_clamp=getattr(spec, 'dhat_ff_clamp', 0.30), dt=spec.dt, integral_leak=spec.integral_leak, integral_scale=spec.integral_scale, decouple_outputs=spec.decouple_outputs,
		action_repeat=spec.action_repeat,
		memory_mode=spec.memory_mode_int(),
	)
	# NOTE: the sim/PID/allocator reference loop that used to live here was DEAD
	# CODE — it was built, never called, and left behind by the Rust port below.
	# It was also actively misleading: it constructed a bare AttitudeSim(), so it
	# read as though the recorder flew the legacy plant on purpose. Removed
	# 13/08/2026; ra.record_address_universe is the only rollout.
	rng = np.random.default_rng(seed)
	tilt = math.radians(tilt_deg)
	init_q, init_om = [], []
	for _ in range(num_episodes):
		ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		q0, om0 = _sample_initial_state(ep_rng, tilt, tilt, 0.5, 0.3)
		init_q.append([float(x) for x in q0])
		init_om.append([float(x) for x in om0])

	# Both reference drivers now run in Rust (record_ops::Driver): PID for the
	# quad path, allocator-LQR on the TRUE rotor table for the overactuated one.
	# Only the episode ICs are drawn in Python and injected — the established
	# parity convention — so this is a bit-exact port of the loop.
	plant = _recorder_plant_kwargs(episode_config, num_episodes, seed)
	# REFUSE a silently-degenerate universe. If the controller carries stage-1
	# features but no vertical plant reached us, the rollout would fly a
	# NON-TRANSLATING aircraft: the three vertical features would sit frozen and
	# the recorded universe would cover one degenerate slice — running clean and
	# meaning nothing, which is far worse than crashing. This guard exists
	# because a pop-6 smoke passed on 13/08/2026 for exactly that reason (the
	# caller handed episode_config=None, so the plant kwargs were {}), and the
	# green result was mistaken for proof the fix worked.
	if any(getattr(spec, n, False) for n in
	       ("obs_collective_cmd", "obs_alt_err", "obs_vz")) and "s1_init_z" not in plant:
		raise ValueError(
			"record_address_universe: the spec has stage-1 vertical features on, but no "
			"translating episode_config reached the recorder (episode_config="
			f"{'None' if episode_config is None else 'translation=' + str(getattr(episode_config, 'translation', False))}"
			"). The recorded universe would be degenerate — pass the EpisodeConfig the "
			"run is flying.")
	if geometry is None:
		s_uni, o_uni = ra.record_address_universe(
			c, init_q, init_om, [0.0, 0.0, 0.0], int(steps), **plant)
	else:
		nominal = (alloc.nominal_rows if alloc is not None and alloc.nominal_rows is not None
		           else geometry.rows)
		s_uni, o_uni = ra.record_address_universe(
			c, init_q, init_om, [0.0, 0.0, 0.0], int(steps),
			geometry_rows=[list(r) for r in geometry.rows],
			nominal_rows=[list(r) for r in nominal],
			rotor_asym=([float(x) for x in geometry.rotor_asym]
			            if geometry.rotor_asym is not None else None),
			q_att=(alloc.q_att if alloc else 12.0), q_rate=(alloc.q_rate if alloc else 1.0),
			r_ctrl=(alloc.r_ctrl if alloc else 1.0), tau_max=(alloc.tau_max if alloc else 0.144),
			f_hover=(alloc.f_hover if alloc else None),
			pinv_lambda=(alloc.pinv_lambda if alloc else 1e-6), **plant)
	return ([(int(n), int(a)) for (n, a) in s_uni],
	        [(int(n), int(a)) for (n, a) in o_uni])



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
		# Mode-native cell draws (ABI 12): QUAD/QSR 0..3 (4-state graded — QSR is a
		# stochastic QUAD read); TERNARY/BINARY/PLN {FALSE=0, TRUE=1} — 2 is the EMPTY
		# sentinel, 3 invalid outside the QUAD family (PLN shares TERNARY's cells).
		hi = 4 if spec.memory_mode_int() in (1, 2, 4) else 2
		_seed = int(rng.integers(0, 1 << 63))
		return cls(
			spec=spec, state_connections=state_conns, output_connections=output_conns,
			state_universe=state_universe, output_universe=output_universe,
			# Rust (counter RNG). The old form drew a compact numpy array and then
			# BOXED every element into a Python int — 10^5-10^6 values per genome,
			# i.e. both a per-cell Python loop and the 156 B/cell representation,
			# created at genesis.
			state_values=list(ra.memory_random_values(
				len(state_universe), hi, _seed, 0, 0, ra.LAYER_STATE)),
			output_values=list(ra.memory_random_values(
				len(output_universe), hi, _seed, 0, 0, ra.LAYER_OUTPUT)),
		)

	def clone(self) -> "MemoryGenome":
		g = MemoryGenome(self.spec, self.state_connections, self.output_connections,
		                 self.state_universe, self.output_universe,
		                 list(self.state_values), list(self.output_values))
		return g

	def mutate(self, rng: np.random.Generator, rate: float) -> "MemoryGenome":
		"""Nudge ~rate fraction of cells one step: QUAD/QSR ±1 (clamped 0..3);
		TERNARY/BINARY/PLN flip FALSE↔TRUE (the 2-state nudge analog).

		Runs in Rust (ram_core counter RNG) — the per-cell Python loop this
		replaced was ~10^9 interpreter iterations per production run. One numpy
		draw still seeds the call, so the caller's rng chain keeps determinism;
		the per-cell draws are counter-based and therefore order-independent."""
		g = self.clone()
		quad = self.spec.memory_mode_int() in (1, 2, 4)
		seed = int(rng.integers(0, 1 << 63))
		g.state_values = list(ra.memory_mutate_values(
			g.state_values, quad, rate, seed, 0, 0, ra.LAYER_STATE))
		g.output_values = list(ra.memory_mutate_values(
			g.output_values, quad, rate, seed, 0, 0, ra.LAYER_OUTPUT))
		return g

	@staticmethod
	def crossover(a: "MemoryGenome", b: "MemoryGenome", rng: np.random.Generator) -> "MemoryGenome":
		"""Uniform per-cell crossover (universe + connectivity shared)."""
		seed = int(rng.integers(0, 1 << 63))
		sv = list(ra.memory_crossover_values(
			a.state_values, b.state_values, seed, 0, 0, ra.LAYER_STATE))
		ov = list(ra.memory_crossover_values(
			a.output_values, b.output_values, seed, 0, 0, ra.LAYER_OUTPUT))
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
		delta_control=spec.delta_control, delta_max=spec.delta_max, delta_leak=spec.delta_leak, delta_gamma=getattr(spec, 'delta_gamma', 1.0),
		obs_tilt_p=spec.obs_tilt_p, obs_tilt_i=spec.obs_tilt_i, obs_peraxis_p=spec.obs_peraxis_p, obs_peraxis_i=spec.obs_peraxis_i, obs_peraxis_yaw=spec.obs_peraxis_yaw, obs_pwm=spec.obs_pwm, obs_yaw_err=spec.obs_yaw_err, obs_yaw_err_i=spec.obs_yaw_err_i,
		# SCOPE C STAGE 1 (13/08/2026). Omitting these builds a 15-feature
		# controller while stage-1 thresholds carry 18, so the constructor refuses
		# it with "thresholds length 144 != 120" — which killed every stage-1 run
		# at the MEMORY phase after ~3h of grid+neurons. EVERY obs_* flag the spec
		# owns must be forwarded here, or this controller is not the one flying.
		obs_collective_cmd=getattr(spec, 'obs_collective_cmd', False),
		obs_alt_err=getattr(spec, 'obs_alt_err', False),
		obs_vz=getattr(spec, 'obs_vz', False),
		dhat_b=(list(spec.dhat_b) if spec.dhat_b is not None else None), dhat_l_gain=spec.dhat_l_gain, dhat_ff=getattr(spec, 'dhat_ff', False), dhat_ff_clamp=getattr(spec, 'dhat_ff_clamp', 0.30), dt=spec.dt, integral_leak=spec.integral_leak, integral_scale=spec.integral_scale, decouple_outputs=spec.decouple_outputs,
		action_repeat=spec.action_repeat,
		memory_mode=spec.memory_mode_int(),
	)
	# ONE FFI call instead of one per cell (see WnnController::load_cells).
	c.load_cells(
		[(n, a, int(v)) for (n, a), v in zip(genome.state_universe, genome.state_values)],
		[(n, a, int(v)) for (n, a), v in zip(genome.output_universe, genome.output_values)],
	)
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
		from wnn.ram.metrics import ControllerMetrics as Metrics
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
			out.append(Metrics(reward=float(mean_reward), stable_rate=float(stable_rate),
			                   fitness=float(mean_reward),
			                   mean_attitude_error_deg=math.degrees(float(mean_err_rad)),
			                   motor_jerk_mean=float(jerk),
			                   mono_violations_total=float(mono),
			                   mean_steady_error_deg=math.degrees(float(steady_rad))))
		return out

	def evaluate_single(self, genome) -> float:
		return -self.evaluate_batch([genome])[0].reward


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
