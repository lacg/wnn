"""ControllerAdaptationStrategy — Lamarckian (stats-guided genesis) for the
controller, the controller-side analog of the IDS AdaptationStrategy.

Unlike GA/TS (which mutate RANDOMLY), genesis edits the architecture
DETERMINISTICALLY from per-genome statistics surfaced by the rollout
(ControllerEvaluator.evaluate_for_adaptation): per-neuron distinct-address
counts ("fill") + closed-loop reward. Two modes are wired here (the third,
axonogenesis-by-entropy, needs per-input-bit stats from new Rust instrumentation
and is deferred — Phase B step 4b-5):

  synaptogenesis — size each layer's sampled-suffix width to the observed address
                   diversity: target bits ≈ log2(mean distinct addresses) + margin
                   (clamped). Grow when neurons exercise many addresses (need
                   resolution), shrink when few (bits are wasted).
  neurogenesis   — prune DEAD state neurons (zero cells → constant prefix bit-pair
                   = dead weight); add a state neuron when reward is poor and no
                   neuron is dead (more memory capacity).

"Lamarckian" in two senses: (1) genesis is guided by acquired training stats, and
(2) with write_back the trained cells are stamped into genome.cells and inherited
— then remapped through any genesis arch change by the 4a remap machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .evaluator import ControllerSpec, arch_shape_from_spec, NUM_FEATURES
from .arch_strategy import default_controller_arch_config, _random_arch_genome
from .recurrent_genome import RecurrentArchGenome, RecurrentArchShape, RecurrentArchConfig


def _binary_entropy(p: float) -> float:
	"""Shannon entropy (bits) of a Bernoulli(p) input bit. 0 for constant bits."""
	if p <= 0.0 or p >= 1.0:
		return 0.0
	return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def record_input_entropy(spec: ControllerSpec, thresholds: list, state_connections: list,
                         output_connections: list, num_episodes: int = 6, steps: int = 800,
                         tilt_deg: float = 15.0, seed: int = 0) -> tuple:
	"""Per-input-bit activation entropy over a PID-driven reference rollout.

	Returns (state_entropy, output_entropy): entropy of each SAMPLEABLE sensor bit
	(the regions the genome's connections index into — sensor_window / sensor_frame).
	Entropy depends on the input distribution (sim + thermometer encoding), not the
	specific genome, so one rollout serves all genomes' axonogenesis.

	Requires the WnnController.last_*_layer_input getters (Phase B step 4b-5) — run
	`maturin develop --release` to build them (held until the 46M flow finishes)."""
	from ram_accelerator import AttitudeSim, WnnController
	import numpy as np
	from .pid import AttitudePID, AttitudePIDConfig
	from .training import _sample_initial_state

	c = WnnController(
		num_motors=spec.num_motors, levels_per_motor=spec.levels_per_motor,
		bits_per_feature=spec.bits_per_feature, input_window_k=spec.input_window_k,
		state_neurons=spec.state_neurons, state_bits_per_neuron=spec.state_bits_per_neuron,
		output_bits_per_neuron=spec.output_bits_per_neuron, thresholds=thresholds,
		state_connections=state_connections, output_connections=output_connections,
		delta_control=spec.delta_control, delta_max=spec.delta_max, delta_leak=spec.delta_leak,
		obs_tilt_p=spec.obs_tilt_p, obs_tilt_i=spec.obs_tilt_i, obs_peraxis_p=spec.obs_peraxis_p, obs_peraxis_i=spec.obs_peraxis_i, obs_pwm=spec.obs_pwm, integral_leak=spec.integral_leak, integral_scale=spec.integral_scale,
	)
	# axonogenesis getters (last_state_layer_input / last_output_layer_input) are live
	# in the compiled accelerator (controller.rs:626/632) — verified 05/06/2026.
	sensor_window = spec.input_window_k * NUM_FEATURES * spec.bits_per_feature
	sensor_frame = NUM_FEATURES * spec.bits_per_feature
	s_act = [0] * sensor_window
	o_act = [0] * sensor_frame
	nsteps = 0
	pid = AttitudePID(AttitudePIDConfig())
	sim = AttitudeSim()
	rng = np.random.default_rng(seed)
	tilt = math.radians(tilt_deg)
	target = (0.0, 0.0, 0.0)
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
			c.step(list(gyro), list(accel), list(target))
			si = c.last_state_layer_input()
			oi = c.last_output_layer_input()
			for i in range(sensor_window):
				if si[i]:
					s_act[i] += 1
			for i in range(sensor_frame):
				if oi[i]:
					o_act[i] += 1
			nsteps += 1
			sim.step(list(pid.step(q, gyro, target)))
	if nsteps == 0:
		return [0.0] * sensor_window, [0.0] * sensor_frame
	return ([_binary_entropy(a / nsteps) for a in s_act],
	        [_binary_entropy(a / nsteps) for a in o_act])


@dataclass
class AdaptationConfig:
	"""Genesis schedule + thresholds (controller analog of IDS AdaptationConfig)."""
	genesis_mode: str = "synaptogenesis"   # "synaptogenesis"|"neurogenesis"|"axonogenesis"|"memory"|"all"
	iterations: int = 20
	population_size: int = 12
	patience: int = 5
	min_improvement: float = 1e-3
	warmup_iterations: int = 2             # explore before adapting
	elite_fraction: float = 0.2
	write_back: bool = True                # Lamarckian cell inheritance
	# synaptogenesis: target_bits ≈ log2(mean fill) + margin, moved ≤ bit_step/iter
	bits_margin: int = 2
	bit_step: int = 2
	# neurogenesis: reward below this (more negative = worse) ⇒ "struggling" ⇒ add
	add_reward_baseline: float = -0.5
	# axonogenesis (mirrors IDS axon_*): rewire a sampled connection whose input-bit
	# entropy < median × ratio, toward an unused bit with entropy > old × improvement.
	axon_entropy_ratio: float = 0.3
	axon_improvement_factor: float = 1.5
	axon_rewire_count: int = 2          # max connections rewired per neuron per pass
	axon_record_episodes: int = 6       # reference rollout for the entropy profile
	axon_record_steps: int = 800


@dataclass
class AdaptationResult:
	best_genome: Optional[RecurrentArchGenome]
	final_fitness: float
	history: list = field(default_factory=list)   # per-iteration best fitness


class ControllerAdaptationStrategy:
	"""Stats-guided genesis loop over RecurrentArchGenome."""

	def __init__(
		self,
		spec: ControllerSpec,
		genesis_mode: str = "synaptogenesis",
		arch_config: Optional[RecurrentArchConfig] = None,
		adapt_config: Optional[AdaptationConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional[Any] = None,
	):
		self._spec = spec
		self._shape: RecurrentArchShape = arch_shape_from_spec(spec)
		self._arch_config = arch_config or default_controller_arch_config(spec)
		self._cfg = adapt_config or AdaptationConfig(genesis_mode=genesis_mode)
		self._cfg.genesis_mode = genesis_mode  # explicit arg wins
		# MEMORY mode's "acquired trait" IS the cells → inheritance is mandatory.
		if genesis_mode == "memory":
			self._cfg.write_back = True
		self._evaluator = batch_evaluator
		self._log = logger or (lambda m: None)
		self._np_rng = np.random.default_rng(0 if seed is None else seed)
		self._seed = seed
		# Seed dims (pinned starting point for the population). Forced prefix =
		# prefix_factor·state_neurons (1-bit since 08/06/2026; the literal 2· was a
		# stale 2-bit assumption that collapsed seed suffixes to ~0).
		_pf = arch_shape_from_spec(spec).prefix_factor
		self._seed_dims = (spec.state_neurons, spec.num_motors * spec.levels_per_motor,
		                   spec.state_bits_per_neuron - _pf * spec.state_neurons,
		                   spec.output_bits_per_neuron - _pf * spec.state_neurons)
		self._mem_seed: Optional[RecurrentArchGenome] = None  # fixed net for memory mode
		self._state_entropy: Optional[list] = None  # axonogenesis input-bit entropy profile
		self._output_entropy: Optional[list] = None

	@property
	def name(self) -> str:
		return f"ControllerLamarckian-{self._cfg.genesis_mode}"

	# ---- public API ---------------------------------------------------------

	def random_genome(self) -> RecurrentArchGenome:
		from wnn.ram.strategies.optimization_dimension import OptimizationDimension
		if self._cfg.genesis_mode == "memory":
			# Lamarckian MEMORY: FIXED architecture + connectivity — the only
			# evolvable trait is the CELLS, acquired by training (write_back) and
			# carried forward (warm-start). Distinct from GA-MEMORY (random cell
			# nudges, no training) and from arch genesis (reshapes the network).
			if self._mem_seed is None:
				self._mem_seed = _random_arch_genome(
					self._shape, OptimizationDimension.CONNECTIONS, self._arch_config,
					self._seed_dims, self._np_rng)
			return self._mem_seed.clone()
		return _random_arch_genome(self._shape, OptimizationDimension.CLUSTER,
		                           self._arch_config, self._seed_dims, self._np_rng)

	def optimize(self, *, initial_population: Optional[list] = None,
	             initial_genome: Optional[RecurrentArchGenome] = None, **_kw) -> AdaptationResult:
		cfg = self._cfg
		pop = list(initial_population) if initial_population else \
			[self.random_genome() for _ in range(cfg.population_size)]
		if initial_genome is not None:
			pop[0] = initial_genome.clone()

		best_g: Optional[RecurrentArchGenome] = None
		best_fit = -math.inf
		stale = 0
		history: list = []

		for it in range(cfg.iterations):
			results = self._evaluator.evaluate_for_adaptation(pop, write_back=cfg.write_back)
			ranked = sorted(zip(pop, results), key=lambda pr: pr[1][0].fitness, reverse=True)
			top_g, (top_m, _top_st) = ranked[0]
			history.append(float(top_m.fitness))
			if top_m.fitness > best_fit + cfg.min_improvement:
				best_g, best_fit, stale = top_g.clone(), float(top_m.fitness), 0
			else:
				stale += 1
			self._log(f"[{self.name}] iter {it+1}/{cfg.iterations}: best={best_fit:.4f} "
			          f"(top={top_m.fitness:.4f}, stale={stale}/{cfg.patience})")
			if stale >= cfg.patience:
				break
			# Genesis: keep elites verbatim, adapt the rest from their stats.
			if it >= cfg.warmup_iterations:
				n_elite = max(1, int(cfg.elite_fraction * len(ranked)))
				newpop = [g.clone() for g, _ in ranked[:n_elite]]
				for g, (_m, st) in ranked[n_elite:]:
					newpop.append(self._genesis(g, st))
				pop = newpop
		return AdaptationResult(best_genome=best_g, final_fitness=best_fit, history=history)

	# ---- genesis operators --------------------------------------------------

	def _genesis(self, genome: RecurrentArchGenome, stats) -> RecurrentArchGenome:
		"""Apply the configured genesis mode(s) to a single genome from its stats."""
		g = genome.clone()
		mode = self._cfg.genesis_mode
		if mode in ("synaptogenesis", "all"):
			self._synaptogenesis(g, stats)
		if mode in ("neurogenesis", "all"):
			self._neurogenesis(g, stats)
		if mode in ("axonogenesis", "all"):
			self._axonogenesis(g, stats)
		g.assert_valid()
		return g

	def _synaptogenesis(self, g: RecurrentArchGenome, stats) -> None:
		"""Size each layer's suffix to its observed address diversity:
		target_bits ≈ log2(mean distinct addresses) + margin, moved ≤ bit_step."""
		cfg, cfgA = self._cfg, self._arch_config
		prefix = g.forced_prefix
		cap_s = min(cfgA.max_suffix, g.shape.state_input_space)
		cap_o = min(cfgA.max_suffix, g.shape.output_input_space)
		g.set_state_suffix(self._suffix_target(stats.state_cell_counts, prefix,
		                                        g.state_suffix_width, cfgA.min_suffix, cap_s), self._np_rng)
		g.set_output_suffix(self._suffix_target(stats.output_cell_counts, prefix,
		                                         g.output_suffix_width, cfgA.min_suffix, cap_o), self._np_rng)

	def _suffix_target(self, counts: list, prefix: int, cur: int, lo: int, hi: int) -> int:
		mean_fill = (sum(counts) / len(counts)) if counts else 0.0
		want_bits = math.ceil(math.log2(max(mean_fill, 2.0))) + self._cfg.bits_margin
		want_suffix = max(lo, min(hi, want_bits - prefix))
		# Move toward the target by at most bit_step (small-neighborhood rule).
		step = self._cfg.bit_step
		if want_suffix > cur:
			return min(want_suffix, cur + step)
		if want_suffix < cur:
			return max(want_suffix, cur - step)
		return cur

	def _ensure_entropy(self, g: RecurrentArchGenome) -> None:
		"""Record the per-input-bit entropy profile ONCE (genome-independent — it's
		the sim+encoder input distribution). Reused for every genome's rewire."""
		if self._state_entropy is not None:
			return
		from .evaluator import spec_from_arch
		sc, oc = g.to_connections()
		spec = spec_from_arch(g, self._spec)
		ev = self._evaluator
		th = None
		if ev is not None:
			ev._ensure_ga_ready()
			th = ev.thresholds
		self._state_entropy, self._output_entropy = record_input_entropy(
			spec, th, sc, oc, num_episodes=self._cfg.axon_record_episodes,
			steps=self._cfg.axon_record_steps, seed=0 if self._seed is None else self._seed)

	def _axonogenesis(self, g: RecurrentArchGenome, stats) -> None:
		"""Rewire each neuron's lowest-entropy sampled connections (near-constant
		input bits = little information) toward the highest-entropy UNUSED bits."""
		self._ensure_entropy(g)
		s_changes = self._rewire_layer(g.state_sampled, self._state_entropy)
		o_changes = self._rewire_layer(g.output_sampled, self._output_entropy)
		if s_changes or o_changes:
			g.rewire_suffix(s_changes, o_changes)

	def _rewire_layer(self, sampled: list, entropy: list) -> dict:
		"""Per-neuron: replace up to axon_rewire_count connections whose target-bit
		entropy < median × ratio, with unused bits of entropy > old × improvement."""
		import statistics
		if not sampled or not entropy or len(sampled[0]) == 0:
			return {}
		cfg = self._cfg
		thresh = statistics.median(entropy) * cfg.axon_entropy_ratio
		changes: dict = {}
		for n, suf in enumerate(sampled):
			weak = sorted((i for i in range(len(suf)) if entropy[suf[i]] < thresh),
			              key=lambda i: entropy[suf[i]])[:cfg.axon_rewire_count]
			if not weak:
				continue
			used = set(suf)
			cands = sorted((b for b in range(len(entropy)) if b not in used),
			               key=lambda b: entropy[b], reverse=True)
			new_suf, ci, changed = list(suf), 0, False
			for i in weak:
				while ci < len(cands) and entropy[cands[ci]] <= entropy[suf[i]] * cfg.axon_improvement_factor:
					ci += 1
				if ci >= len(cands):
					break
				new_suf[i] = cands[ci]
				used.add(cands[ci])
				ci += 1
				changed = True
			if changed:
				changes[n] = new_suf
		return changes

	def _neurogenesis(self, g: RecurrentArchGenome, stats) -> None:
		"""Surgically prune the first DEAD state neuron (zero cells) when one
		exists; else add a state neuron when reward is poor. One edit per iteration
		(small-neighborhood rule). remove_state_neuron excises the specific dead
		neuron's prefix window from every address — not a blunt tail collapse."""
		cfgA = self._arch_config
		dead = [i for i, c in enumerate(stats.state_cell_counts) if c == 0]
		if dead and g.state_neurons - 1 >= cfgA.min_state_neurons:
			g.remove_state_neuron(dead[0], self._np_rng)
		elif not dead and stats.reward < self._cfg.add_reward_baseline \
				and g.state_neurons < cfgA.max_state_neurons:
			g.set_state_neurons(g.state_neurons + 1, self._np_rng)


__all__ = ["ControllerAdaptationStrategy", "AdaptationConfig", "AdaptationResult"]
