"""RecurrentArchGenome — variable-shape genome for a two-layer recurrent RAM FSM.

WHAT IT IS (and is NOT)
-----------------------
The evolvable architecture of a coherent recurrent RAM controller: a STATE
layer (recurrent) + an OUTPUT layer (Mealy), each neuron observing the FULL
recurrent state plus a sampled slice of its layer's input region. It is the
variable-shape successor to `FiniteStateGenome` (genome.py): where that class
fixes the shape in a shared `ControllerSpec` and evolves only connectivity,
this class evolves all three dimensions — neurons, bits, connections — so the
shared GA/TS/Lamarckian framework can drive it exactly as it drives the IDS
`ClusterGenome`.

It is deliberately DOMAIN-FREE: it speaks `state_neurons`, `output_neurons`,
and input-space SIZES — never "motor", "PWM", "sensor". A drone adapter
(evaluator.py) maps `num_motors × levels_per_motor` ↔ `output_neurons`,
`K·F·b` ↔ `state_input_space`, and so on, then builds the Rust `WnnController`.
That keeps this genome reusable by any two-layer recurrent RAM arch and gives
the step-5 WNN-type factory something genuinely generic to unify.

THE INVARIANT, MADE STRUCTURAL
------------------------------
For the network to behave as ONE finite-state machine (not N disjoint
mini-automata), every state and output neuron must observe the full recurrent
state — the forced full-state prefix (see project_controller_state). Rather
than STORE that prefix and self-check it (as FiniteStateGenome does), this
genome stores ONLY each neuron's sampled-input suffix and RECONSTRUCTS the
prefix canonically at `to_connections()` time:

    state neuron i : [ range(state_input_space, +prefix) | state_sampled[i] ]
    output neuron j: [ range(output_input_space, +prefix) | output_sampled[j] ]
    prefix = prefix_factor * state_neurons          (QSR → 2 bits per state neuron)

The prefix carries zero free information (it is fully determined by
`state_neurons`), so by never storing it we make corruption impossible: EVERY
genome any operator ever produces is a valid coherent FSM by construction.

DESIGN DECISIONS (locked 2026-05-25)
------------------------------------
- Uniform bits per layer: all state suffixes share one width, all output
  suffixes share another — matches the scalar `state_bits_per_neuron` /
  `output_bits_per_neuron` of the Rust `WnnController` and `gpu_dims()`. Zero
  accelerator changes.
- Additive neurogenesis: adding a state neuron grows the prefix (+prefix_factor
  bits) for every neuron in BOTH layers; the sampled-suffix width is preserved,
  so adding capacity never destroys learned wiring.
- Two neuron axes: STATE neurogenesis (memory capacity) reshapes the prefix
  globally; OUTPUT neurogenesis (resolution) adds/removes whole blocks in units
  of `output_quantum` and leaves the prefix and the entire state layer untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Reuse the IDS phase taxonomy verbatim so the step-5 factory unifies cleanly.
from wnn.ram.strategies.connectivity.adaptive_cluster import PhaseType


# ---- fixed structural constants (never mutated) -----------------------------

@dataclass(frozen=True)
class RecurrentArchShape:
	"""The structural constants of a recurrent two-layer RAM arch.

	Fixed for a given problem and NEVER touched by any genome operator — only
	the neuron counts and sampled suffixes evolve. A drone adapter fills these
	from a ControllerSpec; another domain would fill them differently.
	"""
	prefix_factor: int        # bits emitted per state neuron (QSR → 2)
	state_input_space: int    # size of the region the STATE layer samples from
	output_input_space: int   # size of the region the OUTPUT layer samples from
	output_quantum: int       # output-neurogenesis granularity (drone: num_motors)


# ---- mutation bounds (mirror of AdaptiveClusterConfig) ----------------------

@dataclass
class RecurrentArchConfig:
	"""Bounds + step sizes for the three mutation dimensions. Generic: output
	bounds are expressed in NEURONS (multiples of output_quantum), not levels."""
	min_state_neurons: int = 2
	max_state_neurons: int = 16
	min_output_neurons: int = 4      # must be a multiple of output_quantum
	max_output_neurons: int = 1024   # must be a multiple of output_quantum
	min_suffix: int = 1              # ≥1 sampled input bit floor (a neuron must see SOMETHING)
	max_suffix: int = 64
	state_neuron_delta: int = 1      # ± neurons per NEURONS-phase step
	output_block_delta: int = 1      # ± quanta (levels) per NEURONS-phase step
	suffix_delta: int = 2            # ± sampled bits per BITS-phase step


# ---- small sampling helpers --------------------------------------------------

def _sample_distinct(space: int, k: int, rng: np.random.Generator,
                     exclude: set[int] | None = None) -> list[int]:
	"""k distinct indices in [0, space), avoiding `exclude`. Clamped to what fits."""
	pool = range(space) if not exclude else [x for x in range(space) if x not in exclude]
	k = min(k, len(pool))
	if k <= 0:
		return []
	idx = rng.choice(len(pool), size=k, replace=False)
	return [int(pool[i]) if exclude else int(i) for i in idx]


def _resample_in_place(suffix: list[int], space: int, rng: np.random.Generator, rate: float) -> None:
	"""Per-entry resample of a sampled suffix, avoiding duplicates within it."""
	used = set(suffix)
	for j in range(len(suffix)):
		if rng.random() < rate:
			used.discard(suffix[j])
			for _ in range(8):  # a few tries to find an unused bit
				cand = int(rng.integers(0, space))
				if cand not in used:
					suffix[j] = cand
					used.add(cand)
					break
			else:
				used.add(suffix[j])


def _resize_suffix(suffix: list[int], space: int, target: int, rng: np.random.Generator) -> None:
	"""Grow (append distinct new bits) or shrink (drop from the tail) IN PLACE."""
	target = min(target, space)
	if target < len(suffix):
		del suffix[target:]
	elif target > len(suffix):
		new = _sample_distinct(space, target - len(suffix), rng, exclude=set(suffix))
		suffix.extend(new)


# ---- the genome --------------------------------------------------------------

@dataclass
class RecurrentArchGenome:
	"""Variable-shape genome = neuron counts + per-neuron sampled suffixes.

	The forced full-state prefix is implicit (reconstructed at materialize), so
	the FSM-coherence invariant cannot be violated by any operator.
	"""
	shape: RecurrentArchShape
	state_neurons: int
	output_neurons: int
	state_sampled: list[list[int]] = field(default_factory=list)   # len = state_neurons
	output_sampled: list[list[int]] = field(default_factory=list)  # len = output_neurons

	# ---- derived quantities -------------------------------------------------

	@property
	def forced_prefix(self) -> int:
		"""Forced full-state prefix length = bits every neuron spends observing state."""
		return self.shape.prefix_factor * self.state_neurons

	@property
	def state_suffix_width(self) -> int:
		return len(self.state_sampled[0]) if self.state_sampled else 0

	@property
	def output_suffix_width(self) -> int:
		return len(self.output_sampled[0]) if self.output_sampled else 0

	@property
	def state_bits_per_neuron(self) -> int:
		return self.forced_prefix + self.state_suffix_width

	@property
	def output_bits_per_neuron(self) -> int:
		return self.forced_prefix + self.output_suffix_width

	# ---- construction -------------------------------------------------------

	@classmethod
	def random(cls, shape: RecurrentArchShape, state_neurons: int, output_neurons: int,
	           state_suffix: int, output_suffix: int, rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Fresh random genome of a given shape (prefix forced; suffixes sampled)."""
		ss = [_sample_distinct(shape.state_input_space, state_suffix, rng) for _ in range(state_neurons)]
		os = [_sample_distinct(shape.output_input_space, output_suffix, rng) for _ in range(output_neurons)]
		return cls(shape=shape, state_neurons=state_neurons, output_neurons=output_neurons,
		           state_sampled=ss, output_sampled=os)

	def clone(self) -> "RecurrentArchGenome":
		return RecurrentArchGenome(
			shape=self.shape,
			state_neurons=self.state_neurons,
			output_neurons=self.output_neurons,
			state_sampled=[list(s) for s in self.state_sampled],
			output_sampled=[list(s) for s in self.output_sampled],
		)

	# ---- materialization (the canonical prefix is rebuilt HERE) -------------

	def to_connections(self) -> tuple[list[int], list[int]]:
		"""Flatten to (state_connections, output_connections) the Rust controller
		expects, prepending the canonical full-state prefix to every neuron."""
		p = self.forced_prefix
		state_prefix = list(range(self.shape.state_input_space, self.shape.state_input_space + p))
		out_prefix = list(range(self.shape.output_input_space, self.shape.output_input_space + p))
		sc: list[int] = []
		for suffix in self.state_sampled:
			sc.extend(state_prefix)
			sc.extend(suffix)
		oc: list[int] = []
		for suffix in self.output_sampled:
			oc.extend(out_prefix)
			oc.extend(suffix)
		return sc, oc

	def fingerprint(self) -> tuple:
		"""Hashable structural identity, used by the GA loop for elite dedup."""
		return (
			self.state_neurons, self.output_neurons,
			self.state_suffix_width, self.output_suffix_width,
			tuple(tuple(s) for s in self.state_sampled),
			tuple(tuple(s) for s in self.output_sampled),
		)

	# ---- phase-aware mutation (the GA/TS/Lamarckian entry point) -------------

	def mutate(self, phase: PhaseType, rate: float, config: RecurrentArchConfig,
	           rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Return a mutated copy. The phase selects which dimension moves; the
		forced prefix is regenerated by to_connections() and so is never at risk."""
		if phase == PhaseType.CLUSTER:
			g = self._mutate_neurons(rate, config, rng)
			g = g._mutate_bits(rate, config, rng)
			return g._mutate_connections(rate, config, rng)
		if phase == PhaseType.NEURONS:
			return self._mutate_neurons(rate, config, rng)
		if phase == PhaseType.BITS:
			return self._mutate_bits(rate, config, rng)
		return self._mutate_connections(rate, config, rng)  # PhaseType.CONNECTIONS

	def _mutate_neurons(self, rate: float, config: RecurrentArchConfig,
	                    rng: np.random.Generator) -> "RecurrentArchGenome":
		"""State + output neurogenesis. Survivors keep their suffixes verbatim;
		growth appends fresh tail blocks (small-neighborhood rule)."""
		g = self.clone()
		# STATE neurogenesis (memory capacity): reshapes the prefix globally.
		if rng.random() < rate and config.state_neuron_delta > 0:
			delta = int(rng.integers(-config.state_neuron_delta, config.state_neuron_delta + 1))
			target = min(config.max_state_neurons, max(config.min_state_neurons, g.state_neurons + delta))
			g._resize_state_neurons(target, rng)
		# OUTPUT neurogenesis (resolution): whole blocks, in units of output_quantum.
		q = g.shape.output_quantum
		if rng.random() < rate and config.output_block_delta > 0 and q > 0:
			delta_blocks = int(rng.integers(-config.output_block_delta, config.output_block_delta + 1))
			lo = max(1, config.min_output_neurons // q)
			hi = max(lo, config.max_output_neurons // q)
			cur_blocks = g.output_neurons // q
			target_blocks = min(hi, max(lo, cur_blocks + delta_blocks))
			g._resize_output_neurons(target_blocks * q, rng)
		return g

	def _mutate_bits(self, rate: float, config: RecurrentArchConfig,
	                 rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Synaptogenesis: grow/shrink sampled-suffix width uniformly per layer."""
		g = self.clone()
		if rng.random() < rate and config.suffix_delta > 0:
			delta = int(rng.integers(-config.suffix_delta, config.suffix_delta + 1))
			cap = min(config.max_suffix, g.shape.state_input_space)
			target = min(cap, max(config.min_suffix, g.state_suffix_width + delta))
			for suffix in g.state_sampled:
				_resize_suffix(suffix, g.shape.state_input_space, target, rng)
		if rng.random() < rate and config.suffix_delta > 0:
			delta = int(rng.integers(-config.suffix_delta, config.suffix_delta + 1))
			cap = min(config.max_suffix, g.shape.output_input_space)
			target = min(cap, max(config.min_suffix, g.output_suffix_width + delta))
			for suffix in g.output_sampled:
				_resize_suffix(suffix, g.shape.output_input_space, target, rng)
		return g

	def _mutate_connections(self, rate: float, config: RecurrentArchConfig,
	                        rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Axonogenesis: resample sampled INPUT bits only (prefix never stored)."""
		g = self.clone()
		for suffix in g.state_sampled:
			_resample_in_place(suffix, g.shape.state_input_space, rng, rate)
		for suffix in g.output_sampled:
			_resample_in_place(suffix, g.shape.output_input_space, rng, rate)
		return g

	# ---- resize primitives (in place; caller has already cloned) ------------

	def _resize_state_neurons(self, target: int, rng: np.random.Generator) -> None:
		width = self.state_suffix_width or 1
		if target < self.state_neurons:
			del self.state_sampled[target:]
		else:
			for _ in range(target - self.state_neurons):
				self.state_sampled.append(_sample_distinct(self.shape.state_input_space, width, rng))
		self.state_neurons = target

	def _resize_output_neurons(self, target: int, rng: np.random.Generator) -> None:
		width = self.output_suffix_width or 1
		if target < self.output_neurons:
			del self.output_sampled[target:]
		else:
			for _ in range(target - self.output_neurons):
				self.output_sampled.append(_sample_distinct(self.shape.output_input_space, width, rng))
		self.output_neurons = target

	# ---- crossover (handles parents of DIFFERENT shape) ---------------------

	@staticmethod
	def crossover(a: "RecurrentArchGenome", b: "RecurrentArchGenome",
	              rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Whole-block uniform crossover. The child inherits ONE parent's shape
		(counts + suffix widths); for each block it then takes the other parent's
		suffix only when shape-compatible, else keeps the shape-parent's. This
		guarantees a structurally valid child even when a and b differ in size."""
		shape_parent, other = (a, b) if rng.random() < 0.5 else (b, a)
		child = shape_parent.clone()
		_mix_blocks(child.state_sampled, other.state_sampled, rng)
		_mix_blocks(child.output_sampled, other.output_sampled, rng)
		return child

	# ---- validity self-check (used by tests) --------------------------------

	def assert_valid(self) -> None:
		"""Raise AssertionError if any structural invariant is violated."""
		sh = self.shape
		assert self.state_neurons >= 1, "need ≥1 state neuron"
		assert len(self.state_sampled) == self.state_neurons, "state count mismatch"
		assert len(self.output_sampled) == self.output_neurons, "output count mismatch"
		assert sh.output_quantum > 0 and self.output_neurons % sh.output_quantum == 0, \
			"output_neurons must be a multiple of output_quantum"
		sw, ow = self.state_suffix_width, self.output_suffix_width
		assert sw >= 1 and ow >= 1, "≥1 sampled bit floor violated"
		for s in self.state_sampled:
			assert len(s) == sw, "non-uniform state suffix width"
			assert len(set(s)) == len(s), "duplicate state sampled bit"
			assert all(0 <= x < sh.state_input_space for x in s), "state sampled bit out of range"
		for o in self.output_sampled:
			assert len(o) == ow, "non-uniform output suffix width"
			assert len(set(o)) == len(o), "duplicate output sampled bit"
			assert all(0 <= x < sh.output_input_space for x in o), "output sampled bit out of range"


def _mix_blocks(into: list[list[int]], other: list[list[int]], rng: np.random.Generator) -> None:
	"""For each block in `into`, with p=0.5 take `other`'s block of the same
	index — but only if it exists and has matching width (keeps suffix uniform)."""
	if not into:
		return
	width = len(into[0])
	for i in range(len(into)):
		if i < len(other) and len(other[i]) == width and rng.random() < 0.5:
			into[i] = list(other[i])


__all__ = ["RecurrentArchGenome", "RecurrentArchShape", "RecurrentArchConfig"]
