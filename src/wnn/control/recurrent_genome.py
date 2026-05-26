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

# The shared, generic optimization-axis taxonomy (NEURONS/BITS/CONNECTIONS/
# CLUSTER/MEMORY) — see optimization_dimension.py. NEURONS/BITS/CONNECTIONS are
# architecture and handled here; MEMORY (cell contents) needs the optional cells
# payload of the unified genome (step 4) and is not yet implemented.
from wnn.ram.strategies.optimization_dimension import OptimizationDimension


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


# ---- memory payload: the optional "content" dimension ------------------------

# QSR value 2 = bits 0b10 = the EMPTY/hover decode (QSR_WEIGHTS[2] = 0.75). Used
# as the neutral branch a freshly-added state neuron's prefix pair defaults to.
NEUTRAL_PAIR = 2


@dataclass
class MemoryPayload:
	"""Evolvable QSR cell contents over a (per-genome) address universe.

	Mirrors ga_memory.MemoryGenome's universe/values split so paradigm-B's
	per-cell mutate + crossover align by index — which holds because a MEMORY
	phase freezes the architecture, giving the whole population one universe.
	Only addresses in the universe are stored; everything else is EMPTY (hover).
	"""
	state_universe: list[tuple[int, int]]   # (neuron_idx, address) keys
	output_universe: list[tuple[int, int]]
	state_values: list[int]                 # QSR 0..3, aligned to state_universe
	output_values: list[int]

	def clone(self) -> "MemoryPayload":
		return MemoryPayload(list(self.state_universe), list(self.output_universe),
		                     list(self.state_values), list(self.output_values))

	def to_triples(self) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
		"""(neuron, address, value) triples the WnnController write methods take."""
		st = [(n, a, v) for (n, a), v in zip(self.state_universe, self.state_values)]
		ot = [(n, a, v) for (n, a), v in zip(self.output_universe, self.output_values)]
		return st, ot

	def fingerprint(self) -> tuple:
		return (tuple(self.state_universe), tuple(self.state_values),
		        tuple(self.output_universe), tuple(self.output_values))


# ---- best-effort cell remap (high-fidelity policy) ---------------------------
# Address model (compute_address_sparse, MSB-first): A = P·2^w + S, prefix P in
# the HIGH bits, sampled suffix S in the LOW w bits. Every operator adds/drops at
# the tail, so changes land on known bit fields.

def _majority(vals: list[int]) -> int:
	"""Most common QSR value among colliders; ties → lower value (deterministic)."""
	from collections import Counter
	counts = Counter(vals)
	return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _remap_grow(universe, values, d):
	"""BITS grow by d LSBs: A → A·2^d + child; value REPLICATED to all 2^d children
	(behavior-preserving — the new low bits don't change the read value)."""
	if d <= 0:
		return list(universe), list(values)
	nu, nv = [], []
	for (n, a), v in zip(universe, values):
		base = a << d
		for child in range(1 << d):
			nu.append((n, base | child))
			nv.append(v)
	return nu, nv


def _remap_shrink(universe, values, d):
	"""BITS shrink by d LSBs: A → A >> d; colliders resolved by majority vote."""
	if d <= 0:
		return list(universe), list(values)
	buckets: dict[tuple[int, int], list[int]] = {}
	for (n, a), v in zip(universe, values):
		buckets.setdefault((n, a >> d), []).append(v)
	nu = list(buckets.keys())
	nv = [_majority(buckets[k]) for k in nu]
	return nu, nv


def _remap_bits(universe, values, d):
	"""Dispatch a BITS-width change of d sampled bits: grow (replicate) if d>0,
	shrink (majority collapse) if d<0."""
	return _remap_grow(universe, values, d) if d > 0 else _remap_shrink(universe, values, -d)


def _remap_prefix_grow(universe, values, k, w):
	"""STATE neurogenesis +k: prefix gains 2k mid-bits (just above the w-bit
	suffix), defaulting to the neutral QSR pair. A = P·2^w + S becomes
	P·2^(2k+w) + neutral·2^w + S — behavior preserved on the neutral branch."""
	if k <= 0:
		return list(universe), list(values)
	mask = (1 << w) - 1
	neutral = 0
	for j in range(k):  # k pairs, lowest pair at j=0
		neutral |= NEUTRAL_PAIR << (2 * j)
	nu, nv = [], []
	for (n, a), v in zip(universe, values):
		P, S = a >> w, a & mask
		nu.append((n, (P << (2 * k + w)) | (neutral << w) | S))
		nv.append(v)
	return nu, nv


def _remap_prefix_shrink(universe, values, k, w):
	"""STATE neurogenesis -k: drop the lowest 2k prefix bits; majority collapse.
	A = P_high·2^(2k+w) + pair·2^w + S  →  P_high·2^w + S."""
	if k <= 0:
		return list(universe), list(values)
	mask = (1 << w) - 1
	buckets: dict[tuple[int, int], list[int]] = {}
	for (n, a), v in zip(universe, values):
		P_high, S = a >> (2 * k + w), a & mask
		buckets.setdefault((n, (P_high << w) | S), []).append(v)
	nu = list(buckets.keys())
	nv = [_majority(buckets[k2]) for k2 in nu]
	return nu, nv


def _drop_neurons_ge(universe, values, limit):
	"""Drop cells whose neuron index ≥ limit (the neurons being removed)."""
	nu, nv = [], []
	for (n, a), v in zip(universe, values):
		if n < limit:
			nu.append((n, a))
			nv.append(v)
	return nu, nv


def _drop_changed_neurons(universe, values, changed: set[int]):
	"""CONNECTIONS remap: drop cells of neurons whose sampled suffix changed (their
	address semantics scrambled); keep the rest verbatim."""
	nu, nv = [], []
	for (n, a), v in zip(universe, values):
		if n not in changed:
			nu.append((n, a))
			nv.append(v)
	return nu, nv


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
	# Optional "content" dimension. None ⇒ paradigm A (cells trained at eval).
	# Populated ⇒ paradigm B (cells evolved) / Lamarckian write-back. Arch
	# mutations remap it best-effort (the universe is keyed by addresses, which
	# move when the architecture changes).
	cells: "MemoryPayload | None" = None

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
			cells=self.cells.clone() if self.cells is not None else None,
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
		"""Hashable structural identity, used by the GA loop for elite dedup.
		Includes the cells payload so MEMORY-mutated genomes are distinct."""
		base = (
			self.state_neurons, self.output_neurons,
			self.state_suffix_width, self.output_suffix_width,
			tuple(tuple(s) for s in self.state_sampled),
			tuple(tuple(s) for s in self.output_sampled),
		)
		return base if self.cells is None else base + (self.cells.fingerprint(),)

	# ---- phase-aware mutation (the GA/TS/Lamarckian entry point) -------------

	def mutate(self, dim: OptimizationDimension, rate: float, config: RecurrentArchConfig,
	           rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Return a mutated copy. The dimension selects which axis moves; the
		forced prefix is regenerated by to_connections() and so is never at risk."""
		if dim == OptimizationDimension.NEURONS:
			return self._mutate_neurons(rate, config, rng)
		if dim == OptimizationDimension.BITS:
			return self._mutate_bits(rate, config, rng)
		if dim == OptimizationDimension.CONNECTIONS:
			return self._mutate_connections(rate, config, rng)
		if dim == OptimizationDimension.CLUSTER:  # all ARCHITECTURE dims at once
			g = self._mutate_neurons(rate, config, rng)
			g = g._mutate_bits(rate, config, rng)
			return g._mutate_connections(rate, config, rng)
		if dim == OptimizationDimension.MEMORY:
			return self._mutate_memory(rate, rng)
		raise ValueError(f"unknown optimization dimension: {dim!r}")

	def _mutate_neurons(self, rate: float, config: RecurrentArchConfig,
	                    rng: np.random.Generator) -> "RecurrentArchGenome":
		"""State + output neurogenesis. Survivors keep their suffixes verbatim;
		growth appends fresh tail blocks (small-neighborhood rule). Cells, if
		present, are remapped: STATE neuro reshapes the prefix in BOTH layers;
		OUTPUT neuro keeps survivors verbatim and drops removed blocks."""
		g = self.clone()
		# STATE neurogenesis (memory capacity): reshapes the prefix globally.
		if rng.random() < rate and config.state_neuron_delta > 0:
			delta = int(rng.integers(-config.state_neuron_delta, config.state_neuron_delta + 1))
			target = min(config.max_state_neurons, max(config.min_state_neurons, g.state_neurons + delta))
			g.set_state_neurons(target, rng)
		# OUTPUT neurogenesis (resolution): whole blocks, in units of output_quantum.
		q = g.shape.output_quantum
		if rng.random() < rate and config.output_block_delta > 0 and q > 0:
			delta_blocks = int(rng.integers(-config.output_block_delta, config.output_block_delta + 1))
			lo = max(1, config.min_output_neurons // q)
			hi = max(lo, config.max_output_neurons // q)
			cur_blocks = g.output_neurons // q
			g.set_output_neurons(min(hi, max(lo, cur_blocks + delta_blocks)) * q, rng)
		return g

	def _mutate_bits(self, rate: float, config: RecurrentArchConfig,
	                 rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Synaptogenesis: grow/shrink sampled-suffix width uniformly per layer.
		Cells remap by replicate-on-grow / majority-collapse-on-shrink (LSBs)."""
		g = self.clone()
		if rng.random() < rate and config.suffix_delta > 0:
			delta = int(rng.integers(-config.suffix_delta, config.suffix_delta + 1))
			cap = min(config.max_suffix, g.shape.state_input_space)
			g.set_state_suffix(min(cap, max(config.min_suffix, g.state_suffix_width + delta)), rng)
		if rng.random() < rate and config.suffix_delta > 0:
			delta = int(rng.integers(-config.suffix_delta, config.suffix_delta + 1))
			cap = min(config.max_suffix, g.shape.output_input_space)
			g.set_output_suffix(min(cap, max(config.min_suffix, g.output_suffix_width + delta)), rng)
		return g

	def _mutate_connections(self, rate: float, config: RecurrentArchConfig,
	                        rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Axonogenesis: resample sampled INPUT bits only (prefix never stored).
		Cells of any neuron whose suffix changed are dropped (address scrambled)."""
		g = self.clone()
		before_s = [list(s) for s in g.state_sampled]
		before_o = [list(s) for s in g.output_sampled]
		for suffix in g.state_sampled:
			_resample_in_place(suffix, g.shape.state_input_space, rng, rate)
		for suffix in g.output_sampled:
			_resample_in_place(suffix, g.shape.output_input_space, rng, rate)
		if g.cells is not None:
			changed_s = {i for i in range(len(g.state_sampled)) if g.state_sampled[i] != before_s[i]}
			changed_o = {i for i in range(len(g.output_sampled)) if g.output_sampled[i] != before_o[i]}
			if changed_s:
				g.cells.state_universe, g.cells.state_values = _drop_changed_neurons(
					g.cells.state_universe, g.cells.state_values, changed_s)
			if changed_o:
				g.cells.output_universe, g.cells.output_values = _drop_changed_neurons(
					g.cells.output_universe, g.cells.output_values, changed_o)
		return g

	def _mutate_memory(self, rate: float, rng: np.random.Generator) -> "RecurrentArchGenome":
		"""MEMORY dimension: nudge ~rate of the stored QSR cells ±1 (clamped 0..3),
		architecture frozen. This is paradigm B / GA-Memory's value mutation."""
		if self.cells is None:
			raise ValueError(
				"MEMORY-dimension mutation needs a recorded cells universe; record "
				"it (record_address_universe) before running a MEMORY phase.")
		g = self.clone()
		for vals in (g.cells.state_values, g.cells.output_values):
			for i in range(len(vals)):
				if rng.random() < rate:
					vals[i] = int(np.clip(vals[i] + (1 if rng.random() < 0.5 else -1), 0, 3))
		return g

	def _remap_state_neuro(self, k: int, sw: int, ow: int, removed_floor: int) -> None:
		"""Remap cells through a STATE-neurogenesis of +k (or -k) neurons. The
		prefix grows/shrinks in BOTH layers; removed state neurons' own cells go."""
		c = self.cells
		if k > 0:
			c.state_universe, c.state_values = _remap_prefix_grow(c.state_universe, c.state_values, k, sw)
			c.output_universe, c.output_values = _remap_prefix_grow(c.output_universe, c.output_values, k, ow)
		else:
			# Drop removed state neurons' own cells first, then collapse the prefix.
			c.state_universe, c.state_values = _drop_neurons_ge(c.state_universe, c.state_values, removed_floor)
			c.state_universe, c.state_values = _remap_prefix_shrink(c.state_universe, c.state_values, -k, sw)
			c.output_universe, c.output_values = _remap_prefix_shrink(c.output_universe, c.output_values, -k, ow)

	# ---- deterministic arch edits with cell remap (in place; caller clones) --
	# These are the single place where a shape change + its cell remap live, so
	# random mutation (_mutate_*) AND stats-guided genesis (ControllerAdaptation)
	# share identical, tested remap behavior. Pass an explicit TARGET value.

	def set_state_neurons(self, target: int, rng: np.random.Generator) -> None:
		"""STATE neurogenesis to `target` neurons; remaps cells in BOTH layers."""
		k = target - self.state_neurons
		if k == 0:
			return
		sw, ow = self.state_suffix_width, self.output_suffix_width  # unchanged by neuro
		self._resize_state_neurons(target, rng)
		if self.cells is not None:
			self._remap_state_neuro(k, sw, ow, removed_floor=target)

	def set_output_neurons(self, target: int, rng: np.random.Generator) -> None:
		"""OUTPUT neurogenesis to `target` neurons (multiple of output_quantum).
		Survivors keep cells verbatim; removed tail blocks' cells are dropped."""
		if target == self.output_neurons:
			return
		if self.cells is not None and target < self.output_neurons:
			self.cells.output_universe, self.cells.output_values = _drop_neurons_ge(
				self.cells.output_universe, self.cells.output_values, target)
		self._resize_output_neurons(target, rng)

	def set_state_suffix(self, target: int, rng: np.random.Generator) -> None:
		"""Synaptogenesis: set state sampled-suffix width to `target`; remap cells."""
		old = self.state_suffix_width
		if target == old:
			return
		for suffix in self.state_sampled:
			_resize_suffix(suffix, self.shape.state_input_space, target, rng)
		if self.cells is not None:
			self.cells.state_universe, self.cells.state_values = _remap_bits(
				self.cells.state_universe, self.cells.state_values, target - old)

	def set_output_suffix(self, target: int, rng: np.random.Generator) -> None:
		"""Synaptogenesis: set output sampled-suffix width to `target`; remap cells."""
		old = self.output_suffix_width
		if target == old:
			return
		for suffix in self.output_sampled:
			_resize_suffix(suffix, self.shape.output_input_space, target, rng)
		if self.cells is not None:
			self.cells.output_universe, self.cells.output_values = _remap_bits(
				self.cells.output_universe, self.cells.output_values, target - old)

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

	@staticmethod
	def crossover_memory(a: "RecurrentArchGenome", b: "RecurrentArchGenome",
	                     rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Per-cell uniform crossover of cell VALUES — the MEMORY-phase recombination
		(paradigm B). Assumes a and b share architecture + universe, which a
		frozen-arch MEMORY phase guarantees. Falls back to `a` if either lacks cells."""
		child = a.clone()
		if child.cells is None or b.cells is None:
			return child
		child.cells.state_values = [
			a.cells.state_values[i] if rng.random() < 0.5 else b.cells.state_values[i]
			for i in range(len(a.cells.state_values))]
		child.cells.output_values = [
			a.cells.output_values[i] if rng.random() < 0.5 else b.cells.output_values[i]
			for i in range(len(a.cells.output_values))]
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
		if self.cells is not None:
			self._assert_cells_valid()

	def _assert_cells_valid(self) -> None:
		c = self.cells
		assert len(c.state_values) == len(c.state_universe), "state values/universe misaligned"
		assert len(c.output_values) == len(c.output_universe), "output values/universe misaligned"
		s_max = 1 << self.state_bits_per_neuron
		o_max = 1 << self.output_bits_per_neuron
		assert len(set(c.state_universe)) == len(c.state_universe), "duplicate state cell key"
		assert len(set(c.output_universe)) == len(c.output_universe), "duplicate output cell key"
		for (n, a), v in zip(c.state_universe, c.state_values):
			assert 0 <= n < self.state_neurons, "state cell neuron out of range"
			assert 0 <= a < s_max, "state cell address exceeds 2^bits"
			assert 0 <= v <= 3, "state cell value not QSR 0..3"
		for (n, a), v in zip(c.output_universe, c.output_values):
			assert 0 <= n < self.output_neurons, "output cell neuron out of range"
			assert 0 <= a < o_max, "output cell address exceeds 2^bits"
			assert 0 <= v <= 3, "output cell value not QSR 0..3"


def _mix_blocks(into: list[list[int]], other: list[list[int]], rng: np.random.Generator) -> None:
	"""For each block in `into`, with p=0.5 take `other`'s block of the same
	index — but only if it exists and has matching width (keeps suffix uniform)."""
	if not into:
		return
	width = len(into[0])
	for i in range(len(into)):
		if i < len(other) and len(other[i]) == width and rng.random() < 0.5:
			into[i] = list(other[i])


__all__ = ["RecurrentArchGenome", "RecurrentArchShape", "RecurrentArchConfig", "MemoryPayload"]
