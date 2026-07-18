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
	# Phase-5c saturation→grow DAMPING (§11b). The splitting trainer emits a
	# `saturation` count (conflicts it could not resolve for lack of free state
	# neurons); _mutate_neurons turns that into state growth. Old behavior FORCED a
	# grow on EVERY offspring whenever saturation>0, churning doomed over-grown
	# genomes (selection prunes them, but it wastes evals). Now growth is
	# PROBABILISTIC: grow_p = min(1, rate + saturation*saturation_grow_gain). Lower
	# gain = gentler/measured growth. Default 0.02 keeps high saturation (≥50) at
	# grow_p≈1; drop toward 0.005 to damp hard on noisy-high-saturation tasks.
	saturation_grow_gain: float = 0.02
	# Per-genome cell budget. BITS-grow replicates cells ×2^d, so without this a
	# long mixed run balloons memory (the dead mixed GA hit 16 GB by gen ~108). A
	# grow is clamped so total cells stay ≤ max_cells. Default huge ⇒ no effect on
	# single-dimension phases; the mixed GA sets it tight.
	max_cells: int = 1_000_000_000
	# Feature-balance cap (26/06/2026): no input FEATURE may capture more than
	# `feature_balance_ratio` × the least-wired feature's connection count. Targets the
	# obs_yaw_err 2.14x over-wiring → coupling. ≤1.0 disables. bits_per_feature maps
	# sampled bit indices → feature groups.
	feature_balance_ratio: float = 0.0
	bits_per_feature: int = 8
	# Memory mode (ABI 12 granularity ablation): "QUAD_WEIGHTED" (±1 lattice
	# cell mutation, values 0..3) / "TERNARY"/"BINARY" (flip FALSE↔TRUE, values
	# {0,1}). Threaded from ControllerSpec.memory_mode like feature_balance_ratio.
	memory_mode: str = "QUAD_WEIGHTED"


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


def _feature_of(idx: int, frame_bits: int, bpf: int) -> int:
	"""Sampled input-bit index → its FEATURE group (window-folded: state bits repeat
	the frame `window` times, so fold by frame_bits first)."""
	return (idx % frame_bits) // bpf


def _rebalance_features(sampled: list[list[int]], space: int, frame_bits: int,
                        bpf: int, ratio: float, rng: np.random.Generator) -> None:
	"""Feature-balance cap (26/06/2026). In place: move sampled bits from OVER-represented
	input features to UNDER-represented ones until no feature's total connection count
	exceeds `ratio` × the least-wired feature's count. Stops the GA from letting one salient
	feature (e.g. obs_yaw_err) capture a disproportionate share of connectivity — the 2.14x
	over-wiring that drove the coupling/brittleness. Maintains per-neuron distinctness.
	ratio<=1 (or too few features) disables."""
	if ratio <= 1.0 or not sampled or frame_bits <= 0 or bpf <= 0:
		return
	nfeat = frame_bits // bpf
	if nfeat <= 1:
		return
	feat_bits: list[list[int]] = [[] for _ in range(nfeat)]
	for b in range(space):
		feat_bits[_feature_of(b, frame_bits, bpf)].append(b)
	counts = [0] * nfeat
	for suf in sampled:
		for b in suf:
			counts[_feature_of(b, frame_bits, bpf)] += 1
	max_iter = sum(counts) * 4 + 100
	for _ in range(max_iter):
		hi = int(np.argmax(counts)); lo = int(np.argmin(counts))
		if counts[hi] <= ratio * max(counts[lo], 1):
			break
		moved = False
		for ni in rng.permutation(len(sampled)):
			suf = sampled[ni]; sufset = set(suf)
			hi_pos = [k for k, b in enumerate(suf) if _feature_of(b, frame_bits, bpf) == hi]
			if not hi_pos:
				continue
			cands = [b for b in feat_bits[lo] if b not in sufset]
			if not cands:
				continue
			suf[hi_pos[0]] = int(rng.choice(cands))
			counts[hi] -= 1; counts[lo] += 1; moved = True
			break
		if not moved:
			break


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

	@classmethod
	def from_triples(cls, state_triples, output_triples) -> "MemoryPayload":
		"""Inverse of to_triples(). Accepts tuples OR lists (YAML round-trip)."""
		return cls(
			state_universe=[(int(n), int(a)) for n, a, _ in state_triples],
			output_universe=[(int(n), int(a)) for n, a, _ in output_triples],
			state_values=[int(v) for _, _, v in state_triples],
			output_values=[int(v) for _, _, v in output_triples],
		)


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


def _remap_prefix_grow(universe, values, k, w, pf=2):
	"""STATE neurogenesis +k: prefix gains `pf·k` mid-bits (just above the w-bit
	suffix), defaulting to the per-neuron neutral feedback. A = P·2^w + S becomes
	P·2^(pf·k+w) + neutral·2^w + S — behavior preserved on the neutral branch.
	`pf` = prefix_factor (bits a state neuron contributes to the address): 2 for
	the legacy QSR pair, 1 for the current 1-bit MSB-only feedback (where a fresh
	neuron emits 0 → neutral feedback bit is 0)."""
	if k <= 0:
		return list(universe), list(values)
	mask = (1 << w) - 1
	per = NEUTRAL_PAIR if pf == 2 else 0   # per-neuron neutral feedback (1-bit → 0)
	neutral = 0
	for j in range(k):  # k groups of `pf` bits, lowest group at j=0
		neutral |= per << (pf * j)
	nu, nv = [], []
	for (n, a), v in zip(universe, values):
		P, S = a >> w, a & mask
		nu.append((n, (P << (pf * k + w)) | (neutral << w) | S))
		nv.append(v)
	return nu, nv


def _remap_prefix_shrink(universe, values, k, w, pf=2):
	"""STATE neurogenesis -k: drop the lowest `pf·k` prefix bits; majority collapse.
	A = P_high·2^(pf·k+w) + group·2^w + S  →  P_high·2^w + S."""
	if k <= 0:
		return list(universe), list(values)
	mask = (1 << w) - 1
	buckets: dict[tuple[int, int], list[int]] = {}
	for (n, a), v in zip(universe, values):
		P_high, S = a >> (pf * k + w), a & mask
		buckets.setdefault((n, (P_high << w) | S), []).append(v)
	nu = list(buckets.keys())
	nv = [_majority(buckets[k2]) for k2 in nu]
	return nu, nv


def _remap_delete_bit_window(universe, values, p_lsb, nbits=2):
	"""Excise `nbits` adjacent address bits starting at position p_lsb (delete a
	mid-address field), majority-collapsing per-neuron collisions. Used by
	surgical state-neuron removal — deleting a NON-tail neuron's prefix bit-pair
	from every address (vs the tail-collapse in _remap_prefix_shrink)."""
	if p_lsb < 0:
		return list(universe), list(values)
	mask_low = (1 << p_lsb) - 1
	buckets: dict[tuple[int, int], list[int]] = {}
	for (n, a), v in zip(universe, values):
		low = a & mask_low
		high = a >> (p_lsb + nbits)
		buckets.setdefault((n, (high << p_lsb) | low), []).append(v)
	nu = list(buckets.keys())
	nv = [_majority(buckets[k]) for k in nu]
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
	# Phase 5c: GA-handshake pressure from the LAST evaluation of THIS genome —
	# (saturation_count, wished_state_input_bits) emitted by the splitting trainer.
	# EVAL metadata, NOT structure: compare=False so it never affects identity, and
	# clone() drops it (a fresh child has no pressure until it is itself evaluated).
	# The mutators read the PARENT's pressure to bias offspring toward what the
	# trainer asked for: grow state_neurons on saturation, route connections to the
	# wished sensor bits.
	pressure: tuple = field(default=(0, ()), compare=False)

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
	           state_suffix: int, output_suffix: int, rng: np.random.Generator,
	           config: "RecurrentArchConfig | None" = None) -> "RecurrentArchGenome":
		"""Fresh random genome of a given shape (prefix forced; suffixes sampled).
		If `config` enables the feature-balance cap, the sampled suffixes are rebalanced."""
		ss = [_sample_distinct(shape.state_input_space, state_suffix, rng) for _ in range(state_neurons)]
		os = [_sample_distinct(shape.output_input_space, output_suffix, rng) for _ in range(output_neurons)]
		if config is not None and config.feature_balance_ratio > 1.0:
			fb, bpf, r = shape.output_input_space, config.bits_per_feature, config.feature_balance_ratio
			_rebalance_features(ss, shape.state_input_space, fb, bpf, r, rng)
			_rebalance_features(os, shape.output_input_space, fb, bpf, r, rng)
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

	# ---- recording interface (so the shared Experiment checkpoint/dashboard
	#      code treats this as a peer of ClusterGenome — no special-casing) -----

	def stats(self) -> dict:
		"""Architecture summary in the shape the dashboard/checkpoint code reads."""
		sc, oc = self.to_connections()
		return {
			"num_clusters": 1,
			"total_neurons": self.state_neurons + self.output_neurons,
			"total_connections": len(sc) + len(oc),
			"min_bits": min(self.state_bits_per_neuron, self.output_bits_per_neuron),
			"max_bits": max(self.state_bits_per_neuron, self.output_bits_per_neuron),
			"min_neurons": self.state_neurons,
			"max_neurons": self.output_neurons,
		}

	def serialize(self) -> dict:
		"""JSON-serializable snapshot (for checkpoint persistence).

		Cells (the bulk — millions of (neuron, addr, value) triples) go through
		the shared int-column packer, so a populated MEMORY genome serializes to a
		few base64 scalars instead of tens of millions of YAML nodes (13/06/2026
		fix). >int64 addresses (state_bits > 63) use the packer's "i128" format —
		the old verbose-list fallback is GONE: it turned a compact write into a
		multi-GB PyYAML node tree (the 12/07/2026 pid@31337004 OOM kill).
		"""
		sh = self.shape
		cells_payload = None
		if self.cells is not None:
			from wnn.ram.strategies.phased.packing import pack_int_columns
			st, ot = self.cells.to_triples()
			cells_payload = {"state": pack_int_columns(st, 3),
			                 "output": pack_int_columns(ot, 3)}
		return {
			"type": "RecurrentArchGenome",
			"shape": [sh.prefix_factor, sh.state_input_space, sh.output_input_space, sh.output_quantum],
			"state_neurons": self.state_neurons,
			"output_neurons": self.output_neurons,
			"state_sampled": [list(s) for s in self.state_sampled],
			"output_sampled": [list(s) for s in self.output_sampled],
			"cells": cells_payload,
		}

	@classmethod
	def deserialize(cls, data: dict) -> "RecurrentArchGenome":
		"""Inverse of serialize(). Native (de)serialization means controller
		checkpoints carry NO pickle — refactor-proof and shell-inspectable."""
		sh = data["shape"]
		shape = RecurrentArchShape(
			prefix_factor=int(sh[0]), state_input_space=int(sh[1]),
			output_input_space=int(sh[2]), output_quantum=int(sh[3]),
		)
		cells = None
		c = data.get("cells")
		if c is not None:
			if isinstance(c, dict):  # new packed format {"state": ..., "output": ...}
				from wnn.ram.strategies.phased.packing import unpack_int_columns, is_packed
				st = unpack_int_columns(c["state"]) if is_packed(c["state"]) else c["state"]
				ot = unpack_int_columns(c["output"]) if is_packed(c["output"]) else c["output"]
			else:  # legacy / overflow-fallback: [state_triples, output_triples]
				st, ot = c
			cells = MemoryPayload.from_triples(st, ot)
		return cls(
			shape=shape,
			state_neurons=int(data["state_neurons"]),
			output_neurons=int(data["output_neurons"]),
			state_sampled=[[int(b) for b in s_] for s_ in data["state_sampled"]],
			output_sampled=[[int(b) for b in s_] for s_ in data["output_sampled"]],
			cells=cells,
		)

	def compute_tier_stats(self, tier_config) -> dict:
		"""Controllers have no tiers — return empty (only called when tier_config set)."""
		return {}

	# ---- phase-aware mutation (the GA/TS/Lamarckian entry point) -------------

	def mutate(self, dim: OptimizationDimension, rate: float, config: RecurrentArchConfig,
	           rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Return a mutated copy. The dimension selects which axis moves; the
		forced prefix is regenerated by to_connections() and so is never at risk."""
		if dim == OptimizationDimension.MEMORY:
			return self._mutate_memory(rate, rng, config.memory_mode)   # no connectivity change → no rebalance
		if dim == OptimizationDimension.NEURONS:
			g = self._mutate_neurons(rate, config, rng)
		elif dim == OptimizationDimension.BITS:
			g = self._mutate_bits(rate, config, rng)
		elif dim == OptimizationDimension.CONNECTIONS:
			g = self._mutate_connections(rate, config, rng)
		elif dim == OptimizationDimension.CLUSTER:  # all ARCHITECTURE dims at once
			g = self._mutate_neurons(rate, config, rng)._mutate_bits(rate, config, rng)._mutate_connections(rate, config, rng)
		else:
			raise ValueError(f"unknown optimization dimension: {dim!r}")
		# Single-point feature-balance enforcement: every connectivity-changing mutation
		# is re-projected into the balanced set (cell-safe; no-op when the cap is off).
		return g._apply_feature_balance(config, rng)

	def _apply_feature_balance(self, config: RecurrentArchConfig,
	                           rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Project this genome's sampled suffixes into the feature-balanced set IN PLACE,
		dropping cells for any neuron whose wiring changed. No-op if the cap is disabled."""
		if config.feature_balance_ratio <= 1.0:
			return self
		fb, bpf, r = self.shape.output_input_space, config.bits_per_feature, config.feature_balance_ratio
		before_s = [list(s) for s in self.state_sampled]
		before_o = [list(s) for s in self.output_sampled]
		_rebalance_features(self.state_sampled, self.shape.state_input_space, fb, bpf, r, rng)
		_rebalance_features(self.output_sampled, self.shape.output_input_space, fb, bpf, r, rng)
		if self.cells is not None:
			changed_s = {i for i in range(len(self.state_sampled)) if self.state_sampled[i] != before_s[i]}
			changed_o = {i for i in range(len(self.output_sampled)) if self.output_sampled[i] != before_o[i]}
			if changed_s:
				self.cells.state_universe, self.cells.state_values = _drop_changed_neurons(
					self.cells.state_universe, self.cells.state_values, changed_s)
			if changed_o:
				self.cells.output_universe, self.cells.output_values = _drop_changed_neurons(
					self.cells.output_universe, self.cells.output_values, changed_o)
		return self

	def _mutate_neurons(self, rate: float, config: RecurrentArchConfig,
	                    rng: np.random.Generator) -> "RecurrentArchGenome":
		"""State + output neurogenesis. Survivors keep their suffixes verbatim;
		growth appends fresh tail blocks (small-neighborhood rule). Cells, if
		present, are remapped: STATE neuro reshapes the prefix in BOTH layers;
		OUTPUT neuro keeps survivors verbatim and drops removed blocks."""
		g = self.clone()
		saturation = self.pressure[0] if self.pressure else 0
		# STATE neurogenesis (memory capacity): reshapes the prefix globally.
		# Phase 5c (DAMPED §11b): under SATURATION pressure (the splitting trainer
		# found conflicts it could not resolve for lack of free neurons), bias toward
		# GROWTH — but PROBABILISTICALLY, not on every offspring. grow_p scales with
		# saturation; only when the gate fires do we bias the delta to a small grow.
		# (Old behavior force-grew every genome whenever saturation>0, churning
		# doomed over-grown offspring that selection then had to prune.)
		grow_p = min(1.0, rate + saturation * config.saturation_grow_gain)
		if rng.random() < grow_p and config.state_neuron_delta > 0:
			delta = int(rng.integers(-config.state_neuron_delta, config.state_neuron_delta + 1))
			if saturation > 0 and delta <= 0:
				delta = 1
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
		# Phase 5c: CONNECTIVITY pressure — route a state neuron to each WISHED
		# sensor bit (a separator the trainer wanted but no neuron observed). Inject
		# the wish into a random state-neuron suffix (replacing a random entry); the
		# wish positions are state-input indices (< state_input_space) by construction.
		wish = self.pressure[1] if self.pressure else ()
		for wb in wish:
			if 0 <= wb < g.shape.state_input_space and g.state_sampled:
				ni = int(rng.integers(0, len(g.state_sampled)))
				suffix = g.state_sampled[ni]
				if wb not in suffix:
					suffix[int(rng.integers(0, len(suffix)))] = wb
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

	def _mutate_memory(self, rate: float, rng: np.random.Generator,
	                   memory_mode: str = "QUAD_WEIGHTED") -> "RecurrentArchGenome":
		"""MEMORY dimension: nudge ~rate of the stored cells one step — QUAD/QSR ±1
		(clamped 0..3); TERNARY/BINARY/PLN flip FALSE↔TRUE (2-state analog, ABI 12).
		Architecture frozen. This is paradigm B / GA-Memory's value mutation."""
		if self.cells is None:
			raise ValueError(
				"MEMORY-dimension mutation needs a recorded cells universe; record "
				"it (record_address_universe) before running a MEMORY phase.")
		g = self.clone()
		# QSR is a stochastic QUAD read → 4-state graded cells; PLN shares TERNARY's
		# 2-state cells. Keep this consistent with ga_memory.MemoryGenome.
		quad = memory_mode.upper() in ("QUAD_WEIGHTED", "QUAD_BINARY", "QSR")
		for vals in (g.cells.state_values, g.cells.output_values):
			for i in range(len(vals)):
				if rng.random() < rate:
					if quad:
						vals[i] = int(np.clip(vals[i] + (1 if rng.random() < 0.5 else -1), 0, 3))
					else:
						# 2-state flip. Mask to the low bit FIRST so an EMPTY (EMPTY_U8=2,
						# the untrained-cell baseline carried in the universe) flips to a
						# definite TRUE(1) instead of 1-2=-1, which overflows the Rust
						# write_state_cell u8 (OverflowError). 0<->1, 2/3->0/1, never negative.
						# Mirror of the ga_memory.py:186 fix — this is the NEURONS-stage
						# Lamarckian joint-mutation twin of that MEMORY-stage flip.
						vals[i] = 1 - (vals[i] & 1)
		return g

	def _remap_state_neuro(self, k: int, sw: int, ow: int, removed_floor: int) -> None:
		"""Remap cells through a STATE-neurogenesis of +k (or -k) neurons. The
		prefix grows/shrinks in BOTH layers; removed state neurons' own cells go."""
		c = self.cells
		pf = self.shape.prefix_factor
		if k > 0:
			c.state_universe, c.state_values = _remap_prefix_grow(c.state_universe, c.state_values, k, sw, pf)
			c.output_universe, c.output_values = _remap_prefix_grow(c.output_universe, c.output_values, k, ow, pf)
		else:
			# Drop removed state neurons' own cells first, then collapse the prefix.
			c.state_universe, c.state_values = _drop_neurons_ge(c.state_universe, c.state_values, removed_floor)
			c.state_universe, c.state_values = _remap_prefix_shrink(c.state_universe, c.state_values, -k, sw, pf)
			c.output_universe, c.output_values = _remap_prefix_shrink(c.output_universe, c.output_values, -k, ow, pf)

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

	def remove_state_neuron(self, k: int, rng: np.random.Generator) -> None:
		"""Surgically remove state neuron `k` (any index, not just the tail): drop
		its own cells + wiring, reindex higher neurons down by one, and excise its
		2-bit prefix window from EVERY address in BOTH layers (majority-collapsing
		collisions). Used by stats-guided neurogenesis to prune a specific dead
		neuron without disturbing the others' learned behaviour beyond the forced
		state-context shrink."""
		n = self.state_neurons
		if not (0 <= k < n) or n <= 1:
			return
		w_s, w_o = self.state_suffix_width, self.output_suffix_width
		# Neuron k's prefix window = connection indices [pf·k, pf·k+pf) → adjacent
		# address bits; the LSB of that pf-bit window sits at (bits - pf - pf·k) for
		# each layer. pf = prefix_factor (2 legacy QSR pair, 1 current MSB-only).
		pf = self.shape.prefix_factor
		p_lsb_s = (pf * n + w_s) - pf - pf * k
		p_lsb_o = (pf * n + w_o) - pf - pf * k
		del self.state_sampled[k]
		self.state_neurons = n - 1
		if self.cells is not None:
			c = self.cells
			# State: drop neuron k's cells, reindex neurons > k, then excise window.
			su, sv = [], []
			for (nn, a), v in zip(c.state_universe, c.state_values):
				if nn == k:
					continue
				su.append((nn - 1 if nn > k else nn, a))
				sv.append(v)
			c.state_universe, c.state_values = _remap_delete_bit_window(su, sv, p_lsb_s, pf)
			# Output: same window excised; output neuron indices unchanged.
			c.output_universe, c.output_values = _remap_delete_bit_window(
				c.output_universe, c.output_values, p_lsb_o, pf)

	def rewire_suffix(self, state_changes: dict, output_changes: dict) -> None:
		"""Axonogenesis: replace specific neurons' sampled suffixes IN PLACE (each
		new suffix must keep the layer's uniform width) and drop those neurons'
		cells (their address semantics changed — same rule as random CONNECTIONS
		mutation). `*_changes` map neuron index → new sampled-bit list."""
		for n, new in state_changes.items():
			self.state_sampled[n] = list(new)
		for n, new in output_changes.items():
			self.output_sampled[n] = list(new)
		if self.cells is not None:
			if state_changes:
				self.cells.state_universe, self.cells.state_values = _drop_changed_neurons(
					self.cells.state_universe, self.cells.state_values, set(state_changes))
			if output_changes:
				self.cells.output_universe, self.cells.output_values = _drop_changed_neurons(
					self.cells.output_universe, self.cells.output_values, set(output_changes))

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
	def crossover_average(a: "RecurrentArchGenome", b: "RecurrentArchGenome",
	                      rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Average-shape variable crossover (the user's spec, 2026-05-28).

		Target shape = ELEMENT-WISE AVERAGE of parent shapes:
		- target_state_neurons   = (a.state_neurons + b.state_neurons) // 2
		- target_output_neurons  = ((a.output_neurons + b.output_neurons) // 2)
		                           rounded to the nearest multiple of output_quantum
		- target_state_suffix    = (a.suffix + b.suffix) // 2 (clamped to input space)
		- target_output_suffix   = same

		For each new neuron position i in the child:
		- If both parents have a neuron at index i → uniformly pick one parent's
		  suffix; resize (truncate or pad with new random bits) to target width
		- If only one parent has a neuron at index i (the larger parent) → use
		  that one's suffix; resize to target width
		- Neither has it → random sample (only fires when target > max(a,b),
		  which is impossible with average; included as a defensive fallback)

		Cells: DROPPED (set to None). Addresses derive from connections + bits,
		and BOTH can differ from either parent → no inherited universe is
		address-aligned with the child's runtime. score_genomes handles
		cells=None gracefully (the controller scores fresh from defaults, will
		typically rank low and be culled, but doesn't crash); the GA's MEMORY
		mutator on subsequent generations refills cells from re-recorded
		universes for the survivors that propagate.

		This is the principled GA recombination the user asked for. The
		one-parent-shape `crossover()` variant above stays as Plan B (faster,
		preserves cells, but doesn't interpolate shape).
		"""
		assert a.shape == b.shape, "crossover requires both parents share the structural shape"
		shape = a.shape
		q = shape.output_quantum

		# ---- Target shape: element-wise average of parents -----------------
		target_state_n = max(1, (a.state_neurons + b.state_neurons) // 2)
		avg_out = (a.output_neurons + b.output_neurons) // 2
		target_output_n = max(q, int(round(avg_out / q)) * q)  # nearest multiple of q

		target_state_suf = max(1, (a.state_suffix_width + b.state_suffix_width) // 2)
		target_output_suf = max(1, (a.output_suffix_width + b.output_suffix_width) // 2)
		# Clamp to input-space limits (can't sample more distinct bits than exist).
		target_state_suf = min(target_state_suf, shape.state_input_space)
		target_output_suf = min(target_output_suf, shape.output_input_space)

		def _pick_or_resample(layer_a: list[list[int]], layer_b: list[list[int]],
		                      i: int, space: int, target_width: int) -> list[int]:
			"""Pick parent suffix at index i (uniform if both have it, else the
			one that does, else random sample), resize to target_width."""
			a_has = i < len(layer_a)
			b_has = i < len(layer_b)
			if a_has and b_has:
				src = layer_a[i] if rng.random() < 0.5 else layer_b[i]
			elif a_has:
				src = layer_a[i]
			elif b_has:
				src = layer_b[i]
			else:
				return _sample_distinct(space, target_width, rng)
			new = list(src)
			_resize_suffix(new, space, target_width, rng)
			return new

		new_state_sampled = [
			_pick_or_resample(a.state_sampled, b.state_sampled, i,
			                  shape.state_input_space, target_state_suf)
			for i in range(target_state_n)
		]
		new_output_sampled = [
			_pick_or_resample(a.output_sampled, b.output_sampled, i,
			                  shape.output_input_space, target_output_suf)
			for i in range(target_output_n)
		]

		return RecurrentArchGenome(
			shape=shape,
			state_neurons=target_state_n,
			output_neurons=target_output_n,
			state_sampled=new_state_sampled,
			output_sampled=new_output_sampled,
			cells=None,  # universe is shape-keyed; let evaluator handle/re-record
		)

	@staticmethod
	def crossover_memory(a: "RecurrentArchGenome", b: "RecurrentArchGenome",
	                     rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Address-KEYED uniform crossover of cell VALUES — the MEMORY-phase
		recombination (paradigm B).

		A QSR cell value is only meaningful for a SPECIFIC cell, identified by its
		``(neuron_idx, address)`` universe key. So we recombine by KEY, not by list
		position: the child keeps ``a``'s architecture + universe, and for each of
		``a``'s cells it adopts ``b``'s value with prob 0.5 IFF ``b``'s universe
		contains that same ``(neuron, address)`` cell — otherwise it keeps ``a``'s.

		Why keyed, not positional: the full-population carry into the MEMORY stage
		mixes genomes of DIFFERENT shapes (varying neurons/bits → different-length,
		differently-keyed universes). Positional zipping crashed (IndexError) and
		was semantically wrong even when lengths matched. Keyed crossover:
		  * never crashes on heterogeneous shapes;
		  * genuinely recombines learned memory wherever two genomes' universes
		    overlap (e.g. same-shape parents within a shape-cluster → full mixing);
		  * for different-shape parents with little overlap, the child ≈ clone(a)
		    (benign — ``b`` has no opinion on cells it never visited);
		  * is EXACTLY the old positional crossover when a and b share a universe
		    (the homogeneous frozen-arch case — every key is present in both).
		Falls back to a clone of ``a`` if either parent lacks cells."""
		child = a.clone()
		if child.cells is None or b.cells is None:
			return child

		def _mix(a_universe, a_values, b_universe, b_values):
			b_map = {key: v for key, v in zip(b_universe, b_values)}
			# Keep a's value unless b shares the cell AND the coin says take b.
			# (key absent from b_map short-circuits → no rng draw, child keeps a;
			# when present, this is exactly `a if rng<0.5 else b` per cell.)
			return [
				a_val if (key not in b_map or rng.random() < 0.5) else b_map[key]
				for key, a_val in zip(a_universe, a_values)
			]

		child.cells.state_values = _mix(
			a.cells.state_universe, a.cells.state_values,
			b.cells.state_universe, b.cells.state_values)
		child.cells.output_values = _mix(
			a.cells.output_universe, a.cells.output_values,
			b.cells.output_universe, b.cells.output_values)
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
