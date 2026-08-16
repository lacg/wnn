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
from wnn.control import _accel as ra   # memory_* cell operators (Rust, counter RNG)


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
	# Per-genome cell budget. BITS-grow replicates cells ×2^d and a WANDERING
	# controller populates ever more distinct addresses, so without this a long
	# run balloons memory (the dead mixed GA hit 16 GB by gen ~108; the 23/07
	# QUAD-dfa study cells hit 200k+ cells/genome and OOM-looped). Enforced in
	# _mutate_neurons/_mutate_bits (wired 23/07/2026 — was documented but dead):
	# once a genome's carried Lamarckian cells reach the budget, structural GROWS
	# are suppressed (shrinks/rewires stay allowed, so selection can still slim
	# it). Default huge ⇒ no effect unless --max-cells sets it tight.
	max_cells: int = 1_000_000_000
	# STRICT budget (31/07/2026, --max-cells-strict). max_cells above is a GROW-
	# SUPPRESSION THRESHOLD, not a ceiling: a genome UNDER budget may still take a
	# legal bits-grow, and that grow replicates its layer ×2^delta. With
	# suffix_delta=2 a genome at 145k lands at 580k — measured 3.22× the 180k
	# budget on 1layer_10feat_QUAD_s31337002, and 8/8 of the cells that overshot
	# were QUAD (no BINARY cell ever tripped 180k at all). That made the
	# granularity ablation NOT budget-matched: QUAD was silently allowed up to 3×
	# the memory of BINARY and still lost every pair.
	# When True, a grow is clamped to the largest delta whose POST-grow count
	# still fits max_cells, so the budget behaves like its name. Default False
	# keeps the historical behaviour bit-for-bit (and adds no RNG draw either
	# way, so the deterministic stream is identical in both modes).
	# NOT covered: neurogenesis, which scales cells roughly linearly rather than
	# ×2^d, and cells written during training/DAgger, which no mutation gate sees.
	strict_cell_budget: bool = False
	# Feature-balance cap (26/06/2026): no input FEATURE may capture more than
	# `feature_balance_ratio` × the least-wired feature's connection count. Targets the
	# obs_yaw_err 2.14x over-wiring → coupling. ≤1.0 disables. bits_per_feature maps
	# sampled bit indices → feature groups.
	feature_balance_ratio: float = 0.0
	bits_per_feature: int = 8
	# Connection-creation POLICY (14/08/2026, Luiz's specialist programme). Governs
	# how FRESH output-layer maps are drawn (init + output neurogenesis); state
	# maps and inherited/crossed-over maps are untouched. Forensics on 5 banked
	# winners (analyze_winner_connectivity.py) showed uniform sampling leaves ~33%
	# of (neuron, feature) pairs at ONE threshold — a half-plane, unable to express
	# "in this band" — which min-2 converts to interval detection.
	#   "spread"          — today's uniform draw (bit-identical control)
	#   "min_per_cluster" — every TOUCHED feature gets >= conn_policy_min bits;
	#                       features that cannot be afforded are DROPPED and their
	#                       budget donated (m=2: b=30 touches 15 of 18 features;
	#                       m=3: 10 — a dose axis from generalist toward specialist)
	conn_policy: str = "spread"
	conn_policy_min: int = 2
	# FRAMED coverage (16/08/2026, Luiz): "framed1" needs to know how many frames
	# the OUTPUT input space holds, which only arm D (output_full_window) makes
	# wider than 1. The sampler cannot derive k and num_features separately from
	# `space` alone (space = k * nfeat * bpf), so k is threaded like bits_per_feature.
	input_window_k: int = 1
	# GA-connectivity mutation SCOPE (16/08/2026, Luiz's connectivity types).
	# Governs where a CONNECTIONS-stage rewire may land, per mutated connection:
	#   "free"    — anywhere in the space (legacy axonogenesis, bit-identical)
	#   "window"  — the original bit's window only (never crosses time; at k=1
	#               this degenerates to free)
	#   "feature" — the original bit's thermometer run only: WHERE on the feature
	#               moves, the feature map itself is frozen at what grid/init chose
	# OUTPUT maps only — state maps keep the free draw (same convention as
	# conn_policy). Implementation: arch_ops::resample_suffix_scoped (Rust).
	conn_mutation_scope: str = "free"
	# Memory mode (ABI 12 granularity ablation): "QUAD_WEIGHTED" (±1 lattice
	# cell mutation, values 0..3) / "TERNARY"/"BINARY" (flip FALSE↔TRUE, values
	# {0,1}). Threaded from ControllerSpec.memory_mode like feature_balance_ratio.
	memory_mode: str = "QUAD_WEIGHTED"


# ---- small sampling helpers --------------------------------------------------

def _sample_distinct(space: int, k: int, rng: np.random.Generator,
                     exclude: set[int] | None = None) -> list[int]:
	"""k distinct indices in [0, space), avoiding `exclude`. Clamped to what fits."""
	if k <= 0 or space <= 0:
		return []
	seed = int(rng.integers(0, 1 << 63))
	return [int(b) for b in ra.arch_sample_distinct(
		int(space), int(k), [int(x) for x in (exclude or ())], seed, 0, 0, 0)]


_CONN_SCOPE_CODE = {"free": 0, "window": 1, "feature": 2}


def _resample_in_place(suffix: list[int], space: int, rng: np.random.Generator, rate: float,
                       config: "RecurrentArchConfig | None" = None, frame_bits: int = 0) -> None:
	"""Per-entry resample of a sampled suffix, avoiding duplicates within it.
	`config.conn_mutation_scope` constrains where each replacement may land
	(window / feature of the ORIGINAL bit); free (or no config) is the legacy
	draw via the legacy symbol, bit-identical."""
	if not suffix or space <= 0:
		return
	seed = int(rng.integers(0, 1 << 63))
	scope = _CONN_SCOPE_CODE.get(getattr(config, "conn_mutation_scope", "free") or "free", 0)
	if scope:
		suffix[:] = [int(b) for b in ra.arch_resample_suffix_scoped(
			[int(b) for b in suffix], int(space), float(rate), scope,
			int(frame_bits), int(config.bits_per_feature), seed, 0, 0, 0)]
	else:
		suffix[:] = [int(b) for b in ra.arch_resample_suffix(
			[int(b) for b in suffix], int(space), float(rate), seed, 0, 0, 0)]


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


def _sample_min_per_cluster(space: int, width: int, bpf: int, m: int,
                            rng: np.random.Generator) -> list[int]:
	"""One fresh suffix under MIN_PER_CLUSTER(m): choose width//m features, give
	each m distinct thresholds, and DONATE the width%m remainder one bit each to
	already-chosen features (never opening a new feature below m — that is the
	whole point of the rule). Falls back to spread when the space has no feature
	structure or the request is unsatisfiable (width > features*bpf).

	m == 1 is the COVERAGE end of the same rule (16/08/2026, Luiz): every feature
	gets >= 1 threshold, so a b == num_features neuron sees each feature exactly
	once. This is NOT what spread does — spread draws width bits uniformly from
	features*bpf, so at 18 features x 8 bits a b=18 neuron covers only ~12.0 of
	18 features in expectation (33.4% of (neuron,feature) pairs get NOTHING; the
	connectivity forensics measured ~14% at b=30). m=1 used to fall through to
	spread here, which made MIN_PER_CLUSTER(1) silently a no-op and left the
	coverage hypothesis untestable.

	Implementation lives in Rust (arch_ops::sample_min_per_cluster, ported
	16/08/2026 per rust-first) including the fallback-to-spread decision."""
	seed = int(rng.integers(0, 1 << 63))
	return [int(b) for b in ra.arch_sample_min_per_cluster(
		int(space), int(width), int(bpf), int(m), seed, 0, 0, 0)]


def _sample_framed1(space: int, width: int, bpf: int, k: int,
                    rng: np.random.Generator, slot: int | None = None) -> list[int]:
	"""One fresh suffix under FRAMED1 (16/08/2026, Luiz's temporal-coverage idea).

	Arm D showed that spreading a neuron's `width` connections across the whole
	K-frame window costs ~k-fold per-frame coverage and buys nothing at 4 ms
	(steady 1.34 deg -> 10.17). FRAMED1 spends the budget the other way: each
	neuron picks ONE frame and covers it COMPLETELY (min1 within that frame, so
	at width == num_features every feature gets exactly one threshold), and the
	POPULATION covers time.

	The frame comes either from the caller (`slot` — genome init passes an EXACT
	per-motor-block quota schedule, see _framed1_slot_schedule) or, when slot is
	None (neurogenesis appends one neuron with no population context), drawn with
	RECENCY weights 2^slot — slot 0 is the oldest frame, slot k-1 the current one
	— so at k=4 the split is 8:4:2:1, i.e. 128/64/32/16 of 240 neurons.

	NOTE what this is and is not: each neuron sees ONE frame and the decode is a
	SUM, so no neuron and no motor computes a temporal DIFFERENCE. Temporal
	structure has to come from the learned cell values (DAgger gives each neuron
	the best response for its own frame's pattern), not from decode arithmetic.

	Implementation lives in Rust (arch_ops::sample_framed1, ported 16/08/2026
	per rust-first) including every degenerate-case decision."""
	seed = int(rng.integers(0, 1 << 63))
	return [int(b) for b in ra.arch_sample_framed1(
		int(space), int(width), int(bpf), int(k),
		-1 if slot is None else int(slot), seed, 0, 0, 0)]


def _framed1_slot_schedule(n_neurons: int, k: int, quantum: int,
                           rng: np.random.Generator) -> list[int]:
	"""EXACT frame-slot quotas for a fresh framed1 population (16/08/2026, Luiz's
	round-2 spec: window0=128n/64/32/16 at 240n — deterministic counts, not the
	weighted draw's 133/56/36/15).

	Output neurons are laid out motor-major (quantum = num_motors), so the quota
	is applied PER MOTOR BLOCK and shuffled within each block: every motor gets
	its exact proportional share of every frame (e.g. 32/16/8/4 of its 60 levels
	at k=4), and no index-keyed structure can leave a motor commanding on stale
	state. Largest-remainder rounding, remainders biased toward NEWER frames.
	Neurogenesis appends still use the per-neuron weighted draw (no population
	context there).

	Implementation lives in Rust (arch_ops::framed1_slot_schedule, ported
	16/08/2026 per rust-first)."""
	seed = int(rng.integers(0, 1 << 63))
	return [int(s) for s in ra.arch_framed1_slot_schedule(
		int(n_neurons), int(k), int(quantum), seed, 0, 0, 0)]


def _fresh_output_suffix(space: int, width: int, rng: np.random.Generator,
                         config: "RecurrentArchConfig | None") -> list[int]:
	"""Policy dispatch for a fresh OUTPUT-layer map. None/spread = the exact
	legacy draw, so every banked result reproduces bit-identically."""
	if config is not None and config.conn_policy == "framed1":
		return _sample_framed1(space, width, config.bits_per_feature,
		                       config.input_window_k, rng)
	if config is not None and config.conn_policy == "min_per_cluster":
		return _sample_min_per_cluster(space, width, config.bits_per_feature,
		                               config.conn_policy_min, rng)
	return _sample_distinct(space, width, rng)


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
	# Rust: the move loop is data-dependent and per-neuron; see arch_ops.
	flat: list[int] = []
	offsets: list[int] = [0]
	for suf in sampled:
		flat.extend(int(b) for b in suf)
		offsets.append(len(flat))
	seed = int(rng.integers(0, 1 << 63))
	out = ra.arch_rebalance_features(
		flat, offsets, int(space), int(frame_bits), int(bpf), float(ratio), seed, 0, 0, 0)
	for ni, suf in enumerate(sampled):
		suf[:] = [int(b) for b in out[offsets[ni]:offsets[ni + 1]]]


# ---- memory payload: the optional "content" dimension ------------------------

# QSR value 2 = bits 0b10 = the EMPTY/hover decode (QSR_WEIGHTS[2] = 0.75). Used
# as the neutral branch a freshly-added state neuron's prefix pair defaults to.
NEUTRAL_PAIR = 2


# STAGE D2 (21/07/2026): the MemoryPayload wrapper class is DELETED — the name
# is a re-export of the Rust GenomeCells class, which carries the identical API
# (4-arg constructor over (neuron, address) pair lists + value lists;
# state_universe/output_universe as on-demand (N,2) uint64 numpy views;
# state_values/output_values as uint8 views; clone / to_triples / from_triples /
# fingerprint / cell_count; the in-place remap + GA-MEMORY operators; and the
# pack_int_columns-compatible export_packed / from_packed). ONE implementation,
# in Rust; cells never exist as Python objects except on-demand views.
# Cells are u64-keyed end-to-end: an address >= 2^64 raises OverflowError.
MemoryPayload = ra.GenomeCells


# ---- best-effort cell remap (high-fidelity policy) ---------------------------
# Address model (compute_address_sparse, MSB-first): A = P·2^w + S, prefix P in
# the HIGH bits, sampled suffix S in the LOW w bits. Every operator adds/drops at
# the tail, so changes land on known bit fields.
#
# STAGE B: the functions below are the REFERENCE SPEC, no longer the live path.
# The operators call the bit-exact Rust ports on the GenomeCells handle
# (cell_remap.rs); tests/test_cell_remap_parity.py holds the two equal. Keep
# these in sync with any semantic change, or the parity suite goes blind.

def _majority(vals: list[int]) -> int:
	"""Most common QSR value among colliders; ties → lower value (deterministic).

	The int() coercion is load-bearing: MemoryPayload stores values as a uint8
	buffer, so colliders arrive as numpy uint8 scalars and the `-kv[0]` tie-break
	WRAPS instead of going negative (-1 → 255). That ranked ties 1 > 2 > 3 > 0,
	the inverse of the documented order, and emitted an overflow RuntimeWarning."""
	from collections import Counter
	counts = Counter(int(v) for v in vals)
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
		if config is not None and config.conn_policy == "framed1":
			# EXACT per-motor-block frame quotas (Luiz 16/08): init knows the whole
			# population, so the 8:4:2:1 recency split is deterministic here, not
			# an expectation. Neurogenesis appends keep the weighted draw.
			slots = _framed1_slot_schedule(output_neurons, config.input_window_k,
			                               shape.output_quantum, rng)
			os = [_sample_framed1(shape.output_input_space, output_suffix,
			                      config.bits_per_feature, config.input_window_k,
			                      rng, slot=slots[j])
			      for j in range(output_neurons)]
		else:
			os = [_fresh_output_suffix(shape.output_input_space, output_suffix, rng, config)
			      for _ in range(output_neurons)]
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
			from wnn.ram.strategies.phased.packing import packed_payload_from_b64
			# Rust-side packing (Stage D1): byte-identical to
			# pack_int_columns(to_triples(), 3), without the ~1.3 GB of transient
			# 3-int tuples a big genome cost per save (x50 genomes per stage
			# checkpoint — the 12/07 OOM's neighbourhood).
			(s_b64, s_n, s_wide), (o_b64, o_n, o_wide) = self.cells.export_packed()
			cells_payload = {"state": packed_payload_from_b64(s_b64, s_n, 3, s_wide),
			                 "output": packed_payload_from_b64(o_b64, o_n, 3, o_wide)}
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
			from wnn.ram.strategies.phased.packing import is_packed, packed_b64_parts
			if isinstance(c, dict) and is_packed(c.get("state")) and is_packed(c.get("output")):
				# Packed → Rust decode, no triple materialisation (Stage D1).
				cells = ra.GenomeCells.from_packed(
					*packed_b64_parts(c["state"]), *packed_b64_parts(c["output"]))
			else:
				if isinstance(c, dict):  # packed-dict shape with legacy list values
					from wnn.ram.strategies.phased.packing import unpack_int_columns
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
				self.cells.drop_changed_state(sorted(changed_s))
			if changed_o:
				self.cells.drop_changed_output(sorted(changed_o))
		return self

	def _cells_at_budget(self, config: RecurrentArchConfig) -> bool:
		"""True when the carried Lamarckian cells already meet the per-genome
		max_cells budget — structural GROWS are then suppressed (shrinks and
		rewires stay allowed, so selection can still slim the genome). Genomes
		without carried cells (paradigm A: cells trained at eval) can't be
		measured here and are exempt."""
		return self.cells is not None and self.cells.cell_count() >= config.max_cells

	def _grow_within_budget(self, config: RecurrentArchConfig, delta: int,
	                        state_layer: bool) -> int:
		"""Largest grow ≤ delta whose POST-grow cell count still fits max_cells.

		Only active under config.strict_cell_budget. A bits-grow of d replicates
		the grown layer ×2^d (the dominant balloon multiplier), so gating on the
		PRE-grow count — as _cells_at_budget does — lets a genome just under the
		line land far above it. Uses cells.counts(), which is O(1) per-layer and
		already exposed, so the projection is exact rather than conservative;
		a total-count approximation would over-suppress, and the config comment
		records where that leads (100k froze all grows → population collapsed to
		one shape). Adds no RNG draw, so the deterministic stream is untouched.
		"""
		if delta <= 0 or self.cells is None or not config.strict_cell_budget:
			return delta
		state_cells, output_cells = self.cells.counts()
		for d in range(delta, 0, -1):
			projected = ((state_cells << d) + output_cells if state_layer
			             else state_cells + (output_cells << d))
			if projected <= config.max_cells:
				return d
		return 0

	def _mutate_neurons(self, rate: float, config: RecurrentArchConfig,
	                    rng: np.random.Generator) -> "RecurrentArchGenome":
		"""State + output neurogenesis. Survivors keep their suffixes verbatim;
		growth appends fresh tail blocks (small-neighborhood rule). Cells, if
		present, are remapped: STATE neuro reshapes the prefix in BOTH layers;
		OUTPUT neuro keeps survivors verbatim and drops removed blocks."""
		g = self.clone()
		at_budget = self._cells_at_budget(config)
		saturation = self.pressure[0] if self.pressure else 0
		# STATE neurogenesis (memory capacity): reshapes the prefix globally.
		# Phase 5c (DAMPED §11b): under SATURATION pressure (the splitting trainer
		# found conflicts it could not resolve for lack of free neurons), bias toward
		# GROWTH — but PROBABILISTICALLY, not on every offspring. grow_p scales with
		# saturation; only when the gate fires do we bias the delta to a small grow.
		# (Old behavior force-grew every genome whenever saturation>0, churning
		# doomed over-grown offspring that selection then had to prune.)
		grow_p = min(1.0, rate + saturation * config.saturation_grow_gain)
		# Every draw here comes from the shared counter RNG (Rust); the numpy call
		# supplies only the per-call seed, so no random number is generated in
		# Python. Sub-draw indices are distinct so the gates cannot alias.
		_sd = int(rng.integers(0, 1 << 63))
		if ra.counter_rng_uniform(_sd, 0, 0, 0, 0, 0) < grow_p and config.state_neuron_delta > 0:
			_d = config.state_neuron_delta
			delta = int(ra.counter_rng_below(2 * _d + 1, _sd, 0, 0, 0, 1, 0)) - _d
			if saturation > 0 and delta <= 0 and not at_budget:
				delta = 1
			if at_budget:                     # cell budget: growth off, shrink allowed
				delta = min(delta, 0)
			target = min(config.max_state_neurons, max(config.min_state_neurons, g.state_neurons + delta))
			g.set_state_neurons(target, rng)
		# OUTPUT neurogenesis (resolution): whole blocks, in units of output_quantum.
		q = g.shape.output_quantum
		if ra.counter_rng_uniform(_sd, 0, 0, 0, 2, 0) < rate and config.output_block_delta > 0 and q > 0:
			_ob = config.output_block_delta
			delta_blocks = int(ra.counter_rng_below(2 * _ob + 1, _sd, 0, 0, 0, 3, 0)) - _ob
			if at_budget:                     # cell budget: growth off, shrink allowed
				delta_blocks = min(delta_blocks, 0)
			lo = max(1, config.min_output_neurons // q)
			hi = max(lo, config.max_output_neurons // q)
			cur_blocks = g.output_neurons // q
			g.set_output_neurons(min(hi, max(lo, cur_blocks + delta_blocks)) * q, rng, config)
		return g

	def _mutate_bits(self, rate: float, config: RecurrentArchConfig,
	                 rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Synaptogenesis: grow/shrink sampled-suffix width uniformly per layer.
		Cells remap by replicate-on-grow / majority-collapse-on-shrink (LSBs).
		At the max_cells budget the ×2^d replicate-on-grow is the dominant
		balloon multiplier, so grows clamp to shrink-only."""
		g = self.clone()
		_sd = int(rng.integers(0, 1 << 63))   # seed only; the draws are Rust-side
		_sfd = config.suffix_delta
		# R9: re-check the budget on the MUTATED intermediate g before EACH grow. A
		# single pre-computed at_budget let a state-suffix grow that reached budget
		# still permit the output-suffix grow in the SAME call — a one-step overshoot
		# up to ×2^suffix per layer (the ×2^d replicate-on-grow is the dominant balloon
		# multiplier). _cells_at_budget reads cell_count only (no RNG draw), so moving
		# it here does not perturb the deterministic RNG stream.
		if ra.counter_rng_uniform(_sd, 0, 0, 0, 0, 0) < rate and _sfd > 0:
			delta = int(ra.counter_rng_below(2 * _sfd + 1, _sd, 0, 0, 0, 1, 0)) - _sfd
			if g._cells_at_budget(config):    # cell budget: growth off, shrink allowed
				delta = min(delta, 0)
			else:                             # strict: clamp so POST-grow still fits
				delta = g._grow_within_budget(config, delta, True)
			cap = min(config.max_suffix, g.shape.state_input_space)
			g.set_state_suffix(min(cap, max(config.min_suffix, g.state_suffix_width + delta)), rng)
		if ra.counter_rng_uniform(_sd, 0, 0, 0, 2, 0) < rate and _sfd > 0:
			delta = int(ra.counter_rng_below(2 * _sfd + 1, _sd, 0, 0, 0, 3, 0)) - _sfd
			if g._cells_at_budget(config):    # re-checked AFTER the state grow above
				delta = min(delta, 0)
			else:                             # strict: clamp so POST-grow still fits
				delta = g._grow_within_budget(config, delta, False)
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
		# OUTPUT maps honour the mutation scope; frame_bits identifies windows
		# (space = k * frame_bits, k threaded like bits_per_feature).
		_k = max(1, getattr(config, "input_window_k", 1)) if config else 1
		_fb = g.shape.output_input_space // _k
		for suffix in g.output_sampled:
			_resample_in_place(suffix, g.shape.output_input_space, rng, rate,
			                   config=config, frame_bits=_fb)
		# Phase 5c: CONNECTIVITY pressure — route a state neuron to each WISHED
		# sensor bit (a separator the trainer wanted but no neuron observed). Inject
		# the wish into a random state-neuron suffix (replacing a random entry); the
		# wish positions are state-input indices (< state_input_space) by construction.
		wish = self.pressure[1] if self.pressure else ()
		_wsd = int(rng.integers(0, 1 << 63))
		for wi, wb in enumerate(wish):
			if 0 <= wb < g.shape.state_input_space and g.state_sampled:
				ni = int(ra.counter_rng_below(len(g.state_sampled), _wsd, 0, 0, 0, wi, 0))
				suffix = g.state_sampled[ni]
				if wb not in suffix:
					suffix[int(ra.counter_rng_below(len(suffix), _wsd, 0, 0, 1, wi, 0))] = wb
		if g.cells is not None:
			changed_s = {i for i in range(len(g.state_sampled)) if g.state_sampled[i] != before_s[i]}
			changed_o = {i for i in range(len(g.output_sampled)) if g.output_sampled[i] != before_o[i]}
			if changed_s:
				g.cells.drop_changed_state(sorted(changed_s))
			if changed_o:
				g.cells.drop_changed_output(sorted(changed_o))
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
		# Rust (ram_core counter RNG). The per-cell Python loop this replaced ran
		# ~10^9 interpreter iterations per production run. One numpy draw seeds the
		# call so the caller's rng chain still determines the outcome.
		seed = int(rng.integers(0, 1 << 63))
		# Same memory_ops draws (seed, gen=0, genome=0, LAYER_*), in place on the
		# handle — the whole-layer Vec<u8> round-trip + list() re-boxing is gone.
		g.cells.mutate_values(quad, rate, seed)
		return g

	def _remap_state_neuro(self, k: int, sw: int, ow: int, removed_floor: int) -> None:
		"""Remap cells through a STATE-neurogenesis of +k (or -k) neurons. The
		prefix grows/shrinks in BOTH layers; removed state neurons' own cells go."""
		# Rust compound op: grow both layers' prefixes, or drop-removed + shrink.
		self.cells.state_neuro(k, sw, ow, self.shape.prefix_factor, removed_floor)

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

	def set_output_neurons(self, target: int, rng: np.random.Generator,
	                       config: "RecurrentArchConfig | None" = None) -> None:
		"""OUTPUT neurogenesis to `target` neurons (multiple of output_quantum).
		Survivors keep cells verbatim; removed tail blocks' cells are dropped.
		`config` carries the connection-creation policy for the FRESH maps grown
		neurons receive (None = legacy spread)."""
		if target == self.output_neurons:
			return
		if self.cells is not None and target < self.output_neurons:
			self.cells.drop_output_neurons_ge(target)
		self._resize_output_neurons(target, rng, config)

	def set_state_suffix(self, target: int, rng: np.random.Generator) -> None:
		"""Synaptogenesis: set state sampled-suffix width to `target`; remap cells."""
		old = self.state_suffix_width
		if target == old:
			return
		for suffix in self.state_sampled:
			_resize_suffix(suffix, self.shape.state_input_space, target, rng)
		if self.cells is not None:
			self.cells.remap_bits_state(target - old)

	def set_output_suffix(self, target: int, rng: np.random.Generator) -> None:
		"""Synaptogenesis: set output sampled-suffix width to `target`; remap cells."""
		old = self.output_suffix_width
		if target == old:
			return
		for suffix in self.output_sampled:
			_resize_suffix(suffix, self.shape.output_input_space, target, rng)
		if self.cells is not None:
			self.cells.remap_bits_output(target - old)

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
			# Rust compound op: drop neuron k's cells + reindex higher state
			# neurons down + excise the pf-bit window from BOTH layers.
			self.cells.remove_state_neuron(k, p_lsb_s, p_lsb_o, pf)

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
				self.cells.drop_changed_state(sorted(state_changes))
			if output_changes:
				self.cells.drop_changed_output(sorted(output_changes))

	# ---- resize primitives (in place; caller has already cloned) ------------

	def _resize_state_neurons(self, target: int, rng: np.random.Generator) -> None:
		width = self.state_suffix_width or 1
		if target < self.state_neurons:
			del self.state_sampled[target:]
		else:
			for _ in range(target - self.state_neurons):
				self.state_sampled.append(_sample_distinct(self.shape.state_input_space, width, rng))
		self.state_neurons = target

	def _resize_output_neurons(self, target: int, rng: np.random.Generator,
	                           config: "RecurrentArchConfig | None" = None) -> None:
		width = self.output_suffix_width or 1
		if target < self.output_neurons:
			del self.output_sampled[target:]
		else:
			for _ in range(target - self.output_neurons):
				self.output_sampled.append(
					_fresh_output_suffix(self.shape.output_input_space, width, rng, config))
		self.output_neurons = target

	# ---- crossover (handles parents of DIFFERENT shape) ---------------------

	@staticmethod
	def crossover(a: "RecurrentArchGenome", b: "RecurrentArchGenome",
	              rng: np.random.Generator) -> "RecurrentArchGenome":
		"""Whole-block uniform crossover. The child inherits ONE parent's shape
		(counts + suffix widths); for each block it then takes the other parent's
		suffix only when shape-compatible, else keeps the shape-parent's. This
		guarantees a structurally valid child even when a and b differ in size."""
		shape_parent, other = (a, b) if ra.counter_rng_uniform(
			int(rng.integers(0, 1 << 63)), 0, 0, 0, 0, 0) < 0.5 else (b, a)
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

		# Parent coins for every neuron position, in ONE Rust call (counter RNG).
		# Sized for the larger layer so both calls below index it safely.
		_pick = ra.arch_pick_mask(
			max(target_state_n, target_output_n, 1),
			int(rng.integers(0, 1 << 63)), 0, 0, 0)

		def _pick_or_resample(layer_a: list[list[int]], layer_b: list[list[int]],
		                      i: int, space: int, target_width: int) -> list[int]:
			"""Pick parent suffix at index i (uniform if both have it, else the
			one that does, else random sample), resize to target_width.
			Parent coins come from a precomputed Rust mask (counter RNG)."""
			a_has = i < len(layer_a)
			b_has = i < len(layer_b)
			if a_has and b_has:
				src = layer_a[i] if _pick[i % len(_pick)] else layer_b[i]
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
		seed = int(rng.integers(0, 1 << 63))
		# Keyed crossover in Rust, in place on the child's handle (the child
		# cloned `a`, so its universe/values ARE a's). Same memory_ops draws
		# (seed, gen=0, genome=0, LAYER_*) as the six-list marshalling this
		# replaces — see memory_ops::crossover_values_keyed for why the coin is
		# coordinate-indexed and universe overlap cannot shift the stream.
		child.cells.crossover_values_from(b.cells, seed)
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
		# Rust-side: duplicate-key, neuron-range, address-range and value checks in
		# one pass over the handle's columns (no numpy materialisation). Called per
		# offspring per generation from the mixed-GA strategy, so it must be cheap.
		try:
			self.cells.validate(
				self.state_neurons, self.state_bits_per_neuron,
				self.output_neurons, self.output_bits_per_neuron)
		except ValueError as e:
			raise AssertionError(str(e)) from None


def _mix_blocks(into: list[list[int]], other: list[list[int]], rng: np.random.Generator) -> None:
	"""For each block in `into`, with p=0.5 take `other`'s block of the same
	index — but only if it exists and has matching width (keeps suffix uniform)."""
	if not into:
		return
	width = len(into[0])
	# One Rust call for all block coins (counter RNG). The Python loop drew per
	# block; the guard is applied here so a skipped block consumes no decision —
	# with coordinates that is free, there is no stream to keep in step.
	seed = int(rng.integers(0, 1 << 63))
	take = ra.arch_pick_mask(len(into), seed, 0, 0, 0)
	for i in range(len(into)):
		if take[i] and i < len(other) and len(other[i]) == width:
			into[i] = list(other[i])


__all__ = ["RecurrentArchGenome", "RecurrentArchShape", "RecurrentArchConfig", "MemoryPayload"]
