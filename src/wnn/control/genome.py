"""FiniteStateGenome — the evolvable connectivity of a coherent-FSM controller.

WHAT IT IS
----------
The GA genome for the drone-attitude controller. It carries ONLY the evolvable
part of the policy: the connectivity (which input bits each state/output neuron
reads). The cells (QSR memory contents) are NOT in the genome — they are a
deterministic function of the genome, produced per-evaluation by the inner
trainer (reward-gated imitation for C1, reinforce-own-wins for C2). The
thresholds are fit once from PID rollouts and held fixed. (Paradigm "GA-Memory"
/ B would instead put the cells IN a genome — a different class.)

THE INVARIANT (why it is its OWN genome class, not a ClusterGenome)
-------------------------------------------------------------------
For the network to behave as ONE finite-state machine rather than N disjoint
mini-automata, every state and output neuron must observe the FULL recurrent
state (the DFA argument in project_controller_state). So each neuron's
connection block is [forced full-state bits | sampled input bits], and the GA
is allowed to mutate ONLY the sampled-input suffix. The forced state-bit prefix
is physically outside the mutable range, so EVERY individual the GA loop ever
sees is a valid coherent FSM *by construction* — no repair pass. That guarantee
is the reason this is a distinct genome type: the invariant lives in mutate/
crossover, which the GA framework calls as black boxes and trusts.

Layout (matches evaluator.random_connectivity):
  state neuron block  (len = state_bits_per_neuron):
      [ prefix_factor*state_neurons forced state-idx | (rest) sampled sensor_window bits ]
  output neuron block (len = output_bits_per_neuron):
      [ prefix_factor*state_neurons forced state-idx | (rest) sampled sensor_frame bits ]
  (prefix_factor = 1 since the 08/06/2026 1-bit MSB-only state migration; was 2.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluator import ControllerSpec, NUM_FEATURES, random_connectivity


@dataclass
class _Layout:
	"""Connectivity layout constants derived from a ControllerSpec."""
	n_state: int
	state_bits: int           # forced prefix length = 2 * n_state
	sensor_window: int        # state-layer input space (K * F * b)
	sensor_frame: int         # output-layer input space (F * b)
	sbpn: int                 # state_bits_per_neuron
	obpn: int                 # output_bits_per_neuron
	num_out: int              # num_motors * levels_per_motor

	@classmethod
	def of(cls, spec: ControllerSpec) -> "_Layout":
		n_state = spec.state_neurons
		return cls(
			n_state=n_state,
			state_bits=2 * n_state,
			sensor_window=spec.input_window_k * NUM_FEATURES * spec.bits_per_feature,
			sensor_frame=NUM_FEATURES * spec.bits_per_feature,
			sbpn=spec.state_bits_per_neuron,
			obpn=spec.output_bits_per_neuron,
			num_out=spec.num_motors * spec.levels_per_motor,
		)


def _resample_block_inputs(
	conn: list[int], base: int, n_forced: int, n_total: int,
	input_space: int, rng: np.random.Generator, rate: float,
) -> None:
	"""Resample the sampled-input suffix of one neuron's connection block, IN
	PLACE. Entries [base, base+n_forced) (forced full-state bits) are never
	touched. New bits avoid duplicates within the block's input set."""
	lo, hi = base + n_forced, base + n_total
	if hi <= lo:
		return
	used = set(conn[lo:hi])
	for j in range(lo, hi):
		if rng.random() < rate:
			used.discard(conn[j])
			for _ in range(8):  # a few tries to find an unused bit
				cand = int(rng.integers(0, input_space))
				if cand not in used:
					conn[j] = cand
					used.add(cand)
					break


@dataclass
class FiniteStateGenome:
	"""GA genome = the two layers' connectivity, with the full-state invariant.

	Only `state_connections` / `output_connections` are evolvable, and only
	their per-neuron sampled-input suffix is ever mutated.
	"""
	spec: ControllerSpec
	state_connections: list[int]
	output_connections: list[int]

	# ---- GA contract --------------------------------------------------------

	@classmethod
	def random(cls, spec: ControllerSpec, seed: int) -> "FiniteStateGenome":
		"""Fresh random genome (full-state forced; input bits sampled)."""
		sc, oc = random_connectivity(spec, seed=seed)
		return cls(spec=spec, state_connections=sc, output_connections=oc)

	def clone(self) -> "FiniteStateGenome":
		return FiniteStateGenome(
			spec=self.spec,
			state_connections=list(self.state_connections),
			output_connections=list(self.output_connections),
		)

	def mutate(self, rng: np.random.Generator, rate: float) -> "FiniteStateGenome":
		"""Return a mutated copy — INPUT bits only; forced state bits preserved."""
		L = _Layout.of(self.spec)
		sc = list(self.state_connections)
		oc = list(self.output_connections)
		for n in range(L.n_state):
			_resample_block_inputs(sc, n * L.sbpn, L.state_bits, L.sbpn, L.sensor_window, rng, rate)
		for n in range(L.num_out):
			_resample_block_inputs(oc, n * L.obpn, L.state_bits, L.obpn, L.sensor_frame, rng, rate)
		return FiniteStateGenome(spec=self.spec, state_connections=sc, output_connections=oc)

	@staticmethod
	def crossover(a: "FiniteStateGenome", b: "FiniteStateGenome",
	              rng: np.random.Generator) -> "FiniteStateGenome":
		"""Per-neuron uniform crossover: each neuron's WHOLE connection block
		comes from one parent or the other. Whole-block swap keeps the forced
		state-bit prefix intact, so the child is a valid coherent FSM."""
		L = _Layout.of(a.spec)
		sc: list[int] = []
		for n in range(L.n_state):
			src = a.state_connections if rng.random() < 0.5 else b.state_connections
			sc.extend(src[n * L.sbpn:(n + 1) * L.sbpn])
		oc: list[int] = []
		for n in range(L.num_out):
			src = a.output_connections if rng.random() < 0.5 else b.output_connections
			oc.extend(src[n * L.obpn:(n + 1) * L.obpn])
		return FiniteStateGenome(spec=a.spec, state_connections=sc, output_connections=oc)

	# ---- invariant self-check (used by tests) -------------------------------

	def state_bits_intact(self) -> bool:
		"""True iff every neuron's forced full-state prefix is the canonical
		state-index range (i.e. the GA never corrupted the recurrent wiring)."""
		L = _Layout.of(self.spec)
		state_state_idx = list(range(L.sensor_window, L.sensor_window + L.state_bits))
		out_state_idx = list(range(L.sensor_frame, L.sensor_frame + L.state_bits))
		for n in range(L.n_state):
			base = n * L.sbpn
			if self.state_connections[base:base + L.state_bits] != state_state_idx:
				return False
		for n in range(L.num_out):
			base = n * L.obpn
			if self.output_connections[base:base + L.state_bits] != out_state_idx:
				return False
		return True


__all__ = ["FiniteStateGenome"]
