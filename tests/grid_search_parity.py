"""
grid_search_parity — byte-parity gate for the IDS GridSearchStrategy refactor (WS4a).

The IDS grid is being refactored to run on the shared `GenericGridSearch` core
(the same template the controller's `ControllerGridSearch` uses) instead of its
own hand-rolled top-K / expand / trim pipeline. That grid seeds GA-Neurons, which
feeds the paper campaign — so the refactor MUST produce the byte-identical seed
population, best genome, and OptimizerResult scalars.

How it works (golden-capture): a deterministic fake batch-evaluator + a fixed seed
make `GridSearchStrategy.optimize()` fully reproducible (see the two determinism
notes below). The first run writes a fingerprint golden to
`tests/fixtures/grid_parity_golden.json`; every later run re-fingerprints and
asserts equality. Capture the golden on the PRE-refactor code, commit it, then the
post-refactor run must match.

Determinism requirements (both encoded here):
  1. `WNN_GRID_SEARCH_PARALLEL=1` — the real grid builds `results` in thread
     completion order, so fitness TIES break non-deterministically under
     concurrency. Sequential eval makes completion order == enumeration order.
  2. Fixed `seed` on the strategy (drives connection RNG) + a fake evaluator whose
     metric is a pure function of genome content (so eval never depends on timing).

Run:  WNN_GRID_SEARCH_PARALLEL=1 python tests/grid_search_parity.py
Recapture (only when the grid algorithm INTENTIONALLY changes):
      rm tests/fixtures/grid_parity_golden.json && python tests/grid_search_parity.py
"""

from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

# Deterministic sequential grid — must be set before importing the strategy path.
os.environ.setdefault("WNN_GRID_SEARCH_PARALLEL", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "wnn"))

from wnn.ram.fitness import FitnessCalculatorType
from wnn.ram.metrics import Metrics
from wnn.ram.strategies.connectivity.grid_search import GridSearchConfig, GridSearchStrategy

_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "grid_parity_golden.json"


class _FakeBatchEvaluator:
	"""A deterministic stand-in for the Rust BitwiseEvaluator.

	`evaluate_batch` returns a Metrics whose values are a pure function of the
	genome's content (neurons, bits, connections) via CRC32 — so the SAME genome
	always scores the SAME, and distinct genomes get well-spread, reproducible
	metrics that make the fitness ranking non-trivial. No Rust, no GPU, no timing
	dependence → the grid's output depends only on the seed + config.
	"""

	train_rows_hint = None  # small-dataset path (concurrent branch gated off anyway)

	def random_train_idx(self, rng) -> int:
		# The real evaluator draws a subset index from the strategy RNG; mirror the
		# draw so the RNG stream advances identically (parity of later draws).
		return rng.randint(0, 4)

	def evaluate_batch(self, genomes, train_subset_idx=None):
		return [self._score(g) for g in genomes]

	@staticmethod
	def _score(genome) -> Metrics:
		payload = repr((list(genome.neurons_per_cluster or []),
		                list(genome.bits_per_neuron or []),
		                list(genome.connections or []))).encode()
		h = zlib.crc32(payload)
		ce = 0.5 + (h % 10_000) / 10_000.0            # [0.5, 1.5)
		acc = (h % 7_919) / 7_919.0                   # [0, 1)
		f1 = ((h >> 3) % 6_997) / 6_997.0             # [0, 1)
		fpr = ((h >> 7) % 5_003) / 5_003.0            # [0, 1)
		return Metrics(ce=ce, acc=acc, f1=f1, fpr=fpr, bit_accuracy=acc)


def _genome_fp(g) -> dict:
	"""Order-and-content fingerprint of one genome. Connections are CRC32'd (not
	stored raw) — identical connections ⇒ identical CRC, so parity is preserved
	while the golden stays tiny (the raw arrays are ~2700 ints/genome)."""
	conns = None if g.connections is None else zlib.crc32(repr(list(g.connections)).encode())
	return {
		"neurons_per_cluster": list(g.neurons_per_cluster or []),
		"bits_per_neuron": list(g.bits_per_neuron or []),
		"connections_crc": conns,
		"connections_len": None if g.connections is None else len(g.connections),
	}


def _metric_fp(m) -> dict:
	return {"ce": round(m.ce, 10), "acc": round(m.acc, 10),
	        "f1": None if m.f1 is None else round(m.f1, 10),
	        "fpr": None if m.fpr is None else round(m.fpr, 10)}


def _run_grid() -> dict:
	"""Run the grid with a fixed seed + fake evaluator; return a full fingerprint of
	the results-affecting output (seed population + best + OptimizerResult scalars)."""
	cfg = GridSearchConfig(
		num_clusters=1,
		neurons_grid=[50, 100, 150],
		bits_grid=[14, 16, 18],
		top_k=4,
		population_size=12,
		total_input_bits=200,
		fitness_calculator_type=FitnessCalculatorType.HARMONIC_RANK,
		fitness_weight_ce=1.0,
		fitness_weight_acc=1.0,
	)
	strat = GridSearchStrategy(cfg, batch_evaluator=_FakeBatchEvaluator(), seed=13337)
	res = strat.optimize()
	return {
		"seed_population": [_genome_fp(g) for g in res.final_population],
		"population_metrics": [_metric_fp(m) for m in res.population_metrics],
		"best_genome": _genome_fp(res.best_genome),
		"scalars": {
			"initial_fitness": round(res.initial_fitness, 10),
			"final_fitness": round(res.final_fitness, 10),
			"final_accuracy": round(res.final_accuracy, 10),
			"initial_accuracy": round(res.initial_accuracy, 10),
			"iterations_run": res.iterations_run,
			"pop_size": len(res.final_population),
		},
	}


def main() -> int:
	fp = _run_grid()
	if not _GOLDEN.exists():
		_GOLDEN.parent.mkdir(parents=True, exist_ok=True)
		_GOLDEN.write_text(json.dumps(fp, indent=2, sort_keys=True))
		print(f"[grid-parity] GOLDEN CAPTURED → {_GOLDEN}")
		print(f"[grid-parity]   seed pop = {fp['scalars']['pop_size']} genomes, "
		      f"final_fitness={fp['scalars']['final_fitness']}")
		print("[grid-parity]   (re-run after the refactor to check parity)")
		return 0
	golden = json.loads(_GOLDEN.read_text())
	if fp == golden:
		print(f"[grid-parity] ✅ PARITY OK — {fp['scalars']['pop_size']} seed genomes "
		      f"+ metrics + best + scalars byte-identical to golden.")
		return 0
	# Pinpoint the first divergence for a useful failure.
	print("[grid-parity] ❌ PARITY MISMATCH vs golden:")
	for key in ("scalars", "best_genome"):
		if fp[key] != golden.get(key):
			print(f"  [{key}] new={fp[key]}")
			print(f"  [{key}] gold={golden.get(key)}")
	if fp["seed_population"] != golden.get("seed_population"):
		g = golden.get("seed_population", [])
		for i, (a, b) in enumerate(zip(fp["seed_population"], g)):
			if a != b:
				print(f"  seed_population[{i}] new={a}")
				print(f"  seed_population[{i}] gold={b}")
				break
		if len(fp["seed_population"]) != len(g):
			print(f"  seed_population length: new={len(fp['seed_population'])} gold={len(g)}")
	if fp["population_metrics"] != golden.get("population_metrics"):
		gm = golden.get("population_metrics", [])
		for i, (a, b) in enumerate(zip(fp["population_metrics"], gm)):
			if a != b:
				print(f"  population_metrics[{i}] new={a} gold={b}")
				break
	return 1


if __name__ == "__main__":
	sys.exit(main())
