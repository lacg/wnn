#!/usr/bin/env python3
"""E1 random immigrants — unit test for the offspring-pool injection.

Verifies (plan .claude/plans/controller_break_90_v2.md E1):
1. immigrant_fraction=0.0 (default) → offspring_generator NEVER calls
   create_random_genome (bit-identical legacy behavior).
2. immigrant_fraction=0.5 → a ~fraction of generated offspring are fresh
   random genomes (marked), the rest bred (tournament+crossover/mutate).
3. Immigrants flow through _generate_offspring's evaluation path like any
   offspring (they arrive as evaluated (genome, metrics) tuples).

Pure-Python: a dummy genome + a GenericGAStrategy subclass with a counting
create_random_genome; batch_evaluate_fn stubs metrics. No Rust, no sim.
"""

import sys

from wnn.ram.strategies.connectivity.framework.configs import GAConfig
from wnn.ram.strategies.connectivity.generic_ga import GenericGAStrategy


class DummyGenome:
	def __init__(self, tag: str):
		self.tag = tag  # 'seed' | 'bred' | 'immigrant'


class DummyGA(GenericGAStrategy):
	def __init__(self, config):
		super().__init__(config, seed=42)
		self.random_calls = 0

	def clone_genome(self, g):
		return DummyGenome("bred")

	def mutate_genome(self, g, rate):
		g.tag = "bred" if g.tag != "immigrant" else g.tag
		return g

	def crossover_genomes(self, p1, p2):
		return DummyGenome("bred")

	def create_random_genome(self):
		self.random_calls += 1
		return DummyGenome("immigrant")


from wnn.ram.metrics import IDSMetrics as Metrics


def _M():
	"""Canonical Metrics stub for the framework's viability/fitness path."""
	return Metrics(ce=1.0, acc=0.5)


def run_offspring(fraction: float, n: int = 200):
	import random
	cfg = GAConfig(population_size=8, generations=1, immigrant_fraction=fraction)
	ga = DummyGA(cfg)
	ga._rng = random.Random(42)  # optimize() normally seeds this; we call _generate_offspring directly
	tags = []

	# Explicit params (no **kwargs — house rule): _build_viable_population passes
	# min_accuracy always, generation/total_generations when provided.
	def batch_eval(genomes, min_accuracy=0.0, generation=None, total_generations=None):
		tags.extend(g.tag for g in genomes)
		return [_M() for _ in genomes]

	ga._batch_evaluate_fn = batch_eval
	ga._evaluate_fn = lambda g: _M()
	population = [(DummyGenome("seed"), float(i), None) for i in range(8)]
	off = ga._generate_offspring(population, n, threshold=0.0, generation=0)
	return ga, off, tags


def main() -> int:
	failures = 0

	# 1. default off → zero immigrant calls
	ga, off, _ = run_offspring(0.0)
	if ga.random_calls != 0:
		print(f"FAIL: fraction=0.0 called create_random_genome {ga.random_calls}x (want 0)")
		failures += 1
	else:
		print(f"PASS: fraction=0.0 → 0 immigrant calls, {len(off)} offspring bred")

	# 2. fraction=0.5 → roughly half immigrants (binomial n=200 p=0.5: [70,130] is >6 sigma)
	ga, off, tags = run_offspring(0.5)
	imm = sum(1 for t in tags if t == "immigrant")
	if not (70 <= ga.random_calls <= 130):
		print(f"FAIL: fraction=0.5 → {ga.random_calls} immigrant calls of ~200 (want ~100)")
		failures += 1
	else:
		print(f"PASS: fraction=0.5 → {ga.random_calls} immigrant calls (~50% of 200)")

	# 3. immigrants are evaluated like any offspring (present in the eval batch)
	if imm == 0:
		print("FAIL: no immigrants reached the evaluation batch")
		failures += 1
	else:
		print(f"PASS: {imm} immigrants flowed through the evaluation path")

	print("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
	return 1 if failures else 0


if __name__ == "__main__":
	sys.exit(main())
