"""
Smoke test for GenericSAStrategy (Garcia-2003 SA on the OptimizationTemplate
framework, 10/06/2026 port).

Toy problem: genomes are float vectors; energy (CE) = squared distance to a
target vector. SA must drive the best energy down monotonically and return
the framework contract (OptimizerResult with final_population of chain states).

Run: PYTHONPATH=src python tests/generic_sa_smoke.py
"""

import random
import sys

from wnn.ram.metrics import IDSMetrics, Metrics
from wnn.ram.strategies.connectivity import GenericSAStrategy, SAConfig

TARGET = [0.7, -0.3, 1.2, 0.0, -1.0]


def energy(g: list[float]) -> float:
	return sum((a - b) ** 2 for a, b in zip(g, TARGET))


def batch_evaluate(genomes: list[list[float]]) -> list[Metrics]:
	return [IDSMetrics(ce=energy(g), acc=1.0 / (1.0 + energy(g))) for g in genomes]


class ToySAStrategy(GenericSAStrategy[list]):
	def clone_genome(self, genome: list) -> list:
		return list(genome)

	def mutate_genome(self, genome: list, mutation_rate: float) -> tuple[list, None]:
		self._ensure_rng()
		g = list(genome)
		idx = self._rng.randrange(len(g))
		g[idx] += self._rng.gauss(0.0, 0.3)
		return g, None

	def genome_to_config(self, genome: list):
		return None  # toy genomes are not tracked


def main() -> int:
	rng = random.Random(42)
	initial = [rng.uniform(-2, 2) for _ in range(len(TARGET))]
	seed_pop = [[rng.uniform(-2, 2) for _ in range(len(TARGET))] for _ in range(6)]

	cfg = SAConfig(iterations=150, initial_temp=1.0, cooling_rate=0.95, chains=8,
	               mutation_rate=1.0, patience=50, check_interval=10)
	strategy = ToySAStrategy(config=cfg, seed=42, log_level=100)  # silence logs

	result = strategy.optimize(
		initial_genome=initial,
		initial_population=seed_pop,
		batch_evaluate_fn=batch_evaluate,
	)

	initial_e = energy(initial)
	best_e = energy(result.best_genome)
	checks = {
		"improved over initial": best_e < initial_e,
		"converged near target (CE<0.5)": best_e < 0.5,
		"final_population == chains": len(result.final_population) == cfg.chains,
		"iterations ran": result.iterations_run > 0,
		"history starts at iter 0": result.history[0][0] == 0,
		"history best is non-increasing": all(
			result.history[i][1] >= result.history[i + 1][1] - 1e-12
			for i in range(len(result.history) - 1)
		),
		"final fitness == best energy": abs(result.final_fitness - best_e) < 1e-9,
	}
	failed = [k for k, ok in checks.items() if not ok]
	for k, ok in checks.items():
		print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
	print(f"  initial CE={initial_e:.4f} → best CE={best_e:.4f} ({result.iterations_run} iters)")
	if failed:
		print(f"FAILED: {failed}")
		return 1
	print("ALL PASS")
	return 0


if __name__ == "__main__":
	sys.exit(main())
