"""IDS-76 flag-off pin: the IDS search results of whichever tree + wheel is on sys.path, as JSON.

Run it twice and compare the two files byte for byte:

  old (ABI 29 = installed wheel + live tree):
    PYTHONPATH=/Users/lacg/wnn/src:/Users/lacg/wnn/src/wnn python scripts/ids76_flag_off_pin.py /tmp/pin_old.json
  new (ABI 30 wheel + this tree, flag off):
    PYTHONPATH=<wheel dir>:<tree>/src:<tree>/src/wnn python scripts/ids76_flag_off_pin.py /tmp/pin_new.json

It never passes ids_fold_rotation, so it runs on both. Covered: the K-fold CV mean
(evaluate_batch), the Rust offspring search, immigrant scoring, and a multi-generation
ArchitectureGAStrategy.optimize run (final population + metrics + incumbent history),
on the control's training recipe (BINARY + vote 0.02, two clusters, rate 0.25,
blend_cluster crossover, desirability fitness, magnitude-aware patience) and on the
QUAD single-cluster default. Floats are written as float.hex (exact).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

os.environ["WNN_ORDER_INDEPENDENT_TRAIN"] = "1"
SEED = 20421


def dataset(n_train=1500, n_test=300, n_bits=8, seed=42):
	from wnn.ids.dataset import IDSDataset
	from wnn.ids.encoded_array import InMemoryEncoded
	from wnn.representations.thermometer import ThermometerEncoder

	rng = np.random.default_rng(seed)
	frame = lambda n: pd.DataFrame({c: rng.normal(size=n) for c in "abcdef"})
	df_train, df_test = frame(n_train), frame(n_test)
	enc = ThermometerEncoder(n_bits=n_bits)
	enc.fit(df_train)
	x_train, total_bits = enc.transform(df_train)
	x_test, _ = enc.transform(df_test)
	label = lambda d: ((d["a"] + 0.5 * d["b"] * d["c"] + 0.3 * rng.normal(size=len(d))) > 0.8).astype(np.int64).to_numpy()
	y_train, y_test = label(df_train), label(df_test)
	return IDSDataset(
		X_train=InMemoryEncoded(x_train, total_bits=total_bits), y_train_binary=y_train, y_train_multi=y_train.copy(),
		X_test=InMemoryEncoded(x_test, total_bits=total_bits), y_test_binary=y_test, y_test_multi=y_test.copy(),
		encoder=enc, category_names=["Normal", "Attack"], feature_names=list("abcdef"),
	)


RECIPES = {
	# the IDS-70 E control's training recipe (BINARY + majority vote 0.02, 2 clusters)
	"control": dict(single_cluster=False, memory_mode=3, write_rule="vote", vote_share=0.02, neuron_sample_rate=0.25),
	# the default QUAD single-cluster path
	"quad_sc": dict(single_cluster=True, memory_mode=2, write_rule="lattice", vote_share=None, neuron_sample_rate=1.0),
}


def evaluator(ds, recipe):
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	r = RECIPES[recipe]
	return IDSEvaluator(
		ds, classification="binary", num_parts=5, k_folds=5, kfold_per_gen=5, seed=SEED, run_seed=SEED,
		single_cluster=r["single_cluster"], memory_mode=r["memory_mode"], empty_value=0.0,
		write_rule=r["write_rule"], vote_share=r["vote_share"], neuron_sample_rate=r["neuron_sample_rate"],
		offspring_folds="all",
	)


def genomes(ev, n, clusters, seed, neurons=10, bits=8):
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	rng = np.random.default_rng(seed)
	tib = ev.total_input_bits
	return [ClusterGenome(bits_per_neuron=[bits] * (neurons * clusters), neurons_per_cluster=[neurons] * clusters,
		connections=rng.integers(0, tib, size=neurons * clusters * bits).tolist()) for _ in range(n)]


def hx(v):
	return None if v is None else float(v).hex()


def mrow(m):
	return [hx(m.ce), hx(m.acc), hx(m.f1), hx(m.fpr)]


def grow(g):
	return [list(map(int, g.bits_per_neuron)), list(map(int, g.neurons_per_cluster)), list(map(int, g.connections))]


def kfold_block(ev, clusters):
	gs = genomes(ev, 6, clusters, 1)
	return [mrow(m) for m in ev.evaluate_batch(gs)]


def offspring_block(ev, clusters):
	pop = [(g, 0.5 + i / 10) for i, g in enumerate(genomes(ev, 8, clusters, 2))]
	res = ev.search_offspring(
		population=pop, target_count=8, max_attempts=40, accuracy_threshold=0.0, min_bits=8, max_bits=8,
		min_neurons=5, max_neurons=20, bits_mutation_rate=0.0, neurons_mutation_rate=0.1, crossover_rate=1.0,
		tournament_size=3, train_subset_idx=1, seed=7, phase_type=0, cluster_crossover_ratio=0.5,
		pool_shuffle_ratio=0.8, assortative_mating_ratio=0.85, crossover_bits="blend_cluster",
	)
	immigrants = genomes(ev, 3, clusters, 3)
	return {"offspring": [grow(g) + [mrow(g.metrics)] for g in res.genomes],
		"immigrants": [mrow(m) for m in ev.evaluate_offspring_batch(immigrants, 1)]}


def ga_block(ev, clusters):
	from wnn.ram.strategies.connectivity.architecture_config import ArchitectureConfig
	from wnn.ram.strategies.connectivity.architecture_ga import ArchitectureGAStrategy
	from wnn.ram.strategies.connectivity.framework import GAConfig
	arch = ArchitectureConfig(
		num_clusters=clusters, min_bits=8, max_bits=8, min_neurons=5, max_neurons=20, default_bits=8,
		default_neurons=10, total_input_bits=ev.total_input_bits, optimize_neurons=True, optimize_bits=False,
		assortative_mating_ratio=0.85, pool_shuffle_ratio=0.8, cluster_crossover_ratio=0.5,
		crossover_bits="blend_cluster",
	)
	cfg = GAConfig(population_size=8, generations=5, patience=50, check_interval=1, fitness_percentile=0.75,
		fitness_aggregation="desirability", fitness_weight_ce=0.1, fitness_weight_f1=0.35,
		fitness_weight_fpr=0.35, fitness_weight_acc=0.2, magnitude_aware_patience=True)
	s = ArchitectureGAStrategy(arch_config=arch, ga_config=cfg, seed=SEED, cached_evaluator=ev)
	res = s.optimize(initial_population=genomes(ev, 8, clusters, 4), train_subset_idx=1)
	return {"best": grow(res.best_genome), "history": [[int(i), hx(f)] for i, f in res.history],
		"final": [grow(g) for g in res.final_population],
		"final_metrics": [mrow(m) for m in (res.population_metrics or [])]}


def main(out):
	import ram_accelerator
	ds = dataset()
	report = {"abi": int(ram_accelerator.ABI_VERSION)}
	for recipe in RECIPES:
		clusters = 1 if RECIPES[recipe]["single_cluster"] else 2
		ev = evaluator(ds, recipe)
		report[recipe] = {"kfold": kfold_block(ev, clusters), "search": offspring_block(ev, clusters),
			"ga": ga_block(ev, clusters)}
	body = {k: v for k, v in report.items() if k != "abi"}
	with open(out, "w") as f:
		json.dump({"abi": report["abi"], "results": body}, f, indent=0, sort_keys=True)
	print(out, "abi", report["abi"], "ga gens", [len(body[r]["ga"]["history"]) for r in body])


if __name__ == "__main__":
	main(sys.argv[1])
