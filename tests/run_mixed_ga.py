"""Mixed-operator neuroevolution of the unified controller genome.

ONE GA, four operators. Each generation: keep the top `elitism` fraction, then
fill the rest with offspring produced by a uniformly-random operator among
{NEURONS, BITS, CONNECTIONS, MEMORY} — so with 20% elitism you get ~20% of the
population from each operator. The genome carries QSR cells (universe recorded
once for the seed arch, remapped through arch mutations).

## Paradigm: training ON or OFF (--training / --no-training)

Default is **--training ON** (paradigm-A): each genome is reward-gated trained
before scoring, so the cells the GA mutations produce get further refinement
from the DAGGER-style training loop. This is what was missing from C-mix-1/2/3
(those silently used paradigm-B because the original default ran score_genomes
instead of evaluate_batch). Their plateaus at 13.69° — 17.89° were therefore
**NOT a fair test of mixed-GA**, only of mixed-GA-WITHOUT-training; flipped to
the default 29/05/2026 after the regression RCA. Use --no-training to reproduce
the historical paradigm-B behavior.

Run (overnight, paradigm-A — the corrected default):
  RAYON_NUM_THREADS=3 python tests/run_mixed_ga.py \
    --pop 200 --gens 10000 --patience 100000 --elitism 0.2 \
    --eval-episodes 20 --steps 1500 --tilt 15 --state-neurons 4 --levels 16 --seed 0
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from wnn.control.evaluator import ControllerSpec, ControllerEvaluator, fit_thresholds_from_pid_rollouts
from wnn.control.arch_strategy import ControllerMixedGAStrategy, default_controller_arch_config
from wnn.control.ga_strategy import default_controller_ga_config
from wnn.control.training import EpisodeConfig


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--pop", type=int, default=200)
	ap.add_argument("--gens", type=int, default=10000)
	ap.add_argument("--patience", type=int, default=100000)
	ap.add_argument("--elitism", type=float, default=0.2)
	ap.add_argument("--eval-episodes", type=int, default=20)
	ap.add_argument("--steps", type=int, default=1500)
	ap.add_argument("--tilt", type=float, default=15.0)
	ap.add_argument("--state-neurons", type=int, default=4)
	ap.add_argument("--levels", type=int, default=16)
	ap.add_argument("--universe-episodes", type=int, default=8)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--crossover-rate", type=float, default=0.5,
		help="Fraction of offspring produced via crossover (vs single-parent mutation). "
		     "0.0=mutation-only (old behavior); 0.5=balanced; 0.7=crossover-heavy.")
	# Paradigm switch — default ON after C-mix-1/2/3 RCA (29/05/2026). See module docstring.
	ap.add_argument("--training", dest="training", action="store_true", default=True,
		help="Paradigm-A: reward-gated train each genome before scoring (DEFAULT, the corrected behavior).")
	ap.add_argument("--no-training", dest="training", action="store_false",
		help="Paradigm-B: score cells directly without training (the historical C-mix-1/2/3 behavior, NOT recommended).")
	# Per-genome ThreadPool worker count. Default 4 on 16-core Mac (leaves room
	# for Rayon-inside-step + the IDS worker). Phase-1 fix landed 29/05/2026
	# alongside the phased-GA RCA; superseded by the planned Rust port (Phase 2).
	ap.add_argument("--train-workers", type=int, default=4,
		help="ControllerEvaluator.max_train_workers; only used when --training. Default 4 = M4 Max sweet spot.")
	args = ap.parse_args()
	t0 = time.time()

	bits = max(24, 2 * args.state_neurons + 16)
	spec = ControllerSpec(num_motors=4, levels_per_motor=args.levels, bits_per_feature=8,
	                      input_window_k=4, state_neurons=args.state_neurons,
	                      state_bits_per_neuron=bits, output_bits_per_neuron=bits)
	ec = EpisodeConfig(dt=0.001, steps_per_episode=args.steps,
	                   max_initial_tilt_rad=math.radians(args.tilt),
	                   max_initial_yaw_rad=math.radians(args.tilt),
	                   max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=args.seed)
	ev = ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes, seed=args.seed,
	                         episode_config=ec, thresholds=thresholds,
	                         max_train_workers=(args.train_workers if args.training else 1))

	# Wider search box than the default (we're exploring all dims for a long run).
	arch_cfg = default_controller_arch_config(spec)
	arch_cfg.max_state_neurons = max(arch_cfg.max_state_neurons, 16)

	gacfg = default_controller_ga_config(population_size=args.pop, generations=args.gens)
	gacfg.patience = args.patience          # high ⇒ never early-stops
	gacfg.elitism_pct = args.elitism        # 20% elite, 80% offspring (mix of crossover + 4 mutators)
	# Crossover ENABLED via RecurrentArchGenome.crossover (variable-shape: one
	# parent's shape + mixed-block suffixes from the other). With rate=0.5,
	# offspring split ~50/50 between crossover children and mutation children;
	# the mutation half is further split across the 4 operators (~12.5% each).
	# Tunable via --crossover-rate.
	gacfg.crossover_rate = args.crossover_rate

	strat = ControllerMixedGAStrategy(spec, arch_config=arch_cfg, ga_config=gacfg,
	                                  seed=args.seed, batch_evaluator=ev,
	                                  record_episodes=args.universe_episodes, record_steps=args.steps)

	paradigm = ("paradigm-A: REWARD-GATED TRAINING then score (corrected default)"
	            if args.training
	            else "paradigm-B: NO training (score build-from-cells) — historical C-mix-1/2/3 behavior")
	print(f"Mixed-operator neuroevolution: pop {args.pop} × gens {args.gens}, elitism {args.elitism:.0%}, "
	      f"4 operators (neurons/bits/connections/memory), {paradigm}.")
	print(f"Seed arch: {args.state_neurons} state neurons × {bits}b, {args.levels} levels. Recording universe…")
	init_pop = [strat.create_random_genome() for _ in range(args.pop)]
	print(f"Universe: {len(init_pop[0].cells.state_universe)} state, {len(init_pop[0].cells.output_universe)} output cells. "
	      f"Init pop built in {time.time()-t0:.0f}s. Starting GA…")

	# Paradigm switch: training ON → evaluate_batch (DAGGER-style training inside);
	# training OFF → score_genomes (cells direct, no training).
	if args.training:
		eval_fn = lambda g: ev.evaluate_batch([g])[0].ce
		batch_eval_fn = ev.evaluate_batch
	else:
		eval_fn = lambda g: ev.score_genomes([g])[0].ce
		batch_eval_fn = ev.score_genomes
	res = strat.optimize(evaluate_fn=eval_fn,
	                     batch_evaluate_fn=batch_eval_fn, initial_population=init_pop)

	best = res.best_genome
	dt = time.time() - t0
	print(f"\n{'='*64}\n  MIXED-OPERATOR NEUROEVOLUTION DONE in {dt/3600:.2f}h ({dt:.0f}s)\n{'='*64}")
	print(f"  best fitness (CE=-reward): {res.final_fitness:.4f}  | iterations: {res.iterations_run}")
	if best is not None:
		s = best.stats()
		print(f"  best genome: state_neurons={best.state_neurons}, levels={best.output_neurons//spec.num_motors}, "
		      f"state_bits={best.state_bits_per_neuron}, output_bits={best.output_bits_per_neuron}, "
		      f"cells={len(best.cells.state_universe)+len(best.cells.output_universe) if best.cells else 0}")
		print(f"  total_neurons={s['total_neurons']}, total_connections={s['total_connections']}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
