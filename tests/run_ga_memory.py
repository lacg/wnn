"""Standalone driver: GA-Memory (paradigm B) — neuroevolve QSR cells directly.

No imitation, no training solver: a genome is the cell values over the recorded
address universe; fitness = closed-loop reward, scored on the GPU. The whole
population evaluates in one Metal batch → IDS-like speed (this is the paradigm
that scales to 50×250, unlike per-genome-trained C1/C2).

Run:  RAYON_NUM_THREADS=3 python tests/run_ga_memory.py --pop 40 --gens 30 \
        --eval-episodes 20 --steps 1500 --tilt 15 --seed 0
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from wnn.control.evaluator import (
	ControllerSpec, ControllerGenome, build_controller, fit_thresholds_from_pid_rollouts,
)
from wnn.control.genome import FiniteStateGenome
from wnn.control.ga_strategy import default_controller_ga_config
from wnn.control.ga_memory import (
	record_address_universe, MemoryGenome, build_controller_from_memory,
	ControllerMemoryEvaluator, ControllerMemoryGAStrategy,
)
from wnn.control.pid import AttitudePID, AttitudePIDConfig
from wnn.control.dagger import eval_closed_loop_reset
from wnn.control.training import EpisodeConfig, make_wnn_action_fn, make_pid_action_fn


def _score(action_fn, reset_fn, ec, n, seed):
	_, m = eval_closed_loop_reset(action_fn, reset_fn, ec, n, seed)
	return m


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--pop", type=int, default=40)
	ap.add_argument("--gens", type=int, default=30)
	ap.add_argument("--eval-episodes", type=int, default=20)
	ap.add_argument("--steps", type=int, default=1500)
	ap.add_argument("--tilt", type=float, default=15.0)
	ap.add_argument("--state-neurons", type=int, default=4)
	ap.add_argument("--levels", type=int, default=16)
	ap.add_argument("--universe-episodes", type=int, default=12)
	ap.add_argument("--seed", type=int, default=0)
	args = ap.parse_args()
	t_start = time.time()

	bits = max(24, 2 * args.state_neurons + 8)
	spec = ControllerSpec(num_motors=4, levels_per_motor=args.levels, bits_per_feature=8,
		input_window_k=4, state_neurons=args.state_neurons,
		state_bits_per_neuron=bits, output_bits_per_neuron=bits, delta_control=False)
	ec = EpisodeConfig(dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt), max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)

	print(f"GA-Memory: {args.state_neurons} state neurons × {bits}b, {args.levels} levels, "
	      f"fixed {args.tilt}° ICs, pop {args.pop} × gens {args.gens}.")
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=args.seed)
	# Fixed connectivity (full-state); GA-Memory evolves the CELLS, not the wiring.
	fsg = FiniteStateGenome.random(spec, seed=args.seed)
	sc, oc = fsg.state_connections, fsg.output_connections

	# Baselines.
	pid = AttitudePID(AttitudePIDConfig())
	pm = _score(make_pid_action_fn(pid), pid.reset, ec, args.eval_episodes, args.seed)
	un = build_controller(ControllerGenome(spec=spec, thresholds=thresholds,
		state_connections=sc, output_connections=oc))
	um = _score(make_wnn_action_fn(un), un.reset, ec, args.eval_episodes, args.seed)
	print(f"[PID]       {pm['mean_attitude_error_deg']:.2f}°  stable={pm['stable_rate']*100:.0f}%")
	print(f"[Untrained] {um['mean_attitude_error_deg']:.2f}°  stable={um['stable_rate']*100:.0f}%")

	# Record the address universe (one-time, PID-driven).
	t = time.time()
	su, ou = record_address_universe(spec, thresholds, sc, oc,
		num_episodes=args.universe_episodes, steps=args.steps, tilt_deg=args.tilt, seed=args.seed)
	print(f"Universe: {len(su)} state cells, {len(ou)} output cells "
	      f"(recorded in {time.time()-t:.0f}s)")

	# Evolve cells on closed-loop reward (GPU-scored, no training).
	ev = ControllerMemoryEvaluator(spec, thresholds, num_eval_episodes=args.eval_episodes,
		seed=args.seed, episode_config=ec)
	gacfg = default_controller_ga_config(population_size=args.pop, generations=args.gens)
	strat = ControllerMemoryGAStrategy(spec, sc, oc, su, ou, ga_config=gacfg,
		seed=args.seed, batch_evaluator=ev)
	import numpy as np
	rng = np.random.default_rng(args.seed)
	init_pop = [MemoryGenome.random(spec, sc, oc, su, ou, rng) for _ in range(args.pop)]

	t = time.time()
	res = strat.optimize(evaluate_fn=ev.evaluate_single, batch_evaluate_fn=ev.evaluate_batch,
		initial_population=init_pop)
	ga_dt = time.time() - t

	# Score the best genome.
	best = res.best_genome
	c = build_controller_from_memory(best, thresholds)
	m = _score(make_wnn_action_fn(c), c.reset, ec, args.eval_episodes, args.seed)

	print(f"\n{'='*64}\n  GA-MEMORY RESULT\n{'='*64}")
	print(f"  GA wall time: {ga_dt:.0f}s ({ga_dt/max(args.gens,1):.1f}s/gen)  best reward={res.final_fitness:.2f}")
	print(f"  {'policy':<14}{'mean_err':>10}{'stable':>8}{'reward':>10}")
	print(f"  {'PID':<14}{pm['mean_attitude_error_deg']:>9.2f}°{pm['stable_rate']*100:>7.0f}%{pm['mean_reward']:>10.2f}")
	print(f"  {'Untrained':<14}{um['mean_attitude_error_deg']:>9.2f}°{um['stable_rate']*100:>7.0f}%{um['mean_reward']:>10.2f}")
	print(f"  {'GA-Memory':<14}{m['mean_attitude_error_deg']:>9.2f}°{m['stable_rate']*100:>7.0f}%{m['mean_reward']:>10.2f}")
	print(f"\n  TOTAL: {(time.time()-t_start)/60:.1f} min")
	return 0 if (m["stable_rate"] > 0 or m["mean_attitude_error_deg"] < um["mean_attitude_error_deg"]) else 1


if __name__ == "__main__":
	sys.exit(main())
