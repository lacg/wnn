"""Standalone driver: GA-Memory (paradigm B) — neuroevolve QSR cells directly.

No imitation, no training solver: a genome is the cell values over the recorded
address universe; fitness = closed-loop reward, scored on the GPU. The whole
population evaluates in one Metal batch → IDS-like speed (this is the paradigm
that scales to 50×250, unlike per-genome-trained C1/C2).

3-WAY + MULTI-SEED (see wnn.seeds): seeds are built-in — omit them for a UTC-date
base, auto-recorded to the `seed_runs` DB table. TRAIN seeds the threshold fit,
connectivity, address-universe recording and the evolve fitness; TEST selects the
final genome from the evolved population (unseen during evolve); VAL is the held-out
report, touched once. `--runs N` repeats with independent seed sets from the base and
summarises val mean±std. Pass the SAME --base-seed as run_mlp_ga.py for a controlled
substrate comparison.

Run:  python tests/run_ga_memory.py --pop 150 --gens 3000 --runs 1      # base=today
      python tests/run_ga_memory.py --base-seed 20260526 --pop 40 --gens 30 --runs 3
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

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
from wnn.seeds import resolve_seed_set, log_seed_set, record_seed_set


def _score(action_fn, reset_fn, ec, n, seed):
	_, m = eval_closed_loop_reset(action_fn, reset_fn, ec, n, seed)
	return m


def _evolve_one_run(args, ec, s):
	"""One GA-Memory run for seed set `s`: evolve cells on TRAIN reward, select the
	final genome from the population on TEST, report TRAIN/TEST/VAL. Returns val metrics."""
	t_run = time.time()
	bits = max(24, 2 * args.state_neurons + 8)
	spec = ControllerSpec(num_motors=4, levels_per_motor=args.levels, bits_per_feature=8,
		input_window_k=4, state_neurons=args.state_neurons,
		state_bits_per_neuron=bits, output_bits_per_neuron=bits, delta_control=False)

	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=s.train)
	# Fixed connectivity (full-state); GA-Memory evolves the CELLS, not the wiring.
	fsg = FiniteStateGenome.random(spec, seed=s.train)
	sc, oc = fsg.state_connections, fsg.output_connections

	# Baselines (on val, the reported partition).
	pid = AttitudePID(AttitudePIDConfig())
	pm = _score(make_pid_action_fn(pid), pid.reset, ec, args.final_episodes, s.val)
	un = build_controller(ControllerGenome(spec=spec, thresholds=thresholds,
		state_connections=sc, output_connections=oc))
	um = _score(make_wnn_action_fn(un), un.reset, ec, args.final_episodes, s.val)

	# Record the address universe from the TRAIN distribution (one-time, PID-driven).
	t = time.time()
	su, ou = record_address_universe(spec, thresholds, sc, oc,
		num_episodes=args.universe_episodes, steps=args.steps, tilt_deg=args.tilt, seed=s.train)
	print(f"  run {s.run_index}: universe {len(su)} state / {len(ou)} output cells ({time.time()-t:.0f}s)")

	# Evolve cells on closed-loop reward over TRAIN ICs (GPU-scored, no training).
	ev = ControllerMemoryEvaluator(spec, thresholds, num_eval_episodes=args.eval_episodes,
		seed=s.train, episode_config=ec)
	gacfg = default_controller_ga_config(population_size=args.pop, generations=args.gens)
	gacfg.patience = args.patience
	strat = ControllerMemoryGAStrategy(spec, sc, oc, su, ou, ga_config=gacfg,
		seed=s.train, batch_evaluator=ev)
	rng = np.random.default_rng(s.train)
	init_pop = [MemoryGenome.random(spec, sc, oc, su, ou, rng) for _ in range(args.pop)]

	t = time.time()
	res = strat.optimize(evaluate_fn=ev.evaluate_single, batch_evaluate_fn=ev.evaluate_batch,
		initial_population=init_pop)
	ga_dt = time.time() - t

	# MODEL SELECTION on TEST: re-score the top candidates on held-out-from-train test
	# ICs and keep the best-on-test (not the best-on-train res.best_genome).
	cands = list(getattr(res, "final_population", None) or [])
	if res.best_genome is not None:
		cands = [res.best_genome] + cands
	cands = cands[:max(1, args.select_topk)] or [res.best_genome]
	def test_reward(g):
		c = build_controller_from_memory(g, thresholds)
		return _score(make_wnn_action_fn(c), c.reset, ec, args.eval_episodes, s.test)["mean_reward"]
	best = max(cands, key=test_reward)

	def score(seed):
		c = build_controller_from_memory(best, thresholds)
		return _score(make_wnn_action_fn(c), c.reset, ec, args.final_episodes, seed)
	m_train, m_test, m_val = score(s.train), score(s.test), score(s.val)
	print(f"  run {s.run_index} ({(time.time()-t_run)/60:.1f}m, GA {ga_dt:.0f}s, {ga_dt/max(args.gens,1):.1f}s/gen) "
	      f"— selected {len(cands)} cand(s) on test:")
	print(f"    {'partition':<20}{'mean_err':>10}{'stable':>8}{'reward':>10}")
	for label, m in (("train", m_train), ("test (selection)", m_test), ("VAL held-out", m_val),
	                 ("PID (val)", pm), ("Untrained (val)", um)):
		print(f"    {label:<20}{m['mean_attitude_error_deg']:>9.2f}°{m['stable_rate']*100:>7.0f}%{m['mean_reward']:>10.2f}")
	return m_val


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--pop", type=int, default=40)
	ap.add_argument("--gens", type=int, default=30)
	ap.add_argument("--eval-episodes", type=int, default=20)   # during evolve / selection
	ap.add_argument("--final-episodes", type=int, default=20)  # report
	ap.add_argument("--steps", type=int, default=1500)
	ap.add_argument("--tilt", type=float, default=15.0)
	ap.add_argument("--state-neurons", type=int, default=4)
	ap.add_argument("--levels", type=int, default=16)
	ap.add_argument("--universe-episodes", type=int, default=12)
	ap.add_argument("--patience", type=int, default=5,
		help="early-stop patience (checks w/o improvement); set high to run all gens")
	ap.add_argument("--select-topk", type=int, default=8,
		help="how many top genomes to re-score on TEST for final model selection")
	# Seeds are built-in (see wnn.seeds): omit for a date-derived base, recorded to DB.
	ap.add_argument("--base-seed", type=int, default=None,
		help="Master seed; default = UTC timestamp (YYYYMMDDHHMMSS). Share across MLP/WNN/PID.")
	ap.add_argument("--runs", type=int, default=1, help="Multi-seed runs (mean±std over held-out val)")
	ap.add_argument("--train-seed", type=int, default=None, help="Override derived train seed")
	ap.add_argument("--test-seed", type=int, default=None, help="Override derived test seed")
	ap.add_argument("--val-seed", type=int, default=None, help="Override derived val seed")
	args = ap.parse_args()
	t_start = time.time()

	ec = EpisodeConfig(dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt), max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)
	print(f"GA-Memory: {args.state_neurons} state neurons, {args.levels} levels, {args.tilt}° ICs, "
	      f"pop {args.pop} × gens {args.gens}, runs={args.runs}, 3-way seeds.")

	val_runs = []
	for run_i in range(args.runs):
		s = resolve_seed_set(base=args.base_seed, run_index=run_i,
			train=args.train_seed, test=args.test_seed, val=args.val_seed)
		log_seed_set(s)
		record_seed_set(s, script="run_ga_memory", extra={
			"state_neurons": args.state_neurons, "levels": args.levels, "pop": args.pop,
			"gens": args.gens, "eval_episodes": args.eval_episodes, "steps": args.steps, "tilt": args.tilt})
		val_runs.append(_evolve_one_run(args, ec, s))

	def agg(key):
		xs = np.array([m[key] for m in val_runs], dtype=float)
		return xs.mean(), xs.std()
	err_m, err_s = agg("mean_attitude_error_deg")
	stab_m, stab_s = agg("stable_rate")
	rew_m, rew_s = agg("mean_reward")
	print(f"\n{'='*64}\n  GA-MEMORY — held-out (val) over {args.runs} run(s), {(time.time()-t_start)/60:.1f} min\n{'='*64}")
	print(f"  mean_attitude_error_deg : {err_m:.2f} ± {err_s:.2f}°")
	print(f"  stable_rate             : {stab_m*100:.1f} ± {stab_s*100:.1f}%")
	print(f"  mean_reward             : {rew_m:.2f} ± {rew_s:.2f}")
	return 0 if any(m["stable_rate"] > 0 for m in val_runs) else 1


if __name__ == "__main__":
	sys.exit(main())
