"""Standalone driver: connectivity GA over the drone controller, C1 vs C2.

Reuses the real GA loop (ControllerGAStrategy ⊂ GenericGAStrategy) — no worker,
no dashboard (those wire in later, drop-in). Runs the SAME connectivity GA with
two inner write-rules and reports both against PID + untrained baselines:

  C1  target_source="pid"     — reward-gated imitation of the expert.
  C2  target_source="student" — reinforce the student's OWN winning actions.

The GA evolves a FiniteStateGenome (input-bit connectivity; full-state recurrent
wiring preserved by construction); cells are trained per genome by the inner
loop. Genome fitness = closed-loop reward of the FINAL trained controller on a
fixed eval set (selection pressure favours connectivity whose gated training
CONVERGES rather than diverges — exactly what we want).

Substrate (locked): absolute-PWM, full-state connectivity, 4 state neurons.

Run (thread-capped to coexist with the IDS worker):
  RAYON_NUM_THREADS=3 python tests/run_controller_ga.py --paradigm both \
      --pop 12 --gens 6 --rounds 5 --episodes 24 --steps 1500 --tilt 15 --workers 1
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from wnn.control.evaluator import (
	ControllerSpec, ControllerGenome, ControllerEvaluator, build_controller,
	fit_thresholds_from_pid_rollouts,
)
from wnn.control.genome import FiniteStateGenome
from wnn.control.ga_strategy import ControllerGAStrategy, default_controller_ga_config
from wnn.control.reward_gated import RewardGatedConfig, reward_gated_train
from wnn.control.dagger import eval_closed_loop_reset
from wnn.control.pid import AttitudePID, AttitudePIDConfig
from wnn.control.training import EpisodeConfig, make_wnn_action_fn, make_pid_action_fn


def _score(action_fn, reset_fn, ec, n, seed):
	_, m = eval_closed_loop_reset(action_fn, reset_fn, ec, n, seed)
	return m


def run_paradigm(name, target_source, spec, thresholds, ec, args, seed):
	print(f"\n{'='*64}\n  {name}  (target_source={target_source!r})\n{'='*64}")
	rg = RewardGatedConfig(
		num_rounds=args.rounds, episodes_per_round=args.episodes,
		steps_per_episode=args.steps, bptt_window=args.window,
		gate_mode="improvement", target_source=target_source,
		curriculum=args.curriculum, full_tilt_deg=args.tilt,
		eval_episodes=args.eval_episodes, episode_config=ec,
	)
	ev = ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes, seed=seed,
		episode_config=ec, thresholds=thresholds, rg_config=rg,
		max_train_workers=args.workers)
	gacfg = default_controller_ga_config(population_size=args.pop, generations=args.gens)
	strat = ControllerGAStrategy(spec, ga_config=gacfg, seed=seed, batch_evaluator=ev)
	init_pop = [FiniteStateGenome.random(spec, seed=seed * 1000 + i) for i in range(args.pop)]

	t0 = time.time()
	res = strat.optimize(evaluate_fn=ev.evaluate_single, batch_evaluate_fn=ev.evaluate_batch,
		initial_population=init_pop)
	dt = time.time() - t0
	best = res.best_genome
	assert best.state_bits_intact(), "GA corrupted the full-state invariant!"

	# Re-train the best genome's connectivity and score the final controller.
	controller, st = reward_gated_train(spec, thresholds,
		best.state_connections, best.output_connections, rg)
	m = _score(make_wnn_action_fn(controller), controller.reset, ec, args.eval_episodes, seed)
	print(f"  GA done in {dt:.0f}s | best closed-loop reward={res.final_fitness:.2f}")
	print(f"  BEST {name}: mean_err={m['mean_attitude_error_deg']:.2f}°  "
	      f"stable={m['stable_rate']*100:.0f}%  reward={m['mean_reward']:.2f}")
	print(f"  best-genome train curve (err°): {[f'{e:.1f}' for e in st['iter_mean_err_deg']]}")
	return m


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--paradigm", choices=["c1", "c2", "both"], default="both")
	ap.add_argument("--pop", type=int, default=12)
	ap.add_argument("--gens", type=int, default=6)
	ap.add_argument("--rounds", type=int, default=5)
	ap.add_argument("--episodes", type=int, default=24)
	ap.add_argument("--eval-episodes", type=int, default=20)
	ap.add_argument("--steps", type=int, default=1500)
	ap.add_argument("--window", type=int, default=32)
	ap.add_argument("--tilt", type=float, default=15.0)
	ap.add_argument("--curriculum", action="store_true")
	ap.add_argument("--workers", type=int, default=1)
	ap.add_argument("--state-neurons", type=int, default=4)
	ap.add_argument("--levels", type=int, default=16)
	ap.add_argument("--seed", type=int, default=0)
	args = ap.parse_args()

	# Locked substrate: absolute-PWM, full-state connectivity, N state neurons.
	bits = max(24, 2 * args.state_neurons + 8)
	spec = ControllerSpec(
		num_motors=4, levels_per_motor=args.levels, bits_per_feature=8,
		input_window_k=4, state_neurons=args.state_neurons,
		state_bits_per_neuron=bits, output_bits_per_neuron=bits,
		delta_control=False,
	)
	ec = EpisodeConfig(dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt),
		max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)

	print(f"Substrate: {args.state_neurons} state neurons × {bits}b, {args.levels} levels, "
	      f"absolute-PWM, full-state. Fixed {args.tilt}° ICs.")
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=args.seed)

	# Baselines on the same eval set.
	pid = AttitudePID(AttitudePIDConfig())
	pm = _score(make_pid_action_fn(pid), pid.reset, ec, args.eval_episodes, args.seed)
	un = build_controller(ControllerGenome(spec=spec, thresholds=thresholds,
		state_connections=FiniteStateGenome.random(spec, 0).state_connections,
		output_connections=FiniteStateGenome.random(spec, 0).output_connections))
	um = _score(make_wnn_action_fn(un), un.reset, ec, args.eval_episodes, args.seed)
	print(f"[PID]       {pm['mean_attitude_error_deg']:.2f}°  stable={pm['stable_rate']*100:.0f}%")
	print(f"[Untrained] {um['mean_attitude_error_deg']:.2f}°  stable={um['stable_rate']*100:.0f}%")

	results = {}
	if args.paradigm in ("c1", "both"):
		results["C1 reward-gated"] = run_paradigm("C1 reward-gated", "pid", spec, thresholds, ec, args, args.seed)
	if args.paradigm in ("c2", "both"):
		results["C2 reinforce-own"] = run_paradigm("C2 reinforce-own", "student", spec, thresholds, ec, args, args.seed)

	print(f"\n{'='*64}\n  SUMMARY (lower err / higher stable = better)\n{'='*64}")
	print(f"  {'policy':<20} {'mean_err':>10} {'stable':>8} {'reward':>10}")
	print(f"  {'PID (teacher)':<20} {pm['mean_attitude_error_deg']:>9.2f}° {pm['stable_rate']*100:>7.0f}% {pm['mean_reward']:>10.2f}")
	print(f"  {'Untrained':<20} {um['mean_attitude_error_deg']:>9.2f}° {um['stable_rate']*100:>7.0f}% {um['mean_reward']:>10.2f}")
	for k, m in results.items():
		print(f"  {k:<20} {m['mean_attitude_error_deg']:>9.2f}° {m['stable_rate']*100:>7.0f}% {m['mean_reward']:>10.2f}")

	beats = any(m["stable_rate"] > 0 or m["mean_attitude_error_deg"] < um["mean_attitude_error_deg"]
	            for m in results.values())
	return 0 if beats else 1


if __name__ == "__main__":
	sys.exit(main())
