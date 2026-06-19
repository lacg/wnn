"""GA-evolved MLP — the FAIR substrate comparison to the WNN.

Evolves a small MLP's weights with the SAME optimization scheme as the WNN
(evolution against closed-loop reward, no gradient) on the SAME sim/ICs/reward/
scorer, with the SAME IMU inputs [gyro, accel, target]. CPU-only (numpy MLP +
Rust sim), so it doesn't contend with the GPU.

3-WAY + MULTI-SEED (see wnn.seeds): seeds are built-in — you don't pass them. A
UTC-date base seed derives three independent partitions: TRAIN (evolve), TEST
(model selection — the gen-best is kept by its *test* reward, not train, which is
what the leaky 2-way run lacked), VAL (held-out, reported once). `--runs N` repeats
with independent seed sets from the same base; the val metrics are summarised
mean±std. Every seed set is logged and stored in the `seed_runs` DB table. Share the
SAME --base-seed across MLP/WNN/PID for a controlled comparison.

Run:  python tests/run_mlp_ga.py --pop 100 --gens 500 --hidden 32 \
        --eval-episodes 10 --runs 3            # base = today's date, auto-recorded
      python tests/run_mlp_ga.py --base-seed 20260526 --runs 5   # pinned, replicable
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from wnn.control._accel import AttitudeSim
from wnn.control.pid import AttitudePID, AttitudePIDConfig
from wnn.control.training import EpisodeConfig, _sample_initial_state, make_pid_action_fn
from wnn.control.dagger import eval_closed_loop_reset
from wnn.seeds import resolve_seed_set, log_seed_set, record_seed_set

TARGET = (0.0, 0.0, 0.0)


def _shapes(hidden):
	return [(hidden, 9), (hidden, hidden), (4, hidden)]


def _param_count(hidden):
	return sum(o * i + o for (o, i) in _shapes(hidden))


def _unflatten(w, hidden):
	out, k = [], 0
	for (o, i) in _shapes(hidden):
		W = w[k:k + o * i].reshape(o, i); k += o * i
		b = w[k:k + o]; k += o
		out.append((W, b))
	return out


def _forward(layers, x):
	h = x
	for li, (W, b) in enumerate(layers):
		z = W @ h + b
		h = np.maximum(z, 0.0) if li < len(layers) - 1 else 1.0 / (1.0 + np.exp(-z))
	return h


def make_np_mlp_action_fn(w, hidden, mean, std):
	layers = _unflatten(w, hidden)
	def fn(gyro, accel, target_rpy, q):
		x = (np.array([*gyro, *accel, *target_rpy], dtype=np.float64) - mean) / std
		return tuple(float(v) for v in _forward(layers, x))
	return fn


def _xavier_init(hidden, rng):
	parts = []
	for (o, i) in _shapes(hidden):
		parts.append(rng.normal(0.0, math.sqrt(1.0 / i), o * i))  # weights
		parts.append(np.zeros(o))                                 # biases
	return np.concatenate(parts)


def _input_stats(ec, n_ep, seed):
	"""mean/std of the IMU input over PID rollouts (for normalization only)."""
	sim, pid = AttitudeSim(), AttitudePID(AttitudePIDConfig())
	rng = np.random.default_rng(seed)
	X = []
	for _ in range(n_ep):
		er = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		q0, om0 = _sample_initial_state(er, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
		                                ec.max_initial_body_rate, ec.max_initial_yaw_rate)
		sim.reset(q=list(q0), omega=list(om0)); pid.reset()
		for _t in range(ec.steps_per_episode):
			if sim.is_unstable():
				break
			g, a = sim.read_imu()
			X.append([*g, *a, *TARGET]); sim.step(list(pid.step(sim.quaternion, g, TARGET)))
	X = np.asarray(X, np.float64)
	mean, std = X.mean(0), X.std(0); std[std < 1e-6] = 1.0
	return mean, std


def _evolve_one_run(args, ec, dim, s):
	"""One GA run for seed set `s`: evolve on TRAIN, select on TEST, report TRAIN/TEST/VAL.
	Returns the held-out (val) metrics dict."""
	t0 = time.time()
	mean, std = _input_stats(ec, 6, s.train + 1234)
	rng = np.random.default_rng(s.train)
	pop = [_xavier_init(args.hidden, rng) for _ in range(args.pop)]
	n_elite = max(1, int(args.elite_frac * args.pop))
	sigma = args.sigma

	def reward_on(w, n_ep, seed):
		r, _m = eval_closed_loop_reset(make_np_mlp_action_fn(w, args.hidden, mean, std),
		                               lambda: None, ec, n_ep, seed)
		return r

	# Selection is on TEST ICs (unseen during the train-fitness step) — this is the
	# model-selection partition that the leaky 2-way run never had.
	best_w, best_test = None, -math.inf
	for gen in range(args.gens):
		fits = np.array([reward_on(w, args.eval_episodes, s.train) for w in pop])
		order = np.argsort(-fits)  # descending (higher reward better)
		elite = [pop[i] for i in order[:n_elite]]
		test_r = reward_on(elite[0], args.eval_episodes, s.test)
		if test_r > best_test:
			best_test, best_w = test_r, elite[0].copy()
		children = [elite[int(rng.integers(0, n_elite))] + rng.normal(0.0, sigma, dim)
		            for _ in range(args.pop - n_elite)]
		pop = elite + children
		if (gen + 1) % 50 == 0 or gen == 0:
			el = time.time() - t0
			print(f"    Gen {gen+1:0{len(str(args.gens))}d}/{args.gens}: train_best={fits[order[0]]:.2f}, "
			      f"sel_test={best_test:.2f} [{el:.0f}s, ETA {(args.gens-gen-1)*el/(gen+1)/60:.0f}m]")

	def score(seed):
		_, m = eval_closed_loop_reset(make_np_mlp_action_fn(best_w, args.hidden, mean, std),
		                              lambda: None, ec, args.final_episodes, seed)
		return m
	m_train, m_test, m_val = score(s.train), score(s.test), score(s.val)
	pid = AttitudePID(AttitudePIDConfig())
	_, pid_val = eval_closed_loop_reset(make_pid_action_fn(pid), pid.reset, ec, args.final_episodes, s.val)
	dt = time.time() - t0
	print(f"  run {s.run_index} ({dt/3600:.2f}h) — selected on test, reported on held-out val:")
	print(f"    {'partition':<20} {'mean_err':>8} {'stable':>6} {'reward':>10}")
	for label, m in (("train", m_train), ("test (selection)", m_test), ("VAL held-out", m_val)):
		print(f"    {label:<20} {m['mean_attitude_error_deg']:>7.2f}°  {m['stable_rate']*100:>5.0f}%  {m['mean_reward']:>9.2f}")
	print(f"    {'PID (val, sees q)':<20} {pid_val['mean_attitude_error_deg']:>7.2f}°  {pid_val['stable_rate']*100:>5.0f}%  {pid_val['mean_reward']:>9.2f}")
	return m_val


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--pop", type=int, default=100)
	ap.add_argument("--gens", type=int, default=500)
	ap.add_argument("--hidden", type=int, default=32)
	ap.add_argument("--elite-frac", type=float, default=0.2)
	ap.add_argument("--sigma", type=float, default=0.05)        # mutation stddev
	ap.add_argument("--eval-episodes", type=int, default=10)    # during evolution (cheap)
	ap.add_argument("--final-episodes", type=int, default=20)   # report (matches WNN runs)
	ap.add_argument("--steps", type=int, default=1500)
	ap.add_argument("--tilt", type=float, default=15.0)
	# Initial-condition severity (defaults = the legacy 15°/0.5 setting). Set these to
	# match a curriculum stage, e.g. Stage E = --tilt 60 --body-rate 4.0 --yaw-rate 2.4
	# --steps 250, for a difficulty-matched baseline vs the WNN full-curriculum report.
	ap.add_argument("--body-rate", type=float, default=0.5, help="max initial body rate (rad/s)")
	ap.add_argument("--yaw-rate", type=float, default=0.3, help="max initial yaw rate (rad/s)")
	# Seeds are built-in (see wnn.seeds): omit for a date-derived base, recorded to DB.
	ap.add_argument("--base-seed", type=int, default=None,
		help="Master seed; default = UTC timestamp (YYYYMMDDHHMMSS). Share across MLP/WNN/PID.")
	ap.add_argument("--runs", type=int, default=1, help="Multi-seed runs (mean±std over held-out val)")
	ap.add_argument("--train-seed", type=int, default=None, help="Override derived train seed (replication)")
	ap.add_argument("--test-seed", type=int, default=None, help="Override derived test seed")
	ap.add_argument("--val-seed", type=int, default=None, help="Override derived val seed")
	args = ap.parse_args()
	t0 = time.time()

	ec = EpisodeConfig(dt=0.001, steps_per_episode=args.steps,
	                   max_initial_tilt_rad=math.radians(args.tilt),
	                   max_initial_yaw_rad=math.radians(args.tilt),
	                   max_initial_body_rate=args.body_rate, max_initial_yaw_rate=args.yaw_rate)
	dim = _param_count(args.hidden)
	print(f"GA-evolved MLP 9→{args.hidden}→{args.hidden}→4 ({dim} weights), pop {args.pop} × gens {args.gens}, "
	      f"σ={args.sigma}, elite {args.elite_frac:.0%}, IMU inputs. CPU-only. runs={args.runs}, 3-way seeds.")

	val_runs = []
	for run_i in range(args.runs):
		s = resolve_seed_set(base=args.base_seed, run_index=run_i,
		                     train=args.train_seed, test=args.test_seed, val=args.val_seed)
		log_seed_set(s)
		record_seed_set(s, script="run_mlp_ga", extra={
			"hidden": args.hidden, "pop": args.pop, "gens": args.gens,
			"eval_episodes": args.eval_episodes, "steps": args.steps, "tilt": args.tilt,
			"body_rate": args.body_rate, "yaw_rate": args.yaw_rate})
		val_runs.append(_evolve_one_run(args, ec, dim, s))

	# Multi-seed summary over the held-out (val) partition — the honest, reportable number.
	def agg(key):
		xs = np.array([m[key] for m in val_runs], dtype=float)
		return xs.mean(), xs.std()
	err_m, err_s = agg("mean_attitude_error_deg")
	stab_m, stab_s = agg("stable_rate")
	rew_m, rew_s = agg("mean_reward")
	dt = time.time() - t0
	print(f"\n{'='*64}\n  GA-EVOLVED MLP — held-out (val) over {args.runs} run(s), {dt/3600:.2f}h\n{'='*64}")
	print(f"  mean_attitude_error_deg : {err_m:.2f} ± {err_s:.2f}°")
	print(f"  stable_rate             : {stab_m*100:.1f} ± {stab_s*100:.1f}%")
	print(f"  mean_reward             : {rew_m:.2f} ± {rew_s:.2f}")
	print(f"  {'WNN GA-Memory*':<24}: 6.43° / 35% TRAIN-SET   *3-way held-out pending (run_ga_memory.py)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
