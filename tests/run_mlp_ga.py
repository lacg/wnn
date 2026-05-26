"""GA-evolved MLP — the FAIR substrate comparison to the WNN.

Evolves a small MLP's weights with the SAME optimization scheme as the WNN
(evolution against closed-loop reward, no gradient) on the SAME sim/ICs/reward/
scorer, with the SAME IMU inputs [gyro, accel, target]. This isolates substrate
(weighted vs weightless) under identical optimization — unlike the gradient/BC
baseline (run_mlp_baseline.py), which suffered behavioral-cloning distribution
shift. CPU-only (numpy MLP + Rust sim), so it doesn't contend with the GPU.

Run:  python tests/run_mlp_ga.py --pop 100 --gens 500 --hidden 32 \
        --eval-episodes 10 --sigma 0.05 --steps 1500 --tilt 15 --seed 0
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from ram_accelerator import AttitudeSim
from wnn.control.pid import AttitudePID, AttitudePIDConfig
from wnn.control.training import EpisodeConfig, _sample_initial_state, make_pid_action_fn
from wnn.control.dagger import eval_closed_loop_reset

TARGET = (0.0, 0.0, 0.0)
LAYERS = None  # set from --hidden: [(9,h),(h,h),(h,4)]


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


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--pop", type=int, default=100)
	ap.add_argument("--gens", type=int, default=500)
	ap.add_argument("--hidden", type=int, default=32)
	ap.add_argument("--elite-frac", type=float, default=0.2)
	ap.add_argument("--sigma", type=float, default=0.05)       # mutation stddev
	ap.add_argument("--eval-episodes", type=int, default=10)   # during evolution (cheap)
	ap.add_argument("--final-episodes", type=int, default=20)  # report (matches WNN runs)
	ap.add_argument("--steps", type=int, default=1500)
	ap.add_argument("--tilt", type=float, default=15.0)
	ap.add_argument("--seed", type=int, default=0)
	args = ap.parse_args()
	t0 = time.time()

	ec = EpisodeConfig(dt=0.001, steps_per_episode=args.steps,
	                   max_initial_tilt_rad=math.radians(args.tilt),
	                   max_initial_yaw_rad=math.radians(args.tilt),
	                   max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)
	mean, std = _input_stats(ec, 6, args.seed + 1234)
	dim = _param_count(args.hidden)
	rng = np.random.default_rng(args.seed)
	pop = [_xavier_init(args.hidden, rng) for _ in range(args.pop)]
	n_elite = max(1, int(args.elite_frac * args.pop))
	sigma = args.sigma

	def fitness(w, n_ep, seed):
		r, _m = eval_closed_loop_reset(make_np_mlp_action_fn(w, args.hidden, mean, std),
		                               lambda: None, ec, n_ep, seed)
		return r

	print(f"GA-evolved MLP 9→{args.hidden}→{args.hidden}→4 ({dim} weights), pop {args.pop} × gens {args.gens}, "
	      f"σ={sigma}, elite {args.elite_frac:.0%}, IMU inputs. CPU-only.")
	best_w, best_fit = None, -math.inf
	for gen in range(args.gens):
		fits = np.array([fitness(w, args.eval_episodes, args.seed) for w in pop])
		order = np.argsort(-fits)  # descending (higher reward better)
		elite = [pop[i] for i in order[:n_elite]]
		if fits[order[0]] > best_fit:
			best_fit, best_w = float(fits[order[0]]), elite[0].copy()
		# next gen: elites + Gaussian-mutated children of random elites
		children = [elite[int(rng.integers(0, n_elite))] + rng.normal(0.0, sigma, dim)
		            for _ in range(args.pop - n_elite)]
		pop = elite + children
		if (gen + 1) % 10 == 0 or gen == 0:
			el = time.time() - t0
			print(f"  Gen {gen+1:0{len(str(args.gens))}d}/{args.gens}: best_reward={best_fit:.2f}, "
			      f"gen_best={fits[order[0]]:.2f}, avg={fits.mean():.2f} "
			      f"[{el:.0f}s, {(gen+1)/el:.1f} gen/s, ETA {(args.gens-gen-1)*el/(gen+1)/60:.0f}m]")

	# Final report on the held-out 20-episode set (seed 0) — comparable to WNN/PID.
	_, m = eval_closed_loop_reset(make_np_mlp_action_fn(best_w, args.hidden, mean, std),
	                              lambda: None, ec, args.final_episodes, args.seed)
	dt = time.time() - t0
	print(f"\n{'='*60}\n  GA-EVOLVED MLP — final ({dt/3600:.2f}h)\n{'='*60}")
	print(f"  {'policy':<24} {'mean_err':>8} {'stable':>6} {'reward':>10}")
	print(f"  {'GA-MLP (IMU, evolved)':<24} {m['mean_attitude_error_deg']:>7.2f}°  {m['stable_rate']*100:>5.0f}%  {m['mean_reward']:>9.2f}")
	print(f"  {'WNN GA-Memory (ref)*':<24} {6.43:>7.2f}°  {35:>5.0f}%  {-27.94:>9.2f}   *pop150×3000")
	print(f"  {'PID (ref, sees q)':<24} {2.49:>7.2f}°  {100:>5.0f}%  {-9.58:>9.2f}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
