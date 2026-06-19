"""Weighted-NN (MLP) baseline on the SAME attitude-control task — for an honest
apples-to-apples table with PID and the WNN.

Fairness controls:
  - Same sim (AttitudeSim), same ICs (tilt/rates), same reward + scorer
    (eval_closed_loop_reset) as the PID / WNN runs.
  - The MLP sees the SAME inputs as the WNN — IMU only: [gyro(3), accel(3),
    target_rpy(3)] = 9 dims. It does NOT get the true quaternion q (PID does),
    so MLP-vs-WNN compares substrates on equal information; PID has the q edge.
  - Trained the NORMAL weighted-NN way: gradient descent (Adam) via behavioral
    cloning of PID over PID-driven rollouts. This measures the weighted-NN
    representational ceiling on this task from IMU — isolating "substrate" from
    "no-training" in the WNN paradigm-B result.

Run:  python tests/run_mlp_baseline.py --episodes 40 --steps 1500 --tilt 15 \
        --eval-episodes 20 --epochs 300 --hidden 64 --seed 0
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn

from wnn.control._accel import AttitudeSim
from wnn.control.pid import AttitudePID, AttitudePIDConfig
from wnn.control.training import EpisodeConfig, _sample_initial_state, make_pid_action_fn
from wnn.control.dagger import eval_closed_loop_reset

TARGET = (0.0, 0.0, 0.0)  # level setpoint — matches the PID/WNN runs


class MLP(nn.Module):
	def __init__(self, hidden: int = 64):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(9, hidden), nn.ReLU(),
			nn.Linear(hidden, hidden), nn.ReLU(),
			nn.Linear(hidden, 4), nn.Sigmoid(),  # 4 motor PWMs in [0,1]
		)

	def forward(self, x):
		return self.net(x)


def collect_pid_dataset(ec: EpisodeConfig, n_ep: int, seed: int):
	"""(state, PID action) pairs over PID rollouts. State = WNN's IMU inputs."""
	sim, pid = AttitudeSim(), AttitudePID(AttitudePIDConfig())
	rng = np.random.default_rng(seed)
	X, Y = [], []
	for _ in range(n_ep):
		ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		q0, om0 = _sample_initial_state(ep_rng, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
		                                ec.max_initial_body_rate, ec.max_initial_yaw_rate)
		sim.reset(q=list(q0), omega=list(om0)); pid.reset()
		for _t in range(ec.steps_per_episode):
			if sim.is_unstable():
				break
			gyro, accel = sim.read_imu()
			pwm = pid.step(sim.quaternion, gyro, TARGET)
			X.append([*gyro, *accel, *TARGET]); Y.append(list(pwm))
			sim.step(list(pwm))
	return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def make_mlp_action_fn(model, mean, std):
	m = torch.from_numpy(mean); s = torch.from_numpy(std)
	model.eval()
	def fn(gyro, accel, target_rpy, q):
		x = torch.tensor([*gyro, *accel, *target_rpy], dtype=torch.float32)
		with torch.no_grad():
			out = model((x - m) / s)
		return tuple(float(v) for v in out)
	return fn


def train_mlp(X, Y, hidden, epochs, seed):
	torch.manual_seed(seed)
	mean, std = X.mean(0), X.std(0)
	std[std < 1e-6] = 1.0  # constant features (target) → no scaling
	Xn = torch.from_numpy((X - mean) / std)
	Yt = torch.from_numpy(Y)
	model = MLP(hidden)
	opt = torch.optim.Adam(model.parameters(), lr=1e-3)
	loss_fn = nn.MSELoss()
	n, bs = len(Xn), 512
	for ep in range(epochs):
		perm = torch.randperm(n)
		for i in range(0, n, bs):
			idx = perm[i:i + bs]
			opt.zero_grad()
			loss = loss_fn(model(Xn[idx]), Yt[idx])
			loss.backward(); opt.step()
	with torch.no_grad():
		final = float(loss_fn(model(Xn), Yt))
	return model, mean, std, final


def _row(name, m):
	return f"  {name:<22} {m['mean_attitude_error_deg']:>7.2f}°  {m['stable_rate']*100:>5.0f}%  {m['mean_reward']:>9.2f}"


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--episodes", type=int, default=40, help="PID rollouts for the imitation dataset")
	ap.add_argument("--steps", type=int, default=1500)
	ap.add_argument("--tilt", type=float, default=15.0)
	ap.add_argument("--eval-episodes", type=int, default=20)
	ap.add_argument("--epochs", type=int, default=300)
	ap.add_argument("--hidden", type=int, default=64)
	ap.add_argument("--seed", type=int, default=0)
	args = ap.parse_args()
	t0 = time.time()

	ec = EpisodeConfig(dt=0.001, steps_per_episode=args.steps,
	                   max_initial_tilt_rad=math.radians(args.tilt),
	                   max_initial_yaw_rad=math.radians(args.tilt),
	                   max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)

	# Collect on a DIFFERENT seed than eval (no train/eval IC leakage).
	X, Y = collect_pid_dataset(ec, args.episodes, seed=args.seed + 1234)
	n_params = lambda mdl: sum(p.numel() for p in mdl.parameters())
	model, mean, std, final_loss = train_mlp(X, Y, args.hidden, args.epochs, args.seed)
	print(f"MLP 9→{args.hidden}→{args.hidden}→4 ({n_params(model)} params), "
	      f"imitation BC of PID: {len(X)} samples, {args.epochs} epochs, final MSE={final_loss:.2e}")

	# Score everything on the SAME held-out eval set (seed = args.seed).
	pid = AttitudePID(AttitudePIDConfig())
	_, pid_m = eval_closed_loop_reset(make_pid_action_fn(pid), pid.reset, ec, args.eval_episodes, args.seed)
	rnd = MLP(args.hidden)  # untrained (random weights) reference
	_, rnd_m = eval_closed_loop_reset(make_mlp_action_fn(rnd, mean, std), lambda: None, ec, args.eval_episodes, args.seed)
	_, mlp_m = eval_closed_loop_reset(make_mlp_action_fn(model, mean, std), lambda: None, ec, args.eval_episodes, args.seed)

	print(f"\n{'='*60}\n  ATTITUDE-CONTROL BASELINE TABLE (lower err / higher stable = better)\n{'='*60}")
	print(f"  {'policy':<22} {'mean_err':>8} {'stable':>6} {'reward':>10}")
	print(_row("PID (teacher, sees q)", pid_m))
	print(_row("MLP random (untrained)", rnd_m))
	print(_row("MLP imitation (IMU)", mlp_m))
	print(f"  {'-'*54}")
	print(f"  {'WNN paradigm-B*':<22} {6.43:>7.2f}°  {35:>5.0f}%  {-27.94:>9.2f}   *prior pop150×3000 run")
	print(f"\n  (MLP + WNN both use IMU only [gyro,accel,target]; PID also sees q.)")
	print(f"  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
	main()
