#!/usr/bin/env python3
"""PID ki=0 ablation — does the I-term explain the PID-vs-WNN stability gap?

Motivation (01/07/2026): the 'missing integrator' diagnosis for the WNN
controller's ~84% vs PID ~98% stable gap has failed 4 independent fix
attempts (delta accumulator, obs integral features, pidmix, --state-integral
A/B). Sajus's quadcopter PID baseline runs KI=0 on all loops and is still
'very stable'. If OUR PID with ki=0 stays ~98%, integral action was never
the differentiator and the diagnosis needs re-examination; if it drops to
~85% ≈ WNN, the diagnosis is confirmed and quantified.

Protocol: exactly the state-integral A/B/C held-out setup —
tilt 5.0deg (bounds tilt AND yaw), body-rate 0.5, yaw-rate 0.3, steps 500,
dt 1ms, seeds {99990001, 99990101, 12345, 67890} x 100 episodes.
Same single-PID-instance-across-episodes convention as
evaluator.evaluate_pid_baseline (no per-episode reset).

Usage: PYTHONPATH=src/wnn /Users/lacg/wnn-venv/bin/python scripts/pid_ki_ablation.py
"""

import math

from wnn.control.pid import AttitudePID, AttitudePIDConfig, PIDGains
from wnn.control.sim import AttitudeSim
from wnn.control.training import EpisodeConfig, fitness_function, make_pid_action_fn

REPORT_SEEDS = [99990001, 99990101, 12345, 67890]
EPISODES_PER_SEED = 100

ARMS = {
	"default (ki .05/.05/.02)": AttitudePIDConfig(),
	"PD-only (ki=0 all axes)": AttitudePIDConfig(
		roll=PIDGains(kp=1.2, ki=0.0, kd=0.30),
		pitch=PIDGains(kp=1.2, ki=0.0, kd=0.30),
		yaw=PIDGains(kp=0.6, ki=0.0, kd=0.20),
	),
}


def episode_config() -> EpisodeConfig:
	# Mirror phased_ga.py:1789 for --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 --steps 500
	return EpisodeConfig(
		dt=0.001, steps_per_episode=500,
		max_initial_tilt_rad=math.radians(5.0),
		max_initial_yaw_rad=math.radians(5.0),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
	)


def eval_arm(cfg: AttitudePIDConfig) -> list[dict]:
	ec = episode_config()
	per_seed = []
	for seed in REPORT_SEEDS:
		pid = AttitudePID(cfg)  # fresh per seed; carried across episodes within a seed (baseline convention)
		sim = AttitudeSim()
		_, metrics = fitness_function(
			make_pid_action_fn(pid), sim, ec,
			num_episodes=EPISODES_PER_SEED, seed=seed,
		)
		per_seed.append({"seed": seed, **metrics})
	return per_seed


def fmt_pct(x: float) -> str:
	return f"{100.0 * x:5.1f}%"


def pooled(vals: list[float]) -> str:
	n = len(vals)
	mean = sum(vals) / n
	sd = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
	return mean, sd


def main() -> None:
	print("========== PID ki=0 ablation — A/B/C held-out protocol ==========")
	print(f"tilt<=5.0deg (tilt+yaw), body<=0.5, yaw-rate<=0.3, steps=500, dt=1ms")
	print(f"seeds={REPORT_SEEDS} x {EPISODES_PER_SEED} eps  |  PID carried across eps (baseline convention)")
	print()
	for name, cfg in ARMS.items():
		rows = eval_arm(cfg)
		print(f"--- {name} ---")
		stables, errs, steadies = [], [], []
		for r in rows:
			stable = r.get("stable_rate")
			err = r.get("mean_attitude_error_deg")
			steady = r.get("mean_steady_error_deg")
			stables.append(stable); errs.append(err); steadies.append(steady)
			print(f"  seed {r['seed']:>9}: stable {fmt_pct(stable)}  err {err:5.2f}deg  steady {steady:5.2f}deg")
		sm, ssd = pooled(stables)
		em, esd = pooled(errs)
		tm, tsd = pooled(steadies)
		print(f"  POOL (n={len(rows)}): stable {fmt_pct(sm)} +-{100*ssd:.1f}  err {em:.2f}+-{esd:.2f}deg  steady {tm:.2f}+-{tsd:.2f}deg")
		print()
	print("Read: PD-only ~= default (~98%) => I-term NOT the differentiator (diagnosis wrong).")
	print("      PD-only drops to ~85% ~= WNN => missing-integrator diagnosis CONFIRMED.")


if __name__ == "__main__":
	main()
