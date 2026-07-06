#!/usr/bin/env python3
"""W2.0 — calibrate the disturbance intensity ladder against PID vs PD.

Targets (plan .claude/plans/w2_disturbances.md):
  L1: PID stays ~100% stable (visible err/steady increase only)
  L2: PID holds ≥95% but PD (ki=0) DEGRADES — the separation zone
  L3: PID degrades too (stress tail)

Sweep: {PID, PD} × {OFF, L1, L2, L3} × steps {500, 2000, 5000}, FRESH seeds
(77770001/77770101/424242/313131) × 100 eps, tilt≤5°. Weather reaches PID
automatically: run_episode arms sim.set_disturbance per episode.

Usage: PYTHONPATH=src/wnn /Users/lacg/wnn-venv/bin/python scripts/w2_calibrate.py
Output: per-cell table + verdict per level; JSON to
logs/controller/W2Calibrate_20260706/results.json
"""

import json
import math
import statistics
from dataclasses import replace
from pathlib import Path

from wnn.control.pid import AttitudePID, AttitudePIDConfig, PIDGains
from wnn.control.sim import AttitudeSim
from wnn.control.training import (
	DisturbanceConfig, EpisodeConfig, fitness_function, make_pid_action_fn,
)

FRESH_SEEDS = [77770001, 77770101, 424242, 313131]
EPISODES_PER_SEED = 100
LEVELS = ["OFF", "L1", "L2", "L3"]
STEPS = [500, 2000, 5000]
OUT = Path("logs/controller/W2Calibrate_20260706")

ARMS = {
	"PID": AttitudePIDConfig(),
	# v1 finding (06/07): the stock integrator (ki=0.05, i_clamp=0.5) trims only
	# ~26% of a constant-torque offset — max I contribution 0.025 vs the ~0.06
	# the L3 bias demands. PID+ raises ki×4 and the windup clamp ×4 so the
	# integrator can actually cancel the bias; this arm defines the honest
	# "with-integrator" ceiling for the L2 separation target.
	"PID+": AttitudePIDConfig(
		roll=PIDGains(kp=1.2, ki=0.20, kd=0.30),
		pitch=PIDGains(kp=1.2, ki=0.20, kd=0.30),
		yaw=PIDGains(kp=0.6, ki=0.08, kd=0.20),
		i_clamp=2.0,
	),
	"PD": AttitudePIDConfig(
		roll=PIDGains(kp=1.2, ki=0.0, kd=0.30),
		pitch=PIDGains(kp=1.2, ki=0.0, kd=0.30),
		yaw=PIDGains(kp=0.6, ki=0.0, kd=0.20),
	),
}


def episode_config(steps: int, level: str, seed: int) -> EpisodeConfig:
	dist = DisturbanceConfig.preset(level, seed=seed)
	return EpisodeConfig(
		dt=0.001, steps_per_episode=steps,
		max_initial_tilt_rad=math.radians(5.0),
		max_initial_yaw_rad=math.radians(5.0),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
		disturbance=dist,
	)


def pool(vals):
	m = statistics.mean(vals)
	sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
	return m, sd


def eval_cell(arm_cfg: AttitudePIDConfig, steps: int, level: str) -> dict:
	stables, errs, steadies = [], [], []
	for seed in FRESH_SEEDS:
		pid = AttitudePID(arm_cfg)
		sim = AttitudeSim()
		ec = episode_config(steps, level, seed)
		_, m = fitness_function(make_pid_action_fn(pid), sim, ec,
		                        num_episodes=EPISODES_PER_SEED, seed=seed)
		stables.append(m["stable_rate"] * 100.0)
		errs.append(m["mean_attitude_error_deg"])
		steadies.append(m["mean_steady_error_deg"])
	sm, ssd = pool(stables)
	em, _ = pool(errs)
	tm, _ = pool(steadies)
	return {"stable": sm, "sd": ssd, "err": em, "steady": tm, "per_seed": stables}


def verdict(results: dict) -> list[str]:
	"""Judge each level vs plan targets using the 2000-step cells. The
	with-integrator reference is PID+ (v1 showed the stock ki/i_clamp trims
	only ~26% of a bias offset — too weak to define the separation zone)."""
	ref = "PID+" if "PID+" in results else "PID"
	lines = []
	for lv, tgt in [("L1", f"{ref} ~100, PD may dip"),
	                ("L2", f"{ref} >=95 AND PD degrades (separation)"),
	                ("L3", f"{ref} degrades too")]:
		pid = results[ref][lv][2000]
		pd = results["PD"][lv][2000]
		sep = pid["stable"] - pd["stable"]
		if lv == "L1":
			ok = pid["stable"] >= 99.0
		elif lv == "L2":
			ok = pid["stable"] >= 95.0 and sep >= 5.0
		else:
			ok = pid["stable"] < 95.0
		stock = results["PID"][lv][2000]["stable"] if ref == "PID+" else None
		extra = f"  (stock PID {stock:.1f}%)" if stock is not None else ""
		lines.append(f"{lv}: {ref} {pid['stable']:.1f}% / PD {pd['stable']:.1f}% "
		             f"(sep {sep:+.1f}pp){extra} — target [{tgt}] → {'MET' if ok else 'MISSED'}")
	return lines


def main() -> None:
	OUT.mkdir(parents=True, exist_ok=True)
	print("========== W2.0 disturbance-ladder calibration — PID vs PD (fresh seeds) ==========")
	print(f"fresh seeds {FRESH_SEEDS} x {EPISODES_PER_SEED} eps | levels {LEVELS} | steps {STEPS}\n")
	results = {arm: {lv: {} for lv in LEVELS} for arm in ARMS}
	for arm, cfg in ARMS.items():
		for lv in LEVELS:
			for steps in STEPS:
				r = eval_cell(cfg, steps, lv)
				results[arm][lv][steps] = r
				print(f"[{arm:>3} {lv:>3} @{steps:>4}] stable {r['stable']:5.1f}±{r['sd']:4.1f}%  "
				      f"err {r['err']:5.2f}°  steady {r['steady']:5.2f}°  "
				      f"(per-seed: {'  '.join(f'{s:.0f}' for s in r['per_seed'])})", flush=True)
		print()
	OUT.joinpath("results.json").write_text(json.dumps(
		{a: {l: {str(s): results[a][l][s] for s in STEPS} for l in LEVELS} for a in ARMS}, indent=1))
	print("================= LADDER VERDICT (@2000-step cells) =================")
	for ln in verdict(results):
		print(ln)
	print(f"\nJSON: {OUT}/results.json")


if __name__ == "__main__":
	main()
