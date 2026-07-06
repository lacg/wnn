#!/usr/bin/env python3
"""W1 common-ruler leg — each horizon-surface winner solo at multiples of its
OWN training horizon: {0.5x, 1x, 2.5x, 5x, 10x, 20x} × H.

The W1 surface (05/07) used own-horizon rulers, so its 4 points are not on a
common ruler. This leg produces the full decay CURVE per winner: where does
each recipe's stability cliff sit relative to its trained horizon? (Finding 3
mechanism predicts a cliff at ~2.5× trained H for singles.)

Protocol matches e4_best_of_k.py: FRESH seeds (77770001/77770101/424242/
313131) × 100 episodes, tilt≤5°, thresholds re-fit per seed. Read-only.

Usage: PYTHONPATH=src/wnn /Users/lacg/wnn-venv/bin/python scripts/w1_common_ruler.py
Output: matrix on stdout + JSON at logs/controller/E4Chain_20260706/common_ruler.json
"""

import gc
import json
import math
import statistics
from pathlib import Path

from wnn.control.checkpoint_io import load_controller_checkpoint
from wnn.control.evaluator import ControllerEvaluator, fit_thresholds_from_pid_rollouts
from wnn.control.training import EpisodeConfig

FRESH_SEEDS = [77770001, 77770101, 424242, 313131]
EPISODES_PER_SEED = 100
MULTIPLES = [0.5, 1.0, 2.5, 5.0, 10.0, 20.0]

# (label, trained horizon H, path)
WINNERS = [
	("e2_imm_s09",    500, "logs/controller/E2Reliability_20260702/IMM_seed20260609/winner.yaml.gz"),
	("e2_imm_s10",    500, "logs/controller/E2Reliability_20260702/IMM_seed20260610/winner.yaml.gz"),
	("w1_h1000_s09", 1000, "logs/controller/W1Surface_20260702/H1000_seed20260609/winner.yaml.gz"),
	("w1_h1000_s10", 1000, "logs/controller/W1Surface_20260702/H1000_seed20260610/winner.yaml.gz"),
	("e2_long_s09",  2000, "logs/controller/E2Reliability_20260702/LONG_seed20260609/winner.yaml.gz"),
	("e2_long_s10",  2000, "logs/controller/E2Reliability_20260702/LONG_seed20260610/winner.yaml.gz"),
	("w1_h4000_s09", 4000, "logs/controller/W1Surface_20260702/H4000_seed20260609/winner.yaml.gz"),
	("w1_h4000_s10", 4000, "logs/controller/W1Surface_20260702/H4000_seed20260610/winner.yaml.gz"),
]

OUT_JSON = Path("logs/controller/E4Chain_20260706/common_ruler.json")


def episode_config(steps: int) -> EpisodeConfig:
	return EpisodeConfig(
		dt=0.001, steps_per_episode=steps,
		max_initial_tilt_rad=math.radians(5.0),
		max_initial_yaw_rad=math.radians(5.0),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
	)


def pool(vals):
	m = statistics.mean(vals)
	sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
	return m, sd


def score_at(spec, genome, steps: int):
	ec = episode_config(steps)
	rows = []
	for rs in FRESH_SEEDS:
		thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=rs)
		ev = ControllerEvaluator(spec, num_eval_episodes=EPISODES_PER_SEED, seed=rs,
		                         episode_config=ec, thresholds=thr)
		m = ev.score_genomes([genome])[0]
		rows.append({
			"seed": rs, "stable": m.acc * 100.0,
			"err": m.mean_attitude_error_deg,
			"steady": getattr(m, "mean_steady_error_deg", float("nan")),
		})
		del ev
		gc.collect()
	sm, ssd = pool([r["stable"] for r in rows])
	em, _ = pool([r["err"] for r in rows])
	tm, _ = pool([r["steady"] for r in rows])
	return {"steps": steps, "stable": sm, "sd": ssd, "err": em, "steady": tm,
	        "per_seed": [r["stable"] for r in rows]}


def main() -> None:
	OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
	print("========== W1 common-ruler leg — solo decay curves (fresh seeds) ==========")
	print(f"fresh seeds {FRESH_SEEDS} x {EPISODES_PER_SEED} eps | multiples {MULTIPLES} x own H\n")
	results = {}
	for label, horizon, path in WINNERS:
		payload = load_controller_checkpoint(path)
		if payload is None or getattr(payload.get("best_genome"), "cells", None) is None:
			print(f"[{label}] LOAD FAILED / no cells — skipping", flush=True)
			continue
		spec, genome = payload["spec"], payload["best_genome"]
		print(f"[{label}] H={horizon} — {path}", flush=True)
		curve = []
		for mult in MULTIPLES:
			steps = int(horizon * mult)
			r = score_at(spec, genome, steps)
			r["mult"] = mult
			curve.append(r)
			print(f"  {mult:>4.1f}x (steps={steps:>6}): stable {r['stable']:5.1f}±{r['sd']:4.1f}%  "
			      f"err {r['err']:.2f}°  steady {r['steady']:.2f}°  "
			      f"(per-seed: {'  '.join(f'{s:.0f}' for s in r['per_seed'])})", flush=True)
		results[label] = {"horizon": horizon, "curve": curve}
		OUT_JSON.write_text(json.dumps(results, indent=1))  # checkpoint after each winner
	print("\n================= DECAY MATRIX (stable% pooled) =================")
	hdr = "winner (H)          " + "".join(f"{m:>8.1f}x" for m in MULTIPLES)
	print(hdr)
	for label, horizon, _ in WINNERS:
		if label not in results:
			continue
		cells = "".join(f"{c['stable']:>8.1f} " for c in results[label]["curve"])
		print(f"{label:<15} ({horizon:>4}) {cells}")
	print(f"\nJSON: {OUT_JSON}")


if __name__ == "__main__":
	main()
