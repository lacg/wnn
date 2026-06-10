#!/usr/bin/env python
"""Report a controller weight-sweep (run_curriculum_ga.py --mode sweep) run.

For each combo it shows: combo #, the 4 fitness weights, the LAST during-search
generation's err/stable, the during-search BEST err/stable, the HELD-OUT err/
stable (a fresh-seed re-eval of the saved winner pkl at matched 5° — the honest
number, since the sweep itself only reports during-search), and wall duration.

Held-out is computed only for combos whose winner pkl exists (completed combos);
in-progress combos show their latest generation and 'running'.

Usage:
  python scripts/report_weight_sweep.py --dir logs/controller/wsweep_20260609
  python scripts/report_weight_sweep.py --dir <DIR> --no-heldout   # log-only, fast
"""
from __future__ import annotations

import argparse
import glob
import math
import pickle
import re
from pathlib import Path


COMBO_RE = re.compile(r"# COMBO (\w+): err²=([\d.]+) stable=([\d.]+) jerk=([\d.]+) mono=([\d.]+)")
GEN_RE = re.compile(r"Gen (\d+)/(\d+): .*?stable=([\d.]+)%, err=([\d.]+)°")
BEST_RE = re.compile(r"best: err=([\d.]+)° +stable=([\d.]+)% +reward=([-\d.]+) +iters=(\d+) +wall=([\d.]+)s")


def parse_log(log_path: Path) -> dict:
	"""Parse the round log into {combo_name: {...}} preserving combo order."""
	combos: dict[str, dict] = {}
	order: list[str] = []
	cur = None
	for line in log_path.read_text(errors="ignore").splitlines():
		mc = COMBO_RE.search(line)
		if mc:
			cur = mc.group(1)
			combos[cur] = {
				"weights": (float(mc.group(2)), float(mc.group(3)), float(mc.group(4)), float(mc.group(5))),
				"last_gen": None, "best": None,
			}
			order.append(cur)
			continue
		if cur is None:
			continue
		mg = GEN_RE.search(line)
		if mg:
			combos[cur]["last_gen"] = (int(mg.group(1)), int(mg.group(2)),
			                            float(mg.group(3)), float(mg.group(4)))  # gen, total, stable%, err°
		mb = BEST_RE.search(line)
		if mb:
			combos[cur]["best"] = (float(mb.group(1)), float(mb.group(2)),
			                        float(mb.group(3)), int(mb.group(4)), float(mb.group(5)))  # err, stable%, reward, iters, wall_s
	return {"order": order, "combos": combos}


def heldout_eval(pkl_path: Path, seed: int, episodes: int, steps: int) -> tuple[float, float] | None:
	"""Honest held-out: RE-TRAIN the saved architecture on a fresh seed (the pkl
	stores arch only, cells=None — and re-training IS the controller's
	generalization measure, same as --report-seed) then score at matched 5°.
	Run with WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 to match the sweep's trainer."""
	from wnn.control.evaluator import ControllerEvaluator, fit_thresholds_from_pid_rollouts
	from wnn.control.training import EpisodeConfig
	from wnn.control.reward_gated import RewardGatedConfig
	d = pickle.load(open(pkl_path, "rb"))
	spec, g = d["spec"], d["best_genome"]
	ec = EpisodeConfig(dt=0.001, steps_per_episode=steps,
	                   max_initial_tilt_rad=math.radians(5.0), max_initial_yaw_rad=math.radians(5.0),
	                   max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)
	rg = RewardGatedConfig(num_rounds=3, episodes_per_round=8, steps_per_episode=steps,
	                       eval_episodes=8, seed=seed, episode_config=ec)
	thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed)
	ev = ControllerEvaluator(spec, num_eval_episodes=episodes, seed=seed, episode_config=ec,
	                         thresholds=thr, rg_config=rg, num_eval_folds=1)
	m = ev.evaluate_batch([g])[0]   # trains fresh (splitting if WNN_STATE_SPLIT=1) + scores
	return float(m.mean_attitude_error_deg), float(m.acc) * 100.0


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dir", required=True, help="wsweep dir (contains round*.out + sweep_*_stageA.pkl)")
	ap.add_argument("--heldout", action="store_true",
	                help="ALSO re-train each completed winner on a fresh seed + score (slow; "
	                     "note: the pkl has cells=None so this is a fresh-train architecture "
	                     "generalization number, NOT the GA winner's held-out — for tiny sweep "
	                     "archs it tends to collapse. Use WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1).")
	ap.add_argument("--heldout-seed", type=int, default=987654321)
	ap.add_argument("--heldout-episodes", type=int, default=50)
	ap.add_argument("--heldout-steps", type=int, default=400)
	args = ap.parse_args()

	d = Path(args.dir)
	logs = sorted(glob.glob(str(d / "round*.out"))) or sorted(glob.glob(str(d / "*.out")))
	if not logs:
		raise SystemExit(f"no round*.out in {d}")
	parsed = parse_log(Path(logs[-1]))
	order, combos = parsed["order"], parsed["combos"]

	hdr = (f"{'#':>3} {'combo':<5} {'err':>4} {'stb':>4} {'jrk':>4} {'mno':>4} | "
	       f"{'lastgen':>8} {'lg_err':>7} {'lg_stb':>7} | {'bst_err':>7} {'bst_stb':>7} | {'dur':>7}")
	if args.heldout:
		hdr += f" | {'ho_err':>7} {'ho_stb':>7}"
	print(f"  Weight sweep: {d.name}   ({sum(1 for c in combos.values() if c['best']) }/{len(order)} done, "
	      f"{len(order)} started)")
	print(hdr)
	print("  " + "-" * len(hdr))
	for i, name in enumerate(order, 1):
		c = combos[name]
		we, ws, wj, wm = c["weights"]
		lg = c["last_gen"]
		lg_s = f"{lg[0]:>3}/{lg[1]:<3}" if lg else "   -   "
		lg_err = f"{lg[3]:.2f}°" if lg else "  -  "
		lg_stb = f"{lg[2]:.1f}%" if lg else "  -  "
		bst = c["best"]
		bst_err = f"{bst[0]:.2f}°" if bst else "running"
		bst_stb = f"{bst[1]:.1f}%" if bst else "  -  "
		dur = f"{bst[4]/60:.0f}m" if bst else "  -  "
		ho_err = ho_stb = "  -  "
		if bst and args.heldout:
			pkl = d / f"sweep_{name}_stageA.pkl"
			if pkl.exists():
				try:
					he, hs = heldout_eval(pkl, args.heldout_seed, args.heldout_episodes, args.heldout_steps)
					ho_err, ho_stb = f"{he:.2f}°", f"{hs:.1f}%"
				except Exception as e:
					ho_err = f"err:{type(e).__name__}"
		row = (f"  {i:>3} {name:<5} {we:>4.2f} {ws:>4.2f} {wj:>4.2f} {wm:>4.2f} | "
		       f"{lg_s:>8} {lg_err:>7} {lg_stb:>7} | {bst_err:>7} {bst_stb:>7} | {dur:>7}")
		if args.heldout:
			row += f" | {ho_err:>7} {ho_stb:>7}"
		print(row)
	print("\n  lastgen = last GA generation's population err/stable; bst = during-search best-FITNESS")
	print("  genome (harmonic err²+stable+jerk+mono, so not the most-stable genome).")
	if args.heldout:
		print("  ho = fresh-train (seed 987654321) re-eval — pkl has cells=None so this loses the GA-refined")
		print("  cells (≈ architecture-generalization lower bound, tends to collapse for tiny sweep archs).")
	else:
		print("  (pass --heldout to add a fresh-train re-eval column; not a true winner held-out — see note.)")


if __name__ == "__main__":
	main()
