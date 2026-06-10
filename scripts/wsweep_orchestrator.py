#!/usr/bin/env python
"""Auto cull-down orchestrator for the phased_ga controller weight sweep.

Flow (fully automatic, writes a report at every round transition):
  Round 1 : all 18 combos, QUICK config (run EXTERNALLY by
            run_weight_sweep_phased.sh — this orchestrator WAITS for it).
  -> rank by MEMORY held-out -> ROUND1_REPORT.txt (full 18 + top-9 survivors)
  Round 2 : top 9, HEAVIER config (pop50/kfold5/more gens), 1 seed.
  -> rank -> ROUND2_REPORT.txt (the 9 + top-3 survivors)
  Round 3 : top 3, HEAVIER config, MULTI-SEED (3 seeds, mean±std).
  -> FINAL_REPORT.txt (the 3 + winner).

Detached so it survives shell/Claude exit. Reports also echo to stdout (the
orchestrator log). Sequential phased_ga runs (no controller thrash; co-resident
with the IDS worker).

Usage: wsweep_orchestrator.py --dir logs/controller/wsweep_phased_20260610
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_weight_sweep import COMBOS, parse_combo  # noqa: E402

WEIGHTS = {name: (e, s, j, m) for (name, e, s, j, m) in COMBOS}

# Per-round phased_ga config. Round 1 is external (quick); 2/3 are heavier.
# Patience HALVED 10/06 after W2's round-2 probe: stages went flat for 30-48
# gens before stopping (172 min/combo) and the longer runway did NOT improve
# held-out (W2: R1 37.5% -> R2 25.0%). Per Luiz: r2 neurons 3 / memory 5,
# r3 neurons 4 / memory 6.
ROUND2_CFG = dict(pop=50, folds=5, steps=800, rg_rounds=4, rg_eps=12, eval_eps=12,
                  uni_eps=6, ngens=50, npat=3, mgens=80, mpat=5, check=5)
ROUND3_CFG = dict(pop=50, folds=5, steps=1000, rg_rounds=5, rg_eps=12, eval_eps=16,
                  uni_eps=8, ngens=60, npat=4, mgens=100, mpat=6, check=5)
ROUND3_SEEDS = [(20260609, 99990001), (20260610, 99990002), (20260611, 99990003)]
KEEP_R1, KEEP_R2 = 9, 3


def log(msg: str):
	print(f"[orch] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def run_phased(name: str, out_dir: Path, cfg: dict, base_seed: int, report_seed: int):
	"""One phased_ga run for a combo into out_dir; returns parsed result dict."""
	out_dir.mkdir(parents=True, exist_ok=True)
	e, s, j, m = WEIGHTS[name]
	env = os.environ.copy()
	env.update(WNN_RUST_DAGGER="1", WNN_STATE_SPLIT="1", RAYON_NUM_THREADS="3",
	           PYTHONPATH="/Users/lacg/wnn/src/wnn")
	cmd = [
		"/Users/lacg/wnn-venv/bin/python", "-u", "tests/run_phased_ga.py",
		"--pop", str(cfg["pop"]), "--num-eval-folds", str(cfg["folds"]),
		"--elitism", "0.2", "--crossover-rate", "0.5",
		"--tilt", "5", "--body-rate", "0.5", "--yaw-rate", "0.3",
		"--lamarckian", "--skip-stages", "bits,connections",
		"--grid-state-neurons", "8", "12", "16", "--grid-bits", "24", "30",
		"--steps", str(cfg["steps"]), "--eval-episodes", str(cfg["eval_eps"]),
		"--universe-episodes", str(cfg["uni_eps"]),
		"--rg-rounds", str(cfg["rg_rounds"]), "--rg-episodes-per-round", str(cfg["rg_eps"]),
		"--neurons-gens", str(cfg["ngens"]), "--neurons-patience", str(cfg["npat"]),
		"--memory-gens", str(cfg["mgens"]), "--memory-patience", str(cfg["mpat"]),
		"--check-interval", str(cfg["check"]), "--saturation-grow-gain", "1.0",
		"--fit-weight-err-sq", str(e), "--fit-weight-stable", str(s),
		"--fit-weight-jerk", str(j), "--fit-weight-mono", str(m),
		"--train-workers", "4", "--base-seed", str(base_seed), "--report-seed", str(report_seed),
		"--save-stage-checkpoints", str(out_dir), "--save-winner", str(out_dir / "winner.pkl"),
	]
	with open(out_dir / "run.out", "w") as f:
		subprocess.run(cmd, cwd="/Users/lacg/wnn", env=env, stdout=f, stderr=subprocess.STDOUT)
	return parse_combo(out_dir / "run.out")


def memory_holdout(parsed: dict):
	"""(err, stable) MEMORY held-out, or None."""
	return parsed.get("ho", {}).get("MEMORY")


def write_report(path: Path, title: str, rows: list, survivors: list):
	"""rows = [(name, weights, mem_ho_or_None, note)], survivors = [names]."""
	lines = [f"  {title}", "  " + "=" * 78,
	         f"  {'combo':<5} {'err':>4} {'stb':>4} {'jrk':>4} {'mno':>4} | "
	         f"{'MEM held-out err':>16} {'MEM held-out stb':>16}  {'note':<12}"]
	for (name, w, mh, note) in rows:
		e, s, j, m = w
		if mh is not None:
			ho = f"{mh[0]:>15.2f}° {mh[1]:>15.1f}%"
		else:
			ho = f"{'—':>16} {'—':>16}"
		lines.append(f"  {name:<5} {e:>4.2f} {s:>4.2f} {j:>4.2f} {m:>4.2f} | {ho}  {note:<12}")
	lines += ["  " + "-" * 78,
	          f"  SURVIVORS → next round ({len(survivors)}): {', '.join(survivors)}",
	          "  ranked by MEMORY held-out stable (then err); held-out = fresh report-seed, matched 5°."]
	text = "\n".join(lines)
	path.write_text(text + "\n")
	print("\n" + text + "\n", flush=True)


def rank_and_cull(results: dict, keep: int):
	"""results: {name: (err,stable) or None}. Returns (ranked_rows_meta, survivors)."""
	scored = [(n, mh) for n, mh in results.items() if mh is not None]
	scored.sort(key=lambda r: (-r[1][1], r[1][0]))   # stable desc, err asc
	survivors = [n for n, _ in scored[:keep]]
	return scored, survivors


def wait_for_round1(base: Path, driver_pid_file="/tmp/wnn_wsweep.pid"):
	"""Block until the external Round-1 driver is done (process gone AND all 18
	combos have a MEMORY held-out OR the combo's run finished)."""
	log("waiting for Round 1 (external driver) to finish…")
	while True:
		done = sum(1 for (n, *_w) in COMBOS if memory_holdout(parse_combo(base / n / "run.out")) is not None
		           or parse_combo(base / n / "run.out")["done"])
		alive = False
		try:
			pid = int(Path(driver_pid_file).read_text().strip())
			os.kill(pid, 0)
			alive = True
		except Exception:
			alive = False
		if not alive and done >= len(COMBOS):
			log(f"Round 1 complete ({done}/{len(COMBOS)}).")
			return
		if not alive and done >= 1:
			# Driver gone but not all combos done → it finished/stopped; proceed with what we have.
			log(f"Round-1 driver gone; {done}/{len(COMBOS)} combos have results. Proceeding.")
			return
		time.sleep(120)


def run_round(combos: list, base: Path, cfg: dict, multiseed: bool, round_name: str):
	"""Run a set of combos through phased_ga; returns {name: (err,stable)|None}.
	Combos (or seeds) whose run.out is already complete are SKIPPED and their
	parsed result reused — so a killed/relaunched orchestrator resumes the round
	without redoing finished work."""
	results = {}
	for name in combos:
		if multiseed:
			vals = []
			for k, (bs, rs) in enumerate(ROUND3_SEEDS):
				prior = parse_combo(base / round_name / name / f"seed{bs}" / "run.out")
				if prior["done"]:
					log(f"{round_name}: {name} seed {k+1}/{len(ROUND3_SEEDS)} already done — reusing")
					mh = memory_holdout(prior)
					if mh:
						vals.append(mh)
					continue
				log(f"{round_name}: {name} seed {k+1}/{len(ROUND3_SEEDS)} (base={bs})")
				p = run_phased(name, base / round_name / name / f"seed{bs}", cfg, bs, rs)
				mh = memory_holdout(p)
				if mh:
					vals.append(mh)
			if vals:
				import statistics
				me = statistics.mean(v[0] for v in vals)
				ms = statistics.mean(v[1] for v in vals)
				results[name] = (me, ms)
			else:
				results[name] = None
		else:
			prior = parse_combo(base / round_name / name / "run.out")
			if prior["done"]:
				log(f"{round_name}: {name} already done — reusing")
				results[name] = memory_holdout(prior)
				continue
			log(f"{round_name}: {name}")
			p = run_phased(name, base / round_name / name, cfg, 20260609, 99990001)
			results[name] = memory_holdout(p)
	return results


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dir", required=True)
	args = ap.parse_args()
	base = Path(args.dir)

	# ---- Round 1: wait for external driver, then rank + report ----
	wait_for_round1(base)
	r1 = {n: memory_holdout(parse_combo(base / n / "run.out")) for (n, *_w) in COMBOS}
	scored1, surv1 = rank_and_cull(r1, KEEP_R1)
	rows1 = [(n, WEIGHTS[n], r1[n], ("✓ survives" if n in surv1 else "culled")) for (n, *_w) in COMBOS]
	rows1.sort(key=lambda r: (r[2] is None, -(r[2][1] if r[2] else -1), (r[2][0] if r[2] else 1e9)))
	write_report(base / "ROUND1_REPORT.txt", "ROUND 1 — all 18 (quick config)", rows1, surv1)

	# ---- Round 2: top 9, heavier ----
	log(f"Round 2 starting with {len(surv1)} survivors: {surv1}")
	r2 = run_round(surv1, base, ROUND2_CFG, multiseed=False, round_name="round2")
	scored2, surv2 = rank_and_cull(r2, KEEP_R2)
	rows2 = [(n, WEIGHTS[n], r2.get(n), ("✓ survives" if n in surv2 else "culled")) for n in surv1]
	rows2.sort(key=lambda r: (r[2] is None, -(r[2][1] if r[2] else -1), (r[2][0] if r[2] else 1e9)))
	write_report(base / "ROUND2_REPORT.txt", "ROUND 2 — top 9 (heavier: pop50/kfold5)", rows2, surv2)

	# ---- Round 3: top 3, heavier + multi-seed ----
	log(f"Round 3 starting with {len(surv2)} survivors: {surv2}")
	r3 = run_round(surv2, base, ROUND3_CFG, multiseed=True, round_name="round3")
	scored3, winner = rank_and_cull(r3, 1)
	rows3 = [(n, WEIGHTS[n], r3.get(n), ("★ WINNER" if n in winner else "")) for n in surv2]
	rows3.sort(key=lambda r: (r[2] is None, -(r[2][1] if r[2] else -1), (r[2][0] if r[2] else 1e9)))
	write_report(base / "FINAL_REPORT.txt", "ROUND 3 FINAL — top 3 (heavier + 3-seed mean)", rows3, winner)
	log(f"SWEEP COMPLETE. Winner: {winner[0] if winner else '(none)'} "
	    f"weights={WEIGHTS.get(winner[0]) if winner else '-'}")


if __name__ == "__main__":
	main()
