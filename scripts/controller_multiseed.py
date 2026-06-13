#!/usr/bin/env python3
"""Multi-seed launcher for the phased-GA drone controller (reusable).

Why a launcher and not `phased_ga --runs N`:
  --runs reuses ONE save dir (clobbers each seed's winner) and reports the
  OPTIMISTIC during-search Stage-4 metric. Proper multi-seed wants:
    * one save dir PER seed (winners + stage checkpoints preserved), and
    * the held-out RESULT line (--report-seed) as the per-seed number to average.

Methodology (matches CLAUDE.md "trust held-out, not the gen-line"):
  * base-seed VARIES per run  → different train/select episode draws (the thing
    whose variance we want to measure).
  * report-seed is FIXED across runs → every seed graded on the IDENTICAL
    held-out test pool, so the spread is optimizer variance, not test variance.

Each seed is launched fully detached (own session, PPID=1) so it survives
Claude /exit. Concurrency is the caller's choice via --stagger / load; this just
fires them. Aggregate afterwards from each seed dir's run.out FINAL_REPORT /
held-out line — do NOT mix in runs with a different episode budget.

Usage (locked 13/06 config is the default — just give seeds + an out root):
  python scripts/controller_multiseed.py \
    --out-root logs/controller/c10_multiseed_20260613 \
    --base-seeds 31337001 31337002 31337003 \
    --report-seed 99990101
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def build_phased_cmd(args, base_seed: int, save_dir: Path) -> list[str]:
	"""One phased_ga invocation for a single seed — the locked C10 recipe with
	per-stage episode allocation. Everything is explicit (no hidden defaults) so
	the command in the manifest fully reproduces the run."""
	save_dir.mkdir(parents=True, exist_ok=True)
	return [
		sys.executable, "-u", "-m", "wnn.control.phased_ga",
		# --- architecture / grid (Stage 0) ---
		"--grid-state-neurons", *map(str, args.grid_state_neurons),
		"--grid-bits", *map(str, args.grid_bits),
		"--levels", str(args.levels),
		"--skip-stages", "bits,connections",     # grid → NEURONS → MEMORY
		"--lamarckian",
		"--saturation-grow-gain", str(args.sat_grow_gain),
		# --- per-stage GA runway ---
		"--neurons-gens", str(args.neurons_gens), "--neurons-patience", str(args.neurons_patience),
		"--memory-gens", str(args.memory_gens), "--memory-patience", str(args.memory_patience),
		"--pop", str(args.pop), "--num-eval-folds", str(args.folds),
		"--check-interval", str(args.check_interval),
		# --- episode allocation (the 13/06 finding: NEURONS 50 / MEMORY 100) ---
		"--eval-episodes", str(args.eval_episodes),
		"--memory-eval-episodes", str(args.memory_eval_episodes),
		"--steps", str(args.steps), "--tilt", str(args.tilt),
		# --- C10 fitness weights (err .40 / stable .30 / jerk .20 / mono .10) ---
		"--fit-weight-err-sq", str(args.w_err), "--fit-weight-stable", str(args.w_stable),
		"--fit-weight-jerk", str(args.w_jerk), "--fit-weight-mono", str(args.w_mono),
		# --- held-out report (FIXED seed across all runs) ---
		"--report-seed", str(args.report_seed), "--report-episodes", str(args.report_episodes),
		"--holdout-pop-sample", str(args.holdout_pop_sample),
		# --- this seed's train/select draw + outputs ---
		"--base-seed", str(base_seed), "--runs", "1",
		"--save-winner", str(save_dir / "winner.yaml.gz"),
		"--save-stage-checkpoints", str(save_dir),
	]


def launch_detached(cmd: list[str], logfile: Path, cwd: Path, env: dict) -> int:
	"""Fire a command in its own session (PPID=1) → survives Claude /exit."""
	logf = open(logfile, "ab", buffering=0)
	p = subprocess.Popen(cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT,
	                     stdin=subprocess.DEVNULL, start_new_session=True, env=env)
	return p.pid


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__,
	                            formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--out-root", required=True, help="parent dir; one seed<N>/ subdir per run")
	ap.add_argument("--base-seeds", type=int, nargs="+", required=True,
	                help="distinct train/select seeds (≥3 for a real multi-seed mean)")
	ap.add_argument("--report-seed", type=int, default=99990101,
	                help="FIXED held-out seed shared by all runs (identical test pool)")
	ap.add_argument("--stagger", type=float, default=3.0, help="seconds between launches")
	# Locked 13/06 config (overridable but defaults reproduce the approved run).
	ap.add_argument("--grid-state-neurons", type=int, nargs="+", default=[8, 12, 16])
	ap.add_argument("--grid-bits", type=int, nargs="+", default=[24, 30])
	ap.add_argument("--levels", type=int, default=16)
	ap.add_argument("--sat-grow-gain", type=float, default=1.0)
	ap.add_argument("--neurons-gens", type=int, default=60)
	ap.add_argument("--neurons-patience", type=int, default=5)
	ap.add_argument("--memory-gens", type=int, default=120)
	ap.add_argument("--memory-patience", type=int, default=8)
	ap.add_argument("--pop", type=int, default=50)
	ap.add_argument("--folds", type=int, default=5)
	ap.add_argument("--check-interval", type=int, default=5)
	ap.add_argument("--eval-episodes", type=int, default=50)
	ap.add_argument("--memory-eval-episodes", type=int, default=100)
	ap.add_argument("--steps", type=int, default=1000)
	ap.add_argument("--tilt", type=float, default=5.0)
	ap.add_argument("--report-episodes", type=int, default=100)
	ap.add_argument("--holdout-pop-sample", type=int, default=8)
	ap.add_argument("--w-err", type=float, default=0.40)
	ap.add_argument("--w-stable", type=float, default=0.30)
	ap.add_argument("--w-jerk", type=float, default=0.20)
	ap.add_argument("--w-mono", type=float, default=0.10)
	ap.add_argument("--rayon-threads", type=int, default=3, help="RAYON_NUM_THREADS per run")
	args = ap.parse_args()

	repo = Path(__file__).resolve().parent.parent
	out_root = Path(args.out_root)
	out_root.mkdir(parents=True, exist_ok=True)

	env = dict(os.environ)
	env["WNN_RUST_DAGGER"] = "1"
	env["WNN_STATE_SPLIT"] = "1"
	env["RAYON_NUM_THREADS"] = str(args.rayon_threads)
	env["PYTHONPATH"] = f"{repo / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")

	manifest = out_root / "manifest.txt"
	lines = [f"# controller multi-seed launched (report-seed FIXED = {args.report_seed})"]
	for i, base_seed in enumerate(args.base_seeds):
		seed_dir = out_root / f"seed{i}_base{base_seed}"
		cmd = build_phased_cmd(args, base_seed, seed_dir)
		logfile = seed_dir / "run.out"
		pid = launch_detached(cmd, logfile, repo, env)
		row = f"seed{i} base={base_seed} pid={pid} dir={seed_dir}"
		lines.append(row)
		lines.append("  cmd: " + " ".join(cmd))
		print(row)
		if i < len(args.base_seeds) - 1:
			time.sleep(args.stagger)
	manifest.write_text("\n".join(lines) + "\n")
	print(f"\nmanifest: {manifest}")
	print("aggregate held-out from each seed dir's run.out FINAL_REPORT line "
	      "(exclude any run with a different episode budget).")
	return 0


if __name__ == "__main__":
	sys.exit(main())
