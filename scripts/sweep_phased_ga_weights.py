"""Sweep phased-GA fitness weights across the 4-metric (err²/stable/jerk/mono) space.

Weight grid (user-requested 29/05/2026):
  err     ∈ {0.1, 0.3, 0.5, 0.7, 0.8}
  stable  ∈ {0.2, 0.4}
  mono    ∈ {0.1, 0.2}
  jerk    = 1 - err - stable - mono  (constraint: weights sum to 1.0)

Filters: jerk > 0 (positive weight required, otherwise the metric is silently
inactive). Combinations where err+stable+mono ≥ 1.0 are skipped.

Per-run config (small + quick): pop=40, 100 gens per stage, --eval-episodes 3,
--steps 300, --pop 40, --rg-rounds 2 (= ~10-12 min per run). 4 stages ≈ ~45 min.

Total expected: ~10 valid combos × ~45 min = ~7.5h sequential. Use --max-runs
to limit (e.g. --max-runs 3 to start). All runs serialized — no concurrent
controllers (M4 Max would thrash with multiple Rust+Rayon controllers).

Usage:
  python scripts/sweep_phased_ga_weights.py                # preview combos
  python scripts/sweep_phased_ga_weights.py --execute      # run all valid combos
  python scripts/sweep_phased_ga_weights.py --execute --max-runs 3
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path


WEIGHTS_ERR    = [0.2, 0.3, 0.4, 0.5]
WEIGHTS_STABLE = [0.2, 0.3, 0.4, 0.5]
WEIGHTS_JERK   = [0.1, 0.2]
WEIGHTS_MONO   = [0.1, 0.2]

# Per-stage budget. SMALL config for fast sweep iteration (29/05/2026).
# Idea: identify the BEST weight combo quickly under a constrained arch,
# then re-run the winner at full-fledge config (wider sn/b grid, more gens,
# larger pop) for paper-grade results.
PER_STAGE_GENS  = 100
PER_STAGE_PATIENCE = 5     # patience=5 × check_interval=10 = 50 raw gens stagnant → early-stop
MEMORY_GENS     = 200
MEMORY_PATIENCE = 10
POP             = 100
EVAL_EPISODES   = 3
STEPS           = 300
UNIVERSE_EPS    = 2
RG_ROUNDS       = 2
RG_EPS_PER_RND  = 4
RG_EVAL_EPS     = 3
TRAIN_WORKERS   = 4

# Constrained architecture (small / fast). The Stage-0 grid picks from these.
GRID_STATE_NEURONS = [4, 8]
GRID_BITS          = [10, 16]
LEVELS             = 12

LOG_DIR = Path("/Users/lacg/wnn/logs/controller/sweep_weights")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def enumerate_combos() -> list[dict]:
	"""Enumerate 4D grid (err × stable × jerk × mono) filtered to exact
	sum=1.0. User-corrected 29/05/2026: jerk is an EXPLICIT knob from
	WEIGHTS_JERK (not derived), keeping all four weights within their
	intended ranges."""
	combos = []
	for err, stable, jerk, mono in itertools.product(
		WEIGHTS_ERR, WEIGHTS_STABLE, WEIGHTS_JERK, WEIGHTS_MONO,
	):
		if abs((err + stable + jerk + mono) - 1.0) > 1e-9:
			continue
		combos.append({"err": err, "stable": stable, "jerk": jerk, "mono": mono})
	return combos


def run_one(combo: dict, run_idx: int, dry: bool) -> dict:
	tag = f"e{int(combo['err']*10):02d}_s{int(combo['stable']*10):02d}_j{int(combo['jerk']*10):02d}_m{int(combo['mono']*10):02d}"
	ts  = time.strftime("%Y%m%d_%H%M%S")
	log = LOG_DIR / f"sweep_{tag}_{ts}.log"
	cmd = [
		sys.executable, "-u", "tests/run_phased_ga.py",
		"--grid-state-neurons", *[str(x) for x in GRID_STATE_NEURONS],
		"--grid-bits",          *[str(x) for x in GRID_BITS],
		"--levels", str(LEVELS),
		"--grid-min-suffix", "4",
		"--neurons-gens", str(PER_STAGE_GENS), "--neurons-patience", str(PER_STAGE_PATIENCE),
		"--bits-gens",    str(PER_STAGE_GENS), "--bits-patience",    str(PER_STAGE_PATIENCE),
		"--conns-gens",   str(PER_STAGE_GENS), "--conns-patience",   str(PER_STAGE_PATIENCE),
		"--memory-gens",  str(MEMORY_GENS),    "--memory-patience",  str(MEMORY_PATIENCE),
		"--pop", str(POP),
		"--elitism", "0.2", "--crossover-rate", "0.5",
		"--eval-episodes", str(EVAL_EPISODES),
		"--steps", str(STEPS),
		"--tilt", "15",
		"--universe-episodes", str(UNIVERSE_EPS),
		"--rg-rounds", str(RG_ROUNDS),
		"--rg-episodes-per-round", str(RG_EPS_PER_RND),
		"--rg-eval-episodes", str(RG_EVAL_EPS),
		"--train-workers", str(TRAIN_WORKERS),
		"--fit-weight-err-sq", str(combo["err"]),
		"--fit-weight-stable", str(combo["stable"]),
		"--fit-weight-jerk",   str(combo["jerk"]),
		"--fit-weight-mono",   str(combo["mono"]),
		"--base-seed", "20260529",
	]
	print(f"\n[{run_idx}] tag={tag}  err={combo['err']:.2f} stable={combo['stable']:.2f} "
	      f"jerk={combo['jerk']:.2f} mono={combo['mono']:.2f}  log={log.name}")
	if dry:
		print("  (dry-run — would launch the above)")
		return {"tag": tag, "log": str(log), "duration_s": 0, "returncode": -1}
	t0 = time.time()
	env = os.environ.copy()
	env["WNN_RUST_DAGGER"]    = "1"
	env["RAYON_NUM_THREADS"]  = "8"
	env["PYTHONPATH"]         = "/Users/lacg/wnn/src/wnn"
	with open(log, "w") as f:
		proc = subprocess.run(
			cmd, cwd="/Users/lacg/wnn", env=env, stdout=f, stderr=subprocess.STDOUT,
		)
	dt = time.time() - t0
	print(f"  done in {dt/60:.1f} min — exit code {proc.returncode}")
	return {"tag": tag, "log": str(log), "duration_s": dt, "returncode": proc.returncode}


def main():
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--execute", action="store_true",
	                help="Actually launch the runs (default is preview).")
	ap.add_argument("--max-runs", type=int, default=None,
	                help="Limit to first N combos (default: all valid).")
	args = ap.parse_args()

	combos = enumerate_combos()
	print(f"Valid combos (jerk > 0): {len(combos)} of "
	      f"{len(WEIGHTS_ERR)*len(WEIGHTS_STABLE)*len(WEIGHTS_MONO)} candidate cells")
	print(f"  Grid: err {WEIGHTS_ERR} × stable {WEIGHTS_STABLE} × mono {WEIGHTS_MONO}")
	print(f"        jerk derived as 1 - err - stable - mono\n")
	print(f"  {'#':>3} {'err':>5} {'stable':>7} {'jerk':>5} {'mono':>5}  Σ")
	for i, c in enumerate(combos, 1):
		total = c['err'] + c['stable'] + c['jerk'] + c['mono']
		print(f"  {i:>3} {c['err']:>5.2f} {c['stable']:>7.2f} {c['jerk']:>5.2f} {c['mono']:>5.2f}  {total:.2f}")

	if args.max_runs is not None:
		combos = combos[:args.max_runs]
		print(f"\nLimiting to first {len(combos)} combo(s).")

	est_min = len(combos) * 45                # ~45 min per run estimate
	print(f"\nPer-run config: pop={POP}, gens/stage={PER_STAGE_GENS} (memory={PER_STAGE_GENS*2}), "
	      f"eval={EVAL_EPISODES} ep × {STEPS} steps, rg={RG_ROUNDS}×{RG_EPS_PER_RND}")
	print(f"Estimated total: {len(combos)} runs × ~45 min ≈ {est_min/60:.1f}h")

	if not args.execute:
		print(f"\nDRY-RUN — pass --execute to launch.")
		return

	results = []
	for i, c in enumerate(combos, 1):
		results.append(run_one(c, i, dry=False))

	print("\n" + "=" * 78)
	print("SWEEP SUMMARY")
	print("=" * 78)
	for r in results:
		print(f"  {r['tag']}  {r['duration_s']/60:5.1f} min  exit={r['returncode']}  {r['log']}")


if __name__ == "__main__":
	main()
