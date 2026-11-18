#!/usr/bin/env python3
"""CTRL-56 recovery for PRE-FIX stage-0 checkpoints (arch-only GRID entries, GRID#2 always).

  ctrl56_grid_ckpt_repair.py verify <run.out>
      Retrain the saved GRID#0 (trained by the held-out at batch position 0 = solo seeds)
      from its architecture alone, with the run's own argv, and compare cell fingerprints.
      MATCH = the installed wheel + this Python reproduce the run's GRID training, so a
      re-materialized arch-only entry is the controller stage-select computed in flight.
      MISMATCH = the trainer changed since the run (older ABI) -> mark the entries invalid.
  ctrl56_grid_ckpt_repair.py repair <run.out>
      verify, then write <ckpt_dir>/ctrl56_stage0_grid.yaml.gz: the same population with
      every arch-only entry trained (phased_ga._materialize_arch_only_cells, solo seeds) and
      meta.cells_materialized_idx set. The original file is NEVER modified (the name does not start with "stage", so
      stage-select / --recalc-headline never pick the copy up by accident).

The ckpt dir is taken from the run's own --save-stage-checkpoints. Read-only otherwise;
never flies a phased_ga. Exit 0 = MATCH (and written, for repair), 3 = MISMATCH.
"""
import copy
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("WNN_CONTROLLER_GPU_EVAL", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run_context(out_path: str):
	"""(args, ec, seeds, ckpt_dir) prepared the way phased_ga.main prepares them."""
	from wnn.control import phased_ga as P
	from wnn.seeds import resolve_seed_set
	lines = [l for l in open(out_path, errors="replace").read().splitlines() if l.startswith("[argv]")]
	if len(lines) != 1:
		raise SystemExit(f"{out_path}: expected one [argv] line, found {len(lines)}")
	# The 11/11 terminology cut-over (same values, same meaning) — as rebuild_argv.py does.
	rename = {"--report-seeds": "--test-seeds", "--report-seed": "--test-seed",
	          "--report-episodes": "--test-episodes"}
	toks = [rename.get(t, t) for t in shlex.split(lines[0][len("[argv]"):])]
	a = P.build_arg_parser().parse_args(toks)
	if not getattr(a, "save_stage_checkpoints", None):
		raise SystemExit(f"{out_path}: the run saved no stage checkpoints")
	ec = P.episode_config_from_args(a)
	P._resolve_ctbr(a, ec)
	a._dhat_b = None
	if getattr(a, "obs_dhat", False):
		raise SystemExit("--obs-dhat runs need the main()-derived d-hat constant — not handled here")
	a._conn_policy, a._conn_policy_min = P.conn_policy_from_name(getattr(a, "conn_policy", "spread"))
	s = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	ck = Path(a.save_stage_checkpoints)
	return a, ec, s, (ck if ck.is_absolute() else Path("/Users/lacg/wnn") / ck)


def verify(a, ec, s, ck) -> bool:
	from wnn.control import phased_ga as P
	_l, spec, res, _it = P._stage_entry_from_checkpoint(ck / "stage0_grid.yaml.gz", top_k=1,
	                                                     allow_arch_only=True)
	g0 = res.final_population[0]
	if g0.cells is None:
		raise SystemExit("GRID#0 has no cells either (no test seeds in this run?) — nothing to verify against")
	probe = copy.copy(g0)
	probe.cells = None
	thr = P._test_thresholds(a, ec, spec, s.train, s.train, use_score=True)
	P._train_arch_only(a, ec, spec, [probe], s.train, thr)
	ok = probe.cells.fingerprint() == g0.cells.fingerprint()
	print(f"[ctrl56] verify {ck.name}: GRID#0 retrained from its architecture "
	      f"{'MATCHES' if ok else 'DOES NOT MATCH'} the saved cells", flush=True)
	return ok


def repair(a, ec, s, ck) -> Path:
	from wnn.control import phased_ga as P
	_l, spec, res, _it = P._stage_entry_from_checkpoint(ck / "stage0_grid.yaml.gz", allow_arch_only=True)
	pop, idx = P._materialize_arch_only_cells(a, ec, spec, res.final_population, s.train)
	out = ck / "ctrl56_stage0_grid.yaml.gz"
	P._save_winner(str(out), a, spec, pop[0], pop, None, stage_num=0, stage_name="grid",
	               cells_materialized_idx=idx)
	print(f"[ctrl56] repaired copy: {out} ({len(idx)} entries materialized)", flush=True)
	return out


if __name__ == "__main__":
	if len(sys.argv) != 3 or sys.argv[1] not in ("verify", "repair"):
		raise SystemExit(__doc__)
	ctx = run_context(sys.argv[2])
	if not verify(*ctx):
		sys.exit(3)
	if sys.argv[1] == "repair":
		repair(*ctx)
