#!/usr/bin/env python3
"""CTRL-56 (17/11/2026): stage checkpoints must not hold arch-only (cells=None) entries
that reload as EMPTY controllers.

Fixture: a REAL pre-CTRL-56 grid checkpoint (Y s1, ABI 57) — its GRID#2/#3 are
arch-only, GRID#0/#1 carry the cells the held-out trained. Skips if it is gone.

  T1 (fast)  the loader refuses arch-only entries by default; allow_arch_only loads them;
             probe_episode_failures._load_stage_genome refuses GRID#2 but loads GRID#1.
  T2 (train) _materialize_arch_only_cells trains COPIES only (live genomes untouched) and
             each copy's cells == a one-genome retrain (= what stage-select and
             --recalc-headline compute for a lone arch-only candidate).
  T3 (train) save -> strict load -> identical fingerprints AND an identical score;
             a resume load strips exactly the materialized entries back to arch-only.

T2/T3 run on a shrunk recipe (steps/episodes/calibration) — both sides of every
comparison use the same args, so the equalities are exact, not approximate.
  python tests/controller_ctrl56_ckpt_cells.py [--fast]
"""
import copy
import os
import shlex
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("WNN_CONTROLLER_GPU_EVAL", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

D = Path("/Users/lacg/wnn/logs/controller/ctrl48")
TAG = "C-S2-d_wposx2_a57_rf_c200_dc_biasin_ms01_P_both_L1_dmyY_full_s1"
CKPT = D / f"ckpt_{TAG}" / "stage0_grid.yaml.gz"
FAILS = []


def check(cond, what):
	print(f"  [{'ok' if cond else 'FAIL'}] {what}", flush=True)
	if not cond:
		FAILS.append(what)


def t1_loader_refuses():
	from wnn.control import phased_ga as P
	from wnn.control.ckpt_cells import ArchOnlyEntryError
	from probe_episode_failures import _load_stage_genome
	print("T1 loader refusal")
	try:
		P._stage_entry_from_checkpoint(CKPT, top_k=3)
		check(False, "strict loader refuses a checkpoint whose top-3 holds GRID#2 arch-only")
	except ArchOnlyEntryError as e:
		check("GRID#2" in str(e), f"strict loader refuses, naming GRID#2 ({str(e)[:70]}...)")
	_l, _s, res, _it = P._stage_entry_from_checkpoint(CKPT, top_k=3, allow_arch_only=True)
	cells = [g.cells is not None for g in res.final_population]
	check(cells == [True, True, False], f"allow_arch_only loads top-3, cells={cells}")
	try:
		_load_stage_genome(str(CKPT.parent), "stage0_grid", 2)
		check(False, "_load_stage_genome refuses GRID#2")
	except ArchOnlyEntryError:
		check(True, "_load_stage_genome refuses GRID#2")
	_lab, _spec, g1 = _load_stage_genome(str(CKPT.parent), "stage0_grid", 1)
	check(g1.cells is not None, "_load_stage_genome loads GRID#1 (has cells) next to an arch-only GRID#2")


def shrunk_run():
	"""The run's own argv, shrunk; prepared the way phased_ga.main prepares it."""
	from wnn.control import phased_ga as P
	from wnn.seeds import resolve_seed_set
	txt = open(D / f"{TAG}.out", errors="replace").read()
	argv = shlex.split([l for l in txt.splitlines() if l.startswith("[argv]")][0][len("[argv]"):])
	a = P.build_arg_parser().parse_args(argv)
	a.steps, a.test_episodes, a.calib_episodes, a.holdout_pop_sample = 300, 6, 20, 8
	ec = P.episode_config_from_args(a)
	P._resolve_ctbr(a, ec)
	a._dhat_b = None
	a._conn_policy, a._conn_policy_min = P.conn_policy_from_name(getattr(a, "conn_policy", "spread"))
	s = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	return a, ec, s


def score(a, ec, spec, g, seed, train):
	from wnn.control import phased_ga as P
	from wnn.control.evaluator import ControllerEvaluator
	thr = P._test_thresholds(a, ec, spec, seed, train, use_score=True)
	ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=seed, episode_config=ec,
	                         thresholds=thr, rg_config=P._rg_config(a, ec, seed), max_train_workers=1,
	                         num_eval_folds=a.num_eval_folds)
	m = ev.score_genomes([g])[0]
	return (m.stable_rate, m.mean_attitude_error_deg, m.mean_steady_error_deg, m.reward)


def t2_t3_materialize_roundtrip():
	from wnn.control import phased_ga as P
	from wnn.control.ckpt_cells import strip_materialized_cells
	a, ec, s = shrunk_run()
	_l, spec, res, _it = P._stage_entry_from_checkpoint(CKPT, top_k=4, allow_arch_only=True)
	live = res.final_population
	before = [g.fingerprint() for g in live]
	print("T2 materialize (copies, solo seeds)")
	mat, idx = P._materialize_arch_only_cells(a, ec, spec, live, s.train)
	check(idx == [2, 3], f"materialized idx={idx}")
	check([g.fingerprint() for g in live] == before and live[2].cells is None,
	      "live genomes untouched (fingerprints equal, GRID#2 still arch-only)")
	check(all(g.cells is not None for g in mat), "every checkpoint entry carries cells")
	check(mat[0] is live[0] and mat[1] is live[1], "entries that had cells are saved as they are")
	thr = P._test_thresholds(a, ec, spec, s.train, s.train, use_score=True)
	for i in idx:
		solo = copy.copy(live[i])
		P._train_arch_only(a, ec, spec, [solo], s.train, thr)   # the held-out path, ONE genome
		check(solo.cells.fingerprint() == mat[i].cells.fingerprint(),
		      f"GRID#{i}: chunk-materialized cells == one-genome retrain (stage-select / recalc)")
	print("T3 save -> load -> score; resume strip")
	with tempfile.TemporaryDirectory() as td:
		a.save_stage_checkpoints = td
		P._save_stage_checkpoint(a, 0, "grid", spec, SimpleNamespace(best_genome=mat[0], final_population=mat),
		                         None, cells_materialized_idx=idx)
		path = Path(td) / "stage0_grid.yaml.gz"
		_l2, spec2, res2, _ = P._stage_entry_from_checkpoint(path)          # STRICT: must not raise
		back = res2.final_population
		check([g.fingerprint() for g in back] == [g.fingerprint() for g in mat],
		      "reloaded population fingerprints == saved (cells included)")
		# Absolute numbers are meaningless on the shrunk recipe (GRID#1's cells were trained
		# under the full calibration); only the equalities are tested — plus the bug itself:
		# the arch-only GRID#2 scores as an EMPTY controller.
		unit = int(a.test_seeds[0])
		for i in (1, 2):
			x, y = score(a, ec, spec, mat[i], unit, s.train), score(a, ec, spec2, back[i], unit, s.train)
			check(x == y, f"GRID#{i} score in memory == reloaded: {x} vs {y}")
		empty = score(a, ec, spec, live[2], unit, s.train)
		check(empty[0] < score(a, ec, spec, mat[2], unit, s.train)[0],
		      f"the arch-only GRID#2 scores as an EMPTY controller (stable {empty[0]:.2f}) — the CTRL-56 bug")
		pay = P._ctl_load(path)
		stripped = strip_materialized_cells(pay)
		pop = pay["population"]
		check(stripped == idx and [g.cells is None for g in pop] == [False, False, True, True],
		      f"resume strip -> as flown (arch-only at {stripped})")
		check([g.fingerprint() for g in pop] == before, "stripped resume population == the live one")


if __name__ == "__main__":
	if not CKPT.exists():
		print(f"SKIP: fixture {CKPT} not present")
		sys.exit(0)
	t1_loader_refuses()
	if "--fast" not in sys.argv:
		t2_t3_materialize_roundtrip()
	print("FAIL:" if FAILS else "PASS", *FAILS, sep="\n  " if FAILS else "")
	sys.exit(1 if FAILS else 0)
