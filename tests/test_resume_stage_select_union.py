"""A RESUMED run must rank the same stage-select pool and publish the same
per-stage held-out rows as an uninterrupted one.

Regression cover for 13/10/2026 (SL_C_..._s31337002_tw20): the watchdog killed
the run in MEMORY (stage 4); the retry resumed with --resume-from-emergency
--resume-mode same, which sliced the orchestrator to MEMORY only and skipped the
grid — so GRID and NEURONS (finished, checkpointed on disk) never re-entered
`_select_headline_stage`. The headline was ranked over 3 MEMORY candidates
instead of 9, and the marker lost its GRID/NEURONS held-out rows.

Everything heavy is monkeypatched (no simulator, no wheel, no controller slot):
the stage checkpoints are toy schema-2 files read through the real bounded
loader, the orchestrator's MEMORY stage is a stub, and the val/held-out scorers
return canned metrics.

Run: PYTHONPATH=src/wnn python -m pytest tests/test_resume_stage_select_union.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnn.control import phased_ga as P
from wnn.control.controller_orchestrator import ControllerOrchestrator
from wnn.ram.strategies.phased import (PhaseCheckpoint, save_checkpoint,
                                       load_checkpoint_top_k)


class _DictCodec:
	"""Toy codec: genomes are plain flow mappings, like the real serialized form."""
	name = "dict_codec"

	def encode(self, genome):
		return genome

	def decode(self, data):
		return data


def _g(name):
	return {"name": name, "rows": [[1, 2, {"k": 3}]] * 4}


def _write_stage(d: Path, sn: int, name: str, n: int) -> Path:
	pop = [_g(f"{name.upper()}#{i}") for i in range(n)]
	ck = PhaseCheckpoint(phase_key=str(sn), phase_name=name, strategy_type="GA",
	                     best_genome=_g(f"{name.upper()}-BEST"), final_population=pop,
	                     iterations_run=7, extra={"spec": {"tag": name}})
	return save_checkpoint(d / f"stage{sn}_{name}.yaml.gz", ck, _DictCodec())


# ---------------------------------------------------------------- bounded loader
def test_top_k_loader_streams_only_the_first_k(tmp_path):
	p = _write_stage(tmp_path, 1, "neurons", 50)
	ck = load_checkpoint_top_k(p, _DictCodec(), 3)
	assert [g["name"] for g in ck.final_population] == ["NEURONS#0", "NEURONS#1", "NEURONS#2"]
	assert ck.best_genome["name"] == "NEURONS-BEST"
	assert ck.phase_name == "neurons" and ck.extra["spec"] == {"tag": "neurons"}
	assert ck.iterations_run == 7


def test_top_k_loader_short_population_and_head_only(tmp_path):
	p = _write_stage(tmp_path, 0, "grid", 2)
	assert len(load_checkpoint_top_k(p, _DictCodec(), 3).final_population) == 2
	head = load_checkpoint_top_k(p, _DictCodec(), 0)
	assert head.final_population is None and head.extra["spec"] == {"tag": "grid"}


# ---------------------------------------------------------------- resume planning
def _args(tmp_path, **kw):
	a = P.build_arg_parser().parse_args([])
	a.save_stage_checkpoints = str(tmp_path)
	a.report_seeds = [99990101, 99990102]
	a.stage_select_top_k = 3
	for k, v in kw.items():
		setattr(a, k, v)
	return a


def test_missing_finished_stage_refuses_loudly(tmp_path):
	_write_stage(tmp_path, 0, "grid", 5)          # NEURONS checkpoint absent
	with pytest.raises(SystemExit, match="stage1_neurons"):
		P._resume_completed_stages(_args(tmp_path), {"stage_num": 4, "resume_mode": "same"},
		                           4, {"bits", "connections"})


def test_no_checkpoint_dir_refuses_loudly(tmp_path):
	with pytest.raises(SystemExit, match="REFUSING"):
		P._resume_completed_stages(_args(tmp_path, save_stage_checkpoints=None),
		                           {"stage_num": 4, "resume_mode": "same"}, 4, set())


def test_required_stages_follow_the_fresh_run(tmp_path):
	for sn, nm in ((0, "grid"), (1, "neurons"), (2, "bits"), (3, "connections")):
		_write_stage(tmp_path, sn, nm, 5)
	st = {"stage_num": 4, "resume_mode": "same"}
	got = P._resume_completed_stages(_args(tmp_path), st, 4, {"bits", "connections"})
	assert [n for _s, n, _p in got] == ["grid", "neurons"]
	got = P._resume_completed_stages(_args(tmp_path), st, 4, set())
	assert [n for _s, n, _p in got] == ["grid", "neurons", "bits", "connections"]
	# No report seeds -> a fresh run never makes GRID a candidate.
	got = P._resume_completed_stages(_args(tmp_path, report_seeds=None, report_seed=None),
	                                 st, 4, {"bits", "connections"})
	assert [n for _s, n, _p in got] == ["neurons"]
	# Resume at NEURONS: only GRID finished before it.
	got = P._resume_completed_stages(_args(tmp_path), {"stage_num": 1}, 1, set())
	assert [n for _s, n, _p in got] == ["grid"]


def test_mode_next_uses_the_snapshot_only_without_a_final_checkpoint(tmp_path):
	_write_stage(tmp_path, 0, "grid", 5)          # NEURONS cut short by --resume-mode next
	got = P._resume_completed_stages(_args(tmp_path), {"stage_num": 1, "resume_mode": "next"},
	                                 2, {"bits"})
	assert [(n, p is None) for _s, n, p in got] == [("grid", False), ("neurons", True)]


# ---------------------------------------------------------------- end to end
@pytest.fixture
def harness(monkeypatch, tmp_path):
	"""_run_one with every expensive call stubbed; records what the selector scored."""
	_write_stage(tmp_path, 0, "grid", 15)
	_write_stage(tmp_path, 1, "neurons", 50)
	scored, holdouts, ran = [], [], []
	fail = {}   # {"val": exc} / {"holdout": (label-prefix, exc)} — deliberate aborts
	ok = SimpleNamespace(acc=1.0, stable_rate=1.0, mean_attitude_error_deg=1.0, reward=-1.0,
	                     fitness=-1.0, mean_steady_error_deg=1.0, motor_jerk_mean=0.1,
	                     mono_violations_total=0.0, mean_effort=0.5)


	def entry_with_toy_codec(path, top_k=None):
		ck = load_checkpoint_top_k(path, _DictCodec(), top_k)
		label = ck.phase_name.upper()
		return (label, SimpleNamespace(tag=label),
		        SimpleNamespace(final_population=[SimpleNamespace(name=g["name"], cells=None)
		                                          for g in ck.final_population],
		                        best_genome=SimpleNamespace(name=ck.best_genome["name"], cells=None)),
		        ck.iterations_run)

	def fake_val(args, ec, spec, genome, final_population, vs, train, stage_label=""):
		if "val" in fail:
			raise fail["val"]
		scored.append(genome.name)
		return ok

	def fake_holdout(args, ec, spec, res, seeds, label):
		if "holdout" in fail and label.startswith(fail["holdout"][0]):
			raise fail["holdout"][1]
		pop = res.final_population or [res.best_genome]
		holdouts.append((label, pop[0].name))
		return ok

	def fake_run_all(self, specs, carry):
		ran.extend(s.name for s in specs)
		if "MEMORY" not in [s.name for s in specs]:
			return
		pop = [SimpleNamespace(name=f"MEMORY#{i}", cells=object()) for i in range(50)]
		self._res_by_stage[4] = SimpleNamespace(best_genome=pop[0], final_population=pop)
		self._row_by_stage[4] = ("Memory", carry.extra["spec"], None, 0.0, 1)

	def no_grid(*a, **k):
		raise AssertionError("the grid must not re-run on resume")

	monkeypatch.setattr(P, "_stage_entry_from_checkpoint", entry_with_toy_codec)
	monkeypatch.setattr(P, "_holdout_report", fake_val)
	monkeypatch.setattr(P, "_maybe_holdout", fake_holdout)
	monkeypatch.setattr(P, "_pid_baseline", lambda *a, **k: [])
	monkeypatch.setattr(P, "stage0_grid", no_grid)
	monkeypatch.setattr(P, "_spec_from_best", lambda best, base: base)   # toy specs
	monkeypatch.setattr(ControllerOrchestrator, "run_all", fake_run_all)
	return SimpleNamespace(dir=tmp_path, scored=scored, holdouts=holdouts, ran=ran, fail=fail)


def _resume_state(mode="same"):
	return {"stage_num": 4, "resume_mode": mode, "spec": SimpleNamespace(tag="MEMORY"),
	        "best_genome": None,
	        "population": [SimpleNamespace(name=f"DUMP#{i}", cells=object()) for i in range(5)]}


def test_resume_at_memory_ranks_the_full_nine_candidate_union(harness):
	args = _args(harness.dir, skip_stages="bits,connections")
	seeds = SimpleNamespace(train=7, val=1000, test=11)
	ho = {}
	P._run_one(args, None, seeds, resume_state=_resume_state(), stage_holdouts=ho)
	assert harness.ran == ["MEMORY"], "resume must slice to the dumped stage"
	unique = set(harness.scored)
	want = {f"{s}#{i}" for s in ("GRID", "NEURONS", "MEMORY") for i in range(3)}
	assert unique == want, f"stage-select pool was {sorted(unique)}"
	assert len(harness.scored) == 9 * 5, "9 candidates x 5 val seeds"
	# The per-stage report is re-published for the reloaded stages, on pop[0], before
	# MEMORY's (the stub MEMORY stage publishes none) — and reaches the caller's dict.
	assert harness.holdouts == [("GRID", "GRID#0"), ("NEURONS", "NEURONS#0")]
	assert set(ho) == {"GRID", "NEURONS"}


def test_seed_winner_start_does_not_reload(harness):
	"""--seed-winner skips stages that never ran in THIS run: nothing to reload."""
	args = _args(harness.dir, skip_stages="bits,connections")
	st = _resume_state(mode="next")
	st["stage_num"] = 3
	st["resume_source"] = "seed_winner"
	P._run_one(args, None, SimpleNamespace(train=7, val=1000, test=11), resume_state=st)
	assert {n.split("#")[0] for n in harness.scored} == {"MEMORY"}
	assert [h for h in harness.holdouts if not h[0].startswith("HEADLINE-")] == []


# ---------------------------------------------------------------- resume from a FINAL checkpoint
def _final_state(tmp_path, sn, name, n=50):
	"""What main() builds for `--resume-from-emergency stage<sn>_<name>.yaml.gz
	--resume-mode next` (the picker's "stage finished, next never dumped" case)."""
	path = next(tmp_path.glob(f"stage{sn}_{name}.yaml.gz"))
	return {"stage_num": sn, "resume_mode": "next", "spec": SimpleNamespace(tag=name.upper()),
	        "best_genome": SimpleNamespace(name=f"{name.upper()}-BEST", cells=object()),
	        "population": [SimpleNamespace(name=f"{name.upper()}#{i}", cells=object()) for i in range(n)],
	        "generation": 7, "source_path": str(path)}


def test_next_from_neurons_final_runs_memory_and_keeps_the_union(harness):
	args = _args(harness.dir, skip_stages="bits,connections")
	P._run_one(args, None, SimpleNamespace(train=7, val=1000, test=11),
	           resume_state=_final_state(harness.dir, 1, "neurons"))
	assert harness.ran == ["BITS", "CONNECTIONS", "MEMORY"]   # skips are the orchestrator's
	assert set(harness.scored) == {f"{s}#{i}" for s in ("GRID", "NEURONS", "MEMORY") for i in range(3)}
	assert harness.holdouts == [("GRID", "GRID#0"), ("NEURONS", "NEURONS#0")]


def test_next_from_memory_final_is_publish_only(harness):
	"""Killed AFTER MEMORY's final save: no GA stage re-runs (it used to clamp to 4
	and re-fly MEMORY); MEMORY is published from the checkpoint, the union is
	complete, and --save-winner gets the FULL population."""
	_write_stage(harness.dir, 4, "memory", 50)
	st = _final_state(harness.dir, 4, "memory")
	_rows, best, pop, _pid = P._run_one(_args(harness.dir, skip_stages="bits,connections"), None,
	                                    SimpleNamespace(train=7, val=1000, test=11), resume_state=st)
	assert harness.ran == [], "publish-only: no GA stage may run"
	assert set(harness.scored) == {f"{s}#{i}" for s in ("GRID", "NEURONS", "MEMORY") for i in range(3)}
	assert [label for label, _g in harness.holdouts] == ["GRID", "NEURONS", "MEMORY"]
	assert best is st["best_genome"] and len(pop) == 50


def test_picker_rule(tmp_path):
	import os
	sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
	import pick_resume_point as K

	def touch(name, ns):
		p = tmp_path / name
		p.write_text("x")
		os.utime(p, ns=(ns, ns))

	assert K.pick(tmp_path) is None                       # killed in the grid: from scratch
	touch("stage0_grid.yaml.gz", 100)
	assert K.pick(tmp_path) == (str(tmp_path / "stage0_grid.yaml.gz"), "next")
	touch("emergency_stage1_neurons.yaml.gz", 200)
	assert K.pick(tmp_path) == (str(tmp_path / "emergency_stage1_neurons.yaml.gz"), "same")
	touch("stage1_neurons.yaml.gz", 200)                  # same mtime: the final save follows the dump
	assert K.pick(tmp_path) == (str(tmp_path / "stage1_neurons.yaml.gz"), "next")
	touch("stage4_memory.yaml.gz.tmp.123", 999)           # a half-written final is never a resume point
	assert K.pick(tmp_path)[0].endswith("stage1_neurons.yaml.gz")


# ---------------------------------------------------------------- deliberate aborts propagate
def test_stage_select_holdout_error_propagates(harness):
	"""HoldoutScoringError in stage-select used to be swallowed by `except Exception`
	in _run_one: rc=0, no headline, a marker with a hole. It must kill the run."""
	harness.fail["val"] = P.HoldoutScoringError("val seed died")
	with pytest.raises(P.HoldoutScoringError, match="val seed died"):
		P._run_one(_args(harness.dir, skip_stages="bits,connections"), None,
		           SimpleNamespace(train=7, val=1000, test=11), resume_state=_resume_state())


def test_runner_up_rescore_error_propagates(harness, monkeypatch):
	"""A selected runner-up whose report-seed score failed used to publish pop[0]'s
	triple under the headline. Make the last candidate win, then fail its re-score."""
	from wnn.ram.fitness import FitnessCalculatorControllerHarmonic
	monkeypatch.setattr(FitnessCalculatorControllerHarmonic, "fitness",
	                    lambda self, ms: [0.0 if i == len(ms) - 1 else 1.0 for i in range(len(ms))])
	harness.fail["holdout"] = ("HEADLINE-", P.HoldoutScoringError("report seed died"))
	with pytest.raises(P.HoldoutScoringError, match="report seed died"):
		P._run_one(_args(harness.dir, skip_stages="bits,connections"), None,
		           SimpleNamespace(train=7, val=1000, test=11), resume_state=_resume_state())


# ---------------------------------------------------------------- banked-run repair
def test_repair_recovers_every_stage_row_newest_attempt_wins(tmp_path):
	"""scripts/repair_resumed_stage_select.py: GRID/NEURONS rows live only in the
	killed attempt's log; a stage re-run by a later attempt takes the newer row."""
	sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
	import repair_resumed_stage_select as R
	out = tmp_path / "T.out"
	row = "  [report-seeds] {} MULTI-SEED held-out (5 seeds [1, 2]): stable={}%  err=1.0°"
	(tmp_path / "T.out.killed1").write_text(
		"\n".join([row.format("GRID", 95), row.format("NEURONS", 99), row.format("MEMORY", 1)]))
	out.write_text(row.format("MEMORY", 98))
	logs = R._logs_oldest_first(str(out))
	assert [Path(p).name for p in logs] == ["T.out.killed1", "T.out"]
	rows = R.stage_rows(logs)
	assert set(rows) == {"GRID", "NEURONS", "MEMORY"}
	assert "stable=98%" in rows["MEMORY"] and "stable=95%" in rows["GRID"]
	assert "  " not in rows["GRID"], "rows are squeezed like the marker's"
