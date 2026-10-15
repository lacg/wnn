"""A SIGTERM mid-stage must NEVER be reported as a completed stage.

Regression cover for 15/10/2026 (SL_C_b24n256_..._s31337003_tw515): the watchdog's
SIGTERM landed in MEMORY gen 2; the run printed "STAGE 4 (MEMORY) done: gen 2/120
... err=0.00° stable=0.0%", wrote a FINAL stage4_memory checkpoint after the
emergency dump, published an all-zero held-out and ranked 9 all-zero stage-select
candidates. Two defects compounded:
  * evaluator.py's cancel-guard read the WORKER wheel's cancel flag (wnn.accel),
    a different static from the controller wheel's the handler sets, so cancelled
    no-op scores (zeros) passed as measurements;
  * the GA's cooperative stop returns a normal result, which every caller took as
    a finished stage.

A TINY real in-process run (~5 s, ~0.6 GB): grid -> NEURONS -> MEMORY, with the
SIGTERM handler invoked (exactly what the signal does) on the second MEMORY score.

Run: PYTHONPATH=src python -m pytest tests/test_cancel_not_stage_done.py -q
"""
import signal
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("ram_controller")

from wnn.control import phased_ga as P
from wnn.control import cancel_state
from wnn.control import _accel
from wnn.control.evaluator import ControllerEvaluator
import pick_resume_point as K

TINY = ["--grid-state-neurons", "0", "--max-state-neurons", "0",
        "--grid-bits", "8", "--grid-output-neurons", "8", "--max-output-neurons", "8",
        "--skip-stages", "bits,connections", "--neurons-gens", "1", "--neurons-patience", "1",
        "--memory-gens", "4", "--memory-patience", "4", "--pop", "4", "--grid-top-k", "4",
        "--eval-episodes", "2", "--memory-eval-episodes", "2", "--report-episodes", "2",
        "--steps", "50", "--universe-episodes", "2", "--report-seeds", "11",
        "--base-seed", "5", "--runs", "1", "--check-interval", "1",
        "--train-workers", "2", "--grid-min-suffix", "1", "--lamarckian"]


@pytest.fixture
def clean_signals():
	"""main() installs process-wide handlers and the test sets the cancel flag +
	witness; restore all of it so no later test inherits a 'shutdown'."""
	saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
	yield
	for s, h in saved.items():
		signal.signal(s, h)
	cancel_state.reset_sigterm()
	_accel.reset_cancel_flag()


def test_sigterm_mid_memory_is_not_a_finished_stage(monkeypatch, tmp_path, clean_signals):
	monkeypatch.setenv("WNN_DB_PATH", str(tmp_path / "seeds.db"))
	state = {"in_memory": False, "memory_scores": 0}
	real_memory = P._run_memory_phase
	real_score = ControllerEvaluator.score_genomes

	def memory_phase(*a, **k):
		state["in_memory"] = True
		return real_memory(*a, **k)

	def score(self, genomes, **kw):
		if state["in_memory"]:
			state["memory_scores"] += 1
			if state["memory_scores"] == 2:          # mid-stage: gen 1's offspring
				P._sigterm_handler(signal.SIGTERM, None)
		return real_score(self, genomes, **kw)

	monkeypatch.setattr(P, "_run_memory_phase", memory_phase)
	monkeypatch.setattr(ControllerEvaluator, "score_genomes", score)
	monkeypatch.setattr(sys, "argv", ["phased_ga", *TINY, "--save-stage-checkpoints", str(tmp_path)])

	# redirect_stdout, NOT capsys/capfd: the framework logger's StreamHandler binds
	# the stderr it first sees, and the fixtures' streams close after the test —
	# every later test that logs would then die on "I/O operation on closed file".
	import contextlib
	import io
	buf = io.StringIO()
	with contextlib.redirect_stdout(buf):
		rc = P.main()
	out = buf.getvalue()

	assert state["memory_scores"] >= 2, "the SIGTERM never fired inside MEMORY"
	assert rc == 143, "a shutdown must exit 143 (watchdog-stop: no marker, resume)"
	assert "STAGE 1 (NEURONS) done" in out, "the stage that DID finish still reports"
	assert "STAGE 4 (MEMORY) done" not in out, "a cancelled stage was reported as done"
	assert "MEMORY MULTI-SEED held-out" not in out, "a cancelled stage published a held-out"
	assert "[stage-select] HEADLINE" not in out, "stage-select ran on cancelled scores"
	assert "[CANCEL-GUARD] cancel flag SET" in out, \
		"the evaluator's cancel-guard must SEE the controller wheel's flag"
	assert not (tmp_path / "stage4_memory.yaml.gz").exists(), "a final checkpoint for a cut stage"
	assert (tmp_path / "emergency_stage4_memory.yaml.gz").exists()
	# Finished stages carry the completion marker the picker requires.
	from wnn.control.stage_completion import checkpoint_is_complete_final
	for name in ("stage0_grid.yaml.gz", "stage1_neurons.yaml.gz"):
		ok, why = checkpoint_is_complete_final(tmp_path / name)
		assert ok and "stage_complete: true" in why, (name, why)
	# And the retry resumes MEMORY itself, not past it.
	assert K.pick(tmp_path) == (str(tmp_path / "emergency_stage4_memory.yaml.gz"), "same")


def test_guard_reads_the_controller_flag(clean_signals):
	"""The direct defect: the two wheels' cancel flags are separate statics."""
	_accel.reset_cancel_flag()
	_accel.set_cancel_flag()
	from wnn.control._accel import require_accel
	assert require_accel().is_cancelled(), "the facade the evaluator now reads sees the flag"
