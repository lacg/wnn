"""The worker's signal handler must stop a dashboard CONTROLLER flow cleanly, and
leave IDS flows exactly as they were.

Background (15/10/2026). Since the 19/06 crate split the controller wheel
(ram_controller) has its own cancel flag, separate from the worker wheel's, and the
controller tells a real shutdown from a stray flag by a Python witness
(wnn.control.cancel_state). FlowWorker._handle_signal set only the worker flag.
_run_controller_flow does install phased_ga's own handler once its imports are
done, so the gap is the window before that, plus any controller path that runs
under the worker's handler. The handler now also marks the witness and sets the
controller flag — but ONLY when the controller path is already loaded (it never
imports it), so an IDS process is untouched.

Run: PYTHONPATH=src python -m pytest tests/test_worker_signal_controller.py -q
"""
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.ram.experiments.worker import FlowWorker


def _bare_worker(current_flow_id=None):
	"""A FlowWorker with only what _handle_signal touches (no dashboard, no DB)."""
	w = FlowWorker.__new__(FlowWorker)
	w.current_flow_id = current_flow_id
	w._running = {}
	w.running = True
	w._stop_current_flow = False
	w.logs = []
	w._log = w.logs.append
	return w


# ---------------------------------------------------------------- IDS: nothing changed
IDS_PROBE = textwrap.dedent("""
	import json, signal, sys
	from wnn.ram.experiments.worker import FlowWorker
	import ram_accelerator
	w = FlowWorker.__new__(FlowWorker)
	w.current_flow_id, w._running, w.running, w._stop_current_flow = {cur}, {{}}, True, False
	w.logs = []; w._log = w.logs.append
	ram_accelerator.reset_cancel_flag()
	w._handle_signal(signal.SIGTERM, None)
	print(json.dumps(dict(
		controller_loaded=any(m in sys.modules for m in
		                      ("wnn.control._accel", "ram_controller", "wnn.control.cancel_state")),
		worker_flag=bool(ram_accelerator.is_cancelled()),
		running=w.running, stop_current=w._stop_current_flow, logs=w.logs)))
	ram_accelerator.reset_cancel_flag()
""")


@pytest.mark.parametrize("cur, want_running, want_stop", [(None, False, False), (6438, True, True)])
def test_ids_process_handler_is_unchanged_and_never_loads_the_controller(cur, want_running, want_stop):
	"""A FRESH interpreter, like an IDS flow subprocess / the scheduler: the handler
	sets the worker flag and the same stop flags as before, and the controller
	wheel is never imported into it."""
	import json
	env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
	r = subprocess.run([sys.executable, "-c", IDS_PROBE.format(cur=cur)],
	                   capture_output=True, text=True, env=env, timeout=120)
	assert r.returncode == 0, r.stderr[-2000:]
	got = json.loads(r.stdout.strip().splitlines()[-1])
	assert got["controller_loaded"] is False, "the signal handler pulled the controller into an IDS process"
	assert got["worker_flag"] is True
	assert (got["running"], got["stop_current"]) == (want_running, want_stop)
	assert not any("controller" in l for l in got["logs"])


def test_ids_flow_shutdown_still_requeues(monkeypatch):
	"""The IDS flow path (FlowWorker._execute_flow's shutdown branch) is not touched
	by this change; pin that a flow stopped by SIGTERM is still requeued, not failed."""
	import wnn.ram.experiments.worker as W
	src = Path(W.__file__).read_text()
	# The IDS requeue branch keys on _stop_current_flow, which the handler still sets.
	assert "requeue" in src.lower()
	w = _bare_worker(current_flow_id=6438)
	w._handle_signal(signal.SIGTERM, None)
	assert w._stop_current_flow is True and w.running is True


# ---------------------------------------------------------------- controller: both flags + witness
@pytest.fixture
def clean_controller():
	from wnn.control import cancel_state, _accel
	cancel_state.reset_sigterm()
	_accel.reset_cancel_flag()
	saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
	yield cancel_state, _accel
	for s, h in saved.items():
		signal.signal(s, h)
	cancel_state.reset_sigterm()
	_accel.reset_cancel_flag()
	try:
		import ram_accelerator
		ram_accelerator.reset_cancel_flag()
	except Exception:
		pass


def test_handler_sets_controller_flag_and_witness_when_controller_loaded(clean_controller):
	cancel_state, ctl = clean_controller
	w = _bare_worker(current_flow_id=7)
	w._handle_signal(signal.SIGTERM, None)
	assert cancel_state.sigterm_received(), "the proper-cancel witness must be marked"
	assert cancel_state.last_signum() == signal.SIGTERM
	assert ctl.is_cancelled(), "the CONTROLLER wheel's flag must be set"
	assert w._stop_current_flow is True


class _Client:
	def __init__(self):
		self.calls = []

	def __getattr__(self, name):
		def rec(*a, **k):
			self.calls.append(name)
			if name == "get_flow":
				return {"id": 99, "status": "running"}
			if name == "list_flow_experiments":
				return []
			return None
		return rec


class _Tracker:
	def __getattr__(self, name):
		return lambda *a, **k: None


TINY_PARAMS = dict(grid_state_neurons=[0], max_state_neurons=0, grid_bits=[8],
                   grid_output_neurons=[8], max_output_neurons=8,
                   skip_stages="bits,connections", neurons_gens=1, neurons_patience=1,
                   memory_gens=4, memory_patience=4, pop=4, grid_top_k=4,
                   eval_episodes=2, memory_eval_episodes=2, report_episodes=2, steps=50,
                   universe_episodes=2, report_seeds=[11], base_seed=5,
                   check_interval=1, train_workers=2, grid_min_suffix=1, lamarckian=True)


@pytest.mark.parametrize("handler", ["phased_ga", "worker_only"])
def test_sigterm_during_dashboard_controller_flow_requeues_cleanly(monkeypatch, tmp_path,
                                                                   clean_controller, handler):
	"""End to end through flow_runner._run_controller_flow with a REAL SIGTERM
	delivered to this process mid-MEMORY: no MEMORY "done", no stage-final
	checkpoint, no headline, flow requeued (not failed, not completed).

	`phased_ga`: the production path (_run_controller_flow installs phased_ga's
	handler). `worker_only`: that install replaced by the WORKER's handler — the
	situation before the install and on any path that never installs it; this is
	the variant FlowWorker._propagate_to_controller makes pass."""
	pytest.importorskip("ram_controller")
	from wnn.ram.experiments import flow_runner as FR
	from wnn.control import phased_ga as P
	from wnn.control.evaluator import ControllerEvaluator
	monkeypatch.setenv("WNN_DB_PATH", str(tmp_path / "seeds.db"))

	state = {"in_memory": False, "n": 0}
	real_memory, real_score = P._run_memory_phase, ControllerEvaluator.score_genomes

	def memory_phase(*a, **k):
		state["in_memory"] = True
		return real_memory(*a, **k)

	def score(self, genomes, **kw):
		if state["in_memory"]:
			state["n"] += 1
			if state["n"] == 2:
				os.kill(os.getpid(), signal.SIGTERM)   # the scheduler's forward, for real
		return real_score(self, genomes, **kw)

	monkeypatch.setattr(P, "_run_memory_phase", memory_phase)
	monkeypatch.setattr(ControllerEvaluator, "score_genomes", score)

	client = _Client()
	w = _bare_worker()
	w.client, w.tracker = client, _Tracker()
	w._open_log_file = lambda name: tmp_path / "flow.log"
	w._start_heartbeat = w._stop_heartbeat = w._close_log_file = lambda *a, **k: None
	if handler == "worker_only":
		def worker_handlers():
			cancel_state, ctl = clean_controller
			cancel_state.reset_sigterm()
			ctl.reset_cancel_flag()
			signal.signal(signal.SIGTERM, w._handle_signal)
			signal.signal(signal.SIGINT, w._handle_signal)
		monkeypatch.setattr(P, "_install_signal_handlers", worker_handlers)
	params = dict(TINY_PARAMS, save_stage_checkpoints=str(tmp_path / "ckpt"))
	flow = {"id": 99, "name": "tiny controller", "config": {"params": params}}

	import contextlib
	import io
	buf = io.StringIO()
	with contextlib.redirect_stdout(buf):
		FR._run_controller_flow(w, flow)
	out = buf.getvalue() + "\n".join(map(str, w.logs))

	assert state["n"] >= 2, "the SIGTERM never fired inside MEMORY"
	assert "requeue_flow" in client.calls, f"not requeued: {client.calls}"
	assert "flow_failed" not in client.calls and "flow_completed" not in client.calls
	assert "STAGE 4 (MEMORY) done" not in out, "a cancelled stage was reported as done"
	assert "[stage-select] HEADLINE" not in out
	assert not (tmp_path / "ckpt" / "stage4_memory.yaml.gz").exists()
	assert (tmp_path / "ckpt" / "emergency_stage4_memory.yaml.gz").exists()
