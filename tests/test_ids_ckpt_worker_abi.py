"""IDS-81 fix 1: IDS GA checkpoints record the worker ABI; resume refuses a
checkpoint from another ABI (or none) and the phase starts fresh.

Run: PYTHONPATH=src/wnn python -m pytest tests/test_ids_ckpt_worker_abi.py -q
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

from wnn.ram.strategies.phased import (
	PhasedCheckpointManager, SaveCadence, PhaseCheckpoint, ClusterGenomeCodec,
)
from wnn.ram.strategies.phased.worker_abi_guard import (
	WORKER_ABI_KEY, installed_worker_abi, stamp_worker_abi, worker_abi_refusal,
)

ABI_N = 30


def _pop(n):
	from wnn.ram.genome import ClusterGenome
	return [ClusterGenome(bits_per_neuron=[8] * 4, neurons_per_cluster=[4],
	                      connections=list(range(i, i + 32))) for i in range(n)]


def _ckpt():
	pop = _pop(5)
	return PhaseCheckpoint(phase_key="GA", phase_name="GA Neurons", strategy_type="GA",
	                       best_genome=pop[0], final_population=pop, iterations_run=9,
	                       patience=2, extra={"complete": False})


def _mgr(path, abi, lines=None):
	log = (lines.append if lines is not None else None)
	return PhasedCheckpointManager(path, ClusterGenomeCodec(), SaveCadence(None, 10),
	                               logger=log, worker_abi=abi)


def _path():
	return Path(tempfile.mkdtemp()) / "flow_ga"


def test_save_stamps_installed_abi_without_mutating_caller():
	p, ck = _path(), _ckpt()
	_mgr(p, ABI_N).save(ck)
	assert WORKER_ABI_KEY not in ck.extra, "save must stamp a copy, not the caller's object"
	assert _mgr(p, None).load().extra[WORKER_ABI_KEY] == ABI_N


def test_same_abi_resumes():
	p = _path()
	_mgr(p, ABI_N).save(_ckpt())
	out = _mgr(p, ABI_N).load()
	assert out is not None and out.iterations_run == 9 and len(out.final_population) == 5


def test_newer_abi_refuses_with_loud_line():
	p, lines = _path(), []
	_mgr(p, ABI_N).save(_ckpt())
	assert _mgr(p, ABI_N + 1, lines).load() is None
	assert len(lines) == 1 and lines[0].startswith("[CKPT] ⚠️")
	assert "ABI mismatch — restarting from scratch" in lines[0]
	assert f"ABI {ABI_N}" in lines[0] and f"ABI {ABI_N + 1}" in lines[0]


def test_legacy_checkpoint_without_abi_refused():
	p, lines = _path(), []
	_mgr(p, None).save(_ckpt())  # pre-IDS-81 writer: no stamp
	assert _mgr(p, ABI_N, lines).load() is None
	assert "no worker ABI" in lines[0] and "restarting from scratch" in lines[0]


def test_controller_strand_unaffected():
	"""worker_abi=None (controller) → no stamp, no check, loads anything."""
	p = _path()
	_mgr(p, ABI_N).save(_ckpt())
	assert _mgr(p, None).load() is not None


def test_refusal_pure_function():
	ck = _ckpt()
	assert worker_abi_refusal(stamp_worker_abi(ck, 7), 7) is None
	assert "saved under worker ABI 7" in worker_abi_refusal(stamp_worker_abi(ck, 7), 8)
	assert "no worker ABI" in worker_abi_refusal(ck, 8)


def _fake_strategy(mgr, calls):
	log = SimpleNamespace(info=lambda m: None, warning=lambda m: None)
	return SimpleNamespace(_checkpoint_mgr=mgr, _log=log, name="GA",
	                       resume_state_from_checkpoint=lambda c: calls.append(c))


def test_ga_resume_starts_fresh_on_mismatch_and_resumes_on_match():
	from wnn.ram.strategies.connectivity.architecture_ga import ArchitectureGAStrategy
	resume = ArchitectureGAStrategy._resume_from_checkpoint
	p, seeds = _path(), _pop(2)
	_mgr(p, ABI_N).save(_ckpt())
	calls = []
	assert resume(_fake_strategy(_mgr(p, ABI_N + 1), calls), seeds) is seeds
	assert calls == [], "a refused checkpoint must not restore GA control state"
	out = resume(_fake_strategy(_mgr(p, ABI_N), calls), seeds)
	assert len(out) == 5 and len(calls) == 1 and calls[0].iterations_run == 9


def test_installed_abi_matches_facade():
	import wnn.accel as accel
	if not accel.AVAILABLE:
		return
	assert installed_worker_abi() == accel.EXPECTED_ABI
