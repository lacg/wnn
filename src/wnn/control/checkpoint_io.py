"""Controller checkpoint I/O — ONE place that maps the historical
payload-dict shape onto the unified phased checkpoint store.

Used by phased_ga and every controller tool (holdout eval, transplant seed,
curriculum GA, memory refinement). All values are PLAIN DATA in the envelope:
genomes via ControllerGenomeCodec (native), spec/metrics as dataclass dicts.
Legacy pickle checkpoints still load through the store's fallback path.
"""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional


def _codec():
	from wnn.ram.strategies.phased import ControllerGenomeCodec
	return ControllerGenomeCodec()


def _spec_to_plain(spec: Any) -> Any:
	return asdict(spec) if is_dataclass(spec) else spec


def _spec_from_plain(d: Any) -> Any:
	if isinstance(d, dict):
		from wnn.control.evaluator import ControllerSpec
		return ControllerSpec(**d)
	return d


def _metrics_to_plain(m: Any) -> Any:
	return asdict(m) if is_dataclass(m) else m


def payload_to_checkpoint(payload: dict):
	"""Historical controller payload dict → PhaseCheckpoint (plain data only)."""
	from wnn.ram.strategies.phased import PhaseCheckpoint
	extra = dict(payload.get("meta", {}))
	if "spec" in payload:
		extra["spec"] = _spec_to_plain(payload["spec"])
	if "fitness_weights" in payload:
		extra["fitness_weights"] = payload["fitness_weights"]
	if "metrics" in payload:
		extra["metrics"] = _metrics_to_plain(payload["metrics"])
	return PhaseCheckpoint(
		phase_key=str(payload.get("stage_num", "")),
		phase_name=payload.get("stage_name", ""),
		strategy_type="GA",
		best_genome=payload.get("best_genome"),
		final_population=list(payload.get("population", [])) or None,
		iterations_run=int(payload.get("generation", 0)),
		extra=extra,
	)


def checkpoint_to_payload(ckpt) -> dict:
	"""Inverse mapping; spec is rebuilt into a ControllerSpec."""
	payload = {
		"stage_num":   int(ckpt.phase_key) if str(ckpt.phase_key).isdigit() else ckpt.phase_key,
		"stage_name":  ckpt.phase_name,
		"best_genome": ckpt.best_genome,
		"population":  list(ckpt.final_population or []),
		"generation":  ckpt.iterations_run,
		"meta":        {k: v for k, v in ckpt.extra.items()
		                if k not in ("spec", "fitness_weights", "metrics")},
	}
	if "spec" in ckpt.extra:
		payload["spec"] = _spec_from_plain(ckpt.extra["spec"])
	if "fitness_weights" in ckpt.extra:
		payload["fitness_weights"] = ckpt.extra["fitness_weights"]
	if "metrics" in ckpt.extra:
		payload["metrics"] = ckpt.extra["metrics"]
	return payload


def save_controller_checkpoint(path: "str | Path", payload: dict) -> Path:
	from wnn.ram.strategies.phased import save_checkpoint
	return save_checkpoint(path, payload_to_checkpoint(payload), _codec())


def save_controller_checkpoint_async(path: "str | Path", payload: dict):
	"""Async variant: encode NOW (race-free snapshot of the population), write on
	a background thread. Returns ``(path, thread)``. Use for between-stage dumps —
	the next stage reads its population from memory, so the disk write is
	resume-only and off the critical path."""
	from wnn.ram.strategies.phased import save_checkpoint_async
	return save_checkpoint_async(path, payload_to_checkpoint(payload), _codec())


def load_controller_checkpoint(path: "str | Path",
                               skip_population: bool = False) -> Optional[dict]:
	"""Load schema-2 yaml.gz OR legacy pickle; returns the payload-dict shape
	(None if the file does not exist).

	skip_population=True: don't materialize the embedded final population
	(payload["population"] = []). Winner checkpoints saved with the full pop
	are 100s of MB / tens of GB in RAM — score-only tools must use this."""
	from wnn.ram.strategies.phased import load_checkpoint
	ckpt = load_checkpoint(path, _codec(), skip_population=skip_population)
	return checkpoint_to_payload(ckpt) if ckpt is not None else None
