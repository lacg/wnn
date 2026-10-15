"""Is a `stageN_<name>.yaml.gz` a stage that genuinely FINISHED?

WHY (15/10/2026, SL_C_..._s31337003_tw515). A watchdog SIGTERM in MEMORY gen 2 was
reported as a completed stage: the GA's cooperative stop returned a normal result,
the orchestrator printed "STAGE 4 (MEMORY) done: ... err=0.00° stable=0.0%", wrote
the FINAL stage4_memory checkpoint 31 s after the emergency dump, and published a
zero held-out. The resume-point picker then saw a final newer than the dump and
would have resumed with --resume-mode next — skipping 118 of MEMORY's 120 gens.

From that date phased_ga never writes a stage-final checkpoint after a cancel
(RunInterrupted), and every final it writes carries `stage_complete: true` in its
extra block. A final WITHOUT the field predates the fix; it is accepted unless its
recorded stage metrics are the cancelled-evaluation signature — stable 0 with err
exactly 0 (the no-op Rust return the broken cancel-guard let through) or >= 180
(the guard's sentinel). A genuinely bad stage has stable 0 but a finite, nonzero err.

Light on purpose (gzip + yaml only): the resume picker (scripts/pick_resume_point.py)
runs this in the shell lib between attempts, and only reads the checkpoint HEAD —
the text before `best_genome:` — never a genome.
"""
from __future__ import annotations

import gzip
from pathlib import Path

_CHUNK = 1 << 16
_CUT = "best_genome:"


def read_checkpoint_extra(path: "str | Path") -> dict:
	"""The `extra` mapping of a schema-2 yaml.gz checkpoint, read from its head only
	(everything before `best_genome:`), or {} when the head cannot be parsed."""
	import yaml
	try:
		Loader = yaml.CSafeLoader
	except AttributeError:      # pure-python PyYAML
		Loader = yaml.SafeLoader
	parts, carry = [], ""
	try:
		with gzip.open(Path(path), "rt", encoding="utf-8") as f:
			while True:
				chunk = f.read(_CHUNK)
				if not chunk:
					return {}
				buf = carry + chunk
				i = buf.find(_CUT)
				if i >= 0:
					parts.append(buf[:i])
					break
				parts.append(buf[:-len(_CUT)])
				carry = buf[-len(_CUT):]
		head = "".join(parts).rstrip().rstrip(",") + "}\n"
		data = yaml.load(head, Loader=Loader)
	except (OSError, EOFError, UnicodeDecodeError, yaml.YAMLError):
		return {}
	extra = data.get("extra") if isinstance(data, dict) else None
	return extra if isinstance(extra, dict) else {}


def cancelled_metrics(metrics) -> bool:
	"""The cancelled-evaluation signature: stable 0 AND err exactly 0 or >= 180."""
	if not isinstance(metrics, dict):
		return False
	stable = metrics.get("stable_rate", metrics.get("acc"))
	err = metrics.get("mean_attitude_error_deg")
	if stable is None or err is None:
		return False
	return float(stable) == 0.0 and (float(err) == 0.0 or float(err) >= 180.0)


def stage_final_status(extra: dict) -> "tuple[bool, str]":
	"""(complete, reason) for a stage-final checkpoint's `extra` block."""
	if extra.get("emergency_dump") is True:
		return False, "an emergency (mid-stage) dump, not a final"
	if "stage_complete" in extra:
		ok = extra.get("stage_complete") is True
		return ok, ("stage_complete: true" if ok else "marked stage_complete: false (interrupted)")
	if cancelled_metrics(extra.get("metrics")):
		return False, ("legacy final (no stage_complete) with cancelled-evaluation metrics "
		               "(stable 0, err 0 or >=180) — a SIGTERM reported as a finished stage")
	return True, "legacy final (no stage_complete), metrics look genuine"


def checkpoint_is_complete_final(path: "str | Path") -> "tuple[bool, str]":
	"""stage_final_status for a file, reading only its head."""
	extra = read_checkpoint_extra(path)
	if not extra:
		return False, "unreadable checkpoint head"
	return stage_final_status(extra)
