"""Unified, versioned checkpoint store for phased searches (D1, 11/06/2026).

ONE schema for both strands (controller phased_ga + experiments phased_search):
yaml.gz envelope, `schema: 2`, with generation + patience so resume CONTINUES
instead of restarting at gen 0 with patience reset. Genome payloads go through
a strand-specific GenomeCodec (see codecs.py). Everything in the envelope is
PLAIN DATA — no pickle, refactor-proof, `zcat file.yaml.gz` readable.

Why YAML: project preference (less verbose to eyeball), and since JSON is a
strict subset of YAML, yaml.safe_load transparently reads the legacy
experiments json.gz checkpoints too. The C loader/dumper is used when
available (pure-Python YAML is slow on big populations).

The loader also accepts both legacy formats:
- legacy experiments json.gz: {"phase_result": {...}, "_metadata": {...}}
- legacy controller pickle: {"spec", "best_genome", "population", ...}
"""

import gzip
import os
import pickle
import threading
import time

import yaml

try:  # libyaml C bindings: ~10× faster parse on large checkpoint payloads
	from yaml import CSafeLoader as _YamlLoader, CSafeDumper as _YamlDumper
except ImportError:  # pragma: no cover
	from yaml import SafeLoader as _YamlLoader, SafeDumper as _YamlDumper
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from wnn.ram.strategies.phased.codecs import GenomeCodec

CHECKPOINT_SCHEMA_VERSION = 2


@dataclass
class PhaseCheckpoint:
	"""Everything needed to resume AFTER (or inside) a phase."""
	phase_key: str
	phase_name: str
	strategy_type: str                       # "GA" / "TS" / "SA" / "GRID" / "MEMORY"
	best_genome: Any = None
	final_population: Optional[list] = None
	# True-resume state (the controller had these; experiments lost them — unified now)
	iterations_run: int = 0
	patience: int = 0
	# Metrics / continuity
	final_fitness: Optional[float] = None
	final_accuracy: Optional[float] = None
	final_threshold: Optional[float] = None
	initial_fitness: Optional[float] = None
	initial_accuracy: Optional[float] = None
	# Strand-specific payload (controller: spec/fitness_weights/meta; etc.)
	extra: dict = field(default_factory=dict)
	saved_at: float = 0.0


def _canonical_path(path: "str | Path") -> Path:
	"""<stem>.yaml.gz regardless of the suffix the caller used."""
	p = Path(path)
	while p.suffix in (".gz", ".json", ".yaml", ".yml", ".pkl"):
		p = p.with_suffix("")
	return p.with_suffix(".yaml.gz")


def _build_payload(ckpt: PhaseCheckpoint, codec: GenomeCodec) -> dict:
	"""Encode a checkpoint into the schema-2 envelope — PLAIN DATA only.

	All ``codec.encode`` work happens here, so the returned dict is an immutable
	snapshot with no references to live genomes: safe to hand to a background
	writer (``save_checkpoint_async``) while the caller keeps evolving its
	population.
	"""
	return {
		"schema": CHECKPOINT_SCHEMA_VERSION,
		"codec": codec.name,
		"phase_key": ckpt.phase_key,
		"phase_name": ckpt.phase_name,
		"strategy_type": ckpt.strategy_type,
		"iterations_run": ckpt.iterations_run,
		"patience": ckpt.patience,
		"final_fitness": ckpt.final_fitness,
		"final_accuracy": ckpt.final_accuracy,
		"final_threshold": ckpt.final_threshold,
		"initial_fitness": ckpt.initial_fitness,
		"initial_accuracy": ckpt.initial_accuracy,
		"extra": ckpt.extra,
		"saved_at": time.time(),
		"best_genome": codec.encode(ckpt.best_genome) if ckpt.best_genome is not None else None,
		"final_population": (
			[codec.encode(g) for g in ckpt.final_population]
			if ckpt.final_population else None
		),
	}


def _write_payload(p: Path, payload: dict) -> None:
	"""Atomically write the yaml.gz envelope (temp file + os.replace), so a crash
	mid-write leaves any prior checkpoint at ``p`` intact rather than truncated."""
	p.parent.mkdir(parents=True, exist_ok=True)
	tmp = Path(str(p) + f".tmp.{os.getpid()}")
	with gzip.open(tmp, "wt", encoding="utf-8") as f:
		yaml.dump(payload, f, Dumper=_YamlDumper, default_flow_style=True, sort_keys=False)
	os.replace(tmp, p)


def save_checkpoint(path: "str | Path", ckpt: PhaseCheckpoint, codec: GenomeCodec) -> Path:
	"""Write a schema-2 yaml.gz checkpoint (synchronous). Returns the path written."""
	p = _canonical_path(path)
	_write_payload(p, _build_payload(ckpt, codec))
	return p


def save_checkpoint_async(path: "str | Path", ckpt: PhaseCheckpoint,
                          codec: GenomeCodec) -> "tuple[Path, threading.Thread]":
	"""Encode NOW (race-free snapshot), write on a background thread.

	The encode (``_build_payload``) runs synchronously so it captures the
	population before the caller mutates it; only the gzip/yaml write — the slow
	part — is offloaded. Returns ``(path, thread)``; the caller may ``thread.join()``
	the FINAL save before exiting (the thread is non-daemon, so the process won't
	exit mid-write regardless). Reusable by any phased strand (controller stages,
	IDS) — between-stage dumps are resume-only (off the next stage's critical path).
	"""
	p = _canonical_path(path)
	payload = _build_payload(ckpt, codec)  # SYNC: immutable plain-data snapshot
	t = threading.Thread(target=_write_payload, args=(p, payload),
	                     name=f"ckpt-save-{p.name}", daemon=False)
	t.start()
	return p, t


def load_checkpoint(path: "str | Path", codec: GenomeCodec) -> Optional[PhaseCheckpoint]:
	"""Load a checkpoint: schema-2, legacy experiments json.gz, or legacy
	controller pickle. Returns None if the file doesn't exist."""
	base = Path(path)
	stem = base
	while stem.suffix in (".gz", ".json", ".yaml", ".yml", ".pkl"):
		stem = stem.with_suffix("")
	candidates = [base] if base.exists() else []
	for suffix in (".yaml.gz", ".yaml", ".json.gz", ".json", ".pkl", ""):
		cand = Path(str(stem) + suffix)
		if cand.exists() and cand not in candidates:
			candidates.append(cand)
	if not candidates:
		return None
	p = candidates[0]

	# --- Try YAML (gz or plain). JSON ⊂ YAML, so legacy json.gz loads here too ---
	data = None
	try:
		opener = gzip.open if p.suffix == ".gz" else open
		with opener(p, "rt", encoding="utf-8") as f:
			data = yaml.load(f, Loader=_YamlLoader)
		if not isinstance(data, dict):
			data = None
	except (OSError, UnicodeDecodeError, yaml.YAMLError):
		data = None

	if data is not None:
		if data.get("schema") == CHECKPOINT_SCHEMA_VERSION:
			return PhaseCheckpoint(
				phase_key=data["phase_key"],
				phase_name=data["phase_name"],
				strategy_type=data["strategy_type"],
				best_genome=codec.decode(data["best_genome"]) if data.get("best_genome") is not None else None,
				final_population=(
					[codec.decode(g) for g in data["final_population"]]
					if data.get("final_population") else None
				),
				iterations_run=int(data.get("iterations_run", 0)),
				patience=int(data.get("patience", 0)),
				final_fitness=data.get("final_fitness"),
				final_accuracy=data.get("final_accuracy"),
				final_threshold=data.get("final_threshold"),
				initial_fitness=data.get("initial_fitness"),
				initial_accuracy=data.get("initial_accuracy"),
				extra=data.get("extra", {}),
				saved_at=data.get("saved_at", 0.0),
			)
		if "phase_result" in data:
			# Legacy experiments format (no generation/patience — defaults 0)
			pr = data["phase_result"]
			meta = data.get("_metadata", {})
			return PhaseCheckpoint(
				phase_key=meta.get("phase_key", ""),
				phase_name=pr.get("phase_name", ""),
				strategy_type=pr.get("strategy_type", ""),
				best_genome=codec.decode(pr["best_genome"]) if pr.get("best_genome") is not None else None,
				final_population=(
					[codec.decode(g) for g in pr["final_population"]]
					if pr.get("final_population") else None
				),
				iterations_run=int(pr.get("iterations_run", 0)),
				patience=0,
				final_fitness=pr.get("final_fitness"),
				final_accuracy=pr.get("final_accuracy"),
				final_threshold=pr.get("final_threshold"),
				initial_fitness=pr.get("initial_fitness"),
				initial_accuracy=pr.get("initial_accuracy"),
				extra=meta,
			)
		raise ValueError(f"Unrecognized checkpoint JSON at {p}")

	# --- Legacy controller pickle ---
	with open(p, "rb") as f:
		payload = pickle.load(f)
	meta = dict(payload.get("meta", {}))
	for k in ("spec", "fitness_weights", "metrics"):
		if k in payload:
			meta[k] = payload[k]
	return PhaseCheckpoint(
		phase_key=str(payload.get("stage_num", "")),
		phase_name=payload.get("stage_name", ""),
		strategy_type="GA",
		best_genome=payload.get("best_genome"),
		final_population=list(payload.get("population", [])) or None,
		iterations_run=int(payload.get("generation", 0)),
		patience=int(payload.get("patience", 0)),
		extra=meta,
	)
