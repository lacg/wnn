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


def find_checkpoint_file(path: "str | Path") -> Optional[Path]:
	"""Return the existing checkpoint file for ``path`` (probing the canonical
	stem across every known suffix), or None. Shared by ``load_checkpoint`` and
	``PhasedCheckpointManager.has_checkpoint`` so existence and load agree."""
	base = Path(path)
	stem = base
	while stem.suffix in (".gz", ".json", ".yaml", ".yml", ".pkl"):
		stem = stem.with_suffix("")
	candidates = [base] if base.exists() else []
	for suffix in (".yaml.gz", ".yaml", ".json.gz", ".json", ".pkl", ""):
		cand = Path(str(stem) + suffix)
		if cand.exists() and cand not in candidates:
			candidates.append(cand)
	return candidates[0] if candidates else None


def _ids_json_to_checkpoint(data: dict, codec: GenomeCodec) -> PhaseCheckpoint:
	"""Map a LEGACY IDS CheckpointManager *.json payload onto a PhaseCheckpoint.

	The IDS genome dicts ({bits_per_neuron, neurons_per_cluster, connections} +
	a 'fitness' field) are a subset of ClusterGenome.serialize(), so the codec
	decodes them directly once 'fitness' is renamed to the key deserialize reads.
	Lets the unified manager resume runs checkpointed by the pre-migration worker."""
	def _heal(gd: dict) -> dict:
		fit = gd.get("fitness")
		if fit is None:
			return gd
		gd = {k: v for k, v in gd.items() if k != "fitness"}
		gd["cached_metrics" if isinstance(fit, dict) else "cached_fitness"] = fit
		return gd
	pop = [codec.decode(_heal(gd)) for gd in data.get("population", [])]
	best = data.get("best_genome")
	bf = data.get("best_fitness") or [None, None]
	return PhaseCheckpoint(
		phase_key=str(data.get("phase_name", "")),
		phase_name=data.get("phase_name", ""),
		strategy_type=data.get("optimizer_type", "GA"),
		best_genome=codec.decode(_heal(best)) if best is not None else None,
		final_population=pop or None,
		iterations_run=int(data.get("current_iteration", 0)),
		patience=int((data.get("extra_state") or {}).get("patience", 0)),
		final_fitness=bf[0] if len(bf) > 0 else None,
		final_accuracy=bf[1] if len(bf) > 1 else None,
		final_threshold=data.get("current_threshold"),
		extra={"config": data.get("config", {}),
		       "extra_state": data.get("extra_state", {}),
		       "legacy_ids_json": True},
		saved_at=0.0,
	)


def _null_scalar_event() -> "yaml.ScalarEvent":
	return yaml.ScalarEvent(anchor=None, tag="tag:yaml.org,2002:null",
	                        implicit=(True, False), value="~")


def _yaml_load_skipping(stream, skip_keys: "frozenset[str]"):
	"""Parse a YAML mapping document while DROPPING the values of the given
	TOP-LEVEL keys — their events are consumed but never materialized, so memory
	stays bounded (an 838 MB winner.yaml.gz's final_population balloons to tens
	of GB if built). Skipped keys come back as None. The reduced event stream is
	re-emitted to text and loaded normally (small once the big subtrees are gone)."""
	kept: list = []
	# Each mapping frame tracks whether the next direct child is a KEY.
	frames: list[dict] = []          # {"map": bool, "next_is_key": bool}
	skipping = 0                     # >0 → inside a dropped subtree
	pending_skip = False             # last kept event was a skip_keys key at top level

	def _child_done():
		if frames and frames[-1]["map"]:
			frames[-1]["next_is_key"] = True

	for ev in yaml.parse(stream, Loader=_YamlLoader):
		if skipping:
			if isinstance(ev, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
				skipping += 1
			elif isinstance(ev, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
				skipping -= 1
				if skipping == 0:
					kept.append(_null_scalar_event())
					_child_done()
			continue
		if isinstance(ev, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
			if pending_skip:
				pending_skip = False
				skipping = 1
				continue
			frames.append({"map": isinstance(ev, yaml.MappingStartEvent), "next_is_key": True})
			kept.append(ev)
			continue
		if isinstance(ev, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
			frames.pop()
			kept.append(ev)
			_child_done()
			continue
		if isinstance(ev, yaml.ScalarEvent):
			if pending_skip:                 # scalar value of a skipped key: null it
				pending_skip = False
				kept.append(_null_scalar_event())
				_child_done()
				continue
			if frames and frames[-1]["map"] and frames[-1]["next_is_key"]:
				frames[-1]["next_is_key"] = False
				if len(frames) == 1 and ev.value in skip_keys:
					pending_skip = True
			else:
				_child_done()
			kept.append(ev)
			continue
		kept.append(ev)                      # stream/document events

	return yaml.load(yaml.emit(iter(kept), Dumper=_YamlDumper), Loader=_YamlLoader)


def extract_checkpoint_head(src: "str | Path", dst: "str | Path",
                            cut_key: str = "final_population") -> Path:
	"""Copy a schema-2 yaml.gz checkpoint WITHOUT its final_population by
	streaming the gz text only UP TO the cut key — which _build_payload dumps
	LAST — and closing the top-level flow mapping. Seconds on multi-100 MB
	full-population winners: the population bytes are never decompressed, let
	alone parsed (the _yaml_load_skipping alternative still walks every event,
	minutes-to-hours at that size). The reduced doc is VALIDATED (yaml.load +
	schema + best_genome present) before the destination is written; raises
	ValueError when the file is not the expected single-doc mapping."""
	marker = f"{cut_key}:"
	head_parts: list[str] = []
	carry = ""
	with gzip.open(Path(src), "rt", encoding="utf-8") as f:
		while True:
			chunk = f.read(1 << 20)
			if not chunk:
				raise ValueError(f"extract_checkpoint_head: '{cut_key}' not found in {src}")
			buf = carry + chunk
			i = buf.find(marker)
			if i >= 0:
				head_parts.append(buf[:i])
				break
			# Keep a marker-sized tail as carry so a split match is still found.
			head_parts.append(buf[:-len(marker)])
			carry = buf[-len(marker):]
	head = "".join(head_parts).rstrip().rstrip(",") + "}\n"
	data = yaml.load(head, Loader=_YamlLoader)
	if (not isinstance(data, dict) or data.get("schema") != CHECKPOINT_SCHEMA_VERSION
			or data.get("best_genome") is None):
		raise ValueError(f"extract_checkpoint_head: reduced doc failed validation for {src}")
	dstp = Path(dst)
	dstp.parent.mkdir(parents=True, exist_ok=True)
	tmp = Path(str(dstp) + f".tmp.{os.getpid()}")
	with gzip.open(tmp, "wt", encoding="utf-8") as f:
		f.write(head)
	os.replace(tmp, dstp)
	return dstp


def load_checkpoint(path: "str | Path", codec: GenomeCodec,
                    skip_population: bool = False) -> Optional[PhaseCheckpoint]:
	"""Load a checkpoint: schema-2, legacy experiments json.gz, legacy IDS *.json,
	or legacy controller pickle. Returns None if the file doesn't exist.

	skip_population=True drops final_population WITHOUT materializing it
	(bounded memory on huge full-population winner checkpoints; score-only
	tools like holdout eval / ensemble harnesses never need the population).
	YAML formats only — a legacy pickle still loads whole (no partial pickle)."""
	p = find_checkpoint_file(path)
	if p is None:
		return None

	# --- Try YAML (gz or plain). JSON ⊂ YAML, so legacy json.gz loads here too ---
	data = None
	try:
		opener = gzip.open if p.suffix == ".gz" else open
		with opener(p, "rt", encoding="utf-8") as f:
			if skip_population:
				data = _yaml_load_skipping(f, frozenset({"final_population"}))
			else:
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
		if "current_iteration" in data and "population" in data:
			# Legacy IDS CheckpointManager *.json (pre-unification worker).
			return _ids_json_to_checkpoint(data, codec)
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
