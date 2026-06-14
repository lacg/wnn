"""CheckpointConfig — checkpoint cadence/location config for the IDS GA/TS.

The CheckpointManager class that used to live here (JSON, ClusterGenome-coupled)
was retired 14/06/2026: both strands now use the unified, codec-based
``wnn.ram.strategies.phased.PhasedCheckpointManager`` (yaml.gz, genome-agnostic).
This config is still how callers (experiment.py, phased_search.py) describe the
cadence; ``architecture_ga`` builds a PhasedCheckpointManager from it, and the
shared store reads any legacy ``*.json`` checkpoint written before the migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CheckpointConfig:
	"""Configuration for checkpoint saving.

	Two save cadences are supported (the active decision lives in
	``phased.SaveCadence``, which a PhasedCheckpointManager is built with):
	  * Legacy gen-count: save every ``interval`` generations.
	  * Dynamic wall-clock (preferred): when ``target_loss_seconds`` is set, save
	    whenever at least that many seconds have elapsed since the last save,
	    capped so we never skip more than ``max_interval`` generations. This
	    self-adjusts to per-gen cost — fast gens accumulate until the budget is
	    hit (throttling I/O), while a single slow gen (e.g. 46M ~40 min/gen)
	    checkpoints the moment it finishes. Bounds the work lost on a crash to
	    ~``target_loss_seconds``.
	"""
	enabled: bool = True
	interval: int = 50                       # Legacy: save every N generations
	checkpoint_dir: Optional[Path] = None    # Directory for checkpoint files
	filename_prefix: str = "checkpoint"      # Prefix for checkpoint filenames
	target_loss_seconds: Optional[float] = None  # Dynamic: max wall-clock to risk losing
	max_interval: int = 10                   # Dynamic: hard cap on gens between saves
