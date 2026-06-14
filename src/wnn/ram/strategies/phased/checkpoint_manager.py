"""PhasedCheckpointManager — ONE checkpoint orchestrator for every phased strand.

Supersedes the IDS-only ``connectivity.CheckpointManager`` (JSON +
ClusterGenome-coupled). Built on the codec-based phased store
(``checkpoint.py``, yaml.gz schema-2), so it is genome-agnostic: hand it the
strand's :class:`GenomeCodec` and any genome with a codec works — controller
(``ControllerGenomeCodec``), IDS (``ClusterGenomeCodec``), and whatever comes
next. No per-strand copy of the save/load/cadence machinery.

Owns three concerns, nothing else:
  * WHEN — the adaptive :class:`SaveCadence` (slow gen → every gen; fast gen →
    throttled to ``max_interval``);
  * WRITE — ``save`` (sync), ``maybe_save`` (cadence-gated), optional async with
    a single in-flight writer so writes never overlap on the pid-keyed temp file;
  * READ — ``load`` / ``has_checkpoint`` over schema-2 yaml.gz, legacy
    experiments json.gz, legacy controller pickle, AND legacy IDS ``*.json``
    (backward-compat, so a worker restart resumes pre-migration checkpoints).

Everything written is a :class:`PhaseCheckpoint` of plain data via the codec —
no pickle, ``zcat``-inspectable.
"""

from pathlib import Path
from typing import Callable, Optional

from wnn.ram.strategies.phased.cadence import SaveCadence
from wnn.ram.strategies.phased.checkpoint import (
	PhaseCheckpoint, save_checkpoint, save_checkpoint_async, load_checkpoint,
	find_checkpoint_file,
)
from wnn.ram.strategies.phased.codecs import GenomeCodec


class PhasedCheckpointManager:
	"""Cadence + write + read for one phased run's checkpoints."""

	def __init__(self, path: "str | Path", codec: GenomeCodec, cadence: SaveCadence,
	             *, async_save: bool = False,
	             logger: Optional[Callable[[str], None]] = None):
		self._path = Path(path)
		self._codec = codec
		self._cadence = cadence
		self._async = async_save
		self._log = logger or (lambda _m: None)
		self._thread = None  # single in-flight async writer

	# --- WHEN ---------------------------------------------------------------
	def should_save_now(self, generation: int) -> bool:
		return self._cadence.should_save_now(generation)

	# --- WRITE --------------------------------------------------------------
	def maybe_save(self, generation: int, checkpoint: PhaseCheckpoint) -> bool:
		"""Checkpoint iff the adaptive cadence says it's due. Returns whether it
		saved. The caller builds the strand-specific PhaseCheckpoint."""
		if not self._cadence.should_save_now(generation):
			return False
		self.save(checkpoint)
		self._cadence.mark_saved(generation)
		return True

	def save(self, checkpoint: PhaseCheckpoint) -> Path:
		"""Write now (cadence-independent). Async mode keeps a single writer in
		flight (joins the prior one first), so two writes never collide."""
		if self._async:
			self.join()
			p, self._thread = save_checkpoint_async(self._path, checkpoint, self._codec)
			return p
		return save_checkpoint(self._path, checkpoint, self._codec)

	def join(self) -> None:
		"""Block until any in-flight async write finishes (≤1 by construction)."""
		t = self._thread
		if t is not None:
			try:
				t.join()
			except Exception:
				pass
			self._thread = None

	# --- READ ---------------------------------------------------------------
	def has_checkpoint(self) -> bool:
		return find_checkpoint_file(self._path) is not None

	def load(self) -> Optional[PhaseCheckpoint]:
		"""Return the resumable PhaseCheckpoint (schema-2 / legacy formats), or
		None if no checkpoint exists for this path."""
		return load_checkpoint(self._path, self._codec)
