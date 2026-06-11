"""EmergencyDump — SIGTERM/SIGINT crash checkpoint for phased searches.

Ported from the controller strand (phased_ga, 29/05/2026 design): on a
termination signal, write the CURRENT carry/phase state through the unified
checkpoint store so the run can resume instead of losing in-RAM progress.
The previous handler is chained (the controller's handler also sets the Rust
cancel flag; that behavior is preserved by chaining).
"""

import signal
from pathlib import Path
from typing import Callable, Optional

from wnn.ram.strategies.phased.checkpoint import PhaseCheckpoint, save_checkpoint
from wnn.ram.strategies.phased.codecs import GenomeCodec


class EmergencyDump:
	"""Register a state provider; dump it on SIGTERM/SIGINT (chained)."""

	def __init__(self, dump_dir: "str | Path", codec: GenomeCodec,
	             log: Callable[[str], None] = print):
		self._dir = Path(dump_dir)
		self._codec = codec
		self._log = log
		self._provider: Optional[Callable[[], Optional[PhaseCheckpoint]]] = None
		self._prev_term = None
		self._prev_int = None
		self._installed = False

	def set_state_provider(self, provider: Callable[[], Optional[PhaseCheckpoint]]) -> None:
		"""Provider returns the checkpoint to dump (or None = nothing to save)."""
		self._provider = provider

	def install(self) -> None:
		if self._installed:
			return
		self._prev_term = signal.signal(signal.SIGTERM, self._handler)
		self._prev_int = signal.signal(signal.SIGINT, self._handler)
		self._installed = True

	def uninstall(self) -> None:
		if not self._installed:
			return
		signal.signal(signal.SIGTERM, self._prev_term or signal.SIG_DFL)
		signal.signal(signal.SIGINT, self._prev_int or signal.SIG_DFL)
		self._installed = False

	def dump_now(self) -> Optional[Path]:
		"""Write the current state (also callable outside a signal)."""
		if self._provider is None:
			return None
		ckpt = self._provider()
		if ckpt is None:
			return None
		name = f"emergency_{ckpt.phase_key or 'phase'}_{ckpt.phase_name}".replace(" ", "_").replace(":", "")
		path = self._dir / f"{name}.json"
		written = save_checkpoint(path, ckpt, self._codec)
		self._log(f"[EMERGENCY-DUMP] wrote {written}")
		return written

	def _handler(self, signum, frame) -> None:
		try:
			self.dump_now()
		except Exception as e:  # never mask the signal on a dump failure
			self._log(f"[EMERGENCY-DUMP] FAILED: {e}")
		prev = self._prev_term if signum == signal.SIGTERM else self._prev_int
		if callable(prev):
			prev(signum, frame)
		elif prev == signal.SIG_DFL:
			signal.signal(signum, signal.SIG_DFL)
			signal.raise_signal(signum)
