"""
LiveProgressObserver — thread-safe live progress reporting for strategies

Split out of architecture_strategies.py (D3, 11/06/2026); that module
re-exports everything, so existing imports keep working.
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING


if TYPE_CHECKING:
	from wnn.ram.strategies.connectivity.adaptive_cluster import (
		ClusterGenome,
		AdaptiveClusterConfig,
	)

class LiveProgressObserver:
	"""Polls Rust evaluator for live progress and POSTs to dashboard.

	Runs in a daemon thread, reading evaluator.get_live_progress() every
	`interval` seconds and sending to the dashboard. Thread-safe: the Rust
	side uses Arc<RwLock> and releases the GIL during search.
	"""

	def __init__(self, evaluator, client, experiment_id: int, interval: float = 5.0):
		self._evaluator = evaluator
		self._client = client
		self._experiment_id = experiment_id
		self._interval = interval
		self._stop_event = threading.Event()
		self._thread = None

	def start(self):
		if not self._client or not self._experiment_id:
			return
		if not hasattr(self._evaluator, 'get_live_progress'):
			return

		def loop():
			while not self._stop_event.wait(self._interval):
				try:
					progress = self._evaluator.get_live_progress()
					if progress and self._client:
						self._client.post_live_progress(self._experiment_id, progress)
				except Exception:
					pass  # Observer must never crash the main thread

		self._thread = threading.Thread(target=loop, daemon=True)
		self._thread.start()

	def stop(self):
		self._stop_event.set()
		if self._thread:
			self._thread.join(timeout=2)
		# Send clear signal
		if self._client and self._experiment_id:
			try:
				self._client.clear_live_progress(self._experiment_id)
			except Exception:
				pass


# =============================================================================
# Shared Mixin for Architecture Strategies
# =============================================================================
