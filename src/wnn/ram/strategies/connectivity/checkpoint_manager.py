"""
CheckpointConfig + CheckpointManager — GA/TS checkpoint save/load/resume

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

@dataclass
class CheckpointConfig:
	"""Configuration for checkpoint saving.

	Two save cadences are supported:
	  * Legacy gen-count: save every `interval` generations.
	  * Dynamic wall-clock (preferred): when `target_loss_seconds` is set, save
	    whenever at least that many seconds have elapsed since the last save,
	    capped so we never skip more than `max_interval` generations. This
	    self-adjusts to per-gen cost — fast gens accumulate until the budget is
	    hit (throttling I/O), while a single slow gen (e.g. 46M ~40 min/gen)
	    checkpoints the moment it finishes. Bounds the work lost on a crash to
	    ~`target_loss_seconds`.
	"""
	enabled: bool = True
	interval: int = 50                       # Legacy: save every N generations
	checkpoint_dir: Optional[Path] = None    # Directory for checkpoint files
	filename_prefix: str = "checkpoint"      # Prefix for checkpoint filenames
	target_loss_seconds: Optional[float] = None  # Dynamic: max wall-clock to risk losing
	max_interval: int = 10                   # Dynamic: hard cap on gens between saves


class CheckpointManager:
	"""
	Reusable checkpoint manager for optimization runs.

	Usage:
		# Create manager
		manager = CheckpointManager(
			config=CheckpointConfig(checkpoint_dir=Path("checkpoints")),
			phase_name="Phase 1a: GA Neurons",
			optimizer_type="GA",
			total_iterations=1000,
			logger=print,
		)

		# In optimization loop:
		for iteration in range(1000):
			# ... do optimization ...

			# Save checkpoint every N iterations
			manager.maybe_save(
				iteration=iteration,
				population=population,
				best_genome=best_genome,
				best_fitness=(ce, acc),
				current_threshold=threshold,
				extra_state={"patience": patience_counter},
			)

		# To resume:
		if manager.has_checkpoint():
			state = manager.load()
			start_iteration = state['current_iteration'] + 1
			population = state['population']
	"""

	def __init__(
		self,
		config: CheckpointConfig,
		phase_name: str,
		optimizer_type: str,
		total_iterations: int,
		logger: Optional[Callable[[str], None]] = None,
	):
		self._config = config
		self._phase_name = phase_name
		self._optimizer_type = optimizer_type
		self._total_iterations = total_iterations
		self._logger = logger or (lambda x: None)

		# Dynamic-cadence tracking (used when target_loss_seconds is set).
		self._last_save_monotonic: Optional[float] = None
		self._last_save_gen: int = -1

		# Create checkpoint directory if needed
		if config.enabled and config.checkpoint_dir:
			config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

	def should_save_now(self, generation: int) -> bool:
		"""Decide whether to checkpoint at this generation.

		When `target_loss_seconds` is configured, throttle by wall-clock: save
		once that budget of seconds has elapsed since the last save, but never
		let more than `max_interval` generations pass without a save. The first
		generation seen establishes the time baseline (no save). When
		`target_loss_seconds` is None, fall back to saving every generation
		(the prior behaviour).
		"""
		if not self._config.enabled:
			return False
		budget = self._config.target_loss_seconds
		if budget is None:
			return True  # legacy: caller saves every gen
		import time
		now = time.monotonic()
		if self._last_save_monotonic is None:
			# Establish baseline on first observed generation; don't save yet.
			self._last_save_monotonic = now
			self._last_save_gen = generation
			return False
		elapsed = now - self._last_save_monotonic
		gens_since = generation - self._last_save_gen
		return elapsed >= budget or gens_since >= max(1, self._config.max_interval)

	@property
	def checkpoint_path(self) -> Optional[Path]:
		"""Path to the checkpoint file."""
		if not self._config.enabled or not self._config.checkpoint_dir:
			return None
		return self._config.checkpoint_dir / f"{self._config.filename_prefix}_{self._optimizer_type.lower()}.json"

	def has_checkpoint(self) -> bool:
		"""Check if a checkpoint file exists."""
		path = self.checkpoint_path
		return path is not None and path.exists()

	def should_save(self, iteration: int) -> bool:
		"""Check if we should save at this iteration."""
		if not self._config.enabled:
			return False
		# Save at interval (1-indexed), but also at iteration 0 for safety
		return iteration > 0 and (iteration + 1) % self._config.interval == 0

	def maybe_save(
		self,
		iteration: int,
		population: list[tuple['ClusterGenome', float]],
		best_genome: 'ClusterGenome',
		best_fitness: tuple[float, float],
		current_threshold: float,
		config_dict: Optional[dict] = None,
		extra_state: Optional[dict] = None,
	) -> bool:
		"""
		Save checkpoint if at the right interval.

		Args:
			iteration: Current iteration (0-indexed)
			population: List of (genome, ce_fitness) tuples
			best_genome: Best genome found so far
			best_fitness: (CE, accuracy) of best genome
			current_threshold: Current threshold value
			config_dict: Optional config as dict
			extra_state: Optional extra state to save (patience, baseline, etc.)

		Returns:
			True if checkpoint was saved, False otherwise
		"""
		if not self.should_save(iteration):
			return False

		self.save(
			iteration=iteration,
			population=population,
			best_genome=best_genome,
			best_fitness=best_fitness,
			current_threshold=current_threshold,
			config_dict=config_dict,
			extra_state=extra_state,
		)
		return True

	def save(
		self,
		iteration: int,
		population: list[tuple['ClusterGenome', float]],
		best_genome: 'ClusterGenome',
		best_fitness: tuple[float, float],
		current_threshold: float,
		config_dict: Optional[dict] = None,
		extra_state: Optional[dict] = None,
	) -> None:
		"""Save checkpoint now (regardless of interval)."""
		import datetime

		path = self.checkpoint_path
		if path is None:
			return

		# Serialize population
		pop_data = []
		for genome, ce in population:
			gd = self._genome_to_dict(genome)
			# Try to get accuracy from cached fitness
			if hasattr(genome, 'metrics') and genome.metrics is not None:
				gd['fitness'] = genome.metrics.to_dict()
			else:
				gd['fitness'] = [ce, 0.0]
			pop_data.append(gd)

		# Build checkpoint data
		data = {
			'phase_name': self._phase_name,
			'optimizer_type': self._optimizer_type,
			'current_iteration': iteration,
			'total_iterations': self._total_iterations,
			'population': pop_data,
			'best_genome': self._genome_to_dict(best_genome),
			'best_fitness': list(best_fitness),
			'current_threshold': current_threshold,
			'config': config_dict or {},
			'extra_state': extra_state or {},
			'saved_at': datetime.datetime.now().isoformat(),
		}

		# Write atomically (temp file + rename)
		temp_path = path.with_suffix('.tmp')
		with open(temp_path, 'w') as f:
			json.dump(data, f, indent=2)
		temp_path.rename(path)

		# Record for dynamic-cadence accounting.
		import time
		self._last_save_monotonic = time.monotonic()
		self._last_save_gen = iteration

		self._logger(f"[Checkpoint] Saved at iteration {iteration + 1}/{self._total_iterations}")

	def load(self, genome_class: type) -> dict:
		"""
		Load checkpoint from file.

		Args:
			genome_class: The ClusterGenome class to use for reconstruction

		Returns:
			Dict with:
				- current_iteration: int
				- population: list of (genome, ce) tuples
				- best_genome: ClusterGenome
				- best_fitness: (CE, accuracy)
				- current_threshold: float
				- config: dict
				- extra_state: dict
		"""
		path = self.checkpoint_path
		if path is None or not path.exists():
			raise FileNotFoundError(f"No checkpoint found at {path}")

		with open(path, 'r') as f:
			data = json.load(f)

		# Reconstruct population
		population = []
		for gd in data['population']:
			genome = self._dict_to_genome(gd, genome_class)
			# fitness may be a dict (Metrics.to_dict) or a [ce, acc, ...] list.
			_fit = gd.get('fitness')
			if isinstance(_fit, dict):
				ce = _fit.get('ce', 0.0)
			elif _fit:
				ce = _fit[0]
			else:
				ce = 0.0
			# Restore cached fitness if available
			if gd.get('fitness'):
				from wnn.ram.metrics import Metrics as _M
				if isinstance(gd['fitness'], dict):
					genome.metrics = _M.from_dict(gd['fitness'])
				else:
					f = gd['fitness']
					genome.metrics = _M(ce=f[0], acc=f[1], f1=f[2] if len(f) > 2 else None, fpr=f[3] if len(f) > 3 else None)
			population.append((genome, ce))

		# Reconstruct best genome
		best_genome = self._dict_to_genome(data['best_genome'], genome_class)

		self._logger(f"[Checkpoint] Loaded from iteration {data['current_iteration'] + 1}")

		return {
			'current_iteration': data['current_iteration'],
			'population': population,
			'best_genome': best_genome,
			'best_fitness': tuple(data['best_fitness']),
			'current_threshold': data['current_threshold'],
			'config': data.get('config', {}),
			'extra_state': data.get('extra_state', {}),
			'saved_at': data.get('saved_at', ''),
		}

	@staticmethod
	def _genome_to_dict(genome: 'ClusterGenome') -> dict:
		"""Convert a ClusterGenome to a serializable dict."""
		return {
			'bits_per_neuron': list(genome.bits_per_neuron),
			'neurons_per_cluster': list(genome.neurons_per_cluster),
			'connections': list(genome.connections) if genome.connections else None,
		}

	@staticmethod
	def _dict_to_genome(d: dict, genome_class: type) -> 'ClusterGenome':
		"""Convert a dict back to a ClusterGenome.

		Supports both new format (bits_per_neuron) and legacy format (bits_per_cluster).
		"""
		if 'bits_per_neuron' in d:
			return genome_class(
				bits_per_neuron=d['bits_per_neuron'],
				neurons_per_cluster=d['neurons_per_cluster'],
				connections=d.get('connections'),
			)
		else:
			# Legacy format: expand bits_per_cluster to bits_per_neuron
			bits_per_cluster = d['bits_per_cluster']
			neurons_per_cluster = d['neurons_per_cluster']
			bits_per_neuron = []
			for bits, neurons in zip(bits_per_cluster, neurons_per_cluster):
				bits_per_neuron.extend([bits] * neurons)
			return genome_class(
				bits_per_neuron=bits_per_neuron,
				neurons_per_cluster=neurons_per_cluster,
				connections=d.get('connections'),
			)
