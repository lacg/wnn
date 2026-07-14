"""
ControllerCancelMixin — the controller strategies' per-generation hook, on the
SHARED cooperative-cancel core (GenericGAStrategy._checkpoint_and_maybe_stop).

The adaptive crash-save + cooperative-shutdown logic lives once in the base; this
mixin only supplies the controller-flavoured `_build_checkpoint` (spec +
fitness_weights + live generation) so `--resume-from-emergency` reloads exactly the
payload the controller resume path expects (via checkpoint_io.payload_to_checkpoint).

The strategy must have (set by `_wire_cancel` in phased_ga.py before optimize()):
`_spec`, `_config` (GAConfig with fitness_weight_*), and — when checkpointing —
`_checkpoint_mgr`, `_shutdown_check`, `_stage_num`, `_stage_name`, `_checkpoint_meta`.
Absent `_checkpoint_mgr`/`_shutdown_check`, the base core is a no-op (a run without
--save-stage-checkpoints and without a cancel).
"""

from __future__ import annotations


class ControllerCancelMixin:
	"""Mixed into the controller GA strategies to share the cancel/crash-save core."""

	def _on_generation_start(self, generation: int, **ctx) -> None:
		# All the logic (adaptive crash-save + cooperative shutdown → StopIteration)
		# is the shared base implementation; the controller only customises the
		# checkpoint payload via _build_checkpoint below.
		self._checkpoint_and_maybe_stop(generation, ctx)

	def _build_checkpoint(self, generation: int, genomes: list, ctx: dict,
	                      complete: bool):
		"""Controller GA state → PhaseCheckpoint via the historical payload shape,
		so the existing resume path (checkpoint_to_payload) round-trips it."""
		from wnn.control.checkpoint_io import payload_to_checkpoint
		cfg = self._config
		payload = {
			"stage_num": getattr(self, "_stage_num", None),
			"stage_name": getattr(self, "_stage_name", None),
			"spec": self._spec,
			"population": genomes,                       # bare genomes (base unpacked)
			"best_genome": ctx.get("best_genome"),
			"generation": generation,
			"fitness_weights": {
				"err_sq": getattr(cfg, "fitness_weight_err_sq", 1.0),
				"stable": getattr(cfg, "fitness_weight_stable", 0.0),
				"jerk":   getattr(cfg, "fitness_weight_jerk", 0.0),
				"mono":   getattr(cfg, "fitness_weight_mono", 0.0),
				"steady": getattr(cfg, "fitness_weight_steady", 0.0),
			},
			"meta": {**getattr(self, "_checkpoint_meta", {}),
			         "emergency_dump": not complete},
		}
		return payload_to_checkpoint(payload)
