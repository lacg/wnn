"""CTRL-56 (17/11/2026): stage checkpoints and ARCH-ONLY genomes (cells=None).

A controller genome with no trained cells is an architecture, not a controller.
Scoring it score-only (`ControllerEvaluator.score_genomes`) builds an EMPTY memory
and measures that: 17/11, Y s1 GRID#2 reloaded from its stage checkpoint scored
0.25 % stable / 49 deg steady, while the same genome had flown 100 % in the run.

How they got into checkpoints: the GRID search never writes cells back, and the
GRID held-out trains only the genomes it scores — pop[0] plus a
`--holdout-pop-sample` draw seeded by the FIRST test seed (99990101 in every run,
so GRID#2 was always left out). The stage-0 checkpoint is written after that, so
42 of 50 GRID entries (incl. GRID#2) were saved arch-only in EVERY grid checkpoint.

Contract since CTRL-56:
  * the stage-0 writer trains COPIES of the arch-only entries before saving
    (phased_ga._materialize_arch_only_cells) and records their indices under
    meta["cells_materialized_idx"], so the checkpoint is scoreable entry-by-entry
    while the live run (and a resume of it) is unchanged;
  * loaders refuse arch-only entries unless the caller says it retrains them
    (the held-out / stage-select / --recalc-headline path does) or only inspects.
"""


class ArchOnlyEntryError(ValueError):
	"""A checkpoint entry carries no trained cells and the caller would score it."""


def arch_only_indices(genomes) -> list:
	"""Indices of the genomes that carry no trained cells."""
	return [i for i, g in enumerate(genomes or []) if getattr(g, "cells", None) is None]


def refuse_arch_only(genomes, where: str, labels=None) -> None:
	"""Raise ArchOnlyEntryError when any genome is arch-only. `labels[i]` names entry i."""
	bad = arch_only_indices(genomes)
	if not bad:
		return
	names = [labels[i] if labels else f"#{i}" for i in bad]
	raise ArchOnlyEntryError(
		f"{where}: {len(bad)} entr{'y' if len(bad) == 1 else 'ies'} {names[:12]}"
		f"{' ...' if len(names) > 12 else ''} carr{'ies' if len(bad) == 1 else 'y'} NO trained cells "
		f"(arch-only; pre-CTRL-56 GRID checkpoint). Scoring it score-only would measure an EMPTY "
		f"controller. Use the run's --save-winner file for the published genome, or retrain it on "
		f"the train seed exactly as the held-out does (phased_ga._holdout_report arch-only branch); "
		f"pass allow_arch_only=True only from a caller that does one of those or just inspects.")


def strip_materialized_cells(payload: dict) -> list:
	"""Undo the checkpoint-only training on a RESUME, so the resumed run seeds its next
	stage from the population the live run had (those entries were arch-only there).
	Returns the stripped indices ([] for pre-CTRL-56 files, which need nothing)."""
	idx = list(((payload or {}).get("meta") or {}).get("cells_materialized_idx") or [])
	pop = payload.get("population") or []
	for i in idx:
		if i < len(pop):
			pop[i].cells = None
	if 0 in idx and payload.get("best_genome") is not None:
		payload["best_genome"].cells = None
	return idx
