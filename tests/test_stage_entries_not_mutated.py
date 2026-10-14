"""A FRESH run's stage-select must rank the genomes its stage checkpoints hold.

Question (13/10/2026, resume-pool fix follow-up): can a later stage mutate an
earlier stage's GRID/NEURONS entries IN PLACE before `_select_headline_stage`
reads them? If it could, a resumed run (which reloads those entries from the
stage checkpoints) would rank different genomes than an uninterrupted one.

Verified, not assumed: a TINY real in-process run (sn=0, 8 output neurons,
2 episodes x 50 steps, pop 4, one generation per stage — ~5 s, ~0.6 GB). The
selector is spied at entry and every candidate's serialized genome (connectivity
+ packed cells) is compared with the matching member of its stage checkpoint.

Code path that makes it hold: OptimizationTemplate.seed_population clones every
carried genome (cells included) before a stage touches it, and
_release_prior_populations rebinds a trimmed list rather than editing one.

Scope of the equality. GRID's checkpoint is written AFTER its held-out, stages
1-4's BEFORE (so a held-out crash still leaves the finished stage on disk). With
--lamarckian (every production recipe) the candidates already carry cells, the
held-out is score-only, and the equality is byte-exact. Without it, the stage's
OWN held-out stamps train-seed cells into its arch-only pop[0] after the save —
not a later stage — so that variant pins connectivity only; a resume re-runs the
same held-out on the reloaded genomes (_reload_completed_stages).

Run: PYTHONPATH=src python -m pytest tests/test_stage_entries_not_mutated.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("ram_controller")

from wnn.control import phased_ga as P
from wnn.ram.strategies.phased import load_checkpoint_top_k, ControllerGenomeCodec

TINY = ["--grid-state-neurons", "0", "--max-state-neurons", "0",
        "--grid-bits", "8", "--grid-output-neurons", "8", "--max-output-neurons", "8",
        "--skip-stages", "bits,connections", "--neurons-gens", "1", "--neurons-patience", "1",
        "--memory-gens", "1", "--memory-patience", "1", "--pop", "4", "--grid-top-k", "4",
        "--eval-episodes", "2", "--memory-eval-episodes", "2", "--report-episodes", "2",
        "--steps", "50", "--universe-episodes", "2", "--report-seeds", "11",
        "--base-seed", "5", "--runs", "1", "--check-interval", "1",
        "--train-workers", "2", "--grid-min-suffix", "1"]


def _fly(monkeypatch, tmp_path, extra):
	"""Run the tiny recipe; return {label: [serialized top-3 at stage-select entry]}."""
	seen = {}
	real = P._select_headline_stage

	def spy(args, ec, seeds, entries, holdouts):
		for label, _spec, res in entries:
			pop = list(res.final_population or [res.best_genome])[:3]
			seen[label] = [g.serialize() for g in pop]
		return real(args, ec, seeds, entries, holdouts)

	monkeypatch.setattr(P, "_select_headline_stage", spy)
	# main() records its seed set in <repo>/db/wnn.db — never let a test row land in
	# the live dashboard DB when this runs from the main checkout.
	monkeypatch.setenv("WNN_DB_PATH", str(tmp_path / "seeds.db"))
	monkeypatch.setattr(sys, "argv", ["phased_ga", *TINY,
	                                  "--save-stage-checkpoints", str(tmp_path), *extra])
	P.main()
	return seen


def _checkpoint_top3(tmp_path, label):
	path = next(tmp_path.glob(f"stage*_{label.lower()}.yaml.gz"))
	ck = load_checkpoint_top_k(path, ControllerGenomeCodec(), 3)
	return [g.serialize() for g in ck.final_population]


def _connectivity(ser):
	return {k: v for k, v in ser.items() if k != "cells"}


def test_lamarckian_entries_equal_their_checkpoints_byte_for_byte(monkeypatch, tmp_path):
	seen = _fly(monkeypatch, tmp_path, ["--lamarckian"])
	assert set(seen) == {"GRID", "NEURONS", "MEMORY"}
	for label in ("GRID", "NEURONS", "MEMORY"):
		ck = _checkpoint_top3(tmp_path, label)
		assert len(seen[label]) == len(ck) == 3
		for i, (live, saved) in enumerate(zip(seen[label], ck)):
			assert live == saved, f"{label}#{i} at stage-select differs from its checkpoint"


def test_non_lamarckian_connectivity_is_never_touched_by_later_stages(monkeypatch, tmp_path):
	seen = _fly(monkeypatch, tmp_path, [])
	for label in ("GRID", "NEURONS"):
		ck = _checkpoint_top3(tmp_path, label)
		assert [_connectivity(s) for s in seen[label]] == [_connectivity(s) for s in ck], label
