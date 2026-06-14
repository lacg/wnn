"""Tests for the shared adaptive-cadence checkpoint decision (13/06/2026).

The clock is injected so we exercise the wall-clock branch with zero real
sleeps. Run: PYTHONPATH=src python tests/test_save_cadence.py
"""
from wnn.ram.strategies.phased.cadence import SaveCadence

PASS = "[PASS]"


class _Clock:
	"""Manually-advanced monotonic clock."""
	def __init__(self): self.t = 1000.0
	def __call__(self): return self.t
	def advance(self, dt): self.t += dt


def test_first_gen_is_baseline_only():
	c = SaveCadence(target_loss_seconds=300, max_interval=10, monotonic=_Clock())
	assert c.should_save_now(0) is False, "first gen only sets the baseline"
	print(f"  {PASS} first generation establishes baseline, no save")


def test_slow_gen_saves_every_gen():
	clk = _Clock()
	c = SaveCadence(target_loss_seconds=300, max_interval=10, monotonic=clk)
	c.should_save_now(0)            # baseline
	clk.advance(2400)              # a 40-min gen elapsed
	assert c.should_save_now(1) is True, "elapsed >> budget ⇒ save"
	c.mark_saved(1)
	clk.advance(2400)
	assert c.should_save_now(2) is True, "every slow gen saves"
	print(f"  {PASS} slow gen (40min) trips budget → saves EVERY gen")


def test_fast_gen_throttles_to_max_interval():
	clk = _Clock()
	c = SaveCadence(target_loss_seconds=300, max_interval=10, monotonic=clk)
	c.should_save_now(0)           # baseline
	# 30s gens: budget (300s) not hit until ~10 gens, but max_interval caps at 10.
	saved = []
	for g in range(1, 25):
		clk.advance(30)
		if c.should_save_now(g):
			saved.append(g)
			c.mark_saved(g)
	# First save at the gen where elapsed≥300 (g=10: 9*30=270 <300; g=11:330≥300) OR
	# max_interval 10 (g=10). max_interval wins at g=10, then g=20, ...
	assert saved and saved[0] in (10, 11), f"first throttled save near gen 10, got {saved[:1]}"
	gaps = [b - a for a, b in zip(saved, saved[1:])]
	assert all(gp <= 10 for gp in gaps), f"never exceed max_interval gens: {gaps}"
	print(f"  {PASS} fast gen (30s) throttles I/O to ≤max_interval (saves at {saved})")


def test_none_budget_saves_every_gen():
	c = SaveCadence(target_loss_seconds=None, max_interval=10, monotonic=_Clock())
	assert all(c.should_save_now(g) is True for g in range(5)), "None budget = legacy every-gen"
	print(f"  {PASS} target_loss_seconds=None → every-gen (legacy)")


def test_zero_budget_saves_every_gen_after_baseline():
	clk = _Clock()
	c = SaveCadence(target_loss_seconds=0.0, max_interval=10, monotonic=clk)
	assert c.should_save_now(0) is False, "still a baseline on gen 0"
	assert c.should_save_now(1) is True, "budget 0 ⇒ every gen after baseline"
	print(f"  {PASS} budget=0 → save every gen (after baseline)")


def test_phased_manager_cadence_from_config():
	"""The unified PhasedCheckpointManager (now used by BOTH IDS and controller)
	drives its save decision from a SaveCadence built off the IDS CheckpointConfig."""
	import tempfile
	from pathlib import Path
	from wnn.ram.strategies.connectivity.checkpoint_manager import CheckpointConfig
	from wnn.ram.strategies.phased import PhasedCheckpointManager, ClusterGenomeCodec
	cfg = CheckpointConfig(enabled=True, target_loss_seconds=300, max_interval=10,
	                       checkpoint_dir=Path(tempfile.mkdtemp()), filename_prefix="ga_checkpoint")
	mgr = PhasedCheckpointManager(cfg.checkpoint_dir / f"{cfg.filename_prefix}_ga",
	                              ClusterGenomeCodec(),
	                              SaveCadence(cfg.target_loss_seconds, cfg.max_interval, monotonic=_Clock()))
	assert mgr.should_save_now(0) is False, "first gen is the cadence baseline"
	print(f"  {PASS} PhasedCheckpointManager cadence built from IDS CheckpointConfig")


if __name__ == "__main__":
	tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
	print(f"Running {len(tests)} save-cadence tests...")
	for t in tests:
		t()
	print("ALL PASS")
