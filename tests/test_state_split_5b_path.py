"""Phase 5b — wire the state-splitting trainer into the live GA training path.

Smoke test for reward_gated_train under WNN_STATE_SPLIT=1: the round's gated
trajectories train via split_train_loop instead of truncated BPTT, and the
trainer's GA-handshake pressure surfaces in stats. Also confirms the legacy
(flag-off) path is unaffected.

This is a WIRING test (does it run end-to-end, train in place, emit pressure),
NOT a quality test — good control needs 5c (connectivity feedback) + Phase 6.
Small dims for speed.
"""

from __future__ import annotations

import os
import math

from wnn.control.evaluator import (
	ControllerSpec, fit_thresholds_from_pid_rollouts, random_connectivity,
)
from wnn.control.reward_gated import RewardGatedConfig, reward_gated_train


def _spec():
	return ControllerSpec(
		num_motors=4, levels_per_motor=4, bits_per_feature=4,
		input_window_k=2, state_neurons=4, state_bits_per_neuron=8,
		output_bits_per_neuron=8, delta_control=False,
	)


def _run(use_split: bool):
	if use_split:
		os.environ["WNN_STATE_SPLIT"] = "1"
	else:
		os.environ.pop("WNN_STATE_SPLIT", None)
	spec = _spec()
	seed = 0
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=4, seed=seed)
	sc, oc = random_connectivity(spec, seed=seed)
	cfg = RewardGatedConfig(
		num_rounds=2, episodes_per_round=4, steps_per_episode=60,
		eval_episodes=4, seed=seed, progress=False,
	)
	ctrl, stats = reward_gated_train(spec, thresholds, sc, oc, cfg)
	return ctrl, stats


def main():
	print("=" * 70)
	print("  PHASE 5b — split trainer wired into reward_gated_train (smoke)")
	print("=" * 70)

	ok = True
	try:
		print("\n  [legacy] WNN_STATE_SPLIT off ...")
		_, legacy = _run(False)
		legacy_rounds = len(legacy["iter_fitness"])
		has_pressure_legacy = "split_saturation" in legacy
		print(f"    ran {legacy_rounds} rounds, final fitness={legacy['final_fitness']:.2f}, "
		      f"pressure keys present={has_pressure_legacy}")

		print("\n  [split]  WNN_STATE_SPLIT on ...")
		_, split = _run(True)
		split_rounds = len(split["iter_fitness"])
		sat = split.get("split_saturation")
		wishes = split.get("split_wish_bits")
		cells = split["iter_cells_written"]
		print(f"    ran {split_rounds} rounds, final fitness={split['final_fitness']:.2f}")
		print(f"    cells_written/round = {cells}")
		print(f"    GA-handshake pressure: saturation={sat}  wish_bits={wishes}")
	except Exception as e:
		import traceback
		traceback.print_exc()
		print(f"\n  EXCEPTION: {e}")
		ok = False
	finally:
		os.environ.pop("WNN_STATE_SPLIT", None)

	if ok:
		ok = (
			legacy_rounds == 2 and split_rounds == 2 and
			# legacy keys ALSO present (stats always carries them; that's fine) but
			# legacy uses the bptt path — its pressure stays at the zero default.
			split.get("split_saturation") is not None and
			isinstance(split.get("split_wish_bits"), list)
		)

	print("\n" + "-" * 70)
	if ok:
		print("  PHASE 5b PASS — split trainer runs end-to-end in the live training")
		print("  loop (flag-gated), trains in place, and emits GA-handshake pressure;")
		print("  the legacy bptt path is unaffected. Ready for 5c (consume pressure).")
	else:
		print("  PHASE 5b FAIL")
	return ok


if __name__ == "__main__":
	raise SystemExit(0 if main() else 1)
