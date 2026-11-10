"""IDS-56 P2 — pre-registered theta=0.05 confirmation on unswt (Luiz 10/11; docs/ids56_readout.md §P2).
The offline theta sweep (read on TEST, hence post-hoc) put the encoff peak at theta~0.05 and encon flat over
0.02-0.15. This flies ms005 (vote, theta=0.05) x {encoff, encon} on the SAME 5 seeds, paired against the IDS-56
lattice and ms01 flows (7481-7500). Pre-registered: adopt theta=0.05 over 0.1 only if it wins F1 (GA best_f1,
margin_val_cal) paired >= 4/5 without an FPR loss, per dedupe arm. Same base/params as
queue_ids56_vote_share_ab.py except the share. Usage: python scripts/queue_ids56_theta005.py [--dry-run]"""
import json, sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_ids56_vote_share_ab as Q  # noqa: E402

THETA = 0.05


def plan() -> list[tuple[str, int, dict, str]]:
	prefix = "IDS56-unswt-BINARY-2m-max250-xo_blend_neuron-16bWb-rate0.25"
	rule = {"ids_write_rule": "vote", "ids_vote_share": THETA}
	return [(f"{prefix}-ms005-{enc}-r{s}", 7125, {**Q.COMMON, **rule, **ep},
	         f"IDS-56 P2 unswt theta=0.05 (pre-registered): ms005 x dedupe {enc}, ABI 28.")
	        for s in Q.SEEDS for enc, ep in Q.DEDUPE]


if __name__ == "__main__":
	Q.plan = plan
	Q.main("--dry-run" in sys.argv)
