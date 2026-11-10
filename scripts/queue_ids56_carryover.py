"""IDS-56 P1 — carry-over of the min-share vote to CICIDS2017 + CIC-IoT-2023 (Luiz 10/11; docs/ids56_readout.md §P1).
n=3 FIRST (seeds r20419-21; extend with r20422-23 as further paired flows only if ambiguous — Luiz's standing rule).
Both arms encon (the vote only paid with dedupe on, unswt): lattice vs ms01 (vote, theta=0.1, as tested).
Bases: cicids clone 7219, ciciot clone 7223 (random_3way). Overrides = the readout's P1 list (train_only ranking per
IDS-53, both arms share it). Budget inherited, identical in both arms.
Usage: python scripts/queue_ids56_carryover.py [--dry-run] [--seeds 20419,20420,20421]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_ids56_vote_share_ab as Q  # noqa: E402

SEEDS = (20419, 20420, 20421)
COMMON = {**Q.COMMON, "ids_feature_ranking": "train_only", "ids_encoding_dedupe": "on"}
DATASETS = (("cicids", 7219, "IDS56C-cicids-CE20-zs-BINARY-2m-max250-xo_blend_neuron-rate0.25"),
            ("ciciot", 7223, "IDS56C-ciciot-B15AC-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25"))


def plan(seeds=SEEDS) -> list[tuple[str, int, dict, str]]:
	"""Dataset-major (cicids ~3-7 h/flow first, then ciciot ~13-19 h/flow), seed-major inside."""
	return [(f"{prefix}-{rule}-encon-r{s}", base, {**COMMON, **rp},
	         f"IDS-56 P1 carry-over ({ds}, n=3 first): {rule} x encon, ABI 28.")
	        for ds, base, prefix in DATASETS for s in seeds for rule, rp in Q.RULES]


if __name__ == "__main__":
	seeds = SEEDS
	if "--seeds" in sys.argv:
		seeds = tuple(int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(","))
	Q.plan = lambda: plan(seeds)
	Q.main("--dry-run" in sys.argv)
