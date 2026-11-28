"""CTRL-84: the MLP baseline as a SCORABLE CONTROLLER KIND for the CTRL-83 stress tooling.

id = "mlp<arm>_s<N>"  (arm a = 2x64, b = WNN-footprint-matched 2x576; N = the seed paired with l2bnm sN)
  payload : logs/controller/ctrl84/data/mlp<arm>_s<N>.npz  (the VAL-selected head, mlp_baseline.py)
  plant   : the l2bnm sN run (argv, episodes, the thresholds' train seed) — the MLP flies exactly the
            plant and the episodes its paired WNN is scored on.
Hook: ctrl83/score_cell.py accepts these ids (score_wnn(..., mlp=<payload>) -> ControllerEvaluator.score_mlp,
the same Rust scorer call the WNN row is captured from, with the same G1-G3 gates; G4 is n/a — no logged
END); stress_table.py reads them with an arm argument, e.g.
  stress_table.py "WNN l2bnm=l2bnm" "MLP a=mlpa" "MLP b=mlpb"
CTRL84_DATA overrides the payload directory (the smoke run writes outside the cohort's data/).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CTRL84_DATA", os.path.join(HERE, "data"))
ARMS = {"a": (64, 64), "b": (576, 576)}
SEEDS = (1, 2, 3, 4, 5)
BASE_ARM = "l2bnm"   # the paired WNN arm (ctrl83/cells.py EXTRA_WNN)
MLP_IDS = tuple(f"mlp{a}_s{s}" for a in ARMS for s in SEEDS)


def is_mlp(k: str) -> bool:
	return k in MLP_IDS


def arm_seed(k: str) -> tuple:
	arm, s = k[3:].split("_s")
	return arm, int(s)


def payload_path(k: str) -> str:
	return os.path.join(DATA, f"{k}.npz")


def base_id(k: str) -> str:
	"""The ctrl83/cells.py controller id of the paired WNN (its run supplies argv / plant / episodes)."""
	return f"{BASE_ARM}_s{arm_seed(k)[1]}"
