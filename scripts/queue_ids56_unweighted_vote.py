"""IDS-56 U — pre-registered UNWEIGHTED min-share vote on unswr (Luiz 10/11: "recommendation 2, n=3 first";
docs/ids56_unswr_tiny_genomes.md §6.2, pre-registration docs/ids56_unweighted_vote_prereg.md).

Arm: clone 7501 (unswr lattice encoff r20419) with ids_write_rule=vote, ids_vote_share=THETA,
balance_classes=false. Audit (prereg §1): balance_classes only feeds the per-visit training weight; lattice ignores
it (CPU eval_export.rs + GPU marker_train.metal), so vs the lattice control the arm changes exactly the writer (the
unweighted min-share tally) and nothing in folds, negatives, fitness, CE or threshold calibration.
Control: the existing IDS-56 unswr lattice encoff flows on the same seeds (7509 / 7513 / 7517), NOT re-flown.
Seeds r20421-23 (n=3) — the seeds theta was NOT tuned on (experiment-design review 10/11, prereg §5); extend with
--seeds 20419,20420 (the tuning seeds, controls 7501/7505, disclosed) ONLY if the prereg's ambiguity rule fires.

Usage: python scripts/queue_ids56_unweighted_vote.py [--dry-run] [--seeds 20419,20420]"""
import json, sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_ids56_vote_share_ab as Q  # noqa: E402

THETA = 0.02
BASE = 7501
SEEDS_N3 = (20421, 20422, 20423)  # clean seeds; extension = 20419,20420 (theta-tuning seeds)
TAG = "nb" + f"{THETA:.2f}".replace("0.", "")  # nb02 = unweighted (no balance) share 0.02
PREFIX = "IDS56U-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy"
# Everything else is inherited from 7501 verbatim; these are the ONLY differences from the control.
ARM = {"ids_write_rule": "vote", "ids_vote_share": THETA, "balance_classes": False,
       "ids_encoding_dedupe": "off"}
# Pinned so a drift in the base flow's params cannot silently change the arm (asserted, not overridden).
PINNED = {"ids_dataset": "unsw-nb15", "ids_split": "random_3way", "ids_classification": "binary",
          "memory_mode": "BINARY", "ids_single_cluster": False, "ids_receptive_fields": "off",
          "ids_feature_ranking": "legacy", "neuron_sample_rate": 0.25, "max_neurons": 250, "max_bits": 34,
          "ids_k_folds": 5, "ids_kfold_per_gen": 5, "fitness_aggregation": "desirability"}


def seeds_from_argv() -> tuple[int, ...]:
	if "--seeds" not in sys.argv:
		return SEEDS_N3
	return tuple(int(s) for s in sys.argv[sys.argv.index("--seeds") + 1].split(","))


def plan() -> list[tuple[str, int, dict, str]]:
	return [(f"{PREFIX}-{TAG}-encoff-r{s}", BASE, dict(ARM),
	         f"IDS-56 U unswr unweighted min-share vote theta={THETA} (pre-registered, "
	         f"docs/ids56_unweighted_vote_prereg.md): clone {BASE} + vote/{THETA}/balance_classes=false, "
	         f"paired vs the IDS-56 lattice encoff flow of the same seed. ABI 28.")
	        for s in seeds_from_argv()]


def check_pinned(name: str, params: dict):
	bad = {k: (params.get(k), v) for k, v in PINNED.items() if params.get(k) != v}
	if bad:
		raise SystemExit(f"{name}: base params drifted from the pre-registration {bad}")


def verify_experiments(flow_id: int):
	db = sqlite3.connect(Q.DB, uri=True)
	n = db.execute("select count(*) from experiments where flow_id=?", (flow_id,)).fetchone()[0]
	if n != len(Q.EXPS):
		raise SystemExit(f"flow {flow_id}: {n} experiments, expected {len(Q.EXPS)} — do NOT let it run")


def main(dry_run: bool):
	if not dry_run:
		Q.require_worker_abi()
	db = sqlite3.connect(Q.DB, uri=True)
	have = {r[0] for r in db.execute("select name from flows")}
	cfg = json.loads(db.execute("select config_json from flows where id=?", (BASE,)).fetchone()[0])
	for name, base, extra, desc in plan():
		if name in have:
			print(f"[skip] {name} exists")
			continue
		p = {**cfg["params"], **extra, "seed": int(name.rsplit("-r", 1)[1])}
		Q.check(name, p)
		check_pinned(name, p)
		if dry_run:
			print(f"[dry] {name}  base {base}  " + " ".join(f"{k}={p.get(k)}" for k in
				("seed", "ids_split", "memory_mode", "ids_write_rule", "ids_vote_share", "balance_classes",
				 "ids_encoding_dedupe", "max_neurons", "max_bits", "ids_k_folds", "ids_kfold_per_gen",
				 "fitness_aggregation", "ga_generations", "patience")))
			print("      params=" + json.dumps(p, sort_keys=True))
			continue
		f = Q.call("POST", Q.API, {"name": name, "description": desc,
		                           "config": {"template": cfg["template"], "params": p}, "experiments": Q.EXPS})
		verify_experiments(f["id"])
		Q.call("PATCH", f"{Q.API}/{f['id']}", {"status": "queued"})
		print(f["id"], name)


if __name__ == "__main__":
	main("--dry-run" in sys.argv)
