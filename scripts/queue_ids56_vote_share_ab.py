"""IDS-56 min-share vote A/B (Luiz approved 09/11; docs/ids55_family_fix_decomposition.md, Lever B + Lever A).
Needs worker ABI 28 deployed (ids_vote_share). Seeds 20419-20423 (n=5 paired), receptive fields off,
BINARY 2-cluster, 500 neurons, every param that matters set explicitly.

 (a) unswt BINARY (clone 7125): {ids_write_rule lattice, vote theta=0.1} x {ids_encoding_dedupe off, on}.
     Harness: spread + distinct columns + MS 0.1 = 92.32 vs 91.35 BIN at equal cells.
 (b) unswr BINARY (clone 7431, the IDS-52 unswr lattice encoff base): the same 2x2. Harness predicts MS LOSES here
     (3.9% attacks: 87.28 vs 90.07) — this arm is the generalisation check.
 (c) unswt MULTICLASS pair: PARKED (Luiz 09/11: after the paper, i.e. after the n=100 flows). No BINARY multiclass
     base exists (5990-5994 MCSD are QUAD + an old recipe); a future draft would clone 7125 with
     ids_classification="multi" and choose neurons/class deliberately.

Report: test_summaries best_f1 margin_val_cal, paired by seed; adopt theta=0.1 only where it wins F1 without an
FPR loss (expect per-dataset). ciciot is NOT in this batch (doc lists it; task scoped it out).

Usage: python scripts/queue_ids56_vote_share_ab.py [--dry-run]"""
import json, sqlite3, ssl, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wnn.ram.experiments.ids_param_resolution import resolve_vote_share, resolve_write_rule  # noqa: E402
from wnn.ram.experiments.params import KNOWN_PARAMS  # noqa: E402

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
API = "https://127.0.0.1:3000/api/flows"
CTX = ssl._create_unverified_context()
EXPS = [{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
        {"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"}]
SEEDS = range(20419, 20424)
MIN_WORKER_ABI = 28
THETA = 0.1
RULES = (("lattice", {"ids_write_rule": "lattice", "ids_vote_share": None}),
         ("ms01", {"ids_write_rule": "vote", "ids_vote_share": THETA}))
DEDUPE = (("encoff", {"ids_encoding_dedupe": "off"}), ("encon", {"ids_encoding_dedupe": "on"}))
COMMON = {"memory_mode": "BINARY", "ids_single_cluster": False, "ids_receptive_fields": "off",
          "ids_feature_ranking": "legacy", "neuron_sample_rate": 0.25, "max_neurons": 250}


def call(method, url, body=None):
	err = None
	for _ in range(8):
		try:
			r = urllib.request.Request(url, method=method, headers={"Content-Type": "application/json"},
			                           data=None if body is None else json.dumps(body).encode())
			return json.load(urllib.request.urlopen(r, context=CTX, timeout=60))
		except Exception as e:
			err = e
			time.sleep(2)
	raise err


def plan() -> list[tuple[str, int, dict, str]]:
	"""(name, base flow, param overrides, description) for every flow, seed-major."""
	unswt = "IDS56-unswt-BINARY-2m-max250-xo_blend_neuron-16bWb-rate0.25"
	unswr = "IDS56-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy"
	# DATASET-major (Luiz 09/11): all unswt (~0.4 h/flow) first, then unswr (~4 h/flow), seed-major inside each,
	# so the fast unswt answer lands first under FIFO (lowest id first).
	out = []
	for prefix, base, label in ((unswt, 7125, "(a) unswt min-share A/B"),
	                            (unswr, 7431, "(b) unswr min-share A/B (harness predicts MS loses)")):
		for s in SEEDS:
			for rule, rp in RULES:
				for enc, ep in DEDUPE:
					out.append((f"{prefix}-{rule}-{enc}-r{s}", base, {**COMMON, **rp, **ep},
						f"IDS-56 {label}: {rule} x dedupe {enc}, ABI 28."))
	return out


def check(name: str, params: dict):
	"""Refuse a flow the worker would mislabel: unknown keys, or a write rule / share the resolvers reject."""
	unknown = sorted(k for k in params if k not in KNOWN_PARAMS)
	if unknown:
		raise SystemExit(f"{name}: unknown params {unknown}")
	resolve_write_rule(params)
	resolve_vote_share(params)


def require_worker_abi():
	import ram_accelerator
	abi = int(getattr(ram_accelerator, "ABI_VERSION", 0))
	if abi < MIN_WORKER_ABI:
		raise SystemExit(f"installed ram_accelerator ABI {abi} < {MIN_WORKER_ABI}: swap the worker wheel first")


def main(dry_run: bool):
	if not dry_run:
		require_worker_abi()
	db = sqlite3.connect(DB, uri=True)
	have = {r[0] for r in db.execute("select name from flows")}
	for name, base, extra, desc in plan():
		if name in have:
			continue
		cfg = json.loads(db.execute("select config_json from flows where id=?", (base,)).fetchone()[0])
		p = {**cfg["params"], **extra, "seed": int(name.rsplit("-r", 1)[1])}
		check(name, p)
		if dry_run:
			print(f"[dry] {name}  base {base}  " + " ".join(f"{k}={p.get(k)}" for k in
				("seed", "ids_split", "ids_classification", "memory_mode", "ids_write_rule", "ids_vote_share",
				 "ids_encoding_dedupe", "ids_receptive_fields", "max_neurons", "max_bits")))
			continue
		f = call("POST", API, {"name": name, "description": desc,
		                       "config": {"template": cfg["template"], "params": p}, "experiments": EXPS})
		call("PATCH", f"{API}/{f['id']}", {"status": "queued"})
		print(f["id"], name)


if __name__ == "__main__":
	main("--dry-run" in sys.argv)
