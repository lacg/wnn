"""IDS58 — the two unswr adoptions flown together: unweighted min-share vote (IDS56U, theta=0.02) + max_bits 64 (IDS-57
A1). Luiz approved 13/11/2026 ("yes, let's do this first"); pre-registration docs/ids58_vote_b64_prereg.md.

Per seed r20421-23, ONE new flow C = the lattice control of that seed (7509 / 7513 / 7517) with exactly four overrides:
ids_write_rule=vote, ids_vote_share=0.02, balance_classes=false, max_bits=64. No new controls. The script proves the
pairing by construction before it creates anything:
  C - lattice control (7509/7513/7517)  == the four overrides            (descriptive contrast)
  C - A1 flow         (7546/7547/7548)  == the IDS56U package, 3 keys    (PRIMARY: does the vote add at 64 bits?)
  C - IDS56U flow     (7543/7544/7545)  == max_bits 34 -> 64, 1 key      (SECONDARY: do 64 bits add on top of the vote?)
and that every base flow is completed with the pinned recipe.

The flows are created PAUSED (pending -> PATCH queued -> POST /pause -> GET confirms paused). That is race-free only
while another IDS flow is RUNNING (IDS needs 10 of 13 cores, so the scheduler cannot admit a second one); the script
refuses otherwise. Release is NOT this script's job: prereg §5 (VAL futility gate on the A1 genomes) must pass first,
then POST /api/flows/:id/resume on Luiz's go.

Usage: python scripts/queue_ids58_vote_b64.py [--dry-run]"""
import json, sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_ids56_vote_share_ab as Q  # noqa: E402

WORKER_ABI = 29  # the A1 controls of the primary contrast ran on 29; another ABI makes the primary cross-build
THETA = 0.02
MAX_BITS = 64
UNWEIGHTED_VOTE = {"ids_write_rule": "vote", "ids_vote_share": THETA, "balance_classes": False}
OVERRIDES = {**UNWEIGHTED_VOTE, "max_bits": MAX_BITS}
# seed -> (lattice control, IDS56U arm, IDS-57 A1 arm); all three completed, all on the same seed.
BASES = {20421: (7509, 7543, 7546), 20422: (7513, 7544, 7547), 20423: (7517, 7545, 7548)}
LATTICE_PREFIX = "IDS56-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-lattice-encoff-r"
NEW_PREFIX = "IDS58-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-b64-encoff-r"
# Asserted on the lattice control, never overridden (same list as scripts/queue_ids57_a1.py).
PINNED = {"ids_dataset": "unsw-nb15", "ids_split": "random_3way", "ids_classification": "binary",
          "memory_mode": "BINARY", "ids_single_cluster": False, "ids_receptive_fields": "off",
          "ids_write_rule": "lattice", "ids_vote_share": None, "ids_encoding_dedupe": "off",
          "ids_feature_ranking": "legacy", "neuron_sample_rate": 0.25, "max_neurons": 250, "max_bits": 34,
          "ids_k_folds": 5, "ids_kfold_per_gen": 5, "ids_offspring_folds": "all",
          "fitness_aggregation": "desirability", "ga_generations": 250, "patience": 5, "population_size": 50,
          "balance_classes": True, "wnn_order_independent_train": True}


def diff(base: dict, new: dict) -> dict:
	return {k: (base.get(k), new.get(k)) for k in sorted(set(base) | set(new)) if base.get(k) != new.get(k)}


def expected(base: dict, overrides: dict) -> dict:
	return {k: (base.get(k), v) for k, v in sorted(overrides.items())}


def read_flow(db, flow_id: int) -> tuple[str, str, str, dict]:
	name, status, cfg = db.execute("select name, status, config_json from flows where id=?", (flow_id,)).fetchone()
	cfg = json.loads(cfg)
	return name, status, cfg["template"], cfg["params"]


def check_lattice_control(db, flow_id: int, seed: int) -> tuple[str, str, dict]:
	name, status, template, p = read_flow(db, flow_id)
	bad = {k: (p.get(k), v) for k, v in PINNED.items() if p.get(k) != v}
	if status != "completed" or bad or int(p["seed"]) != seed or not name.startswith(LATTICE_PREFIX):
		raise SystemExit(f"control {flow_id} ({name}, {status}): not the pinned r{seed} lattice control {bad}")
	return name, template, p


def check_paired_arm(db, flow_id: int, seed: int, template: str, new: dict, must_differ_by: dict, label: str) -> dict:
	"""The new flow must differ from this completed arm in exactly `must_differ_by`, nothing else."""
	name, status, arm_template, p = read_flow(db, flow_id)
	if status != "completed" or int(p["seed"]) != seed or arm_template != template:
		raise SystemExit(f"{label} {flow_id} ({name}): status {status}, seed {p.get('seed')}, template {arm_template}")
	d = diff(p, new)
	if d != expected(p, must_differ_by):
		raise SystemExit(f"{label} {flow_id}: new flow differs by {d}, expected exactly {must_differ_by}")
	return d


def require_ids_running():
	running = [f for f in Q.call("GET", f"{Q.API}?status=running&limit=50")
	           if (f.get("config") or {}).get("params", {}).get("architecture_type", "ids") == "ids"]
	if not running:
		raise SystemExit("no IDS flow is running: pending->queued->pause could be admitted mid-way; refusing")


def require_worker_abi():
	import ram_accelerator
	abi = int(getattr(ram_accelerator, "ABI_VERSION", 0))
	if abi != WORKER_ABI:
		raise SystemExit(f"installed ram_accelerator ABI {abi} != {WORKER_ABI}: the primary contrast (vs A1 on ABI 29) "
		                 f"would be cross-build; prereg §7 needs an equivalence record first")


def verify_experiments(flow_id: int):
	db = sqlite3.connect(Q.DB, uri=True)
	n = db.execute("select count(*) from experiments where flow_id=?", (flow_id,)).fetchone()[0]
	if n != len(Q.EXPS):
		raise SystemExit(f"flow {flow_id}: {n} experiments, expected {len(Q.EXPS)} — leave it pending, fix by hand")


def create_paused(name: str, desc: str, template: str, params: dict) -> int:
	f = Q.call("POST", Q.API, {"name": name, "description": desc,
	                           "config": {"template": template, "params": params}, "experiments": Q.EXPS})
	verify_experiments(f["id"])
	require_ids_running()
	Q.call("PATCH", f"{Q.API}/{f['id']}", {"status": "queued"})
	Q.call("POST", f"{Q.API}/{f['id']}/pause")
	got = Q.call("GET", f"{Q.API}/{f['id']}")
	if got.get("status") != "paused":
		raise SystemExit(f"flow {f['id']} {name}: status {got.get('status')} after pause — check it NOW")
	return f["id"]


def describe(control: int, u: int, a1: int) -> str:
	return (f"IDS58 (pre-registered, docs/ids58_vote_b64_prereg.md): clone of lattice control {control} with "
	        f"ids_write_rule=vote, ids_vote_share={THETA}, balance_classes=false, max_bits={MAX_BITS} only. "
	        f"Primary pair: A1 {a1} (differs by the vote package). Secondary pair: IDS56U {u} (differs by max_bits). "
	        f"Worker ABI {WORKER_ABI}.")


def plan_seed(db, seed: int) -> tuple[str, str, dict, str]:
	control, u, a1 = BASES[seed]
	control_name, template, cp = check_lattice_control(db, control, seed)
	params = {**cp, **OVERRIDES}
	Q.check(control_name, params)
	d_ctl = diff(cp, params)
	if d_ctl != expected(cp, OVERRIDES):
		raise SystemExit(f"r{seed}: diff vs control {d_ctl} != the four overrides")
	d_a1 = check_paired_arm(db, a1, seed, template, params, UNWEIGHTED_VOTE, "A1")
	d_u = check_paired_arm(db, u, seed, template, params, {"max_bits": MAX_BITS}, "IDS56U")
	name = NEW_PREFIX + str(seed)
	print(f"{name}\n  base {control} (lattice, 34b)  diff-vs-base {json.dumps(d_ctl)}  "
	      f"(inherited {len(params) - len(OVERRIDES)} keys verbatim)\n"
	      f"  vs A1 {a1} (PRIMARY)      {json.dumps(d_a1)}\n"
	      f"  vs IDS56U {u} (SECONDARY) {json.dumps(d_u)}")
	return name, template, params, describe(control, u, a1)


def main(dry_run: bool):
	if not dry_run:
		require_worker_abi()
		require_ids_running()
	db = sqlite3.connect(Q.DB, uri=True)
	have = {r[0] for r in db.execute("select name from flows")}
	for seed in BASES:
		name, template, params, desc = plan_seed(db, seed)
		if name in have:
			print(f"  [skip] {name} exists")
			continue
		if dry_run:
			print("  [dry] not created")
			continue
		print(" ", create_paused(name, desc, template, params), name, "paused")


if __name__ == "__main__":
	main("--dry-run" in sys.argv)
