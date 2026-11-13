"""IDS56UC — cross-dataset validation of the IDS56U unweighted min-share vote (theta=0.02) on CICIDS2017 + CIC-IoT-2023
(Luiz 13/11/2026; pre-registration docs/ids56u_crossdataset_prereg.md, reviewed by experiment-design).

Per dataset (cicids, ciciot) x seed (r20419, r20420, r20421), against the PAUSED, never-run IDS-56 carry-over lattice
encon flow of that dataset+seed (C_on: cicids 7531/7533/7535, ciciot 7537/7539/7541), three NEW flows:
  U_on  = C_on + the IDS56U package (ids_write_rule=vote, ids_vote_share=0.02, balance_classes=false)  -> vs C_on
  C_off = C_on + ids_encoding_dedupe=off                                                                -> new control
  U_off = C_on + ids_encoding_dedupe=off + the IDS56U package                                          -> vs C_off
Every new flow clones its C_on base verbatim (same seed, same recipe); the overrides above are the ONLY differences,
and the per-dataset recipe keys are asserted (pinned), not overridden. 18 flows.

The flows are left PAUSED, not runnable. A new flow is created 'pending' and the API has no pending->paused
transition, so each flow goes pending -> PATCH queued -> POST /pause in immediate succession. That is race-free only
while another IDS flow is RUNNING (IDS needs 10 of 13 cores, so the scheduler cannot admit a second IDS flow); the
script refuses to run otherwise. Release later, one seed-quad at a time, with POST /api/flows/:id/resume (prereg §6).

Usage: python scripts/queue_ids56u_crossdataset.py [--dry-run]"""
import json, sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_ids56_vote_share_ab as Q  # noqa: E402

MIN_WORKER_ABI = 29
THETA = 0.02
SEEDS = (20419, 20420, 20421)
UNWEIGHTED_VOTE = {"ids_write_rule": "vote", "ids_vote_share": THETA, "balance_classes": False}
ENC_OFF = {"ids_encoding_dedupe": "off"}
# (cell tag, overrides on top of C_on); creation order inside a seed = U_on, C_off, U_off.
ARMS = (("nb02-encon", dict(UNWEIGHTED_VOTE)),
        ("lattice-encoff", dict(ENC_OFF)),
        ("nb02-encoff", {**ENC_OFF, **UNWEIGHTED_VOTE}))
CONTROLS = {"cicids": {20419: 7531, 20420: 7533, 20421: 7535},
            "ciciot": {20419: 7537, 20420: 7539, 20421: 7541}}
PINNED_COMMON = {"ids_split": "random_3way", "ids_classification": "binary", "memory_mode": "BINARY",
                 "ids_single_cluster": False, "ids_receptive_fields": "off", "ids_feature_ranking": "train_only",
                 "neuron_sample_rate": 0.25, "max_neurons": 250, "ids_n_bits": 96, "ids_k_folds": 5,
                 "ids_kfold_per_gen": 5, "ga_generations": 250, "patience": 5, "population_size": 50,
                 "ids_encoded_storage": "memory", "ids_write_rule": "lattice", "ids_vote_share": None,
                 "balance_classes": True, "ids_encoding_dedupe": "on"}
PINNED = {"cicids": {**PINNED_COMMON, "ids_dataset": "cicids2017", "max_bits": 34, "fitness_aggregation": "zscore",
                     "ga_crossover_bits": "blend_neuron"},
          "ciciot": {**PINNED_COMMON, "ids_dataset": "ciciot2023_neto_subsample", "max_bits": 100,
                     "fitness_aggregation": "desirability", "ga_crossover_bits": "blend_cluster"}}


def plan() -> list[tuple[str, str, int, int, dict]]:
	"""(dataset, cell tag, seed, base C_on id, overrides): dataset-major (cicids first), seed-major inside."""
	return [(ds, tag, s, CONTROLS[ds][s], dict(extra))
	        for ds in ("cicids", "ciciot") for s in SEEDS for tag, extra in ARMS]


def flow_name(base_name: str, tag: str) -> str:
	if not base_name.startswith("IDS56C-") or "-lattice-encon-r" not in base_name:
		raise SystemExit(f"base name {base_name!r} is not an IDS-56 carry-over lattice encon flow")
	return "IDS56UC-" + base_name[len("IDS56C-"):].replace("-lattice-encon-r", f"-{tag}-r")


def check_base(db, ds: str, seed: int, base: int) -> tuple[str, dict, str]:
	"""The base must be the never-run carry-over control of this dataset+seed, with the pre-registered recipe."""
	name, status, started, cfg = db.execute(
		"select name, status, started_at, config_json from flows where id=?", (base,)).fetchone()
	cfg = json.loads(cfg)
	params = cfg["params"]
	if status != "paused" or started is not None:
		raise SystemExit(f"base {base} is {status} (started {started}): C_on must be paused and never run")
	if params.get("seed") != seed or f"-{ds}-" not in name:
		raise SystemExit(f"base {base} ({name}) is not the {ds} r{seed} control")
	if params.get("ids_streaming"):
		raise SystemExit(f"base {base}: ids_streaming is on; the balance_classes audit assumes in-memory")
	bad = {k: (params.get(k), v) for k, v in PINNED[ds].items() if params.get(k) != v}
	if bad:
		raise SystemExit(f"base {base}: params drifted from the pre-registration {bad}")
	return name, params, cfg["template"]


def diff(base: dict, new: dict) -> dict:
	return {k: (base.get(k), new.get(k)) for k in sorted(set(base) | set(new)) if base.get(k) != new.get(k)}


def require_ids_running():
	running = [f for f in Q.call("GET", f"{Q.API}?status=running&limit=50")
	           if (f.get("config") or {}).get("params", {}).get("architecture_type", "ids") == "ids"]
	if not running:
		raise SystemExit("no IDS flow is running: pending->queued->pause could be admitted mid-way; refusing")


def require_worker_abi():
	import ram_accelerator
	abi = int(getattr(ram_accelerator, "ABI_VERSION", 0))
	if abi < MIN_WORKER_ABI:
		raise SystemExit(f"installed ram_accelerator ABI {abi} < {MIN_WORKER_ABI}: the controls will run on >= 29")


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


def describe(ds: str, tag: str, base: int) -> str:
	return (f"IDS56UC {ds} {tag}: clone of carry-over control {base} (lattice encon) with only the listed overrides; "
	        f"pre-registered in docs/ids56u_crossdataset_prereg.md. Worker ABI >= {MIN_WORKER_ABI}.")


def main(dry_run: bool):
	if not dry_run:
		require_worker_abi()
		require_ids_running()
	db = sqlite3.connect(Q.DB, uri=True)
	have = {r[0] for r in db.execute("select name from flows")}
	for ds, tag, seed, base, extra in plan():
		base_name, base_params, template = check_base(db, ds, seed, base)
		name = flow_name(base_name, tag)
		if name in have:
			print(f"[skip] {name} exists")
			continue
		params = {**base_params, **extra}
		Q.check(name, params)
		if dry_run:
			print(f"[dry] {name}  base {base}  diff-vs-base {json.dumps(diff(base_params, params))}  "
			      f"(inherited {len(params) - len(extra)} keys verbatim)")
			continue
		print(create_paused(name, describe(ds, tag, base), template, params), name, "paused")


if __name__ == "__main__":
	main("--dry-run" in sys.argv)
