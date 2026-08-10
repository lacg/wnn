"""Create 2 CIC-IoT-2023 46M flows paired against the SP100 subsample cohort.

WHY. Every SP100-ciciot flow runs `ciciot2023_neto_subsample` — 1.43M rows sampled
from the 46.7M canonical Neto (ciciot2023.py:241). That is inherited from the
SP-abl cohort and the PUB50 neto-sub batches, so all the ciciot numbers are
internally consistent but say nothing about full scale. These two flows are the
same recipe at 46.7M, on the SAME SEEDS as the first two SP100-ciciot flows, so
the only variable is dataset scale.

TWO PARAMS CHANGE, and the second is not optional:
  ids_dataset          ciciot2023_neto_subsample -> ciciot2023_neto_full
  ids_encoded_storage  memory -> memmap

`memory` at 46.7M is how the box gets killed: the encoded matrix is materialized
in RAM, and the 03/07/2026 jetsam investigation traced the 46M OOMs to exactly
this class of sizing (calculate_pool_size at 3000/neuron reached 63 GB). Every
prior 46M flow in the DB (4534-4538, 4403) used memmap; those were PAUSED by hand,
not crashed, so memmap is the configuration that was actually surviving.

Everything else is byte-copied from flow 4984 (a completed SP100-ciciot run), so
the comparison is recipe-identical: 250 neurons, 250 GA generations, top20
features, 96 bits, random_3way, QUAD_WEIGHTED (memory_mode absent = worker
default).

Flows are created via POST /api/flows and then PATCHed to `queued` — POST leaves
them `pending`, which the worker never polls (worker.py list_flows(status=
"queued")), so a created-but-unqueued cohort looks perfect and does nothing.
"""
import json
import sqlite3
import sys

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DB = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
API = "https://localhost:3000/api/flows"
REFERENCE_FLOW = 4984          # completed SP100-ciciot subsample run
SEEDS = [24530, 29083]         # the first two SP100-ciciot seeds → paired comparison

EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search",
	 "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def main():
	db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
	cfg = json.loads(db.execute("select config_json from flows where id=?",
	                            (REFERENCE_FLOW,)).fetchone()[0])
	assert cfg["params"]["ids_dataset"] == "ciciot2023_neto_subsample", "reference moved"

	existing = {r[0] for r in db.execute(
		"select name from flows where name like 'SP100-ciciot46m-%'").fetchall()}

	created = 0
	for seed in SEEDS:
		name = f"SP100-ciciot46m-quad-96bWc-r{seed}"
		if name in existing:
			print(f"  {name}: exists, skipping")
			continue
		params = dict(cfg["params"])
		params["ids_dataset"] = "ciciot2023_neto_full"
		params["ids_encoded_storage"] = "memmap"   # NOT memory — see module docstring
		params["seed"] = seed
		body = {
			"name": name,
			"description": (f"CIC-IoT-2023 46.7M (neto_full), recipe-identical to "
			                f"SP100-ciciot subsample seed {seed} — scale is the only variable"),
			"config": {**cfg, "params": params},
			"experiments": EXPERIMENTS,
		}
		r = requests.post(API, json=body, verify=False, timeout=30)
		if r.status_code not in (200, 201):
			sys.exit(f"FAILED {name}: {r.status_code} {r.text[:200]}")
		fid = (r.json() or {}).get("id")
		if fid is None:
			sys.exit(f"{name}: created but no id returned — cannot queue")
		q = requests.patch(f"{API}/{fid}", json={"status": "queued"}, verify=False, timeout=30)
		if q.status_code != 200:
			sys.exit(f"{name}: created but QUEUE failed {q.status_code}")
		created += 1
		print(f"  created + queued {name} (id {fid})")

	db2 = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
	rows = db2.execute(
		"select id, name, status, json_extract(config_json,'$.params.ids_dataset'), "
		"json_extract(config_json,'$.params.ids_encoded_storage'), "
		"(select count(*) from experiments e where e.flow_id=f.id) "
		"from flows f where f.name like 'SP100-ciciot46m-%' order by id").fetchall()
	print(f"\ncreated {created}; verify:")
	for r in rows:
		print(f"  {r[0]} {r[1]} {r[2]} {r[3]} storage={r[4]} experiments={r[5]}")
		if r[5] == 0 or r[2] == "pending" or r[4] != "memmap":
			sys.exit("VERIFY FAILED — zero experiments, still pending, or wrong storage")
	print("verify OK")


if __name__ == "__main__":
	main()
