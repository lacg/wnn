"""UNSW lever screens on unswt (Plane IDS-64 C, IDS-65 D, IDS-72 MDL), designed 22/11/2026 for Luiz's
approval — NOTHING is queued until he picks the arms. Every arm is a clone of the PRODUCTION unswt recipe
(IDS-73 Eon, adopted 22/11 = flows 7600/7601/7602, seeds r20421-23) with ONE setting changed, paired per
seed vs that flow (the control is NOT re-flown):

  C    (IDS-64) ids_feature_ranking=train_only: the top-20 re-derived by RF on this split's own TRAIN rows
       (feature_rankings/train_only_v1/unsw-nb15_temporal_3way.json). 12/20 features shared; drops the 3
       categoricals; cut-E budget 288 -> 320, 318 bits after dedupe (control: 287).
  D24  (IDS-65) equal-cell width-vs-count: max_bits 34 -> 24, max_neurons 250 -> 455. The control's GA sits
       at both caps (499 neurons, 32-34 b, 0.265 M cells); 24 b trains 292.7 cells/neuron (evolved-prefix
       measurement on the 3 control genomes), so 2 x 455 neurons ~ 0.266 M cells.
  D16  (IDS-65, optional 2nd rung) max_bits 16, max_neurons 985 (134.6 cells/neuron -> ~0.265 M cells).
  MDL  (IDS-72) ids_cut_method=info_gain_mdl: no global budget, Fayyad-Irani MDLP decides each feature's
       cuts -> 1200 bits after dedupe (control 287). NEEDS the info_gain_mdl encoder deployed first.
  E64  (IDS-72 control, optional) ids_n_bits 16 -> 64: cut E at a 1104-bit budget (1103 after dedupe),
       separates "more input bits" from "MDLP placement" if MDL moves.

Every body is dry-constructed through the worker's own IdsEvaluatorFactory.load (real unswt data, TRAIN-fit
encoder) and its encoded width asserted against the pre-check value, so a combination the encoder refuses,
or a silent encoding drift, stops the script before anything is POSTed. Run with PYTHONPATH=<tree the worker
will run>/src.

  --arms C,D24,MDL  (required)   --write-bodies | --dry-run | (no flag = POST + verify + queue)"""
import argparse
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import queue_ids56_vote_share_ab as Q  # noqa: E402  (DB, API, EXPS, call, check)
from queue_ids60_unswt_ce20 import diff  # noqa: E402

CONTROLS = {20421: 7600, 20422: 7601, 20423: 7602}
CONTROL_PREFIX = "IDS73-unswt-BINARY-2m-max250-xo_blend_neuron-16bWb-rate0.25-ms01-encon-cutE"
WORKER_ABI = 29
CONTROL_BITS = 287
# arm -> (Plane story, params changed vs the control, name, encoded width after dedupe on unswt TRAIN)
ARMS = {
	"C": ("IDS-64", {"ids_feature_ranking": "train_only"},
		"IDS64-unswt-BINARY-2m-max250-xo_blend_neuron-16bWb-rate0.25-ms01-encon-cutE-rktrain", 318),
	"D24": ("IDS-65", {"max_bits": 24, "max_neurons": 455},
		"IDS65-unswt-BINARY-2m-max455-xo_blend_neuron-16bWb-rate0.25-ms01-encon-cutE-b24", CONTROL_BITS),
	"D16": ("IDS-65", {"max_bits": 16, "max_neurons": 985},
		"IDS65-unswt-BINARY-2m-max985-xo_blend_neuron-16bWb-rate0.25-ms01-encon-cutE-b16", CONTROL_BITS),
	"MDL": ("IDS-72", {"ids_cut_method": "info_gain_mdl"},
		"IDS72-unswt-BINARY-2m-max250-xo_blend_neuron-16bWb-rate0.25-ms01-encon-cutMDL", 1200),
	"E64": ("IDS-72", {"ids_n_bits": 64},
		"IDS72-unswt-BINARY-2m-max250-xo_blend_neuron-64bWb-rate0.25-ms01-encon-cutE", 1103),
}
WHY = {
	"C": "feature list ranked on this split's TRAIN only (12/20 shared with legacy, no categoricals)",
	"D24": "width vs count at the control's cell budget: 24 b x ~910 neurons ~ 0.266 M cells",
	"D16": "width vs count at the control's cell budget: 16 b x ~1970 neurons ~ 0.265 M cells",
	"MDL": "MDLP-stopped info-gain cuts, data-chosen input width (1200 b vs 287)",
	"E64": "cut E at a 64-bit quantile budget (1103 b), the more-bits control for MDL",
}
# Measured trained cells per neuron (evolved-prefix of the 7600-7602 genomes, mean of 3; .scratch pre-check
# 22/11/2026) — used only to print the predicted D operating point.
CELLS_PER_NEURON = {34: 534.0, 24: 292.7, 20: 208.1, 16: 134.6}
LIVE_CUT_METHOD = Path("/Users/lacg/wnn/src/wnn/representations/cut_method.py")
BODIES = ROOT / "docs" / "ids_levers_cdm_flows.json"


def control(db, seed: int) -> tuple[dict, str]:
	"""The production unswt control for one seed, asserted to be the adopted IDS-73 Eon recipe."""
	name, cfg = db.execute("select name, config_json from flows where id=?", (CONTROLS[seed],)).fetchone()
	cfg = json.loads(cfg)
	p = cfg["params"]
	assert name == f"{CONTROL_PREFIX}-r{seed}" and p["seed"] == seed, name
	assert p["ids_split"] == "temporal_3way" and p["ids_k_folds"] == 5 and p["ids_kfold_per_gen"] == 5
	assert p["max_bits"] == 34 and p["max_neurons"] == 250 and p["ids_n_bits"] == 16
	assert p["ids_cut_method"] == "info_gain_mc" and p["ids_encoding_dedupe"] == "on"
	assert p["ids_feature_ranking"] == "legacy" and p["ids_feature_selection"] == "top20"
	return p, cfg["template"]


def dry_construct(params: dict, want_bits: int) -> None:
	"""Load + encode unswt exactly as the worker does (TRAIN-fit encoder) and check the encoded width."""
	from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
	from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
	ds = IdsEvaluatorFactory(log=lambda m: None).load(IdsEvaluatorSettings.from_params(params))
	got = ds.X_train.total_bits
	if got != want_bits or ds.encoder.dropped_features_:
		raise SystemExit(f"seed {params['seed']}: encoded {got} bits (want {want_bits}), "
			f"dropped {ds.encoder.dropped_features_}")


def not_a_repeat(db, params: dict) -> None:
	"""Never repeat a recipe on the same seed: no live flow may already carry these exact params."""
	for fid, cfg in db.execute("select id, config_json from flows where status != 'cancelled'"):
		if json.loads(cfg).get("params") == params:
			raise SystemExit(f"flow {fid} already runs this exact recipe on seed {params['seed']}")


def mdl_deployed() -> None:
	"""info_gain_mdl is a Python encoder change: refuse to queue it before the live tree has it."""
	if "INFO_GAIN_MDL" not in LIVE_CUT_METHOD.read_text():
		raise SystemExit("MDL: the live tree has no info_gain_mdl yet (merge + worker restart at idle first)")


def body(db, arm: str, seed: int) -> dict:
	story, change, prefix, want_bits = ARMS[arm]
	base, template = control(db, seed)
	params = {**base, **change}
	assert sorted(diff(base, params)) == sorted(change), arm
	Q.check(f"{arm}-r{seed}", params)
	not_a_repeat(db, params)
	dry_construct(params, want_bits)
	desc = (f"{story} arm {arm} (unswt screen, 22/11/2026): {WHY[arm]}. Clone of production {CONTROLS[seed]} "
		f"(IDS-73 Eon, r{seed}) with {json.dumps(change)}; paired control = {CONTROLS[seed]}. Worker ABI {WORKER_ABI}.")
	return {"name": f"{prefix}-r{seed}", "description": desc,
		"config": {"template": template, "params": params}, "experiments": Q.EXPS}


def print_operating_point(arm: str) -> None:
	change = ARMS[arm][1]
	if "max_neurons" in change:
		b, n = change["max_bits"], 2 * change["max_neurons"]
		print(f"  {arm}: predicted {n} neurons x {b} b x {CELLS_PER_NEURON[b]} = {n * CELLS_PER_NEURON[b] / 1e6:.3f} M cells "
			f"(control 0.265 M); realized cells must land within +-10% for an equal-budget read")


def post_once(b: dict) -> dict:
	"""POST without retry: a retried POST whose first attempt landed would create a duplicate flow."""
	req = urllib.request.Request(Q.API, method="POST", headers={"Content-Type": "application/json"},
		data=json.dumps(b).encode())
	return json.load(urllib.request.urlopen(req, context=Q.CTX, timeout=120))


def verify(db, flow_id: int, b: dict) -> None:
	"""Rule 2: the flow has its experiments and stored every param we sent."""
	n = db.execute("select count(*) from experiments where flow_id=?", (flow_id,)).fetchone()[0]
	if n != len(b["experiments"]):
		raise SystemExit(f"flow {flow_id}: {n} experiments, expected {len(b['experiments'])}; left pending")
	stored = json.loads(db.execute("select config_json from flows where id=?", (flow_id,)).fetchone()[0])["params"]
	wrong = {k: [v, stored.get(k)] for k, v in b["config"]["params"].items() if stored.get(k) != v}
	if wrong:
		raise SystemExit(f"flow {flow_id}: stored params differ from the body: {wrong}; left pending")


def queue(db, planned: list[dict]) -> None:
	have = {r[0] for r in db.execute("select name from flows where status != 'cancelled'")}
	for b in planned:
		if b["name"] in have:
			print("exists, skipped", b["name"])
			continue
		f = post_once(b)
		verify(db, f["id"], b)
		Q.call("PATCH", f"{Q.API}/{f['id']}", {"status": "queued"})
		print(f["id"], b["name"], "verified + queued")


def main(mode: str, arms: list[str]) -> None:
	import ram_accelerator
	if int(ram_accelerator.ABI_VERSION) != WORKER_ABI:
		raise SystemExit(f"installed ABI {ram_accelerator.ABI_VERSION} != {WORKER_ABI}")
	unknown = [a for a in arms if a not in ARMS]
	if unknown:
		raise SystemExit(f"unknown arms {unknown}; choose from {list(ARMS)}")
	if "MDL" in arms and mode == "queue":
		mdl_deployed()
	db = sqlite3.connect(Q.DB, uri=True)
	planned = [body(db, arm, seed) for arm in arms for seed in CONTROLS]  # arm-major: one arm's seeds together
	for arm in arms:
		print_operating_point(arm)
	if mode == "write":
		BODIES.write_text(json.dumps(planned, indent=1) + "\n")
		print("wrote", BODIES, len(planned), "bodies")
		return
	saved = json.loads(BODIES.read_text())
	if any(b not in saved for b in planned):
		raise SystemExit(f"{BODIES} lacks (or differs from) a body re-derived from the DB; refusing")
	if mode == "dry":
		for b in planned:
			print("[dry]", b["name"], json.dumps(diff(json.loads(sqlite3.connect(Q.DB, uri=True).execute(
				"select config_json from flows where id=?", (CONTROLS[b["config"]["params"]["seed"]],)).fetchone()[0])["params"],
				b["config"]["params"])), len(b["experiments"]), "experiments")
		return
	queue(db, planned)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--arms", required=True, help="comma list from " + ",".join(ARMS))
	g = ap.add_mutually_exclusive_group()
	g.add_argument("--write-bodies", action="store_true")
	g.add_argument("--dry-run", action="store_true")
	a = ap.parse_args()
	main("write" if a.write_bodies else "dry" if a.dry_run else "queue", a.arms.split(","))
