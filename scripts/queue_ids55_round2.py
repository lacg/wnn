"""IDS-55 round 2 (Luiz 08/11, after the unswt A/B: family -0.83 pp F1, ~48% fewer cells; docs/ids55_ab_readout.md).
Seeds 20419-20423, worker ABI 27, receptive fields on families_v2. Two A/Bs, every param that matters set explicitly:

 (a) EQUAL MEMORY, unswt BINARY: clone 7125 like the round-1 A/B, receptive fields OFF, max_neurons_total=262
     (off used ~428 cells/neuron -> ~112K cells = the family arm's 111.8K). Pairs with the round-1 family flows
     7452-7460 (same seeds/ABI; those ran families_v1 — the v2 merges add nothing at dedupe off except
     protocol->context, so v1 is the family arm of record). Question: is family better F1 PER CELL?
 (b) CIC-IoT QUAD: clone 7344 (IDS-47 ciciot QUAD 1m), off vs family cross 0.2 (96-bit neurons -> 19 out-of-family).
     The probe's coverage-limited dataset. Both arms fresh on ABI 27 (the IDS-47 rows ran an older ABI: never pair).

 (c) families_v2 (Luiz 09/11, after docs/ids55_why_family_loses.md): unswt family 0.2 on the v2 map (protocol->context),
     isolates the protocol-starvation cause; harness predicts the gap shrinks to ~-0.45 pp. Cross 0.4 deferred.

Usage: python scripts/queue_ids55_round2.py [--dry-run]"""
import json, sqlite3, ssl, sys, time, urllib.request

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
API = "https://127.0.0.1:3000/api/flows"
CTX = ssl._create_unverified_context()
EXPS = [{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
        {"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"}]
SEEDS = range(20419, 20424)
OFF = {"ids_receptive_fields": "off"}
FAMILY = {"ids_receptive_fields": "family", "ids_rf_cross": 0.2}


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


def plan(db) -> list[tuple[str, int, dict, str]]:
	"""(name, base flow, param overrides, description) for every flow of round 2, seed-major."""
	common = {"ids_encoding_dedupe": "off", "ids_feature_ranking": "legacy"}
	unswt = "IDS55-unswt-BINARY-lattice-2m-max250-xo_blend_neuron-16bWb-rate0.25-encoff"
	ciciot = "IDS55-ciciot-QUAD-1m-encoff"
	out = []
	for s in SEEDS:
		out.append((f"{unswt}-rfoff-tot262-r{s}", 7125, {**common, **OFF, "max_neurons_total": 262},
			"IDS-55 round 2a EQUAL MEMORY (Luiz 08/11): rf off at ~112K cells, pairs with round-1 family flow, ABI 27."))
		out.append((f"{unswt}-rffam02v2-r{s}", 7125, {**common, **FAMILY},
			"IDS-55 round 2c (Luiz 09/11): family 0.2 on families_v2 (protocol->context), pairs with round-1 v1 family + off, ABI 27."))
		for arm, extra in (("rfoff", OFF), ("rffam02", FAMILY)):
			out.append((f"{ciciot}-{arm}-r{s}", 7344, {**common, **extra, "memory_mode": "QUAD_WEIGHTED"},
				"IDS-55 round 2b CIC-IoT QUAD (Luiz 08/11): receptive fields off vs family 0.2, families_v2, ABI 27."))
	return out


def main(dry_run: bool):
	db = sqlite3.connect(DB, uri=True)
	have = {r[0] for r in db.execute("select name from flows")}
	for name, base, extra, desc in plan(db):
		if name in have:
			continue
		cfg = json.loads(db.execute("select config_json from flows where id=?", (base,)).fetchone()[0])
		p = {**cfg["params"], **extra, "seed": int(name.rsplit("-r", 1)[1])}
		if dry_run:
			print(f"[dry] {name}  base {base}  " + " ".join(f"{k}={p.get(k)}" for k in
				("seed", "memory_mode", "ids_receptive_fields", "ids_rf_cross", "max_neurons", "max_neurons_total", "max_bits")))
			continue
		f = call("POST", API, {"name": name, "description": desc,
		                       "config": {"template": cfg["template"], "params": p}, "experiments": EXPS})
		call("PATCH", f"{API}/{f['id']}", {"status": "queued"})
		print(f["id"], name)


if __name__ == "__main__":
	main("--dry-run" in sys.argv)
