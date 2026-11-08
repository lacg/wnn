"""IDS-52 dedupe A/B, stage 1 (docs/ids52_53_ab_design.md §5; rule pre-registered 377df937).

Queues, after the IDS-48c cicids calib pairs (7380-7389, whose calibfull flows are the cicids dedupe-OFF control):
  1a unswt  BINARY bn max250 (base 7125): seeds 20419..20438, off then on      (40 flows)
  1b round k=1..8, seed s=20418+k:
       k<=5 cicids QUAD (base 7343): dedupe ON, calibfull, train_only        (5 flows)
       unswr BINARY bc max250 (base 7248): off then on, legacy ranking        (16 flows)
Both new params are set explicitly on every flow. Idempotent by name.
"""
import json, sqlite3, ssl, sys, time, urllib.request

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
API = "https://127.0.0.1:3000/api/flows"
CTX = ssl._create_unverified_context()
EXPS = [{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
        {"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"}]
DESC = "IDS-52 dedupe A/B stage 1 (docs/ids52_53_ab_design.md; non-inferiority pre-registered 377df937). Clone of {b}."


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


def base_cfg(db, fid):
	return json.loads(db.execute("select config_json from flows where id=?", (fid,)).fetchone()[0])


def plan():
	out = []
	for s in range(20419, 20439):
		for enc in ("off", "on"):
			out.append((7125, f"IDS52-unswt-BINARY-lattice-2m-max250-xo_blend_neuron-16bWb-rate0.25-rklegacy-enc{enc}-r{s}",
			            {"seed": s, "ids_encoding_dedupe": enc, "ids_feature_ranking": "legacy"}))
	for k in range(1, 9):
		s = 20418 + k
		if k <= 5:
			out.append((7343, f"IDS52-cicids-CE20-zs-QUAD-1m-rktrain-encon-calibfull-r{s}",
			            {"seed": s, "ids_encoding_dedupe": "on", "ids_feature_ranking": "train_only"}))
		for enc in ("off", "on"):
			out.append((7248, f"IDS52-unswr-WbCTRL-desir-BINARY-lattice-2m-max250-xo_blend_cluster-rate0.25-rklegacy-enc{enc}-r{s}",
			            {"seed": s, "ids_encoding_dedupe": enc, "ids_feature_ranking": "legacy"}))
	return out


def main():
	db = sqlite3.connect(DB, uri=True)
	have = {r[0] for r in db.execute("select name from flows")}
	dry = "--dry-run" in sys.argv
	for b, name, diff in plan():
		if name in have:
			continue
		cfg = base_cfg(db, b)
		p = dict(cfg["params"])
		p.update(diff)
		if dry:
			print("DRY", b, name, diff)
			continue
		f = call("POST", API, {"name": name, "description": DESC.format(b=b),
		                       "config": {"template": cfg["template"], "params": p}, "experiments": EXPS})
		call("PATCH", f"{API}/{f['id']}", {"status": "queued"})
		print(f["id"], name)


if __name__ == "__main__":
	main()
