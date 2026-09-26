import json, re, sqlite3, statistics as st, sys, pickle
from datetime import datetime
DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
MODES = ["train_cal","fixed_05","platt","beta","empirical","empirical_cumulative","val_cal"]
SQL = """
SELECT f.id fid, f.name, f.started_at, f.completed_at, f.config_json cj,
       e.phase_type ph, vs.genome_type gt, vs.genome_hash gh, vs.experiment_id eid, vs.threshold_metadata tm
FROM validation_summaries vs
JOIN flows f ON f.id = vs.flow_id
JOIN experiments e ON e.id = vs.experiment_id
WHERE f.status = 'completed' AND vs.validation_point = 'final'
  AND json_extract(f.config_json,'$.params.ids_dataset') IN
      ('cicids2017','ciciot2023','ciciot2023_canonical','ciciot2023_full','ciciot2023_neto_full','ciciot2023_neto_subsample')
"""
con = sqlite3.connect(DB, uri=True); con.row_factory = sqlite3.Row
def ts(s):
	s = s.replace("Z","+00:00")
	return datetime.fromisoformat(s)
def shape(gh, eid):
	r = con.execute("SELECT tiers_json FROM genomes WHERE experiment_id=? AND (genome_hash=? OR config_hash=?) LIMIT 1", (eid, gh, gh)).fetchone()
	if not r: return None
	t = json.loads(r[0])
	if isinstance(t, dict):
		b = t.get("bits_per_neuron") or []
		return (len(b), round(st.mean(b),1) if b else 0, max(b) if b else 0)
	if isinstance(t, list) and t and isinstance(t[0], dict):
		n = sum(x.get("neurons",0)*x.get("clusters",1) for x in t); bb = [x.get("bits",0) for x in t]
		return (n, max(bb), max(bb))
	return None
rows = []
for r in con.execute(SQL):
	p = json.loads(r["cj"]).get("params", {})
	try: tm = json.loads(r["tm"]) if r["tm"] else {}
	except Exception: tm = {}
	dur = (ts(r["completed_at"]) - ts(r["started_at"])).total_seconds()/3600 if r["started_at"] and r["completed_at"] else None
	for md in MODES:
		v = tm.get(md)
		if isinstance(v, dict) and v.get("f1") is not None:
			rows.append(dict(fid=r["fid"], name=r["name"], start=r["started_at"], dur=dur, ds=p.get("ids_dataset"), split=p.get("ids_split"),
				mem=p.get("memory_mode","QUAD"), nb=p.get("ids_n_bits"), maxb=p.get("max_bits"), w=(p.get("fitness_weight_ce"),p.get("fitness_weight_acc"),p.get("fitness_weight_f1"),p.get("fitness_weight_fpr")),
				agg=p.get("fitness_aggregation","whm"), feat=p.get("ids_feature_selection"), flip=p.get("flip_labels"), oi=p.get("wnn_order_independent_train"),
				ph="GS" if r["ph"]=="grid_search" else ("GA" if (r["ph"] or "").startswith("ga") else r["ph"]), gt=r["gt"], gh=r["gh"], eid=r["eid"], md=md,
				f1=v["f1"]*100, fpr=(v.get("fpr") or 0)*100, acc=(v.get("acc") or 0)*100))
pickle.dump(rows, open(sys.argv[1],"wb"))
print(len(rows))
