import json, re, sqlite3, statistics as st, collections
from datetime import datetime
con = sqlite3.connect("file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro", uri=True); con.row_factory=sqlite3.Row
SQL = """
SELECT f.id fid, f.name, f.started_at, f.completed_at, f.config_json cj, e.phase_type ph,
       vs.genome_type gt, vs.genome_hash gh, vs.experiment_id eid, vs.threshold_metadata tm
FROM validation_summaries vs JOIN flows f ON f.id=vs.flow_id JOIN experiments e ON e.id=vs.experiment_id
WHERE f.status='completed' AND vs.validation_point='final'
  AND json_extract(f.config_json,'$.params.ids_dataset') IN ('cicids2017','ciciot2023_neto_subsample')
  AND json_extract(f.config_json,'$.params.ids_split') LIKE '%_3way'
  AND replace(f.started_at,'Z','+00:00') >= '2026-08-22' AND f.name LIKE 'IDSXD-%' AND json_extract(f.config_json,'$.params.max_bits')<=64
"""
def ts(s): return datetime.fromisoformat(s.replace("Z","+00:00"))
def shape(gh,eid):
	r=con.execute("SELECT tiers_json FROM genomes WHERE experiment_id=? AND (genome_hash=? OR config_hash=?) LIMIT 1",(eid,gh,gh)).fetchone()
	if not r: return "n/a"
	t=json.loads(r[0]); b=t.get("bits_per_neuron") or [] if isinstance(t,dict) else []
	return f"{len(b)}n x {st.mean(b):.1f}b(max {max(b)})" if b else str(t)[:40]
R=[]; flows={}
for r in con.execute(SQL):
	p=json.loads(r["cj"])["params"]
	flows[r["fid"]]=(r["name"],p["ids_dataset"],r["started_at"])
	tm=json.loads(r["tm"]); v=tm.get("val_cal")
	if not isinstance(v,dict): continue
	R.append(dict(fid=r["fid"],name=r["name"],start=r["started_at"],dur=(ts(r["completed_at"])-ts(r["started_at"])).total_seconds()/3600,
		ds=p["ids_dataset"],p=p,ph="GS" if r["ph"]=="grid_search" else "GA",gt=r["gt"],gh=r["gh"],eid=r["eid"],f1=v["f1"]*100,fpr=v["fpr"]*100,acc=v["acc"]*100))
cnt=collections.Counter((d, n.split("-")[0], ("QSR" if "-qsr-" in n else "")+("abi13" if "abi13" in n else "")+("w64fix" if "w64fix" in n else "")) for n,d,s in flows.values())
for k,v in sorted(cnt.items()): print(k,v)
print("B34:", [ (n,s) for n,d,s in flows.values() if "B34" in n])
print("QSR pre:", [ (n,s) for n,d,s in flows.values() if json.loads(con.execute("select config_json from flows where name=?",(n,)).fetchone()[0])["params"].get("memory_mode")=="QSR" and s<"2026-09-19T15:40"])
def cfg(p): return f"{p.get('fitness_aggregation','whm')} f1/fpr/ce/acc={p['fitness_weight_f1']}/{p['fitness_weight_fpr']}/{p['fitness_weight_ce']}/{p['fitness_weight_acc']} {p.get('memory_mode','QUAD')} nb={p['ids_n_bits']} bits={p['min_bits']}-{p['max_bits']} n={p['min_neurons']}-{p['max_neurons']} seed={p['seed']} feat={p['ids_feature_selection']} pop={p['population_size']} gens={p['ga_generations']} pat={p['patience']} nsr={p['neuron_sample_rate']} OI={p.get('wnn_order_independent_train')}"
for ds in ("cicids2017","ciciot2023_neto_subsample"):
	X=sorted([r for r in R if r["ds"]==ds], key=lambda r:(-r["f1"],r["fpr"]))
	print(f"\n=== {ds} val_cal rows={len(X)}")
	seen=set();k=0
	for r in X:
		key=(r["fid"],r["ph"],r["gh"])
		if key in seen: continue
		seen.add(key);k+=1
		gts="/".join(sorted(x["gt"].replace("best_","") for x in X if (x["fid"],x["ph"],x["gh"])==key and abs(x["f1"]-r["f1"])<1e-9))
		print(f"{r['fid']} {r['name']} {r['ph']} {gts} {r['f1']:.2f}/{r['fpr']:.2f}/{r['acc']:.2f} {shape(r['gh'],r['eid'])} {r['dur']:.2f}h {r['start'][:16]} | {cfg(r['p'])}")
		if k>=5: break
	C=collections.defaultdict(list)
	for r in X:
		if r["ph"]!="GA" or r["gt"]!="best_f1": continue
		base=re.sub(r"-r\d+","",r["name"]).replace("IDSXD2-","IDSXD-")
		C[base].append(r)
	print("-- arms (GA best_f1 val_cal)")
	for b,rs in sorted(C.items(), key=lambda kv:-st.mean(x["f1"] for x in kv[1])):
		f=[x["f1"] for x in rs];q=[x["fpr"] for x in rs]
		sd=lambda a: st.stdev(a) if len(a)>1 else 0
		print(f"{b} n={len(rs)} seeds={sorted(x['p']['seed'] for x in rs)} F1 {st.mean(f):.2f}±{sd(f):.2f} FPR {st.mean(q):.2f}±{sd(q):.2f} dur {st.mean(x['dur'] for x in rs):.2f}h")
