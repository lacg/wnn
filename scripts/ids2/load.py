import sqlite3,json,re
from collections import defaultdict
DB="file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
MODES=["train_cal","fixed_05","platt","beta","empirical","empirical_cumulative","val_cal"]
GT=["best_f1","best_fpr","best_acc","best_ce","best_fitness"]
RE=re.compile(r"^IDSXD2?-(cicids|ciciot)-quad-96b-([A-Za-z0-9]+-?[A-Z]*)-r(\d+)(-w64fix)?$")
FIX="2026-08-30T02:27"
SQL="""SELECT f.id,f.name,f.started_at,e.id eid,e.phase_type,vs.genome_type,vs.genome_hash,vs.threshold_metadata
FROM validation_summaries vs JOIN flows f ON f.id=vs.flow_id JOIN experiments e ON e.id=vs.experiment_id
WHERE (f.name LIKE 'IDSXD-cicids-%' OR f.name LIKE 'IDSXD-ciciot-%' OR f.name LIKE 'IDSXD2-ciciot-%')
AND f.status='completed' AND vs.validation_point='final'"""
GENS="""SELECT f.name, e.phase_type, COUNT(i.id), MAX(i.iteration_num) FROM flows f JOIN experiments e ON e.flow_id=f.id
LEFT JOIN iterations i ON i.experiment_id=e.id WHERE (f.name LIKE 'IDSXD-cicids-%' OR f.name LIKE 'IDSXD-ciciot-%' OR f.name LIKE 'IDSXD2-ciciot-%')
AND f.status='completed' AND e.phase_type='ga_neurons' GROUP BY e.id"""
def load():
	con=sqlite3.connect(DB,uri=True)
	D=defaultdict(dict)  # D[(ds,arm,seed,phase,gt,mode)] = (f1,fpr,acc) ; runs meta
	meta={}
	for fid,name,st,eid,ph,gt,gh,tm in con.execute(SQL):
		m=RE.match(name); ds,arm,seed,fix=m.group(1),m.group(2),int(m.group(3)),bool(m.group(4))
		era="post" if st>=FIX else "pre"
		# ciciot: keep only post-fix runs (w64fix replaces pre)
		if ds=="ciciot" and era=="pre": continue
		meta[(ds,arm,seed)]=(fid,name,era)
		p="GS" if ph=="grid_search" else "GA"
		t=json.loads(tm)
		for md in MODES:
			v=t.get(md)
			if isinstance(v,dict) and v.get("f1") is not None:
				D[(ds,arm,seed,p,gt,md)]=(v["f1"]*100,v["fpr"]*100,v["acc"]*100,gh)
	gens={}
	for name,ph,c,mx in con.execute(GENS):
		m=RE.match(name); ds,arm,seed=m.group(1),m.group(2),int(m.group(3))
		if ds=="ciciot" and (ds,arm,seed) in meta and meta[(ds,arm,seed)][1]!=name: continue
		gens[(ds,arm,seed)]=c
	return D,meta,gens
