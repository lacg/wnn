"""Shared cohort selection for IDS-24 / IDS-25 (GA plateau spike IDS-23). Read-only on the DB.

Cohorts (QUAD, Protocol v2 _3way, completed, own-flow rows only):
  SP100  : SP100-{cicids,ciciot,unswr,unswt}-quad-*   (production config, one arm per dataset)
  SWEEP  : IDSXD-{ds}-quad-*  (B34-CTRL excluded: min_bits=34 pins bits, pre-fix hybrids r20401-04)
           IDSAGG-/IDSAGGX-{cicids,ciciot}-quad-*
Excluded: zINVALID*, QSR, SP-*-bin (same seeds/config as SP100 -> would double count),
validation rows listed as inherited in the 10/10 audit pickle (inh4.pkl) unless since
re-validated (cache_key NOT NULL = re-scored on the fixed cache = own).
"""
import sqlite3, json, re, pickle
DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
INH = "/private/tmp/claude-501/-Users-lacg-wnn/ca7e9679-f517-4eb2-ba35-ad1a4d41366b/scratchpad/inh4.pkl"
MODES = ["train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal"]
GT = ["best_f1", "best_fpr", "best_acc", "best_ce", "best_fitness"]
DSETS = ["unswr", "unswt", "cicids", "ciciot"]
RX = [
	("SP100", re.compile(r"^SP100-(cicids|ciciot|unswr|unswt)-quad-\d+bW[abc]-r(\d+)$")),
	("SWEEP", re.compile(r"^IDSXD-(cicids|ciciot|unswr|unswt)-quad-\d+b-(?!B34-CTRL)(.+)-r(\d+)$")),
	("SWEEP", re.compile(r"^IDSAGGX?-(cicids|ciciot)-quad-96b-(.+)-r(\d+)$")),
]


def connect():
	return sqlite3.connect(DB, uri=True)


def inherited_ids():
	_, inh = pickle.load(open(INH, "rb"))
	return {r["id"] for r in inh}


def select_flows(con):
	out = {}
	for fid, name, status, cfg in con.execute("SELECT id,name,status,config_json FROM flows WHERE status='completed'"):
		for fam, rx in RX:
			m = rx.match(name)
			if not m:
				continue
			p = json.loads(cfg).get("params", {})
			assert p.get("ids_split", "").endswith("_3way"), name
			assert p.get("memory_mode") in (None, "QUAD_WEIGHTED"), name
			out[fid] = dict(name=name, fam=fam, ds=m.group(1), seed=int(m.group(m.lastindex)), params=p)
	return out


def load_cells(con, flows):
	"""D[(fid, phase, gt)] = {mode: (f1, fpr, acc) in %}, plus genome hash/eid; own rows only."""
	inh = inherited_ids()
	D, GH = {}, {}
	ids = ",".join(map(str, flows))
	q = f"""SELECT vs.id, vs.flow_id, e.id, e.phase_type, vs.genome_type, vs.genome_hash, vs.threshold_metadata, vs.cache_key
		FROM validation_summaries vs JOIN experiments e ON e.id=vs.experiment_id
		WHERE vs.validation_point='final' AND vs.flow_id IN ({ids})"""
	nin = 0
	for vid, fid, eid, ph, gt, gh, tm, ck in con.execute(q):
		if vid in inh and ck is None:
			nin += 1
			continue
		p = {"grid_search": "GS", "ga_neurons": "GA"}.get(ph)
		if p is None or not tm:
			continue
		t = json.loads(tm)
		cells = {}
		for md in MODES:
			v = t.get(md)
			if isinstance(v, dict) and v.get("f1") is not None:
				cells[md] = (v["f1"] * 100, v["fpr"] * 100, v["acc"] * 100)
		D[(fid, p, gt)] = cells
		GH[(fid, p, gt)] = (eid, gh)
	return D, GH, nin


def genome_size(con, eid, gh):
	r = con.execute("SELECT tiers_json FROM genomes WHERE experiment_id=? AND (genome_hash=? OR config_hash=?) LIMIT 1", (eid, gh, gh)).fetchone()
	if not r:
		return None
	t = json.loads(r[0])
	b = t.get("bits_per_neuron") if isinstance(t, dict) else None
	if not b:
		return None
	return len(b), sum(b) / len(b)
