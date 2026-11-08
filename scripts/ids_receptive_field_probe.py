"""Offline probe: do family-restricted ("receptive field") RAM neurons generalize
better than spread neurons on IDS data? (IDS analogue of CTRL-45.)

Read-only w.r.t. the worker/DB. Two subcommands:

  prep  <cfg> <dedupe 0|1>   load via the production loader + encoder, subsample
                              TRAIN / VAL (never TEST), save a column-major bool
                              cache + feature bit ranges to --out.
  probe <cfg> <dedupe 0|1>   draw random neurons (spread / family / family+cross),
                              RAM-train them on TRAIN (address = the neuron's bits,
                              per-neuron 25% row sample like neuron_sample_rate),
                              read VAL, report coverage / per-neuron accuracy /
                              distinct addresses / ensemble F1+FPR.

RAM semantics (faithful, offline numpy — a measurement harness, not a substrate):
  address   = the neuron's bits, MSB-first (bit 0 of the connection list = MSB);
              > 64 bits: two u64 words hashed to one u64 key (collisions ~1e-9).
  QUAD      = order-independent tally: net = sum(+w attack, -w benign) with the
              balanced class weight w_c = N/(2 N_c); binned exactly as
              neuron_memory::oi_bin_to_cell (obs, round(net)) -> cell_to_weight
              {FALSE .0, WEAK_FALSE .25, WEAK_TRUE .75, TRUE 1.}; unseen = WEAK_FALSE.
  BINARY    = 2-discriminator WiSARD, lattice/one-shot Set: disc c's neuron cell
              = 1 iff any class-c train row hit the address; response = #ones;
              argmax (ties -> benign). Disjoint neuron sets per discriminator.
  coverage  = val row's address was written in TRAIN (EMPTY = abstain).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CFGS = {
	"unswr": dict(ds="unsw-nb15", split="random_3way", n_bits=64, ranking="legacy", width=34),
	"unswt": dict(ds="unsw-nb15", split="temporal_3way", n_bits=16, ranking="legacy", width=34),
	"cicids": dict(ds="cicids2017", split="random_3way", n_bits=96, ranking="train_only", width=34),
	"ciciot": dict(ds="ciciot2023_neto_subsample", split="random_3way", n_bits=96, ranking="train_only", width=96),
}

FAMILIES = {
	"unsw": {
		"ttl_state": ["sttl", "dttl", "ct_state_ttl", "state"],
		"handshake": ["tcprtt", "synack", "ackdat", "swin"],
		"volume": ["sbytes", "dbytes", "smeansz", "dmeansz", "Dpkts", "Spkts"],
		"rate": ["Sload", "Dload", "sload", "dload", "rate"],
		"timing": ["dur", "Sintpkt", "Dintpkt", "Sjit"],
		"context": ["ct_srv_src", "ct_srv_dst", "ct_dst_src_ltm", "ct_dst_sport_ltm", "ct_src_dport_ltm"],
		"protocol": ["proto", "service"],
	},
	"cicids": {
		"size": ["Average Packet Size", "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "Max Packet Length"],
		"bwd_size": ["Bwd Packet Length Std", "Bwd Packet Length Mean", "Avg Bwd Segment Size", "Bwd Packet Length Max",
			"Subflow Bwd Bytes", "Total Length of Bwd Packets"],
		"fwd_size": ["Subflow Fwd Bytes", "Total Length of Fwd Packets"],
		"header_window": ["Init_Win_bytes_forward", "Init_Win_bytes_backward", "Bwd Header Length", "Fwd Header Length"],
		"timing": ["Flow IAT Max", "Flow IAT Std"],
		"port": ["Destination Port"],
	},
	"ciciot": {
		"flags": ["syn_count", "rst_count", "urg_count"],
		"rate": ["Rate", "Srate"],
		"size": ["Tot size", "Tot sum", "AVG", "Min", "Max", "Variance", "Radius", "Magnitue", "Header_Length"],
		"timing": ["IAT", "flow_duration", "Duration"],
		"context": ["Number", "Weight", "HTTPS"],
	},
}

MAX_TRAIN = 400_000
MAX_VAL = 100_000
SAMPLE_RATE = 0.25


def _strat_idx(y, cap, rng):
	if len(y) <= cap:
		return np.arange(len(y))
	return np.sort(rng.choice(len(y), cap, replace=False))


def _rows_bool(X, idx):
	"""bits x rows bool; unpacks the encoder's little-endian packed rows."""
	r = np.asarray(X[idx])
	if r.dtype != bool:
		r = np.unpackbits(r, axis=1, bitorder="little", count=X.total_bits).astype(bool)
	assert r.shape[1] == X.total_bits, (r.shape, X.total_bits)
	return np.ascontiguousarray(r.T)


def prep(cfg_key, dedupe, out):
	from wnn.ids import load_ids_dataset
	c = CFGS[cfg_key]
	t0 = time.time()
	d = load_ids_dataset(c["ds"], n_bits=c["n_bits"], split=c["split"], feature_selection="top20",
		encoding_dedupe=bool(dedupe), feature_ranking=c["ranking"])
	rng = np.random.default_rng(12345)
	ytr, yva = np.asarray(d.y_train_binary), np.asarray(d.y_val_binary)
	itr, iva = _strat_idx(ytr, MAX_TRAIN, rng), _strat_idx(yva, MAX_VAL, rng)
	Xtr = _rows_bool(d.X_train, itr)
	Xva = _rows_bool(d.X_val, iva)
	ranges = d.encoder.feature_bit_ranges()
	np.savez(out, Xtr=Xtr, ytr=ytr[itr].astype(np.int8), Xva=Xva, yva=yva[iva].astype(np.int8),
		meta=json.dumps(dict(ranges=ranges, n_train_full=int(len(ytr)), n_val_full=int(len(yva)),
			dropped=list(getattr(d.encoder, "dropped_features_", [])))))
	print(f"[prep] {cfg_key} dedupe={dedupe} bits={Xtr.shape[0]} train={Xtr.shape[1]}/{len(ytr)} "
		f"val={Xva.shape[1]}/{len(yva)} attack_tr={ytr.mean():.3f} attack_va={yva.mean():.3f} {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------ probe

def family_bits(ranges, fam_key):
	fams = FAMILIES[fam_key]
	out, assigned = {}, set()
	for name, feats in fams.items():
		bits = [b for f in feats if f in ranges for b in range(*ranges[f])]
		assigned |= {f for f in feats if f in ranges}
		if bits:
			out[name] = np.array(bits)
	rest = [f for f in ranges if f not in assigned]
	if rest:
		out["other"] = np.array([b for f in rest for b in range(*ranges[f])])
	return out, rest


def draw(scheme, cross, width, n_total, fams, rng):
	"""One neuron's connection list + its family name ('spread' for a)."""
	if scheme == "spread":
		return rng.choice(n_total, width, replace=False), "spread"
	names = list(fams)
	sizes = np.array([len(fams[n]) for n in names], dtype=float)
	name = names[rng.choice(len(names), p=sizes / sizes.sum())]
	fb = fams[name]
	n_cross = int(round(cross * width))
	n_in = min(width - n_cross, len(fb))
	inside = rng.choice(fb, n_in, replace=False)
	outside_pool = np.setdiff1d(np.arange(n_total), fb, assume_unique=False)
	n_out = min(width - n_in, len(outside_pool))  # cross budget (+ fill if the family is narrower than width)
	outside = rng.choice(outside_pool, n_out, replace=False) if n_out else np.array([], dtype=int)
	return np.concatenate([inside, outside]), name


def addresses(Xt, conn, cols=None):
	"""MSB-first address of each row (Xt is bits x rows)."""
	sub = Xt[conn] if cols is None else Xt[conn][:, cols]
	k = len(conn)
	words = []
	for s in range(0, k, 64):
		chunk = sub[s:s + 64]
		w = np.zeros(chunk.shape[1], dtype=np.uint64)
		for row in chunk:
			w = (w << np.uint64(1)) | row.astype(np.uint64)
		words.append(w)
	key = words[0]
	for w in words[1:]:
		key = key * np.uint64(0x9E3779B97F4A7C15) ^ w
	return key


def oi_cell_weight(obs, net):
	"""neuron_memory::oi_bin_to_cell -> cell_to_weight (QUAD)."""
	netr = np.rint(net)
	w = np.full(obs.shape, 0.25)
	one = obs == 1
	w[one & (net > 0)] = 0.75
	two = obs >= 2
	w[two & (netr <= -1)] = 0.0
	w[two & (netr == 1)] = 0.75
	w[two & (netr >= 2)] = 1.0
	return w


def train_read(Xtr, ytr, Xva, conn, cw, rng):
	n = Xtr.shape[1]
	rows = np.sort(rng.choice(n, int(n * SAMPLE_RATE), replace=False))
	ka = addresses(Xtr, conn, rows)
	yr = ytr[rows]
	uk, inv = np.unique(ka, return_inverse=True)
	n1 = np.bincount(inv, weights=yr, minlength=len(uk))
	obs = np.bincount(inv, minlength=len(uk))
	n0 = obs - n1
	net = n1 * cw[1] - n0 * cw[0]
	cell = oi_cell_weight(obs, net)
	kv = addresses(Xva, conn)
	pos = np.searchsorted(uk, kv)
	pos[pos >= len(uk)] = 0
	hit = uk[pos] == kv
	q = np.where(hit, cell[pos], 0.25)                 # QUAD read (unseen = WEAK_FALSE)
	p1 = np.where(hit, (n1 * cw[1] / np.maximum(n1 * cw[1] + n0 * cw[0], 1e-12))[pos], np.nan)  # balanced tally
	b0 = hit & (n0[pos] > 0)
	b1 = hit & (n1[pos] > 0)
	return dict(hit=hit, q=q.astype(np.float32), p1=p1.astype(np.float32), b0=b0, b1=b1, n_addr=len(uk))


def neuron_stats(r, yva):
	hit = r["hit"]
	cov = hit.mean()
	cov_att = hit[yva == 1].mean()
	if hit.sum() == 0:
		return cov, np.nan, np.nan, np.nan, cov_att
	y = yva[hit]
	pred = r["p1"][hit] > 0.5
	acc = (pred == y).mean()
	tpr = pred[y == 1].mean() if (y == 1).any() else np.nan
	tnr = (~pred[y == 0]).mean() if (y == 0).any() else np.nan
	bacc = np.nanmean([tpr, tnr])
	sep = r["p1"][hit][y == 1].mean() - r["p1"][hit][y == 0].mean() if (y == 1).any() and (y == 0).any() else np.nan
	return cov, acc, bacc, sep, cov_att


def f1_fpr(pred, y):
	tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
	f1 = 2 * tp / max(2 * tp + fp + fn, 1)
	fpr = fp / max((y == 0).sum(), 1)
	return f1, fpr


def auc(score, y):
	from scipy.stats import rankdata
	r = rankdata(score)
	n1 = (y == 1).sum(); n0 = len(y) - n1
	return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def best_f1(score, y):
	order = np.argsort(-score, kind="stable")
	ys = y[order]
	tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
	# only cut between distinct scores
	s = score[order]
	cut = np.r_[s[1:] != s[:-1], True]
	tp, fp = tp[cut], fp[cut]
	fn = ys.sum() - tp
	f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
	i = int(np.argmax(f1))
	return f1[i], fp[i] / max((y == 0).sum(), 1)


def ensembles(pool, yva, n_quad, n_bin, n_rep, rng):
	Q = np.stack([r["q"] for r in pool]); B0 = np.stack([r["b0"] for r in pool]); B1 = np.stack([r["b1"] for r in pool])
	M = len(pool)
	out = []
	for _ in range(n_rep):
		idx = rng.choice(M, n_quad, replace=False)
		s = Q[idx].mean(0)
		qf1, qfpr = best_f1(s, yva)
		f05, fpr05 = f1_fpr(s > 0.5, yva)
		a = auc(s, yva)
		perm = rng.permutation(M)
		d0, d1 = perm[:n_bin], perm[n_bin:2 * n_bin]
		r0 = B0[d0].sum(0); r1 = B1[d1].sum(0)
		bf1, bfpr = f1_fpr(r1 > r0, yva)
		out.append((qf1, qfpr, f05, fpr05, a, bf1, bfpr))
	return np.array(out)


def probe(cfg_key, dedupe, cache, M, crosses, seeds, out_json):
	c = CFGS[cfg_key]
	z = np.load(cache)
	Xtr, ytr, Xva, yva = z["Xtr"], z["ytr"].astype(np.int64), z["Xva"], z["yva"].astype(np.int64)
	meta = json.loads(str(z["meta"]))
	ranges = {k: tuple(v) for k, v in meta["ranges"].items()}
	fam_key = "unsw" if cfg_key.startswith("unsw") else cfg_key
	fams, rest = family_bits(ranges, fam_key)
	n_total = Xtr.shape[0]
	width = c["width"]
	N, N1 = len(ytr), ytr.sum()
	cw = (N / (2 * (N - N1)), N / (2 * N1))
	print(f"[probe] {cfg_key} dedupe={dedupe} bits={n_total} width={width} train={N} val={len(yva)} "
		f"families={ {k: len(v) for k, v in fams.items()} } unassigned={rest}", flush=True)
	schemes = [("spread", 0.0)] + [("family", x) for x in crosses]
	res = dict(cfg=cfg_key, dedupe=dedupe, bits=n_total, width=width, families={k: int(len(v)) for k, v in fams.items()},
		unassigned=rest, n_train=int(N), n_val=int(len(yva)), attack_val=float(yva.mean()), schemes={})
	for scheme, cross in schemes:
		tag = scheme if scheme == "spread" else f"family_x{cross:.1f}"
		per_seed, fam_rows = [], {}
		for seed in seeds:
			rng = np.random.default_rng(seed * 1000 + int(cross * 10))
			t0 = time.time()
			pool, stats, naddr = [], [], []
			for _ in range(M):
				conn, fname = draw(scheme, cross, width, n_total, fams, rng)
				r = train_read(Xtr, ytr, Xva, conn, cw, rng)
				st = neuron_stats(r, yva)
				pool.append(dict(q=r["q"], b0=r["b0"], b1=r["b1"]))
				stats.append(st); naddr.append(r["n_addr"])
				fam_rows.setdefault(fname, []).append((*st, r["n_addr"]))
			S = np.array(stats, dtype=float)
			E = ensembles(pool, yva, n_quad=500, n_bin=min(250, M // 2), n_rep=5, rng=rng)
			per_seed.append(dict(cov=np.nanmean(S[:, 0]), cov_att=np.nanmean(S[:, 4]), acc=np.nanmean(S[:, 1]), bacc=np.nanmean(S[:, 2]),
				sep=np.nanmean(S[:, 3]), addr_med=float(np.median(naddr)), addr_mean=float(np.mean(naddr)),
				quad_valcal_f1=E[:, 0].mean(), quad_valcal_fpr=E[:, 1].mean(), quad_05_f1=E[:, 2].mean(),
				quad_05_fpr=E[:, 3].mean(), quad_auc=E[:, 4].mean(), bin_f1=E[:, 5].mean(), bin_fpr=E[:, 6].mean()))
			print(f"  {tag:12s} s{seed} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
				for k, v in per_seed[-1].items()) + f" ({time.time()-t0:.0f}s)", flush=True)
		fam_summary = {k: dict(n=len(v), cov=float(np.nanmean([x[0] for x in v])), acc=float(np.nanmean([x[1] for x in v])),
			bacc=float(np.nanmean([x[2] for x in v])), sep=float(np.nanmean([x[3] for x in v])),
			cov_att=float(np.nanmean([x[4] for x in v])), addr_med=float(np.median([x[5] for x in v]))) for k, v in fam_rows.items()}
		res["schemes"][tag] = dict(per_seed=per_seed, per_family=fam_summary)
	Path(out_json).write_text(json.dumps(res, indent=1, default=float))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("cmd", choices=["prep", "probe"])
	ap.add_argument("cfg", choices=list(CFGS))
	ap.add_argument("dedupe", type=int, choices=[0, 1])
	ap.add_argument("--dir", default="/Users/lacg/.claude/jobs/f3e0e76f/tmp")
	ap.add_argument("--neurons", type=int, default=1000)
	ap.add_argument("--crosses", default="0,0.2,0.4")
	ap.add_argument("--seeds", default="1,2,3")
	ap.add_argument("--width", type=int, default=None)
	a = ap.parse_args()
	cache = os.path.join(a.dir, f"rf_{a.cfg}_d{a.dedupe}.npz")
	if a.cmd == "prep":
		prep(a.cfg, a.dedupe, cache)
	else:
		if a.width:
			CFGS[a.cfg]["width"] = a.width
		out = os.path.join(a.dir, f"rf_{a.cfg}_d{a.dedupe}_w{CFGS[a.cfg]['width']}.json")
		probe(a.cfg, a.dedupe, cache, a.neurons, [float(x) for x in a.crosses.split(",")],
			[int(s) for s in a.seeds.split(",")], out)


if __name__ == "__main__":
	sys.exit(main())
