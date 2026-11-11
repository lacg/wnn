"""CTRL-52: bit-flip (single-event-upset) fault injection on the deployed controller image.

Simulation only. Scores through the INSTALLED ram_controller wheel (ABI 49) — the
scorer is never reimplemented; this file only corrupts stored parameters in the
exact format the hardware stores them, decodes that corrupted image back into the
genome the scorer flies, and calls the scorer.

THE STORAGE MODEL (WNN) = the MCU export, mcu/README.md + scripts/export_controller_c.py:
  wnn_conn[N*bits]    uint8 (int16 if any index > 255) input index per address bit;
                      a negative index reads 0 (neuron_memory.rs address_over, C twin)
  wnn_keys[K]         uint32, each output neuron's sorted TRUE on-set, concatenated
  wnn_off[N+1]        uint32 start offset per neuron into wnn_keys
  wnn_thresholds[F*8] float32 thermometer cut-points
BINARY memory stores ONLY the written-TRUE set; every other address reads FALSE. So a
flipped key bit deletes a TRUE entry and creates one at the XORed address, and it can
break the sort order, hiding OTHER keys from the C binary search. The decode below
emulates mcu/wnn_cascade.c key_present() bit-for-bit (vectorised) to get the on-set the
firmware would actually see. Two approximations, both documented in the report:
  * a corrupted conn index >= input bits would read past frame[] in C (adjacent
    struct memory) — modelled as the constant-0 bit (-1), as for a negative index;
  * a corrupted wnn_off past the end of wnn_keys would read adjacent .rodata —
    modelled as clamped to K (fewer keys, never garbage keys).
Not in the universe: the firmware rate-PID gains and the CTBR/d-hat constants, which
the WNN cascade shares with the MPCOF cascade (see the `rate` mode for those).

Modes:
  wnn    --tag T --ks 0,1,4,... --draws 10 [--seeds N]   -> JSON of per-draw triples
  rate   --who wnn|mpcof --tag T --seeds N               -> every single bit of the
         12 float32 firmware rate-PID gains (384 bits), flipped one at a time
  mpcof  --tag T [--seeds N]                             -> the k=0 MPCOF>rate(h10) row
  report --inputs a.json ... --png P                     -> tables (markdown) + plot
"""
import argparse, json, re, shlex, sys, time
from pathlib import Path

import numpy as np

D = "/Users/lacg/wnn/logs/controller/ctrl48/"
STAGE = {"GRID": "stage0_grid", "NEURONS": "stage1_neurons", "MEMORY": "stage4_memory"}
KEYS = ("acc", "mean_attitude_error_deg", "mean_steady_error_deg")


# ---------------------------------------------------------------------------- subject
def load_subject(tag: str):
	"""(args, ec, spec, genome, thresholds, seedset, headline) exactly as the run flew."""
	from wnn.control import phased_ga as P
	from wnn.seeds import resolve_seed_set
	import ram_controller
	assert ram_controller.ABI_VERSION == 49, ram_controller.ABI_VERSION
	text = open(D + tag + ".out", errors="replace").read()
	argv = shlex.split([l for l in text.splitlines() if l.startswith("[argv]")][0][len("[argv]"):])
	head = re.findall(r"stage-select\] HEADLINE stage.*?genome=(\S+)", text)[-1]
	stage, idx = head.split("#")[0], int(head.split("#")[1])
	a = P.build_arg_parser().parse_args(argv)
	ec = P.episode_config_from_args(a); P._resolve_ctbr(a, ec); a._dhat_b = None
	a._conn_policy, a._conn_policy_min = P.conn_policy_from_name(getattr(a, "conn_policy", "spread"))
	ck = Path(D + f"ckpt_{tag}/{STAGE[stage]}.yaml.gz")
	_, spec, res = P._stage_entry_from_checkpoint(ck, top_k=idx + 1)[:3]
	g = res.final_population[idx]
	assert g.cells is not None, "headline genome carries no cells (arch-only) — not handled"
	assert g.state_neurons == 0 and g.forced_prefix == 0, "sn>0 genomes are not MCU-exportable"
	assert spec.memory_mode == "BINARY", spec.memory_mode
	s = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	thr = [float(t) for t in P._test_thresholds(a, ec, spec, int(a.test_seeds[0]), s.train, use_score=True)]
	return a, ec, spec, g, thr, s, head


# ---------------------------------------------------------------------------- image
class Image:
	"""The deployed WNN image as flat integer arrays, plus the global bit universe."""

	def __init__(self, g, thr, input_bits: int):
		self.n = g.output_neurons
		self.bits = g.output_bits_per_neuron
		self.input_bits = input_bits
		conn = np.asarray(g.output_sampled, dtype=np.int64)
		assert conn.shape == (self.n, self.bits), conn.shape
		self.conn_w = 8 if conn.max() <= 255 and conn.min() >= 0 else 16
		self.conn = conn.reshape(-1).astype(np.uint8 if self.conn_w == 8 else np.int16)
		ou = np.asarray(g.cells.output_universe, dtype=np.uint64)
		ov = np.asarray(g.cells.output_values)
		assert (ov == 1).all(), "BINARY image must hold only TRUE cells"
		assert ou[:, 1].max() < (1 << 32), "keys must fit uint32"
		order = np.lexsort((ou[:, 1], ou[:, 0]))
		nn, aa = ou[order, 0].astype(np.int64), ou[order, 1].astype(np.uint32)
		self.keys = aa
		self.off = np.searchsorted(nn, np.arange(self.n + 1), side="left").astype(np.uint32)
		self.thr = np.asarray(thr, dtype=np.float32)
		self.seg = [("conn", self.conn.size * self.conn_w), ("keys", self.keys.size * 32),
		            ("off", self.off.size * 32), ("thr", self.thr.size * 32)]
		self.total_bits = sum(b for _, b in self.seg)

	def corrupt(self, k: int, rng: np.random.Generator, only: str | None = None):
		"""Flip k distinct uniformly random bits of the image. Returns
		(conn int64 (N,bits) for the sim, (neuron, addr) on-set arrays, thr list,
		 per-segment flip counts)."""
		lo, nb = 0, self.total_bits
		if only:  # attribution: restrict the universe to ONE segment (same flip semantics)
			lo = sum(b for n, b in self.seg[:[n for n, _ in self.seg].index(only)])
			nb = dict(self.seg)[only]
		pos = lo + rng.choice(nb, size=k, replace=False) if k else np.empty(0, np.int64)
		conn, keys, off = self.conn.copy(), self.keys.copy(), self.off.copy()
		thr = self.thr.copy()
		counts, base = {}, 0
		for name, nb in self.seg:
			p = pos[(pos >= base) & (pos < base + nb)] - base
			counts[name] = int(p.size)
			base += nb
			if not p.size:
				continue
			if name == "conn":
				w = self.conn_w
				u = conn.view(np.uint8 if w == 8 else np.uint16)
				np.bitwise_xor.at(u, p // w, (1 << (p % w)).astype(u.dtype))
			elif name == "keys":
				np.bitwise_xor.at(keys, p // 32, (np.uint64(1) << (p % 32).astype(np.uint64)).astype(np.uint32))
			elif name == "off":
				np.bitwise_xor.at(off, p // 32, (np.uint64(1) << (p % 32).astype(np.uint64)).astype(np.uint32))
			else:
				u = thr.view(np.uint32)
				np.bitwise_xor.at(u, p // 32, (np.uint64(1) << (p % 32).astype(np.uint64)).astype(np.uint32))
		c = conn.astype(np.int64)
		c[(c < 0) | (c >= self.input_bits)] = -1
		nn, aa = self._visible_onset(keys, off, counts["keys"] > 0 or counts["off"] > 0)
		return c.reshape(self.n, self.bits), nn, aa, thr.astype(np.float64).tolist(), counts

	def _visible_onset(self, keys, off, touched: bool):
		"""The on-set the firmware's key_present() can actually find, per neuron."""
		K = keys.size
		lo = np.minimum(off[:-1].astype(np.int64), K)
		end = np.minimum(off[1:].astype(np.int64), K)
		if not touched:
			nn = np.repeat(np.arange(self.n), (end - lo))
			return nn, keys.astype(np.uint64)
		# Candidates: every array slot a neuron's range covers (only stored keys can match).
		cnt = np.maximum(end - lo, 0)
		nn = np.repeat(np.arange(self.n), cnt)
		starts = np.repeat(lo, cnt)
		ends = np.repeat(end, cnt)
		idx = starts + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
		v = keys[idx].astype(np.int64)
		l, h = starts.copy(), ends.copy()
		for _ in range(40):
			act = l < h
			if not act.any():
				break
			m = (l + h) >> 1
			mm = np.where(act, m, 0)
			less = keys[mm].astype(np.int64) < v
			l = np.where(act & less, m + 1, l)
			h = np.where(act & ~less, m, h)
		found = (l < ends) & (keys[np.minimum(l, K - 1)].astype(np.int64) == v)
		nn, v = nn[found], v[found]
		# Addresses beyond the neuron's width can never be formed by neuron_address().
		ok = v < (1 << self.bits)
		nn, v = nn[ok], v[ok]
		# Duplicates (a flip landing on an existing key) are one entry, not two.
		u = np.unique(np.stack([nn, v], 1), axis=0) if nn.size else np.empty((0, 2), np.int64)
		return u[:, 0], u[:, 1].astype(np.uint64)


def make_genome(g, conn, nn, aa):
	import ram_controller as rc
	gp = g.clone()
	gp.output_sampled = [list(map(int, r)) for r in conn]
	z32, z64, z8 = [], [], []
	gp.cells = rc.GenomeCells.from_columns(z32, z64, z8, nn.astype(np.uint32).tolist(),
	                                       aa.astype(np.uint64).tolist(), [1] * int(nn.size))
	return gp


# ---------------------------------------------------------------------------- scoring
def score_wnn(a, ec, spec, thr, genomes, seeds, ctbr_override=None):
	"""[[(stable, err, steady) per seed] per genome] via the installed scorer, as the
	held-out report flies (fresh evaluator per seed => fold 0 of the report pool)."""
	import dataclasses
	sp = spec if ctbr_override is None else dataclasses.replace(spec, ctbr=ctbr_override)
	out = [[] for _ in genomes]
	for rs in seeds:
		try:
			_score_wnn_seed(a, ec, sp, thr, genomes, rs, out)
		except BaseException as e:
			if isinstance(e, KeyboardInterrupt):
				raise
			for o in out:
				o.append([0.0, float("nan"), float("nan"), f"CRASH: {str(e)[:120]}"])
	return out


def _score_wnn_seed(a, ec, sp, thr, genomes, rs, out):
	from wnn.control import phased_ga as P
	from wnn.control.evaluator import ControllerEvaluator
	ev = ControllerEvaluator(sp, num_eval_episodes=a.test_episodes, seed=rs, episode_config=ec,
	                         thresholds=thr, rg_config=P._rg_config(a, ec, rs),
	                         max_train_workers=a.train_workers, num_eval_folds=a.num_eval_folds)
	scored = ev.score_genomes(genomes)  # all-or-nothing: append only after the whole batch scored
	for i, m in enumerate(scored):
		out[i].append([float(getattr(m, k)) for k in KEYS])


def score_mpcof(a, ec, seeds, rate_override=None):
	"""MPCOF>rate(h<action_repeat>) est-fed, the phased_ga cascade-rival row, per seed."""
	from wnn.control import classical_baseline as CB
	rows = []
	for rs in seeds:
		draw = CB.HoldoutDraw(seed=rs, episodes=a.test_episodes, steps=ec.steps_per_episode,
		                      eval_folds=a.num_eval_folds)
		if rate_override is None:
			r = CB.cascade_rival_metrics(ec, draw, CB.TeacherFeed(use_estimator=True), 4, int(a.action_repeat))
			rows.append([r["stable_rate"], r["mean_attitude_error_deg"], r["mean_steady_error_deg"]])
			continue
		from wnn.control._accel import score_classical_baseline
		q0, w0, f = CB._episode_fields(ec, draw)
		f = {**f, **ec.airframe_kwargs(), **ec.motor_lag_kwargs(), **CB.TeacherFeed(use_estimator=True).fields(),
		     **CB._stage1_fields(ec, draw), **CB._abi32_fields(ec), "ctbr_outer_hold": int(a.action_repeat)}
		f["af_pid_rate"] = list(rate_override)
		try:
			r = score_classical_baseline(4, list(q0), list(w0), draw.steps, draw.stable_deg, **f)
			rows.append([r[0], r[1], r[2]])
		except BaseException as e:  # pyo3 PanicException is a BaseException
			if isinstance(e, KeyboardInterrupt):
				raise
			# The controller cannot run (e.g. the QP Hessian built from a corrupted gain is
			# not positive definite): on hardware that is a crashed control task — every
			# episode of this unit is a failure. Recorded as such, flagged.
			rows.append([0.0, float("nan"), float("nan"), f"CRASH: {str(e)[:120]}"])
	return rows


def summarise(per_seed, eps):
	crashes = sum(1 for p in per_seed if len(p) > 3)
	r = np.asarray([p[:3] for p in per_seed], dtype=np.float64)
	return {"crashed_units": crashes, "stable": float(r[:, 0].mean() * 100), "err": float(np.nanmean(r[:, 1])),
	        "steady": float(np.nanmean(r[:, 2])), "fails": int(round(((1 - r[:, 0]) * eps).sum()))}


def test_seeds(a, n):
	s = [int(x) for x in a.test_seeds]
	return s if n is None else s[:n]


# ---------------------------------------------------------------------------- modes
def mode_wnn(args):
	a, ec, spec, g, thr, s, head = load_subject(args.tag)
	from wnn.control.evaluator import NUM_FEATURES  # noqa: F401 (documents the input space source)
	input_bits = g.shape.output_input_space
	img = Image(g, thr, input_bits)
	seeds = test_seeds(a, args.seeds)
	print(f"[ctrl52] {args.tag} {head}: N={img.n} bits={img.bits} K={img.keys.size} conn u{img.conn_w} "
	      f"input_bits={input_bits} thr={img.thr.size} total_bits={img.total_bits} seeds={len(seeds)}", flush=True)
	res = {"only": args.only, "tag": args.tag, "genome": head, "image": {n: b for n, b in img.seg}, "total_bits": img.total_bits,
	       "neurons": img.n, "bits": img.bits, "keys": int(img.keys.size), "input_bits": int(input_bits),
	       "seeds": seeds, "episodes": a.test_episodes, "draws": []}
	out = Path(args.out)
	for k in [int(x) for x in args.ks.split(",")]:
		t0 = time.time()
		nd = 1 if k == 0 else args.draws
		groups = {}  # thresholds tuple -> [(draw, genome, counts, cells)]
		for d in range(nd):
			rng = np.random.default_rng([52, args.subject_id, k, d])
			conn, nn, aa, thr_d, counts = img.corrupt(k, rng, args.only)
			groups.setdefault(tuple(thr_d), []).append((d, make_genome(g, conn, nn, aa), counts, int(nn.size)))
		for thr_d, items in groups.items():
			scores = score_wnn(a, ec, spec, list(thr_d), [it[1] for it in items], seeds)
			for (d, _, counts, ncell), ps in zip(items, scores):
				row = {"k": k, "draw": d, "counts": counts, "visible_cells": ncell, "per_seed": ps,
				       **summarise(ps, a.test_episodes)}
				res["draws"].append(row)
				print(f"  k={k:>7} d={d} st={row['stable']:6.2f} err={row['err']:.2f} sty={row['steady']:.2f} "
				      f"fails={row['fails']:>4} cells={ncell} flips={counts}", flush=True)
		json.dump(res, open(out, "w"))
		print(f"[ctrl52] k={k} done in {time.time() - t0:.0f}s", flush=True)
		ks = [r for r in res["draws"] if r["k"] == k]
		if k and np.mean([r["stable"] for r in ks]) < args.stop_below:
			print(f"[ctrl52] collapse (mean stable < {args.stop_below}) at k={k} — stopping", flush=True)
			break


def _flip_f32(x: float, bit: int) -> float:
	u = np.array([x], dtype=np.float32).view(np.uint32)
	u ^= np.uint32(1 << bit)
	return float(u.view(np.float32)[0])


def mode_rate(args):
	a, ec, spec, g, thr, s, head = load_subject(args.tag)
	seeds = test_seeds(a, args.seeds)
	if args.who == "wnn":
		base = list(spec.ctbr["rate_gains"])
	else:
		base = list(ec.airframe_kwargs()["af_pid_rate"])
	print(f"[ctrl52-rate] who={args.who} base gains={base}", flush=True)
	res = {"who": args.who, "tag": args.tag, "base": base, "seeds": seeds, "episodes": a.test_episodes, "flips": []}
	targets = [(-1, -1)] + [(i, b) for i in range(12) for b in range(32)]
	for i, b in targets:
		gains = list(base)
		if i >= 0:
			gains[i] = _flip_f32(base[i], b)
		t0 = time.time()
		if args.who == "wnn":
			ct = dict(spec.ctbr); ct["rate_gains"] = gains
			ps = score_wnn(a, ec, spec, thr, [g], seeds, ctbr_override=ct)[0]
		else:
			ps = score_mpcof(a, ec, seeds, rate_override=gains)
		row = {"gain": i, "bit": b, "value": gains[i] if i >= 0 else None, "per_seed": ps,
		       **summarise(ps, a.test_episodes)}
		res["flips"].append(row)
		print(f"  gain={i:>2} bit={b:>2} val={row['value']} st={row['stable']:6.2f} err={row['err']:.2f} "
		      f"({time.time() - t0:.1f}s)", flush=True)
		json.dump(res, open(args.out, "w"))


def mode_mpcof(args):
	a, ec, spec, g, thr, s, head = load_subject(args.tag)
	seeds = test_seeds(a, args.seeds)
	t0 = time.time()
	ps = score_mpcof(a, ec, seeds)
	row = {"per_seed": ps, **summarise(ps, a.test_episodes), "seeds": seeds, "secs": time.time() - t0}
	json.dump(row, open(args.out, "w"))
	print(f"[ctrl52-mpcof] {row['stable']:.2f} / {row['err']:.2f} / {row['steady']:.2f} fails={row['fails']} "
	      f"({row['secs']:.0f}s)", flush=True)


def _agg(rows, key):
	v = np.asarray([r[key] for r in rows], dtype=np.float64)
	return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0


def _first_below(ks, means, bar):
	for i, (k, m) in enumerate(zip(ks, means)):
		if m < bar:
			return (ks[i - 1] if i else None, k)
	return None


def mode_report(args):
	"""Markdown tables for the doc + the PNG. Measured numbers only; nothing inferred here."""
	wnn = [json.load(open(f)) for f in args.wnn]
	out = []
	curves = []
	for r in wnn:
		ks = sorted({d["k"] for d in r["draws"]})
		out.append(f"\n#### {r['tag']} ({r['genome']}) — {r['total_bits']:,} stored bits "
		           f"(conn {r['image']['conn']:,} / keys {r['image']['keys']:,} / off {r['image']['off']:,} / thr {r['image']['thr']:,})\n")
		out.append("| k flips | fraction of stored bits | draws | stable % | err ° | steady ° | fails /3500 | visible TRUE cells |")
		out.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
		means = []
		for k in ks:
			rows = [d for d in r["draws"] if d["k"] == k]
			st, sts = _agg(rows, "stable"); er, ers = _agg(rows, "err"); sy, sys_ = _agg(rows, "steady")
			fa, fas = _agg(rows, "fails"); vc, _ = _agg(rows, "visible_cells")
			means.append(st)
			out.append(f"| {k:,} | {k / r['total_bits']:.2e} | {len(rows)} | {st:.2f} ± {sts:.2f} | {er:.2f} ± {ers:.2f} | "
			           f"{sy:.2f} ± {sys_:.2f} | {fa:.1f} ± {fas:.1f} | {vc:,.0f} |")
		for bar in (99.0, 90.0):
			fb = _first_below(ks, means, bar)
			out.append(f"\n- first k with mean stable < {bar:.0f}%: " +
			           ("not reached in the grid" if fb is None else
			            f"**{fb[1]:,}** (= {fb[1] / r['total_bits']:.2e} of stored bits; previous grid point {fb[0]:,} still ≥ {bar:.0f}%)"))
		curves.append((r, ks))
	rate = {}
	for f in args.rate:
		r = json.load(open(f))
		rate[r["who"]] = r
	if rate:
		out.append("\n#### Firmware rate-PID gain table (12 float32 = 384 bits), every single bit flipped alone\n")
		out.append("| controller | units | baseline stable % | flips < 99% stable | flips < 90% | flips < 50% | crashes (planner refused to build) |")
		out.append("|---|---:|---:|---:|---:|---:|---:|")
		n7 = min(len(r["seeds"]) for r in rate.values())
		for who, r in rate.items():
			for nseed in sorted({len(r["seeds"]), n7}):
				def st(fl):
					return float(np.mean([p[0] for p in fl["per_seed"][:nseed]]) * 100)
				base = st(r["flips"][0]); fl = r["flips"][1:]
				c = lambda bar: sum(1 for x in fl if st(x) < bar)
				cr = sum(1 for x in fl if any(len(p) > 3 for p in x["per_seed"][:nseed]))
				out.append(f"| {who} | {nseed} | {base:.2f} | {c(99)} / {len(fl)} | {c(90)} | {c(50)} | {cr} |")
	print("\n".join(out))
	if args.png:
		_plot(curves, rate, args.png)


def _plot(curves, rate, png):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
	fig, axs = plt.subplots(1, 2 if rate else 1, figsize=(12 if rate else 7, 4.6), facecolor="#fcfcfb")
	axs = np.atleast_1d(axs)
	ax = axs[0]
	for i, (r, ks) in enumerate(curves):
		x, m, sd = [], [], []
		for k in ks:
			if k == 0:
				continue
			v = [d["stable"] for d in r["draws"] if d["k"] == k]
			x.append(k / r["total_bits"]); m.append(np.mean(v)); sd.append(np.std(v, ddof=1) if len(v) > 1 else 0)
		x, m, sd = map(np.asarray, (x, m, sd))
		lab = r["tag"].replace("C-S2-d_wposx2_", "").replace("_full", "") + f" ({r['genome']})"
		ax.plot(x, m, color=cols[i % len(cols)], lw=2, marker="o", ms=5, label=lab)
		ax.fill_between(x, m - sd, np.minimum(m + sd, 100), color=cols[i % len(cols)], alpha=0.15, lw=0)
	for bar in (99, 90):
		ax.axhline(bar, color="#52514e", lw=0.8, ls="--")
		ax.text(ax.get_xlim()[0] if False else 1.2e-7, bar - 3, f"{bar}%", color="#52514e", fontsize=9)
	ax.set_xscale("log"); ax.set_xlabel("upsets per stored bit (k / stored bits)")
	ax.set_ylabel("stable % (35 units x 100 episodes)"); ax.set_ylim(-2, 102)
	ax.set_title("WNN cascade image: persistent bit flips", fontsize=11)
	ax.legend(fontsize=8, frameon=False, loc="lower left")
	for sp in ("top", "right"):
		ax.spines[sp].set_visible(False)
	ax.grid(axis="y", color="#e6e5e0", lw=0.6)
	if rate:
		ax = axs[1]
		n7 = min(len(r["seeds"]) for r in rate.values())
		for j, who in enumerate(("wnn", "mpcof")):
			if who not in rate:
				continue
			fl = rate[who]["flips"][1:]
			y = [float(np.mean([p[0] for p in x["per_seed"][:n7]]) * 100) for x in fl]
			ax.plot(np.arange(len(y)) + (0.25 if j else 0), y, ls="none", marker="o", ms=4,
			        color=(cols[0] if who == "wnn" else cols[4]), label=f"{'WNN cascade (s1)' if who == 'wnn' else 'MPCOF>rate(h10)'}")
		for g in range(1, 12):
			ax.axvline(g * 32 - 0.5, color="#e6e5e0", lw=0.6)
		ax.set_xticks([g * 32 + 16 for g in range(12)])
		ax.set_xticklabels(["kp", "ki", "kd", "ilim"] * 3, fontsize=8)
		ax.set_xlabel("flipped bit: roll | pitch | yaw gains (each: mantissa, exponent, sign)")
		ax.set_ylabel(f"stable % ({n7} units x 100 episodes)"); ax.set_ylim(-2, 102)
		ax.set_title("Shared firmware rate-PID gains: each of 384 bits alone", fontsize=11)
		ax.legend(fontsize=8, frameon=False, loc="lower left")
		for sp in ("top", "right"):
			ax.spines[sp].set_visible(False)
	fig.tight_layout()
	fig.savefig(png, dpi=150, facecolor=fig.get_facecolor())
	print(f"[ctrl52] wrote {png}")


def main():
	p = argparse.ArgumentParser()
	sub = p.add_subparsers(dest="mode", required=True)
	w = sub.add_parser("wnn")
	w.add_argument("--tag", required=True); w.add_argument("--ks", required=True)
	w.add_argument("--draws", type=int, default=10); w.add_argument("--seeds", type=int, default=None)
	w.add_argument("--subject-id", type=int, default=0); w.add_argument("--stop-below", type=float, default=50.0)
	w.add_argument("--out", required=True)
	w.add_argument("--only", choices=("conn", "keys", "off", "thr"), default=None,
	               help="attribution: flip only this segment's bits")
	r = sub.add_parser("rate")
	r.add_argument("--who", choices=("wnn", "mpcof"), required=True); r.add_argument("--tag", required=True)
	r.add_argument("--seeds", type=int, default=None); r.add_argument("--out", required=True)
	m = sub.add_parser("mpcof")
	m.add_argument("--tag", required=True); m.add_argument("--seeds", type=int, default=None)
	m.add_argument("--out", required=True)
	rp = sub.add_parser("report")
	rp.add_argument("--wnn", nargs="*", default=[]); rp.add_argument("--rate", nargs="*", default=[])
	rp.add_argument("--png", default=None)
	args = p.parse_args()
	{"wnn": mode_wnn, "rate": mode_rate, "mpcof": mode_mpcof, "report": mode_report}[args.mode](args)


if __name__ == "__main__":
	main()
