#!/usr/bin/env python3
"""Live progress + results report for the ceiling pipeline (phases S, B, A, C).

READ-ONLY. It opens the pipeline's log, its per-phase stdout files and its JSON
markers, and prints a table. It never writes, never signals, never touches a
running process — it is safe to run against the pipeline mid-flight, which is the
only time it is useful.

Phase S is the one that carries the claim. The dfa1l study ran a 12-bit state
prefix + an 18-bit sensor SUFFIX and collapsed on held-out (~35-49% stable, with
one training seed diverging outright); a 10-episode smoke suggested suffix 32
moves held-out to 90-95%. So the phase-S table is built around ONE question —
does held-out stable% rise with the suffix? — and everything in it serves reading
that trend: arms are grouped by (sn, suffix), aggregated across the 5 training
seeds (the smoke showed single-seed scatter swamping real effects), sorted by
suffix, and the CONTROL arm (sn=12, suf=18) is marked because it is the number to
beat, not just another row.

Mid-run partiality is the normal case, not an edge case: rows are parsed out of
the live .out files as the sweep emits them, so a phase with 1 of 65 arms done
reports 1 arm. A phase that has not started prints "not started" and nothing else.

Usage:  build_ceiling_report.py [--phase S|B|A|C|all] [--compact] [--raw]
"""
import argparse
import glob
import json
import os
import re
import statistics
from datetime import datetime, timedelta, timezone

# The pipeline's own constants, mirrored. If run_ceiling_pipeline.sh changes its
# grid, these change with it or the progress fractions silently lie.
CONTROL_SN = 12
CONTROL_SUF = 18
PHASE_S_EXPECTED_ARMS = 65      # 1x1x5 control + 2sn x 6suf x 5 tseed
FAULT_BAR = ("MPCOF", 20.0)     # phase C's beat-the-classical bar, % stable

_PHASE_START = re.compile(r"\[ceiling\] =+ PHASE ([SBAC]): (.*?) =+ (\S+Z)")
_PHASE_RC = re.compile(r"\[ceiling\] phase ([SBAC]) rc=(-?\d+) (\S+Z)")
_PHASE_FAIL = re.compile(r"\[ceiling\] phase ([SBAC]) (.*FAILED) (\S+Z)")
# ' 30   12   18   16  31337002    38.6±16.3     6.18±1.02      8,577    265s'
_SWEEP_ROW = re.compile(
	r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
	r"([\d.]+)±([\d.]+)\s+([\d.]+)±([\d.]+)\s+([\d,]+)\s+([\d.]+)s\s*$")
_GEN_LINE = re.compile(
	r"\[ControllerGA-(\w+)\] Gen (\d+)/(\d+):.*?stable=([\d.]+)%,\s*err=([\d.]+)°")
# Architecture is NEVER passed on a phase's command line — bits_levels_sweep.py and
# phased_ga --seed-winner both INHERIT substrate/features/memory-mode from the winner
# checkpoint. So the winner path is the only source of truth, and each phase echoes
# its own: the sweep in its '# bits x levels sweep:' header, phased_ga in its
# '[main] CURRICULUM seed-winner from ...' line. Parsed, never assumed — a phase that
# is later re-pointed at a different winner then reports ITS winner, not phase S's.
_WINNER_DECL = re.compile(
	r"(?:# bits x levels sweep:|seed-winner from)\s+(\S+_winner\.yaml\.gz)")
_WINNER_NAME = re.compile(
	r"([A-Za-z0-9]+)_(\d+feat)_(BINARY|QUAD|TERNARY|PLN|MPLN|QSR)_s(\d+)_winner"
	r"\.yaml\.gz$")
_HELD_OUT = re.compile(
	r"RESULT — during-search winner \(held-out\):\s+stable=([\d.]+)%\s+"
	r"err=([\d.]+)°\s+steady=([\d.]+)°")
_BUDGET_SLOPE = re.compile(r"^# slope .*$")

PHASE_TITLES = {
	"S": "split sweep — state/sensor bit split (the suffix trend)",
	"B": "data-budget curve — 1x/4x/16x DAgger episodes",
	"A": "nominal long memory-GA — past the imitation plateau?",
	"C": "motor-fault memory-GA — beat the classical bar",
}
PHASE_GLOBS = {
	"S": ("phaseS_control.out", "phaseS_sweep.out"),
	"B": ("phaseB_probe.out",),
	"A": ("phaseA_*.out",),
	"C": ("phaseC_baselines.out", "phaseC_fault_memga.out"),
}


# ---------------------------------------------------------------- primitives --

def _read(path: str) -> str:
	"""File contents, or '' when the phase has not produced the file yet."""
	try:
		with open(path, "r", errors="replace") as f:
			return f.read()
	except OSError:
		return ""


def _load_json(path: str) -> dict:
	try:
		with open(path, "r") as f:
			return json.load(f)
	except (OSError, ValueError):
		return {}


def _utc(stamp: str) -> datetime:
	"""'2026-07-31T18:59:51Z' -> aware datetime; None when unparseable."""
	try:
		return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
	except (TypeError, ValueError):
		return None


def _mtime(path: str) -> datetime:
	try:
		return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
	except OSError:
		return None


def _hms(seconds: float) -> str:
	if seconds is None:
		return "—"
	h, rem = divmod(int(max(seconds, 0)), 3600)
	return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def _stamp(when: datetime) -> str:
	return when.strftime("%d/%m/%Y %H:%M") if when else "—"


def _stat(values: list) -> tuple:
	"""(mean, population sd). sd is 0 for n=1 — an honest 'no spread measured'."""
	vals = [v for v in values if v is not None]
	if not vals:
		return (None, None)
	return (statistics.mean(vals),
	        statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def _fmt_ms(mean: float, sd: float, width: int, prec: int) -> str:
	if mean is None:
		return "—".rjust(width * 2 + 1)
	return f"{mean:{width}.{prec}f}±{sd:{width}.{prec}f}"


# ---------------------------------------------------------------- winner arch --

def parse_architecture(paths: list) -> dict:
	"""substrate/feature/mode/seed of the winner a phase inherited, or None.

	Returns None — never a guessed default — when no file exists, no winner is
	declared, or the basename does not match the naming convention. A wrong
	architecture label on a results table is worse than an absent one.
	"""
	for path in paths:
		for line in _read(path).splitlines():
			decl = _WINNER_DECL.search(line)
			if not decl:
				continue
			name = _WINNER_NAME.search(os.path.basename(decl.group(1)))
			if name:
				return {"substrate": name.group(1), "feat": name.group(2),
				        "mode": name.group(3), "seed": name.group(4),
				        "path": decl.group(1)}
	return None


def arch_token(arch: dict) -> str:
	"""Compact one-token form for the tick line."""
	if not arch:
		return "arch:unknown"
	return f"{arch['substrate']}/{arch['feat']}/{arch['mode']}"


def print_architecture(arch: dict) -> None:
	if not arch:
		print("  architecture: unknown (no winner declared in this phase's output)")
		return
	print(f"  architecture: {arch['substrate']} / {arch['feat']} / {arch['mode']}"
	      f"   (seeded from s{arch['seed']} winner)")


# --------------------------------------------------------------- phase state --

def parse_phase_log(path: str) -> dict:
	"""Pipeline log -> {phase: {'start','end','rc','label'}} (only what it says)."""
	events = {}
	for line in _read(path).splitlines():
		m = _PHASE_START.search(line)
		if m:
			events.setdefault(m.group(1), {})["start"] = _utc(m.group(3))
			events[m.group(1)]["label"] = m.group(2).strip()
			continue
		m = _PHASE_RC.search(line)
		if m:
			ev = events.setdefault(m.group(1), {})
			ev["rc"], ev["end"] = int(m.group(2)), _utc(m.group(3))
			continue
		m = _PHASE_FAIL.search(line)
		if m:
			ev = events.setdefault(m.group(1), {})
			ev["rc"], ev["end"] = ev.get("rc", 5), _utc(m.group(3))
			ev["label"] = m.group(2)
	return events


def phase_paths(outdir: str, phase: str) -> list:
	"""Existing per-phase stdout files, newest last."""
	found = []
	for pat in PHASE_GLOBS[phase]:
		found.extend(sorted(glob.glob(os.path.join(outdir, pat))))
	return [p for p in found if os.path.exists(p)]


def phase_state(phase: str, events: dict, paths: list, now: datetime) -> dict:
	"""Resolve not-started / running / done(rc), with elapsed and last activity.

	The log is authoritative for start and rc; file mtimes are the only evidence a
	RUNNING phase is still alive, so they are reported rather than inferred from.
	"""
	ev = events.get(phase, {})
	start, end, rc = ev.get("start"), ev.get("end"), ev.get("rc")
	last = max([m for m in (_mtime(p) for p in paths) if m], default=None)
	if start is None and not paths:
		return {"state": "not started", "start": None, "elapsed": None,
		        "rc": None, "last": None}
	if rc is not None:
		return {"state": f"done rc={rc}", "start": start, "rc": rc,
		        "elapsed": (end - start).total_seconds() if start and end else None,
		        "last": end or last}
	return {"state": "running", "start": start, "rc": None,
	        "elapsed": (now - start).total_seconds() if start else None,
	        "last": last}


def print_phase_status(states: dict, now: datetime) -> None:
	print("=" * 88)
	print("  CEILING PIPELINE — phase status")
	print(f"  now {_stamp(now)} UTC")
	print("=" * 88)
	print(f"  {'ph':2}  {'state':12} {'started (UTC)':16} {'elapsed':>8}  "
	      f"{'last write':16}  what")
	print("  " + "-" * 84)
	for ph in ("S", "B", "A", "C"):
		st = states[ph]
		print(f"  {ph:2}  {st['state']:12} {_stamp(st['start']):16} "
		      f"{_hms(st['elapsed']):>8}  {_stamp(st['last']):16}  {PHASE_TITLES[ph]}")
	print()


# ------------------------------------------------------------------- phase S --

def parse_sweep_text(text: str) -> list:
	"""Arm rows out of a live bits_levels_sweep stdout.

	Rows are interleaved with '[progress] dagger-batch: ...' lines and preceded by
	'#' commentary and a column header; only the numeric row shape matches, so no
	filtering pass is needed and a half-written trailing line simply does not match.
	"""
	arms = []
	for line in text.splitlines():
		m = _SWEEP_ROW.match(line)
		if not m:
			continue
		g = m.groups()
		arms.append({"ob": int(g[0]), "sn": int(g[1]), "suf": int(g[2]),
		             "levels": int(g[3]), "tseed": int(g[4]),
		             "stable": float(g[5]), "stable_sd": float(g[6]),
		             "err": float(g[7]), "err_sd": float(g[8]),
		             "cells": int(g[9].replace(",", "")), "train_s": float(g[10])})
	return arms


def steady_index(paths: list) -> dict:
	"""(ob, sn, suf, cells) -> held-out steady° mean, from the sweep JSONs.

	The stdout table drops steady° for width; the JSON keeps it, but is only written
	when a sweep invocation FINISHES. So this enriches completed invocations and
	leaves the in-flight one blank, which is exactly the honest state.
	"""
	index = {}
	for path in paths:
		for row in _load_json(path).get("rows", []):
			key = (row.get("ob"), row.get("sn"), row.get("suffix"), row.get("cells"))
			steady = row.get("steady_deg")
			if steady:
				index[key] = steady[0]
	return index


def collect_phase_s(outdir: str, markdir: str) -> tuple:
	"""(arms, steady-by-arm). Both sweep invocations are one logical phase."""
	arms = []
	for name in ("phaseS_control.out", "phaseS_sweep.out"):
		arms.extend(parse_sweep_text(_read(os.path.join(outdir, name))))
	jsons = [os.path.join(markdir, n)
	         for n in ("split_control.json", "split_sweep.json")]
	return arms, steady_index(jsons)


def group_by_split(arms: list, steady: dict) -> list:
	"""Aggregate arms across training seeds into one row per (sn, suffix)."""
	groups = {}
	for a in arms:
		groups.setdefault((a["sn"], a["suf"]), []).append(a)
	rows = []
	for (sn, suf), members in sorted(groups.items()):
		stables = [m["stable"] for m in members]
		steadies = [steady.get((m["ob"], m["sn"], m["suf"], m["cells"]))
		            for m in members]
		rows.append({
			"sn": sn, "suf": suf, "ob": members[0]["ob"], "n": len(members),
			"stable": _stat(stables), "err": _stat([m["err"] for m in members]),
			"steady": _stat(steadies), "cells": _stat([m["cells"] for m in members]),
			"train_s": _stat([m["train_s"] for m in members]),
			"lo": min(stables), "hi": max(stables),
			# A diverged training seed (0% stable, ~80° error) is a qualitatively
			# different failure from a merely weak arm, and it drags the mean hard.
			# Counting it separately keeps the mean readable instead of mysterious.
			"diverged": sum(1 for s in stables if s < 1.0),
			"converged": _stat([s for s in stables if s >= 1.0]),
			"is_control": sn == CONTROL_SN and suf == CONTROL_SUF})
	return rows


def phase_s_eta(arms: list, start: datetime, now: datetime) -> str:
	"""Wall-clock projection from arms completed so far. Coarse by construction."""
	if not arms or not start:
		return "—"
	done, spent = len(arms), (now - start).total_seconds()
	if done >= PHASE_S_EXPECTED_ARMS or spent <= 0:
		return "—"
	eta = now + timedelta(seconds=(PHASE_S_EXPECTED_ARMS - done) * spent / done)
	return f"{_stamp(eta)} UTC (avg {_hms(spent / done)}/arm over {done} done)"


def print_phase_s_header(arms: list, arch: dict, state: dict, now: datetime) -> None:
	print("=" * 88)
	print("  PHASE S — does the SENSOR SUFFIX lift held-out stability?")
	print("  ob = sn + suffix, state prefix FORCED. Each arm: cells wiped, one")
	print("  training pass, scored on 5 report seeds; rows aggregate training seeds.")
	print_architecture(arch)
	print(f"  arms done {len(arms)}/{PHASE_S_EXPECTED_ARMS}   |   "
	      f"ETA {phase_s_eta(arms, state['start'], now)}")
	print("=" * 88)


def print_phase_s_groups(rows: list) -> None:
	print(f"  {'sn':>3} {'suf':>4} {'ob':>3} {'n':>2} {'div':>3}  "
	      f"{'stable%':>12} {'lo–hi':>13} {'err°':>12} {'steady°':>12} "
	      f"{'cells':>8}")
	print("  " + "-" * 84)
	for r in rows:
		tag = "  <= CONTROL (baseline to beat)" if r["is_control"] else ""
		print(f"  {r['sn']:>3} {r['suf']:>4} {r['ob']:>3} {r['n']:>2} "
		      f"{r['diverged']:>3}  {_fmt_ms(*r['stable'], 5, 1):>12} "
		      f"{r['lo']:5.1f}–{r['hi']:5.1f}  {_fmt_ms(*r['err'], 5, 2):>12} "
		      f"{_fmt_ms(*r['steady'], 5, 2):>12} "
		      f"{r['cells'][0]:8,.0f}{tag}")
	print("  " + "-" * 84)
	print("  div = training seeds that DIVERGED (stable < 1%); they stay in the mean")
	print("        but are counted here so a dragged mean is never a mystery. Read")
	print("        div together with lo–hi: a wide range with div>0 is a bimodal arm")
	print("        (some seeds train, some fall over), NOT a uniformly mediocre one.")
	for r in [r for r in rows if r["diverged"]]:
		print(f"        sn={r['sn']} suf={r['suf']}: {r['stable'][0]:.1f}% over all "
		      f"{r['n']} seeds, {r['converged'][0]:.1f}% over the "
		      f"{r['n'] - r['diverged']} converged.")


def print_phase_s_trend(rows: list) -> None:
	"""The suffix trend, per sn block, as the delta against the control arm."""
	control = next((r for r in rows if r["is_control"]), None)
	if not control:
		return
	base = control["stable"][0]
	swept = [r for r in rows if not r["is_control"]]
	if not swept:
		print()
		print(f"  SUFFIX TREND: no swept arm finished yet — control sits at "
		      f"{base:.1f}% stable.")
		return
	print()
	print(f"  SUFFIX TREND vs CONTROL (sn={CONTROL_SN} suf={CONTROL_SUF} = "
	      f"{base:.1f}% stable):")
	print(f"  {'sn':>3} {'suf':>4} {'n':>2}  {'stable%':>8}  {'Δ vs control':>13}")
	print("  " + "-" * 84)
	for r in sorted(swept, key=lambda x: (x["sn"], x["suf"])):
		print(f"  {r['sn']:>3} {r['suf']:>4} {r['n']:>2}  {r['stable'][0]:8.1f}  "
		      f"{r['stable'][0] - base:+13.1f}")


def print_phase_s_arms(arms: list) -> None:
	"""Per-training-seed rows. Hidden by default (--raw): the aggregated table is
	the one to read, and this is only for identifying WHICH seed diverged."""
	print()
	print("  RAW ARMS (one training seed each, mean±SD over 5 report seeds):")
	print(f"  {'ob':>3} {'sn':>3} {'suf':>4} {'lvl':>3} {'tseed':>9}  "
	      f"{'stable%':>12} {'err°':>12} {'cells':>8} {'train':>7}")
	print("  " + "-" * 84)
	for a in sorted(arms, key=lambda x: (x["sn"], x["suf"], x["tseed"])):
		print(f"  {a['ob']:>3} {a['sn']:>3} {a['suf']:>4} {a['levels']:>3} "
		      f"{a['tseed']:>9}  {_fmt_ms(a['stable'], a['stable_sd'], 5, 1):>12} "
		      f"{_fmt_ms(a['err'], a['err_sd'], 5, 2):>12} "
		      f"{a['cells']:>8,} {a['train_s']:>6.0f}s")


def print_phase_s(outdir: str, markdir: str, state: dict, now: datetime,
                  raw: bool) -> None:
	arms, steady = collect_phase_s(outdir, markdir)
	arch = parse_architecture(phase_paths(outdir, "S"))
	print_phase_s_header(arms, arch, state, now)
	if not arms:
		print("  (no arms scored yet)")
		print()
		return
	rows = group_by_split(arms, steady)
	print_phase_s_groups(rows)
	print_phase_s_trend(rows)
	if raw:
		print_phase_s_arms(arms)
	print()


# ------------------------------------------------------------------- phase B --

def print_phase_b(outdir: str, markdir: str) -> None:
	print("=" * 88)
	print("  PHASE B — data-budget curve (is the gap DATA STARVATION?)")
	print("  Read the SHAPE across retrain arms, not saved-vs-retrain: `saved`")
	print("  accumulated over 5 folds x every GA generation.")
	print("=" * 88)
	text = _read(os.path.join(outdir, "phaseB_probe.out"))
	doc = _load_json(os.path.join(markdir, "data_budget_probe.json"))
	arms = doc.get("arms", {})
	if not arms and not text:
		print("  (not started)")
		print()
		return
	if arms:
		print(f"  {'arm':12} {'episodes':>9}  {'stable%':>12} {'err°':>12} "
		      f"{'train':>7}")
		print("  " + "-" * 84)
		for name, arm in arms.items():
			met = arm.get("metrics", {})
			st, er = met.get("stable", [None, None]), met.get("err_deg", [None, None])
			eps = arm.get("episodes")
			tr = arm.get("train_s")
			print(f"  {name:12} {eps if eps else '—':>9}  "
			      f"{_fmt_ms(st[0], st[1], 5, 1):>12} "
			      f"{_fmt_ms(er[0], er[1], 5, 2):>12} "
			      f"{(f'{tr:.0f}s' if tr else '—'):>7}")
		print("  " + "-" * 84)
	for line in text.splitlines():
		if _BUDGET_SLOPE.match(line):
			print(f"  {line[2:]}")
	print()


# ----------------------------------------------------------------- phases A/C --

def parse_memga(text: str) -> dict:
	"""Latest gen-line + every HELD-OUT winner triple from a phased_ga stdout."""
	gens = _GEN_LINE.findall(text)
	held = _HELD_OUT.findall(text)
	return {"gen": gens[-1] if gens else None,
	        "held": [tuple(float(x) for x in h) for h in held]}


def print_memga_phase(phase: str, path: str, title: str, note: str) -> None:
	print("=" * 88)
	print(f"  PHASE {phase} — {title}")
	print("=" * 88)
	text = _read(path)
	if not text:
		print("  (not started)")
		print()
		return
	# This phase's OWN winner — it seeds from --seed-winner independently, so it is
	# not safe to reuse phase S's label here.
	print_architecture(parse_architecture([path]))
	info = parse_memga(text)
	if info["gen"]:
		stage, gen, total, stable, err = info["gen"]
		print(f"  during search  : stage {stage}  gen {gen}/{total}  "
		      f"stable={float(stable):.1f}%  err={float(err):.2f}°")
	else:
		print("  during search  : no generation line yet")
	if info["held"]:
		print(f"  {'#':>2}  {'stable%':>8} {'err°':>8} {'steady°':>8}   "
		      f"(HELD-OUT REPORT — the honest numbers)")
		print("  " + "-" * 84)
		for i, (stable, err, steady) in enumerate(info["held"], 1):
			print(f"  {i:>2}  {stable:8.1f} {err:8.2f} {steady:8.2f}")
	else:
		print("  held-out       : no HELD-OUT REPORT block yet")
	if note:
		print(f"  {note}")
	print()


# ---------------------------------------------------------------- baselines --

def print_baselines(path: str, title: str) -> None:
	"""Classical controllers on the SAME held-out seed the WNN rows were scored on."""
	doc = _load_json(path)
	baselines = doc.get("baselines", {})
	if not baselines:
		return
	paired = str(doc.get("meta", {}).get("report_seed", ""))
	print(f"  {title} (paired report seed {paired}):")
	print(f"  {'controller':12} {'stable%':>8} {'err°':>8} {'steady°':>8}")
	print("  " + "-" * 84)
	for name in ("PID", "LQR", "MPC", "LQI", "MPCOF"):
		b = baselines.get(name)
		if not b:
			continue
		tri = b.get("per_seed", {}).get(paired)
		st, er, sy = tri if tri else (b["stable"], b["err_deg"], b["steady_deg"])
		print(f"  {name:12} {st:8.1f} {er:8.2f} {sy:8.2f}")
	print()


def fault_baseline_paths(markdir: str) -> list:
	return sorted(glob.glob(os.path.join(markdir, "baselines_fault_*.json")))


# ------------------------------------------------------------------- compact --

def print_compact(states: dict, outdir: str, markdir: str, now: datetime) -> None:
	"""A few lines for a 30-minute status tick: status + best arm + control."""
	parts = [f"{ph}:{states[ph]['state']}" for ph in ("S", "B", "A", "C")]
	print(f"[ceiling {_stamp(now)}Z] " + "  ".join(parts))
	arms, steady = collect_phase_s(outdir, markdir)
	arch = arch_token(parse_architecture(phase_paths(outdir, "S")))
	if not arms:
		print(f"  phase S [{arch}]: no arms scored yet")
		return
	print(f"  phase S [{arch}]: {len(arms)}/{PHASE_S_EXPECTED_ARMS} arms  |  ETA "
	      f"{phase_s_eta(arms, states['S']['start'], now)}")
	rows = group_by_split(arms, steady)
	best = max(rows, key=lambda r: r["stable"][0])
	control = next((r for r in rows if r["is_control"]), None)
	print(f"  best (sn={best['sn']} suf={best['suf']}, n={best['n']}): "
	      f"stable {_fmt_ms(*best['stable'], 5, 1)}%  err {_fmt_ms(*best['err'], 5, 2)}°")
	if control:
		print(f"  CONTROL (sn={control['sn']} suf={control['suf']}, n={control['n']}): "
		      f"stable {_fmt_ms(*control['stable'], 5, 1)}%  "
		      f"err {_fmt_ms(*control['err'], 5, 2)}°")


# ---------------------------------------------------------------------- main --

def build_states(logpath: str, outdir: str, now: datetime) -> dict:
	events = parse_phase_log(logpath)
	return {ph: phase_state(ph, events, phase_paths(outdir, ph), now)
	        for ph in ("S", "B", "A", "C")}


def print_full(phase: str, states: dict, outdir: str, markdir: str,
               now: datetime, raw: bool) -> None:
	print_phase_status(states, now)
	if phase in ("S", "all"):
		print_phase_s(outdir, markdir, states["S"], now, raw)
	if phase in ("B", "all"):
		print_phase_b(outdir, markdir)
	if phase in ("A", "all"):
		print_memga_phase("A", os.path.join(outdir, "phaseA_nominal_memga.out"),
		                  "nominal long memory-GA (800 gens, patience 20)", "")
		print_baselines(os.path.join(markdir, "baselines.json"),
		                "CLASSICAL BASELINES — nominal plant, L2D")
	if phase in ("C", "all"):
		bar_name, bar = FAULT_BAR
		print_memga_phase("C", os.path.join(outdir, "phaseC_fault_memga.out"),
		                  "motor-fault memory-GA",
		                  f"bar to beat on this plant: {bar_name} {bar:.1f}% stable")
		for path in fault_baseline_paths(markdir):
			print_baselines(path, f"FAULT BASELINES — {os.path.basename(path)}")


def main() -> None:
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	ap = argparse.ArgumentParser(description="Read-only ceiling-pipeline report.")
	ap.add_argument("--root", default=root)
	ap.add_argument("--outdir", default=None, help="default <root>/logs/controller/ceiling")
	ap.add_argument("--markdir", default=None, help="default <root>/experiments/dfa1l_markers")
	ap.add_argument("--log", default="/private/tmp/ceiling_pipeline.log")
	ap.add_argument("--phase", default="all", choices=["S", "B", "A", "C", "all"])
	ap.add_argument("--compact", action="store_true")
	ap.add_argument("--raw", action="store_true",
	                help="also list per-training-seed arms (which seed diverged)")
	a = ap.parse_args()

	outdir = a.outdir or os.path.join(a.root, "logs/controller/ceiling")
	markdir = a.markdir or os.path.join(a.root, "experiments/dfa1l_markers")
	now = datetime.now(timezone.utc)
	states = build_states(a.log, outdir, now)
	if a.compact:
		print_compact(states, outdir, markdir, now)
		return
	print_full(a.phase, states, outdir, markdir, now, a.raw)


if __name__ == "__main__":
	main()
