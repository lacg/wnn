#!/usr/bin/env python3
"""Progress tracker for the ciciot 96b-Wc-250n100b n=30 cohort.

Fill-rate + running best F1/FPR (across ALL genomes × ALL 7 thresholds) vs the
paper's CICIOT Pareto targets. Read-only DB (live file). No args.
"""
import sqlite3, re, json
from datetime import datetime, timezone
from collections import Counter

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
PREFIX = "XDS-ciciot-subsample-96b-Wc-C35-250n100b-OI-r"
TARGET = 30
# Paper CICIOT reference points (F1, FPR, Acc). best_f1/best_acc = the paper's
# UNCONSTRAINED 108.7MB champions (not yet beaten); fpr<6/fpr<5 = low-FPR Pareto.
PAPER = {"fpr<6": (93.05, 5.89, 96.47), "fpr<5": (88.31, 0.98, 93.35),
         "best_f1": (93.26, 7.27, 96.63), "best_acc": (93.22, 8.71, 96.65)}


def main():
	c = sqlite3.connect(DB, uri=True); c.row_factory = sqlite3.Row
	flows = c.execute(
		"SELECT id,name,status,started_at,completed_at FROM flows WHERE name LIKE ?",
		(PREFIX + "%",)).fetchall()
	st = Counter(f["status"] for f in flows)
	done = st.get("completed", 0)
	now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
	print(f"========== ciciot 96b-Wc-250n100b — n={TARGET} tracker — {now} ==========")
	print(f"fill: {done}/{TARGET} completed  |  running={st.get('running',0)}  queued={st.get('queued',0)}  "
	      f"pending={st.get('pending',0)}")

	# avg completed-flow duration → rough ETA
	durs = []
	for f in flows:
		if f["status"] == "completed" and f["started_at"] and f["completed_at"]:
			d = c.execute("SELECT (julianday(?)-julianday(?))*1440", (f["completed_at"], f["started_at"])).fetchone()[0]
			if d and d > 0: durs.append(d)
	if durs:
		avg = sum(durs) / len(durs)
		rem = TARGET - done
		print(f"avg run: {avg:.0f} min  |  remaining: {rem}  |  serial est: ~{rem*avg/60:.1f}h "
		      f"(longer if queued behind other flows)")

	# running best across all completed flows: scan all genome×threshold points
	pts = []  # (f1, fpr, acc, seed, gt, mode)
	rows = c.execute(
		"""SELECT f.name, vs.genome_type, vs.threshold_metadata
		FROM validation_summaries vs JOIN experiments e ON vs.experiment_id=e.id
		JOIN flows f ON e.flow_id=f.id
		WHERE f.name LIKE ? AND f.status='completed'""", (PREFIX + "%",)).fetchall()
	for r in rows:
		seed = re.search(r"-r(\d+)$", r["name"]).group(1)
		try: tm = json.loads(r["threshold_metadata"])
		except Exception: continue
		for mode, md in tm.items():
			if not isinstance(md, dict) or md.get("f1") is None: continue
			pts.append((md["f1"]*100, md["fpr"]*100, md["acc"]*100, seed, r["genome_type"], mode))
	if not pts:
		print("\n(no completed validation summaries yet)"); return

	def best_f1(pred):
		cand = [p for p in pts if pred(p)]
		return max(cand, key=lambda p: p[0]) if cand else None
	def best_fpr(pred):
		cand = [p for p in pts if pred(p)]
		return min(cand, key=lambda p: p[1]) if cand else None
	def best_acc(pred):
		cand = [p for p in pts if pred(p)]
		return max(cand, key=lambda p: p[2]) if cand else None

	print(f"\nrunning best over {done} flow(s), ALL genomes × ALL thresholds:")
	def line(label, p, paper=None, metric="f1"):
		if not p: print(f"  {label:<26} —"); return
		tag = ""
		if paper:
			if metric == "acc":
				tag = f"  [vs paper Acc {p[2]-paper[2]:+.2f} / FPR {p[1]-paper[1]:+.2f}]"
			else:
				tag = f"  [vs paper F1 {p[0]-paper[0]:+.2f} / FPR {p[1]-paper[1]:+.2f}]"
		print(f"  {label:<26} F1={p[0]:.2f}%  FPR={p[1]:.2f}%  Acc={p[2]:.2f}%  (r{p[3]} {p[4]} {p[5]}){tag}")
	line("Best F1 (any FPR)", best_f1(lambda p: True), PAPER["best_f1"])
	line("Best Acc (any FPR)", best_acc(lambda p: True), PAPER["best_acc"], metric="acc")
	line("Best F1 @ FPR<6%", best_f1(lambda p: p[1] < 6), PAPER["fpr<6"])
	line("Best F1 @ FPR<5%", best_f1(lambda p: p[1] < 5), PAPER["fpr<5"])
	line("Best FPR @ F1>90%", best_fpr(lambda p: p[0] > 90))
	print(f"\n  paper CICIOT targets: best-F1 93.26/7.27/96.63 | best-Acc 93.22/8.71/96.65 "
	      f"| FPR<6% 93.05/5.89/96.47 | FPR<5% 88.31/0.98/93.35")


if __name__ == "__main__":
	main()
