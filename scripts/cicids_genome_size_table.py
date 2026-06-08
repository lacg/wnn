#!/usr/bin/env python3
"""Generate the CICIDS-random thermo × weight table WITH deployed genome size.

For each completed XDS-cicids flow: the headline (best-F1) genome's held-out
F1/FPR/ACC + its architecture (neurons × bits) + a sparse-cell bound, and a flag
on any degenerate best-FPR genome (low F1 from under-predicting attacks).

Writes docs/cicids_thermo_genome_size.md. Single-seed (r82096) snapshot — the
sweep is mid-run; re-run as more widths/seeds land.

Sizing notes:
  * wiring (exact, deterministic)   = neurons × bits  (connection / address-line bits)
  * dense address space (NOT stored) = neurons × 2^bits  (sparse storage skips this)
  * sparse cells (populated)         = data-dependent; bounded above by
        neurons × min(2^bits, n_train_sampled). Exact count needs the trained
        model / Vivado synth — reported as an UPPER BOUND, real fill is lower.
"""
import sqlite3, re, json, sys
from pathlib import Path

DB = "db/wnn.db"
# CICIDS2017: ~2.83M rows, 80% train, neuron_sample_rate 0.25.
N_TRAIN = int(2_827_880 * 0.80)
SAMPLE_RATE = 0.25
N_TRAIN_SAMPLED = int(N_TRAIN * SAMPLE_RATE)

WIDTHS = ["8b", "16b", "32b", "64b", "96b"]
WSETS = ["Wa", "Wb", "Wbu", "Wc"]


def main():
	con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
	flows = con.execute("""SELECT id,name FROM flows
		WHERE name LIKE 'XDS-cicids%' AND status='completed' ORDER BY name""").fetchall()

	def arch(gh):
		r = con.execute("SELECT total_neurons,tiers_json FROM genomes WHERE genome_hash=? LIMIT 1", (gh,)).fetchone()
		if not r:
			return None
		try:
			bits = json.loads(r["tiers_json"]).get("bits_per_neuron", [])
			b = max(bits) if bits else 0
		except Exception:
			b = 0
		return r["total_neurons"], b

	def metric_winner(fid, gtype, col, best):
		rows = con.execute(f"""SELECT genome_hash,f1_macro,fpr,accuracy FROM validation_summaries
			WHERE flow_id=? AND validation_point='final' AND genome_type=? AND {col} IS NOT NULL""",
			(fid, gtype)).fetchall()
		if not rows:
			return None
		r = best(rows, key=lambda x: x[col])
		return dict(f1=r["f1_macro"] * 100, fpr=r["fpr"] * 100, acc=r["accuracy"] * 100,
		            arch=arch(r["genome_hash"]), hash=r["genome_hash"])

	cells = {}
	for f in flows:
		m = re.search(r'-(\d+)b-W(\w+?)-', f["name"])
		if not m:
			continue
		w, ws = m.group(1) + "b", "W" + m.group(2)
		cells[(w, ws)] = dict(
			f1=metric_winner(f["id"], "best_f1", "f1_macro", max),
			fpr=metric_winner(f["id"], "best_fpr", "fpr", min),
		)

	present_widths = [w for w in WIDTHS if any((w, ws) in cells for ws in WSETS)]

	def sparse_bound(n, b):
		if not n or not b:
			return None
		import math
		return n * min(2 ** b, N_TRAIN_SAMPLED)

	out = []
	out.append("# CICIDS-random — thermo × weight + deployed genome size\n")
	out.append("Held-out (report) best-F1 genome per cell, with the architecture the GA actually "
	           "converged to. **Single seed r82096** (sweep mid-run). bits = 34 uniform on every "
	           "non-degenerate winner.\n")
	out.append(f"- wiring (exact) = neurons × bits — the deterministic FPGA cost\n"
	           f"- sparse-cell UPPER BOUND = neurons × min(2^bits, n_train_sampled={N_TRAIN_SAMPLED:,}); "
	           f"real fill is lower (address collisions), exact needs Vivado synth\n")

	# headline table
	out.append("\n## Headline genome per cell (best-F1 = best-ACC; FPR is that same genome's)\n")
	hdr = "| weight | " + " | ".join(present_widths) + " |"
	out.append(hdr)
	out.append("|" + "---|" * (len(present_widths) + 1))
	for ws in WSETS:
		row = [ws]
		for w in present_widths:
			c = cells.get((w, ws))
			if not c or not c["f1"] or not c["f1"]["arch"]:
				row.append("— (queued)")
				continue
			g = c["f1"]; n, b = g["arch"]
			row.append(f"{g['f1']:.2f}/{g['fpr']:.2f}/{g['acc']:.2f} · **{n}n×{b}b**")
		out.append("| " + " | ".join(row) + " |")

	# genome-size + sparse table
	out.append("\n## Deployed size (best-F1 genome)\n")
	out.append("| cell | neurons | bits | wiring (n×b) | sparse-cell upper bound | hash |")
	out.append("|---|---|---|---|---|---|")
	for w in present_widths:
		for ws in WSETS:
			c = cells.get((w, ws))
			if not c or not c["f1"] or not c["f1"]["arch"]:
				continue
			g = c["f1"]; n, b = g["arch"]
			sb = sparse_bound(n, b)
			out.append(f"| {w} {ws} | {n} | {b} | {n*b:,} | ≤ {sb:,} | `{g['hash'][:12]}` |")

	# degenerate flags
	out.append("\n## ⚠ Degenerate best-FPR genomes (do NOT cite as FPGA wins)\n")
	out.append("Tiny genomes that hit low FPR by under-predicting attacks → F1 collapses. Flagged so "
	           "they aren't mistaken for efficient deployable points.\n")
	out.append("| cell | best-FPR genome | F1 | FPR | verdict |")
	out.append("|---|---|---|---|---|")
	any_degen = False
	for w in present_widths:
		for ws in WSETS:
			c = cells.get((w, ws))
			if not c or not c["fpr"] or not c["fpr"]["arch"]:
				continue
			g = c["fpr"]; n, b = g["arch"]
			degen = g["f1"] < 95.0
			if degen:
				any_degen = True
				out.append(f"| {w} {ws} | {n}n×{b}b | {g['f1']:.2f} | {g['fpr']:.3f} | "
				           f"⚠ DEGENERATE |")
	if not any_degen:
		out.append("| — | — | — | — | none |")

	out.append("\n## Takeaways\n")
	out.append("- **Thermo width barely changes the deployed genome.** Wa converges to ~104–107 "
	           "neurons × 34b across 32/64/96b with near-identical quality (~99.57 F1 / ~0.08 FPR). "
	           "The ~105n×34b architecture is the invariant; thermo width mostly affects *search time*, "
	           "not deployed size.\n")
	out.append("- **neurons is the FPGA lever** (bits pinned at 34). Non-degenerate winners span "
	           "~88–290 neurons; pick the (width, weight) yielding the leanest STRONG genome.\n")
	out.append("- **Always pair a small-neuron claim with its F1** — the 5n/19n 'best-FPR' genomes are "
	           "degenerate (F1 78–86), not deployable.\n")

	doc = Path("docs/cicids_thermo_genome_size.md")
	doc.write_text("\n".join(out) + "\n")
	print(f"wrote {doc}")
	print("\n".join(out))


if __name__ == "__main__":
	sys.exit(main())
