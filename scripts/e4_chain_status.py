#!/usr/bin/env python3
"""Status for the E4 chain (scripts/e4_chain_driver.sh). Read-only.

Shows: driver log tail, per-leg markers, the live .out's latest progress line,
and every finished result summary (decay rows / truth-serum pools / committee
ENSEMBLE POOL lines).
"""

import json
import re
from pathlib import Path

OUT = Path("logs/controller/E4Chain_20260706")
LOG = Path("logs/controller/E4Chain_20260706.log")
MARKERS = ["/tmp/wnn_e4chain_lega.json", "/tmp/wnn_e4chain_legb.json", "/tmp/wnn_e4chain_done.json"]


def tail(path: Path, n: int) -> list[str]:
	if not path.exists():
		return []
	return path.read_text(errors="replace").splitlines()[-n:]


def main() -> None:
	print("=========== E4 chain status ===========")
	for m in MARKERS:
		p = Path(m)
		state = json.loads(p.read_text()).get("ts", "?") if p.exists() else "—"
		print(f"marker {p.name:<28} {state}")
	print("\n--- driver log (last 6) ---")
	for ln in tail(LOG, 6):
		print(ln)
	if not OUT.exists():
		print("\n(no output dir yet)")
		return
	outs = sorted(OUT.glob("*.out"), key=lambda p: p.stat().st_mtime)
	if outs:
		live = outs[-1]
		print(f"\n--- live: {live.name} (last 4 lines) ---")
		for ln in tail(live, 4):
			print(ln)
	print("\n--- finished summaries ---")
	for p in outs:
		txt = p.read_text(errors="replace")
		if p.name.startswith("leg_a"):
			mrows = re.findall(r"^\S+ \(\s*\d+\)\s+.*$", txt, re.M)
			if mrows:
				print(f"[{p.name}] decay matrix:")
				for r in mrows:
					print(f"  {r}")
		elif p.name.startswith("leg_b"):
			rank = txt.split("RANKING (fresh seeds)")
			if len(rank) > 1:
				print(f"[{p.name}]:")
				for ln in rank[1].splitlines()[2:16]:
					if ln.strip():
						print(f"  {ln}")
		elif p.name.startswith("leg_c"):
			for ln in txt.splitlines():
				if "ENSEMBLE POOL" in ln:
					print(f"[{p.name}] {ln.strip()}")


if __name__ == "__main__":
	main()
