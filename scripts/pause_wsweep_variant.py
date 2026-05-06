"""Pause all queued flows of a weight-sweep variant.

Sets status from 'queued' → 'pending' so the worker skips them but they
stay alive (no IDs lost). Use /restart later to resume.

Usage:
    python scripts/pause_wsweep_variant.py CE40
    python scripts/pause_wsweep_variant.py FPRH

Variants: CE20, F1H, FPRH, CE40
Pass --all-but-running to pause everything still queued (keeps the
currently running flow going).
"""

import sqlite3
import sys
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DASHBOARD = "https://localhost:3000"
DB_PATH = Path(__file__).resolve().parents[1] / "db" / "wnn.db"
VALID_VARIANTS = ["CE20", "F1H", "FPRH", "CE40", "DDR", "C35", "F40", "TRI",
                  "B05-CE", "B05-AC", "B10-CE", "B10-AC", "B15-CE", "B15-AC"]

if len(sys.argv) < 2:
	print(f"Usage: {sys.argv[0]} <VARIANT>")
	print(f"Variants: {', '.join(VALID_VARIANTS)}")
	sys.exit(1)

variant = sys.argv[1].upper()
if variant not in VALID_VARIANTS:
	print(f"Unknown variant: {variant}. Valid: {', '.join(VALID_VARIANTS)}")
	sys.exit(2)

# Find queued flows of this variant
con = sqlite3.connect(str(DB_PATH))
con.row_factory = sqlite3.Row
cur = con.execute(
	"SELECT id, name, status FROM flows WHERE name LIKE ? AND status = 'queued' ORDER BY id",
	(f"WSWEEP-96b-{variant}-r%",),
)
targets = list(cur)
con.close()

if not targets:
	print(f"No queued WSWEEP-96b-{variant} flows found.")
	sys.exit(0)

print(f"Pausing {len(targets)} flows for variant {variant}:")
for r in targets:
	resp = requests.patch(
		f"{DASHBOARD}/api/flows/{r['id']}",
		json={"status": "pending"},
		verify=False,
		timeout=15,
	)
	flag = "✓" if resp.ok else "✗"
	print(f"  {flag} id={r['id']}  {r['name']}  → pending")

print()
print(f"To resume: python scripts/resume_wsweep_variant.py {variant}")
