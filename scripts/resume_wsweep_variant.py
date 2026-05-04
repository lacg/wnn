"""Resume previously-paused weight-sweep variant.

Flips status from 'pending' → 'queued' via /restart endpoint.

Usage:
    python scripts/resume_wsweep_variant.py CE40
"""

import sqlite3
import sys
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DASHBOARD = "https://localhost:3000"
DB_PATH = Path(__file__).resolve().parents[1] / "db" / "wnn.db"
VALID_VARIANTS = ["CE20", "F1H", "FPRH", "CE40"]

if len(sys.argv) < 2:
	print(f"Usage: {sys.argv[0]} <VARIANT>")
	sys.exit(1)

variant = sys.argv[1].upper()
if variant not in VALID_VARIANTS:
	print(f"Unknown variant: {variant}. Valid: {', '.join(VALID_VARIANTS)}")
	sys.exit(2)

con = sqlite3.connect(str(DB_PATH))
con.row_factory = sqlite3.Row
cur = con.execute(
	"SELECT id, name, status FROM flows WHERE name LIKE ? AND status = 'pending' ORDER BY id",
	(f"WSWEEP-96b-{variant}-r%",),
)
targets = list(cur)
con.close()

if not targets:
	print(f"No pending WSWEEP-96b-{variant} flows found.")
	sys.exit(0)

print(f"Resuming {len(targets)} flows for variant {variant}:")
for r in targets:
	resp = requests.post(f"{DASHBOARD}/api/flows/{r['id']}/restart", json={}, verify=False, timeout=15)
	flag = "✓" if resp.ok else "✗"
	print(f"  {flag} id={r['id']}  {r['name']}  → queued")
	time.sleep(0.5)
