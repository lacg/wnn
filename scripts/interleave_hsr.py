#!/usr/bin/env python3
"""Reassign the HSR cohort flows to round-robin interleaved order.

For 8 seeds × 7 ratios = 56 queued flows tagged with HSR in the name, walks
them in id-DESC order (worker pickup order) and reassigns each to
HSR_VALUES[i % 7]. Result: every 7 consecutive worker pickups span all
ratios once, giving the earliest possible cross-ratio signal as data lands.

Renames flows and updates wnn_hybrid_speed_ratio in config in-place; flows
that already match the target HSR are skipped.
"""
import json, urllib.request, ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
API = "https://localhost:3000"

def fetch(p):
    with urllib.request.urlopen(f"{API}{p}", context=ctx) as r:
        return json.loads(r.read())

def patch(p, b):
    req = urllib.request.Request(f"{API}{p}", data=json.dumps(b).encode(),
        method="PATCH", headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, context=ctx) as r:
        return json.loads(r.read())

HSR_VALUES = [1, 2, 3, 5, 7, 8, 10]

flows = fetch("/api/flows?limit=2000")
hsr_queued = sorted([f for f in flows
                     if "HSR" in f.get("name","") and f.get("status")=="queued"],
                    key=lambda x: -x["id"])

print(f"Found {len(hsr_queued)} queued HSR flows (expecting 56).")
print()

# Compute target HSR for each by round-robin (i % 7) over id-DESC order
to_update = []
for i, f in enumerate(hsr_queued):
    target_hsr = HSR_VALUES[i % len(HSR_VALUES)]
    current_hsr = f["config"]["params"].get("wnn_hybrid_speed_ratio")
    if current_hsr != target_hsr:
        to_update.append((f, target_hsr))

print(f"Flows needing reassignment: {len(to_update)}")
print()

# Show first round + sample for verification
print("New assignment (first 14 flows = first 2 rounds):")
for i, f in enumerate(hsr_queued[:14]):
    target = HSR_VALUES[i % len(HSR_VALUES)]
    print(f"  pos={i} id={f['id']} → HSR={target}")
print("...")
print()

# Apply
for f, hsr in to_update:
    seed = f["config"]["params"].get("seed")
    new_name = f"WSWEEP-T20-96b-C35-250n100b-OI-HSR{hsr}-r{seed}"
    cfg = json.loads(json.dumps(f["config"]))
    cfg["params"]["wnn_hybrid_speed_ratio"] = hsr
    patch(f"/api/flows/{f['id']}", {
        "name": new_name,
        "config": cfg,
        "description": f"OI 250n×100b with WNN_HYBRID_SPEED_RATIO={hsr} (HSR sweep, round-robin interleaved, n=8 per ratio). Seed={seed}.",
    })

# Verify final counts + ordering
from collections import Counter
after = fetch("/api/flows?limit=2000")
hsr_after = sorted([f for f in after
                    if "HSR" in f.get("name","") and f.get("status")=="queued"],
                   key=lambda x: -x["id"])
print(f"Updated {len(to_update)} flows. Final counts per HSR:")
counts = Counter()
for f in hsr_after:
    counts[f["config"]["params"].get("wnn_hybrid_speed_ratio")] += 1
for h in HSR_VALUES:
    print(f"  HSR={h:>2}: {counts[h]} flows")

# Show id-DESC sequence (first 14) — should be the perfect round-robin now
print()
print("Worker pickup order (first 14 = first 2 rounds, id DESC):")
for f in hsr_after[:14]:
    p = f["config"]["params"]
    print(f"  id={f['id']} HSR={p.get('wnn_hybrid_speed_ratio'):>2} seed={p.get('seed')}")
