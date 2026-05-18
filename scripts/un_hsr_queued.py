#!/usr/bin/env python3
"""Remove explicit wnn_hybrid_speed_ratio from queued HSR-cohort flows so
they'll use the worker's new predict_hsr_from_params() default. Also rename
them to drop the HSR{X} suffix.

Only touches QUEUED flows (status='queued'). Running flow keeps its explicit
HSR. Completed HSR flows keep their HSR-tagged names as historical record.

Usage:
    python3 scripts/un_hsr_queued.py            # preview
    python3 scripts/un_hsr_queued.py apply      # apply
"""
import json, re, sys, urllib.request, ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
API = "https://localhost:3000"


def fetch(p):
	with urllib.request.urlopen(f"{API}{p}", context=ctx) as r:
		return json.loads(r.read())


def patch(p, b):
	req = urllib.request.Request(f"{API}{p}", data=json.dumps(b).encode(),
		method="PATCH", headers={"Content-Type": "application/json"})
	with urllib.request.urlopen(req, context=ctx) as r:
		return json.loads(r.read())


def main():
	apply = (len(sys.argv) > 1 and sys.argv[1] == "apply")
	flows = fetch("/api/flows?limit=2000")
	hsr_queued = [
		f for f in flows
		if "OI-HSR" in f.get("name", "")
		and "CEILING" not in f.get("name", "")
		and f.get("status") == "queued"
	]
	print(f"Found {len(hsr_queued)} queued HSR flows. "
	      f"{'APPLYING' if apply else 'PREVIEW (no changes)'}.")
	print()

	for f in hsr_queued:
		old_name = f["name"]
		# Strip the -HSR{X} segment from the name
		new_name = re.sub(r"-OI-HSR\d+-r", "-OI-r", old_name)
		cfg = json.loads(json.dumps(f["config"]))
		old_hsr = cfg["params"].pop("wnn_hybrid_speed_ratio", None)
		old_desc = f.get("description", "")
		new_desc = f"OI 250n×100b (post-HSR-experiment). HSR now picked by worker's predict_hsr() function. Seed={cfg['params'].get('seed')}."

		print(f"  {f['id']}: '{old_name}' → '{new_name}' (was HSR={old_hsr})")
		if apply:
			patch(f"/api/flows/{f['id']}", {
				"name": new_name,
				"config": cfg,
				"description": new_desc,
			})

	if not apply:
		print()
		print("To apply: python3 scripts/un_hsr_queued.py apply")
	else:
		print()
		# Verify
		after = fetch("/api/flows?limit=2000")
		remaining = [f for f in after if "OI-HSR" in f.get("name", "")
		             and "CEILING" not in f.get("name", "")
		             and f.get("status") == "queued"]
		print(f"After: {len(remaining)} HSR-tagged queued flows remaining (should be 0)")
		oi_queued_plain = [f for f in after
		                   if "WSWEEP-T20-96b-C35-250n100b-OI-r" in f.get("name", "")
		                   and "OLD" not in f.get("name", "")
		                   and f.get("status") == "queued"]
		print(f"Total plain OI queued flows: {len(oi_queued_plain)}")


if __name__ == "__main__":
	main()
