"""Queue 2 ceiling-test flows: 500n × 100b OI cohort, HSR=10."""
import json, urllib.request, ssl, random

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

def post(p, b):
    req = urllib.request.Request(f"{API}{p}", data=json.dumps(b).encode(),
        method="POST", headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, context=ctx) as r:
        return json.loads(r.read())

# Pull template from 2671 (OI-OLD canonical OI flow)
template = fetch("/api/flows/2671")
template_exps = fetch("/api/flows/2671/experiments")

# Get all existing seeds to avoid collision
all_flows = fetch("/api/flows?limit=2000")
existing_seeds = {f.get("config",{}).get("params",{}).get("seed") for f in all_flows}
existing_seeds.discard(None)

# Build experiment specs (preserve grid_search + ga_neurons phases)
exp_specs = []
for e in sorted(template_exps, key=lambda x: x["sequence_order"]):
    exp_specs.append({
        "name": e["name"],
        "phase_type": e["phase_type"],
        "experiment_type": "ga" if e["phase_type"] in ("ga_neurons","ga_bits","ga_connections") else e["phase_type"],
    })

rng = random.Random(20260518)
new_seeds = []
while len(new_seeds) < 2:
    s = rng.randint(1, 99999)
    if s in existing_seeds: continue
    existing_seeds.add(s)
    new_seeds.append(s)

created = []
for seed in new_seeds:
    cfg = json.loads(json.dumps(template["config"]))
    # Override architecture ceiling
    cfg["params"]["max_neurons"] = 500
    cfg["params"]["min_neurons"] = 5
    cfg["params"]["max_bits"] = 100
    cfg["params"]["min_bits"] = 4
    # OI + HSR
    cfg["params"]["wnn_order_independent_train"] = True
    cfg["params"]["wnn_hybrid_speed_ratio"] = 10
    cfg["params"]["seed"] = seed
    name = f"WSWEEP-T20-96b-C35-500n100b-CEILING-OI-HSR10-r{seed}"
    body = {
        "name": name,
        "description": f"Architecture ceiling test (500n × 100b, OI, HSR=10). Tests whether the GA wants more headroom than 250n × 100b under OI training. Seed={seed}.",
        "config": cfg,
        "experiments": exp_specs,
    }
    r = post("/api/flows", body)
    created.append(r.get("id"))
    print(f"Created flow id={r.get('id')} name={name} max_neurons=500 max_bits=100 HSR=10")

# Newly-created flows start as 'pending' — transition to 'queued' so worker picks them up
for fid in created:
    patch(f"/api/flows/{fid}", {"status": "queued"})
    print(f"  {fid} → queued")

print()
print(f"2 ceiling flows queued. Worker will pick them up FIRST (highest IDs).")
