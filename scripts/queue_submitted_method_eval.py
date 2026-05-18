"""Queue 2 evaluation flows at SUBMITTED methodology (8b thermo, 500n×34b)
WITH bug fixes (OI training, empirical_cumulative). Compares apples-to-apples
against the submission's CIC-IoT-2023 numbers to quantify how much of the
+11pp F1 gain in the current cohort comes from the bug fixes alone vs the
encoding/architecture changes."""
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

# Template: flow 1801 (PUB50-neto-sub-ciciot-random-r112, submitted methodology)
template = fetch("/api/flows/1801")
template_exps = fetch("/api/flows/1801/experiments")

# Existing seeds to avoid
all_flows = fetch("/api/flows?limit=2000")
existing_seeds = {f.get("config",{}).get("params",{}).get("seed") for f in all_flows}
existing_seeds.discard(None)

# Build experiment specs
exp_specs = []
for e in sorted(template_exps, key=lambda x: x["sequence_order"]):
    exp_specs.append({
        "name": e["name"],
        "phase_type": e["phase_type"],
        "experiment_type": "ga" if e["phase_type"] in ("ga_neurons","ga_bits","ga_connections") else e["phase_type"],
    })

# Sample 2 fresh seeds
rng = random.Random(0x5b1)
new_seeds = []
while len(new_seeds) < 2:
    s = rng.randint(1, 99999)
    if s in existing_seeds: continue
    existing_seeds.add(s)
    new_seeds.append(s)

print(f"Template flow 1801 architecture:")
tp = template["config"]["params"]
print(f"  ids_n_bits        = {tp.get('ids_n_bits')}    (thermometer width)")
print(f"  max_neurons       = {tp.get('max_neurons')}")
print(f"  max_bits          = {tp.get('max_bits')}")
print(f"  fitness weights   = ce={tp.get('fitness_weight_ce')} f1={tp.get('fitness_weight_f1')} fpr={tp.get('fitness_weight_fpr')} acc={tp.get('fitness_weight_acc')}")
print()
print(f"New flows will OVERRIDE: wnn_order_independent_train=True (bug fix)")
print(f"  + use new HSR predict_hsr_from_params() default (silent)")
print()

created = []
for seed in new_seeds:
    cfg = json.loads(json.dumps(template["config"]))
    cfg["params"]["seed"] = seed
    cfg["params"]["wnn_order_independent_train"] = True  # OI bug fix
    # Do NOT set wnn_hybrid_speed_ratio — let worker predict
    name = f"EVAL-SUBMITTED-METHOD-8b-500n34b-OI-r{seed}"
    body = {
        "name": name,
        "description": (
            f"Evaluation flow at SUBMITTED methodology (8b thermo, 500n × 34b) "
            f"WITH bug fixes (OI training). Compares apples-to-apples against "
            f"the submission's CIC-IoT-2023 Table 5 numbers. Seed={seed}."
        ),
        "config": cfg,
        "experiments": exp_specs,
    }
    r = post("/api/flows", body)
    created.append(r.get("id"))
    print(f"Created flow id={r.get('id')} {name}")

# Transition pending → queued so worker picks them up
for fid in created:
    patch(f"/api/flows/{fid}", {"status": "queued"})
    print(f"  {fid} → queued")

print()
print("Worker will pick these up FIRST (highest ids) once it finishes the current flow.")
print("Each flow at 8b/500n×34b should run faster than the 96b/250n×100b cohort flows")
print("(smaller per-fold work — ~9× less per-example bits, despite more neurons).")
