"""IDS-65 (lever D) pre-check, 22/11/2026: trained cells per neuron vs neuron width on a flow's real encoding.

Re-trains the flow's GA best_f1 genome (prints stored vs re-trained footprint), then the same genome with
every neuron cut to its first b evolved connections, and a random-connection genome at b. Cells are the keys
training writes on the 80% TRAIN (the footprint); no val/test metric is read or used.

  PYTHONPATH=<tree>/src python scripts/ids65_cells_per_neuron_precheck.py 7600 34,24,20,16
"""
import json, sqlite3, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ids57_abi_equivalence import apply_production_env  # noqa: E402
from ids_val_rescore_phases import genome_for, stored_cell  # noqa: E402
from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

db = sqlite3.connect("file:/Users/lacg/wnn/db/wnn.db?mode=ro", uri=True)
fid = int(sys.argv[1])
widths = [int(x) for x in sys.argv[2].split(",")]
params = json.loads(db.execute("select config_json from flows where id=?", (fid,)).fetchone()[0])["params"]
apply_production_env(params)
s = IdsEvaluatorSettings.from_params(params)
fac = IdsEvaluatorFactory(lambda m: None)
ds = fac.load(s)
_, test = fac.flat(ds, s)
_, h = stored_cell(db, fid, "ga_neurons")
g = genome_for(db, fid, h)
stored = db.execute("select logical_cells from test_summaries where genome_hash=? and flow_id=? and test_point='final' and genome_type='best_f1'", (h, fid)).fetchone()[0]


def footprint(genome):
	return test.evaluate_multiclass_at_thresholds(genome)["footprint"]


fp = footprint(g)
print(f"{fid} stored GA best_f1 cells {stored} | re-trained {fp.logical_cells} neurons {fp.logical_neurons} bits {min(g.bits_per_neuron)}-{max(g.bits_per_neuron)}", flush=True)
# per-neuron connection lists of the evolved genome
offs = np.cumsum([0] + list(g.bits_per_neuron))
conns = [g.connections[offs[i]:offs[i + 1]] for i in range(len(g.bits_per_neuron))]
rng = np.random.default_rng(0)
total_bits = ds.X_train.total_bits
for b in widths:
	# (a) evolved: each neuron keeps the first b of its own evolved connections
	ev = ClusterGenome(bits_per_neuron=[min(b, len(cl)) for cl in conns], neurons_per_cluster=list(g.neurons_per_cluster),
		connections=[int(c) for cl in conns for c in cl[:b]])
	# (b) random: b distinct random input bits per neuron
	rd = ClusterGenome(bits_per_neuron=[b] * len(conns), neurons_per_cluster=list(g.neurons_per_cluster),
		connections=[int(c) for _ in conns for c in rng.choice(total_bits, b, replace=False)])
	fe, fr = footprint(ev), footprint(rd)
	print(f"  b={b:2d}: evolved-prefix {fe.logical_cells:8d} cells = {fe.logical_cells / fe.logical_neurons:7.1f}/neuron | "
		f"random {fr.logical_cells:8d} = {fr.logical_cells / fr.logical_neurons:7.1f}/neuron", flush=True)
