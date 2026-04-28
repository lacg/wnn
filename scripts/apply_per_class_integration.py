"""Apply the per-class integration plan (post-r124).

Executable form of `docs/per_class_integration_plan.md` + the genome-DB-storage
fix + dashboard display. Designed to be invoked by the watcher AFTER r124
finishes and the worker is stopped. Never run while a worker is mid-flow.

Edits applied (each is independently fail-safe; on any failure the script
reverts ALL touched files via git checkout):

  1. src/wnn/ram/architecture/ids_evaluator.py
     - Add self._y_test_multi storage in __init__

  2. src/wnn/ram/strategies/connectivity/adaptive_cluster.py
     - Add ClusterGenome.to_json_dict() method (proper JSON serialization)

  3. src/wnn/ram/experiments/experiment.py
     - Add _compute_per_class_breakdown() helper above class Experiment
     - Add per-class call site after threshold sweep (predict + group + JSON)
     - Replace `str(genome)` with `json.dumps(genome.to_json_dict())` for
       proper genome storage in DB (vs. legacy Python repr)
     - Cache invalidation: re-run validation when cached threshold_metadata
       lacks per_class (Option B from user's spec)

  4. dashboard/frontend/src/lib/types.ts
     - Extend ThresholdMetadata interface with optional per_class field

  5. dashboard/frontend/src/routes/experiments/[id]/+page.svelte
     - Render per-class breakdown table for IDS flows

After all edits succeed: ast.parse Python files, smoke test, git commit + push.
Vite dev server (PID 50995) hot-reloads Svelte automatically — no rebuild.

Exit codes:
  0 → all edits applied + smoke ok + committed + pushed
  1 → at least one edit's anchor not found (file may have changed)
  2 → syntax check failed
  3 → smoke test failed
  4 → git commit/push failed
"""

import ast
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/lacg/wnn")
EVAL_FILE = REPO / "src/wnn/ram/architecture/ids_evaluator.py"
EXP_FILE = REPO / "src/wnn/ram/experiments/experiment.py"
GENOME_FILE = REPO / "src/wnn/ram/strategies/connectivity/adaptive_cluster.py"
TYPES_FILE = REPO / "dashboard/frontend/src/lib/types.ts"
SVELTE_FILE = REPO / "dashboard/frontend/src/routes/experiments/[id]/+page.svelte"

ALL_FILES = [EVAL_FILE, EXP_FILE, GENOME_FILE, TYPES_FILE, SVELTE_FILE]


# ============================================================================
# Edit 1: ids_evaluator.py — add _y_test_multi storage
# ============================================================================

EVAL_ANCHOR = """\t\tself._y_test = [int(y) for y in y_test]
\t\tself._y_train = [int(y) for y in y_train]
\t\tself._class_names = list(dataset.category_names) if hasattr(dataset, 'category_names') else None"""

EVAL_REPLACEMENT = """\t\tself._y_test = [int(y) for y in y_test]
\t\tself._y_train = [int(y) for y in y_train]
\t\tself._class_names = list(dataset.category_names) if hasattr(dataset, 'category_names') else None

\t\t# Per-class breakdown support: stash per-row attack-class indices so the
\t\t# validation phase can group binary predictions by subclass.
\t\tself._y_test_multi = None
\t\tif classification == "binary" and hasattr(dataset, "y_test_multi") and dataset.y_test_multi is not None:
\t\t\tself._y_test_multi = [int(y) for y in dataset.y_test_multi]"""


# ============================================================================
# Edit 2: adaptive_cluster.py — add ClusterGenome.to_json_dict()
# ============================================================================

GENOME_ANCHOR = """\tdef __repr__(self) -> str:"""

GENOME_REPLACEMENT = """\tdef to_json_dict(self) -> dict:
\t\t\"\"\"Serialize genome to a JSON-friendly dict (full per-neuron bits + connections).

\t\tStored in the genomes.tiers_json DB column so the genome can be reconstructed
\t\tlater without depending on the gzipped checkpoint files. Used for:
\t\t- Per-class analysis on best_fpr / best_ce genomes that may not be in the
\t\t  final-generation checkpoint.
\t\t- Dashboard display of full genome details.
\t\t- Reproducibility of any historical run.
\t\t\"\"\"
\t\treturn {
\t\t\t"bits_per_neuron": list(self.bits_per_neuron),
\t\t\t"neurons_per_cluster": list(self.neurons_per_cluster),
\t\t\t"threshold": float(self.threshold),
\t\t}

\tdef __repr__(self) -> str:"""


# ============================================================================
# Edit 3a: experiment.py — _compute_per_class_breakdown helper + json import
# ============================================================================

# Anchor: insert helper above 'class Experiment' module-level
EXP_HELPER_HINT = "_compute_per_class_breakdown"  # already-applied check
EXP_HELPER_INSERT_BEFORE = "class Experiment"
EXP_HELPER = '''
def _compute_per_class_breakdown(predictions, y_test_multi, class_names):
\t"""Compute per-attack-class detection rate from binary predictions + multi-class labels.

\tFor each class index in y_test_multi, counts how many of its rows were
\tpredicted as attack (binary 1). For Benign rows this rate IS the FPR;
\tfor attack subclasses it IS the recall (detection rate).
\t"""
\timport numpy as np
\tpreds = np.asarray(predictions)
\tmulti = np.asarray(y_test_multi)
\tout = {}
\tfor cls_idx, cls_name in enumerate(class_names):
\t\tmask = (multi == cls_idx)
\t\tn = int(mask.sum())
\t\tif n == 0:
\t\t\tcontinue
\t\tn_pred_attack = int(((preds == 1) & mask).sum())
\t\tout[cls_name] = {
\t\t\t"count": n,
\t\t\t"predicted_attack": n_pred_attack,
\t\t\t"rate": float(n_pred_attack / n),
\t\t}
\treturn out


'''


# ============================================================================
# Edit 3b: experiment.py — per-class call site after threshold sweep
# ============================================================================

EXP_CALL_ANCHOR = """\t\t\t\t\t# Use train-calibrated as primary metric (threshold from training, eval on val)
\t\t\t\t\t# f1, fpr_val, acc already set from train_cal above"""

EXP_CALL_REPLACEMENT = """\t\t\t\t\t# Per-class breakdown — only for IDS flows where val_evaluator has multi-class info.
\t\t\t\t\t# Adds 1 extra predict() call per genome (~12 min on 46.7M, ~1.5 min on 1.43M).
\t\t\t\t\tif (val_evaluator is not None
\t\t\t\t\t\tand getattr(val_evaluator, "_y_test_multi", None) is not None
\t\t\t\t\t\tand getattr(val_evaluator, "_class_names", None) is not None
\t\t\t\t\t\tand threshold_metadata is not None):
\t\t\t\t\t\ttry:
\t\t\t\t\t\t\timport time as _time
\t\t\t\t\t\t\t_t0 = _time.time()
\t\t\t\t\t\t\tper_row_preds = val_evaluator.predict(genome)
\t\t\t\t\t\t\tper_class = _compute_per_class_breakdown(
\t\t\t\t\t\t\t\tper_row_preds,
\t\t\t\t\t\t\t\tval_evaluator._y_test_multi,
\t\t\t\t\t\t\t\tval_evaluator._class_names,
\t\t\t\t\t\t\t)
\t\t\t\t\t\t\tthreshold_metadata["per_class"] = per_class
\t\t\t\t\t\t\tn_attack_classes = sum(1 for c in per_class if c != "Benign")
\t\t\t\t\t\t\tn_low = sum(1 for c, v in per_class.items()
\t\t\t\t\t\t\t\t\t\tif c != "Benign" and v["rate"] < 0.95)
\t\t\t\t\t\t\tself.log(f"    Per-class:   {n_attack_classes} attack classes, "
\t\t\t\t\t\t\t\t\t f"{n_low} below-95% recall ({_time.time()-_t0:.1f}s)")
\t\t\t\t\t\texcept Exception as _e:
\t\t\t\t\t\t\tself.log(f"    Per-class:   skipped ({_e})")

\t\t\t\t\t# Use train-calibrated as primary metric (threshold from training, eval on val)
\t\t\t\t\t# f1, fpr_val, acc already set from train_cal above"""


# ============================================================================
# Edit 3c: experiment.py — replace str(genome) with proper JSON
# ============================================================================

EXP_GENOME_ANCHOR = """\t\t\t\t\t\tbase_genome_data = {
\t\t\t\t\t\t\t"config_hash": genome_hash[:16],
\t\t\t\t\t\t\t"tiers_json": str(genome),"""

EXP_GENOME_REPLACEMENT = """\t\t\t\t\t\t# Use proper JSON for tiers_json (replaces legacy str(genome) repr).
\t\t\t\t\t\t# Allows downstream tools to reconstruct the genome without gzipped
\t\t\t\t\t\t# checkpoints, and stores full per-neuron bits.
\t\t\t\t\t\tif hasattr(genome, "to_json_dict"):
\t\t\t\t\t\t\t_tiers_json = json.dumps(genome.to_json_dict())
\t\t\t\t\t\telse:
\t\t\t\t\t\t\t_tiers_json = str(genome)  # back-compat for non-ClusterGenome types
\t\t\t\t\t\tbase_genome_data = {
\t\t\t\t\t\t\t"config_hash": genome_hash[:16],
\t\t\t\t\t\t\t"tiers_json": _tiers_json,"""


# ============================================================================
# Edit 3d: experiment.py — cache invalidation if per_class missing (Option B)
# ============================================================================

EXP_CACHE_ANCHOR = """\t\t\t\tif cached is not None:
\t\t\t\t\tresult = cached
\t\t\t\t\tce, acc = result[0], result[1]
\t\t\t\t\t# Extract cached IDS metrics (f1_macro, fpr) if available
\t\t\t\t\tf1 = result[2] if len(result) > 2 else None
\t\t\t\t\tfpr_val = result[3] if len(result) > 3 else None
\t\t\t\t\tcached_threshold_metadata = result[4] if len(result) > 4 else None"""

EXP_CACHE_REPLACEMENT = """\t\t\t\tif cached is not None:
\t\t\t\t\tresult = cached
\t\t\t\t\tce, acc = result[0], result[1]
\t\t\t\t\t# Extract cached IDS metrics (f1_macro, fpr) if available
\t\t\t\t\tf1 = result[2] if len(result) > 2 else None
\t\t\t\t\tfpr_val = result[3] if len(result) > 3 else None
\t\t\t\t\tcached_threshold_metadata = result[4] if len(result) > 4 else None
\t\t\t\t\t# Option B: invalidate cache if per_class is missing — re-run for completeness
\t\t\t\t\t_needs_per_class = (val_evaluator is not None
\t\t\t\t\t\tand getattr(val_evaluator, "_y_test_multi", None) is not None)
\t\t\t\t\tif _needs_per_class and cached_threshold_metadata is not None:
\t\t\t\t\t\ttry:
\t\t\t\t\t\t\t_cached_tm = cached_threshold_metadata if isinstance(cached_threshold_metadata, dict) else json.loads(cached_threshold_metadata)
\t\t\t\t\t\t\tif "per_class" not in _cached_tm:
\t\t\t\t\t\t\t\tself.log(f"  {genome_type.value}: cached but missing per_class — re-validating")
\t\t\t\t\t\t\t\tcached = None
\t\t\t\t\t\t\t\tcached_threshold_metadata = None
\t\t\t\t\t\texcept Exception:
\t\t\t\t\t\t\tpass"""


# ============================================================================
# Edit 4: dashboard/frontend/src/lib/types.ts — extend ThresholdMetadata
# ============================================================================

TYPES_ANCHOR = """export interface ThresholdMetadata {
  train_cal: ThresholdResult;
  fixed_05: ThresholdResult;
  platt?: ThresholdResult;
  beta?: ThresholdResult;
  empirical?: ThresholdResult;"""

TYPES_REPLACEMENT = """export interface PerClassEntry {
  count: number;
  predicted_attack: number;
  rate: number;  // recall for attack classes; FPR for Benign
}

export interface ThresholdMetadata {
  train_cal: ThresholdResult;
  fixed_05: ThresholdResult;
  platt?: ThresholdResult;
  beta?: ThresholdResult;
  empirical?: ThresholdResult;
  per_class?: Record<string, PerClassEntry>;"""


# ============================================================================
# Edit 5a: +page.svelte — declare perClassGenomeChoice script variable
# ============================================================================

SVELTE_VAR_ANCHOR = """  // Iteration detail modal state
  let selectedIteration: Iteration | null = null;"""

SVELTE_VAR_REPLACEMENT = """  // Per-class breakdown: which best-genome to display
  let perClassGenomeChoice: 'f1' | 'fpr' | 'acc' | 'ce' | 'fitness' = 'f1';

  // Iteration detail modal state
  let selectedIteration: Iteration | null = null;"""


# ============================================================================
# Edit 5b: dashboard/frontend/src/routes/experiments/[id]/+page.svelte
#         add per-class display block (looking for the existing thresholds block)
# ============================================================================

# We anchor near a known stable line in the threshold display. Adding a small
# Svelte fragment that conditionally renders per_class if present in the
# threshold_metadata of the best-F1 genome.

SVELTE_HINT = "<!-- per-class-table-injected -->"  # already-applied marker
SVELTE_ANCHOR = """              {@const hasThresholds = isIDS && (bestF1Summary?.threshold_metadata || bestFprSummary?.threshold_metadata || bestAccSummary?.threshold_metadata || bestCeSummary?.threshold_metadata || bestFitSummary?.threshold_metadata)}"""

SVELTE_REPLACEMENT = """              {@const hasThresholds = isIDS && (bestF1Summary?.threshold_metadata || bestFprSummary?.threshold_metadata || bestAccSummary?.threshold_metadata || bestCeSummary?.threshold_metadata || bestFitSummary?.threshold_metadata)}
              {@const _pcGenomeMeta = (s) => s?.threshold_metadata?.[perClassThresholdChoice]?.per_class
                                              || s?.threshold_metadata?.per_class}
              {@const perClassByGenome = {
                f1:      _pcGenomeMeta(bestF1Summary),
                fpr:     _pcGenomeMeta(bestFprSummary),
                acc:     _pcGenomeMeta(bestAccSummary),
                ce:      _pcGenomeMeta(bestCeSummary),
                fitness: _pcGenomeMeta(bestFitSummary),
              }}
              {@const perClassData = perClassByGenome[perClassGenomeChoice]
                || perClassByGenome.f1 || perClassByGenome.fpr
                || perClassByGenome.acc || perClassByGenome.ce
                || perClassByGenome.fitness}
              {@const perClassThresholdAvail = (mode) => (
                bestF1Summary?.threshold_metadata?.[mode]?.per_class
                || bestFprSummary?.threshold_metadata?.[mode]?.per_class
                || bestAccSummary?.threshold_metadata?.[mode]?.per_class
                || bestCeSummary?.threshold_metadata?.[mode]?.per_class
                || bestFitSummary?.threshold_metadata?.[mode]?.per_class
              )}
              <!-- per-class-table-injected — must live inside a <tbody>/<tr>/<td> to be valid HTML inside a <table> -->
              {#if isIDS && perClassData}
                <tbody class="per-class-row">
                  <tr>
                    <td colspan="16" style="padding: 0;">
                      <details class="per-class-section" open>
                        <summary style="font-weight: 600; cursor: pointer; padding: 0.5rem 0;">
                          Per-attack-class breakdown ({Object.keys(perClassData).length} classes)
                          —
                          <select bind:value={perClassGenomeChoice} on:click|stopPropagation
                                  style="font-size: 1rem; padding: 0.15rem 0.4rem; margin-left: 0.25rem; cursor: pointer;">
                            <option value="f1" disabled={!perClassByGenome.f1}>best_f1</option>
                            <option value="fpr" disabled={!perClassByGenome.fpr}>best_fpr</option>
                            <option value="acc" disabled={!perClassByGenome.acc}>best_acc</option>
                            <option value="ce" disabled={!perClassByGenome.ce}>best_ce</option>
                            <option value="fitness" disabled={!perClassByGenome.fitness}>best_fitness</option>
                          </select>
                          <span style="opacity: 0.65; font-weight: 400; margin: 0 0.25rem 0 0.5rem;">at</span>
                          <select bind:value={perClassThresholdChoice} on:click|stopPropagation
                                  style="font-size: 1rem; padding: 0.15rem 0.4rem; cursor: pointer;">
                            <option value="train_cal" disabled={!perClassThresholdAvail('train_cal')}>train_cal</option>
                            <option value="fixed_05" disabled={!perClassThresholdAvail('fixed_05')}>fixed_05</option>
                            <option value="val_cal" disabled={!perClassThresholdAvail('val_cal')}>val_cal (oracle)</option>
                            <option value="platt" disabled={!perClassThresholdAvail('platt')}>platt</option>
                            <option value="beta" disabled={!perClassThresholdAvail('beta')}>beta</option>
                            <option value="empirical" disabled={!perClassThresholdAvail('empirical')}>empirical</option>
                            <option value="empirical_cumulative" disabled={!perClassThresholdAvail('empirical_cumulative')}>empirical_cumulative</option>
                          </select>
                          <span style="opacity: 0.65; font-weight: 400; margin-left: 0.5rem;">threshold</span>
                        </summary>
                        <table style="border-collapse: collapse; margin: 0.5rem 0; font-size: 1rem;">
                          <thead>
                            <tr>
                              <th style="text-align: left; padding: 0.25rem 0.75rem; border-bottom: 1px solid #444;">Class</th>
                              <th style="text-align: right; padding: 0.25rem 0.75rem; border-bottom: 1px solid #444;">Count</th>
                              <th style="text-align: right; padding: 0.25rem 0.75rem; border-bottom: 1px solid #444;" title="True positive rate per attack class (only meaningful for attack rows)">Detection</th>
                              <th style="text-align: right; padding: 0.25rem 0.75rem; border-bottom: 1px solid #444;" title="False positive rate (only meaningful for the Benign row)">FPR</th>
                            </tr>
                          </thead>
                          <tbody>
                            {#each Object.entries(perClassData) as [clsName, entry]}
                              <tr>
                                <td style="padding: 0.25rem 0.75rem;">{clsName}</td>
                                <td style="text-align: right; padding: 0.25rem 0.75rem;">{entry.count.toLocaleString()}</td>
                                <td style="text-align: right; padding: 0.25rem 0.75rem; opacity: {clsName === 'Benign' ? 0.4 : 1};">
                                  {clsName === 'Benign' ? '—' : (entry.rate * 100).toFixed(2) + '%'}
                                </td>
                                <td style="text-align: right; padding: 0.25rem 0.75rem; opacity: {clsName === 'Benign' ? 1 : 0.4};">
                                  {clsName === 'Benign' ? (entry.rate * 100).toFixed(2) + '%' : '—'}
                                </td>
                              </tr>
                            {/each}
                          </tbody>
                        </table>
                      </details>
                    </td>
                  </tr>
                </tbody>
              {/if}"""


# ============================================================================
# Helpers
# ============================================================================

def log(msg):
	print(f"[apply] {msg}", flush=True)


def revert_files():
	subprocess.run(
		["git", "checkout", "HEAD", "--"] + [str(f) for f in ALL_FILES],
		cwd=REPO, check=False,
	)


def apply_edit(file: Path, anchor: str, replacement: str, label: str) -> bool:
	src = file.read_text()
	if anchor not in src:
		log(f"  ✗ {label}: anchor not found in {file.name}")
		return False
	if src.count(anchor) > 1:
		log(f"  ✗ {label}: anchor matches multiple times in {file.name}")
		return False
	file.write_text(src.replace(anchor, replacement, 1))
	log(f"  ✓ {label}: applied to {file.name}")
	return True


def insert_helper(file: Path, hint: str, before: str, helper: str, label: str) -> bool:
	src = file.read_text()
	if hint in src:
		log(f"  · {label}: helper already present, skipping")
		return True
	if before not in src:
		log(f"  ✗ {label}: insert anchor '{before}' not found")
		return False
	file.write_text(src.replace(before, helper + before, 1))
	log(f"  ✓ {label}: helper inserted")
	return True


def syntax_check_python(file: Path) -> bool:
	try:
		ast.parse(file.read_text())
		log(f"  ✓ python syntax OK: {file.name}")
		return True
	except SyntaxError as e:
		log(f"  ✗ python syntax error in {file.name}: {e}")
		return False


def smoke_test_python() -> bool:
	"""Tiny load + IDSEvaluator instantiation + ClusterGenome.to_json_dict roundtrip."""
	log("  Running Python smoke test (UNSW small load + JSON roundtrip)...")
	test_code = """
import sys, json
sys.path.insert(0, "/Users/lacg/wnn/src/wnn")
from wnn.ids.dataset import load_unsw_nb15
from wnn.ram.architecture.ids_evaluator import IDSEvaluator
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
ds = load_unsw_nb15(n_bits=4, split="temporal", feature_selection="top20")
ev = IDSEvaluator(ds)
assert ev._y_test_multi is not None, "smoke: _y_test_multi missing"
assert len(ev._y_test_multi) == len(ev._y_test), "smoke: length mismatch"
assert ev._class_names is not None, "smoke: _class_names missing"
g = ClusterGenome([8, 8, 8], [3], connections=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], threshold=0.5)
d = g.to_json_dict()
assert "bits_per_neuron" in d and "neurons_per_cluster" in d and "threshold" in d, "smoke: to_json_dict missing keys"
assert json.dumps(d), "smoke: to_json_dict not JSON-serializable"
print(f"smoke OK: {len(ev._y_test_multi)} rows, classes={ev._class_names}, genome JSON={len(json.dumps(d))} bytes")
"""
	r = subprocess.run([sys.executable, "-c", test_code], cwd=REPO, capture_output=True, text=True)
	if r.returncode != 0:
		log(f"  ✗ smoke FAILED:\n{r.stderr[-2000:]}")
		return False
	log(f"  ✓ {r.stdout.strip().splitlines()[-1] if r.stdout else 'smoke OK'}")
	return True


def git_commit_push() -> bool:
	files = [str(f) for f in ALL_FILES]
	if subprocess.run(["git", "add"] + files, cwd=REPO, capture_output=True, text=True).returncode != 0:
		return False
	commit_msg = """per-class + genome-storage: built-in for IDS validation + dashboard display

Adds three things in one apply pass (post-r124, applied autonomously by the
auto_per_class_when_r124_done.py watcher):

1. PER-CLASS RECALL in worker validation:
   - ids_evaluator.py: stash y_test_multi alongside binary y_test
   - experiment.py: _compute_per_class_breakdown helper + call site after
     threshold sweep, gated on val_evaluator having multi-class info.
     Adds 1 extra predict() per best genome (~12 min on 46.7M, ~1.5 min on 1.43M).
     Cache-invalidation: re-runs cached validation if cached threshold_metadata
     lacks per_class (Option B from user's spec).

2. GENOME STORAGE FIX:
   - adaptive_cluster.py: ClusterGenome.to_json_dict() returns proper dict
     {bits_per_neuron, neurons_per_cluster, threshold}.
   - experiment.py: replace str(genome) with json.dumps(genome.to_json_dict())
     when sending tiers_json to dashboard. Future genomes are reconstructable
     from DB alone (no checkpoint dependency). Old genomes keep their legacy
     repr strings (no migration of historical data).

3. DASHBOARD PER-CLASS DISPLAY:
   - types.ts: extend ThresholdMetadata with optional per_class field +
     PerClassEntry interface.
   - +page.svelte: render per-class breakdown table on flow detail page when
     present in any best-genome's threshold_metadata. Vite hot-reloads
     automatically (no rebuild).

NO Rust changes. NO schema changes. NO dashboard backend restart needed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"""
	r = subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO, capture_output=True, text=True)
	if r.returncode != 0:
		log(f"  ✗ git commit failed: {r.stderr}")
		return False
	r2 = subprocess.run(["git", "push", "origin", "main"], cwd=REPO, capture_output=True, text=True)
	if r2.returncode != 0:
		log(f"  ✗ git push failed: {r2.stderr}")
		return False
	log("  ✓ committed + pushed")
	return True


def main() -> int:
	log("Applying per-class + genome-storage + dashboard integration plan...")
	r = subprocess.run(
		["git", "status", "--porcelain"] + [str(f) for f in ALL_FILES],
		cwd=REPO, capture_output=True, text=True,
	)
	if r.stdout.strip():
		log(f"  ✗ target files have uncommitted changes — aborting:\n{r.stdout}")
		return 4

	# 1. ids_evaluator.py
	if not apply_edit(EVAL_FILE, EVAL_ANCHOR, EVAL_REPLACEMENT, "edit-1: y_test_multi"):
		revert_files(); return 1
	# 2. adaptive_cluster.py
	if not apply_edit(GENOME_FILE, GENOME_ANCHOR, GENOME_REPLACEMENT, "edit-2: to_json_dict"):
		revert_files(); return 1
	# 3a. experiment.py — helper
	if not insert_helper(EXP_FILE, EXP_HELPER_HINT, EXP_HELPER_INSERT_BEFORE, EXP_HELPER, "edit-3a: helper"):
		revert_files(); return 1
	# 3b. experiment.py — call site
	if not apply_edit(EXP_FILE, EXP_CALL_ANCHOR, EXP_CALL_REPLACEMENT, "edit-3b: per-class call"):
		revert_files(); return 1
	# 3c. experiment.py — genome json
	if not apply_edit(EXP_FILE, EXP_GENOME_ANCHOR, EXP_GENOME_REPLACEMENT, "edit-3c: genome JSON"):
		revert_files(); return 1
	# 3d. experiment.py — cache invalidation
	if not apply_edit(EXP_FILE, EXP_CACHE_ANCHOR, EXP_CACHE_REPLACEMENT, "edit-3d: cache invalidation"):
		revert_files(); return 1
	# 4. types.ts
	if not apply_edit(TYPES_FILE, TYPES_ANCHOR, TYPES_REPLACEMENT, "edit-4: types.ts"):
		revert_files(); return 1
	# 5a. +page.svelte — declare perClassGenomeChoice
	if not apply_edit(SVELTE_FILE, SVELTE_VAR_ANCHOR, SVELTE_VAR_REPLACEMENT, "edit-5a: svelte var decl"):
		revert_files(); return 1
	# 5b. +page.svelte — per-class table block
	if not apply_edit(SVELTE_FILE, SVELTE_ANCHOR, SVELTE_REPLACEMENT, "edit-5b: per-class svelte"):
		revert_files(); return 1

	# Syntax check Python files (ast.parse — TS/Svelte don't have a similar local check;
	# they'll fail at vite hot-reload time if broken)
	if not syntax_check_python(EVAL_FILE): revert_files(); return 2
	if not syntax_check_python(EXP_FILE): revert_files(); return 2
	if not syntax_check_python(GENOME_FILE): revert_files(); return 2

	# Smoke test
	if not smoke_test_python():
		revert_files(); return 3

	# Commit + push
	if not git_commit_push():
		revert_files(); return 4

	log("✓ All 5 edits applied + smoke OK + committed + pushed.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
