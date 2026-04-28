# Per-Class Recall Integration Plan (post-r124)

**Status**: Planning only — to be implemented + tested + committed by watcher when r124 finishes.

**Goal**: Make per-attack-class recall a built-in part of every IDS flow's validation phase, so future flows automatically produce per-class data alongside the existing aggregate F1/FPR/Acc/CE.

**Decision drivers**:
- We've been computing per-class via a separate ad-hoc script (`per_class_analysis.py`) that re-trains best genomes on the fly.
- Built-in integration eliminates the ad-hoc step + GPU contention concerns.
- For paper flows (Phase D + PUB50 reruns on neto-full/neto-subsample), having per-class as a first-class metric makes draft-camera-ready trivial.

---

## Scope

| Component | Change | Risk |
|---|---|---|
| `src/wnn/ram/architecture/ids_evaluator.py` | Add `_y_test_multi` storage in `__init__` | Additive — old code paths unchanged |
| `src/wnn/ram/experiments/experiment.py` | After threshold-mode sweep, call `predict()` once + compute per-class + add to `threshold_metadata` JSON | Additive — failure mode is wrapped in try/except |
| `src/wnn/ram/strategies/accelerator/*` (Rust) | **NO change needed** — `predict_examples` already returns per-row predictions | — |
| Dashboard (`dashboard/src/*`) | **NO change needed** — `threshold_metadata` is an existing TEXT column; we just store more JSON | Forward-compatible |
| Database schema | **NO change** | — |

**Compute cost**: 1 extra `predict()` call per genome × 5 genome types per flow.
- Phase D (46.7M): ~12 min/genome × 5 = ~60 min extra per flow → ~2h over both flows
- PUB50 (1.43M): ~1.5 min/genome × 5 = ~7.5 min extra per flow → ~14h over 112 flows
- **Total: ~16h extra compute over the entire ~weeks of experiments. Acceptable.**

---

## Detailed changes

### 1. `src/wnn/ram/architecture/ids_evaluator.py`

In `IDSEvaluator.__init__`, after the `self._class_names = ...` line (around line 103), add:

```python
# For per-class breakdown during validation: stash per-row attack-class
# indices so the validation phase can group binary predictions by subclass
# without reloading the dataset. Only populated for binary classification.
self._y_test_multi = None
if classification == "binary" and hasattr(dataset, "y_test_multi") and dataset.y_test_multi is not None:
    self._y_test_multi = [int(y) for y in dataset.y_test_multi]
```

**Backward compatibility**: existing flows ignore the new attribute. Code paths using `self._y_test` are unchanged.

### 2. `src/wnn/ram/experiments/experiment.py`

**2a. New helper function** (top-level, around the imports section):

```python
def _compute_per_class_breakdown(predictions, y_test_multi, class_names):
    """Compute per-attack-class recall from binary predictions + multi-class labels.

    For each class index in y_test_multi, count how many of its rows were
    predicted as attack (binary 1). For Benign rows this rate IS the FPR;
    for attack subclasses it IS the recall (detection rate).

    Args:
        predictions: list of binary predictions (0/1), length = len(y_test_multi)
        y_test_multi: list of attack-class indices per test row
        class_names: list mapping class_index → class_name string

    Returns:
        dict {class_name: {count, predicted_attack, rate}}, omitting classes
        with zero rows.
    """
    import numpy as np
    preds = np.asarray(predictions)
    multi = np.asarray(y_test_multi)
    out = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = (multi == cls_idx)
        n = int(mask.sum())
        if n == 0:
            continue
        n_pred_attack = int(((preds == 1) & mask).sum())
        out[cls_name] = {
            "count": n,
            "predicted_attack": n_pred_attack,
            "rate": float(n_pred_attack / n),
        }
    return out
```

**2b. Per-class call site** in the validation loop (in the function around line 940-1100, inside the `else: # validate live` branch, AFTER the threshold-mode sweep finishes around line 1071-1073, BEFORE line 1078 `# Use train-calibrated as primary metric`):

```python
# Per-class breakdown — only for IDS flows (val_evaluator is IDSEvaluator)
# with multi-class info available. Adds 1 extra predict() call (~12 min on
# 46.7M training data, ~1.5 min on 1.43M).
if (val_evaluator is not None
    and getattr(val_evaluator, "_y_test_multi", None) is not None
    and getattr(val_evaluator, "_class_names", None) is not None
    and threshold_metadata is not None):
    try:
        t0 = time.time()
        per_row_preds = val_evaluator.predict(genome)
        per_class = _compute_per_class_breakdown(
            per_row_preds,
            val_evaluator._y_test_multi,
            val_evaluator._class_names,
        )
        threshold_metadata["per_class"] = per_class
        # Compact log line
        n_attack_classes = sum(1 for c in per_class if c != "Benign")
        n_low_recall = sum(1 for c, v in per_class.items()
                           if c != "Benign" and v["rate"] < 0.95)
        self.log(f"    Per-class:   {n_attack_classes} attack classes, "
                 f"{n_low_recall} below-95% recall ({time.time()-t0:.1f}s)")
    except Exception as e:
        self.log(f"    Per-class:   skipped ({e})")
```

**Required import** at top of file (if not present):
```python
import time  # likely already imported
```

---

## How the watcher applies these changes

The watcher (`scripts/auto_per_class_when_r124_done.py`) will be extended with a new step BEFORE the worker restart:

```
1. Detect r124 = completed
2. Stop worker
3. → NEW: Run scripts/apply_per_class_integration.py
        (This script edits the two files above, runs ast.parse to syntax-check,
         optionally runs a tiny end-to-end smoke test on UNSW, commits, pushes.)
   - On success: continue to step 4.
   - On failure: log error, fall back to "no per-class" path, continue to step 4.
4. Restart worker (loads new code on success, old code on fall-back)
5. Queue 2 Phase D flows on neto-full + 112 PUB50 flows on neto-subsample
6. Exit
```

The `apply_per_class_integration.py` script needs to be written before the watcher chain change. It's the ONLY new file required.

**Apply script structure**:
```python
"""Auto-apply the per-class integration plan from docs/per_class_integration_plan.md.

Edits ids_evaluator.py + experiment.py with the changes documented in the plan.
Runs ast.parse() to verify syntax. Optionally runs a tiny smoke test.
On success, commits + pushes. On failure, reverts and exits with non-zero code.
"""
# 1. Apply edit 1 to ids_evaluator.py via str.replace on the anchor line
# 2. Apply edit 2a + 2b to experiment.py via str.replace on anchor lines
# 3. ast.parse both files — fail if syntax broken
# 4. (Optional smoke test): import IDSEvaluator + run a tiny load → predict
#    on UNSW (small dataset) to verify nothing broke
# 5. git add / commit with message
# 6. git push origin main
```

---

## Rollback plan

If the watcher's apply step fails:
1. The apply script reverts file changes (git checkout HEAD -- <files>).
2. Watcher logs the failure and continues with the un-modified codebase.
3. Worker restarts with the OLD code (no per-class).
4. Flows still run with existing aggregate metrics, just no per-class.
5. User can investigate + apply manually later via `per_class_analysis.py`.

---

## Testing notes (post-implementation)

A sane smoke test for the apply script:

```bash
# Use a tiny UNSW dataset to verify load + predict + per-class math
python -c "
from wnn.ids.dataset import load_unsw_nb15
from wnn.ram.architecture.ids_evaluator import IDSEvaluator
ds = load_unsw_nb15(n_bits=4, split='temporal', feature_selection='top20')
ev = IDSEvaluator(ds)
assert ev._y_test_multi is not None, 'multi-class storage failed'
print(f'OK: {len(ev._y_test_multi)} test rows, classes: {ev._class_names}')
"
```

If this prints "OK: ..." → apply succeeded.
If it errors → apply failed; rollback.

---

## Open questions for daytime review

- Should we extend `apply_per_class_integration.py` to ALSO update the dashboard
  to display per-class info? (Lower priority — JSON is queryable via SQL.)
- Should per-class also run for cached-validation cases (re-using cached
  `threshold_metadata`)? Currently the cache contains per-class if computed
  during the original validation; subsequent cached reads include it transparently.
- Do we want a `--no-per-class` flag for fast development flows? (Optional.)
