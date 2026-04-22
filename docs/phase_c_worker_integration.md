# Phase C: Worker Integration Plan for Raw Datasets + invalid_encoding

**Status**: Design doc. **Do NOT execute while r98 (exp 5818) is running** — worker code changes could affect a future worker restart, and we don't want any possibility of r98 losing progress.

**When to execute**: After r98 finishes and worker is idle (or intentionally restarted).

## Goals

Plug four things into the worker pipeline so new flows can use the Phase A/B work:

1. **New HF dataset names** recognized by the worker's loader
2. **`ids_invalid_encoding` flow parameter** threaded through to `ThermometerEncoder`
3. **3-way split reactivation** (optional for this phase — gated on when we want test/val separation)
4. **Documentation updates** (`CLAUDE.md` IDS Datasets section)

## File-by-file edits

### 1. `src/wnn/ids/ciciot2023.py` — add `-raw` variant support

Currently the loader accepts `dataset_size="full"` which maps to `lacg030175/CIC-IoT-2023-full`. Add a sibling `raw=False` flag (or extend `dataset_size` to accept `"full-raw"` and `"raw"`) that maps to the new HF repos.

```python
# Around line 53-54 of ciciot2023.py
HF_DATASET_ID = "lacg030175/CIC-IoT-2023"
HF_DATASET_FULL_ID = "lacg030175/CIC-IoT-2023-full"
HF_DATASET_RAW_ID = "lacg030175/CIC-IoT-2023-raw"            # NEW
HF_DATASET_FULL_RAW_ID = "lacg030175/CIC-IoT-2023-full-raw"  # NEW

# In load_ciciot2023(), after dataset_size check:
def load_ciciot2023(..., dataset_size: str = "subsample", raw: bool = False, ...):
    if raw and dataset_size == "full":
        repo_id = HF_DATASET_FULL_RAW_ID
    elif raw and dataset_size == "subsample":
        repo_id = HF_DATASET_RAW_ID
    elif dataset_size == "full":
        repo_id = HF_DATASET_FULL_ID
    else:
        repo_id = HF_DATASET_ID
```

### 2. `src/wnn/ids/cicids2017.py` — add raw variant

Analogous: add `HF_DATASET_RAW_ID = "lacg030175/CICIDS2017-raw"` and `raw: bool = False` flag to `load_cicids2017`.

### 3. `src/wnn/ids/dataset.py` (`load_unsw_nb15`) — no change needed

UNSW dataset already preserves NaN. Just ensure the `invalid_encoding` param flows to the encoder (step 4 below).

### 4. `src/wnn/ids/encoder.py` — already done (commit `2b61dd3a`)

`ThermometerEncoder(invalid_encoding=...)` is already live with Option A (single_bit). Nothing to change here.

### 5. `src/wnn/ram/experiments/worker.py` — thread params through

Currently `worker.py` reads these params (around line 820-850):

```python
n_bits = params.get("ids_n_bits", 8)
val_fraction = params.get("ids_val_fraction", 0.25)
# ... etc ...
auto_max_bits = params.get("ids_auto_max_bits", 32)
```

**Add**:

```python
raw_dataset = params.get("ids_raw", False)  # NEW — opt into raw HF variant
invalid_encoding = params.get("ids_invalid_encoding", "none")  # NEW — encoder param
```

Then in the dataset-loading dispatch (around line 843-855):

```python
if dataset_name == "ciciot2023":
    from wnn.ids.ciciot2023 import load_ciciot2023
    full_dataset = load_ciciot2023(n_bits=n_bits, split=split,
                                   feature_selection=feature_selection,
                                   raw=raw_dataset,  # NEW
                                   invalid_encoding=invalid_encoding)  # NEW
elif dataset_name == "ciciot2023_full":
    full_dataset = load_ciciot2023(n_bits=n_bits, split=split,
                                   feature_selection=feature_selection,
                                   dataset_size="full",
                                   raw=raw_dataset,  # NEW
                                   invalid_encoding=invalid_encoding)  # NEW
# ... similarly for cicids2017 and unsw-nb15 ...
```

### 6. `src/wnn/ids/{ciciot2023,cicids2017,dataset}.py` loaders — accept invalid_encoding

Each loader creates a `ThermometerEncoder`. Pass the param through:

```python
# In each loader:
def load_ciciot2023(..., invalid_encoding: str = "none", ...):
    ...
    encoder = ThermometerEncoder(n_bits=n_bits, method=method_enum,
                                 auto_max_bits=auto_max_bits,
                                 invalid_encoding=invalid_encoding)  # NEW
    encoder.fit(df_train)
    # ... (transform calls same as today)
```

### 7. Dashboard flow-creation API — accept new params

`dashboard/src/api/mod.rs` (or wherever flow params are validated) — accept optional:
- `ids_raw: bool` (default false)
- `ids_invalid_encoding: str` (default "none", valid: "none" | "single_bit")

Or leave the API as-is and accept free-form params via the existing `params` HashMap (minimum change).

## Flow creation recipe (after integration)

Once worker is restarted with the edits above, creating a new flow with raw + single_bit looks like:

```python
flow_config = {
    "template": "ids-binary-2-phase",
    "params": {
        "ids_dataset": "ciciot2023_full",
        "ids_raw": True,                         # NEW — use full-raw HF variant
        "ids_invalid_encoding": "single_bit",    # NEW — Option A encoder
        "ids_split": "random",                   # or "random_3way"
        "ids_n_bits": 8,
        # ... other params unchanged ...
    }
}
```

## Rollout plan (post-r98)

1. Stop the worker gracefully (`kill -TERM`, not -9)
2. Apply all edits above in a single commit
3. Run `cargo-check-hook.sh` if applicable (Rust crate sanity)
4. Start worker
5. Smoke-test: create one flow with `ids_raw=False` (existing dataset) — verify back-compat
6. Create one flow with `ids_raw=True, ids_invalid_encoding="single_bit"` — verify raw path
7. If both pass, create production flows

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Back-compat break for existing flows | All new params have defaults matching current behavior; no existing flow config changes |
| Worker restart kills in-flight flow | Drain worker queue before restart; we're already gated on r98 completion |
| New HF datasets have upload errors | CICIDS + CIC-IoT 1.3M verified loading with row counts matching expected; 46M verifies post-upload |
| Encoder bug corrupts training | Unit tests in `tests/test_encoder_invalid.py` cover NaN, ±Inf, binary/categorical NaN, fit robustness |

## References

- **Phase A commit** `2b61dd3a`: ThermometerEncoder `invalid_encoding` param + 9 unit tests
- **Phase B commit** `544a849e`: three `create_*_raw_dataset.py` scripts
- **Memories**: `project_positioning_vs_pruning.md` (why raw matters for camera-ready)
