# `wnn` public API — draft for review (12/09/2026)

Status: DRAFT, no code. Scope agreed with Luiz 12/09: **A + B + C** — a scikit-learn estimator,
a thin PyPI package over the Rust core, and `ram_core` on crates.io. Release with the paper
(camera-ready), built now.

Everything below is grounded in what exists: cell modes are `core/neuron_memory.rs` (u8 codes
0–5), the GPU forward is `core/metal_sparse.rs::forward_batch_sparse`, the encoder is
`wnn/representations/thermometer.py::ThermometerEncoder`, the export is `SparseGpuExport`.
The public names are new; the semantics are not.

---

## 1. What ships, what does not

| ships | does not ship |
|---|---|
| `ram_core` crate (cells, packed bits, sparse memory, forward) | `ram_accelerator` (GA evaluators, training kernels with atomics, LM) |
| `wnn` PyPI wheel: `CellMode`, `WiSARDClassifier`, `ThermometerEncoder`, key export | `ram_controller`, the drone stack, the dashboard, the worker |
| CPU forward (rayon) everywhere; GPU forward on Apple silicon now, wgpu later | GA/TS/SA connectivity search (a separate package if ever) |

Dependencies of the public wheel: `numpy`, `scikit-learn`. **No torch, no datasets, no tiktoken.**
The current `pyproject.toml` (`ram-wnn`) is the research monorepo and stays as it is; the public
package is a second, thin `pyproject` under `packages/wnn/` (name TBD — see §7).

---

## 2. `CellMode` — the ladder

One enum, six members, ordered by information per cell. Internal codes are the existing u8s.

| public name | code | states | read | training | `empty_value` | `seed` |
|---|---|---|---|---|---|---|
| `BINARY` | 3 | 1 bit: FALSE/TRUE | TRUE→1.0 else 0.0 | one-shot own-class set-TRUE, negatives ignored (classical WiSARD / n-tuple) | unused | unused |
| `TERNARY` | 0 | FALSE / u / TRUE | FALSE→0, TRUE→1, u→`empty_value` | majority vote, negatives write FALSE | used (default 0.5) | unused |
| `PLN` | 5 | FALSE / u / TRUE | as TERNARY, but u fires a **fair coin** | as TERNARY | unused (coin is 0.5 by design) | used |
| `QUAD_BINARY` | 1 | 4-state nudging | cell ≥ WEAK_TRUE → 1.0 else 0.0 | nudge lattice, WEAK_FALSE initial | unused | unused |
| `QUAD_WEIGHTED` | 2 | 4-state nudging | graded 0 / .25 / .75 / 1 | as QUAD_BINARY | unused | unused |
| `QSR` | 4 | 4-state nudging | fires a coin with p = graded weight | as QUAD_BINARY | unused | used |

Documented invariants (they are already true in Rust, the doc just states them):
- `PLN` ≡ `TERNARY@0.5` and `QSR` ≡ `QUAD_WEIGHTED` **in expectation**; the difference is sampling
  variance. Same `seed` → identical output (hash-PRNG per read, `qsr_key`/`qsr_hash`).
- Physical packing is 2 bits/cell for all modes; `BINARY` is *semantically* 1 bit and the FPGA
  sizing (`export_keys`) counts it that way.
- Default: `QUAD_WEIGHTED` (the project default). `BINARY` is documented as "classical WiSARD" and is
  the mode a `wisardpkg` user expects — first thing in the README's "coming from wisardpkg" note.

Open: expose `QUAD_BINARY` publicly, or keep it internal? It is an ablation arm, not a mode anyone
would choose. My call: keep it, mark it `# ablation` in the docstring — hiding a member means two
enums.

---

## 3. `WiSARDClassifier` — the scikit-learn surface

```python
class WiSARDClassifier(BaseEstimator, ClassifierMixin):
	def __init__(
		self,
		neurons_per_class: int = 50,
		bits_per_neuron: int = 16,
		cell_mode: CellMode = CellMode.QUAD_WEIGHTED,
		empty_value: float = 0.5,          # TERNARY only
		connections: np.ndarray | None = None,   # (n_classes, neurons, bits) int; None = random from random_state
		coverage_aware: bool = False,
		backend: Backend = Backend.AUTO,   # AUTO | CPU | GPU  (GPU = Metal today, wgpu later)
		n_jobs: int | None = None,         # rayon threads; None = all cores
		random_state: int | None = None,   # connectivity draw AND the PLN/QSR coin seed
	)
	def fit(self, X: np.ndarray[bool | uint8], y) -> Self
	def partial_fit(self, X, y, classes=None) -> Self      # RAM writes accumulate — natural fit
	def predict(self, X) -> np.ndarray                     # argmax of per-class vote
	def predict_proba(self, X) -> np.ndarray               # per-class vote / neurons_per_class, row-normalised
	def decision_function(self, X) -> np.ndarray           # raw per-class vote (n, n_classes); binary → margin
	def export_keys(self) -> KeyExport                     # TRUE-only sorted keys per neuron: the H743 / FPGA path
	# attributes after fit
	classes_, n_features_in_, connections_, memory_  (opaque handle), fill_fraction_
```

Rules baked in:
- `X` is **bits**. The estimator does not encode. Encoding is `ThermometerEncoder` (§4) in a
  `Pipeline`, exactly like `StandardScaler` before an SVM. Passing floats raises.
- `fit` = reset + `partial_fit`. `partial_fit` is the honest primitive: RAM training is a write, so
  streaming/warm-start is free (this is what fold-accumulation already relies on).
- **Thresholds are not the estimator's job.** `predict` is argmax; `decision_function` gives the
  margin; calibration is sklearn's `CalibratedClassifierCV` (Platt/isotonic) or the user's own
  threshold on `decision_function`. Our seven IDS threshold modes stay in the research code; the
  paper's `val_cal` row is reproducible as "threshold on `decision_function` chosen on a validation
  split", which is one line of sklearn.
- `predict_proba` for PLN/QSR is the **expected** read (deterministic) so probabilities are stable;
  `predict` uses the stochastic read seeded by `random_state`. Documented, because it is the one
  place expectation and sample differ.
- `n_jobs`/`backend` are runtime, not model state — `get_params` returns them but `clone` with a
  different backend must give identical predictions (parity test).

`export_keys()` returns the exact structure `scripts/export_controller_c.py` / `count_true_keys.py`
compute today (TRUE-only sorted keys, uint32/packed byte counts) so the "fits the H743" number a
user gets is the one we publish.

---

## 4. `ThermometerEncoder` as a Transformer

It already has `fit(df)` / `transform(df)`; the public version adds `TransformerMixin` and accepts a
numpy array or DataFrame:

```python
class ThermometerEncoder(BaseEstimator, TransformerMixin):
	def __init__(self, n_bits: int = 8, method: ThermometerType = DISTRIBUTIVE,
	             feature_config: dict[str, str] | None = None)
	def fit(self, X, y=None) -> Self
	def transform(self, X) -> np.ndarray[uint8]     # (n, total_bits)
	n_bits_out_, bit_ranges_
```

The streaming t-digest path (`pytdigest`) stays optional (`extras = ["streaming"]`).

Pipeline the README opens with:

```python
clf = make_pipeline(ThermometerEncoder(n_bits=8),
                    WiSARDClassifier(neurons_per_class=50, bits_per_neuron=48))
cross_val_score(clf, X, y, cv=5)
```

That line is the whole adoption argument: the RF/XGB comparison in the paper becomes three lines
for a reviewer.

---

## 5. Rust — `ram_core` and the `GpuForward` boundary

`ram_core` is already the domain-free rlib (cells, `PackedBits`, `SparseLayerMemory`,
`SparseGpuExport`, `metal_sparse`). Publishing it means two changes, both boundary work:

**5a. Feature-gate Metal.** Today `metal_sparse` is `#[cfg(target_os = "macos")]` with a stub
elsewhere. Make it a cargo feature (`gpu-metal`, default on macOS) and add `gpu-wgpu` later, so a
Linux `cargo add ram_core` builds without a stub that lies.

**5b. One trait for the forward.** The Metal call is a 16-argument function. The public boundary is
the same contract, named:

```rust
pub struct ForwardRequest<'a> {
	pub input: &'a PackedBits,          // rows of packed bits
	pub connections: &'a [i64],         // flat (neurons × bits_per_neuron)
	pub memory: &'a SparseGpuExport,    // sorted keys / values / offsets / counts
	pub layout: ClusterLayout,          // neurons_per_cluster, num_clusters, bits_per_neuron
	pub read: ReadParams,               // cell_mode: CellMode, empty_value: f32, coverage_aware: bool, run_seed: u64
}

pub trait Forward {
	fn forward(&self, req: &ForwardRequest) -> Result<Vec<f32>, ForwardError>;   // (n_examples × num_clusters)
}

pub struct CpuForward;                     // rayon; the existing sparse_memory::forward_batch_sparse
#[cfg(feature = "gpu-metal")] pub struct MetalForward;   // the existing MetalSparseEvaluator
#[cfg(feature = "gpu-wgpu")]  pub struct WgpuForward;    // the port (§6)
```

`ReadParams.cell_mode` is a real enum (`CellMode`, `#[repr(u8)]` with the six codes) — the u8
constants stay as its discriminants so nothing internal moves. `forward_batch_general` (tiered
clusters) is the same trait with `ClusterLayout::Tiered(...)`; one trait, two layouts.

Training is **not** on the trait. `SparseLayerMemory::write_cell` + `train_batch_sparse` are CPU
(DashMap) and documented as such — the GPU only ever reads sorted arrays. That is the existing
architecture, now stated on the public surface.

Parity: `cpu_fallback_matches_gpu` becomes the trait test — every `Forward` impl must match
`CpuForward` bit-for-bit at a fixed `run_seed`, all six modes.

---

## 6. Portable GPU — the wgpu port, scoped

Port scope is the public forward only: `common.metal` (284 lines) + `sparse_forward.metal` (564)
+ `marker_slots.metal` (218) → one WGSL module. **No atomics** in any of them (checked 12/09).

- Keys: store `(hi: u32, lo: u32)` pairs in the wgpu upload path and compare two-step in the binary
  search. Avoids `SHADER_INT64` entirely; works on Vulkan/DX12/Metal **and WebGPU**.
- `WgpuForward` **replaces** `MetalForward` once it passes parity and a benchmark on our Macs (wgpu
  runs on Metal underneath). One kernel, not two — the no-duplicates rule. Until then both exist
  behind features and the benchmark decides.
- `ram_accelerator` training kernels and `controller_rollout.metal` stay Metal — research only.

Sequencing: v0 ships CPU + Metal ("GPU on Apple silicon; CPU elsewhere"). wgpu is the first
post-v0 item, ~1–2 days, done in a worktree; it touches `ram_core` so it lands at a wheel-swap
window (worker idle + HOLD sentinel), never mid-chain.

---

## 7. Decisions needed from Luiz

1. **Name.** `wnn` on PyPI if free; fallback `ramwnn` / `weightless`. Crate name `ram_core` is fine
   on crates.io if free, else `ram-wnn-core`.
2. **Default `cell_mode`** — `QUAD_WEIGHTED` (project default, proposed) vs `BINARY` (what a
   WiSARD user expects). I propose QUAD_WEIGHTED default + a loud "classical = BINARY" note.
3. **Expose `QUAD_BINARY`?** Proposed yes, marked ablation.
4. **Multi-class from day one** (per-class discriminators, argmax) — proposed yes; binary is the
   n_classes=2 case, no special path.
5. **Versioning**: CalVer like the crates (`2026.9.x`) or SemVer `0.1.0`? SemVer is what PyPI users
   read; proposed `0.1.0`.
6. **Release timing**: build now on a branch, publish at camera-ready (S&P double-blind).
7. **License**: MIT (already). Author line and citation entry (the paper) in the README.

---

## 8. Effort, in order

| step | what | size |
|---|---|---|
| 1 | `CellMode` enum in `ram_core` (`#[repr(u8)]`), u8 consts become discriminants; `Forward` trait + `CpuForward`/`MetalForward` impls wrapping existing functions; feature gate | small |
| 2 | `packages/wnn/`: maturin abi3 wheel over `ram_core` with a minimal PyO3 module (`forward`, `train`, `export_keys`, `PackedBits` from numpy) | small–medium |
| 3 | `WiSARDClassifier` + `ThermometerEncoder(TransformerMixin)` + sklearn's `check_estimator` suite passing | medium |
| 4 | README (pipeline example, "coming from wisardpkg", the ladder table), CI wheels (macOS arm64, Linux x86_64/aarch64, Windows) | small |
| 5 | wgpu forward (§6) | 1–2 days, post-v0 |

Nothing here changes research behaviour: steps 1–2 wrap existing functions; the parity test is the
gate. No launch, no wheel swap, until reviewed.
