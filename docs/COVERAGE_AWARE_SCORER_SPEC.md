# Coverage-aware scorer — design spec (DRAFT, 28/08/2026)

Status: **IMPLEMENTED, built, NOT YET DEPLOYED** (28/08/2026). Decisions in §6 are
settled (Luiz): rule = `p0=0.0, kappa=m_c` (minimal); scope = **search AND decode**.
Flag: flow param `ids_coverage_aware` (default `false` = bit-exact today).
Wheel is NOT swapped — the worker is mid-cohort; swap at idle per §8.

## 1. The defect (verified in code and in data, 28/08/2026)

Per-class score today (`adaptive/eval_export.rs:116`, `:211`):

```
s_c(x) = ( Σ_{j=1..n_c} w(cell_{c,j}[addr_j(x)]) ) / n_c
```

with `w = QUAD_WEIGHTS[cell] = [0.0, 0.25, 0.75, 1.0]`.

`oi_bin_to_cell` commits an **untouched** cell (`obs == 0`) to `QUAD_WEAK_FALSE`
= **0.25**, while a class that **learned to reject** the row (`obs >= 2,
net <= -1`) commits to `QUAD_FALSE` = **0.0**.

The score is a mean over the class's own neurons, so neuron count gives no
protection: a class whose memory is entirely untouched scores a flat **0.25**
whatever its size. Raw argmax (`predict_genome_hybrid`) then hands it every row
where the true class's mean falls below 0.25.

**Ignorance outranks knowledge.** Confident rejection is scored *below*
never having learned anything.

Measured consequence (UNSW temporal_3way, flows 6000-6006, 7 runs): a class
absorbs misclassifications in inverse proportion to its train support.
Over-absorption = (share of all errors absorbed) / (share of the test set):

| destination | train sup | over/under | | destination | train sup | over/under |
|---|---|---|---|---|---|---|
| Worms | 97 | **508.3x** | | Recon | 7868 | 1.3x |
| Shellcode | 850 | 13.0x | | DoS | 9198 | 0.5x |
| Analysis | 1500 | 9.1x | | Exploits | 25045 | 0.4x |
| Backdoor | 1309 | 4.6x | | Normal | 42000 | 0.4x |
| Fuzzers | 13638 | 2.4x | | Generic | 30000 | 0.3x |

Spearman rho(train support, over-absorption) = **-0.930** (n=10, t=-7.16, df=8).
Companion, victim-side: rho(own recall, leak into Worms) = **-0.900** (n=9).

This is NOT a QUAD bug — it is a decode that conflates *evidence* with
*ignorance*. It is why the four smallest classes (Worms F1 0.98%, Analysis
1.28%, Backdoor 7.59%, DoS 11.70%) sit at the floor while occupying 40% of
macro-F1's weight.

## 2. Why the signal is free (sparse) and absent (dense)

`SparseExport::lookup(neuron_in_group, address, miss_default)` performs a binary
search and, **on a miss, substitutes `miss_default`** = `default_cell_for_mode`
(QUAD -> cell 1 -> 0.25). The hit/miss bit is computed and then discarded.

- **Sparse groups (bits > 12):** coverage is already known. Cost of keeping it:
  one bool per neuron-lookup. tiered3 (b=34-50) is entirely sparse.
- **Dense groups (bits <= 12):** the cell is read straight out of the packed
  word; untouched is indistinguishable from a learned tie. `oi_bin_to_cell`
  maps `obs==0`, `obs==1 & net<0`, and `obs>=2 & net==0` ALL to WEAK_FALSE, so
  it cannot be recovered post-hoc either. Dense needs an explicit coverage
  bitmap (1 bit/cell, <= 512 B/neuron at b=12) or must keep current behavior.

`GroupMemory::neuron_fill_rate` ("fraction of a neuron's addresses that are
non-EMPTY") already exists, so the concept is not foreign to the codebase.

Note `default_cell_for_mode` already returns **0 for BINARY** ("unwritten = no
vote"). This proposal gives QUAD the semantics BINARY already has.

## 3. The parameterization

Split the sum into evidence and ignorance. Let `h_c` = number of neurons whose
address **hit**, `m_c = n_c - h_c` = misses.

```
s_c(x) = ( Σ_{hits} w_j  +  m_c * p0 ) / n_c
```

- **`p0 = 0.25` reproduces today's behavior bit-exactly** (the current code is
  this with p0 pinned at WEAK_FALSE). So the change is a strict generalization
  and can ship default-OFF.
- `p0 = 0.0` is the minimal fix: a miss is *no vote*, not a quarter-vote.

A richer form treats it as a posterior mean with tunable prior strength `kappa`:

```
s_c(x) = ( Σ_{hits} w_j  +  kappa * p0 ) / ( h_c + kappa )
```

`kappa = m_c` recovers the form above. `kappa` constant makes ignorance's pull
independent of genome size — worth having, but it is a second knob.

### Why NOT "mean over hits only"

`s_c = Σ_{hits} w_j / max(h_c,1)` looks like the obvious fix and is **wrong**:
a class with one lucky TRUE scores 1.0 and beats a class with 30 hits averaging
0.9. That trades the abstention pathology for a low-support overfit pathology —
Worms would win on a single hit. Any accepted rule MUST keep the denominator
anchored to evidence *volume*, not just evidence *quality*.

## 4. Why this should not repeat the classnorm failure

`argmax_classnorm` (z-normalize each class's score column) attacks the same
defect and half-works: Worms predictions -49% at UNCHANGED recall (pure FP
removal), but macro-F1 goes DOWN 0.95pp (n=30 paired, wins 8/30) because
dividing by sigma_c rescales the well-trained classes too (Exploits -11.10pp F1).

Subtracting a pedestal is right; rescaling everything is not. The coverage rule
touches a class **in proportion to how ignorant it is** — a dense class has
`m_c ~ 0` and is left alone. That is the specific property classnorm lacks.

## 5. Plumbing / correctness requirements

1. **CPU + Metal parity.** Both `compute_per_example_scores` fallbacks and the
   sparse Metal kernel must change identically; add a case to
   `cpu_fallback_matches_gpu`. A sparse-miss that diverges CPU vs GPU is silent.
2. **Mode coherence.** BINARY already behaves as p0=0; TERNARY/PLN use
   `empty_value` (0.5) for their u-state, which is a *different* concept
   (a genuine 3rd state, not ignorance) and must NOT be folded in. QSR/PLN
   stochastic reads must route through `cell_to_weight_rng` or parity breaks.
3. **Read-side only.** No retraining: applies to already-exported memories, so
   it can be A/B'd on existing genomes.
4. **Threshold recalibration.** Shifting scores down changes the S0 gate's
   operating point; the 7 threshold modes are refit per mode, but benign FPR
   must be re-read, not assumed.
5. **Dense fallback must be explicit** (see §2) — never silently apply the
   sparse rule to dense groups.

## 6. DECISIONS (settled 28/08/2026, Luiz)

- **(a) Rule: `p0 = 0.0`, `kappa = m_c` (minimal).** A miss is *no vote*;
  denominator stays `n_c`. No new hyperparameter. Implemented as
  `default_cell_for_coverage(mode, true) -> cell 0`, whose weight is 0.0 in
  EVERY mode — so one value change covers CPU and Metal, all six modes, with no
  new branch in the inner loop. BINARY is a no-op under it (already 0).
- **(b) Scope: search AND decode.** The scorer drives GA fitness as well as the
  final decode, so the search selects genomes that are good under the rule
  actually used at inference. **Consequence to honour at read-out:** the delta
  is "scorer + search trajectory", NOT the scorer alone, and it needs a full
  re-search per seed. Do not report it as an isolated decode effect.

## 6b. What was implemented

| layer | change |
|---|---|
| `core/metal_sparse.rs` | `default_cell_for_coverage(memory_mode, coverage_aware)` (+ non-macOS stub in `core/lib.rs`) |
| `core/neuron_memory.rs` | `EvalSettings.coverage_aware: bool` (default `false`) + 5 invariant tests |
| `ids_cache.rs` | `IDSCache.coverage_aware` threaded into all 9 `EvalSettings` sites |
| sparse forward | `forward_batch_sparse` / `forward_batch_general` / `evaluate_group_sparse_gpu` / `compute_per_example_scores` take the flag |
| IDS GA evaluators | `compute_ce`, `eval_sparse_groups_batched` take the flag (per decision (b)) |
| PyO3 | `IDSCacheWrapper.set_coverage_aware(bool)` |
| Python | `ids_coverage_aware` in `KNOWN_PARAMS`; read in BOTH `_create_ids_evaluators` and `_create_hierarchical_ids_evaluators`; passed to all 7 `IDSEvaluator` constructions |

**Deliberately NOT changed** — dense `evaluate_group_metal` (no miss signal by
construction); `adaptive_eval.rs` neuron-stat / axonogenesis passes (architecture
*growth* heuristics, not the cascade scorer); LM multistage / bitwise / TERNARY
tiered-sparse paths and `ids_streaming` (all pass `false` explicitly).

Tests: ram_core **78/78** (73 + 5 new), ram_accelerator **111/111**,
ram_controller **187/187**. The new tests assert cell 0 weighs 0.0 in all six
modes (including the QSR/PLN stochastic reads over 64 coin values), that OFF is
bit-exact per mode, that BINARY is a no-op, and encode the defect itself as an
executable statement.

## 8. Deployment (NOT done — worker is mid-cohort)

`ram_core` changed, so BOTH wheels need rebuilding. The controller wheel installs
anytime; the WORKER wheel can only be swapped at worker-idle
(`scripts/worker_swap.py`), and the Python must land WITH it (the worker reads
`ids_coverage_aware` and calls `set_coverage_aware`, which a stale wheel lacks).
Do not swap while MCST/MCSB flows are running.

## 7. Pre-registered read-out

Primary: **over-absorption ratio per class** and rho(train support,
over-absorption) — NOT macro-F1 alone. Secondary: macro-F1, benign FPR,
per-class recall table (QSR lesson: an aggregate win with recall losses across
classes is not "detects better"). Control: paired same-seed runs under the
current scorer. n=5 seeds, paired majority.

Related: `docs/MULTICLASS_DESIGN.md` (decode modes), `docs/MCST_TIERED_ARM_SPEC.md`.
