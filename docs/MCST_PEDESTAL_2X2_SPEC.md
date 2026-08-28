# MCST pedestal 2x2 — spec (28/08/2026)

Status: **SPEC ONLY — nothing armed.** Requires Luiz's go before any flow is created.

## 1. What is being tested

One defect, two independent routes to it.

**The defect** (verified in code and data, `docs/COVERAGE_AWARE_SCORER_SPEC.md` §1):
in QUAD an UNTOUCHED cell commits to `WEAK_FALSE = 0.25` while a class that
LEARNED to reject commits to `FALSE = 0.0`, and the per-class score is a mean
over that class's own neurons. So **ignorance outranks knowledge** and the
emptiest class wins a raw argmax by abstention. UNSW Worms (97 train rows)
absorbs **508x** its fair share of every misclassification; over-absorption
against train support runs **rho = -0.930** (n=10 classes).

Two routes, attacking it at different times:

| | route | when it acts |
|---|---|---|
| **A. sizing** | give each class an address space it can actually populate | **train time** — there are no untouched cells to score |
| **B. scorer** | score a sparse miss as 0.0, not 0.25 | **read time** — untouched cells stop voting |

They are NOT redundant: A raises evidence density, B silences the residue. The
interaction cell is the point of the design — do they overlap, or compound?

## 2. Factors

**Factor A — sizing rule** (2 levels). The bits band exists only to bound the
sizing formula, so the rule and its band move together as ONE treatment:

- `A0 = current`: legacy rule `b = bmax*log2(s)/log2(S_model)`, band **34-50**.
  This is exactly MCST tiered3 (flows 6005-6009). Per-class bits 34-45.
- `A1 = constant-fill`: corrected rule `b = log2(s*sample_rate/FILL_TARGET)`,
  band **4-14**. Per-class bits 4-12.

Derived centres (UNSW S1, sample_rate 0.25, FILL_TARGET 2):

| class | supp | A0 | A1 | | class | supp | A0 | A1 |
|---|---|---|---|---|---|---|---|---|
| Worms | 97 | 34 | **4** | | Recon | 7868 | 39 | 10 |
| Shellcode | 850 | 34 | **7** | | DoS | 9198 | 40 | 10 |
| Backdoor | 1309 | 34 | **7** | | Fuzzers | 13638 | 42 | 11 |
| Analysis | 1500 | 34 | **8** | | Exploits | 25045 | 44 | 12 |
| | | | | | Generic | 30000 | 45 | 12 |

Under A0 the four rare classes are **FLOOR-CLAMPED** (the formula wanted
20/30/31/32, the band floor of 34 overrode it) — and those four are **exactly
the four biggest over-absorbers** (Worms 508x, Shellcode 13.0x, Analysis 9.1x,
Backdoor 4.6x). The 5th-ranked class (Fuzzers 2.4x) is the first unclamped one.
Perfect separation.

**Factor B — scorer** (2 levels): `ids_coverage_aware ∈ {false, true}`.
Shipped, default-OFF and bit-exact there (`docs/COVERAGE_AWARE_SCORER_SPEC.md`).

## 3. Design

2x2 x 5 seeds (20401-20405) = **20 runs**. Everything else identical to MCST
tiered3: UNSW `temporal_3way`, quad 16b top20, hierarchical, cap 150,
`ids_tier_sizing=true`, k_folds=5, patience 5.

| cell | sizing | scorer | name |
|---|---|---|---|
| A0B0 | current | off | `MCSP-a0b0-s2040X` (= tiered3, RE-RUN) |
| A0B1 | current | on | `MCSP-a0b1-s2040X` |
| A1B0 | constant-fill | off | `MCSP-a1b0-s2040X` |
| A1B1 | constant-fill | on | `MCSP-a1b1-s2040X` |

**A1B1 was briefly dropped and is RESTORED (28/08/2026).** The reason given for
dropping it — "A1's bits are 4-12, below SPARSE_THRESHOLD=12, so the groups are
dense and `coverage_aware` is inert" — was WRONG. That threshold governs the
CLASSIC `GroupMemory` path. The live path is the marker path (Option B): both of
its `GenomeExport` constructions set `dense_exports: vec![]` and push
`(is_sparse=true, ...)` for every group regardless of bits, and the worker log
shows `PATH2_FALLBACK` has never fired. A b=4 group is therefore still read
through the sparse binary search, `miss_default` applies, and the flag acts. The
full 2x2 is measurable and no dense coverage machinery is needed.

**Re-run A0B0 rather than reusing flows 6005-6009.** The banked runs are on the
pre-coverage-aware wheel; `coverage_aware=false` is bit-exact by construction
and tested, so reuse is defensible — but runs are 4-20 min and a same-wheel
control removes the argument entirely. Cost of the caution is ~1 h.

Cost: ~20 runs x ~10 min ≈ 3.5 h wall clock, one at a time on the 13-core budget.

## 4. Pre-registered read-out

**Primary — the mechanism, not the headline:**
1. **Worms over-absorption** (share of all misclassifications absorbed / share of
   the test set). A0B0 baseline ≈ 460-530x. Report all four cells + the
   interaction.
2. **rho(train support, over-absorption)** across the 10 classes. A0B0 ≈ -0.93.
   Toward 0 means the sink structure is gone.

**Secondary:** macro-F1, benign FPR, accuracy, and the **per-class recall table
(MANDATORY** — QSR lesson: an aggregate win with recall losses across classes is
NOT "detects better").

**Interaction is the question.** Report `(A1B1 - A0B0)` against
`(A1B0 - A0B0) + (A0B1 - A0B0)`. Sub-additive ⇒ the two routes are draining the
same pedestal and one suffices. Additive/super-additive ⇒ they are distinct and
the fix is both.

**Expected costs, stated in advance so they are not read as surprises:**
- The BINARY probe (n=1, flow 6010 vs 6005) drained the sink 415x → 10.1x but
  cost **-2.28pp macro-F1** with Worms TP 17 → 6. Silencing abstention removes
  over-prediction AND detection. Expect B1 to trade macro-F1 for benign FPR
  (BINARY gained **-5.82pp FPR**, better than any QUAD run in either cohort).
- A1 puts Exploits/Generic at 12 bits vs 44/45. That is a large drop in
  discrimination for the big classes; A1 may lose accuracy outright. **If it
  does, that is the finding** — it bounds how far density can be traded for
  address width, and it is why the low band is 4-14 rather than 4.

**Falsified if:** A1B0 leaves Worms over-absorption above ~100x. That would mean
evidence density is not what drives the sink and only the read-side fix matters.

## 5. Confounds, stated up front

- **A moves formula AND band together.** Deliberate — the band only exists to
  bound the formula, and a band of 34-50 under the corrected rule clamps every
  class to 34 (uniform, no tiering at all), which is not a meaningful control.
- **A also changes what the GA searches**, since `ids_coverage_aware` drives
  fitness as well as decode (Luiz, 28/08). Deltas are "treatment + search
  trajectory", not the treatment alone. Do not report either as isolated.
- **Not comparable to tiered2/tiered3 headline numbers** except through the
  re-run A0B0 cell.

## 6. Dependencies (all landed)

- Corrected `bits_centre` + `FILL_TARGET` + `BITS_MIN 10→4`, legacy rule kept as
  `bits_centre_logratio` for reproducing banked cohorts. `tests/test_tier_sizing.py`
  16/16, including a test that pins the legacy rule's >50x fill spread.
- `ids_coverage_aware` flow param, worker wheel swapped 28/08 11:5x ET.
- **A0 can no longer be produced by the default code path** — reproducing
  tiered2/tiered3 now requires `bits_centres_logratio` explicitly. The A0 cell
  needs that wired behind a flow param before this can run. **THIS IS THE ONE
  OPEN IMPLEMENTATION ITEM.**

Related: `docs/COVERAGE_AWARE_SCORER_SPEC.md` · `docs/MCST_TIERED_ARM_SPEC.md`
