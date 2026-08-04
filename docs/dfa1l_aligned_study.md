# The dfa1l study, on the aligned measurement axis

Regenerated 03/08/2026 from `experiments/rescore/all.json` via
`scripts/build_dfa_aligned_table.py`.

**Do not read `scripts/build_dfa_1layer_table.py` or the per-cell marker triples.**
Every cell in this sweep ran before the 03/08/2026 fix, so the triple each marker
carries was printed through a *refit* address function — thresholds fit on the report
seed while the genome's cells were written under the train seed, i.e. the memory was
read where nothing had been written. See
[`threshold_misalignment_finding.md`](threshold_misalignment_finding.md). The numbers
below are the same frozen winners replayed on the aligned axis by
`scripts/rescore_winners.py`.

The sweep was stopped at **25 run cells** (of 40 planned) once the ranking settled;
8 further corners were **cost-skipped** and never ran, so there is no winner to
rescore and nothing untrustworthy about them. 0 cells remain on the broken axis.

## Held-out results

Disturbance L2D, tilt 5°, 100 report-episodes × 2000 steps, 5 report seeds per cell.
`n` = training seeds; ±SD is **across training seeds**, using each cell's report-seed
mean — the same axis as the old marker table's ±SD, so the two columns are comparable.
(The per-cell test-set ±SD across the 5 report seeds is a different quantity and is
kept in `all.json`, not printed here.)

```
  cell                          n       stable%        err°     steady°   vs BROKEN axis
  ----------------------------------------------------------------------------------
  1layer 9feat  BINARY          4     87.2±1.9      3.3±0.1     3.3±0.1   was  59.2% /   4.8°
  1layer 9feat  QUAD            4     42.4±14.1     6.1±0.9     7.2±1.9   was  14.2% /  10.8°
  1layer 10feat BINARY          4     99.4±0.7      2.1±0.3     1.6±0.4   was  28.4% /   7.6°
  1layer 10feat QUAD            4     79.2±9.2      4.2±0.4     4.3±0.5   was  23.2% /  10.0°
  dfa    9feat  BINARY          4     88.5±1.0      3.2±0.1     3.3±0.1   was  76.3% /   3.9°
  dfa    9feat  QUAD            1     26.8±0.0      7.7±0.0     9.9±0.0   was  35.2% /   6.8°
  dfa    10feat BINARY          3     99.1±0.6      2.3±0.1     1.9±0.2   was  21.0% /   8.4°
  dfa    10feat QUAD            1     56.4±0.0      5.3±0.0     5.9±0.0   was  14.0% /   8.6°
  ----------------------------------------------------------------------------------
  CLASSICAL BASELINES (compute_baselines.py, same 5 report seeds):
  PID                           5     90.4±7.5      4.0±0.4     4.0±0.5
  LQR                           5    100.0±0.0      1.6±0.1     1.3±0.2
  MPC                           5    100.0±0.0      1.7±0.1     1.4±0.2
  LQI                           5    100.0±0.0      1.4±0.1     1.0±0.1
  MPCOF                         5    100.0±0.0      0.8±0.0     0.2±0.0
  ----------------------------------------------------------------------------------
  aligned: 25 | cost-skipped (never ran): 8 | markers: 33 | STILL ON THE BROKEN AXIS: 0
```

The broken axis did not merely add noise — it **inverted the ranking**. Under it,
`dfa 9feat BINARY` (76.3%) looked like the best WNN and `1layer 10feat BINARY`
(28.4%) looked like one of the worst. Aligned, that ordering reverses exactly.

## What the aligned axis says

### 1. 10 features beat 9 — 12/12 paired, no exceptions

Paired by (substrate, mode, training seed), Δstable = 10feat − 9feat:

| substrate | mode | n | mean Δstable | all positive |
|---|---|---|---|---|
| 1layer | BINARY | 4 | **+12.2 pp** | yes |
| 1layer | QUAD | 4 | **+36.8 pp** | yes |
| dfa | BINARY | 3 | **+10.7 pp** | yes |
| dfa | QUAD | 1 | +29.6 pp | (n=1) |

Every one of the 12 pairs moves the same direction. This is the conclusion that most
cleanly inverted: the broken axis had 9feat ahead.

### 2. Both 10feat BINARY rows beat PID

PID sits at 90.4% ± 7.5 / 4.0°. `1layer 10feat BINARY` (99.4% ± 0.7 / 2.1°) and
`dfa 10feat BINARY` (99.1% ± 0.6 / 2.3°) clear it on stability, error, and steady-state
error, with roughly a tenth of PID's seed-to-seed spread. They do **not** reach the
model-based teachers (LQR 1.6°, MPC 1.7°, LQI 1.4°, MPCOF 0.8°), which remain the
ceiling.

### 3. The single layer ties the DFA — at 19–37× less search

Paired by (feature, mode, training seed), Δstable = 1layer − dfa:

| feature | mode | n | mean Δstable | pairs (1layer / dfa) |
|---|---|---|---|---|
| 9feat | BINARY | 4 | −1.4 pp | 86.0/89.8, 86.2/88.2, 90.0/87.4, 86.6/88.8 |
| 10feat | BINARY | 3 | **+0.2 pp** | 99.6/99.8, 100.0/98.6, 98.4/99.0 |

At 10feat the two substrates are indistinguishable (Δ +0.2 pp, sign splits 1–2). Yet
the measured wall-clock cost per cell is not close:

| feature | mode | 1layer | dfa | ratio |
|---|---|---|---|---|
| 9feat | BINARY | 0.32 h | 6.02 h | **18.6×** |
| 10feat | BINARY | 0.50 h | 18.45 h | **37.2×** |
| 9feat | QUAD | 0.26 h | 30.23 h | 114.5× |
| 10feat | QUAD | 0.44 h | 90.95 h | 208.3× |

(Earlier notes cited "1/60th the search cost" from memory. The measured figure at the
headline corner, 10feat BINARY, is **37×**, not 60×. The 60–200× range only appears in
the QUAD corners — which is precisely why those dfa corners were cost-skipped: a single
`dfa 10feat QUAD` cell cost 91 hours.)

**Caveat that limits this conclusion: it is an L2D result.** A single layer having
nothing left to gain from a recurrent state at L2D does not imply the same at L3D,
where every arm collapses and history may start to pay. That is the pre-registered
question P3 is measuring, and this table cannot answer it.

### 4. BINARY beats QUAD 10/10 — but the comparison is confounded

Paired by (substrate, feature, seed), Δstable = QUAD − BINARY:

| substrate | feature | n | Δstable per seed |
|---|---|---|---|
| 1layer | 9feat | 4 | −51.8, −30.4, −63.2, −34.0 |
| 1layer | 10feat | 4 | −31.6, −24.2, −14.4, −10.6 |
| dfa | 9feat | 1 | −63.0 |
| dfa | 10feat | 1 | −43.4 |

Ten pairs, ten losses for QUAD, several of them enormous. Taken at face value this
says the 4-state nudging cell is simply worse than the 1-bit cell for this task.

**It does not establish that**, because in this sweep BINARY and QUAD differ in more
than the cell alphabet — the output decode topology moves with it. Attributing the
whole gap to cell granularity is the confound flagged in
`project_gran_ablation_winner_variance`. De-confounding it is exactly what the
`--output-decode antagonist` axis (shipped 03/08/2026, ABI 21, CPU/GPU parity 10/10)
exists to do, and is the subject of the P4 run.

## Reproducing

```bash
python3 scripts/build_dfa_aligned_table.py    # prints the table above
```

Source of truth: `experiments/rescore/all.json` (git-tracked, 25 cells, each holding
both the aligned `train` replay and the broken `per_seed` replay for every report seed).
