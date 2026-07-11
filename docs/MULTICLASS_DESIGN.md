# Multiclass IDS Design (S&P 2027 Phase 1 design doc)

Drafted 11/07/2026. Answers Review A ("extend to multi-class attack type
identification") for the S&P 2027 Montreal submission. Binary stays the paper's
spine; this adds a major multiclass evaluation section. See
`docs/SP2027_MONTREAL_PLAN.md`.

## 0. Shared prerequisite: Evaluation Protocol v2 (3-way splits)

Applies to BOTH the binary fresh-seed confirmation cohorts and all multiclass
runs. Motivated by two audit findings (11/07):
- C1.3: configuration selection previously used held-out probe results.
- val_cal today calibrates its threshold ON the same held-out set it reports
  on (why it dominates; reviewer-attackable as "threshold tuned on test").

**Protocol v2:** use the HF `_3way` configs (80/10/10 train/test/val; all
datasets have them). The worker STOPS merging test+val:
- **train (80%)**: GA search with 5-fold CV (unchanged).
- **val (10%)**: threshold calibration (val_cal, Platt, beta fit here),
  configuration/model selection, early-stopping peeks.
- **test (10%)**: final report ONLY. Nothing is ever fit, selected, or
  calibrated on it.

**Investigation results (11/07):** X_val/y_val are loaded and encoded by
every loader and carried on IDSDataset — but NOTHING consumes them (no
merge exists; the CLAUDE.md "worker merges test+val" note was stale). For
`_3way` runs today, eval = the 10% test partition only. Additionally,
val_cal is computed as the ORACLE (threshold −1.0 sentinel = F1-optimal on
the eval scores themselves, `experiment.py` validation block), and
Platt/beta/empirical are fit on TRAIN scores (the paper text says
"held-out" — a claim-code mismatch fixed by v2).

**Implementation plan (additive, no merge to remove):**
1. Rust `ids_cache.rs`: optional val arrays on the cache;
   `evaluate_at_thresholds_ids_cached` also scores val → returns
   (eval_scores, train_scores, val_scores, metrics). pyapi signatures
   follow. (Val is 10% — negligible cache memory growth.)
2. Python `ids_evaluator.py`: upload X_val/y_val when the dataset has them;
   `evaluate_at_thresholds` returns val_scores.
3. Python `experiment.py` validation block, when val_scores present:
   val_cal = F1-optimal threshold on VAL scores applied to TEST scores;
   Platt/beta/empirical/emp-cumulative fit on VAL scores (matches the
   paper's stated semantics); train_cal/fixed_05 unchanged. Log
   `[PROTOCOL-V2]`. When no val partition (legacy 2-way): current behavior.
4. Rebuild worker wheel (`maturin develop --release`) + worker restart —
   safe now, zero flows running.
5. Parity/smoke: 2-way flow unchanged; `_3way` flow logs PROTOCOL-V2 and
   val_cal threshold comes from val.

## 1. Scope and class structure

| Dataset | Classes | Mapping |
|---|---|---|
| UNSW-NB15 (temporal + random) | 10 = 9 attack cats + Normal | `attack_cat` as-is |
| CICIDS2017 (random) | 15 = 14 attack labels + BENIGN | labels as-is |
| CIC-IoT-2023 (1.4M subsample) | 8 = 7 attack categories + Benign | Neto's own 33→7 grouping — enables DIRECT comparison with their published 8-class baselines (Table 7) |
| CIC-IoT-2023 46M | — | stays BINARY (scale demonstration; multiclass at 46M is out of compute scope) |

Class imbalance is severe (UNSW Worms ≈ 130 train rows). Macro-F1 is the
headline metric precisely because it exposes this; we do NOT resample the
data (matches binary protocol; keeps comparability with RF/XGB baselines).

## 2. Architecture: K clusters, one per class

The substrate already supports this — `ClusterGenome` is inherently
multi-cluster (`neurons_per_cluster: [num_clusters]`), and Option-B B5
K-class GPU training exists (any K, parity-verified). Binary IDS has been
the special case K=2 all along.

- Each class c gets a cluster of N_c neurons; per-cluster score =
  mean QSR cell output across its neurons (same as binary).
- Training: one pass; each example writes TRUE-nudges to its class's
  cluster and FALSE-nudges to sampled negative clusters (B5 semantics,
  neuron_sample_rate=0.25 as production).
- **The GA evolves per-cluster neuron counts natively** (`_mutate_neurons`
  is already per-cluster) — rare classes may get more/fewer neurons as the
  search dictates. This is a genuinely interesting research knob we get for
  free: does evolution allocate capacity toward rare classes?

## 3. Decode + thresholds (the genuinely new design work)

Binary's 7 threshold modes don't map 1:1 to K classes. v1 keeps decode
minimal and principled — two decode rules × two calibration sources:

1. **Plain argmax** (no threshold): predict argmax_c score_c. Baseline
   decode; what Di Mauro/Neto-style comparisons use.
2. **Benign-margin threshold** (deployable FPR control): binarize first via
   margin m = max_{c≠benign} score_c − score_benign against threshold τ;
   if attack, assign argmax over attack clusters only. τ calibrated:
   (a) `train_cal`: sweep on train to maximize the fitness function;
   (b) `val_cal`: sweep on val (protocol v2 — now legitimate).
   This is a 2-stage cascade AT DECODE TIME over one shared memory — the
   cascade-paper architecture appears only as an evaluation composition,
   keeping the deeper cascade contributions for paper #2.
3. Deferred to v2 if time allows: per-class Platt/beta calibrated argmax,
   per-class τ_c vectors, reject/unknown option.

## 4. Metrics (full-stack plumbing)

- **Headline:** macro-F1, benign-FPR (= 1 − benign recall), accuracy.
- Per-class precision/recall/F1 + full confusion matrix stored in
  `validation_summaries.threshold_metadata` (extend the per-mode JSON with
  `per_class` — the binary schema already has a per_class rate block).
- Weighted-F1 secondary (comparability with literature that reports it).
- **GA fitness:** the existing weighted harmonic-rank (Eq. fitness) with
  metric substitutions: F1→macro-F1, FPR→benign-FPR, CE→K-class CE,
  Acc→accuracy. Weights: reuse the rebalanced scheme (0.1/0.35/0.35/0.2)
  FROZEN a priori — no weight probing for multiclass (C1.3-clean by
  construction; any weight exploration would select on val, never test).

## 5. Full-stack change list (Rule 6: no forwarding gaps)

| Layer | Change | Est. |
|---|---|---|
| Rust `ids_cache`/eval | K-class scoring exists (B5); ADD: K-class CE, confusion matrix, macro-F1/per-class metrics, benign-margin decode + τ sweep | 1-2 d |
| Rust pyapi | expose multiclass eval results struct | 0.5 d |
| Python worker | label-map loaders (attack_cat / 14-label / Neto-7), K-cluster genome config, protocol v2 val/test roles, validation-summary writer per-class JSON | 1-2 d |
| params registry | `ids_num_classes`, `ids_label_mapping`, `eval_protocol`, `decode_mode` keys | 0.5 h |
| Dashboard | macro-F1/benign-FPR/Acc columns for multiclass flows (flow-type-aware, like the F1/FPR/ACC vs CE/ACC switch) | 0.5-1 d |
| Baselines | RF/XGB multiclass on identical splits + protocol (extend existing baseline scripts); pull Neto 8-class Table 7 numbers (verified source) | 0.5 d |
| `ids_stats.py` | metric key passthrough (macro_f1 etc.) — near-free | 1 h |

Total implementation: ~1-1.5 weeks alongside watcher duties. Deploy per the
standing rule: `maturin build` + worker swap at idle only.

## 6. Experiment sequence (compute plan)

1. **Smoke** (post-drain, ~20/07): K-class training parity vs B5 tests; tiny
   UNSW multiclass run end-to-end through dashboard.
2. **Screening** (n=5/dataset, fixed 250n×100b-style caps): verify signal,
   lock decode default, confirm fitness weights behave. Selection decisions
   on VAL only.
3. **Cohorts** (Sep): UNSW-temporal n=100 first (multiclass temporal is the
   novel result nobody reports), then UNSW-random n=50, CICIDS n=50,
   CIC-IoT subsample n=50. Interleaved seeds (rule: never batch same combo).
4. **Baselines**: RF/XGB multiclass per dataset (hours, CPU).
5. Analysis via `ids_stats.py` (BCa CIs + rank tests, macro-F1).

Contention: shares the box with the controller chain (PID-full + seed-pairs
+ hybrid runs) — max 2 heavy runners rule stands; multiclass cohorts get
priority from ~01/09 per the Montreal timeline.

## 7. Open questions (decide at screening, on val data only)

1. **IN SCOPE (Luiz 11/07): rare-class capacity allocation.** Does evolution
   shift neuron allocation toward rare classes, and does frequency-scaled
   init (`token_frequencies` path exists) beat uniform? Instrumentation:
   log per-cluster neuron counts per generation (population checkpoints
   already carry genomes); screening runs one uniform-init arm + one
   frequency-scaled-init arm; report the per-class N trajectory vs class
   frequency + per-class recall. Target: a paper subsection.
2. Benign-margin τ: one global τ vs per-class τ_c (v1: global).
3. CICIDS: 14 labels vs grouping web attacks (v1: 14 as-is; grouping only
   if per-class supports collapse).
4. Multiclass memory footprint at 250n×100b caps × K clusters — measure at
   screening; shrink caps if RAM-bound (46M lesson: max 2 heavy runners).

## 8. What this buys the paper

- Review A's multiclass ask: answered with cohort-grade rigor (n=50-100,
  BCa CIs, rank tests), not a token experiment.
- First (to our knowledge) TEMPORAL-split multiclass WNN IDS results.
- Direct 8-class comparison against Neto et al.'s published CIC-IoT
  baselines on their own grouping.
- The per-class capacity-allocation question (7.1) is a novel micro-finding
  candidate that falls out of existing GA machinery.
