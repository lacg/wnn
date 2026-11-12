# IDS-57 options: closing the WNN gap to the classical models on UNSW-NB15

Written 12/11/2026 (ids-security agent) for Luiz. **Proposal only. Nothing was queued, paused, cancelled, resumed or
reordered; no wheel was swapped; no worker code changed.** The worker (ABI 28) ran IDS56U (7543) throughout.

- Probes ran at `nice -n 10`, one process at a time, 4 threads, through the worker's own construction path
  (`IdsEvaluatorFactory` -> `IDSCacheWrapper`, Rust `evaluate_multiclass_at_thresholds`). No Python reimplementation
  of network logic. Scratch scripts: `/tmp/ids57/probe1.py` .. `probe4.py` (not committed).
- **TEST** = `margin_val_cal` (threshold fitted on VAL, scored on the 10% TEST). TEST was read only for genomes that
  production flows already reported, and for the XGBoost comparator.
- **GATE (VAL)** = threshold fitted on VAL and scored on VAL. It is a tuning read. It is shown only as a difference
  and a pass/fail verdict, never as a result. No exploratory genome was scored on TEST.
- Fidelity: the five lattice-encoff GA genomes retrained offline reproduce their DB TEST numbers exactly
  (7501 94.17/0.19, 7505 93.89/0.91, 7509 94.12/0.20, 7513 94.11/0.23, 7517 94.33/0.20). XGBoost refit reproduces the
  banked 95.53/0.42.

## 1. Headline

1. On unswr the gap is **one attack category and one cap**. 76% of the extra misses are Fuzzers (WNN recall 45% vs
   XGBoost 76%, TEST). Every neuron of every GA genome sits at `max_bits=34`, and the search score is still rising at
   that cap.
2. Raising the cap passes its gate by a wide margin: random genomes (no search) at 64 bits are **+1.18 pp F1 on VAL**
   over the same genomes at 34 bits, 5/5 draws. That is 3/4 of the VAL distance to XGBoost. At 80-128 bits the
   remaining VAL distance is 0.1-0.2 pp.
3. So the IDS-23 reading "UNSW-random plateau = dataset ceiling" does not hold as stated. XGBoost reaches 95.53 on
   the **same bits**. The plateau is the ceiling of the 34-bit cap. This is inferred for TEST until an arm runs.
4. The price is size: 64-bit genomes hold about 2.2M cells against 0.66M today (about the size of the XGBoost model).
5. On unswt none of this applies. The same ladder is flat from 34 to 48 bits and falls after. No arm is proposed there.

## 2. Diagnosis: where the 1.3-1.7 pp on unswr comes from

Base: best WNN = lattice encoff, 94.12 +- 0.16 F1 / 0.35 FPR (n=5). XGBoost on the same 64-bit thermometer bits =
95.53 / 0.42. Test partition: 152,390 normal / 6,036 attack.

### 2a. Feature list equality (MEASURED: equal)

The caveat is closed. The WNN flow's loader resolves exactly the 20 names of `TOP20_RF_FEATURES`, in the same order,
20/20 present in the HF `random_3way` columns:

`ct_dst_sport_ltm, ct_src_dport_ltm, ct_srv_dst, ct_state_ttl, Dintpkt, dmeansz, Dpkts, dttl, dur, sbytes, Sintpkt,
Sjit, smeansz, Spkts, sttl, swin, tcprtt, proto, service, state`

- Thermometer baselines call the same loader (`load_unsw_nb15(..., feature_selection="top20")`, ranking `legacy` by
  default), so they see the identical 1104-bit matrix. The XGBoost refit in this probe used the very matrix the WNN
  cache was built from.
- Raw baselines read the same constant (`unsw_dataset.TOP20_RF_FEATURES`) from the same HF config.
- Both sides use the **legacy** ranking. `train_only` was not used by either.

### 2b. Per-attack-category recall on the same TEST rows (MEASURED)

WNN = the four seeds whose threshold landed on the low-FPR point (7501, 7509, 7513, 7517). Seed 7505 landed on the
high-recall point and is shown separately.

| Category | rows | XGBoost | WNN mean [min-max] | WNN 7505 | extra WNN misses (rows) |
|---|---|---|---|---|---|
| Normal (correctly passed) | 152,390 | 99.6 | 99.8 [99.8-99.8] | 99.1 | -322 (WNN has fewer false alarms) |
| **Fuzzers** | 1,344 | **75.7** | **44.9 [43.8-46.4]** | 89.6 | **+413** |
| Exploits | 1,705 | 97.4 | 94.1 [94.1-94.2] | 99.0 | +55 |
| Reconnaissance | 808 | 99.8 | 95.4 [94.3-96.2] | 99.9 | +35 |
| DoS | 351 | 97.4 | 92.7 [91.7-94.6] | 99.7 | +17 |
| Generic | 1,435 | 99.6 | 98.6 [98.3-99.0] | 99.9 | +14 |
| Shellcode | 94 | 93.6 | 82.7 [79.8-88.3] | 98.9 | +10 |
| Analysis | 171 | 86.0 | 84.8 | 95.3 | +2 |
| Backdoor | 117 | 99.1 | 98.1 | 100.0 | +1 |
| Worms | 11 | 100.0 | 97.7 | 100.0 | 0 |

Totals: WNN 968 missed attacks / 313 false alarms; XGBoost 420 / 635.

### 2c. Error overlap (MEASURED): the same hard rows, not different ones

| TEST rows | n | WNN wrong (mean of 4 seeds) |
|---|---|---|
| attacks XGBoost misses | 420 | 368 (88%) |
| attacks XGBoost catches | 5,616 | 600 (11%) |
| benign XGBoost flags | 635 | 235 (37%) |
| benign XGBoost passes | 151,755 | 78 (0.05%) |

- The WNN's misses contain 88% of XGBoost's misses, plus 600 more. 75% of the WNN's false alarms are also XGBoost
  false alarms.
- Both models struggle on the same block: Fuzzers that look like benign traffic. XGBoost splits that block; the WNN
  calls the whole block one way or the other.

### 2d. Operating point (MEASURED)

XGBoost scored at the WNN's own false-alarm count, same TEST rows:

| seed | WNN F1 / FPR | XGBoost at that FPR | gap at equal FPR |
|---|---|---|---|
| 7501 | 94.17 / 0.19 | 95.06 | 0.89 |
| 7509 | 94.12 / 0.20 | 95.09 | 0.97 |
| 7513 | 94.11 / 0.23 | 95.22 | 1.11 |
| 7517 | 94.33 / 0.20 | 95.12 | 0.79 |

- About **0.9 pp** of the gap is ranking quality at equal FPR.
- About **0.4 pp** is XGBoost moving to FPR 0.42 (95.53). The WNN has no usable point there: its stored modes jump
  from 94.18 / 0.21 straight to 93.28 / 0.41 (`margin_classnorm`) and then to the high-recall point (about 93.9 / 0.9).
- Seed 7505 landed on the high-recall point (93.89 / 0.91). That costs the n=5 mean about 0.06 pp and most of the SD.

### 2e. Search and caps (MEASURED)

- Grid best_f1 vs GA best_f1 on TEST: 93.84-94.01 (all five on the high-recall point, FPR 0.9) vs 94.12 mean. The GA
  adds about +0.23, mostly by moving 4/5 seeds to the low-FPR point.
- The GA Neurons phase changes neuron counts only. Connections are never mutated and bit widths are not searched
  (docs/ids_ga_plateau_ids26.md).
- **All 2,477 neurons in the five GA genomes are 34 bits wide, the `max_bits` cap.** Cluster sizes are 242-250, the
  `max_neurons` cap.
- Grid CV F1 (the search metric, 5 seeds, never a result) is still rising at the cap: 500 neurons, 28 -> 32 -> 34
  bits gains +0.13 then +0.05, with FPR falling.
- No 2-cluster BINARY flow on unswr ever ran above `max_bits=34` (29 completed flows, all 34). The only unswr flows
  above 34 were single-cluster (`abl2big`, max 100), a different architecture.

### 2f. Representation (MEASURED from existing docs + this probe)

- Thermometer costs the trees nothing: XGBoost 95.53 on the bits vs 95.60 raw.
- XGBoost uses 258 of the 1104 bits. 84.5% of its gain is on `sttl`.
- Interaction order, XGBoost as the instrument: depth 1 (no interactions) 93.67 / 0.98, depth 2 94.71, depth 3 95.11,
  depth 6 95.53. The WNN (94.12) sits between depth 1 and depth 2. Its high-recall point (93.5 / 1.1) is where the
  no-interaction model sits.
- Ranking `legacy` vs `train_only`: moves raw RF +0.32 and XGBoost +0.05 (docs/ids53_leak_size.md). Not measured on
  the WNN. It moves the comparator too, so it is not a gap lever.

### 2g. Verdict

| Candidate cause | Verdict | Basis |
|---|---|---|
| Features differ from the baselines | **Refuted** | 2a, measured |
| Thermometer encoding loses information | **Refuted** | 2f, XGBoost on the same bits |
| Capacity: bits per neuron capped at 34 | **Confirmed as binding; size of the TEST effect inferred** | 2e measured; gate in section 3 |
| Capacity: neurons per cluster capped at 250 | Binding but small | 2e; gate +0.16 at 64 bits |
| Random connectivity starves key features | **Refuted at gate** | section 3, no weighting scheme beats uniform |
| Decision rule (2-cluster margin) | Partly: no operating point between FPR 0.2 and 0.9 | 2d measured; more bits removes the bistability at gate |
| Search (GA plateau) | Secondary: the GA cannot widen neurons or rewire | 2e |
| Dataset ceiling (IDS-23) | **Not supported as stated** | XGBoost 95.53 on the same bits; gate ladder |

Mechanism (INFERRED): with 34 address bits, 1.2M benign training rows cover most addresses a benign-looking Fuzzer
row hits, so benign neurons fire on it. Wider addresses collide less. The unweighted vote (IDS56U) attacks the same
collisions from the other side, which is why its extra gain shrinks as bits grow (section 3).

## 3. Offline gates already run (VAL only, differences only)

Random 2-cluster genomes, 250 neurons per cluster, uniform random connections, built like grid genomes, trained on the
full train partition, 5 draws each. Baseline = the same genomes at 34 bits under lattice. For scale: on the same VAL
read, XGBoost is +1.58 over that baseline and the production GA genome of 7501 is +0.57.

**unswr, bits ladder (lattice / IDS56U rule = unweighted vote theta 0.02):**

| bits | lattice dF1 | draws on the low-FPR point | unweighted vote dF1 | vote's extra | cells |
|---|---|---|---|---|---|
| 24 | -0.36 | 0/5 | -0.17 | +0.19 | - |
| 34 | 0 (baseline) | 0/5 | +0.68 | +0.68 | 0.66M (GA genome) |
| 48 | +0.67 | 3/5 | +1.06 | +0.39 | - |
| **64** | **+1.18** | **5/5** | **+1.33** | +0.15 | 2.15M |
| 80 | +1.42 | 5/5 | +1.46 | +0.04 | 3.14M |
| 96 | +1.38 | 5/5 | +1.42 | +0.04 | 4.23M |
| 128 | +1.47 | 5/5 | +1.47 | 0.00 | 6.64M |

- The worst 64-bit draw beats the best 34-bit draw by 1.03. The ladder flattens at 80.
- Per category at 64 bits (2 draws, VAL): Fuzzers recall 62% at FPR 0.3. The 34-bit GA genomes reach 45% at their
  low-FPR point (TEST, different partition, indicative only). XGBoost is at 76%.

**unswr, other levers at the gate:**

| Lever | dF1 on VAL | Verdict |
|---|---|---|
| 500 neurons per cluster at 34 bits | +0.32 | small |
| 500 per cluster at 64 bits (vs 250 at 64) | +0.16 | small, 2x cells |
| 125 per cluster at 64 bits (vs 250 at 34) | +0.65 | half the cells of 250 at 64, half the gain |
| Connections drawn per feature, not per bit (34 / 48 bits) | +0.07 / +0.15 | within draw noise: **killed** |
| Connections over distinct bits only (what dedupe does) | -0.15 / -0.19 | **killed**; consistent with dedupe hurting on unswr |
| Connections weighted by train XGBoost gain (full / half) | -0.32 / -0.12 at 34 bits | **killed** |
| `neuron_sample_rate` 1.0 at 64 bits (3 draws, same genomes) | -1.12 | **killed**; rate stays 0.25 |

**unswt (ms01 + dedupe, the best unswt recipe), same ladder:**

| Lever | dF1 on VAL vs 250 x 34 | Verdict |
|---|---|---|
| 24 bits | -0.50 | - |
| 48 bits | -0.05 | flat |
| 64 bits | -0.60 | worse |
| 96 bits | -2.69 | worse |
| 500 per cluster (34 / 48 bits) | -0.03 / -0.03 | flat |

unswt has 175K training rows against unswr's 1.27M. Its addresses are already thin at 34 bits. **Both capacity levers
are killed on unswt.**

## 4. Menu of IDS-57 arms (ranked)

Hours: unswr lattice flows measured 4.0-6.1 h (mean 5.08 h, 70-100 generations); grid phase 0.02 h. Offline, training
one 500-neuron genome takes 0.27 s at 34 bits, 0.87 s at 64 and 1.0 s at 80. **The GA wall time at 64 bits is not
measured.** The estimate below scales the measured 5.08 h by 1.6-3.3x.

| # | Arm | Hypothesis | Params that change | Expected TEST effect and basis | Cost | Gate | Adoption rule |
|---|---|---|---|---|---|---|---|
| **A1** | **b64**: raise the bits cap, lattice encoff | The 34-bit cap is what holds unswr at 94.1 | `max_bits`: 34 -> 64. Nothing else. Grid becomes 4,10,20,...,60,64 by the worker's own rule. No code, no swap | **+0.6 to +1.0 pp F1**, FPR about +0.1 (stays under XGBoost's 0.42). Basis: gate +1.18; the 34-bit GA genome is already +0.57 of that; VAL-to-TEST shrink on 7501 was 0.46 | 3 flows x 8-17 h = **25-50 h** | **PASSED** (5/5 draws) | below |
| A2 | **b64 + nb02**: A1 with the IDS56U rule | The unweighted vote still adds on top of wide neurons | A1 plus `ids_write_rule=vote`, `ids_vote_share=0.02`, `balance_classes=false` | +0.1 to +0.2 over A1 and a tighter spread. Basis: gate +0.15, every nb02 draw above every lattice draw | 3 flows, 25-50 h | PASSED, small | Run **only if IDS56U adopts**. Same rule as A1, control = A1 flows, win >= +0.05 |
| B | **b80**: cap at 80 | The ladder keeps paying until 80 | `max_bits`: 80 | +0.2 over A1 (gate +0.24). 3.1M cells. Above 64 bits addresses are hashed, not identity: affects the FPGA story | 3 flows, 30-60 h | PASSED, small | Only if A1 adopts and leaves >= 0.3 pp to XGBoost |
| C | **n500**: 500 neurons per cluster at 64 bits | More voters, not wider ones | `max_neurons`: 500 | +0.1 to +0.2 (gate +0.16), 2x cells and time | 3 flows, 50-100 h | PASSED, small | Not recommended before B |
| D | **unswt: no flow** | The 0.2-0.5 pp on unswt is not capacity | - | Bits and neurons are flat or negative at the gate | 0 | **KILLED** | See section 6 |
| E | **Per-neuron vote weights / stacked readout** | A learned readout over neuron outputs could rank the hard block | Needs a Rust export of per-neuron outputs (worker wheel swap) before it can even be gated | Unknown; not measurable today | code + swap | not run | Park until A1 reads out |

Checked and dropped: GA Connections phase (the operator only shifts a connection by +-2 bit positions, and no
connectivity weighting passes the gate); feature-family receptive fields (IDS-55, lost); dedupe on unswr (IDS-56,
lost; gate agrees); `train_only` ranking as a gap lever (moves the comparator equally); sample rate 1.0.

### A1 pre-registration (draft, to be ruled by experiment-design before queueing)

- **Arm:** clone 7509 / 7513 / 7517 (unswr lattice encoff, r20421-23), override **only** `max_bits=64`. Name
  `IDS57-unswr-WbCTRL-desir-BINARY-2m-max250-b64-xo_blend_cluster-rate0.25-rklegacy-lattice-encoff-r<seed>`.
  `scripts/queue_ids56_unweighted_vote.py` is the template (POST /api/flows with the two-experiment array, verify the
  experiment count, then PATCH status=queued). Its PINNED check must be changed to `max_bits: 64`.
- **Control:** the existing 34-bit flows of the same seeds (7509, 7513, 7517). Not re-flown. Readout guard:
  `test_summaries.worker_abi` equal on arm and control (28).
- **Seeds:** r20421-23 first. The value 64 was chosen on VAL with genome draws that do not use flow seeds, and on the
  grid CV slope of all five seeds (train only). Extension seeds r20419-20 are disclosed as part of that CV evidence.
- **Metric:** GA `best_f1`, `test_point='final'`, `margin_val_cal`, from `test_summaries.threshold_metadata`. Paired
  by seed. Macro-F1; report attack F1, TPR, FPR, cells and hours beside it. Rule-7 tables for all modes and genome
  types. Never `iterations.best_f1`.
- **Per seed:** dF1 = arm - control. Win if dF1 >= +0.15 pp (about one SD of the control).
- **n=3:** ADOPT if 3/3 wins AND mean dF1 >= +0.40 AND arm mean FPR <= 0.42 (XGBoost's). REJECT if wins <= 1/3 OR
  mean dF1 < +0.15. Otherwise AMBIGUOUS -> n=5 with r20419-20; then ADOPT if >= 4/5 wins AND mean dF1 >= +0.40 AND
  mean FPR <= 0.42, else REJECT. No further extension.
- **What adoption means:** b64 becomes the unswr accuracy row. The 34-bit genome stays as the small-model row
  (0.66M vs about 2.2M cells). The paper's unswr size claim changes and must be restated.
- **Early stop for cost:** if the first flow's GA runs past 20 h, stop queueing more and bring the timing to Luiz.

## 5. Protocol guardrails (every arm)

| Guardrail | How it is kept |
|---|---|
| `_3way` split | `ids_split=random_3way`, inherited from the control, asserted by the queue script |
| K-fold 5 on train drives the GA | `ids_k_folds=5`, `ids_kfold_per_gen=5`, `ids_offspring_folds=all`, inherited and asserted |
| Thresholds calibrated on VAL | `margin_val_cal` only |
| TEST touched once | Only at the flow's own final report. No offline TEST read of any arm genome before the readout |
| No parameter chosen on TEST | 64 / 80 / theta come from VAL gates and train CV. Section 2 TEST reads were diagnosis on already-reported genomes |
| One change per arm | A1 changes `max_bits` only; A2 adds the three IDS56U keys only |
| Base rate | unswr is 3.8% attack: read F1 and FPR, not accuracy |
| Flows | POST /api/flows with the experiments array; verify count; no SQL; no worker change; cohort recipe untouched |

## 6. If the gap does not close

- **unswr.** The gate leaves 0.1-0.4 pp to XGBoost on VAL even at 64-128 bits, before any GA gain. A1 may land at
  94.7-95.1 against 95.5. The residue is Fuzzers (62% vs 76% at the gate). If A1 and B both leave it, the honest
  statement is: the WNN matches the classical FPR and is within about 0.5 pp F1 at about the same model size, and the
  remaining difference is one category.
- **unswt.** The gap is 0.2-0.5 pp against single fixed-seed baselines, and the WNN SD is 0.07-0.08. No capacity
  lever moves it at the gate. The open items there are already held (IDS-52 extra seeds). Recommended: seed-repeat
  the baselines before spending flows, then report parity-within-0.3 if that is what it shows.
- **Size.** Closing unswr costs about 3x the cells. The paper cannot claim both the b64 accuracy and the 34-bit size
  in the same row.

## 7. Recommendation

**Run A1 (b64, lattice encoff, 3 flows) first.**

- It is the only lever with a large, clean gate (5/5 draws, +1.18), it changes one existing param, and it needs no
  code and no swap.
- Its controls already exist, so only 3 flows are new.
- It does not depend on the IDS56U verdict. A2 does, and should wait for it.

**Queue position.** FIFO is by oldest id, so three new flows run right after IDS56U (7543-7545, last one expected
around 04:00 UTC 13/11) as long as the paused sets stay paused. Their cost to the backlog is their own wall time:
**25-50 h** of delay to the IDS56C carry-over (12 flows) and the 59 held flows if those are resumed afterwards. If
the carry-over is resumed first, its lower ids run first and A1 waits behind it. A cheaper first look exists inside
A1 itself: each flow writes its grid TEST row within minutes of starting, but the adoption line is the GA row only.

## 8. Not verified

- GA wall time and memory at 64 bits (estimate only; the batched trainer has a 16 GB budget and falls back to a slower
  path when exceeded, seen offline at rate 1.0, not at rate 0.25).
- The TEST effect of any arm. Every arm number above is a VAL gate on random genomes trained on the full train
  partition. The flow's search uses 5-fold CV and a GA on top.
- Whether the CV fitness rewards 64-bit cells as VAL does (CV is still rising at 34; not measured above).
- `train_only` ranking on the WNN.
- The classical baselines are single fixed-seed fits; their seed spread on unswr/unswt was not measured here.
- Mechanism in 2g (address collisions) is inferred from the ladder and the category table, not measured per address.
- Thresholds in the A1 adoption rule are a draft: hand-off to `experiment-design`.
