# IDS-77: where the unswr WNN loses to RF raw, and what can close it

Written 23/11/2026 (EST) by ids-security for Luiz (Plane IDS-77). Investigation and offline VAL checks only.
No flow was created, paused, resumed, cancelled or reordered. The worker was not swapped and `src/` was not
touched. Flow 7593 finished at 17:25 EST while this work ran, and 7594 (IDS-70 B s1) is running. Every probe ran at
`nice -n 19` with 4 Rust threads, through the worker's own construction path (`IdsEvaluatorFactory` →
`IDSCacheWrapper`, Rust training and scoring, ABI 29).

**Which partition each number comes from.**
- **Diagnosis (§1–§4) uses VAL and TRAIN only.** The probes rebind the dataset's report slot to VAL
  (`val_as_report`), so no probe scored TEST.
- **TEST numbers** appear only in §0. They are the stored `test_summaries` cells and the classical rows Luiz quoted.
  No lever below was chosen by looking at TEST.
- **Fidelity check.** The re-scored VAL `margin_val_cal` of each production genome equals the IDS-76 re-score
  exactly (for example 7591 GA 95.7611 and 7592 GA 95.5517). The Python confusion matrix also equals the Rust one
  on all 6 genomes.

Scripts and raw outputs are in `logs/ids/ids77/`:
- `wnn_val_diag.py`, `wnn_val_analyze.py`: per-row VAL margins and misses of the WNN genomes.
- `rf_val_cat.py`, `cat_table.py`: RF and XGB per-category recall on VAL.
- `roc_match.py`: recall at a matched FPR.
- `info_budget.py`: information kept per feature under a bigger cut budget.
- `gate_conn.py`, `gate_ladder.py`: the random-genome gates.
- `ens_val.py`: ensembles of the E winners.
- `*.jsonl`: the raw records.

## 0. State (TEST, stored cells; context only)

Cell: GA `ga_neurons`, `best_f1`, `test_point='final'`, `margin_val_cal`, TEST.

```
seed     flow   F1 (%)   FPR (%)  acc (%)   FP    FN    cells       hours   | 80b ctrl flow  F1      FPR     acc
r20421   7591   95.366   0.278    99.332    423   635   1,306,543   11.4    | 7585           95.154  0.296   99.301
r20422   7592   95.466   0.257    99.349    391   641   1,298,510   10.3    | 7586           95.111  0.244   99.304
r20423   7593   95.262   0.270    99.320    411   667   1,290,622   14.7    | 7587           95.173  0.255   99.310
mean            95.365   0.268    99.334                                    |                95.146  0.265   99.305
RF raw top-20 (5 seeds)  95.812   0.288    99.390   XGB raw top-20  95.603  0.427  99.342
```

- **7593 is new.** E won F1 on 3/3 seeds against the control (+0.212, +0.355, +0.089). Mean acc is up +0.029.
- **Mean FPR is +0.003 pp worse** (0.268 vs 0.265). The IDS-70 pre-registered rule (§5) requires mean dFPR ≤ 0 for
  an automatic ADOPT, so at n=3 the rule's outcome is **"Luiz decides"**, not ADOPT. Luiz adopted E at n=2. The
  difference is 0.003 pp, which is about 5 benign rows out of 152K.
- **Gap to RF raw on TEST:** −0.45 pp F1, with FPR better by 0.02 pp. Gap to XGB raw: −0.24 pp F1, with FPR better
  by 0.16 pp.
- **TEST confusion, for scale.** The WNN has the same FP count as RF (391–423 vs RF s20421's 426) and about 90–120
  more FN. On TEST the whole gap is recall. §1 checks this on VAL.

## 1. Where the F1 is lost (VAL)

### 1a. Recall, not precision

Each model at its own F1-optimal VAL threshold (the `margin_val_cal` rule). WNN = the 3 GA `best_f1` genomes.

```
model                    VAL F1   VAL FPR   FP     FN
WNN E GA (3 seeds)       95.681   0.255     389    597
WNN E grid (3 seeds)     95.645   0.275     420    579
RF raw top-20 (3 seeds)  96.229   0.277     423    451
RF on the cut-E bits(1)  96.092   0.291     443    463
XGB raw top-20 (1)       95.626   0.449     684    358
```

Matched-FPR read on the same VAL rows (recall % / macro F1 % at the best threshold with FPR ≤ the bound):

```
source            FPR<=0.20       FPR<=0.25       FPR<=0.30       FPR<=0.40
WNN 7591 GA       88.57 95.583    90.12 95.723    90.37 95.746    92.84 95.595
WNN 7592 GA       88.17 95.468    89.41 95.543    90.14 95.535    90.14 95.535
WNN 7593 GA       88.43 95.545    90.06 95.704    90.27 95.725    90.27 95.725
RF raw 20421      90.12 96.032    91.25 96.083    92.69 96.166    94.48 96.112
RF raw 20422      90.22 96.101    91.80 96.221    92.83 96.218    94.43 96.068
RF raw 20423      90.31 96.092    91.85 96.259    92.91 96.249    94.86 96.142
RF on E bits      89.44 95.862    91.23 96.038    92.51 96.087    94.42 96.024
XGB raw           87.69 95.332    89.21 95.462    90.42 95.503    92.56 95.520
```

- **The gap is recall: about 1.6 pp at the same FPR.** At FPR ≤ 0.25 % the WNN detects 89.4–90.1 % of attacks and
  RF detects 91.3–91.9 %. That is about 100 attack rows out of 6,035.
- **The WNN beats XGB on recall at low FPR on VAL.** XGB's F1 comes from its 0.45 % FPR operating point.
- **The WNN's operating curve has a hole.** On 2 of 3 GA genomes there is no threshold between FPR 0.26 % and
  0.40 % (7592 and 7593 have identical rows at 0.30 and 0.40).

### 1b. The operating point leaves nothing on VAL

- **Sweep check.** For every genome, sweeping every distinct VAL margin gives exactly the `margin_val_cal` F1
  (7591 GA: sweep max 95.761 = val_cal 95.761, at FPR 0.254).
- **Threshold choice is not a lever.** No other threshold on VAL beats val_cal. The VAL → TEST shrink is −0.32 pp for
  the WNN (95.68 → 95.37) and −0.42 pp for RF raw (96.22 → 95.81). The WNN is not losing more in transfer than RF.

### 1c. Which attacks are missed (VAL, detected / support, mean over seeds; the Normal row = FP)

```
category        support |  WNN E GA (3)    WNN E grid (3)   RF raw (3)       RF on E bits(1)  XGB raw (1)
Normal           152391 |   389.3 (0.26%)   419.7 (0.28%)   422.7 (0.28%)    443.0 (0.29%)    684.0 (0.45%)
Analysis            163 |   143.0 (87.7%)   142.7 (87.5%)   145.7 (89.4%)    144.0 (88.3%)    143.0 (87.7%)
Backdoor            135 |   134.0 (99.3%)   134.0 (99.3%)   135.0 (100%)     135.0 (100%)     135.0 (100%)
DoS                 336 |   328.3 (97.7%)   327.3 (97.4%)   330.3 (98.3%)    329.0 (97.9%)    331.0 (98.5%)
Exploits           1706 |  1655.0 (97.0%)  1656.7 (97.1%)  1666.3 (97.7%)   1668.0 (97.8%)   1674.0 (98.1%)
Fuzzers            1331 |   825.3 (62.0%)   847.3 (63.7%)   954.7 (71.7%)    950.0 (71.4%)   1039.0 (78.1%)
Generic            1469 |  1464.3 (99.7%)  1463.7 (99.6%)  1465.0 (99.7%)   1465.0 (99.7%)   1466.0 (99.8%)
Reconnaissance      787 |   786.0 (99.9%)   786.3 (99.9%)   784.7 (99.7%)    782.0 (99.4%)    787.0 (100%)
Shellcode            98 |    92.0 (93.9%)    88.3 (90.1%)    92.3 (94.2%)     89.0 (90.8%)     92.0 (93.9%)
Worms                10 |    10.0 (100%)     10.0 (100%)     10.0 (100%)      10.0 (100%)      10.0 (100%)
total FN                |   597.0           578.7           451.0            463.0            358.0
Fuzzers FN              |   505.7           483.7           376.3            381.0            292.0
```

**The gap is Fuzzers.**
- The WNN misses 146 more attacks than RF raw. 129 of them (88 %) are Fuzzers: Fuzzers recall is 62.0 % for the WNN
  and 71.7 % for RF.
- Exploits add 11 and DoS/Analysis/Shellcode about 5 more. Every other category is level.

### 1d. What the misses look like (VAL, `wnn_val_analyze.py`)

- **The misses are rows where both clusters fire.** The WNN's score per cluster is the fraction of its neurons
  whose cell is TRUE. On the 582–622 FN rows of the GA genomes:
  - the attack cluster fires a median 0.98 of its neurons;
  - the benign cluster fires 0.99–1.00;
  - the margin sits a median 0.008–0.011 below 0.
- **The FPs mirror them**: attack cluster 0.95–0.96, benign cluster 0.89–0.90.
- **Almost all misses sit just below the threshold.** 81–86 % of FN are within 0.05 of τ.
- **The misses are not novel rows.** Only 0–2 FN rows per genome have both clusters below 5 % firing.

**Mechanism (measured, then explained).**
- Under the vote rule with θ = 0.02, a cell commits TRUE whenever its own class is more than 2 % of that address's
  training visits.
- A Fuzzers row whose 80-bit addresses are also visited by benign traffic therefore turns both clusters on.
- The WNN keeps only "this class is not a clear minority here" per address. RF's leaves keep the class proportion.

**Testing whether keeping the proportion helps (§3, gates G3 and G4): it does not.** Changing θ, or grading the
share with a ladder of θ across neurons, moves the genome along the same recall/FPR frontier. On VAL, these Fuzzers
rows overlap benign at 80-bit address granularity.

## 2. Headroom: caps, limits, phases

### 2a. What `max250` caps, and is it binding?

- **What it caps.** `max_neurons=250` caps neurons **per cluster**. There are 2 clusters, so at most 500 neurons
  per genome.
- **Grid.** `neurons: [5, 50, 100, 150, 200, 250]` × `bits: [80]`. The grid's `best_f1` is the 250 × 250 corner on
  all 3 seeds.
- **GA.** Its `best_f1` genome keeps cluster 0 at exactly 250 on all 3 seeds (250/243, 250/249, 250/249).
- **Grid CV F1 by neurons** (k-fold search metric, train only, one draw per size, never a result):

  ```
  flow   n=5     50      100     150     200     250
  7591   93.03   94.60   94.85   95.05   95.13   95.17
  7592   92.72   94.48   94.94   95.11   95.17   95.21
  7593   92.75   94.50   94.93   (<=)    95.12   95.16
  ```

  It is still rising at the cap: +0.04 from 200 to 250.
- **The cap binds.** The random-genome gate (§3, G1) measures what lifting it is worth.

### 2b. Did the GA hit its limits?

- **Generations: no.** The GA ran 90, 80 and 110 of 250. Every flow stopped on the magnitude patience rule (5 checks
  × 10 generations, min_delta 0.05 %). The last checks read ρ = 1.00, a WARNING, with no improvement.
- **Bits.** Bits are fixed by design (`min_bits = max_bits = 80`). All neurons are 80 bits wide.
- **Neurons.** Cluster 0 is pinned at the 250 cap.
- **The GA buys almost nothing on held-out data**, comparing the GA `best_f1` genome with the grid `best_f1` genome:

  | where measured | GA − grid | detail |
  |---|---|---|
  | TEST | +0.031, +0.150, −0.012 | mean +0.06 |
  | VAL | +0.096, −0.149, +0.160 | mean +0.04 |
  | CV, own folds (IDS-76; 7591, 7592) | +0.38, +0.32 | |
  | CV, fresh folds (IDS-76; 7591, 7592) | +0.28, +0.27 | |

  Five *random unsearched* 250 × 250 × 80 genomes average **95.642** on VAL. The GA winners average 95.681. That is
  10–15 h of GA per flow for about +0.04 pp held-out.

### 2c. Phases that ran and phases that exist

- **What ran.** The template is `ids-binary-2-phase`: Grid Search (`optimize bits/neurons/connections = False`), then
  GA Neurons (`neurons=True, bits=False, connections=False`).
- **Connections are never mutated.** A neuron's connections are drawn once, uniformly per bit, and only move between
  genomes by crossover.
- **A GA Connections phase exists.** The `ids-binary-5-phase` and `ids-binary-7-phase` templates add GA Bits /
  Synaptogenesis / GA Connections (`optimize_connections=True`).
- **No IDS flow has run it since flows 430–433.** The DB holds 4 `ga_connections` experiments on IDS flows, all in
  flows 430–433, against 5,796 `ga_neurons` experiments.
- **Why it was dropped (`docs/ids57_options.md`).** The connections operator only shifts a connection by ±2 bit
  positions. Under thermometer cuts that is the neighbouring threshold of the same feature, so it is a very local
  move.
- **The cheaper question: does where connections land matter at all?** Gate G2 asks it with random draws and finds a
  small effect at most (§3). A GA Connections phase can only exploit what that gate can see, so it ranks low.

## 3. Offline VAL gates (random unsearched genomes; cut-E encoding; 7591 params; differences paired by draw)

- **Genomes.** 5 draws, seeds 1000–1004, built like grid genomes: 2 clusters, connections uniform per bit with
  replacement.
- **Training.** Rust trains each genome on the full TRAIN partition and scores it on VAL.
- **Readout.** The F1-optimal VAL threshold, i.e. the `margin_val_cal` rule.
- **Baseline.** 250 × 250 × 80 bits, θ = 0.02, per-bit draw: **95.642 / FPR 0.305 / recall 91.03 / Fuzzers 66.0 %**
  (mean of 5 draws).
- **Draw-to-draw SD** of the baseline is 0.07 pp.

```
gate  variant                                   F1 (VAL)  dF1     wins  FPR    recall  Fuzzers  cells vs base
G1    500 x 500 x 80, per-bit                   95.782    +0.140  5/5   0.313  91.70   68.6 %   ~2x
G1    750 x 750 x 80, per-bit                   95.763    +0.121  4/5   0.329  91.98   69.7 %   ~3x
G1    1000 x 1000 x 80, per-bit                 95.803    +0.161  5/5   0.325  92.04   69.8 %   ~4x
G1    500 x 500 x 96, per-bit                   95.791    +0.148  5/5   0.300  91.46   68.0 %   >2x
G1    250 x 250 x 64, per-bit                   95.533    -0.109  1/5   0.295  90.43   63.7 %   <1x
G2    per-feature draw (uniform over 20 feats)  95.384    -0.258  0/5   0.402  92.18   70.4 %   1x
G2    sqrt(bits)-weighted feature draw          95.690    +0.048  5/5   0.318  91.49   67.3 %   1x
G2    50% uniform-feature + 50% RF importance   95.321    -0.322  0/5   0.347  90.79   65.2 %   1x
G2    per-feature draw at 500 x 500             95.519    -0.124  0/5   0.339  91.32   66.4 %   ~2x
G2    sqrt draw at 500 x 500                    95.733    +0.091  4/5   0.289  91.00   65.3 %   ~2x
G3    theta 0.01                                95.656    +0.014  4/5   0.300  90.98   65.8 %   1x
G3    theta 0.05                                95.625    -0.017  0/5   0.325  91.41   67.8 %   1x
G3    theta 0.10                                95.599    -0.043  1/5   0.393  92.77   73.7 %   1x
G4    theta ladder .02/.1/.2/.35/.5 (5 x 50n)   95.464    -0.185  0/5   0.461  93.75   79.3 %   1x
G4    theta ladder .02/.05/.1/.2/.3             95.599    -0.050  1/5   0.465  94.32   80.7 %   1x
G4    theta ladder .02 x3 / .2 / .5             95.548    -0.101  1/5   0.440  93.60   78.7 %   1x
G4    all theta 0.5 (majority vote)             95.139    -0.510  0/5   0.470  92.76   77.4 %   1x
G5    ensemble of 2 E GA winners (3 pairs)      95.749    +0.068* -     0.263  90.51   63.7 %   2x
G5    ensemble of the 3 E GA winners            95.761    +0.080* -     0.254  90.36   63.2 %   3x
```

\* G5 is measured against the mean single E GA winner (95.681), not against the random baseline.

**Validity checks.**
- **Determinism.** The G3 θ = 0.02 row reproduces the baseline exactly on every draw (dF1 = 0.000).
- **G4 split.** The G4 control (the same 2 × 250 genome cut into 5 sub-genomes of 2 × 50, all at θ = 0.02) reads
  95.649 against the whole genome's 95.642. Scoring the 5 parts separately is the same model.

**Reading the gates.**
- **G1, capacity, is the only lever that passes.** At 500 per cluster it gains +0.14 pp, on 5/5 draws.
  - **It saturates at 500.** Against 500 × 500 × 80 on the same draws: 750 per cluster −0.02 (1/5),
    1000 per cluster +0.02 (3/5), and 96 bits at 500 per cluster +0.01 (2/5). The random-genome ceiling of this
    architecture on VAL is about **95.78–95.80**: 0.43 pp under RF raw (96.23) and 0.30 under RF on the same bits
    (96.09).
  - Fuzzers recall rises +2.6 pts and FPR barely moves.
  - It doubles the cells: about 2.6M (estimated from the 250-per-cluster E genomes' 1.29–1.34M, not measured on
    the gate genomes). That is roughly the old quantile-80b control's footprint (2.74–2.95M).
  - Cut E freed exactly this memory.
- **Width is not the lever.** 64 bits loses 0.11 pp (1/5 draws). Note that IDS-71 (cut E at ≤64 bits) is already
  queued.
- **G2, connectivity: no.**
  - A uniform-per-feature draw, or RF-importance weighting, starves the wide numeric features. It raises Fuzzers
    recall but costs F1 through FPR.
  - The production draw is per bit. Because cut E gives sbytes 379 bits but sttl only 5, about 70 % of neurons never
    see sttl, the feature that carries 86 % of XGB's gain. Even so, re-weighting does not help.
  - Only the mild sqrt weighting is positive: +0.05, 5/5, which is within about 1 baseline SD. It does not add on
    top of capacity: at 500 × 500 it is −0.05 against the per-bit draw (2/5).
  - This repeats the IDS-57 finding on the quantile encoding.
- **G3 and G4, the write rule: no.**
  - θ is flat from 0.01 to 0.10.
  - Every θ ladder (a graded per-address share at equal cells) catches more Fuzzers, up to 80.7 %, but pays in FPR.
    Each lands at or below the baseline F1.
  - The Fuzzers rows overlap benign at 80-bit address granularity. A finer share readout does not separate them.
- **G5, ensembles: about half of G1 for the same 2x cells.** IDS-62 already did not adopt the seed ensemble.

## 4. Inputs: is RF's advantage in features the encoding loses?

- **The encoding is not where the gap is.**
  - RF trained on the cut-E bit matrix (the same 1104 bits the WNN reads) scores **96.092** on VAL.
  - RF on raw values, same seed, scores 96.174.
  - The encoding costs RF 0.08 pp. The WNN on the same bits scores 95.68, **0.41 pp below RF on identical inputs.**
  - Same seed, raw → E costs RF 25 Fuzzers detections (975 → 950). The WNN is 125 Fuzzers below RF-on-E.
  - **Most of the gap (about 0.41 of 0.55 pp on VAL) is the learner, not the input.**
- **A bigger E budget for the top features does nothing**, because the top features have no more cuts to give.
  - **TTL group: only a handful of bits.** sttl (13 unique values), ct_state_ttl (7) and dttl (10) carry 51 % of RF
    importance and 89 % of XGB gain. E already gives them 5/5/4 bits: every cut that has information gain on TRAIN.
  - **Discrete features: already near-lossless.** Reallocating 2x / 4x the numeric budget on TRAIN
    (`info_budget.py`, I(bins; attack_cat) relative to the raw values) raises the share of information kept by
    +0.3..+2.5 pts at 2x and +0.4..+3.6 pts at 4x for sbytes, dmeansz, smeansz, Dpkts and Spkts. These are already at
    91–99 %.
  - **Continuous timing features: tiny gains too.** Dintpkt, dur, Sintpkt, Sjit and tcprtt sit at 34–66 % of the
    raw-value plug-in information. That plug-in number is inflated: each of their 0.4–0.8M unique values counts as a
    bin. Even 4x the bits adds only 1.3–2.7 pts.
- **How connections spread over features (GA genomes, mean of 3).** Connections follow bits (connection share =
  bit share within 0.4 pts), and all 1104 bits are used:

  ```
  feature          RF imp %  XGB gain %  E bits  bit %   WNN conn %
  sttl              22.26     86.38        5     0.45     0.53
  ct_state_ttl      19.25      2.99        5     0.45     0.55
  dttl               9.58      0.01        4     0.36     0.35
  dmeansz            7.52      0.74      154    13.95    14.16
  tcprtt             7.21      0.26       17     1.54     1.58
  sbytes             4.72      2.14      379    34.33    34.19
  smeansz            4.66      1.40      152    13.77    14.12
  (13 more features, each <= 3.9 % RF importance)
  ```

  G2 shows that pushing connections toward the TTL group does not help. A thermometer TTL bit already splits the
  classes in one look, and the neurons that see none still vote usefully on the other features.

## 5. Other items

- **Fold rotation (IDS-76).**
  - On its own folds the GA's CV edge over grid is +0.38 / +0.32 (7591 / 7592). On fresh folds it is
    +0.28 / +0.27.
  - The fold-specific part (0.05–0.10) is about the size of the whole held-out GA effect (+0.04 to +0.06).
  - Rotation can at best turn more of the CV edge into held-out gain. Bound: ≤ +0.1 pp. It cannot reach 0.45.
- **Fitness weights on unswr.** The GA's total held-out effect is about +0.05, so re-weighting what the GA chases is
  bounded by the same small number. Not ranked.
- **Threshold mode.** val_cal is already the VAL optimum (§1b). The 7-mode tables are context only.
- **QUAD memory.** Not proposed. Luiz fixed BINARY for unswr on 07/11, and the vote rule is BINARY/TERNARY/PLN only.
  G4 asked the question QUAD would answer (a graded cell) and found no F1 in it.

## 6. Ranked levers

Effects are VAL gate differences on random genomes unless stated. How these gate effects transfer to TEST is not
established. The bounds below are engineering guesses, stated as such. **No measured lever closes the 0.45 pp TEST gap to RF raw.**

| # | Lever | Gap it targets | Evidence | Expected TEST effect (honest bound) | Exact change / code area | Cost | Screen on unswt first? |
|---|---|---|---|---|---|---|---|
| **1** | **max_neurons 500 on cut E, fixed 80b** | Fuzzers/benign collisions: more independent voters (Fuzzers +2.6 pts at the gate) | G1: **+0.140 VAL, 5/5**, FPR +0.008; the grid best and the GA cluster 0 sit on the 250 cap on 3/3 seeds; grid CV still +0.04 from 200 to 250 | **0 to +0.15 pp F1**, FPR flat to +0.03. Lands about 95.4–95.5. Reaches XGB-raw F1 (95.60) only at the top of the bound. Does **not** reach RF raw | Clone 7591/7592/7593 `config_json` verbatim, change only `max_neurons: 250 → 500`. Same template, experiments array, seeds r20421-23. The worker derives the grid neuron list from min/max (250 gave [5..250]): check the list in the first flow's log. Cells about 2.6M, the old 80b control's footprint. No code, no swap | Per flow about 2–2.6x today's 10–15 h: offline training of one genome went 5.0 s → 12.9 s at 2x neurons. So **~25–40 h per flow, ~75–115 h for n=3**. The grid row lands in the first ~10 minutes of each flow | **No: must run on unswr.** IDS-57 killed n500 on unswt (−0.03, 175K train rows, addresses already thin) |
| 2 | **Fold rotation (IDS-76)** | The GA's CV edge that does not transfer | IDS-76: own-fold edge +0.38/+0.32, fresh-fold +0.28/+0.27, held-out GA−grid only +0.04 (VAL) / +0.06 (TEST) | **≤ +0.1 pp**. Most of the CV edge is not fold-specific, and the whole held-out GA effect is +0.05 | Already its own story (IDS-76); no new params from here | as IDS-76 | unswt can show the mechanism; the size needs unswr |
| 3 | sqrt(bits)-weighted feature draw for connections | Starvation of the low-cardinality TTL features under cut E (about 70 % of neurons never see sttl) | G2: +0.048 at 250 (5/5, about 1 baseline SD), **−0.049 at 500** (not additive with lever 1) | **About 0 to +0.05**, and only without lever 1 | Needs code. A new `KNOWN_PARAMS` key (e.g. `ids_connection_draw`) in the grid/init/immigrant genome builders (Python) and in the Rust new-neuron draw of the GA operators. That is a **worker wheel swap**. Not worth it at this size | code + swap + 3 flows | either; the effect is under one SD |
| 4 | Ensemble of 2 cut-E winners | Same as lever 1, by averaging two searched genomes | G5: pairs +0.068 VAL over the single winner at 2x cells (3 pairs: −0.036, +0.046, −0.015 vs their better member); triple +0.080 at 3x | Dominated by lever 1 (half the gain for the same cells). IDS-62 seed ensemble already not adopted | Needs the IDS-62 ensemble scorer (`score_multiclass_sets` is absent from the installed ABI-29 wheel) | 0 new flows (re-scores 7591-93) | n/a |

**Killed at the VAL gate on unswr (do not queue):**
- θ re-tune 0.01–0.10 (G3: −0.04..+0.01).
- θ ladder / graded per-address share at equal cells (G4: −0.05..−0.19). This is also the question QUAD would
  answer.
- Per-feature or RF-importance connection draws (G2: −0.26 / −0.32).
- A larger cut-E budget for the top features (§4: the TTL group is already at its maximum useful cuts; the others
  gain ≤ 3.6 pts of information at 4x the bits).
- 1000 or 750 per cluster, and 96 bits at 500 per cluster: no gain over lever 1 (G1 saturates at 500).
- Threshold-mode changes (§1b).
- Fitness re-weighting (bounded by the GA's +0.05 held-out effect).
- A GA Connections phase (±2-bit operator; G2 shows where connections land barely matters).

**Prediction for the already queued IDS-71** (cut E at ≤64b, config C), from the gate: 250 × 250 × 64 per-bit is
**−0.11 VAL** against 80b (1/5). Expect IDS-71 to trade about 0.1 pp F1 for fewer cells. That is stated before its
readout, so it can be checked.

## 7. Bottom line

- **On VAL the WNN trails RF raw by 0.55 pp F1. It is ahead on FPR at the val_cal point (0.255 vs 0.277).**
- **The whole deficit is recall**: about 1.6 pp at matched FPR, and **88 % of the extra misses are Fuzzers** (62 %
  vs 72 % recall).
- **The misses are not novel rows.** Their 80-bit addresses are shared with benign traffic, so both clusters fire.
- **About 0.08 pp of the deficit is the cut-E encoding.** RF on the same 1104 bits scores 96.09.
- **About 0.41 pp is the learner on identical inputs.**
- **What does not help, measured:**
  - a finer per-address readout (θ, θ ladder);
  - steering connections toward RF's important features;
  - a bigger cut budget;
  - the GA itself (+0.04 held-out over random draws).
- **The only lever that passes is more neurons per cluster.** +0.14 pp VAL at 500, saturating there.
- **The random-genome ceiling of this architecture on VAL is about 95.80, still 0.43 under RF raw.**
- **So surpassing RF raw on F1 is not in reach of any lever measured here.**
- **What is realistic**, if lever 1 transfers:
  - matching XGB raw on F1, with FPR about 0.16 pp lower;
  - keeping FPR at or under RF's, at about 2.6M cells (vs RF's 93 MB, 1.16M nodes).
- **For the paper's comparator rows, that is a Pareto claim (FPR × size), not an F1 win.** Whether that framing is
  safe to state goes to `experiment-design`, together with the IDS-70 rule outcome at n=3 (F1 3/3 but mean FPR
  +0.003, so the pre-registered result is "Luiz decides").
