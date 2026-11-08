# IDS-52 x IDS-53 A/B, re-planned IDS-49, IDS-48: pre-registered design (08/11/2026)

Status: DESIGN ONLY. No flows created and nothing committed. The DB was read read-only.
Author: experiment-design agent, for Luiz. Every number below is an actual DB read unless marked JUDGEMENT.

---

## 0. Facts (DB / code / doc reads)

### F1. IDS-48 on unswt is a no-op by construction (new finding)
- The unswt (`unsw-nb15`, `temporal_3way`) train partition has **175,341 rows** (docs/ids_top20_train_only/unsw-nb15_temporal_3way.json, branch /private/tmp/ids_fix_wt).
  - A fold-train is 4/5 of that, about 140k rows.
  - `ids_calib_sample=200000` is larger than the fold, so `stratified_rows` returns the full set (adaptive/calib_sample.rs:79, `sample >= n`), and the path is bit-identical.
- `balance_classes` reweights rows and does not duplicate them (flow.py:238), so the row count stays below 200k.
- Measured on pair r25052 (7348 full vs 7349 sub): **identical genome hashes** for grid and GA final, best_f1 and best_fitness (23ba9a09…, ae5f6e37…, c23f9b5b…), and identical val_cal 85.88 F1 / 18.41 FPR. GA time was 0.45 h vs 0.51 h, so there was no speed-up.
- The queue script's docstring ("~1.6M k-fold train rows") is wrong for unswt.
- Train sizes where the subsample does bite:

```
    dataset (3way train)   rows       fold-train   200k share
    cicids random_3way     2,262,300  ~1.81M       11%
    ciciot subsample       1,143,802  ~0.92M       22%
    unswr random_3way      1,267,407  ~1.01M       20%   (BINARY two-cluster: calib_sample REJECTED)
    unswt temporal_3way      175,341  ~0.14M       >100% -> full set, no-op
```

### F2. Recipes (flows.config_json, template `ids-binary-2-phase`, 2 experiments: grid_search + ga_neurons)

```
    Dataset / config                        Base flow  Key params
    unswt  BINARY bn  (IDS-46 max250)       7125       unsw-nb15 temporal_3way 16b, BINARY lattice, single_cluster=false,
                                                       max_neurons 250, max_bits 34, xo blend_neuron, rate 0.25, Wb-CTRL weights
    unswr  BINARY bc  (IDS-47)              7248       unsw-nb15 random_3way 64b, BINARY lattice 2m, max250, max_bits 34,
                                                       xo blend_cluster, rate 0.25, desirability WbCTRL (.10/.20/.35/.35)
    cicids QUAD 1m    (IDS-47)              7343       cicids2017 random_3way 96b, single_cluster, max_neurons 500,
                                                       max_bits 34, rate 1.0, zscore CE20 (.20/.10/.30/.40)
    ciciot QUAD 1m    (IDS-47)              7344       ciciot neto_subsample random_3way 96b, single_cluster,
                                                       max_neurons 250, max_bits 100, rate 1.0, desirability B15AC
    ciciot BINARY bc  (IDS-47)              7244       as 7344 but BINARY lattice 2m, max250, xo blend_cluster, rate 0.25
```

- Queued IDS-49 flows 7368-7379 are exact clones of the IDS-47 QUAD flows with only `ga_crossover_bits` added (diff 7343→7368, 7344→7374).
- IDS-47 QUAD r20421 flows (7237/7241/7245) and 7233 lack `ids_offspring_folds`, but the default has been "all" since ABI 20, so the recipe is the same.
- IDS-47 arms mix worker ABIs 21-24 within an arm.
- **unswt cap is open.** IDS-47 used per-cluster max250 with no total cap. The IDS-45 tot250 arm (7333-7342) adds `max_neurons_total=250`. The memory note records the cap decision (max250 vs tot250 vs tot500) as pending. This design uses 7125 (max250) and needs a one-word confirmation from Luiz.

### F3. Per-flow wall time, old wheel (completed_at − started_at)

```
    Cell                          flows                      h/flow (each)           mean
    unswt BINARY bn max250        IDS-46 r20419-24           0.39-0.43               0.42
    unswr BINARY bc               7226/7236/7248             3.95/6.19/3.60          4.58
    cicids QUAD 1m                7343/7346 (7237 = 28.9)    5.26/5.80               5.53  (7237 outlier excluded, unexplained)
    ciciot QUAD 1m                7344/7347/7241             16.75/19.57/13.39       16.57
    ciciot BINARY bc              7223/7232/7244             12.99/15.00/19.73       15.91
    unswt QUAD (IDS-48)           7348/7349                  0.47/0.51               0.49
```

### F4. Between-seed spread of the current controls (GA phase, final, held-out TEST partition, n=3 unless stated)

```
    Control                          genome x mode           F1 %          FPR %
    unswt BINARY bn max250 (n=7)     best_f1 x margin_val_cal 90.82±0.09    5.26±1.23
    unswr BINARY bc                  best_f1 x margin_val_cal 94.23±0.19    0.21±0.02
    cicids QUAD 1m                   best_f1 x val_cal        99.54±0.05    0.08±0.03
    ciciot QUAD 1m                   best_f1 x val_cal        92.90±0.30    7.39±0.43
    ciciot BINARY bc                 best_f1 x argmax         92.60±0.09    4.12±0.25
    ciciot BINARY bc                 best_f1 x margin_val_cal 92.68±0.12    7.82±1.27
```

All the rate-0.25 rows above predate the IDS-50 fix (positional sampling mask), so their spreads include that noise.

### F5. Ranking lists
- unswt legacy list vs train_only (both ranked on the same 175,341 train rows): **12/20 overlap**.
  - The legacy list contains `proto/service/state`; the train_only list contains none of them.
  - So for unswt, train_only changes the ranking *procedure* (categorical handling). It does not fix a leak.
- unswr: legacy has 0 overlap with test/val, and agrees with train_only on 13/20.

---

## 1. IDS-52 x IDS-53: reduced design, not a full factorial (JUDGEMENT)

**Ranking (IDS-53).** The factor is not a candidate on any dataset.

```
    dataset  ranking        why
    cicids   train_only     legacy = selection leak (76% of test/val rows in ranking partition) -> correctness, fixed
    ciciot   train_only     legacy = selection leak (84%) -> correctness, fixed
    unswr    legacy         0 overlap: clean. train_only is a different list (13/20), not a fix -> no reason to change
    unswt    legacy         ranked on its own train: clean. 12/20 = procedure change (F5), not a fix
```

**Leak size, cheap version.** Measuring it is a footnote, not a decision. Two parts:
- (a) A free, confounded estimate: IDS-47 cicids/ciciot QUAD r20419-21 (legacy, old wheel, rate 1.0, so IDS-50 does not apply) vs the new train_only dedupe-off QUAD cells at the same seeds. The only extra confound is the QUAD CPU determinism fix.
- (b) A model-agnostic estimate in minutes: RF/XGB on legacy vs train_only lists, 5 seeds each.
- (b) is needed anyway: **the RF/XGB/Ada/MLP baselines for cicids/ciciot must be recomputed on the train_only lists** for parity with the new WNN numbers.
- Optional WNN-specific cell: `IDS53-cicids … rklegacy` at n=3 (3 flows, about 17 h). Run it only if (a) and (b) disagree, or a reviewer-facing number is wanted.

**Encoding (IDS-52).** This is the one real factor. Each dataset gets an off/on pair at its production config, paired by seed.

```
    Stage  dataset  config                        cells (paired by seed)                 n     seeds
    1      unswt    BINARY bn max250, legacy      enc off | enc on                       20    20419-20438
    1      unswr    BINARY bc, legacy             enc off | enc on                        8    20419-20426
    1      cicids   QUAD 1m inherit, train_only   off-calibfull | off-calibsub | on-calibfull  5  20419-20423
    3      ciciot   chosen config (Stage 2)       the opposite encoding to E*             3    20419-20421 (pairs with Stage 2)
```

Why these n (rough one-sided 95% t, true Δ=0, σΔ ≈ √2·σ_seed from F4, assuming no pairing gain):
- unswt FPR σΔ≈1.7 → UB 0.66 pp at n=20, vs margin 1.32.
- unswr F1 σΔ≈0.27 → LB 0.18 at n=8, vs margin 0.25. At n=5 it would be 0.26, a coin flip.
- cicids F1 σΔ≈0.07 and FPR σΔ≈0.04 → 0.07 / 0.04 at n=5, vs 0.25 / 0.05.
- ciciot n=3 is a sign/disaster check only.

**Primary readout, fixed now.** Genome `best_f1`, GA phase final, held-out TEST partition.
- QUAD: `val_cal`, which Protocol v2 calibrates on VAL.
- unswt/unswr BINARY: `margin_val_cal`.
- ciciot BINARY: `argmax`, Luiz's low-FPR operating point.

**Secondary readout.** Reported, but it does not decide anything:
- Each arm at its own best genome_type × mode, labelled best-of-35 for QUAD and best-of-50 for BINARY.
- The low-FPR zone: best F1 at FPR<1% (cicids/unswr) or FPR<5% (ciciot/unswt), at a genome×mode fixed by seed-mean, not per-seed best.
- The full Rule-7 tables (all modes).
- Effective input bits, GA h/flow, and generations.

**Pre-registered adoption rule for `ids_encoding_dedupe=on`.**
- Δ = on − off per seed pair, on the primary readout, one-sided 95% paired t bounds.
- Margins: δF1 = 0.25 pp everywhere. δFPR = max(0.05 pp, 25% of the control FPR in F4), fixed now:

```
    unswt 1.32 pp · unswr 0.05 pp · cicids 0.05 pp · ciciot 1.85 pp (QUAD val_cal) / 1.03 pp (BINARY argmax)
```

- Non-inferior (NI) = LB(ΔF1) > −δF1 **and** UB(ΔFPR) < +δFPR.
- Inferior (INF) = UB(ΔF1) < −δF1 **or** LB(ΔFPR) > +δFPR.
- Anything else = INCONCLUSIVE.
- **Adopt dedupe for all four datasets** if no Stage-1 dataset is INF and at least 2 of the 3 are NI.
  - Superiority is not required. The case for dedupe stands on its own: it removes dead and duplicate bits, roughly halves the input, makes "bits" findings mean effective bits, and shrinks the FPGA encoder.
  - **Luiz must OK the non-inferiority framing before any data exists.** If he wants superiority, change it now.
- Otherwise adopt per dataset only where NI.
- Stage 3 ciciot reverts to off only if it is INF at n=3.
- Superiority is reported separately: a two-sided 95% CI that excludes 0 in the good direction.

**IDS-48' (calibration subsample), nested in cicids Stage 1.**
- Pair: off-calibfull vs off-calibsub200k.
- Rule: the original pre-registered one from scripts/queue_ids_calib_paired.py. 95% CI of ΔF1 and ΔFPR both inside ±0.10 pp, two-sided, now t(0.975,4)=2.776 at n=5. Report the GA time ratio.
- Equivalence on cicids is assumed to transfer to ciciot QUAD (22% subsample). This is flagged.

---

## 2. IDS-49 re-planned (IDS-49b)

- **Cancel 7368-7379.** Their control is legacy lists on the old wheel.
- Run IDS-49b **after** the dedupe decision, under the adopted encoding E*. The crossover question is equally clean at any fixed encoding, but only E* ships. Running it under off risks a crossover × encoding interaction that would never be tested.
- The inherit control already exists from Stage 1 for both values of E*, so this order costs no extra flows.

```
    cicids  QUAD 1m, train_only, calibfull, E*:  xo blend_cluster | blend_neuron        r20419-21   6 flows
            control = Stage-1 cicids inherit E* calibfull r20419-21 (no new flows)
    ciciot  QUAD 1m, train_only, calib C*, E*:   xo inherit | blend_cluster | blend_neuron   r20419-21   9 flows
            BINARY bc 2m, train_only, rate 0.25, E*:  (BINARY vs QUAD on the fixed wheel)   r20419-21   3 flows
```

- **Why ciciot BINARY bc is rerun.** The ciciot BINARY-vs-QUAD decision is compared against IDS-47 BINARY bc, which is rate 0.25 on the pre-IDS-50 wheel and legacy lists, so that comparison is invalid.
- **C\*:** calibsub200k if IDS-48' passes, else calibfull. Within ciciot all QUAD cells are new, so there is no mixing. BINARY two-cluster rejects calib_sample, which is accepted because the configs differ anyway.
- **Pre-registered IDS-49b read, at n=3, so provisional by construction.**
  - A crossover rule replaces inherit only if mean ΔF1 > +0.10 pp on the primary readout, 3/3 seeds have the same sign, and mean ΔFPR < +δFPR.
  - Otherwise inherit stays, since it is the default.
  - ciciot BINARY vs QUAD is Luiz's call, on the primary readout plus the low-FPR zone, labelled n=3.
- **Optional cut (JUDGEMENT; default is to run it as Luiz asked).** If cicids IDS-49b shows no separation under the rule above, the ciciot blend_cluster and blend_neuron cells add little: 6 flows, about 100 h. The prior is weak separation, since IDS-46/47 BINARY showed none.

---

## 3. IDS-48

- **Do not rerun it on unswt.** Cancel 7350-7367; 7350 is a no-op, so cancel rather than wait, and the swap gets its idle point sooner.
- 7348/7349 already show the no-op, with identical hashes.
- The question is real only where the fold-train exceeds 200k rows. It moves to cicids as IDS-48' (§1), reusing the dedupe-off control flows, so it costs only 5 extra flows.
- On whether to keep it legacy/dedupe-off: moot on unswt. On cicids it is train_only (mandatory) and dedupe off.

---

## 4. Order and worker time (serial FIFO; means from F3, assuming the new wheel and dedupe do not change h/flow)

```
    Step                                                    flows   est. h    cum. days
    0  cancel 7350-7367 (IDS-48 unswt) + 7368-7379 (IDS-49); swap ABI 25/26 at idle
    1a unswt  enc off/on x20 (seed-major, off then on)         40     16.8      0.7   <- full dataset readout in <1 day
    1b rounds k=1..8: [cicids off-full, off-sub, on-full (k<=5)], [unswr off, on]
                                                               31    155.7      7.2
       (optional IDS53 cicids legacy n=3)                     (+3)  (+16.6)
    -- decision: E* (dedupe), C* (calib)
    2a cicids IDS-49b  rounds r=1..3: [bc, bn]                  6     33.2      8.6
    2b ciciot rounds r=1..3: [Q-inh, Q-bc, Q-bn, B-bc]         12    196.8     16.8   (QUAD part shorter if C*=sub)
    -- decision: ciciot config
    3  ciciot opposite-encoding check at chosen config x3      3    ~49       18.8
    TOTAL before cohorts                                       92    ~451 h   ~18.8 days  (~14.7 d with the 2b cut)
```

- Seeds 20419+ at the final config are byte-identical to the cohort recipe, so they count toward the n=100 cohort (seeds 20419-20518).
- **Remaining n=100 cohort time at F3 rates:**

```
    unswt   80 x 0.42 =   34 h
    unswr   92 x 4.58 =  421 h
    cicids  95 x 5.53 =  525 h  (less if calibsub adopted)
    ciciot  97 x ~16  = 1550 h  (~65 days alone)
    total  ~2,530 h ≈ 105 days
```

- The A/Bs are about 18% of that. **The ciciot n=100 is the critical path**; that is flagged for Luiz, not decided here.

**Pre-swap results that must not be cited, and their replacements.**
- **Rate 0.25 (IDS-50 positional mask):**
  - Affected: all BINARY lattice work (IDS-40/44/45/46 unswt, IDS-47 BINARY ×27), about 1,519 flows since id 4000.
  - Replaced by the Stage-1/2 new-wheel cells. Not rerun wholesale.
  - The crossover and cap "no separation" verdicts from IDS-46/47 are relabelled **provisional (measured under positional-mask noise)**.
- **cicids/ciciot on legacy lists (IDS-53 selection leak):**
  - Affected: every cicids/ciciot WNN number (IDS-47 QUAD + BINARY, IDS-4/5/Z/X/D cohorts) and the cicids/ciciot RF/XGB/Ada/MLP baselines.
  - Superseded by Stage 1/2 and the cohorts. The baselines need a rerun on the train_only lists.
- **IDS-47 unswr QUAD (rate 1.0, clean list):** not affected by IDS-50/53. Possibly touched by the QUAD determinism fix. No longer the unswr config, so nothing to do.
- **IDS-48 unswt:** a no-op, so there is nothing to rerun.
- **PUB50-ciciot (IDS-51, swapped weights):** historical. Stop citing it as comparable to the other PUB50 batches.

---

## 5. Flow specs, ready to POST (clone the base flow's config_json, change only the listed keys)

- Body shape as in scripts/queue_param_clone.py.
- `config = {template: "ids-binary-2-phase", params}`.
- `experiments = [{"name":"Grid Search (neurons x bits)","experiment_type":"grid_search","phase_type":"grid_search"}, {"name":"GA Neurons","experiment_type":"ga","phase_type":"ga_neurons"}]`.
- Set both new params **explicitly** in every flow. Dry-run first to verify that an explicit "off"/"legacy" is byte-identical to absent.
- Creation order = run order.

```
STAGE 1a  base 7125 (unswt BINARY bn max250)   seeds 20419..20438, for each seed: off then on
  IDS52-unswt-BINARY-lattice-2m-max250-xo_blend_neuron-16bWb-rate0.25-rklegacy-enc{off|on}-r{seed}
  diff: seed={seed}, ids_encoding_dedupe="{off|on}", ids_feature_ranking="legacy"

STAGE 1b  round k = 1..8 (seed s = 20418+k):
  if k<=5, base 7343 (cicids QUAD 1m):
    IDS52-cicids-CE20-zs-QUAD-1m-rktrain-encoff-calibfull-r{s}
      diff: seed=s, ids_encoding_dedupe="off", ids_feature_ranking="train_only"
    IDS48c-cicids-CE20-zs-QUAD-1m-rktrain-encoff-calibsub200k-r{s}
      diff: seed=s, ids_encoding_dedupe="off", ids_feature_ranking="train_only", ids_calib_sample=200000
    IDS52-cicids-CE20-zs-QUAD-1m-rktrain-encon-calibfull-r{s}
      diff: seed=s, ids_encoding_dedupe="on",  ids_feature_ranking="train_only"
  base 7248 (unswr BINARY bc):
    IDS52-unswr-WbCTRL-desir-BINARY-lattice-2m-max250-xo_blend_cluster-rate0.25-rklegacy-enc{off|on}-r{s}
      diff: seed=s, ids_encoding_dedupe="{off|on}", ids_feature_ranking="legacy"
  (optional, k<=3) IDS53-cicids-CE20-zs-QUAD-1m-rklegacy-encoff-calibfull-r{s}
      diff: seed=s, ids_encoding_dedupe="off", ids_feature_ranking="legacy"

STAGE 2a  base 7343, rounds r = 20419..20421: blend_cluster then blend_neuron
  IDS49b-cicids-CE20-zs-QUAD-1m-xo_{bc|bn}-rktrain-enc{E*}-calibfull-r{r}
      diff: seed=r, ga_crossover_bits="blend_cluster|blend_neuron", ids_encoding_dedupe=E*, ids_feature_ranking="train_only"

STAGE 2b  rounds r = 20419..20421, order Q-inh, Q-bc, Q-bn, B-bc
  base 7344:  IDS49b-ciciot-B15AC-desir-QUAD-1m-xo_{inherit|bc|bn}-rktrain-enc{E*}-calib{C*}-r{r}
      diff: seed=r, [ga_crossover_bits for bc/bn only; inherit = key absent, as in the controls],
            ids_encoding_dedupe=E*, ids_feature_ranking="train_only", [ids_calib_sample=200000 iff C*=sub]
  base 7244:  IDS49b-ciciot-B15AC-desir-BINARY-lattice-2m-max250-xo_blend_cluster-rate0.25-rktrain-enc{E*}-r{r}
      diff: seed=r, ids_encoding_dedupe=E*, ids_feature_ranking="train_only"

STAGE 3   chosen ciciot config, seeds 20419..20421, encoding = NOT E*; name prefix IDS52-ciciot-…-enc{¬E*}-r{r}
```

**Verify after each POST batch:**
- COUNT(experiments)=2 per flow.
- Params diff vs base = exactly the listed keys.
- `ids_feature_ranking="train_only"` on every cicids/ciciot flow.
- Worker ABI ≥ the new wheel on the first completed flow's validation_summaries.worker_abi.

---

## PRE-REGISTRATION — FIXED 08/11/2026 by Luiz, BEFORE any dedupe-on flow exists

- **IDS-52 dedupe adoption = NON-INFERIORITY** (the recommended framing, accepted). Primary readout: best_f1 genome on the held-out
  TEST partition, threshold mode fixed per dataset: val_cal for QUAD (cicids, ciciot QUAD), margin_val_cal for unswt/unswr BINARY,
  argmax for ciciot BINARY. Paired by seed (on − off). For each dataset, dedupe is NON-INFERIOR iff the paired one-sided 95% bound
  of ΔF1 ≥ −0.25 pp AND the paired one-sided 95% bound of ΔFPR ≤ max(0.05 pp, 25% of the control's mean FPR).
  ADOPT on all four datasets iff no dataset is clearly worse (both bounds violated) AND ≥ 2 of the 3 stage-1 datasets
  (unswt, unswr, cicids) are non-inferior. Secondary (reported, never decisive): each arm's own best genome × mode, the low-FPR zone.
- **IDS-53 leak size:** measured with RF/XGB (legacy vs train_only lists), no WNN legacy cell (Luiz).
- **unswt cap:** max250 per cluster (= IDS-46/47; equivalent to tot500 in practice: 490 vs 444 neurons, F1 90.82 vs 90.78).
- **IDS-49:** the 12 legacy flows (7368-7379) CANCELLED; IDS-49b (§2) runs after the dedupe decision on the adopted encoding.
- **IDS-48:** moved to cicids, flows 7380-7389 (queued 08/11).
