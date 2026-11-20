# IDS70: supervised thermometer cut points (ids_cut_method E and B) on the unswr fixed-80b recipe

Written 20/11/2026 (EST) by ids-security, on Luiz's approval of 20/11/2026 (Plane IDS-70). No IDS70 flow
exists yet and no IDS70 number has been computed. Pre-check: Plane IDS-63, `logs/ids/encoding_B/`
(PREREG.txt, choice.json, summary.txt). Queue: `logs/ids/queue_ids70.sh` →
`scripts/queue_ids70_cut_method.py`; the exact POST bodies are in `docs/ids70_flows.json`.

## 1. Hypothesis

The thermometer's quantile cuts cost the classical model part of its accuracy. In the offline pre-check,
RF on raw top-20 scored 95.81 TEST F1 and RF on thermo64 scored 95.33. Placing the cuts where the classes
separate recovers part of that gap: method E got 95.59 TEST F1 / 0.329 FPR (it recovered 78% of the gap on
VAL and ~54% on TEST), and method B got 95.66 / 0.359 with only 457 bits. XGB hardly moved (about ±0.1 pp),
so the loss is specific to RF. **A WNN gain is a hypothesis, not an implication.**

H: on the unswr fixed-80b recipe, supervised cuts raise the WNN's TEST F1 without making FPR or accuracy worse.

## 2. Arms (only `ids_cut_method` changes)

| arm | ids_cut_method | what the encoder does (fitted on TRAIN only) | bits |
|-----|----------------|----------------------------------------------|------|
| E | `info_gain_mc` | Best-first information-gain cuts against the 10 attack classes, under one global budget equal to today's numeric + categorical bits. proto/service/state become a thermometer over their TRAIN attack-rate rank. | 1104 (same as today) |
| B | `xgb_splits` | Every split threshold of an XGB fitted on TRAIN (100 trees, depth 6, lr 0.1, binary label) over the 17 numeric features. Categoricals keep their binary code. | 457 |

- **Seeds.** r20421, r20422, r20423, paired with the controls.
- **Params.** Each flow is the control flow's `config_json` copied verbatim, plus `ids_cut_method`.
  - Template: `ids-binary-2-phase`. Experiments: Grid Search (neurons x bits) and GA Neurons.
  - Recipe: unswr `random_3way`, top20, `ids_n_bits` 64, BINARY, vote θ 0.02, `ids_encoding_dedupe` off,
    `ids_feature_ranking` legacy, min_bits = max_bits = 80, bits_grid [80], max_neurons 250, 5-fold CV (5/gen),
    desirability fitness (0.1/0.35/0.35/0.2), and so on.
  - The queue script asserts that the diff against the control is exactly `{ids_cut_method}`.
- **Names.** `IDS70-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-fixb80-encoff-cut{E|B}-r<seed>`.
- **Order: arm-major.** E r20421, r20422, r20423, then B r20421, r20422, r20423 (the worker runs FIFO, lowest id
  first). E is the pre-check's locked winner and the primary hypothesis. Its n=3 verdict lands after about 3 flows
  (~35 h). Seed-major order would delay it to about 5 flows (~58 h).
- **Leak guard.** The cuts, the categorical ranks, and the XGB whose splits are used are all fitted on the TRAIN
  partition and TRAIN labels only. They are fitted inside `ThermometerEncoder.fit`, which never receives val or
  test rows (`tests/test_ids_cut_method.py::test_fit_never_touches_val_or_test`). The fit is deterministic:
  information gain has no RNG, and XGB has no subsampling (its splits are seed-invariant, checked for 20419 and
  20421) and runs on 6 pinned threads.
- **Parity (verified 20/11).** On UNSW `random_3way` through the worker path (flow 7585 params):
  - `quantile` matches the main-line 7fd1a425 encoder by sha256 on train, val and test.
  - `info_gain_mc` and `xgb_splits` reproduce `cuts_E.json` / `cuts_B.json` exactly, and the pre-check's encoded
    matrices byte-for-byte.

## 3. Controls (IDS62 fixed-80b, flows 7585-7587; not re-flown)

Values are from `test_summaries`: GA `ga_neurons`, `best_f1`, `test_point='final'`, `margin_val_cal`, TEST
(worker ABI 29).

```
seed     flow   F1 (%)    FPR (%)   acc (%)   cells       hours
r20421   7585   95.154    0.296     99.301    2,891,690   12.6
r20422   7586   95.111    0.244     99.304    2,741,219   12.6
r20423   7587   95.173    0.255     99.310    2,948,267    9.5
```

The controls' cache keys have no `cut=` part. IDS70's keys carry `cut=info_gain_mc:v1` or `cut=xgb_splits:v1`,
so the cross-flow test cache can never serve an IDS70 genome a control's row, or the reverse.

## 4. Validity check (HARD; applied before any F1 is read)

1. **Encoder line, per flow, in the worker log.**
   - The flow's `Loading unsw-nb15 dataset (...)` line shows `cut_method=info_gain_mc` (E) or
     `cut_method=xgb_splits` (B).
   - Its `[CUTS] cut_method=<m> seed=<flow seed> fitted on TRAIN (1,267,407 rows): <N> total bits` line shows
     N = 1104 for E and N = 457 for B.
   - Its `Encoder:` line ends `, cut_method=<m>)`.
   - There is no `UNKNOWN PARAM` warning for `ids_cut_method`.
   - If any of these fails, that flow is VOID: it ran the old encoder, which makes it a control re-fly.
2. **Cache key.** Every IDS70 `test_summaries` row's `cache_key` contains `|cut=<m>:v1|`.
3. **Width.** Every neuron of the arm's GA `best_f1` final genome is exactly 80 bits, as in IDS62 §4. Any seed that
   fails is void.
4. **Count.** With fewer than 3 valid seeds in an arm, that arm has no verdict: report and stop.

## 5. Metric and decision rule (Luiz's rule, IDS-70)

- **Cell.** GA `ga_neurons` · `best_f1` · `test_point='final'` · `margin_val_cal` · TEST, from
  `test_summaries.threshold_metadata` (`macro_f1`, `fpr`, `acc`), paired by seed with the control.
- **Per arm, separately against the control:**
  - **ADOPT** if dF1 > 0 on 3/3 seeds, AND mean dFPR ≤ 0, AND mean dAcc ≥ 0. This is the reading that adopted
    IDS62: F1 up 3/3, mean FPR lower, acc higher.
  - If F1 is up on 3/3 seeds but mean FPR or mean acc is worse → **Luiz decides**. He reads the whole vector.
  - Otherwise → **NOT ADOPTED**. The fixed-80b quantile control stays.
- **If both arms adopt.** Both rows are reported side by side and Luiz picks. B spends 41% of E's input bits, and an
  80-bit neuron then samples 17.5% of the input instead of 7.2%. That sampling difference is part of B's treatment,
  not a confound to remove.
- **Reported beside the rule** (all from the DB, never estimated):
  - attack-class F1 and TPR, cells (`logical_cells`), GA generations, hours per flow;
  - the grid `best_f1` neuron count;
  - the 7-mode Rule-7 tables, for context only.
- **Caveat** (for `experiment-design` before anything enters the paper): n=3, and two arms are tested against one
  control, so no multiplicity correction is applied. This is an adopt/not-adopt engineering decision under Luiz's
  rule, not a significance claim.

**Expectation (not a result).** Somewhere between 0 and +0.2 pp TEST F1 for E. RF gained +0.26 on TEST, but XGB, the
stronger tree learner, gained nothing. B is the more uncertain arm: it uses fewer bits and the address density
changes.

## 6. Cost and schedule

- **Per flow.** About 11.6 h, the mean of the controls (12.6 / 12.6 / 9.5 h). B's 457-bit input could change the
  GA's generation count.
- **Total.** Six flows take about 70 h.
- **Schedule.** Queued after flow 7590 (IDS63, ~23:30 EST 20/11) and the worker restart. E's verdict should arrive
  around 22/11 midday EST, and B's around 23/11 night EST.
- **Deploy.** Python only: no Rust, no wheel, worker ABI stays 29. The worker picks up the new code when it is
  restarted while idle. `logs/ids/queue_ids70.sh` refuses to queue unless both hold:
  - the worker checkout's `KNOWN_PARAMS` contains `ids_cut_method`;
  - the running worker started after that code landed.
