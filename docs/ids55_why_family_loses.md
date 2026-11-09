# IDS-55: why `family` (cross 0.2) loses F1 on UNSW-temporal BINARY (08/11/2026, investigation only)

Flows 7451-7460 (families_v1, dedupe off, legacy top-20, 288 encoded bits). Production numbers = validation_summaries `best_f1` final, `margin_val_cal`. Diagnosis = the 20 matched best_f1 genomes (grid + GA, located in checkpoints/ by genome_hash) re-scored in an offline numpy BINARY-lattice harness on the full temporal_3way TRAIN (175,341) / VAL (41,166) from the probe cache (VAL split in halves: F1-optimal tau on half A, report on half B; TEST never touched). The harness is diagnostic only, not a reportable number: it reproduces the arm gap on the arm means and on 4/5 seed pairs (off GA 91.12 vs family 90.69 cal-F1); harness address counts are ~1.6x DB logical cells, but the off/family ratio holds (2.05x vs 1.91x). Scripts: /tmp/ids55_why/{harness,match,run1-4}.py.

## Ranked causes

**1. Address collapse makes the BENIGN discriminator saturate (main mechanism).** A family-confined 34-bit neuron reads 28+ redundant thermometer bits of 2-5 correlated features, so it sees very few distinct addresses. Under the lattice TRUE-wins write, a benign-disc cell fires if ANY benign train row hit it, so low-diversity neurons fire on attacks too.

| benign-disc neuron (random, 34b) | distinct addrs | fires on benign | fires on attack |
|---|---|---|---|
| spread | 701 | 0.975 | 0.582 |
| home ttl_state / handshake | 84 / 86 | 0.993 / 0.997 | 0.932 / 0.965 |
| home timing / context / volume | 445 / 387 / 449 | 0.990 / 0.989 / 0.977 | 0.855 / 0.623 / 0.480 |

In the PRODUCTION GA genomes, per-neuron Youden (P(fire|own) - P(fire|other)) is off 0.460 vs family 0.306. On the benign discriminator it is off 0.426 vs family 0.254 (homes: ttl_state 0.070, handshake 0.039, timing 0.120). Ranking quality drops: harness AUC 0.970 (off) vs 0.965 (family). The ~48% cell saving is this same collapse seen from the other side.

**2. Protocol starvation (a design consequence, not a bug).** proto+service = 12 bits, so it can be HOME to no neuron and is reached only through the 6-bit cross budget. It gets 0.8-1.0% of family connections vs 4.2-4.6% in off (its fair share is 4.2%). In off genomes, Youden rises with protocol bits (0 bits 0.442, 4+ bits 0.478). Per-bit OLS has protocol highest (0.0215 vs volume 0.0145). Alone protocol is weak (12-bit neuron Youden 0.18), so its value comes as a modifier on other features. Counterfactuals with random 500n x 34b genomes, 3 seeds:

| scheme | cal F1 | AUC | addrs |
|---|---|---|---|
| A spread | 91.15+-0.10 | 0.9714 | 373K |
| B family 0.2 v1 (Rust projection) | 89.78+-0.23 | 0.9626 | 165K |
| C spread WITHOUT protocol bits | 90.44+-0.07 | 0.9685 | 361K |
| D family 0.2, protocol->context (~v2) | 90.70+-0.20 | 0.9661 | 191K |
| E family 0.4 | 91.06+-0.35 | 0.9701 | 265K |
| F spread 262n (equal memory) | 91.03+-0.01 | 0.9705 | 195K |
| G benign spread / attack family | 90.78+-0.02 | 0.9695 | 269K |
| H benign family / attack spread | 89.60+-0.49 | 0.9581 | 269K |

Removing protocol costs spread 0.71 pp. Merging protocol into context recovers 0.92 of family's 1.37 pp gap, so protocol is about 2/3 of the harness gap and collapse (#1) is the remaining ~0.45 pp. The damage is on the benign side (H << G).

**3. H2, allocation: partly supported, but it acts through #1.** Homes are proportional to family size, as designed: ttl_state 17.6%, handshake 8.8%, volume 30.4%, timing 23.8%, context 19.4%, protocol 0. About 50% of neurons sit in ttl_state/handshake/timing, the homes that saturate on the benign side. Volume homes are the best (Youden 0.400) but still below the off average.

## Refuted
- **H1 as "specific cross-family pairs"**: within off genomes Youden barely depends on mixing (corr with majority-family fraction -0.04; most vs least mixed quartile 0.461 vs 0.448). What mixing buys is address RESOLUTION from independent features, not particular pairs. "6 cross bits too few" holds: cross 0.4 is within noise of spread (-0.09 pp).
- **H3 capacity**: at equal memory, spread 262n loses only 0.12 pp vs 500n and stays 1.25 pp above family (harness).
- **H4 GA**: the gap is already there at grid (90.65 vs 89.85). Grid->GA gain is +0.11 (off) vs +0.07 (family). GA leaves composition unchanged (majority-family fraction 0.31 / 0.82 in both phases).
- **H5 bug**: encoder ranges tile 288 bits. Family sizes match the worker `[RF]` line exactly (52/32/80/64/48/12). 0 of 2,457 family GA neurons break `in-family >= w - floor(0.2w)`. bits_per_neuron are identical across arms (34/32). Protocol 0 homes is by design.

## Why the probe missed it
1. Its BINARY metric was **argmax (r1 > r0), uncalibrated**. Spread is biased there (FPR ~20%), and family's "gain" was bias correction (FPR 20 -> 13%). Production calibrates the margin on val, which gives spread that correction for free. Harness: argmax F1 89.91 vs 89.78 (tie), calibrated 91.15 vs 89.78. The probe never computed BINARY AUC or a calibrated F1.
2. Its family draw differs from production on protocol. Protocol was drawn as a home (4%) and padded with 22 random out-of-family bits, so it behaved like spread. Cross was round(0.2x34)=7, not floor=6. Protocol therefore kept its share.
3. Its own warning was there: pure family dropped per-neuron balanced accuracy by 2-11 pp.

## Round-2 prediction (7461+ rf-off, max_neurons_total 262, ~112K cells)
off-262 should land within ~0.1-0.2 pp of off-500 (around 90.6 F1) and beat family-500 v1 (89.92) at equal memory. So the memory saving is not specific to families: halving the neuron count gets it at near-zero F1 cost.

## Cheap next experiments (n=5 paired r20419-23, ~0.43 h/flow)
1. unswt `family` on **families_v2** (protocol->context), cross 0.2: isolates cause #2. Harness predicts the gap shrinks from -1.37 to about -0.45 pp; expect production around -0.3.
2. unswt `family` v1 with **ids_rf_cross=0.4**: tests the resolution account (#1). Harness predicts parity (-0.09) at ~30% fewer cells.
3. (Needs code, design talk first.) Family on the ATTACK discriminator only: harness G is -0.37 vs spread, H is -1.55. Before any of this, read the round-2 off-262 result; if it matches the prediction, families have no memory argument on unswt.
