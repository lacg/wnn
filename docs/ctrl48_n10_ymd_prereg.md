# CTRL-48 n=10 extension, M / Y / D: prereg (Luiz 18/11/2026 ~06:40 EST)

## Question
Is the base Y (`--delta-max-yaw 0.5`, yaw bank only) or D (`--delta-max 0.5`, all banks)? Both are compared against
M (the adopted base).

## Evidence so far (s1-s5, Plane CTRL-55)
**Headlines as flown under rule P** (TEST means):

| arm | err | steady | pos | fails |
|---|---|---|---|---|
| M | 1.070 | 0.426 | 0.340 | 2 |
| Y | 1.026 | 0.428 | 0.344 | 1 |
| D | 0.994 | 0.352 | 0.352 | 0 |

**Every stage-select candidate scored offline** (commit 84439b4c, logs/controller/ctrl48/ymd_stage/):
- Rule P costs Y up to 0.053° of steady (it lands Y on GRID picks) and costs D nothing.
- TEST D−Y steady by headline rule: −0.081 (rule P), −0.055 (fitness-only), −0.024 (best-of-MEMORY).
- err D−Y is a tie; pos D−Y is +0.004 to +0.008 m.
- Both Y and D beat M on err (−0.04 to −0.07) under every rule.

**Noise:** the paired D−Y SD is about 0.07° per seed, so n=10 resolves gaps of roughly 0.06° or more only.

## Arms
All arms are flown on ABI 58 with `--yaw-trim-ff` off (bit-identical to ABI 57 by the determinism gate). Each is
rebuilt from the seed's arm-1 both_L1 `.out`.
- **M:** both_L1 + `--zrank-pos-floor auto` + `--obs-yaw-from-att-est` (`--delta-max 0.6`).
- **Y:** M + `--delta-max-yaw 0.5`.
- **D:** M with `--delta-max 0.6 → 0.5`, substituted in place and asserted.

**Seeds:** s6-s10 (base 31337006-010), flown seed by seed in the order M, Y, D. That is 15 runs, about 15-17 h.
Tags: `C-S2-d_wposx2_a58_rf_c200_dc_biasin_ms01_P_both_L1_n10{M,Y,D}_full_sN`.
Driver: `logs/controller/ctrl48/ab_n10_ymd.sh`. The runs still fly with rule P; the decision uses the offline
re-pick below.

## Decision (Luiz 18/11): fitness-only headline + stage table
- **Pooled sample:** n=10, s1-s5 already flown (M a57 / Y a57 / D a56) plus s6-s10 (a58).
- **Headline per run:** the fitness-only pick, i.e. the top VAL fit among the 9 stage-select candidates (no p99 tail),
  scored on TEST.
- **Every candidate scored offline:** after the block, with `dmy_axis.py` on VAL+TEST, the same as the ymd_stage
  analysis. GRID entries without cells go through the CTRL-56 repair; no empty controller is scored.
- **Comparisons:** paired by seed, D−Y, Y−M and D−M on the WHOLE vector: stable, err, steady, pos, fails.
- **Adoption thresholds against M** (Luiz's whole-vector rule):
  - steady better by > 0.03°;
  - err not worse by > 0.03°;
  - pos not worse by > 0.010 m;
  - fails within the gate and not significantly worse.
- **Choosing between Y and D:** the arm better on steady by > 0.03 with pos within 0.010 of the other. If neither
  clears that, it is Luiz's whole-vector call. The trade on record: D gives more steady at a small pos/alt cost; Y is
  pos-neutral.
- **Pre-registered companion:** the stage-for-stage NEURONS/MEMORY table (fit-best and top-3 mean, per seed and
  pooled), reported next to the fitness-only verdict.
- **Expectation** (from s1-s5): D − Y steady about −0.03 to −0.06; err tie; D pos +0.005 m.
- **Stated limit:** n=10 detects only gaps of about 0.06° or more at 80% power, so a smaller gap may stay unresolved.

## Reported
- per-seed compact pair tables, with M's and D's re-flies versus their ABI 56/57 counterparts not applicable
  (these are new seeds);
- the rule-P headlines as flown, for transparency.

Nothing above may change after the first N10 END line.
