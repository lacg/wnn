# CTRL-48 block 9 — pre-registration: dc_ms01 vs both, head-to-head on new seeds (Luiz 13/11/2026)

Written 13/11/2026 BEFORE any block-9 run exists. Driver logs/controller/ctrl48/ab_block9.sh (starts after 'S5P DONE').

## Why
On s1-s3 (docs/ctrl48_block8_result.md + the all-blocks survivor table), dc_ms01 = 0/10500 fails, steady 0.94±0.09°,
err 1.42±0.11°; both (= dc_ms01 + biasin + yaw-centred ladder + yaw leak 0.9995) = 2/10500, 0.77±0.08°, 1.37±0.10°.
0 vs 2 fails is NOT a measured difference (exact test p≈0.25; 95% upper bounds ~3 and ~7 per 10500), and the two were
selected under different rules (dc_ms01 = B, ABI 49; both = P, ABI 50). Luiz leans dc_ms01 (fails) vs both (steady).

## Design
- 2 arms x NEW seeds s4, s5, s6 (base seeds 31337004-06), SEED-MAJOR, controller ABI 50, BOTH with rule P.
- dcms01 = block-6 dc_ms01 s1 argv + P; both = block-8 both s1 argv (carries P). Seed = --base-seed only (verified the only
  seed-dependent token). Test = 35 CF2.1 units x 100 episodes (99990101-35), fixed scorer.

## Decision rule (on s4-s6 ONLY — same ABI, same selection rule, same seeds), applied in order
1. Safety gate per arm: pooled fails over s4-s6 <= 5/10500 (0.05%) AND not significantly worse than the other arm
   (exact one-sided conditional binomial test on the two fail counts, p < 0.05).
2. Both pass: choose **both** only if its mean steady is better than dc_ms01's by MORE than 0.03° (the tie band) AND its mean
   err is not worse by more than 0.03°; otherwise choose **dc_ms01** (fewer ingredients, ties go to the simpler recipe).
3. Exactly one passes: that one. Neither passes: report to Luiz, no choice made.
The chosen arm becomes the base for the next block.

## Reported, descriptive only
Pooled s1-s6 per arm (fails/21000, steady, err, pos) — mixes ABI 49/50 and rules B/P for dc_ms01 s1-s3, disclosed, never decides.
Nothing above may change after the first block-9 END line.

## Amendment 1 (13/11/2026, Luiz; written BEFORE any block-9 run — the driver was still waiting for 'S5P DONE')
- **Seeds s4-s10** (base seeds 31337004-31337010), SEED-MAJOR = **14 runs**; n=10 per arm counting the earlier s1-s3.
- The decision rule is unchanged in form and is evaluated on **s4-s10** (same ABI, same rule P, same seeds for both arms):
  gate = pooled fails over s4-s10 <= 0.05% (<= 12/24500) AND not significantly worse than the other arm (exact one-sided
  conditional binomial, p < 0.05); then both only if mean steady better by > 0.03° AND mean err not worse by > 0.03°, else dc_ms01.
- Added, descriptive only: per arm, the number of seeds whose controller had 0 fails, out of 7 (s4-s10) and out of 10 (s1-s10),
  with the 95% lower bound on the fail-free-seed fraction (exact binomial; 10/10 -> ~74%).
