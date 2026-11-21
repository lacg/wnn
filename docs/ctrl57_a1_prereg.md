# CTRL-57 A1: looser stage stopper on Y+F+L — prereg (Luiz 20/11/2026 ~20:00 EST)

## Question
Does the search keep improving steady (and err) when the stage stopper lets it run longer?

Source: the fitness audit of 18/11 (`logs/controller/ctrl48/fitness_audit/`). Magnitude patience watches only err and
stable, and needs a 0.05° err gain per 2 generations. NEURONS stopped at gen 8/15 in 15/15 runs, and MEMORY at gen
6/120 in 14/15. MEMORY moved the best steady by only −0.008°.

## Arm
- **A1** = base Y+F+L with `--neurons-patience 3 → 6` (every occurrence in the base argv) and
  `--memory-patience 2 → 10`. Nothing else changes, and there is no code change. Refine flags are NOT included:
  A1 is tested alone.
- **Base / B rows** = Y+F+L `..._a59_..._ffF_yld_full_s1..s5`, not re-flown. They are valid via the flag-off
  determinism gate ABI 60 vs 59 (CTRL-58 chain).
- **Seeds** s1–s5 seed-major on ABI 60. Tags `C-S2-d_wposx2_a60_rf_c200_dc_biasin_ms01_P_both_L1_ffF_yld_a1_full_sN`.
- **Driver** `logs/controller/ctrl48/ab_a1.sh`. It starts after the A2 block's `A2 DONE`, whatever the A2 verdict
  (Luiz 20/11), via `logs/controller/ctrl48/a1_after_a2.sh`. It ends with `A1 DONE`.

## Rule (Luiz's adoption reading, 20/11)
A1 − B, paired by seed, TEST over 35 CF2.1 units (stable / err / steady / pos / fails). Adopt if A1 is consistently
better in the expected direction (steady, and/or err, down) and not worse on the rest of the vector. There is no
magnitude bar. Flat or worse means not adopted.

## Expectation
Weak. In the audit, MEMORY ranks anti-predicted VAL steady. A null result would mean MEMORY needs more evaluation
episodes, not more generations. Runs will be longer, roughly +0.5 to +1.5 h each.

## After
If A2 and A1 both have merit, discuss A1+A2 with Luiz (CTRL-58 prereg: "A1 and A1+A2 follow as their own block").

## Addendum 21/11/2026 ~07:40 EST — A1+A2 queued after A1 (Luiz)
Luiz agreed A2 is not adopted on its own at n=9: err −0.009° and steady −0.002° moved in the right direction but not
consistently, while pos rose +0.003 m (worse on 6/9 seeds) and fails were 6 vs 1. He asked for **A1+A2** to test
whether each lever covers the other's weakness. A2's refine ranking may need a longer search to turn into gains,
which A1's looser stopper provides; A1 alone was expected to be weak.
- **Arm A1+A2** = Y+F+L + `--neurons-patience 6 --memory-patience 10` + `--refine-fitness --refine-w-steady 0.40
  --refine-w-mid 0.25 --refine-w-reward 0.10`, s1–s5, ABI 60. Tags `..._ffF_yld_a1a2_full_sN`; driver
  `logs/controller/ctrl48/ab_a1a2.sh`; waiter `a1a2_after_a1.sh` (fires on `A1 DONE`). Ends `A1A2 DONE`.
- **Read** paired vs Y+F+L (primary), and vs A1 and A2 on the same seed: does the combination keep A2's err/steady
  direction while closing its pos and fails gap?
- **Rule:** the same adoption reading as above.
