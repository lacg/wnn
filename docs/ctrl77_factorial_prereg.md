# CTRL-77 — L1 × L2 × L3 factorial on the G+T base (CTRL-74/75/76) — pre-registration

Written 25/11/2026 before any flight of any arm below. Luiz approved (25/11) the 11 non-empty combinations of
L1 {off, on} × L2 {off, a, b} × L3 {off, on}, flown at n = 3 first and winnowed in stages. **L1 is flown
gain-matched only.** The levers come from `docs/ctrl73_gap_analysis_gt.md` §6. Plane stories:
- CTRL-77 (this block);
- CTRL-74 (L1);
- CTRL-75 (L2);
- CTRL-76 (L3).

## Control: G+T (the model-free base), already flown — never re-flown

- **Tags.** `C-S2-d_wposx2_a62_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0g3gz_full_sN`, s1–s10, in
  `logs/controller/ctrl48`. ABI 62, rate-loop flags off.
- **The ABI-62 controls are valid on the ABI-63 wheel.**
  - `tests/controller_ctrl74_delta_leak_yaw.py` (merge e53a65ac) pins that, with `--delta-leak-yaw` unset, ABI 63
    is EQUAL to ABI 62 on the flown G+T recipe. The pinned items: fitted thresholds, calibration token, resolved
    ctbr dict, every parsed argument, provenance, 40 steps of features, the Lamarckian write-back digest after a
    short DAgger, and the scorer rows.
  - `tests/controller_ctrl75_76_ladders.py` (merge c454f70d) pins `--gyro-xy-ladder` and `--yaw-err-outer-cut`
    unset == today's G+T (thresholds, token, ctbr dict, every parsed argument).
- **Pairing.** Arm sN vs G+T sN, same seed (base seed 31337000+N), same 35 TEST units 99990101..99990135 ×
  100 episodes (the HEADLINE block).
- **No recipe repeats on a seed.** Every arm is a new recipe, and the G+T recipe is not re-flown.

## Arms

**Argv.** Each arm sN argv = the **flown G+T sN `[argv]` line** with only three changes:
- the arm's flags appended;
- for L1, the in-place swap `--delta-max-yaw 0.5` → `--delta-max-yaw 0.08333`;
- the two save paths (the tag).

**Tags.** `C-S2-d_wposx2_a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_<arm>_full_sN`.

```
#   arm       L1  L2   L3   flags added to the G+T sN argv (+ L1's --delta-max-yaw 0.5 -> 0.08333 swap)
1   l1        on  -    -    --delta-leak-yaw 0.95
2   l2a       -   a    -    --gyro-xy-ladder dense
3   l2b       -   b    -    --gyro-xy-ladder dense_hold
4   l3        -   -    on   --yaw-err-outer-cut 4.5
5   l1l2a     on  a    -    --delta-leak-yaw 0.95 --gyro-xy-ladder dense
6   l1l2b     on  b    -    --delta-leak-yaw 0.95 --gyro-xy-ladder dense_hold
7   l1l3      on  -    on   --delta-leak-yaw 0.95 --yaw-err-outer-cut 4.5
8   l2al3     -   a    on   --gyro-xy-ladder dense --yaw-err-outer-cut 4.5
9   l2bl3     -   b    on   --gyro-xy-ladder dense_hold --yaw-err-outer-cut 4.5
10  l1l2al3   on  a    on   --delta-leak-yaw 0.95 --gyro-xy-ladder dense --yaw-err-outer-cut 4.5
11  l1l2bl3   on  b    on   --delta-leak-yaw 0.95 --gyro-xy-ladder dense_hold --yaw-err-outer-cut 4.5
```

**What each flag moves (checked by the driver through phased_ga's own resolver).**
- **L1** moves only the yaw bank's delta law: `delta_leak_yaw` None → 0.95 and `delta_max_yaw` 0.5 → 0.08333.
  Thresholds and the calibration token are unchanged.
- **L2-a / L2-b** move only the calibration token (`+,gyroxy:dense` / `+,gyroxy:dense_hold`).
- **L3** moves only the calibration token (`+,yawout:4.5`).
- **Everything else is equal to the control:** the yaw gamma (1, 2, 2, 3), the gyro_z ladder, every other parsed
  argument, the resolved ctbr dict, folds 5 and the 35 units (digest).
- L2 and L3 are new threshold lineages (thresholds are the address function). L1 is a new output lineage (the
  decode and the DAgger label baseline). None is pooled with another lineage.

### L1 gain match (the exact formula, from the code)

The CTBR delta law per yaw action (`controller.rs` `decode_ctbr` / `leaked_baseline` / `delta_leak_bank`, ABI 63)
is

    s_k = clamp( λ_yaw · s_{k−1} + d_k , ±range_r ),   d_k = shape_γ(t_k) · delta_max_yaw · range_r

with the yaw bank anchor 0 under ctbr and range_r = 2.0 rad/s. A held code t gives the steady setpoint
s* = shape_γ(t) · range_r · **delta_max_yaw / (1 − λ_yaw)**. So the code-to-held-setpoint map (the yaw bank's
steady-state gain) is identical when delta_max_yaw / (1 − λ_yaw) is equal:

```
today (G+T)   0.5     / (1 − 0.70) = 1.666667
L1 0.08333    0.08333 / (1 − 0.95) = 1.666600   -> 1.667 to 3 decimals   (USED)
L1 0.083      0.083   / (1 − 0.95) = 1.660000   -> 1.660, does NOT match to 3 decimals (rejected)
```
L1 therefore uses **`--delta-leak-yaw 0.95 --delta-max-yaw 0.08333`**. The per-action yaw step is 6× smaller
(0.167 vs 1.0 rad/s at full code), and the leak no longer bleeds the held trim. Sustaining a trim s now needs
0.05·s per action instead of 0.3·s. The ±range clamp is unchanged.

## Hypotheses (from CTRL-73, TEST + VAL diagnosis)

- **H-L1 (hold the yaw trim).** In the hold the WNN keeps a unit-specific yaw-rate trim (firmware yaw-rate
  Ti ≈ 7 s vs a 2 s episode). Under the 0.7 leak it is rebuilt every action (mean delta = 0.3 × trim; per-action
  SD ≈ the trim), which gives a ~2.8 Hz hold-yaw chatter plus a wear-proportional offset (0.32 °/c1 on G+T).
  Gain-matched, the yaw loop's steady gain is unchanged, but the trim persists and the per-action noise is
  1/6.
  - **Expected vs G+T: steady lower, err lower** (hold yaw 0.219° vs MPCOF 0.078°; err slope on |c1| 0.33 vs
    0.016).
  - stable / fails / pos: no change expected.
  - Risk: a smaller per-action yaw step could slow the yaw transient (the Y3 lesson). That would show as whole
    yaw up.
- **H-L2 (roll/pitch rate ladder).** gyro_x / gyro_y are beyond their outer cut on 91 / 75 / 49 % of the
  0–0.6 s ticks (VAL). The roll/pitch pre-hold excess is +0.097 / +0.093° vs MPCOF, and every fail is a
  mid-window roll/pitch excursion. T fixed exactly this on yaw.
  - **Expected vs G+T: err lower; steady not worse.** Fails possibly fewer (not registered).
  - **L2-a (`dense`):** ±0.004 / ±0.013 / ±0.05 / ±0.80 rad/s.
  - **L2-b (`dense_hold`):** ±0.004 / ±0.013 / ±0.026 / ±0.80. It keeps today's 0.026 hold cut and guards
    against the hold-roll cost T showed.
- **H-L3 (yaw-error outer cut).** The yaw_err code saturates at ±2.62°, and the command is flat above it
  (0.56 → 0.57 rad/s). The gap still grows with |yaw0| (0.20 → 0.44°). L3 moves the outermost pair to ±4.5°.
  - **Expected vs G+T: err lower; steady not worse.** Expected to be the smallest of the three.
- **Combinations.** The levers act on different parts: L1 on the yaw output accumulator, L2 on the roll/pitch
  rate inputs, L3 on the yaw-error input. Registered direction for every combination:
  - every arm (Luiz 25/11, his usual rule for all arms): **err lower and steady lower** (the levers target
    the err/steady gap to MPCOF; consistently better in that direction, not worse elsewhere).
  - Combination vs its single-lever parts (and vs the other combinations) is **descriptive only**: does it add
    up, or do the parts interfere?

## Readout numbers

- **Source.** For every run (arm and control), the HEADLINE TEST block of its `.out`.
  - err and steady are the mean over the 35 per-unit RESULT lines (4 dp; the `ctrl70/cmp_logged.py` parse).
  - fails = the run's `stable_fail` count over 3500 episodes.
  - stable = 100·(1 − fails/3500), the same quantity as fails.
  - pos is the 3-D position error (includes altitude).
- **Same scorer.** Arm and control were both scored by the in-flight GPU scorer, so the known +0.002–0.004°
  GPU offset (CTRL-70/73) cancels in the pair.
- **Ties.** A tie at 4 dp counts as NOT lower.
- **No other numbers.** Gen-line and VAL numbers are never read.

## Staged winnowing rule (n = 3 → 5 → 7 → 10)

**Stages.** n = 3 = s1–s3, all 11 arms, seed-major (all 11 on s1, then s2, then s3; 33 runs). Survivors fly
s4–s5 (n = 5), then s6–s7 (n = 7), then s8–s10 (n = 10). G+T controls exist on s1–s10.

**Drop rule, applied to each arm at n = 3, 5 and 7 (k = the number of pairs read).** An arm is **DROPPED** if
ANY of (A), (B) or (C) holds:
- **(A) Not majority-better on either target.** It is lower than G+T on err in fewer than a majority of the k
  pairs AND lower than G+T on steady in fewer than a majority of the k pairs. Majority = 2 of 3, 3 of 5, 4 of 7.
  To survive, it must be majority-better on at least one of err or steady.
- **(B) More fails in total.** Its fails summed over the k seeds exceed G+T's summed fails on the same seeds by
  more than **2** (k = 3), **3** (k = 5) or **4** (k = 7).
- **(C) Worse on stable on a majority.** It has a lower stable % than G+T on a majority of the k pairs (2/3,
  3/5, 4/7). Because stable = 1 − fails/3500, this is "more failed episodes than G+T on a majority of seeds".

**Everything else survives.**
- pos is **not** a winnowing criterion (CTRL-73: G+T itself moved pos by +1.4 mm). It enters at the n = 10
  adoption read.
- A drop is final for the block: a dropped arm is never re-flown on later seeds.
- The rule is mechanical, and Luiz may overrule any drop or survival at a stage (his standing practice).

**n = 10: adoption (Luiz's rule), per surviving arm vs G+T.**
- **ADOPT if:**
  - it is consistently better in the registered direction (majority of the 10 pairs AND mean delta in that
    direction) on err and steady — the SAME rule for every arm (Luiz 25/11: "use my usual rule for all arms");
  - AND it is not worse on stable, fails and pos (3-D) over the whole vector.
- No magnitude bar. Cells (TRUE keys) and wall minutes are tie-breakers only.
- If more than one arm passes, Luiz chooses on the whole vector, with the combination-vs-parts tables as
  descriptive input.
- Flat or worse → not adopted; G+T stays the base.

**Budget.** n = 3 is 33 flights, about 30–36 h at ~55–65 min each. In the worst case (no drops) the stages add
22, 22 and 33 flights. No cap is registered here. A cap at a stage, if wanted, is Luiz's decision before that
stage flies.

## Execution

- **Driver.** `logs/controller/ctrl63/ab_fact77.sh`, one run at a time; it waits for any other controller run.
  - Seeds and arms come from `F77_SEEDS` / `F77_ARMS`, so later stages fly only the survivors.
  - Every argv is rebuilt from the flown G+T sN `[argv]` line and checked by fixed text: arm minus its flags
    (and the L1 swap) and the tag == the control argv.
  - Then through phased_ga's resolver and `resolve_block10_args`: control → (0.5, leak unset, gamma
    (1, 2, 2, 3), the G+T token); arm → only its fields move; the digest of everything else is equal; the L1
    gain is 1.667.
- **Safety checks.** Before every flight:
  - wheel ram_controller + facade ABI 63;
  - `src/wnn/control` clean;
  - this prereg committed.
- **Logging.** It never reuses an `.out` tag, takes the lock `/tmp/ab_fact77.pid`, and writes `F77` and
  START/END lines in `logs/controller/ctrl48/runs.log`. It ends with `F77 DONE` or `F77 ABORT: why`.
- **Hard rule.** The controller yields to the IDS worker and is never in the way of it.
- **Settings.** `--num-eval-folds 5`, rule P (`--val-units 20 --select-tail p99`), BINARY, everything else as
  G+T.
