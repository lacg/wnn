# CTRL-58 A2: staged refinement fitness — prereg DRAFT (Luiz 19/11/2026 ~09:30 EST)

Status: DRAFT until the base is fixed (D+F vs Y+F, decided after the F-on-Y block). Nothing flies before that.

## Decisions taken (Luiz 19/11)
- **A2 goes before A1** (CTRL-57, patience 6/10).
- **Keep the defaults, refine from the start.** Arm A2 = base + `--refine-fitness` and nothing else:
  - trigger: ≥ 80% of the TRAIN pool at stable 100%, err ≤ 1.5°, steady ≤ 1.0°, pos ≤ 0.40 m for 2 consecutive
    generations, or MEMORY start. On this base the pool already meets those bounds, so the switch fires at
    about NEURONS gen 2 ("refine from the start"); the GRID stage is not a GA and is unaffected;
  - refinement gate: stable ≥ 0.99, err ≤ 1.5°, steady ≤ 1.0°, pos ≤ 0.40 m;
  - weights: steady 0.50, reward 0.25, p99 tail 0.15 (ABI 59 tail scorer row), pos 0.10;
  - patience: same counter and max per stage; at the switch the counter resets and steady is watched too, with a
    0.15° floor (5% of the distance above the floor counts as a gain); the trigger streak carries across stages;
  - `--refine-select` is NOT part of A2 (stage-select stays rule P).
- **Arms:** A2 vs the base (B = the base's existing F-block rows, argv-identical, ABI 59). If A2 is adopted, A1 and
  A1+A2 follow as their own block (A1 alone runs only if Luiz asks).
- **Seeds:** s1-s5 seed-major, ABI 59; extend to n=10 if promising.
- **Rule (a starting point, Luiz):** A2 − B, paired by seed, TEST over 35 CF2.1 units: steady better by > 0.03°,
  err not worse by > 0.03°, pos not worse by > 0.010 m, fails within the gate. Luiz judges the whole vector: a large
  gain on two metrics is not vetoed by a hair-thin miss on the third.
- **Expectation:** steady down (the fitness weights steady 0.50 and patience now sees steady gains), err flat or
  down, pos flat; runs somewhat longer (patience kept alive by steady gains).

## Amendment 19/11 — mid-window err term, Luiz approved (19/11/2026 ~14:15 EST)

Still a DRAFT. This amendment supersedes the "weights" bullet and the "Expectation" bullet above; everything
else stands. Source: the Y+F gap investigation, `logs/controller/ctrl48/yf_gap/`.

### Why
- The gap to MPCOF eq-info on the Y+F base (TEST, n=3 seeds, a direction) is **err +0.147°** and steady +0.050°.
  About 93% of the err gap lies before the hold window (t < 1.6 s).
- The fitness does not weight that window. Its err term is the reward = integrated per-step err²
  (`FitnessCalculatorControllerHarmonic._columns`, "err² → ranked on reward"), and on the Y+F s1 winner (VAL,
  6 units) **at least 90% of Σerr² accrues in the first 0.3 s**, where the 0.2 s motor lag binds every
  controller. steady covers t ≥ 1.6 s. Nothing covers 0.3–1.6 s, where the yaw error creeps (mean |yaw| 0.53° at
  0.3–0.4 s, at its floor only by 1.4 s) and where the seed-variable roll / pitch transient sits.
- A2 as drafted (steady 0.50, reward 0.25) keeps that blind spot: it pushes steady (the 0.05° gap) and barely
  touches err (the 0.15° gap).
- Limit of this lever, measured: on VAL every one of the 27 stage candidates of Y+F s1-s3 sits at err 0.81–0.95°
  against the label law's 0.69°. Selection alone cannot close the err gap; this term makes the search see the
  window, it does not replace the representation levers (`docs/ctrl48_yawladder_prereg.md`,
  `docs/ctrl59_yaw_pterm_spec.md`).

### The term
- **`mid` = mid-window mean attitude error, degrees, lower is better.** Per episode: the mean of the scorer's
  geodesic error (true attitude against the scorer's own G5 tilt reference, the quantity err and steady
  average) over the steps `mid_start ≤ t < tail_start`, with `mid_start = ceil(0.15 · steps)` and
  `tail_start = ceil(0.80 · steps)` (the hold window's existing start). At 2000 steps and dt 1 ms that is
  **0.3 s ≤ t < 1.6 s**. The genome's value is the mean over its evaluation episodes.
- **Diverged episodes:** an episode that ends before `tail_start` contributes its whole-episode mean, the same
  fallback the WNN scorers already use for steady (`axis_split.rs` `HoldPolicy::FallbackWhole`).
- **Pools:** the mean over the CRN pools, as every other scalar column (`evaluator.combine_pool_scores`).
- It enters the refinement fitness as one more ZRank cost column (`FitnessCalculatorControllerRefine._columns`),
  z-scored and clamped like the others, with its own weight `--refine-w-mid` (default 0.0 = column absent =
  the pre-amendment calculator, bit-identical). When the weight is > 0 and a candidate has no finite value the
  calculator REFUSES (as the tail column does); it never drops the weight silently.
- It is NOT added to the gate, to the trigger, or to stage-select (rule P unchanged). Magnitude patience is
  unchanged: it already watches mean err, which moves with `mid`.

### Weights (proposal — needs Luiz's confirmation, because the 19/11 morning ruling was "keep the defaults")
```
column             | A2 as drafted | A2 amended
-------------------+---------------+-----------
steady             | 0.50          | 0.40
mid (0.3–1.6 s)    | —             | 0.25
p99 tail           | 0.15          | 0.15
reward (Σerr²)     | 0.25          | 0.10
pos                | 0.10          | 0.10
```
- The new weight comes mostly out of reward, which is ≥ 90% the plant-bound first 0.3 s. reward stays at 0.10
  as the guard on that early transient.
- steady keeps the largest weight (A2's purpose), mid is second because the err gap is three times the steady gap.
- Arm A2 argv = base + `--refine-fitness --refine-w-steady 0.40 --refine-w-mid 0.25 --refine-w-reward 0.10`
  (tail 0.15 and pos 0.10 are the defaults).

### Plumbing — verified 19/11 (no code was changed)
**Neither scorer returns the mid-window mean today.** What each path returns per genome
(`cpu_score.rs:633-663`, `metal_controller.rs:1466-1500`, parsed at `evaluator.py:2616-2682`): reward, err,
stable, jerk, mono, steady, rise, settle ×2, ITAE, IAE, ISE, effort, pos, alt, dx, dy, pos_hold, dz_hold
(19 values), plus the ABI 59 per-episode error list.

**Rust needed (rides the ABI 60 build of `docs/ctrl59_yaw_pterm_spec.md`):**
1. `controller/cpu_score.rs`, `rollout_one`: next to `tail_sum_err` / `tail_cnt` (lines ~485-510) add a
   mid-window accumulator (`mid_start ≤ _t < tail_start`), fold it per episode with the FallbackWhole rule, and
   append the mean as **row index 19** after `dz_hold` (appending keeps the `PRE_ABI51_ROW_METRICS` pins valid).
2. `controller/shaders/controller_rollout.metal`: the kernel twin — `mid_start` next to `tail_start`
   (line 1415), the accumulator beside the `t >= tail_start` blocks (2380-2395), and one more per-episode output
   slot (widen `out_effort` from 7 to 8 slots per episode, lines 2434-2445, or a new buffer).
3. `controller/metal_controller.rs`: read the new slot and aggregate it per genome with the same rule
   (aggregation block ~1391-1500), append it as row index 19; update the struct-size / slot-count comments.
4. One definition of the window in Rust (a `pub(crate)` constant `MID_START_FRAC = 0.15` beside the 0.80 the
   scorers share), not a second literal in each file.
5. Tests, in `cargo test -p ram_controller --lib --no-default-features`: CPU-vs-kernel parity of row 19 in the
   existing parity sweeps; a unit test that `0.15·first + 0.65·mid + 0.20·steady == err` on a completed
   episode (to f32 tolerance), where `first` is the mean over t < mid_start; the diverged-episode fallback.
6. The classical rival scorer does not need the row (it is not ranked). Optional, not required.

**Python needed (every site; the "--fit-weight-alt was a no-op at 4/5 sites" class of bug):**
- `evaluator.py:2616-2682`: read `row[19]` (None on a shorter row) → `"mean_mid_error_deg"` in the metrics dict.
- `wnn/ram/metrics.py` (`ControllerMetrics`, beside `mean_position_hold_m`): the field, Optional, default None.
- `evaluator.py:3141` and `:3270`: the two `ControllerMetrics(...)` construction sites.
- `phased_ga.py:2279-2304` (`_mean_metric`, the val-unit average used by stage-select) and the stage table /
  END-row printers (`phased_ga.py:1747`, `:1784`), so the column is visible wherever steady is.
- `refine_config.py`: `RefineConfig.w_mid`, `refine_config_from_args`, `_validate_weights`, `refine_token`
  (`mid<w>`), `refine_weights_record`; `phased_ga._add_refine_args`: `--refine-w-mid`.
- `refine_fitness.py`: the column in `_columns` (cost, lower better) with the refuse-if-missing rule, and the
  name string.
- `evaluator.combine_pool_scores` needs no change (it averages every numeric field).
- A Python test that a refine calculator with `w_mid > 0` changes the ranking when only `mean_mid_error_deg`
  differs, and refuses when it is None.

**No-Rust fallback, if ABI 60 slips (Python only, ABI 59):** the pre-hold mean
`pre = (err − h · steady) / (1 − h)`, `h = (steps − tail_start) / steps` (0.2 here). It is an exact identity
for completed episodes and uses two columns both scorers already return and that are already parity-pinned.
It covers 0–1.6 s, so it still includes the first 0.3 s (about 58% of mean err on the s1 winner against 33%
for the mid window): a blunter instrument than `mid`, but far less early-dominated than Σerr². If used, the
prereg must name it `pre`, not `mid`.

**Considered and not chosen: ITAE** (`mean_itae`, Σ t·|err|·dt, already computed by both scorers). On the s1
profile it weights the mid window 61%, the hold 21%, the first 0.3 s 18%. Rejected because a diverged episode
accumulates fewer steps and so scores a LOWER ITAE, it has no ControllerMetrics field, and I did not find it
in the CPU/GPU parity pins.

### Arms, base, rule (as amended)
- **Base: open — D+F or Y+F, Luiz picks after F on Y reaches n=10. Luiz is leaning Y+F (19/11).**
- **ABI:** with the Rust row, A2 flies on ABI 60. The base's F-block rows (ABI 59) remain the B rows provided
  the flag-off determinism gate ABI 60 vs ABI 59 passes (`scripts/ctrl_determinism_gate.sh`), as for every
  default-off bump. With the fallback, A2 flies on ABI 59.
- **Expectation (replaces the earlier bullet):** steady −0.01 to −0.03°; err −0.01 to −0.03° (through the
  roll / pitch transient, which differs by seed: +0.017 / +0.016 above the label law on s1, +0.076 / +0.062 on
  s3); pos flat; 0 fails; runs somewhat longer.
- **Rule (replaces the earlier bullet; a starting point, Luiz):** A2 − B, paired by seed, TEST over 35 CF2.1
  units: err + steady together better by > 0.03° with neither worse by > 0.01°, pos not worse by > 0.010 m,
  fails within the gate. Luiz judges the whole vector; a metric better on the seed is a win on that seed.
- Report the per-axis split and the `mid` column of each headline next to the triple.

## To fill at launch
- Base (D+F or Y+F — Luiz leaning Y+F) and its B rows; driver path; tags `..._a<ABI>_..._a2_full_sN`
  (a60 with the Rust `mid` row, a59 with the `pre` fallback).
- Luiz's confirmation of the amended weights.
