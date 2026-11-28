# CTRL-83 — out-of-distribution stress panel: pre-registration (27/11/2026)

Approved by Luiz 27/11/2026. Written by `experiment-design`. Code: `logs/controller/ctrl83/` (`cells.py` = the grid,
`score_cell.py` = one controller × cell, `rowb.py` = the eq-info MPCOF row, `run_grid.py` = the resumable runner).

**Partition of every number this panel produces: TEST** = the 35 CF2.1 units 99990101..99990135 × 100 episodes = 3500
episodes per controller per cell, REPORT-ONLY. Each (controller × cell) is scored once. Nothing is trained, selected,
tuned or re-thresholded on these numbers, and no cell result may feed back into any recipe, rival or grid choice.

## 1. Question and claim shape

Do the published controllers keep working beyond the plant range they were trained on, and does the WNN's standing
against the classical rivals survive? Two reads, both fixed here:

- **R1 shape** — per controller and axis: GRACEFUL / DEGRADED / CLIFF (§5.1).
- **R2 rank** — per primary cell: is the in-range ordering WNN vs each rival preserved, on each metric (§5.2)?

## 2. Controllers (frozen)

```
id       | controller                                                       | source
---------+------------------------------------------------------------------+-----------------------------------------------
wnn_s1-5 | WNN adopted base l2b, seeds 1..5 (the published winners)          | ctrl48/C-S2-d_wposx2_a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2b_full_s{1..5}_winner.yaml.gz
mpcofB   | MPCOF>rate(h10) est-fed + attitude-bias comp — THE comparator    | cargo probe rival_probe_tests::test_units_rival_rows_rescore_f32_lineage (rowb.py)
mpcofA   | MPCOF>rate(h10) est-fed, no bias comp (row A)                    | classical_baseline.teacher_metrics
mpcofC   | MPCOF>rate(h10) oracle-fed (row C: bound + label law)            | classical_baseline.teacher_metrics
pid      | PID firmware cascade, est-fed (row F, the only as-deployed row)  | classical_baseline.teacher_metrics (pid_metrics)
lqi      | LQI motor-level, est-fed                                         | classical_baseline.teacher_metrics
```

Row naming follows `docs/ctrl_rival_row_ruling.md` §0–1: row B is the comparator for every "vs MPCOF" sentence; A and C
are shown and labelled. **The WNN is frozen as published: genome + thermometer thresholds.** Thresholds are part of the
address function (`phased_ga._test_thresholds`), so they are always fitted on the NOMINAL plant (the run's own
train-seed fit); a stressed cell changes only the plant the episodes fly. Re-fitting thresholds on a stressed plant
would be re-adaptation, not robustness.

## 3. The grid (one factor at a time)

Training values of l2b (every seed's `[argv]`): `--disturbance L4C` (motor wear ±0.20, torque jitter ±0.20, ADIS16448
sensor noise), `--mass-jitter` default 0.15, `--motor-lag-s 0.20`. **`--motor-lag-s` is Molchanov's 2 % settling time
T, not τ (τ = T/4 = 50 ms)**; the levels below are T values.

```
cell    | axis  | level (training)              | prim | mechanism (no src change)                          | reaches WNN + rivals identically?
--------+-------+-------------------------------+------+----------------------------------------------------+-----------------------------------------------
nom     | —     | training values               | yes  | the run's own EpisodeConfig                         | yes (G2 + G4)
wear30  | wear  | motor_asym_mag 0.30 (0.20)    | yes  | dc.replace(DisturbanceConfig.motor_asym_mag)        | yes — same disturbance_stream draw, G2 gated
wear25  | wear  | 0.25                          | no   | same                                                | yes
wear40  | wear  | 0.40                          | no   | same                                                | yes
mass25  | mass  | mass_jitter 0.25 (0.15)       | yes  | dc.replace(EpisodeConfig.mass_jitter)               | same draws (G2); see §8 D2 (hover anchor)
mass35  | mass  | 0.35                          | no   | same                                                | same; §8 D2
lag30   | lag   | motor_lag_s T 0.30 (0.20)     | yes  | dc.replace(EpisodeConfig.motor_lag_s)               | same PLANT (G2); model-based rivals re-derive, §8 D1
lag10   | lag   | T 0.10                        | yes  | same                                                | same; §8 D1
lag40   | lag   | T 0.40                        | no   | same                                                | same; §8 D1
noise2  | noise | gyro σ, bias walk, accel σ ×2 | yes  | dc.replace(DisturbanceConfig.gyro_*/accel_sigma)    | yes for WNN + est-fed rivals; oracle row C cannot see it
noise3  | noise | ×3                            | no   | same                                                | same
—       | payload step mid-episode          | —    | UNSUPPORTED: mass is fixed per episode (stage1.rs Stage1Cfg.mass, set once at reset) | not scored; needs a src change (§8 D3)
```

How far beyond training the draws go (same units, same uniform draws scaled — the wear and mass draws are
`1 + U(-mag, +mag)` from the same seeded stream, so each unit's wear vector is the nominal one × mag/0.20):

```
cell    | per-unit max |wear|: median / max | units beyond ±20 %       || cell   | episodes beyond ±15 % mass | max
wear25  | 0.206 / 0.249                      | 20/35                    || mass25 | 1389/3500                  | 0.250
wear30  | 0.248 / 0.299                      | 28/35                    || mass35 | 1984/3500                  | 0.350
wear40  | 0.330 / 0.398                      | 33/35                    ||
```

Every cell flies the SAME 3500 episodes: same pool seeds (`fold_pool_seed(unit, 0)`), same ICs, same weather seeds;
only the stressed parameter changes. Cells are therefore PAIRED with `nom`, per unit and per episode.

## 4. Metrics (per controller × cell)

The whole vector, every cell, every controller (Luiz 17/11):

- **stable %** floored from fails (19/11 rule), **err °**, **steady °**, **pos m (3-D)**, **fails** (episodes, x/3500;
  WNN also pooled x/17500 and per seed).
- **Worst-case tail: p99.9** of the per-episode mean attitude error over the 3500 episodes (numpy `inverted_cdf`, the
  3497th order statistic = the 4th-worst episode), every episode capped at 8° (`episode_tail.rs`'s gate err limit) on
  BOTH sides at report time (raw values are cached). WNN: the scorer's `episode_errs` (it caps only the episodes it
  flags diverged — a flipped-but-not-diverged episode came back at 33.8° in the dry run, hence the uniform cap). Rivals:
  the scorer has no per-episode output, so the call is re-issued on every prefix 1..100 and the means differenced
  (gate: the episodes average back to the row). Once a controller has ≥ 4 episodes beyond 8°, p99.9 saturates at 8.000;
  **p99** (the `--select-tail` statistic) is printed beside it as a descriptive, non-primary column. Row B has no
  per-episode output (cargo probe): fails only.
- **Worst unit**: max over the 35 units of the unit-mean err.
- **Degradation**: Δ = cell − nom, paired per WNN seed (mean ± std, N = 5) and per rival (deterministic, one value);
  ratio cell / nom for err and steady.

WNN cells are `mean ± std (N = 5)` over seeds; rivals are deterministic laws, one value, no spread. The 5 seeds fly the
same 3500 episodes: 17 500 is not 17 500 independent episodes.

## 5. How the paper reads it

### 5.1 Shape per (controller, axis) — applied identically to every controller

Over all levels of the axis, on the controller's cell value (WNN: seed mean; fails pooled):

- **CLIFF** if at any level: fails ≥ 2.0 % (≥ 70/3500), or err ≥ 3 × nom err, or steady ≥ 3 × nom steady.
- **GRACEFUL** if at every level: fails ≤ 0.5 % (≤ 17/3500) and err ≤ 2 × nom and steady ≤ 2 × nom.
- **DEGRADED** otherwise.

The paper sentence per axis states the class AND the whole curve (all levels), for every controller — never one level.

### 5.2 Rank preserved vs a rival (primary cells only)

For metric X ∈ {fails, err, steady, pos, p99.9}, the in-range sign is sign(WNN − rival) at `nom`. At a primary cell:

- **flip** = the WNN seed mean is on the other side AND ≥ 4 of 5 seeds individually are on the other side;
- **preserved** = same side on ≥ 4 of 5 seeds;
- **indistinguishable at N = 5** otherwise (including any case where |WNN mean − rival| < WNN seed SD).

Comparator for "vs MPCOF" sentences = row B. Row C is a bound, row A the stock-estimator MPCOF, PID the as-deployed
firmware. The paper reports R2 against B and PID; A, C, LQI are context rows.

### 5.3 Falsifiers (written before any WNN stressed score)

- "The WNN degrades gracefully under stress S" is FALSE if the WNN is CLIFF on axis S.
- "The WNN keeps its stability edge over PID" is FALSE at a primary cell where fails flips (§5.2) against PID.
- "The WNN is as reliable as MPCOF" (the in-range claim, ruling §3) is FALSE at a primary cell where WNN fails exceed
  row B's by more than 17/3500 (0.5 %) on ≥ 4/5 seeds.

## 6. Multiplicity

- **Primary family (fixed): 5 stressed cells** — wear30, mass25, lag30, lag10, noise2 (the levels Luiz named). R2 is
  read on these only. wear25 / wear40 / mass35 / lag40 / noise3 are SECONDARY: they draw the dose-response curve for
  R1 and carry no rank claim.
- **No best-cell headline.** Every cell of the grid is reported, in the order of `cells.CELLS`, flattering or not.
  No cell is dropped, re-levelled or added after the first score.
- **No significance claim.** At N = 5 seeds against deterministic rivals, the strongest sign test is 5/5 (two-sided
  p = 0.0625), and 5 cells × 5 metrics × 2 rivals = 50 comparisons; nothing here survives a multiplicity correction,
  and the doc does not pretend otherwise. R2 is descriptive (seed consistency + magnitude vs seed SD) and is worded so.

## 7. Execution

- Interleaving: cell-major in `cells.CELLS` order (primary first), inside a cell every controller — a partial grid
  always holds complete cells.
- Gates (exit 2, nothing cached): G1 re-issued WNN row == scored metrics; **G2 plant identity** — in every WNN job, per
  unit, every plant / episode kwarg of the WNN call equals the rival call on the same stressed config (ICs, weather
  seed, resolved wear, mass draws, lag, noise, drag, IMU, airframe); G3 the stressed kwarg differs from nominal; G4 at
  `nom` the WNN reproduces its logged END and rows A / F reproduce the run's printed per-unit rival lines; row B: the
  probe's self-check (build == installed wheel on unit 1 of the stressed fixture) and probe row A == score_cell row A.
- Wheel: **ram_controller ABI 65** is what is installed (checked 28/11 04:10 UTC; the brief expected 64). G4 passed on
  ABI 65 for wnn_s1 and rows A at `nom`. Every cached result records its ABI; the panel must be completed on ONE ABI.
- Runtime (measured in the dry run, nice 15, RAYON_NUM_THREADS 6, ~1 core per job): WNN 20 s per (seed × cell), rival
  with per-episode tail 77 s, row B 55 s (build already cached). 11 cells × (5 × 20 + 4 × 77 + 55) s ≈ 1.4 h serial;
  105 jobs pending after the dry run (`run_grid.py --list`: 1.28 h at the planning estimates).
- **Disclosure.** While the tooling was being timed, before this doc was written, `mpcofA` was scored at `nom` (the
  known published row A, reproduced exactly: 1.352 / 0.684) and at `wear30`. The grid levels and the primary set in
  `cells.py` were written before any stressed score and were not changed after it; wear30 is primary because Luiz named
  ±30 %, not because of that look. After this doc: the dry-run cell `wear30` for wnn_s1, row A and row B (cached; the
  launch skips them). No other stressed cell has been scored.

## 8. Decisions for Luiz

- **D1 lag cells are matched-model for the model-based rivals.** `motor_lag_s` arms the plant AND every actuator twin:
  MPCOF and LQI re-derive their lag models from it (`optimal.rs` `motor_lag_tau`, `ctbr_mpcof.rs:100`), the WNN's d̂ /
  pwm twin is re-armed (`set_observer_actuator_core(sim.actuator())`), but the WNN's cells and PID's gains are fixed. So a
  lag cell reads "a different aircraft, with MPCOF/LQI redesigned for it" — a WNN-vs-MPCOF rank at a lag cell is
  confounded. A plant-only lag (model at T = 0.20, plant at T') needs a src change. Options: (a) score as is, R2 vs
  MPCOF/LQI on lag cells marked confounded, R2 vs PID and R1 kept; (b) drop lag from the rank claim; (c) build the
  plant-only kwarg first.
- **D2 mass: the WNN's collective anchor reads the TRUE episode mass.** `cpu_score.rs:285` anchors the delta
  accumulator at `Stage1Cfg::collective_pwm` = √(m_true g / 4k) (`stage1.rs:150`); the rivals' altitude PD is built
  from NOMINAL mass (`dagger_train.rs` ~4180-4194). Pre-existing and also in range; under mass25/35 it advantages the
  WNN by construction. Recommend: report mass cells with that label, no R2 "WNN beats rival on mass" claim, and ask
  `flight-dynamics` to confirm the rival side before the paper.
- **D3 payload step**: unsupported (mass fixed per episode). Needs a mid-episode mass change in `Stage1Cfg` + both
  scorers.
- **D4 noise**: the oracle row C never reads the IMU, so it is invariant to noise cells by construction; B/A/PID/LQI
  and the WNN see the same noise.
- **D5 ABI**: the panel runs on installed ABI 65. If ABI 65 is not adopted, re-run `nom` on the adopted wheel first
  (G4 decides).

## 9. Amendment 1 (28/11/2026, Luiz: "fix it properly") — controller ABI 66, branch `ctrl83-sim-fairness`

- **D1 resolved (option c, now the panel default).** `EpisodeConfig.model_motor_lag_s` holds every controller's
  design-time actuator model at the nominal T = 0.20 while the plant flies the cell's lag: MPCOF / LQI / LQR / MPC
  are built on 0.20 (`AirframeRs::designed`), their observer twins too, and the WNN's actuator twin is armed with
  0.20 (CPU `set_observer_actuator_core`, Metal `Params.twin_settling_time_s`). `cells._lag` sets it. Audit: no
  rival re-derives from wear, noise, rotor drag or the mass draw. At T = 0.20 it is bit-identical
  (`ctrl83_flag_off_pins`). **Every lag cell scored on ABI 65 is matched-model and is superseded.**
- **D2: investigated, flag only, default unchanged.** Both sides traced: the WNN's anchor AND its `coll_cmd` input
  are √(m_true·g/4k)·(1 + c) in training (`episode_regime::draw_translation_episode`), calibration, the recorder and
  both scorers; the l2b winners wire `coll_cmd` into 99–112 of 256 output neurons. The rivals use the nominal mass
  (`af_pid_hover_n`) and ignore c. Flag: `--anchor-mass nominal` (trainer + scorers + recorder, a lineage switch);
  default `true`. Mass cells keep the §8 label until Luiz picks the D2 option.
- **D3 resolved.** `payload_step_s` / `payload_mass_scale`: at t_step the plant mass becomes draw × scale (inertia
  unchanged: point payload at the CoM), the identical event on the WNN CPU / Metal scorers and every rival; no model
  is told. SECONDARY cells `pay25`, `pay50`, `drop20` (t = 1.0 s) — added after the prereg, no rank claim.

## 10. Amendment (Luiz 28/11) — the combined joint-OOD cell `all` is the single headline; controller ABI 69, branch `ctrl83-joint-ood`

**Decision (Luiz, 28/11/2026):** one COMBINED out-of-distribution cell, `all`, becomes "the default standard golden
value to pursue". From now on it is the **only robustness number on our tables**. The one-factor-at-a-time cells
(§3: `wear30`, `mass25`, `lag30`, `lag10`, `noise2`, and the secondary levels / payload cells) are **diagnostic-only
and are not reported**. §5.2's "rank preserved" read and §6's multiplicity budget apply to `all` alone; `nom` stays
as the in-range reference row and as gate G4.

**The cell.** Every episode draws ALL of these at once; everything else is the TEST preset (35 CF2.1 units × 100
episodes, the same ICs / weather seeds / pool seeds as `nom`, so `all` is paired with `nom` per episode):

```
stress            | all                                         | TEST preset (= nom)
------------------+---------------------------------------------+---------------------
motor wear        | motor_asym_mag 0.30 (±30 %)                 | 0.20
mass spread       | mass_jitter 0.25 (±25 %)                    | 0.15
motor lag (plant) | T_e ~ U(0.15, 0.30) s, drawn PER EPISODE    | T = 0.20 s
sensor noise      | gyro σ, gyro bias walk, accel σ ×2          | ×1 (ADIS16448)
every MODEL       | T = 0.20 (D1: rival designs, observer twins, the WNN's actuator twin) | 0.20
```

**Identical plant for every scorer.** The per-episode lag is drawn in Rust by ONE function,
`train_lag_range::scorer_episode_lags` (T_e from `disturbance_episode_seed(dist_seed, e)`, the ABI-68 CTRL-87 draw),
from ONE kwarg, `train_motor_lag_range=(0.15, 0.30)`, which `EpisodeConfig.stress_motor_lag_range` hands to every
scorer through `stress_kwargs()`: the WNN CPU / Metal scorers (ABI 68 already), `score_classical_baseline` (rows A,
PID, LQI — **new in ABI 69**) and the row-B cargo probe (fixture key `train_motor_lag_range`, ABI 69). The trainer
never sees it. Gates: G2 (`score_cell.plant_identity`) now also requires the lag-draw range to be equal on the WNN
and rival calls — with the weather seed and episode count already gated, that fixes T_e per episode; G3 requires
wear, mass, noise and the range to all reach the WNN call, the base lag (= the WNN twin) to stay 0.20 and
`model_motor_lag_s` to be 0.20. Rust pins (`ctrl83_all_cell_tests`): a degenerate range [0.20, 0.20] is the ABI-68
rival row bit for bit; a point range [T, T] on the 0.20 plant equals the fixed-T plant with the model held at 0.20,
bit for bit, on the rival AND the WNN CPU scorer (T = 0.30, 0.15); WNN CPU == GPU under the joint draw; the spread
moves every rival row. Every existing flag-off pin passes `None` and is unchanged.

**Read caveat (written before any `all` result is reported).** l2bnmw was TRAINED with `--train-motor-asym-mag 0.30
--train-motor-lag-range 0.15 0.30`, so for that arm the wear and lag axes of `all` are inside its training spread;
only mass ±25 % and noise ×2 are out of distribution for it. l2b and l2bnm trained at wear 0.20 and a fixed T = 0.20.
Any arm-vs-arm comparison on `all` must name this.
