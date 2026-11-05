# Multi-axis programme (CTRL-5) — specification (draft 4, 05/11/2026)

Status: DESIGN, not armed. Draft 4 re-bases draft 3 (19/09/2026, commit 3b8affdb — `git show
3b8affdb:docs/multi_axis_programme_spec.md` for the full D0 / stale-altitude history it carried)
onto the current controller: wheel ABI 33, the ABI-32 real-life stage-2 plant, rotor drag k_d 0.78,
and the CTRL-8 recipe. Nothing is launched until the Stage 0 checklist (§5) is closed AND the CTRL-8
arm-vs-twin verdict has named the anchor.

## What changed since draft 3

| # | draft 3 (19/09, ABI 29/30) | draft 4 (05/11, ABI 33) | evidence |
|---|---|---|---|
| 1 | Anchor `_hd29` (stage-1 ladder recipe, 180k-cell cap, MEMORY row) | Anchor = **the CTRL-8 winner recipe**, expected the ARM (position inputs) — §1 | experiments/ctrl8_ab_rule.json; logs/controller/ctrl8_ab_pop50/ |
| 2 | ABI 29/30, stage-1 plant, inclinometer accel | ABI 33: exact box-QP MPC (9dd6d762); ABI 32 real-life plant under --translation = specific-force IMU, rotor drag, WNN attitude features from the Mahony estimator, --att-err-ref tilt (f99611bf) | src/wnn/control/_accel.py:87-96 (EXPECTED_ABI 33) |
| 3 | No rotor drag | k_d = 0.78 1/s default (Folk 2026, mass-scaled; 1e9b12c9), chosen by the pop-50 drag sweep | phased_ga.py:3451-3459; logs/controller/ctrl8_drag_sweep_pop50/sweep.log |
| 4 | 3.2 h/run | **~9 h/run measured** (7.2-10.8 h); axis C budgeted 18 h | §4 |
| 5 | Round 1 = 12 runs ~46 h; total 40 runs ~215 h | Round 1 = 9 runs ~99 h; total 48-50 runs ~520-540 h at n=5 | §4 |
| 6 | n=4 per condition + anchor extended to 8 | n=5 per condition (house rule), anchor s2-s6 from CTRL-8 + D4 extension s7-s9 | §4 |
| 7 | ONE primary column per axis is the verdict | Holistic verdict (CTRL-8 rule_v2 template): attitude block decides, pos3d/alt tie-break; per-axis focus metric only sizes power | §2, §6 |
| 8 | MEMORY multi-seed row is the surface | REPORT headline only (report seeds), five metrics stable/err/steady/alt/pos3d (+dx/dy) | R7, R8 |
| 9 | Baselines via compute_baselines (stage-1, attitude-only) | Rival bar per condition via scripts/ctrl22_stage2_rivals.py on the condition's own argv | R6 |
| 10 | Axis F plumbing open; --motor-lag 0.0375 | CLOSED at ABI 30; flag is `--motor-lag-s` = 2% SETTLING TIME T (T=0.15 nominal, 0.30 stress) | phased_ga.py:2700-2707 |
| 11 | Trainer-airframe gap (#14/#15) open | CLOSED at ABI 30; `_hd30` re-fly superseded by the ABI-33 lineage | evaluator.py:1021-1033 |
| 12 | Accelerometer hover-approximation disclosure | RESOLVED by ABI 32 (specific-force accel + drag; WNN reads attitude via Mahony) | f99611bf |
| 13 | multi_axis_chain.sh ready | STALE: wraps the `_hd` ladder (sweep_ladder_gamma.sh / seed_arm_chain.sh), would fly the wrong recipe — rewrite | scripts/multi_axis_chain.sh:3-14 |
| 14 | — | NEW open items: cf2x_firmware has no sourced k_d; axis C memory budget re-opened (anchor now ~1.6 M cells uncapped); stage-2 rescoring tool; wheel freeze | §5 |

## 0. Closed history (D0, stale-altitude, trainer-airframe, CTRL-17)

All CLOSED; detail in draft 3 (3b8affdb) and the memory notes. One line each:
- D0 teacher hover anchoring — derived hover is DEFAULT (a88cb7c1, A/B equivalent 4/4); the label
  re-base is teacher-agnostic through `teacher.hover()` (dagger_train.rs:1240-1259), so PidFw/LQR
  labels are re-based too — the draft-3 "pid trainer saturates under translation" blocker is fixed in
  code, but NO pid/lqr-teacher translation run has ever flown (Stage 0 #7 smoke checks it).
- D8 stale-altitude replay — FIXED ABI 28 (ReplayObs; ABI 32 adds `att_up` to the stream).
- Trainer never got the airframe — FIXED ABI 30 (evaluator.py:1021-1033 packs `ec.airframe_kwargs()`).
- CTRL-17 stage-2 trainer gaps G1-G4 — FIXED ABI 31 (_accel.py:79-86).
- Every banked marker from those eras is a DIFFERENT LINEAGE: never paired with ABI-33 runs (R9).

## 1. The claim under test

Every stage-2 controller result sits at ONE point of the design space:

    airframe cf21_brushless · ABI-32 plant (specific-force IMU, rotor drag k_d 0.78, Mahony attitude
    features, tilt reference) · disturbance L4C · teacher mpcof (ABI-33 exact MPC) · lag 0 · sn=0 ·
    stage 2 (xy-offset 0.5 m, 2 s episodes, w_pos 0.10) · b24 n256→512 · seeds 31337002..

A reviewer asks whether the ranking measured there survives a move along any axis. The programme
answers it one axis at a time from the anchor, recipe FROZEN (the thermometer encoder is re-fit per
condition from that condition's PID rollouts — stated, not hidden), LQR/LQI/MPC/MPCOF re-deriving
their gains from the plant, PID flying the sourced firmware gains of each airframe, and the position
and altitude outer loops (position_loop.rs, altitude_pd.rs) deriving from the plant for every teacher.

### 1.1 THE ANCHOR = the CTRL-8 winner recipe (verdict pending; the ARM is expected)

The CTRL-8 A/B (rule experiments/ctrl8_ab_rule.json `rule_v2`, holistic: attitude block decides, pos3d
tie-breaks; one interim look at n=3) decides between ARM (position inputs) and TWIN (position-blind).
The anchor is the WINNER's exact argv. Expected: the ARM. Argv as flown (arm_s3.out `[argv]`, last-wins
duplicates kept verbatim — this is what the chain must reproduce, not a cleaned version):

```
--levels 16 --lamarckian --skip-stages neurons,bits --max-cells 180000 --max-cells-strict --neurons-gens 5
--neurons-patience 3 --conns-gens 5 --conns-patience 3 --memory-gens 120 --memory-patience 2 --pop 50
--num-eval-folds 5 --check-interval 2 --magnitude-aware-patience --eval-episodes 100 --memory-eval-episodes 200
--steps 2000 --tilt 5.0 --fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0
--delta-gamma 1.0 --grid-bits 24 --grid-output-neurons 256 --max-output-neurons 256 --report-episodes 100
--holdout-pop-sample 8 --runs 1 --memory-mode BINARY --airframe cf21_brushless --disturbance L4C --teacher mpcof
--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i --obs-collective-cmd
--obs-alt-err --obs-vz --translation --reward-lambda-alt 0 --grid-state-neurons 0 --max-state-neurons 0
--report-seeds 99990101 99990102 99990103 99990104 99990105
--obs-pwm --skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000 --teacher-hover derived
--base-seed <seed>
--fit-weight-err-sq 0.28125 --fit-weight-stable 0.225 --fit-weight-steady 0.39375 --fit-weight-pos 0.1
--xy-offset 0.5 --att-err-ref tilt --neurons-gens 15 --neurons-patience 3 --eval-mem-budget auto --rotor-drag 0.78
--obs-pos-err-xy --obs-vel-xy                      # ARM only; the TWIN omits this line
```
Effective: pipeline GRID → NEURONS (15 gens / patience 3) → MEMORY (120 / 2); obs-pwm; output neurons ≤ 512;
cells uncapped; S16noJM weights x0.9 + w_pos 0.10, zscore combine, alt weight 0.0 (CTRL-19 closed), no MAD
floor (CTRL-20 closed); 2 s episodes; provenance abi=33 wheel_sha256=a6398538ce04588c (arm_s3.out:7).
Built by scripts/ctrl8_ab_pop50.sh (BASE = experiments/crn_noise_probe/ARGS_…_s31337002_op31.txt minus
save/seed/weights, + W + S2 + POS). FILL IN at verdict: winner = {arm|twin}, decided on n={3|5}, date.

PROVISIONAL anchor numbers — REPORT headline (stage-select crowned genome on report seeds 99990101-05),
ABI 33, k_d 0.78; mean ± SD over search seeds. NOT a result until the A/B closes.
```
controller          n   stable %      err °        steady °     alt m          pos3d m
ARM  (s2,s3,s4)     3   95.5 ± 1.4    2.67 ± 0.14  1.99 ± 0.15  0.181 ± 0.020  0.422 ± 0.009
TWIN (s2,s3)        2   89.5          3.17         2.84         0.171          0.444      (twin_s4 flying)
rival bar, xy 0.5 m, 2000 steps, k_d 0.78, estimator-fed (experiments/ctrl22/rivals_abi33_tilt_kd0.78.json):
MPCOF               5   100.0         1.80 ± 0.03  1.06 ± 0.05  0.16 ± 0.01    0.35 ± 0.02   the teacher
LQI                 5   100.0         1.92 ± 0.05  1.27 ± 0.05  0.16 ± 0.01    0.35 ± 0.01
LQR                 5   100.0         1.94 ± 0.09  1.20 ± 0.06  0.16 ± 0.01    0.35 ± 0.01
MPC                 5   100.0         2.24 ± 0.23  1.46 ± 0.19  0.16 ± 0.01    0.36 ± 0.01
PID                 5   100.0         2.99 ± 0.38  2.50 ± 0.45  0.16 ± 0.02    0.35 ± 0.01   weakest classical
```
(rival SDs are over the 5 report seeds. WNN headline per seed: arm s2 97.0/2.55/2.02/0.163/0.411,
s3 95.0/2.63/1.83/0.202/0.429, s4 94.4/2.83/2.13/0.178/0.425; twin s2 91.4/3.04/2.64/0.179/0.451,
s3 87.6/3.30/3.03/0.163/0.436.)

What the provisional anchor supports: the WNN beats PID on err and steady and is below every rival
on stable and pos3d; it sits 0.87° err above its own teacher. Whether that position is LOCALLY robust
is the programme's question.

THE HEADLINE QUESTION:
    Q1. With the recipe frozen, does a one-axis move change the controller as a whole (stable, err,
        steady, alt, pos3d), and does the WNN's position relative to the rival bar at that condition
        (above PID on err/steady, below all on stable/pos3d) persist?

### 1.2 Already flown at n=1 — the drag axis (reference, not part of the programme)

The pop-50 drag sweep is a one-seed OFAT on k_d (arm recipe, s31337002, ABI 33), headline:
```
k_d    WNN stable/err/steady/alt/pos3d            MPCOF rival err/steady (xy 0.5, 2 s)
0.78   97.0 / 2.55 / 2.02 / 0.163 / 0.411          1.80 / 1.06
0.50   93.2 / 2.66 / 2.05 / 0.170 / 0.406          1.95 / 1.29
0.25   83.6 / 3.64 / 3.58 / 0.209 / 0.453          2.11 / 1.53
0.00   86.2 / 3.72 / 4.17 / 0.180 / 0.420          2.31 / 1.81
```
n=1 per value: a direction, not a measurement. It is not re-flown here unless Luiz asks (Q6).

## 2. Rules (R1-R11)

R1. VERDICT is HOLISTIC on the five metrics together (§6), on the CTRL-8 rule_v2 template: the attitude
    block (stable, err, steady) decides; alt and pos3d tie-break; mixed outcomes go to Luiz without an
    automatic verdict. Each axis names a FOCUS metric that sizes its power (R2); it is not a gate.
    Win/loss tallies are never a verdict.
R2. SIZE BEFORE FLYING with exact noncentral-t MDEs (80 % power, α .05 two-sided). From the provisional
    ARM n=3 SDs (stable 1.36 pp, err 0.144°, steady 0.152°, alt 0.020 m, pos3d 0.009 m):
        paired n=5 (SD_d = √2·SD)    stable 3.2 pp  err 0.34°  steady 0.36°  alt 0.047 m  pos3d 0.022 m
        Welch anchor 8 vs cond 5     stable 2.4 pp  err 0.25°  steady 0.27°  alt 0.035 m  pos3d 0.017 m
    A 3-seed SD is itself uncertain (95 % CI [0.52x, 6.3x]) — QUOTE RANGES and refresh at the anchor's
    n=5 (and again per condition after its 2nd seed). A null whose MDE exceeds the axis MEI is
    INDETERMINATE, never a refutation.
R3. pos3d is the best-resolved channel at stage 2 (SD 0.009 m); alt next (0.020 m). They tie-break, and
    are the focus for axes whose read is about position/altitude hold.
R4. Same search seeds (31337002..) and the same report seeds everywhere. Pairing by search seed is
    cosmetic (pair correlation ≈ 0 in every earlier arm), so PRIMARY analysis = Welch vs the extended
    anchor; paired is the robustness line.
R5. Train ORACLE, compare ESTIMATOR-FED: every rival row is the [est] row (ctrl22 `feed: estimator`).
    The WNN's own attitude features come from the Mahony estimator (ABI 32) — stated in every table.
R6. Every condition gets its OWN rival bar, scored by scripts/ctrl22_stage2_rivals.py on the
    condition's own argv (per-condition ARGS file; xy 0.5 m, 2000 steps cell) BEFORE its WNN runs,
    and recomputed on any new report-seed set (R10). Never compare across conditions' bars. Conditions
    that do not change the plant (B teacher, C state neurons) share the anchor's bar.
R7. Surface = the REPORT HEADLINE only (stage-select crowned genome on the report seeds). No
    STAGE-SELECT/VAL tables in reports (Luiz 25/10). Gen-line numbers are never results.
R8. Five metrics on every surface: stable %, err °, steady °, alt m, pos3d m (dx/dy alongside). Rival
    rows carry their SDs. Altitude and pos3d are never omitted.
R9. PROVENANCE + WHEEL FREEZE. Every run records `[provenance]` (abi, wheel_sha256). The programme flies
    on ONE wheel: ABI 33, sha a6398538ce04588c (the CTRL-8 anchor's). Never pair runs across ABIs or
    wheel hashes. Any controller-wheel change during the programme stops it; resume only after an
    anchor re-fly or a demonstrated bit-identity pin on the anchor argv (pop-6 smoke RESULT lines
    identical pre/post).
R10. REPORT SEEDS: D5 DECIDED 19/09 — final table on 99990201..05; interim reads keep 99990101..05. The
    CTRL-8 anchor was read on 99990101..05, so the final table needs every winner RE-SCORED on D5 (Stage 0
    #12) and every rival bar recomputed on D5 (#6).
R11. stable is a bounded per-episode pass rate: report failures as counts (`stable_fail=k/n`, 50a9bdba)
    with an exact Clopper-Pearson CI where exported; a t-CI on stable % is approximate.

## 3. The axes — values, focus metric, MEI, prerequisites (re-checked 05/11 against the tree)

### Axis A — DISTURBANCE (L4A, L4B vs anchor L4C)                 2 conditions, 10 runs, ~90 h
Values: the sourced L4 rung (training.py:231-235): L4A clean plant, L4B 10 %, L4C 20 %. L4A removes ONLY
the per-episode static plant draw (torque_scale_jitter, motor_asym); sensor noise (ADIS16448) is shared
by every rung (training.py:236-239), and the translation draws (mass/collective jitter, alt offset,
xy start) stay randomised. The plausible "L4A is harder" mechanism is teacher-label collapse (mpcof
d̂ ≈ 0 → more labels in the dead zone), not dither.
Focus: ERR. MEI 0.6° (the plant-jitter step moved PID 0.58 → 1.79° at stage 1; the stage-2 bar at
L4A/L4B is not yet scored — Stage 0 #6).
A-cross (scoring only, minutes): the anchor winners scored at L4A/L4B — needs the stage-2 rescoring
tool (#12).
Prereqs: rival bars at L4A, L4B (#6); smoke (#7). No code.

### Axis B — TEACHER (pid, lqr vs anchor mpcof)                    2 conditions, 10 runs, ~90 h
Values: pid (the weakest classical; quality span) and lqr (memoryless). Both fly the SAME outer
position + altitude cascade as mpcof — the loops are teacher-independent (episode_regime.rs:40-74,
dagger_train.rs:1612) — so the swap changes only the inner attitude law.
NEW READ (stage-2 rival bar, xy 0.5): MPCOF 1.80°, LQR 1.94°, PID 2.99°. The lqr-vs-mpcof quality gap
is now 0.14° — below every MDE — so the lqr arm tests "memoryless vs observer teacher at ≈ equal
quality" only; the TEACHER-QUALITY question rests on pid (gap 1.19°).
Focus: ERR. The "student ignores its teacher" null is an equivalence claim: TOST margin 0.6° (half
the pid gap), decidable only if |d| < 0.2° and SD ≤ 0.3°.
ABI 33's exact MPC changed only mpc/mpcof; pid/lqr arms are unaffected, the anchor's teacher is ABI 33.
Prereqs: smoke per teacher (#7) — no pid/lqr-teacher translation run has EVER flown (no
logs/controller/*.out `[argv]` line with --translation and --teacher pid|lqr); the smoke must show the
re-based labels do not saturate. Shares the anchor's rival bar.

### Axis C — STATE NEURONS (sn=4, sn=8 vs anchor sn=0)             2 conditions, 10 runs, ~180 h (budget)
Values: --grid-state-neurons N --max-state-neurons N. Prior (attitude-only, Aug 2026): S1_lqi_sn8
hd 0.0535, sn4 0.0618 — no row under translation, none under stage 2.
Focus: ALT and pos3d (the hold channels a recurrent state could smooth); err descriptive.
Path: sn>0 trains through the CPU EDRA-BPTT window trainer (no GPU split with the vertical channel).
The replay now must carry vert_obs, horiz_obs, pwm_dev (obs-pwm) and att_up (ABI 32) per step;
`replay_obs_guard` refuses a misaligned stream. sn>0 with obs-pwm + Mahony attitude features +
position inputs has NEVER run — the smoke (#7) is the first test.
MEMORY BUDGET RE-OPENED (#8): the 19/09 decision (sn=4 uncapped ~18 GB; sn=8 at --max-cells 4000000)
was sized against a 275k-cell sn=0 anchor under the 180k cap. The anchor is now UNCAPPED at ~1.6 M
cells per genome (_op31 s2 marker: cells μ 1584k, peak RSS 11.4 GB). sn=4 uncapped will be larger
(unmeasured); a 4 M cap on sn=8 is now only ~2.5x the anchor, a different condition than it was.
Watchdog HOG_GB 28 (controller_mem_watchdog.sh:89) and the IDS worker (~10 GB) bound the box.
--eval-mem-budget auto (50 % of available) limits eval, not training cells.
Wall-clock: unmeasured on this recipe; budgeted 18 h/run (2x anchor). The first sn=4 run IS the
measurement; re-cost after it.
Prereqs: #7 smoke (plumbing), #8 budget decision, Q4.

### Axis D — AIRFRAME (cf2x_firmware vs cf21_brushless)             1 condition, 5 runs, ~45 h
Values: cf2x_firmware only (D1); cf2x_urdf refused (training.py `_pid_cascade_kwargs` ValueError, landed
12/09) until a citable DSL PID exists. The trainer now gets the airframe (ABI 30), so D is no longer void.
NEW BLOCKER — ROTOR DRAG HAS NO SOURCE FOR cf2x_firmware: DEFAULT_ROTOR_DRAG = 0.78 is airframe-
independent (phased_ga.py:3459, applied at :3477 to any --translation run) and is sourced only for the
39.3 g CF2.1-Brushless (Folk 2026). cf2x_firmware is 29 g (airframe.py:229). Options (Q3): (i) Förster
2015 rotor-only CF2.0 ≈ 0.25 (cited in the DEFAULT_ROTOR_DRAG comment); (ii) Folk mass-scaled
0.647·√(57/29) ≈ 0.91 — a cross-airframe extrapolation, unsourced; (iii) keep 0.78 and disclose. Since
k_d moved the n=1 WNN by ~1.1° err between 0.78 and 0.25 (§1.2), this choice is not cosmetic: the
condition is (airframe, k_d) and the airframe effect is confounded with the drag choice unless (i)/(ii)
is sourced. 
Focus: ALT and pos3d (does the hold transfer to a different thrust/inertia map); err descriptive.
cf21_brushless inertia is DERIVED, cf2x_firmware's MEASURED — D is also the inertia-sensitivity check.
Prereqs: Q3; rival bar at (cf2x_firmware, chosen k_d) (#6); smoke (#7).

### Axis F — ACTUATOR LAG (T = 0.15 s, 0.30 s vs anchor 0)            2 conditions, 10 runs, ~90 h
Molchanov eq. 7 motor lag. UNITS: `--motor-lag-s` is the 2 % SETTLING TIME T, τ = T/4 (phased_ga.py:2700-
2707): nominal T = 0.15 (τ 37.5 ms), stress T = 0.30. Passing 0.0375 would model a 4x faster motor.
PLUMBING CLOSED (ABI 30): trainer packed cfg + per-round eval, both scorers, classical baselines,
address recorder, thermometer calibration (_accel.py:71-78; classical_baseline.py:183,206; dagger.py:420-424;
evaluator.py:677-679,900-902,1010; ga_memory.py:54-57). The ctrl22 rival script builds the regime through
episode_config_from_args, so a per-condition ARGS file with --motor-lag-s carries the lag to the rivals.
Never flown at lag > 0 (no .out argv with --motor-lag-s > 0).
Focus: ERR (lag degrades every controller; the read is whether the WNN degrades MORE than the rivals).
Expected effect large. Reviewer-priority axis.
Prereqs: rival bars at T 0.15/0.30 (#6); smoke (#7).

### Axis G — CONTROL RATE (action_repeat 2 = 500 Hz)                optional, 5 runs, ~45 h
D7: optional tail after round 2. Not costed in the total.

### Axis E — LEVELS                                                     NOT FLOWN (D2)
Dropped as an attitude axis (19/09). Re-open on altitude only if Luiz asks.

DISCLOSURES (not axes): wind — all L4 rungs windless by design (disturbance_param_sources.md); inertia/
mass mismatch on the attitude plant IS torque_scale_jitter (axis A) plus axis D; the outer position/
altitude loops of every RIVAL are derived cascades (position_loop.rs:1-36) — the WNN is monolithic.
The draft-3 accelerometer disclosure is RESOLVED by the ABI-32 specific-force IMU.

## 4. Design: rounds, anchor, budget

INTERLEAVE, DON'T STAGE (D3): round r flies seed r of every condition before any condition gets
seed r+1; a condition's SD exists after its 2nd seed, which is the re-size point.

ANCHOR. The CTRL-8 A/B supplies the anchor's s2..s6 (arm or twin, whichever wins) on the same wheel
and argv. If the n=3 interim stop fires, only s2..s4 exist and s5, s6 must fly first (+18 h). D4
extension s7..s9 (+27 h) feeds the Welch primary (anchor n=8 vs condition n=5).

COST BASIS (measured, pop 50, ABI 33, k_d sweep + A/B, box shared with the IDS worker):
arm_kd078 7.2 h · arm_kd050 8.7 · arm_kd025 9.7 · arm_kd000 9.9 · twin_kd078 10.1 · arm_s3 8.2 ·
twin_s3 7.4 · arm_s4 7.5 · twin_s4 > 10.8 (flying). Mean of the 8 finished 8.6 h; PLANNING FIGURE 9 h.
(For scale: _hd29 3.1 h — marker dur_s ~11.2k; _pipeN30 8.6-11.6 h; _op30 11.5-12.9 h; _op31 8.0-15.3 h.)
Axis C: 18 h/run budgeted, unmeasured. One controller at a time; IDS has priority and is never
paused — controller wall-clock stretches when the IDS load rises.

ROUND 1 (once Stage 0 is closed and the anchor is named):
```
order  condition                runs  h      why here
(0)    anchor s5, s6            2     ~18    ONLY if the CTRL-8 n=3 interim stop fired
1      A  L4A, L4B              2     ~18    no code; largest expected effect
2      F  T=0.15, T=0.30        2     ~18    reviewer priority; plumbing closed
3      B  pid, lqr              2     ~18    smoke must show no label saturation
4      D  cf2x_firmware         1     ~9     after Q3 (k_d source)
5      C  sn=4, sn=8            2     ~36    last: slowest, memory budget (#8) gates it
       round 1 total            9     ~99 h  (~4 days); +18 h if (0)
```
The anchor already has s2..s6 when round 1 starts, so every round-1 run reads as a DIRECTION against an
n=5 anchor immediately; the D4 extension (s7..s9) flies in round 2 where Welch needs it.

BUDGET (n=5 per condition):
```
anchor completion (s5,s6 only if interim stop)       0-2 runs    0-18 h
anchor extension D4 (s7..s9)                           3          27
A disturbance (L4A, L4B)                              10          90
F actuator lag (T 0.15, 0.30)                         10          90
B teacher (pid, lqr)                                  10          90
D airframe (cf2x_firmware)                             5          45
C state neurons (sn=4, sn=8), 18 h budget             10         180
total                                              48-50     ~520-540 h  ≈ 22-23 days of box time
G optional tail                                       +5         +45
```
Draft 3 was 40 runs / ~215 h; the ~2.5x is the 3.2 → 9 h run cost plus n=5. Trims, each Luiz's call
(Q1): n=4 per condition (-9 runs, ~-99 h); drop sn=8 (-5, ~-90 h); a pre-registered n=3 interim stop
per condition on the rule_v2 pattern (saves up to 2 runs per decisive condition).
Escalation (pre-registered): a condition whose verdict is "similar" while its CIs still contain the
MEI after n=5 gets +2 seeds — a human decision, not automated.
Every chain: marker-gated, idempotent, fails closed, HOLD-aware, one controller at a time, never edits
a running .sh, never launched beside IDS memory pressure (the controller is the sacrifice).

## 5. Stage 0 — prerequisites (status 05/11/2026)

```
#   item                                   status    evidence / what closes it
1   R9 provenance on every run              CLOSED    50a9bdba; arm_s3.out:7 `[provenance] … abi=33 wheel_sha256=a6398538ce04588c`
2   R11 failure counts                      CLOSED    50a9bdba; `stable_fail=` on MULTI-SEED lines (3 in arm_s3.out)
3   HOLD sentinel                           CLOSED    controller_arm_lib.sh; honoured by ctrl8_ab_pop50.sh:24
4   D0 teacher hover / label re-base        CLOSED    a88cb7c1; dagger_train.rs:1240-1259 (teacher.hover(), all teachers)
5   D5 report-seed set                      CLOSED    decided 19/09 (§8); implementation is #6 + #12
6   Rival bars per condition (ABI 33)       OPEN      exist: cf21/L4C at k_d 0.78/0.5/0.25/0 on 99990101-05
                                                      (experiments/ctrl22/rivals_abi33_tilt_kd*.json). Need: L4A, L4B, T0.15, T0.30,
                                                      cf2x_firmware@k_d — each from a per-condition ARGS file via
                                                      scripts/ctrl22_stage2_rivals.py; then all (+anchor) on D5. Minutes each.
7   Smokes, one per condition (pop 6)       OPEN      none of lag>0, sn>0+translation, cf2x_firmware+translation, pid/lqr
                                                      +translation has ever flown. Plumbing only (rc 0, GRID line, HEADLINE,
                                                      [PLANT] + [provenance] abi=33); B: no label saturation. Decisions at pop 50.
8   Axis C memory budget                    RE-OPENED anchor now ~1.6 M cells uncapped (_op31 marker), the 19/09 cap decision was
                                                      sized vs 275k; first sn=4 run measures; Q4
9   Round-major chain                       OPEN      scripts/multi_axis_chain.sh (111626af) wraps seed_arm_chain.sh →
                                                      sweep_ladder_gamma.sh = the `_hd` ladder recipe (180k cap, no stage 2,
                                                      no rotor-drag/estimator flags). REWRITE on the ctrl8_ab_pop50.sh pattern:
                                                      anchor argv + condition flags last-wins, winner-file gated, HOLD-aware,
                                                      --dry-run; keep its table/escape-hatch design and the T-units fix.
10  Power statement per axis                OPEN      refresh R2 at the anchor's n=5; paired_power.py has --welch
                                                      (scripts/paired_power.py:19,61) but reads banked sweepladder MARKERS —
                                                      the CTRL-8 runs bank .out + winner only. Needs a HEADLINE-from-.out input.
11  Actuator-lag plumbing (F)               CLOSED    ABI 30; see §3-F evidence
12  Stage-2 rescoring (A-cross + D5)        OPEN      rescore_winners.py builds argv from (disturbance, airframe, translation,
                                                      tilt) only (rescore_winners.py:152-163) → would score a stage-1 regime.
                                                      Add a --run-args-file mode (as ctrl22_stage2_rivals.py does). Code, Python.
13  Anchor restated                         OPEN      blocked on #18; §1.1 table is provisional
14  Trainer gets the airframe               CLOSED    ABI 30; evaluator.py:1021-1033
15  `_hd30` anchor re-fly                   SUPERSEDED _hd30 4/4 banked (ABI 30) but the lineage moved to ABI 33 / stage 2
16  CTRL-17 stage-2 trainer gaps G1-G4      CLOSED    ABI 31; _accel.py:79-86
17  Accelerometer disclosure                CLOSED    ABI 32 specific-force IMU (f99611bf)
18  CTRL-8 arm-vs-twin verdict              OPEN      rule_v2 n=3 interim check pending twin_s4; names the anchor
19  cf2x_firmware rotor-drag value          OPEN      no source (§3-D); Q3
20  Wheel freeze for the programme          OPEN      Luiz confirms R9 freeze at ABI 33 sha a6398538ce04588c (Q5)
```
BLOCKS ROUND 1: #6, #7, #9, #10, #18 (all axes); #12 (A-cross and the D5 final table, not the runs);
#19 (axis D only); #8 (axis C only); #20 (any wheel change resets everything).

Idle-window procedure (unchanged in shape): smokes and rival bars need no wheel install; the controller
wheel is NOT reinstalled for the programme (R9). Smoke shape (no marker, numbers NOT read):
    cd /Users/lacg/wnn && PYTHONPATH=src/wnn /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m \
      wnn.control.phased_ga <anchor argv §1.1> --pop 6 --steps 200 --neurons-gens 1 --memory-gens 1 \
      --eval-episodes 2 --memory-eval-episodes 2 --report-episodes 2 --base-seed 1 [+ condition flags, last-wins]
    assert rc=0; grep -E '^\[provenance\]|^\[PLANT\]|HEADLINE|GRID'.

## 6. Verdict protocol (pre-registered, per condition)

  V1. Δ(WNN_condition − WNN_anchor) on ALL FIVE metrics (stable, err, steady, alt, pos3d): Welch mean
      and 95 % CI (anchor n=8 once extended, else n=5), paired as robustness. dx/dy printed alongside.
  V2. HOLISTIC READ on the rule_v2 template, margins = the axis MEIs (similar = CI inside ±margin):
        1. attitude block (stable, err, steady): if every metric is better-or-similar in one direction
           and at least one is beyond its margin, the condition MOVED the controller in that direction;
        2. all three similar → alt and pos3d decide (beyond margin = moved; else UNCHANGED);
        3. mixed (one better, one worse beyond margin) → no automatic verdict, reported to Luiz.
  V3. Position vs the condition's own rival bar (R6), as a table with SDs: is the WNN still above PID on
      err/steady and below every rival on stable/pos3d? That is the robustness READ of Q1.
  V4. An indeterminate condition is written up with the n it would need and, if its CIs include the
      MEI after n=5, triggers the §4 escalation (human decision).
  Reports: REPORT headline only; all five metrics together; never a single-metric gate.

## 7. What this programme is NOT

- Not a search for a better operating point: hyperparameters frozen; the encoder re-fit per condition
  is part of the recipe, stated.
- Not a factorial: interactions are out of scope; the claim is LOCAL to the anchor.
- Not a replacement for CTRL-8: it waits for CTRL-8's verdict and inherits its winner.

## 8. Decisions

DECIDED 19/09/2026 (Luiz): D1 cf2x_firmware only · D2 levels dropped · D3 round-major · D4 anchor to 8
seeds · D5 final report seeds 99990201..05, interim 99990101..05 · D6 axis F added (T units per §3-F) ·
D7 axis G optional after round 2 · D9 anchor = `_hd29` — SUPERSEDED 05/11 by the CTRL-8 winner (draft 4).
DECIDED since: alt weight 0.0 (CTRL-19, 28/10); MAD floor not adopted (CTRL-20, 30/10); ABI-32 real-life
plant (31/10, option A); k_d 0.78 default (03/11); CTRL-8 rule_v2 holistic (05/11).

OPEN for Luiz:
  Q1. Scope/cost: n=5 per condition (~520-540 h) as the house rule says, or a trim — n=4 (-99 h), drop
      sn=8 (-90 h), or a pre-registered n=3 interim stop per condition (rule_v2 pattern)?
  Q2. D4 anchor extension to 8 seeds (+27 h): keep, or anchor n=5 with paired + Welch 5-vs-5?
  Q3. Axis D rotor drag for cf2x_firmware: Förster 0.25 (sourced, CF2.0 rotor-only), Folk mass-scaled
      ~0.91 (unsourced extrapolation), or 0.78 disclosed?
  Q4. Axis C memory: sn=4 uncapped and sn=8 at --max-cells 4000000 as decided 19/09, or re-cap now that
      the anchor is ~1.6 M cells (e.g. sn=8 uncapped only with HOG_GB raised — a safety change)?
  Q5. Freeze the controller wheel at ABI 33 (sha a6398538ce04588c) for the whole programme (~3 weeks)?
  Q6. Re-fly the drag axis (§1.2) to n=5, or leave it as the n=1 reference it is?

## 9. Reviews

Draft 2 folded the experiment-design review (11/09, 19 findings); draft 3 the flight-dynamics review
(11/09, 15 findings) — both listed in draft 3 §9 (3b8affdb). Draft 4 has NOT been reviewed: before
arming, route §2/§4/§6 (holistic verdict with Welch, n, interim stops) to experiment-design and §3-C/D/F
(sn>0 replay under the ABI-32 plant, cf2x k_d, lag units) to flight-dynamics.
