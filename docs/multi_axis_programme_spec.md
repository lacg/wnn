# Multi-axis programme — specification (draft 3, 11/09/2026)

Status: DESIGN, not armed. Draft 2 folded in the experiment-design review (19
findings); draft 3 folds in the flight-dynamics review (15 findings). Both are in §9.
Nothing is launched until the Stage 0 checklist (§5) is closed AND the §0 blocker is
resolved.

## 0. D0 — the training-teacher hover anchoring: INVESTIGATED (controller agent, 11/09)

Draft 3 recorded the physics reviewer's claim as a blocker. The controller agent traced
it read-only, line by line, and the picture is narrower and different:

0.1 THE LABEL IS ANCHORED CORRECTLY BY CONSTRUCTION — no off-hover label exists.
    The live DAgger label is the teacher's absolute pwm through output_decode_target
    (controller.rs:5559-5567; legacy branch clamp(0,1) at :3452), and the student
    decodes it as a DELTA around self.neutral = 0.5 (:5636-5640; cell_mode.rs:147-157)
    added to an accumulator that leaks toward collective_anchor = the episode's TRUE
    hover 0.694·(1+jitter) (dagger_train.rs:1287-1306, controller.rs:3001). So a
    0.5-based teacher emitting 0.5/motor at level-and-on-altitude means "delta zero,
    hold the anchor" = hover. The feared 0.194 pwm offset is exactly zero; from_id's
    0.5 IS the label's neutral (optimal.rs:1399-1403, documented only as "legacy").
    The reviewer's proposed fix (anchor the trainer at 0.694) would label +0.194 pwm =
    12 grid levels at L=64 on every level step, integrating at G=4 to anchor+0.78 —
    saturation. DO NOT ADOPT IT AS WRITTEN.

0.2 WHAT IS ASYMMETRIC: THE LINEARIZATION POINT, and it is teacher-dependent.
    All four builds call calibrate_control_gains_rs(…, hover, 0.05) (optimal.rs:349,
    491, 906, 1209), probing at mix_to_motors_f64(hover, u) (controller.rs:6789-6812)
    on a pwm² plant (:2371), so b ∝ hover and b(0.5)/b(0.694) = 0.720 (reviewer right
    on this). Effect on K for cf21_brushless, from the closed forms:
        lqr / lqi   k1 unchanged; k2 +0.1% roll/pitch, +2% yaw   → same law as rival
        mpc / mpcof first-move gain +17% (angle 2.194 vs 1.871, rate 0.638 vs 0.544)
                    because the 25 ms horizon is b-sensitive       → a HOTTER law
    So the anchor's mpcof TRAINER is ~17% hotter (in normalized u) than the mpcof
    RIVAL the scorer uses; lqr/lqi trainers match their rivals.

0.3 MPCOF-SPECIFIC: the disturbance observer is mis-scaled. update_dhat uses the
    trainer's b(0.5) against the student's applied pwm (optimal.rs:1233-1260) while
    the student flies at 0.694 where true effectiveness is 1.39·b. Steady state: d̂
    absorbs 0.39·b·u_student, so u_ff = d̂/b (:1271-1276, τ≈20 ms, clamp 0.2)
    subtracts ~0.39·u_student from the label — a spurious negative feedback on the
    student's own action — and over-compensates real L4C disturbances by 39%. The
    scorer's mpcof (b at 0.694) has neither term.

0.4 BLAST RADIUS (rule, not list): trainer-vs-rival law mismatch exists iff
    cfg.translation AND the teacher id ∈ {lqr, mpc, lqi, mpcof}, on every wheel since
    35b1328d (14/08). Every SL_* ladder passes --translation --teacher mpcof, so ALL 78
    sweepladder markers (the 4 anchors, _leak090, _ls*, _mut1tap, _win2) and the
    translation A/B ON arms carry finding 0.2/0.3. OFF arms and any pid-teacher run
    do not. "Carry" means a hotter, DOB-distorted mpcof trainer — NOT a droop; the
    anchor's 0.346 m altitude is not explained by this.
    Separate, noted: the STUDENT's anchor is the true-mass hover in train and score
    (dagger_train.rs:1287, stage1.rs:107-111, cpu_score.rs:193) while the rivals get
    nominal-mass PD (:2819-2827) — a WNN-favouring asymmetry on the altitude column.
    Add to the disclosure list.

0.5 WHY NO TEST CAUGHT IT: Teacher never exposes its hover; no test compares the
    bank's teacher to the scorer's; the six label pins (controller.rs:9234-9330) use a
    constant-label fixture at neutral 0.5 and pin the label rule, not the teacher;
    every CPU/GPU parity sweep compares STUDENT rollouts, and the DAgger teacher runs
    CPU-only (dagger_train.rs:1240) with no GPU twin. The pin that would have caught a
    semantic mismatch: "for id 1..4 under a translation cfg, the bank's teacher at
    level / zero rates / alt_err 0 / vz 0 returns the label neutral on all motors" —
    passes today, would FAIL under the reviewer's fix. Add it regardless.

0.6 THE CORRECT FIX (if the effect is material): split hover into a LINEARIZATION
    point and a MIXING base. Add probe_hover to the four builds and pass it to
    calibrate_control_gains_rs while mix_to_motors_f64(self.hover = 0.5) stays;
    AirframeRs carries probe_hover = √(af_mass·g/4k) when cfg.translation else 0.5
    (dagger_train.rs:895-914, :940). CONTROLLER WHEEL ONLY (no ram_core, no worker
    swap). Off-translation: probe_hover = 0.5 → bit-identical, all pins and parity
    sweeps unchanged. Under translation: lqr/lqi labels move ≤0.1% (2% yaw); mpc/mpcof
    gain moves ~17% and the DOB scale becomes consistent — a LINEAGE BREAK for the
    anchor's mpcof trainer. Deploy only at an idle window (never while a chain is
    armed), then A/B at n=4 on the anchor seeds.

0.7 CHEAPER THAN A RE-FLY: the Python ctors take hover (optimal.rs:398, 580, 1109,
    1316) and AttitudeSim exposes set_translation / hover_pwm / set_vertical_state
    (controller.rs:998-1036), so a teacher_step_histogram-style rollout of
    AttitudeMpcOfRs(hover=0.5) vs (hover=0.694) on the same L4C episodes — the 0.5
    variant's output shifted by +0.194 to sit on the student's anchor — gives the
    per-step label difference and each variant's own err/steady/alt in MINUTES, no
    wheel, no chain risk. (Caveat: scripts/teacher_step_histogram.py:119 builds
    AttitudeSim() on the synthetic plant despite its docstring — pass the airframe.)
    Regenerating the banked winner's labels is NOT possible (markers hold cells only;
    rollout_and_label_rs is not exported). The STUDENT-side effect needs a re-fly;
    one smoke seed is not a measurement (n=4 MDE 0.61-0.69° err).

0.8 RECOMMENDATION (controller agent): RECORD-AND-PROCEED, THEN MEASURE BEFORE FIXING.
    Record §0.2/0.3 as the asymmetry (not "off-hover labels"). Run probe 0.7 first;
    if the mpcof label delta is material, implement 0.6 and A/B it. This BLOCKS AXIS B
    (the teacher-swap compares exactly the ids whose trainer and rival laws diverge:
    mpc, mpcof vs lqr, pid) but NOT axes A, C, D or F, and it does not invalidate the
    anchor's altitude finding.

0.9 TRAINING-ALGORITHMS TRACE (11/09) — agrees with 0.1/0.2 and adds three things.
    (a) The 0.194 offset is cancelled by the label's COORDINATE CONVENTION: the
        antagonist grid labels net = p − 0.5 (cell_mode.rs antagonist_target) and the
        student decodes around n = 0.5 on top of leaked_baseline = its TRUE-hover
        anchor (controller.rs:90-102, :5573-5583, :5633-5640). So the label reads
        "teacher pwm − 0.5 = student deviation from its own anchor", and the 0.5
        teacher anchor is the ONE thing that makes that correct. Naive fix → +12
        E-levels per level step → accumulator clamps at 1.0 → PD swings to −0.25 →
        limit cycle between full throttle and anchor − 0.22. Confirms 0.1.
    (b) K-sensitivity: k1 = √(q_att/r) is b-free and k2 moves ~0.06% on cf21 (b ≈
        2e3 rad/s² per u, so 2√(qr)/b ≪ q_rate) — LQR/LQI nearly inert, and this
        trace reads the MPC QP as b-insensitive too (cost structure), whereas the
        controller trace computed +17% first-move gain for MPC/MPCOF from the
        condensed QP. The two traces DISAGREE on MPC's gain sensitivity; the probe
        in 0.7/0.10 measures the actual label-magnitude ratio and settles it. Both
        agree the live channel is the mpcof OBSERVER (0.3): d̂ absorbs 0.28·b·u_student
        and u_ff subtracts ~0.39·u_student from every label, cutting the closed-loop
        DC attitude gain from 4 to ~1.56 below ~8 Hz — a TRANSIENT (err) effect, not
        a hold-floor or altitude one. Magnitude ≈ 0.016 pwm on the trim motors, the
        same order as the hold-window label and the 1/64 dead zone.
    (c) TWO CONSEQUENCES THE SPEC HAD BACKWARDS, both urgent:
        · PID AS A TRAINING TEACHER UNDER --translation IS BROKEN TODAY. PidFw's
          mixer sits at the true hover (pid_firmware.rs:388-399, pinned), so through
          path (a) it emits a permanent +12-level label → saturation. No pid-teacher
          translation run has ever banked. Axis B's pid arm CANNOT FLY until the
          label is re-based; it is not "confounded", it is impossible.
        · ARM B (--dagger-label-delta --obs-pwm) AS QUEUED IS VOID by the same
          mechanism in the other direction: the delta label is pid_pwms − label_base
          with label_base = leaked_baseline ≈ 0.694 (controller.rs:5538-5563), while
          the mpcof teacher emits ~0.5 at level → label −0.194 → clamped to −dmax →
          every motor labelled "max descend" on every level step. The pin
          delta_label_mode_labels_against_the_recorded_baseline uses anchor 0.5 and
          cannot catch it. ACTION TAKEN 11/09 10:30 EDT: arm B is HELD in
          scripts/post_arma_queue.sh behind the sentinel
          experiments/labelscale_markers/LABEL_REBASE_LANDED.json (queue killed in
          its pure-wait phase and relaunched; the 2x2 and window-k are unaffected).
    (d) The correct fix is TWO coupled changes landed together, default-off,
        bit-identical at s=1 off-translation: teacher probe/anchor at nominal hover
        AND a label re-base (label = neutral + (p − hover_teacher)) for ALL teachers
        including PidFw and the delta-label path; new pins at anchor 0.694
        (mis-anchored mpcof → neutral, fixed mpcof → neutral, PidFw → neutral, arm B
        delta label → 0 at level). Controller wheel only.
    (e) Probe (minutes, no wheel): solo oracle-fed rollouts as in
        scripts/teacher_step_histogram.py:114-147 (pass the airframe — it currently
        builds the synthetic plant), 20 episodes x 2000 steps, cf21/L4C/tilt 5°, same
        IC + weather seeds, teachers {mpcof, lqr} x anchors {0.5, 0.6942}. Report per
        motor at L=64: teacher-relative label floor64(p − h_teacher) histogram
        (dead-zone share, mean |levels|, saturation at s ∈ {1,2,4,8}); the raw
        floor64(p − 0.5) histogram at 0.6942 (exposes the +12-level offset); and the
        mean |u_cmd| ratio between anchors per teacher. MATERIAL = mpcof ratio
        outside [0.8, 1.25] or motor-1/3 hold dead-zone share moving > 15 pp
        (calibration: arm A's s=2 DOUBLED every label deviation and moved no column
        at n=4). lqr ≈ 1.0 confirms (b).
    (f) A/B if fixed: control = the (extended) anchor; arm = same recipe + flag; paired
        seeds; MEMORY row. Expected for mpcof: offset → 0, K → ~0, observer term gone
        → DC gain 1.56 → 4, and both banked gain arms (leak 0.90, L3) lost 4/4 when
        G fell, so ERR should improve, steady move little, stable ≈ 0, and ALT must be
        0 ± 0.08 m (collective untouched — alt is the bug detector). PRIMARY = ERR.
    (g) Q2 verdict: at level-and-on-altitude the student is taught the CORRECT hover
        as a neutral label; the hold floor and the altitude column are untouched by
        D0. The WNN's 0.35 m altitude gap is collective-jitter anchor error corrected
        at gain 4 (≈0.12 m at ±10%) plus the 0.11 m dead-zone bound — not D0.

0.10 PROBE RESULT (11/09 14:15 EDT, scripts/hover_anchor_probe.py, output in
     docs/d0_hover_anchor_probe.txt; 20 episodes x 2000 steps, cf21/L4C/tilt 5°, L=64,
     same episodes across variants; h_nom = √(m·g/4k) = 0.6942 from the airframe):
        teacher  variant  hover   dead_m13  mean|dev|  sat_s8  tilt°     today/fixed
        mpcof    today    0.5000  0.0774    0.0346     0.101   0.496
        mpcof    fixed    0.6942  0.0711    0.0342     0.088   0.496     1.010, +0.6 pp
        lqr      today    0.5000  0.0853    0.0363     0.135   1.015
        lqr      fixed    0.6942  0.0852    0.0362     0.134   1.015     1.002, +0.0 pp
     ("today" = teacher at 0.5 with its output applied on the student's true-hover
     anchor and observe() fed the applied pwm, i.e. what DAgger does; "fixed" = teacher
     at h_nom.) VERDICT: IMMATERIAL at the pre-registered threshold (ratio outside
     [0.8, 1.25] or dead-zone shift > 15 pp). Neither the disputed +17% MPC gain nor the
     0.39·u observer term shows up in the label the trainer would build; the two
     traces' first-order claim (zero label offset by coordinate convention) holds.
     Caveat carried from the training trace: this is SOLO flight (u_cmd = u_mpc/1.39
     partially self-cancels); in the DAgger loop the student applies the label at
     gain 4, so the observer term could be larger there — the A/B is still the check.
     CONSEQUENCE FOR THE PLAN: the fix is REQUIRED for correctness (derived hover, the
     broken PID trainer, arm B's void delta label) but its expected effect on the
     anchor is ≈ 0. So (i) the A/B is an equivalence check, err primary, alt
     no-regression; (ii) once it passes, the derived hover becomes the DEFAULT for
     every future run, and the banked s=1 controls stay valid comparators to first
     order (state the caveat on every 2x2 / arm B report).

DECISION D0 (Luiz), restated after both traces AND the probe:
  (a) run probe 0.9(e) now (minutes, read-only) and decide on its number;
  (b) implement the coupled fix 0.9(d) + A/B 0.9(f) regardless — it is REQUIRED
      anyway before arm B or any pid-teacher translation run can exist;
  (c) record and proceed with A, C, D, F only.
Probe done (0.10): immaterial. Luiz (11/09): Priority 0 — land the fix before ANY
next run, derive the hover, use it going forward. IN PROGRESS: rust-code agent on
branch hover-anchor-derived (worktree), default-off switch --teacher-hover
{legacy,derived}, pins at 0.694, controller wheel only. Deploy at the current idle
window, smoke ONE, then A/B (4 anchor seeds, derived vs banked legacy) as the FIRST
thing in the queue; then the 2x2 / window-k with derived ON; arm B once the delta
label is re-based (same change).

### 0.10 D0 A/B VERDICT — 4/4 banked 12/09/2026 02:10 EDT: EQUIVALENT, derived stays DEFAULT

Paired vs the banked legacy anchors (s2 control = _crn), MEMORY multi-seed row, n=4,
exact paired t, ONE primary column (docs/controller_d0_ab_verdict.txt):
    column   mean delta (derived − legacy)   95% CI            verdict
    err      −0.008°                         [−0.300, +0.285]  straddles — PRIMARY: indistinguishable
    alt      +0.004 m                        [−0.225, +0.232]  straddles — no-regression check PASSES
    steady   −0.075°                         [−0.593, +0.443]  straddles (descriptive)
    stable   −0.25 pp                        [−1.63, +1.13]    straddles (descriptive)
    failures/500 (MEMORY): derived 3/3/5/5 vs legacy 2/5/12/2 — Fisher p 0.73/0.14/0.45 per pair
MDE at n=4: err 0.39°, alt 0.31 m (the observed paired SDs 0.18° / 0.14 m are lower on
err and higher on alt than the arm-A priors). Bound on the effect: |Δerr| < 0.3°,
|Δalt| < 0.23 m. Both CIs straddle zero ⇒ per the pre-registered read, derived teacher
hover is CONFIRMED as the default (flipped a88cb7c1); no ALT regression, so the label
re-base is not a bug. Per-run: s2 +0.24°/+0.03 m, s3 −0.01°/−0.06 m, s4 −0.20°/−0.15 m,
s5 −0.06°/+0.19 m — the s5 alt swing (+0.19 m, headline CONNECTIONS#0 at 0.538 m) is a
single-seed draw and the reason the alt CI is 0.46 m wide.
CAVEAT (D8, §0b): every run in this A/B, both arms, trains the vertical channel through
the stale-vert_obs replay; the A/B is internally consistent (same trainer both arms) but
its alt column is not yet a controller property.

## 0b. D8 — the CPU replay trainer feeds STALE vertical features (audit finding, 11/09)

STATUS: FOUND BY CODE READING, NOT YET MEASURED. Needs a Rust probe before any decision.

Mechanism (file:line, tree 3b5fc37d):
  · At rollout, `rollout_and_label_rs` sets the vertical observation EVERY step
    (dagger_train.rs:1618-1630: `set_vertical_obs(collective, target_alt − z, vz)`),
    so the student's live addresses carry per-step [collective, alt_err, vz] bits.
  · The trajectory it records has NO vertical fields: `TrajectoryRs`
    (dagger_train.rs:586-613) = gyros, accels, targets, pid/student pwms, integrals,
    att_errs, label_base. Nothing about z.
  · Training replays that record: `bptt_train_window` (controller.rs:3764-3800) takes
    gyros/accels/targets/pwms only, and rebuilds each step's frame with
    `compute_features(gyros[t], accels[t], targets[t])` (controller.rs:3892), which
    appends `self.vert_obs` (controller.rs:5451-5464) — a field NOTHING in the replay
    updates. `reset()` (controller.rs:949) does not clear it either.
  ⇒ Every record of a training round is written at addresses whose three vertical
    features equal the LAST rollout step of the LAST episode of that round (near
    hover), while scoring/deploy reads addresses with the real per-step values. This
    is the DOB frozen-accumulator bug (Fix A, 06/08) in the vertical channel — the
    13/08 stage-1 review saw exactly this hazard for the GPU recorder and refused
    that path (dagger_train.rs:2231-2237: "would append the vertical features as
    ZEROS while the CPU/score path appends the real values") but called the CPU path
    safe without checking it. The CPU split trainer (controller_split.rs) has no
    vertical handling either (grep clean).

Scope: every run with --obs-collective-cmd/--obs-alt-err/--obs-vz since 12/08/2026
(d56576d7) — the anchor recipe sets all three (sweep_ladder_gamma.sh:62-63) — at
sn=0 AND sn>0. That is the four anchors, arm A, the D0 A/B, the leak/label arms.

Why it can still fly at 99% stable: the stale value is near hover, and so are most
steps, so the thermometer bins agree on the majority of records; they DISAGREE on
the transient steps after a disturbance — exactly where altitude error is made.
With ~20 features x 16 bits and 24-bit neurons essentially every neuron addresses
at least one vertical bit, so the effect is on address identity, not on a minority
of neurons. This is a candidate mechanism for the 0.27 m altitude gap that no
attitude lever has moved, and for "the floor is STRUCTURAL" (L1b).

What must happen BEFORE axis C (and before treating the anchor's alt as a controller
property):
  1. MEASURE (Rust, read-only): a probe that, at scoring time, counts the fraction of
     visited addresses that are EMPTY, vertical features on vs off, on one banked
     anchor winner. Stale-write divergence shows as a high EMPTY fraction on the
     transient steps. No wheel deploy needed for a probe binary; the run's cells are
     in the banked winner.yaml.gz.
  2. If confirmed, FIX by the Fix-A pattern: record [collective, alt_err, vz] (and
     the stage-2 horizontal quad) per step in TrajectoryRs at rollout, slice them
     into `bptt_train_window` like `student_pwms`, and call `set_vertical_obs` /
     `set_horizontal_obs` before `compute_features(t)` at controller.rs:3892. Pin
     test: replay frame bits == rollout frame bits for the vertical features on a
     synthetic trajectory. Ship behind a flag, default legacy bit-identical (D0
     precedent), A/B on the 4 anchor seeds, then flip. Wheel change ⇒ R9: anchor
     re-fly or bit-identity, and the ABI bumps.
  3. Until then, axis C's primary column (ALTITUDE) is measured against an anchor
     whose altitude channel may be trained wrong — do not arm C.

## 1. The claim under test

Every controller result to date sits at ONE point of the design space:

    airframe cf21_brushless · disturbance L4C · teacher mpcof · L=64 levels/motor
    · sn=0 (single layer) · altitude regimen ON · b24 n256 · 4 seeds 31337002..5

A reviewer's first question is whether the ranking measured there survives a move
along any axis. The programme answers it one axis at a time from the anchor, with
the WNN's hyperparameters FROZEN (the thermometer encoder is re-fit per condition
from that condition's PID rollouts — evaluator.py:463-490 — and that is stated, not
hidden), LQR/LQI/MPC/MPCOF re-deriving their gains automatically, and PID flying the
sourced firmware gains bound to each airframe.

THE ANCHOR, correctly stated as a 4-seed mean±SD on the MEMORY multi-seed row
(draft 1 printed seed 31337003's row and called it the mean — corrected):

    controller    stable        err          steady       alt          hd
    MPCOF         100.0 ± 0.0   0.70 ± 0.01  0.01 ± 0.00  0.076 ± 0.04 0.0483  the teacher
    LQI           100.0 ± 0.0   0.89 ± 0.06  0.45 ± 0.06  0.076 ± 0.04 0.0621
    LQR           100.0 ± 0.0   1.05 ± 0.08  0.59 ± 0.11  0.076 ± 0.04 0.0726
    MPC           100.0 ± 0.0   1.38 ± 0.17  1.04 ± 0.22  0.076 ± 0.04 0.0958
    PID           100.0 ± 0.0   1.79 ± 0.36  1.03 ± 0.36  0.076 ± 0.04 0.1241  weakest classical
    WNN anchor    98.95 ± 0.94  1.63 ± 0.20  1.17 ± 0.25  0.346 ± 0.035 0.127 ± 0.025  n=4
    (classical SDs are over the 5 report seeds x 5 folds; classical alt is the outer
     loop's, not the controller's; lower hd = better)

What that table supports and what it does not:
- The WNN's hd 95% CI at n=4 is [0.086, 0.167]. It contains MPC and PID alike. So the
  anchor is NOT "level with PID" in any testable sense; it is "somewhere between LQR
  and worse-than-PID". Draft 1's "its best seeds beat PID" was best-of-archive and
  is withdrawn.
- LQR vs LQI is a real order (SDs 0.06-0.08 on a 0.15 gap). PID vs WNN is not.
- The WNN's altitude gap to every classical (0.35 vs 0.08 m) IS resolved: SD 0.035
  on a 0.27 m gap. That is the one anchor fact with statistical content.

THE HEADLINE QUESTION, narrowed to what OFAT can answer:
    Q1. With the recipe frozen, is the anchor's position LOCALLY robust to a
        one-axis move — on err (the attitude primary) and on altitude (the
        resolved primary) — and does the 0.27 m altitude gap persist?
hd is reported everywhere but is DESCRIPTIVE: it composes the best-resolved (err)
and worst-resolved (stable) attitude columns, and it is absent from the verdict tool.

## 2. Rules (R1-R11)

R1. VERDICT = the mean delta and its 95% CI on ONE pre-registered PRIMARY column
    per axis (§3). The other three columns are descriptive. A Holm-4 CI is printed
    as a robustness line (paired_power.py --primary). Win/loss tallies are never a
    verdict ("k of n" is a sign test: 3/4 fires 31% under the null).
R2. SIZE BEFORE FLYING, with the exact paired-t MDE at n=4 from the anchor arm's SDs
    (scripts/paired_power.py, docs/controller_paired_power.txt):
        err 0.61-0.69°   steady 0.69-1.00°   stable 1.9-2.0 pp   alt 0.08-0.16 m
    and a 4-pair SD is itself uncertain (95% CI [0.57x, 3.7x]). Quote RANGES. Every
    axis names its minimum effect of interest (MEI) against these. A null on a
    column whose MDE exceeds the MEI is INDETERMINATE, never a refutation.
R3. Altitude is the well-resolved channel (SD 0.04-0.08 m). It is the primary for
    any axis whose "read" is about the WNN's altitude gap.
R4. Same four seeds (31337002..5) and the same report seeds everywhere, for
    bookkeeping. BUT the pairing is cosmetic: pair correlation on err/steady/alt is
    ≈0 (paired-delta SD ≈ √2 x the anchor's between-seed SD), because the search seed
    is a random draw, not a block. The PRIMARY analysis is therefore Welch two-sample
    against an EXTENDED anchor (§4), with the paired analysis as robustness.
R5. Train ORACLE, compare ESTIMATOR-FED: every classical row is the [est] row.
R6. Every condition gets its OWN baseline file (compute_baselines.py --airframe
    --disturbance --translation) BEFORE its WNN runs, and baselines are recomputed
    on any new report-seed set (R10). Never compare across conditions' baselines.
R7. Surface = MEMORY multi-seed row (stage-matched; the headline is a val draw that
    moved 0.12 m alt / 0.022 hd on one anchor seed). Every table ALSO shows the
    headline row with its crowned stage named — that is the deployed genome and
    the one the H743 key count is measured on. Axis C additionally reports the
    CONNECTIONS row, because MEMORY can regress on CONNECTIONS and sn>0 adds a stage.
R8. Four columns on every surface. Altitude is never omitted. Classical rows carry
    their SDs (§1) so V3's "order" says which gaps are real.
R9. PROVENANCE. Every marker records the wheel hash, ABI and fitness_pools. The four
    anchor markers come from three sweeps (04-07/09) and all PREDATE the 08/09 22:20
    label-fix wheel; the fix was pinned bit-identical at s=1 in arm A's smoke, and
    that pin is the anchor's validity argument — cite it. Any wheel change during the
    programme forces either an anchor re-fly or a demonstrated bit-identity.
R10. REPORT SEEDS ARE A VALIDATION SET. Every ladder decision so far was read on
    99990101..05; stage-select avoids them but the recipe was chosen on them. The
    programme's FINAL table uses a fresh never-used report-seed set (D5), with the
    classical baselines recomputed on it (R6). Interim ticks may use the old set.
R11. stable% is a bounded per-episode pass rate; a t-CI on it is approximate and its
    paired SD swings 0.4-3.4 pp between arms. Report failures as counts with an exact
    binomial CI where the episode counts are exported; otherwise stable is
    descriptive only (hd already folds it in).

## 3. The axes — values, primary column, MEI, power, confounds

### Axis A — DISTURBANCE  (L4A, L4B vs anchor L4C)          8 runs, ~40 h
Values: the sourced L4 rung only (training.py `_L4_LEVELS`): L4A clean plant, L4B
10%, L4C 20% (Molchanov's ceiling). L1-L3 and the D variants are unsourced; not used.
Primary: ERR (V1, WNN absolute). MEI 0.6° = the n=4 MDE; the plant-jitter step is
expected to exceed it comfortably.
Power caveat: Q1's V2 (gap to PID) is UNSIGNED here — PID's own 1.79±0.36° is
jitter-driven and will move with the rung. State V1 as the resolvable quantity and
V2 as the read, not the test.
What L4A actually removes (physics review): ONLY the per-episode static plant draw —
torque_scale_jitter and motor_asym_mag (training.py:231-235). Gyro/accel sigma and
bias walk are the same ADIS16448 model on every rung (training.py:236-239), so the
sensors stay noisy and the old "noise = dither" finding (which was disturbance-OFF,
sensors clean) does NOT apply. The plausible "L4A is harder" mechanism is instead
TEACHER-LABEL COLLAPSE: with no static torque offset, mpcof's d̂ ≈ 0 and more labels
fall into the 1/16 dead zone. Name that, not dither.
Also: under --translation the VERTICAL plant stays randomised at every rung
(--mass-jitter 0.15, --collective-jitter 0.1, --alt-offset 0.3, --init-vz 0.2 are
regimen defaults, phased_ga.py:2338-2347, drawn per episode independent of the rung).
So L4A cleans the ATTITUDE plant only; Read (b) below is about attitude noise.
Two protocols, both flown, named separately:
  A-same  train and evaluate at the same rung (as written): "does the student need
          plant randomisation to learn?"  — the 8 runs.
  A-cross SCORING ONLY: the banked anchor winners (trained L4C) scored at L4A/L4B —
          "how robust is one controller to plant uncertainty?" This is Molchanov's
          own protocol (train randomised → deploy nominal), costs minutes, and the
          classical rows are identical either way. Free; add it.
Prereqs: baselines at L4A and L4B with --translation (minutes each).

### Axis B — TEACHER  (pid, lqi vs anchor mpcof)             8 runs, ~40 h
Values: pid and LQR (changed from lqi on physics review). LQR is MEMORYLESS
(optimal.rs:419); lqi carries an integrator, mpcof a d̂ observer, the PID cascade two
integrators plus an LPF. With pid + lqi both stateful, a null could not separate
"student floor" from "hidden teacher state makes labels non-functions of the
observation". pid + lqr keeps the quality span (1.79 / 1.05 / 0.70°) AND adds the
memoryless control. mpc stays excluded, but for its own physics (no integral/
observer, so the L4C torque offset is a steady 1.04° it cannot absorb; the +27%
under translation is that, not a regimen artefact).
BLOCKED BY §0 — and harder than draft 3 said: the pid TRAINER under --translation
saturates the label today (§0.9(c)); axis B cannot fly at all until the label re-base
(§0.9(d)) lands. Then pid vs lqr is a clean swap.
Primary: ERR. What n=4 resolves: FULL tracking (student err moves by ~the 1.1°
teacher gap) versus NONE. PARTIAL tracking (0.3-0.6°) is indeterminate at n=4.
The null is NOT free: claiming "the student ignores its teacher" is an equivalence
claim and needs a pre-registered margin. Margin = 0.55° (half the teacher gap),
TOST at 90%; decidable only if |d| < 0.2° AND the observed SD ≤ 0.3°. State this.
Prior: the live-label work found the steady floor teacher-independent (0.57-0.87°
for every teacher), so the null is the prior on steady; err is the open column.
Confound: the prior teacher screen was attitude-only AND through the ~6x mis-scaled
thermometer — none of its numbers carry over. This is the first under the regimen.
Prereqs: none new (TeacherBank fix is in since 05/08).

### Axis C — STATE NEURONS  (sn=4, sn=8 vs anchor sn=0)      8 runs, ~50-60 h
Values: --grid-state-neurons N --max-state-neurons N. The only lever family that
ever reached sub-0.06 hd (attitude-only S1_lqi_sn8 0.0535, sn4 0.0618, Aug 2026,
rotation-era scorer), and it has ZERO rows under the altitude regimen.
Primary: ALTITUDE. MEI = 0.16 m of the 0.27 m gap (the n=4 MDE at the pessimistic
SD 0.077; 0.08 m at SD 0.039). A state layer that closes less than that is not
distinguishable from nothing at n=4 — say so up front.
Also report the CONNECTIONS row (R7): sn>0 adds a stage, so a MEMORY-row loss may
be a pipeline artefact, not a controller one.
Confound: sn>0 changes the search (an extra stage); a win is "recurrent state + its
search", which is how it would ship. Acceptable.
Path (physics review, resolved): the ladder sets no WNN_* env, so `use_split` is
false and sn>0 trains through the EDRA-BPTT window trainer ON CPU
(controller.rs:3731-3747); the GPU split path refuses the vertical channel anyway
(dagger_train.rs:1988-2001). Several anchor-era flags are sn=0-ONLY: write-priority
/ err-floor (dagger_train.rs:266-275), --dagger-label-delta (evaluator.py:147),
output_full_window (controller.rs:1817-1820). So if arm B or a window arm becomes the
anchor before axis C flies, C cannot inherit it; a C win reads "recurrent state + a
different trainer + its search". Observability: no trap — the vertical channel is
fully observed and the state-prefix offset is derived from num_features(); do not
expect a state layer to close the altitude gap by observability, only by smoothing
the collective channel.
Prereqs: (1) a 4-minute smoke at sn=4 with --translation on the CPU BPTT path;
(2) memory budget vs the 180k-cell watchdog cap, stated before arming; (3) wall-clock
estimate for the CPU trainer (budget 6-7 h per run, may be more).

### Axis D — AIRFRAME  (cf2x_firmware vs cf21_brushless; cf2x_urdf deferred)  4 runs, ~20 h
Values: cf2x_firmware only. VERIFIED 11/09 by constructing the controller: the
firmware PID cascade builds on cf2x_firmware with ITS OWN sourced gains
(platform_defaults_cf2.h). cf2x_urdf: the ValueError I hit fires only under
--calib-airframe; on the NORMAL path `_pid_cascade_kwargs` returns {} for rate=None
(training.py:513-514) and trainer AND baseline scorer fall back to the legacy
retired-plant loop SILENTLY (dagger_train.rs:930). So cf2x_urdf is not refused, it
is quietly wrong — worse. Stage 0 adds a hard refusal there. Because the WNN's thermometer is fit from PID rollouts
(evaluator.py:463), a fallback PID would contaminate the WNN's ENCODER, not just the
comparator — so cf2x_urdf cannot be flown at all until a citable DSL single-loop PID
is ported (Python + Rust + Metal, parity). Draft 1's "re-derive gains" is WITHDRAWN:
airframe.py records Luiz's 05/08 rule that invariant-preserving derivation
"manufactures an UNSOURCED number". docs/disturbance_param_sources.md also forbids
blending the URDF plant with firmware gains (different Crazyflie builds).
Primary: ALTITUDE (the transfer question is whether the recipe's altitude hold
survives a different thrust/inertia map); err descriptive.
Reads to state (physics review): cf2x_firmware's k_drag is 5x cf2x_urdf's
(airframe.py:166,233) — yaw authority per pwm — on the one axis the student cannot
observe (yaw is dead-reckoned), so expect the yaw-dither cost to scale with airframe;
cf21_brushless's inertia is DERIVED ("treat as an assumption", airframe.py:183-217)
while cf2x_firmware's is MEASURED, so axis D is also the inertia-sensitivity check a
paper is asked for. A self-deriving comparator, if ever needed, should be LQI
(integral), not MPC — moot under D1.
Power caveat: the cf21 SDs give NO basis for the SD at another airframe. D's power
statement is a guess until its first two seeds land (§4 round rule).
Prereqs: baseline at (cf2x_firmware, L4C, --translation); one smoke.

### Axis F — ACTUATOR LAG  (τ = 0.0375 s nominal, sweepable; anchor = 0)   8 runs, ~40 h  [NEW]
Found by the physics review as the axis a controls reviewer asks about FIRST: motor
settling (Molchanov eq. 7, τ = T/4 = 0.0375 s) is SOURCED, IMPLEMENTED in the sim
(controller.rs:381-399, 2290-2297) with a Metal twin and a parity assertion, defaults
to 0.0, and is reachable from NO recipe: nothing under src/wnn or scripts references
it, RewardGatedConfigPacked has no field, and score_classical_baseline does not take
it. The anchor is therefore a LAG-FREE 1 kHz high-gain loop — the most attackable
modelling choice in the whole programme.
Values: τ = 0.0375 s (nominal, sourced) and 2τ (stress) vs 0. Primary: ERR (lag
degrades every controller; the read is whether the WNN degrades MORE than the
classicals, i.e. V2). Expected effect: large — resolvable.
Prereqs: PLUMBING (Python only, swap-free): phased_ga flag → EpisodeConfig → packed
cfg + the baseline scorer; per-condition baselines; one smoke. Goes AHEAD of axis E
and, on reviewer priority, ahead of axis D.

### Axis G — CONTROL RATE  (action_repeat 2 = 500 Hz vs anchor 1 kHz)   optional, 4 runs
The WNN acts every 1 ms step (action_repeat=1) while the firmware cascade it is
compared to runs at 500 Hz with hold; action_repeat already reaches the Metal scorer.
Cheap, plumbed, and it is the H743 deployment question. Consider after F.

DISCLOSURE (not an axis): the sim's accelerometer is the hover approximation —
specific force = −gravity_body only (controller.rs:598-606). With translation on, a
real IMU on a thrust-only quad carries no tilt information without a drag model; the
sim's accel carries tilt at all times. This flatters the WNN's accel-derived features
and the Mahony rival EQUALLY, so it is not a confound between rows, but the paper
must state it. Wind: no sourced attitude-torque path exists and all L4 rungs are
windless by design (disturbance_param_sources.md:239-251, 410-470) — cite, do not
add. Inertia/mass mismatch on the attitude plant IS torque_scale_jitter, so axis A
(±20%) plus axis D (measured vs derived inertia) already cover it — cite.

### Axis E — LEVELS  (L=32, L=128 vs anchor L=64)            NOT FLOWN
Prior: the alphabet probe and levels ablation were refuted at their bars, and the
bits ladder found no width separates at n≤5; expected attitude effect is below the
n=4 MDE on every column. Flying it would manufacture an indeterminate null (R2).
One caveat for flight-dynamics (§9): the live-label dead-zone mechanism predicts L
moves ALTITUDE (label grid ∝ 1/L), which would be a large-effect prior on the
resolved channel. If flight-dynamics endorses that, E re-opens with alt as primary.

## 4. Design: rounds, anchor extension, budget

INTERLEAVE, DON'T STAGE (the standing sweep rule, and R2's own logic: a condition's
SD is known after its 2nd seed, which is when "the n it would need" is actionable):
    round 1  one seed each of  L4A, L4B  (A has no prereqs), plus A-cross scoring
             + pid, lqr once D0 (§0) is decided; + τ-lag once its plumbing lands;
             + cf2x_firmware and sn=4/sn=8 as their prereqs land
    round 2  second seed of every condition  → per-condition SD → re-size
    round 3-4  third and fourth seeds
    escalation rule (pre-registered): if a condition's primary CI straddles zero
             AND includes the MEI after 4 seeds, add 2 seeds to that condition.
ANCHOR EXTENSION: because the pairing is cosmetic (R4), the efficient spend is more
ANCHOR seeds shared by all conditions. Fly 31337006..9 at the anchor (4 runs, ~20 h)
on the programme's wheel; primary analysis Welch (anchor n=8 vs condition n=4, SE
≈0.61 SD) with paired-4 as robustness (SE 0.79 SD). This beats "5 seeds everywhere"
(+9 runs for SE 0.63 SD) and also gives the anchor its own R9-clean re-fly.
BUDGET
    anchor extension            4 runs   ~20 h   (round 1, first)
    A disturbance (+ cross-rung scoring, minutes)   8   ~40
    B teacher (pid, lqr) — blocked on D0            8   ~40
    F actuator lag [NEW] — needs plumbing           8   ~40
    C state neurons (CPU BPTT path)                 8   ~55+
    D airframe (cf2x_firmware)                      4   ~20
    G control rate (optional)                       4   ~20
    E levels                                        0   —
    total (A,B,F,C,D + anchor)                     40  ~215 h ≈ 9 days, plus escalation
Every chain: marker-gated, idempotent, fails closed, one controller at a time, never
edits a running .sh. Queues behind the post-arm-A queue (~90 h).

## 5. Stage 0 — prerequisites, no controller runs

  [ ] Marker provenance fields (wheel hash, ABI, fitness_pools) exported by the
      ladder — R9. Check whether they already exist; add if not (Python only).
  [ ] Decision D5: fresh report-seed set for the final table — R10.
  [x] Baselines --translation for (cf21, L4A), (cf21, L4B), (cf2x_firmware, L4C) —
      DONE 11/09/2026 18:21 EDT, same args as the anchor file (5 report seeds
      99990101-05, 100 ep x 2000 steps, tilt 5, sim_seed 911, fold 0):
      experiments/l4teach_markers/baselines_{L4A,L4B}_cf21bl_translation.json and
      baselines_L4C_cf2xfw_translation.json (log logs/controller/stage0_baselines.log).
      [est] rows, stable 100.0% everywhere, err / steady / alt / hd:
        L4A cf21:   PID 0.58/0.18/0.000/0.040  LQI 0.68/0.25/0.001/0.048  MPCOF 0.70/0.01/0.001/0.048
        L4B cf21:   PID 1.08/0.54/0.038/0.075  LQI 0.76/0.31/0.038/0.053  MPCOF 0.70/0.01/0.038/0.048
        L4C cf2xfw: PID 1.39/0.80/0.076/0.096  LQI 0.88/0.43/0.099/0.061  MPCOF 0.69/0.01/0.076/0.048
        (anchor L4C cf21: PID 1.79/1.03/0.076/0.124  LQI 0.89/0.44/0.076/0.062  MPCOF 0.70/0.01/0.076/0.048)
      Reads: MPCOF's 0.70 deg / 0.01 deg floor is IDENTICAL across all four conditions,
      so V2 (gap to the teacher) is disturbance- and airframe-invariant and any WNN
      movement along axis A/D is the student's, not the teacher's. PID's err is the
      most disturbance-sensitive row (0.58 -> 1.08 -> 1.79 deg across L4A/L4B/L4C) and
      is 0.40 deg lower on cf2x_firmware than on cf21 at L4C — the firmware gains fly
      their own airframe better, so axis D's PID gap is NOT comparable to axis A's
      (R6). alt m is set by the disturbance, not the controller (0.000/0.038/0.076
      for L4A/L4B/L4C on cf21). PENDING: rerun all four on the D5 seed set once chosen.
  [ ] Failure-count export for stable (R11): does the marker carry per-episode
      counts? If not, add them or declare stable descriptive.
  [x] sn>0 path audit (11/09/2026, current tree 3b5fc37d). Every claim of the §3-C
      path paragraph re-verified at today's lines, plus the D0 question the review
      predates:
        · Trainer dispatch: `use_split = WNN_STATE_SPLIT=="1" && sn>0`
          (dagger_train.rs:2223-2226); the ladder exports no WNN_* (grep clean), so
          sn>0 takes the non-split branch (dagger_train.rs:2341-2352) →
          `train_on_trajectory_rs` (1759) → `bptt_train_window` (controller.rs:3764-4417),
          CPU. GPU split needs WNN_CONTROLLER_GPU_TRAIN=1 AND refuses the vertical
          channel (`!vert_on`, dagger_train.rs:2238-2247).
        · sn=0-ONLY: write-priority / err-floor gated by `state_bits_in == 0`
          (controller.rs:3981-3987); --dagger-label-delta asserts sn==0
          (controller.rs:3832-3837; evaluator.py:151); output_full_window refuses sn>0
          (controller.rs:1821-1824). Confirmed: axis C cannot inherit arm B / arm D.
        · D0 REACHES sn>0: the derived hover enters through TeacherBank::get
          (dagger_train.rs:968-975), consumed by `rollout_and_label_rs` (1481), which
          both the split and the BPTT branch share — so a C arm under
          --teacher-hover derived trains against the same re-based labels as the anchor.
        · State-prefix offset from num_features(): controller.rs:8096 (pin test).
      ⚠️ NEW FINDING — D8, applies to sn=0 TOO (i.e. to the ANCHOR). See §0b below.
  [ ] Smokes: one 4-minute phased_ga per new flag combination (L4A, L4B, pid, lqi,
      sn=4, sn=8, cf2x_firmware) — rc 0 and a sane grid line.
  [ ] Memory budget for sn=4/8 vs the 180k cap.
  [ ] Round-major chain written (one script, conditions x seeds, marker-gated),
      ending each round in a paired_power.py --primary verdict per condition against
      the extended anchor AND the condition's own baseline.
  [ ] Power statement per axis written INTO the chain header (R2).
  [ ] D0 resolved (§0) before axis B or the anchor extension flies.
  [ ] Hard refusal in `_pid_cascade_kwargs` for an airframe whose registered gains
      have rate=None (training.py:513) so cf2x_urdf cannot be flown by accident.
      Python-only, inert on cf21 — but it is live-imported source: land it at an idle
      window, never while a chain is armed.
  [ ] Actuator-lag plumbing (axis F): flag → EpisodeConfig → cfg → baseline scorer.
  [ ] A-cross scoring script: score the 4 banked anchor winners at L4A/L4B.
  [ ] Stale notes fixed so draft 1's error cannot recur: `_FW_UNIT_NOTE`
      (airframe.py:304-308) still says the mapping "must be derived and TESTED";
      memory note project_pid_not_airframe_retuned.md still says "re-derive via
      derive_sim_pid_rp" (removed and rejected). The memory note is fixed 11/09; the
      code comment is a one-line edit for the next idle window.

## 6. Verdict protocol (pre-registered, per condition)

  V1. Δ(WNN_condition − WNN_anchor) on the PRIMARY column: Welch mean, 95% CI
      (anchor n=8 when the extension has landed, else paired n=4). The other three
      columns: same numbers, labelled descriptive. Holm-4 CI printed as robustness.
  V2. Δ(WNN − PID) at the condition vs at the anchor. PID is fixed on the report
      seeds, so V2 = V1 − a constant: it is the same test, reported as the READ.
  V3. The classical order at the condition from its own baseline file, WITH SDs, and
      where the WNN's hd falls in it — as a table, never a bare inequality.
  V4. Altitude always shown; hd always shown and always descriptive.
  A condition has MEASURED something if the primary CI excludes zero. An
  indeterminate primary is written up with the n it would need (the tool prints it)
  and, if the CI includes the MEI, triggers the §4 escalation rule.

## 7. What this programme is NOT

- Not a search for a better operating point: hyperparameters are frozen and moved.
  The encoder IS re-fit per condition (it is fit from that condition's PID rollouts);
  that is part of the recipe, not tuning, and it is stated.
- Not a factorial: 5 axes x 3 values x 4 seeds = 972 runs; this is 32. Interactions
  are out of scope, and the claim in §1 is LOCAL to the anchor for that reason.
- Not a replacement for the queued 2x2 / window-k. Arm B is HELD (§0.9(c)) until the
  label re-base lands; this programme queues behind whatever runs.

## 8. Open decisions for Luiz

  D1. Axis D: fly cf2x_firmware only now (recommended); cf2x_urdf waits for a citable
      DSL single-loop PID port, or is dropped.
  D2. Axis E: drop with justification, or re-open on altitude if flight-dynamics
      endorses the 1/L altitude prior.
  D3. Round-major interleaving (§4) in place of stage-major — confirm.
  D4. Anchor extension to 8 seeds (+4 runs, Welch primary) in place of "5 seeds
      everywhere" (+9 runs) — recommended.
  D5. Fresh report-seed set for the final table (R10): e.g. 99990201..05 — confirm
      the numbers, and whether interim ticks may keep using 99990101..05.
  D0. (§0) trainer-vs-rival linearization asymmetry (mpc/mpcof ~17% + DOB 1.39x):
      probe first (recommended), fix+A/B regardless, or record and proceed?
  D6. Add axis F (actuator lag) — recommended, and ahead of D on reviewer priority.
  D7. Axis G (control rate) — include as an optional tail?

## 9. Reviews

### 9.1 experiment-design review — received 11/09/2026, 19 findings; all adopted
Adopted into draft 2: anchor row corrected to the n=4 mean±SD (F1); "level with
PID" and best-of-archive withdrawn, hd CI stated (F2); pairing shown to be cosmetic,
Welch-vs-extended-anchor made primary (F3, F12); MEMORY row kept, headline row with
crowned stage always shown, CONNECTIONS row for axis C (F4); ONE primary column per
axis + Holm-4 robustness, V2 recognised as V1 − const (F5); exact noncentral-t
sizing and MDE in the tool, normal approximation removed (F6, F19); axis B null as
an equivalence claim with a 0.55° TOST margin (F7); axis A power restated on V1,
encoder re-fit made explicit in §1/§7 (F8); axis C MEI 0.16 m (F9); axis D deepened —
the PID-fed thermometer means a fallback PID contaminates the WNN encoder, which
strengthens deferring cf2x_urdf (F10); Q1 narrowed to a local claim on err and alt,
hd descriptive (F11); R9 provenance (F13); R10 fresh report seeds (F14); round-major
interleaving with a per-condition escalation rule (F15); R11 stable as counts (F16);
classical SDs carried (F17); axis E do-not-fly accepted with the 1/L altitude caveat
routed to flight-dynamics (F18).

### 9.2 flight-dynamics review — received 11/09/2026, 15 findings; all adopted or routed
Adopted into draft 3: §0 training-teacher hover-anchoring mismatch surfaced as a
BLOCKER and routed to Luiz/controller agents, not fixed (F1); axis D refusal claim
corrected — cf2x_urdf falls back silently on the normal path; hard refusal added to
Stage 0 (F2); option (ii) moot, LQI would be the right self-deriving comparator if
ever needed (F3); axis D reads: 5x k_drag on the unobservable yaw axis, derived vs
measured inertia (F4); axis A confound renamed from dither to teacher-label collapse,
sensors stay noisy on every rung (F5); vertical plant stays randomised at L4A (F6);
A-cross scoring protocol added, free (F7); axis C path named — CPU EDRA-BPTT, sn=0-only
flags listed (F8); axis C observability accepted, mechanism caveat kept (F9); axis B
second teacher changed lqi → lqr, mpc exclusion re-reasoned (F10); axis F actuator lag
added — sourced, implemented, reachable from no recipe, anchor is lag-free (F11);
accelerometer hover-approximation disclosure (F12); axis G control rate as optional
(F13); wind and inertia coverage cited (F14); the two stale notes that produced
draft 1's error listed in Stage 0, memory note fixed (F15).
