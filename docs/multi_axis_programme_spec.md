# Multi-axis programme — specification (draft 1, 11/09/2026)

Status: DESIGN, not armed. Written while the post-arm-A queue runs (2x2 on s=2 →
arm B → window-k). Review by the experiment-design and flight-dynamics agents is
part of this spec (§9) and must happen before anything is launched.

## 1. The claim under test

Every controller result to date sits at ONE point of the design space:

    airframe cf21_brushless · disturbance L4C · teacher mpcof · L=64 levels/motor
    · sn=0 (single layer) · altitude regimen ON · b24 n256 · 4 seeds 31337002..5

A reviewer's first question is whether the ranking measured there survives a move
along any axis. The programme answers it one axis at a time, from the anchor, with
every condition paired against its own classical baselines on the same report seeds.

The thing being tracked is NOT the WNN's absolute number. It is the WNN's position
in the classical ORDER at that condition, and its GAP to the weakest classical (PID),
on all four columns. At the anchor (translation regimen, estimator-fed, 5 report seeds,
from experiments/l4teach_markers/baselines_L4C_cf21bl_translation.json):

    controller   stable   err    steady  alt     hd        note
    MPCOF        100.0    0.70   0.01    0.076   0.0483    the teacher
    LQI          100.0    0.89   0.44    0.076   0.0621
    LQR          100.0    1.05   0.59    0.076   0.0726
    MPC          100.0    1.38   1.04    0.076   0.0958
    WNN (anchor) 99.0     1.65   1.35    0.351   0.1266    b24 n256 MEMORY-row mean, n=4
    PID          100.0    1.79   1.03    0.076   0.1241    weakest classical
    (lower hd = better; classical alt is the outer loop's, not the controller's)

The anchor WNN is level with PID on hd (its best seeds beat PID: 0.0988, 0.1001,
0.1029) and 0.03 behind MPC. The 01/09 note "every classical beats every WNN" is
STALE for hd; it remains true on stable (99% vs 100%) and on altitude (0.35 m vs
0.08 m). The programme's pre-registered headline question is therefore:

    Q1. Does the WNN stay level with PID on hd across conditions, and does its
        altitude gap stay ~0.3 m, or does either blow up off the anchor?

## 2. Rules inherited from the measurement fix (commit 4b2bb46a)

R1. Verdicts are the mean paired delta and its 95% CI on the MEMORY multi-seed row,
    from scripts/paired_power.py. Win/loss tallies are descriptive only.
R2. Size before flying. At the anchor's paired SD, n=4 resolves ~0.6° steady, ~0.6°
    err, ~1.3 pp stable, ~0.1 m altitude. Every axis below states what it can and
    cannot see at n=4. A null on an under-powered column is INDETERMINATE.
R3. Altitude is the well-resolved channel (SD 0.04-0.08 m). Any axis that moves
    altitude is decidable at n=4; most that only move steady are not.
R4. Same four seeds everywhere (31337002..5), same five report seeds, CRN scorer,
    so every condition is paired to the anchor AND to its own baselines.
R5. Train ORACLE, compare ESTIMATOR-FED: every classical row is the [est] row.
R6. Every condition gets its OWN baseline file before its WNN runs start
    (scripts/compute_baselines.py --airframe --disturbance --translation). Never
    compare a WNN at condition X to a classical measured at the anchor.
R7. Stage-matched comparison: MEMORY row vs MEMORY row. Headlines only as an aside.
R8. Report all four columns on every surface. Altitude is never omitted.

## 3. The axes

Order is by (expected effect size × paper value) / (cost + prerequisite risk).
Expected effects are stated so that the power check in R2 is explicit.

### Axis A — DISTURBANCE  (L4A, L4B vs anchor L4C)          8 runs, ~40 h

Values: the sourced L4 rung only (training.py `_L4_LEVELS`): L4A = clean plant,
L4B = 10% jitter, L4C = 20% (Molchanov's ceiling). L1-L3 and the D variants are
unsourced and are NOT used.
Prereqs: baselines at L4A and L4B with --translation (minutes each). Nothing else.
Expected effect: large. Going from 20% plant jitter to a clean plant should move
err by well over 0.6°, so this axis IS resolvable at n=4 on every column.
Confound to name: "noise = dither" (memory). A clean plant previously DEGRADED a
noisy-trained winner 1.1°→6.5°. So L4A is not "easier" for the student; it may be
harder, and that is itself a finding. Train and evaluate at the SAME rung (never
train L4C / evaluate L4A) — the cross-rung transfer is a separate question, not this.
Reads: (a) robustness curve WNN vs classicals across three plant-uncertainty rungs;
(b) whether the WNN's altitude gap to the classicals is a plant-noise artefact.

### Axis B — TEACHER  (pid, lqi vs anchor mpcof)             8 runs, ~40 h

Values: pid (weakest, transfers to translation at +0%) and lqi (integral, +10%).
mpc is excluded because translation costs it +27%, which would confound the teacher
swap with the regimen; lqr is skipped as redundant with lqi for a first pass.
Prereqs: the TeacherBank fix (05/08) is in — anything before it was PID-taught; all
runs here are post-fix by construction. No new baselines (same condition as anchor).
Expected effect: the teachers differ by 1.1° err (mpcof 0.70 vs pid 1.79). If the
student tracks its teacher, err moves by ~0.5-1.0°: resolvable. If it does NOT track
(student pinned at its own floor regardless of teacher), the delta is ~0 and the
CI will say so cleanly — that is also a resolvable, publishable answer.
Confound to name: the prior teacher screen (docs/l4_teacher_screen_results.md)
was attitude-only AND measured through the ~6× mis-scaled thermometer; none of its
numbers carry over. This is the first teacher screen under the current regimen.
Reads: does the DAgger student inherit teacher quality, or does it saturate at a
student floor? The WNN-minus-teacher gap per teacher is the number.

### Axis C — STATE NEURONS  (sn=4, sn=8 vs anchor sn=0)      8 runs, ~50-60 h

Values: sn=4 and sn=8 (--grid-state-neurons N --max-state-neurons N). These are
the ONLY lever family that ever produced sub-0.06 hd (attitude-only S1_lqi_sn8
0.0535, sn4 0.0618, Aug 2026, rotation-era scorer), and they have NEVER been flown
under the altitude regimen: the leaderboard has 0 sn>0 rows with alt. That is the
biggest unmeasured cell in the archive.
Prereqs: (1) confirm the recurrent trainer path is live at sn>0 under --translation
(the L4 memory says sn>0 gates `use_split` on WNN_STATE_SPLIT=1 — check what the
default path does at sn>0 and state it); (2) a 4-minute smoke at sn=4 with the
anchor flags; (3) a memory-budget check — a state layer grows the pool, and the
watchdog cap is 180k cells; state the expected cells before arming.
Expected effect: unknown under altitude; attitude-only history says large. Cost is
higher per run (bigger pool, longer MEMORY stage): budget 6-7 h per run.
Confound to name: sn>0 changes the search (an extra stage), so a win is "recurrent
state + its search" not "recurrent state" alone. Acceptable — that is how it ships.
Reads: does a state layer close the altitude gap (the column where the WNN is
worst, 0.35 m vs 0.08 m) — altitude is exactly the well-resolved channel (R3).

### Axis D — AIRFRAME  (cf2x_urdf, cf2x_firmware vs cf21_brushless)   8 runs, ~40 h

Values: the two other presets in airframe.py `_AIRFRAMES`.
Prereqs — THIS AXIS HAS A BLOCKING CONFOUND that must be resolved first:
LQR/LQI/MPC/MPCOF re-derive their gains from the airframe automatically; PID does
not, and only cf21_brushless has re-derived PID gains (airframe.py `PidGains`). On
the other two presets PID flies the retired plant's tuning, so the "PID baseline"
there is a mis-tuned controller, not a baseline. Two ways out, DECISION NEEDED:
  (i) re-derive PID gains for cf2x_urdf and cf2x_firmware first (the routine exists
      in airframe.py; needs a tuning pass and a test), then PID stays the comparator;
  (ii) use MPC (the next-weakest, self-deriving) as the comparator on those
      airframes and say so — but then Q1's "gap to PID" is not the same quantity.
Recommendation: (i). It is a one-time cost and keeps Q1 the same question everywhere.
Also: docs/disturbance_param_sources.md §"The two sources disagree on the airframe"
— cf2x_firmware and cf2x_urdf embody that disagreement; both are worth flying for
exactly that reason, and the spec should cite the section.
Expected effect: large (different inertia/thrust maps). Resolvable.
Reads: does the recipe (b24 n256, the weights, γ, leak) transfer across vehicles,
or was it tuned to cf21? Ties directly to the H743 deployability story.

### Axis E — LEVELS  (L=32, L=128 vs anchor L=64)            OPTIONAL, 8 runs, ~40 h

Values: via --grid-output-neurons 128 / 512 (levels = neurons/4 per motor).
Prior: the alphabet probe (09/08) and the levels ablation were REFUTED at their
bars, attitude-only; the bits ladder found no width separates at n≤5.
Expected effect: below the n=4 noise floor on every attitude column. Flying it at
n=4 would produce an indeterminate null by construction (R2).
Verdict: DO NOT FLY at n=4. Either drop it with the justification above, or fly it
only if a large-effect prior appears (e.g. from axis C). Listed for completeness.

## 4. Stage order, gates, budget

    stage  axis                    runs  ~hours  gate to start
    0      prerequisites (§5)      0     ~1 day  none — do now, in parallel with the queue
    1      A disturbance           8     40      baselines L4A/L4B banked
    2      B teacher               8     40      stage 1 markers 8/8
    3      C state neurons         8     55      sn>0 smoke + memory budget banked
    4      D airframe              8     40      PID gains re-derived (or decision ii)
    5      E levels                0     —       not flown unless a prior appears
    total                          32    ~175 h  ≈ 7.5 days of box time

Every stage is a marker-gated chain in the arm-A style (idempotent, fails closed,
one controller at a time, never edits a running .sh). The post-arm-A queue (~90 h)
runs first; this programme queues behind it. Stage 0 has no compute and is the
work to do NOW.

## 5. Stage 0 — prerequisites, no controller runs

  [ ] Baselines: compute_baselines.py --translation for (cf21, L4A), (cf21, L4B).
  [ ] Baselines: same for (cf2x_urdf, L4C), (cf2x_firmware, L4C) — only meaningful
      after the PID-gain decision (axis D prereq).
  [ ] Decision: axis D comparator, (i) re-derive PID or (ii) use MPC. Luiz's call.
  [ ] sn>0 path audit: what the trainer does at sn=4 with --translation and without
      WNN_STATE_SPLIT; one paragraph in this doc, with the file:line.
  [ ] Smokes: one 4-minute phased_ga per NEW flag combination (L4A, L4B, pid, lqi,
      sn=4, sn=8, each airframe) — rc 0 and a sane grid line. Same as arm A's smoke.
  [ ] Memory budget for sn=4/8: expected cells vs the 180k watchdog cap.
  [ ] Chains written for stages 1-4, modelled on arm_b_delta_label_chain.sh, each
      ending in a paired_power.py verdict against the anchor AND the condition's
      own baselines.
  [ ] Power statement per stage written INTO the chain header (R2).
  [ ] Review (§9) signed off.

## 6. Verdict protocol (pre-registered, per stage)

For each condition, on the MEMORY multi-seed row, paired by seed, four columns:
  V1. Δ(WNN_condition − WNN_anchor): mean, SD, 95% CI. The direct effect of the axis.
  V2. Δ(WNN − PID) at the condition vs Δ(WNN − PID) at the anchor: does the gap move?
  V3. The classical order at the condition (from its baseline file) and where the
      WNN's hd falls in it. Reported as a table, never as a bare inequality.
  V4. hd is reported but never the sole verdict; altitude is always shown (R8).
A stage "succeeds" if V1/V2 CIs exclude zero in either direction on any column —
i.e. it MEASURED something. An all-straddling result on an under-powered column
is written up as indeterminate with the n it would need (paired_power.py prints it).

## 7. What this programme is NOT

- Not a search for a better operating point. Nothing is re-tuned per condition; the
  recipe is frozen at the anchor's and moved. (Re-tuning per condition is a
  different, larger programme.)
- Not a factorial. 5 axes × 3 values × 4 seeds = 972 runs; OFAT from the anchor is
  32. Interactions are out of scope and said to be.
- Not a replacement for the queued 2x2 / arm B / window-k. It queues behind them.

## 8. Open decisions for Luiz

  D1. Axis D comparator: re-derive PID gains (recommended) or switch to MPC?
  D2. Axis E: drop with justification, or keep as a conditional tail?
  D3. Stage order: A→B→C→D as proposed, or move C (state neurons) first because it
      targets the WNN's worst column (altitude) on the best-resolved channel?
  D4. Seeds: 4 (matches everything banked) or 5 (adds 31337006, +25% cost, brings
      steady resolution from ~0.6° to ~0.5°)? Recommendation: 4 — the axes chosen
      have expected effects well above 0.6°, so the extra seed buys little here.

## 9. Review before arming

  - experiment-design agent: R1-R8, §6, the power statements, and whether OFAT from
    a single anchor supports the claim in §1 as written.
  - flight-dynamics agent: axis D's PID-gain confound and the airframe disagreement;
    whether L4A "clean plant" is a legitimate rung for a student trained with
    dither; whether sn>0 under translation has any observability trap.
  Both reviews go into this doc as dated sections before any chain is written.
