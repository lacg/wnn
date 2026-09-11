# Multi-axis programme — specification (draft 2, 11/09/2026)

Status: DESIGN, not armed. Draft 2 folds in the experiment-design review (§9, 19
findings, received 11/09). The flight-dynamics review is pending; §9 will carry it.
Nothing is launched until both are in and the Stage 0 checklist (§5) is closed.

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
Confound: "noise = dither". A clean plant previously degraded a noisy-trained winner
1.1°→6.5°, so L4A may be HARDER for the student; train and evaluate at the same rung
only (cross-rung transfer is a different question). Encoder re-fit per rung (§1).
Prereqs: baselines at L4A and L4B with --translation (minutes each).

### Axis B — TEACHER  (pid, lqi vs anchor mpcof)             8 runs, ~40 h
Values: pid (weakest; translation costs it +0%) and lqi (integral; +10%). mpc is
excluded (translation +27% would confound the swap with the regimen); lqr deferred
as near-redundant with lqi for a first pass (flight-dynamics to confirm, §9).
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
Prereqs: (1) audit what the trainer does at sn>0 under --translation without
WNN_STATE_SPLIT (the L4 memory says `use_split` is gated on both; one paragraph
with file:line); (2) a 4-minute smoke at sn=4; (3) memory budget vs the 180k-cell
watchdog cap, stated before arming. Budget 6-7 h per run.

### Axis D — AIRFRAME  (cf2x_firmware vs cf21_brushless; cf2x_urdf deferred)  4 runs, ~20 h
Values: cf2x_firmware only. VERIFIED 11/09 by constructing the controller: the
firmware PID cascade builds on cf2x_firmware with ITS OWN sourced gains
(platform_defaults_cf2.h). cf2x_urdf is REFUSED (DSL single-loop gains, no rate
loop) and the Rust teacher then silently falls back to the legacy retired-plant
loop (dagger_train.rs:930). Because the WNN's thermometer is fit from PID rollouts
(evaluator.py:463), a fallback PID would contaminate the WNN's ENCODER, not just the
comparator — so cf2x_urdf cannot be flown at all until a citable DSL single-loop PID
is ported (Python + Rust + Metal, parity). Draft 1's "re-derive gains" is WITHDRAWN:
airframe.py records Luiz's 05/08 rule that invariant-preserving derivation
"manufactures an UNSOURCED number". docs/disturbance_param_sources.md also forbids
blending the URDF plant with firmware gains (different Crazyflie builds).
Primary: ALTITUDE (the transfer question is whether the recipe's altitude hold
survives a different thrust/inertia map); err descriptive.
Power caveat: the cf21 SDs give NO basis for the SD at another airframe. D's power
statement is a guess until its first two seeds land (§4 round rule).
Prereqs: baseline at (cf2x_firmware, L4C, --translation); one smoke.

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
    round 1  one seed each of  L4A, L4B, pid, lqi           (A and B have no prereqs)
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
    A disturbance               8        ~40
    B teacher                   8        ~40
    C state neurons             8        ~55
    D airframe (cf2x_firmware)  4        ~20
    E levels                    0        —
    total                      32       ~175 h  ≈ 7.5 days, plus escalation seeds
Every chain: marker-gated, idempotent, fails closed, one controller at a time, never
edits a running .sh. Queues behind the post-arm-A queue (~90 h).

## 5. Stage 0 — prerequisites, no controller runs

  [ ] Marker provenance fields (wheel hash, ABI, fitness_pools) exported by the
      ladder — R9. Check whether they already exist; add if not (Python only).
  [ ] Decision D5: fresh report-seed set for the final table — R10.
  [ ] Baselines --translation for (cf21, L4A), (cf21, L4B), (cf2x_firmware, L4C);
      and all of them again on the D5 seed set once chosen.
  [ ] Failure-count export for stable (R11): does the marker carry per-episode
      counts? If not, add them or declare stable descriptive.
  [ ] sn>0 path audit paragraph with file:line (axis C prereq 1).
  [ ] Smokes: one 4-minute phased_ga per new flag combination (L4A, L4B, pid, lqi,
      sn=4, sn=8, cf2x_firmware) — rc 0 and a sane grid line.
  [ ] Memory budget for sn=4/8 vs the 180k cap.
  [ ] Round-major chain written (one script, conditions x seeds, marker-gated),
      ending each round in a paired_power.py --primary verdict per condition against
      the extended anchor AND the condition's own baseline.
  [ ] Power statement per axis written INTO the chain header (R2).
  [ ] Flight-dynamics review received and folded in (§9).

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
- Not a replacement for the queued 2x2 / arm B / window-k. It queues behind them.

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

### 9.2 flight-dynamics review — PENDING
Questions posed: axis D PID-gain path (now resolved in code, see §3 D — reviewer
to confirm); L4A legitimacy for a dither-trained student; sn>0 under translation
observability; lqr vs lqi as the second teacher; the 1/L altitude prior for axis E;
anything physically missing from the axis list (control rate, actuator lag,
mass/inertia mismatch, wind).
