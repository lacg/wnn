# CTRL: which MPCOF row is the comparator, and how the WNN-vs-MPCOF claims must be worded (ruling, 14/11/2026)

Ruling by `experiment-design`, requested by Luiz 14/11/2026 (N3 of `docs/ctrl_indi_teacher_spec.md` §11.4). No code, no runs, no new look at the test units. Every number below is copied from an existing doc or from `logs/controller/ctrl48/runs.log`.

**Partition of every number here: TEST** = 35 CF2.1 units 99990101..99990135 × 100 episodes = 3500 episodes per controller, fixed scorer, unless a line says VAL.

## 0. The rows (TEST)

```
row                                                   | stable % | err °      | steady °   | pos m       | fails           | source
------------------------------------------------------+----------+------------+------------+-------------+-----------------+------------------------------
A  MPCOF cascade, est-fed, as shipped (no bias comp)  | 100.00   | 1.352      | 0.684      | 0.341       | 0/3500          | spec §12.1 arm 1
B  MPCOF cascade, est-fed + attitude-bias comp        | 100.00   | 0.691      | 0.165      | 0.357       | 0/3500          | spec §12.1 arm 2 (f6e084e8)
C  MPCOF cascade, oracle-fed (label law; a bound)     | 100.00   | 0.690      | 0.163      | 0.357       | 0/3500          | spec §12.1 arm 3
D  WNN "both", rule P, mean±std over seeds s1-s9      |  99.98   | 1.37±0.07  | 0.77±0.06  | 0.361±0.003 | 6/31500 (0.019%)| runs.log END *_P_both_full_s1..s9
E  MPCOF direct (no cascade)                          | 100.0    | 1.81       | 1.06       | 0.354       | 0/3500          | ctrl48_fixed_scorer_table.md
F  PID (firmware cascade, as shipped)                 |  91.1    | 3.33       | 2.76       | 0.352       | 310/3500        | ctrl48_fixed_scorer_table.md
```

- Row D is N = 9 seeds (s10 had not landed when this was written). Per seed steady: 0.82 / 0.82 / 0.68 / 0.83 / 0.72 / 0.71 / 0.75 / 0.86 / 0.74. Per seed fails: 2 / 0 / 0 / 1 / 0 / 1 / 1 / 1 / 0. **5 of 9 seeds are fail-free; 4 of 9 are not.** Recompute with s10 before any of this goes into the paper.
- Rows A, B, C, E, F are deterministic laws: one row each, no seed spread. The nine WNN seeds fly the same 3500 episodes, so 31,500 is not 31,500 independent episodes.
- Row B was pre-registered on val (1e01dea6), read on val (38f15727), and scored on test once (f6e084e8). One pre-registered val condition was missed (pos +0.016 m against an allowance of +0.005 m) and was accepted by Luiz before the test look. That must be disclosed wherever row B is introduced.

## 1. Which rows the paper shows, and which one is THE comparator

**Ruling: row B (equal-information MPCOF) is the comparator for every sentence of the form "matches / approaches / is within X of MPCOF". Rows A and B must both be in the table. Row C must be shown, labelled as a bound and as the label source.**

Why B and not A:
- **The comparison must hold information constant.** Since block 7 the WNN reads `--obs-att-bias-pred`. Row A does not have that signal; row B has exactly that signal (Mahony estimate + horizontal velocity, no true quaternion). A WNN-vs-A gap mixes two things: the controller, and a 0.52° estimator handicap that only the rival carries. A reviewer who knows that will read "within 0.09° of MPCOF" as a comparison against a weakened baseline.
- **The handicap is ours to remove, and we removed it for one side only.** The bias predictor is our own code, added to the student. Leaving it off the rival and then claiming near-parity is the baseline-strength artifact this project has been burnt by before (TeacherBank, pre-fix scorers).
- **Row A's 0.68° is an estimator limit, not a controller limit** (spec §11.4 item 2). Its closeness to the WNN's 0.77° is a coincidence of that limit, not evidence about the WNN.

Why A must still be shown:
- It is the row every earlier doc, tick and memory entry quoted as "the MPCOF teacher". Dropping it silently would hide the correction.
- It is the honest "what an MPCOF cascade does on the stock estimator" row, and the A→B step (−0.52° steady, 35/35 units) is itself a result worth one sentence.
- It is NOT "as deployed firmware": MPCOF is not what the CF2.1 ships. The only as-deployed row is F (the firmware PID cascade). Do not call A "the deployed rival".

Why C must be shown:
- The DAgger labels come from it. The honest statement of the imitation gap is D vs C: **steady 0.77° vs 0.163°, +0.61°**. That is the number the paper owes the reader about imitation quality.

Open item on the same principle: **row F (PID) has no equal-information twin.** PID's failure is stability (310 fails), which bias compensation probably does not fix, but that is not measured. See §3 claim 1.

## 2. Wordings for the current numbers

How big the gaps are (row D, N = 9 seeds; standard error of the mean steady 0.02°, of the mean err 0.02°):

```
WNN "both" minus ...            | Δ err °  | Δ steady ° | read
--------------------------------+----------+------------+-----------------------------------------------
B  equal-information MPCOF      | +0.68    | +0.61      | WNN clearly worse on both (about 10 seed-SD)
C  oracle-fed label law         | +0.68    | +0.61      | same; this is the imitation gap
A  shipped MPCOF, no bias comp  | +0.02    | +0.09      | err indistinguishable at N=9; steady worse by 0.09°
F  PID cascade                  | −1.96    | −1.99      | WNN clearly better; see §3 for the caveat
```

- Stability: 6 fails in 31,500 (0.019%) vs 0 in 3500 for A/B/C. The upper 95% bound on a 0/3500 row is about 0.09%, so the rates are **not distinguishable**. That is not the same as equal: the WNN has shown fails, MPCOF has shown none.
- Position: every row sits at 0.34-0.36 m because the task fixes it (`ctrl_gap_after_block9.md` bottom line 5). **No position claim in either direction.**

**Is "within 0.09° steady of MPCOF" publishable?** Only as a statement about row A, never alone, and never as the headline. It must carry the qualifier in the same sentence and row B in the same table.

Safe wordings:
- "Against an MPCOF cascade given the same attitude-bias information as the WNN (row B), the WNN holds 0.77±0.06° steady and 1.37±0.07° mean error (N = 9 seeds, test, 35 units × 100 episodes), against 0.17° and 0.69°."
- "The WNN's steady error is 0.09° above an MPCOF cascade that runs on the uncorrected Mahony estimate (0.68°); that rival lacks the bias-prediction input the WNN receives, and with it reaches 0.17°."
- "The WNN is stable in 99.98% of test episodes (6 failures in 31,500 over 9 seeds; 5 of 9 seeds have none). The MPCOF cascade had no failure in 3500."
- "The labels come from the oracle-fed MPCOF law (0.16° steady); the WNN reaches 0.77°, an imitation gap of 0.61°."
- "Correcting the predicted Mahony bias brings the est-fed MPCOF cascade to within 0.002° of its oracle-fed bound (35 of 35 units)."

Forbidden wordings:
- "matches / reaches / is on par with MPCOF", "approaches MPCOF", "teacher parity", in any form, on err or steady.
- "within 0.09° of MPCOF" or "within 0.09° of its teacher" without the qualifier. The teacher (label source) is row C at 0.163°.
- "the student closed the gap to its teacher" for steady or err. Only the **stability** gap is closed in the sense above.
- "100% stable" for the WNN. Say 99.98% and give the count.
- "0.68° = the MPCOF teacher's 0.68°" (the block-8 line on seed s3). It quotes the best of 9 seeds against the handicapped row: best-of-N plus the wrong comparator.
- "the WNN beats MPCOF on roll/pitch" (block 6: 0.62° vs 0.67°). True only of row A; row B holds 0.094° / 0.083°.
- Any single-seed WNN number next to a rival row. The WNN row is the seed mean±std with N stated.
- Any VAL number (spec §11, `ctrl_gap_after_block9.md`) beside these rows without the VAL label.

## 3. What the WNN can defensibly claim now, and what each claim still needs

The paper can no longer be "a lookup table that matches MPC". It can be "a lookup table that is as reliable as MPC on this task, at a measured accuracy cost, with bounded execution time". Verdicts:

1. **Beats the deployed firmware PID cascade — supported on test, with one hole.** 99.98% vs 91.1% stable, 1.37° vs 3.33° err, 0.77° vs 2.76° steady; the gap is far outside the seed spread. Hole: PID has no bias compensation and the WNN does. Smallest experiment: the PID cascade + the same bias compensation through the existing rival probe, val first, then one test look, pre-registered. Also confirm the PID row's feed label (est-fed) in the table before quoting.
2. **Stability on par with MPCOF across 35 airframes — provisional.** "Not distinguishable" (0.019% vs 0/3500) is what the data support; "equal" is not. The block-9 descriptive figure (fail-free seeds out of 10 and its lower bound) belongs beside it. A longer MPCOF row (more episodes per unit) would tighten the 0.09% bound; not required, but the wording must stay at "not distinguishable".
3. **Bounded, flat execution time — supported on the M4 bench, on an older genome.** `ctrl51_onboard_cost.md`: WNN worst window 1.006× its mean; exact MPCOF 7-14× its mean when the box binds (1.1-2.2M instructions in software double). Needs: the bench re-run on a "both" genome, and the MPCOF side stated for both f64 and f32.
4. **"Cheaper than MPCOF" — NOT supported, and must not be claimed.** Same doc: WNN averages 257k instructions per 10 ms window against MPCOF's 161k (f64), MPCOF in f32 is far cheaper still, and the WNN's 1.2 MB flash is about 30× MPCOF's 42 KB. Safe form: "constant-time, multiplier-free table lookup with a flat worst case; higher average cost and about 30× the flash of the MPCOF it imitates". The "no online QP" point is a worst-case-time argument, not a cost argument.
5. **Graceful degradation of the lookup memory under bit flips — supported for the table memory only; the head-to-head is NOT measured.** `ctrl52_bitflip_robustness.md`: ≥99.9% stable to about 1e-4 upsets per stored bit, on N = 3+1 genomes of the older dres lineage, 10 draws per rate. The thresholds and the shared rate-PID floats are as fragile as any float. "More robust than MPCOF" needs the MPCOF parameter hook (spec'd there, not built); the "about 33% MPCOF survival" figure is inferred and must not appear as a result. On the shared rate-gain table the WNN was slightly *more* sensitive (38 vs 29 of 384). Needs also: a re-run on "both" genomes.
6. **The method findings — supported as WNN-internal ablations (paired, same seed, test).** Making the estimator bias observable (biasin), the centred yaw ladder + yaw leak (both), receptive fields. These never depended on the rival row.
7. **A rival-side finding we can report — supported.** The est-fed MPCOF steady floor on this plant is the Mahony bias, and a velocity-driven predictor removes it (A→B, val then one test look).

What is not a contribution: accuracy parity with MPC, lower compute than MPC, any position result.

## 4. Are earlier blocks' conclusions affected?

**No gate decision changes.** What was checked:
- **Blocks 8 and 9, the 11/11 failure gate, rule P.** Every decision rule is WNN arm vs WNN arm: paired against the same seed's in-block control, pooled fail rate ≤ 0.05%, then steady, then err, tie band ±0.03° (`ctrl48_block8_result.md`, `ctrl48_block9_prereg.md`, `ctrl_select_tail_prereg.md`). The MPCOF row appears in none of them. Rule P selects on val units by the WNN's own tail; the rival is not an input. **Unaffected.**
- **Block 6** (`ctrl_steady_gap_block6.md`). The mechanism verdicts (mlab wrong sign, biasin makes the bias observable, the rest is yaw) are measured on the students' own flights. **Unaffected.** Two descriptive lines are now wrong as comparisons and need a note: "biasin already beats the MPCOF teacher on roll/pitch" and the "teacher parity" target of 0.65°. Both used row A.
- **Blocks 4, 5, 7.** No result doc exists under `docs/` for these; from `runs.log` and the block-8/9 docs they used the same paired-vs-control gate. I did not find a rival-referenced decision, but I did not read a pre-registration for them either. **Unaffected as far as checked; not independently verified.**
- **"Stable gap closed"** (memory entry of 09/11). Still true: rows A and B are both 100% / 0 fails. **Unaffected.**
- **Block-9 decision (dc_ms01 vs both, s1-s10).** WNN-internal, pre-registered, **unaffected**. Its motivation changes, though: "both" was preferred for steady partly because 0.77° looked close to 0.68°. Against 0.165° both arms are far away, so the 0.03° tie band decides between two controllers that are both about 0.6-0.75° from the comparator. Apply the rule as written; do not describe the winner as "closer to MPCOF" as if that were material.

What does change is the **framing** of the whole CTRL-48 programme: it was narrated as closing a 0.2-0.3° gap to the teacher. The gap to the equal-information rival and to the label source is 0.61°, and `ctrl_gap_after_block9.md` (VAL) already locates it in the student's reference tracking and yaw parking, not in the teacher.

## 5. Reporting from now on

- **Reference rows in every controller table, in this order:** C "MPCOF cascade, oracle-fed (label source, bound)", B "MPCOF cascade, est + bias comp (equal information) — COMPARATOR", A "MPCOF cascade, est, no bias comp", then the WNN arms, then F PID. Never a bare "MPCOF" or "MPCOF teacher" label again.
- **"Teacher" means the label source (row C).** "Rival" or "comparator" means row B. Row A is "MPCOF without bias compensation".
- **Gap columns are against row B** (err, steady) and against row C when the subject is imitation. A gap to row A may be printed only as a second column, labelled.
- **Ticks and the cohort table:** the reference line reads `MPCOF eq-info 100.0 / 0.69 / 0.17 / 0.357 (0)`; the old `100.0 / 1.35 / 0.68 / 0.341 (0)` line stays underneath as `MPCOF no-bias-comp`. The tick-format change needs Luiz's OK (the tick format is a fixed requirement).
- **Memory:** entries that say "MPCOF steady 0.68°" or "gap to MPCOF" need the row named; the stable-gap entry stands. Add: comparator = row B since 14/11; row A is the handicapped rival; no parity wording.
- **Old docs:** do not rewrite results. Add a one-line dated note at the top of `ctrl48_block8_result.md`, `ctrl_steady_gap_block6.md` and `ctrl48_fixed_scorer_table.md` pointing here (not done in this commit, which touches only this file).
- **WNN rows:** seed mean±std with N, fails as count/episodes, and the number of fail-free seeds. No best seed in a headline.
- **Row B is frozen.** It had its one test look. Any further rival tuning (observer gain 0.05, PID + bias comp) goes val first, pre-registered, then one test look, and enters the table as a new labelled row.

## Verdict

- "WNN matches / approaches MPCOF" on err or steady: **not supported** (TEST, N = 9 seeds; +0.61° steady, +0.68° err against the equal-information rival).
- "WNN stability not distinguishable from MPCOF": **provisional** (6/31,500 vs 0/3500).
- "WNN beats the firmware PID cascade": **supported**, pending the PID + bias-compensation row.
- Smallest experiments that would firm up the paper: (1) PID + bias compensation, val then one test look; (2) M4 cost bench and bit-flip sweep re-run on "both" genomes; (3) the MPCOF parameter-flip hook, if the robustness claim is to be comparative.
