# CTRL-48: the steady gap, student side (09/11/2026)

Investigation only (training-algorithms agent). Scratch: /tmp/steady_student/ (trace.py, ana*.py, thr.py, lt_*.csv). The teacher and plant side is in docs/ctrl_steady_gap_teacher.md.

**Probe.** Each saved dres winner (s1-s3) runs one DAgger round on its own trained cells: 12 student-flown episodes at 5° (report seed 99990101, 2000 steps, report-seed thresholds). `WNN_LABEL_TRACE` records, at every decision, the teacher label at the student's state and the leaked baseline. These give the emitted setpoint sp_k = base_{k+1}/0.7 and the student's own decoded label L̂ = (sp_k − base_k)/0.6. "Tail" means t ≥ 1.6 s, the steady window.
- Probe err 1.49/1.78/1.81° matches the runs.
- Caveat: `_winner.yaml.gz` may not be the headline genome (s1 NEURONS#2, s2 MEMORY#1, s3 MEMORY#2).

**Correction to the brief.** The memory is **BINARY**, not QUAD: `--memory-mode BINARY` comes from the ctrl22 anchor argv.
- Cells are 1 bit, with the antagonist E/I decode, last-write-wins writes, and `protect_learned=False`.
- Writing FALSE deletes the cell, so FALSE is the same as a miss. There are no WEAK states and no confidence. Q3's "EMPTY/WEAK fraction" therefore does not exist by construction. The measurable proxy is the share of exactly neutral decodes (ΣE = ΣI).
- Levels per bank by genome: 118 (s1), 68 (s2), 120 (s3).

## 1. Finding: in the hold, the student does not reproduce the teacher's roll/pitch command (all 3 seeds)
Tail values, s1/s2/s3 ("r" = correlation of L̂ with the teacher label):

| bank | teacher label rms | student L̂ rms | r | slope L̂~L | sign agree | neutral decodes | sign flips/decision (student vs teacher) |
|---|---|---|---|---|---|---|---|
| p | .081/.084/.073 | .019/.027/.014 | .11/.19/.15 | .03/.07/.03 | .56/.55/.60 | .11/.19/.17 | .26/.34/.32 vs .02/.05/.05 |
| q | .093/.076/.154 | .018/.020/.037 | −.02/.06/.11 | .00/.02/.04 | .48/.53/.60 | .13/.24/.14 | .30/.40/.31 vs .05/.10/.02 |
| r | .16/.25/.19 | .13/.14/.12 | .67/.45/.70 | .54/.22/.43 | .88/.79/.88 | .03/.02/.02 | |
| T | .023/.028/.027 | .023/.023/.022 | .95/.91/.93 | .93/.74/.77 | .96/.94/.98 | .07/.02/.02 | |

- **The teacher's hold command is a slow, sustained, learnable signal.** It is a per-episode offset (episode-ID R² 0.76-0.94) that is linear in the own-family position inputs: p on (e_y, v_y) R² .76/.54/.85, q on (e_x, v_x) .83/.77/.64. This is the position loop asking for tilt back toward the target.
- **Those inputs are well resolved in the hold.** Tail codes of e_x, e_y, v_x, v_y spread over 6-7 of the 9 ladder codes (s1). The family map gives e_y and v_y to ROLL and e_x and v_x to PITCH.
- **The student does not act on it.** Its per-episode mean L̂ is about 0 (|·| < 0.025, against teacher means up to 0.26 rad/s), and its R² on the same inputs is 0.01-0.05.
- **Why this shows up as steady error (plausible, not measured):** the scorer uses `--att-err-ref tilt`, so the tilt the student fails to follow is scored as attitude error. Tail |e| has a median of 0.16 m.
- Roll/pitch fail and yaw/collective learn because only the roll/pitch hold labels depend on a small part of each neuron's address (see §5.1).

## 2. Output resolution (Q1): not the floor
- **Alphabet:** δ = 0.6·range·t² (γ = 2), with t on a grid of 1/(L/2).
  - Finest p/q δ: s1 0.0007, s2 0.0022 rad/s.
  - Fixed-point setpoint resolution is δ/0.3. The leak and the 10× repeat at 100 Hz are the same for the teacher rival "MPCOF>rate(h10)", so they do not differentiate the two.
- **Label flooring** (`antagonist_target` floors |t|·L): at the 0.08 rad/s hold label the step is 0.020 rad/s (s1) or 0.033 (s2). That is a −12% / −27% magnitude bias, second order next to r ≈ 0.1.
- **Dither:** the student's p/q setpoint is 0.02-0.06 rad/s rms and flips sign every 3-4 decisions. That is about 0.05° per flip, so the floor this alone implies is ≤ 0.1° and does not explain the 0.3-0.4° gap.

## 3. Labels and data (Q2)
- **Label convention mismatch (code-verified).** `dagger_label_delta` is off. The label is the teacher's ABSOLUTE setpoint, built by `output_target_at` → `output_decode_target` → `CtbrSpec::encode` on ±range. The decode, however, is a δ on 0.6·range added to 0.7·sp.
  - So the student executes sp_k = 0.7·sp_{k−1} + 0.6·L̂: DC gain 2 and a 28 ms pole.
  - Measured: the collective setpoint follows 1.5-1.85× its label (slope sp~L).
  - In transients, memory shrinks p/q to 0.56-0.59 and the gain of 2 restores an executed slope of 0.83-0.87. This plausibly explains why the delta law beat the absolute law in ct40 (grid steady 1.22° vs 1.92°).
  - It has no effect in the hold, where L̂ ≈ 0.
- **Hold share:** 85% of decisions have t ≥ 0.3 s at any tilt. Scorer-like states (max |p,q| label < 0.15 rad/s) are 49% of decisions at 5° but 41% at 8° and 12% at 30°. The 8→30° curriculum therefore under-weights them about 2×, though they are still abundant in absolute count.

## 4. Memory near steady (Q3)
- Neutral decodes on p/q in the tail are 11-24%, against 2-7% on r and T. Where p/q do read cells, the sign is at chance level.
- The trainer counter is not a fill measure: 1.06 M "writes" per 12 episodes are mostly FALSE writes, which are no-ops (FALSE is stored as a delete).

## 5. Student-side causes, ranked
1. **SUPPORTED (effect), PLAUSIBLE (mechanism): the hold command is erased by single-neuron, last-write-wins thresholds.**
   - A 0.08 rad/s label sets only the first 8 E or I levels (s1; 4 on s2). Each level is ONE neuron with one 24-bit address, and only about 6 of those bits are e/v.
   - So the other ~18 bits scatter hold states across addresses. Each cell keeps the sign of the last episode that wrote it, so the output sign is random (agreement 0.48-0.60).
   - Yaw survives with more levels (about 19). The collective survives because its whole family (alt_err, vz, coll_cmd) is relevant to the label.
2. **SUPPORTED: label/decode gain mismatch** (gain 2, 28 ms pole). It is not the hold cause today. It is a trap for any fix: once the hold is learned, the hold command will run at about 2×.
3. **Curriculum under-weights 5°-like hold states about 2×.** Minor.
4. **NOT causes:** alphabet and quantization (≤ 0.1°), action-repeat (the rival also decides every 10 ms), and output EMPTY/WEAK cells (BINARY has none).

## 6. Cheap tests (gate offline with trace.py before flying; the observable is tail p/q r and per-episode mean L̂)
1. **`--memory-mode QUAD_WEIGHTED --output-decode antagonist`** (same topology, neutral 0.5). The nudge keeps a short tally instead of the last write.
   - Expect: tail sign agreement > 0.7 and r > 0.4 → steady toward 0.7-0.8°.
   - If r stays at ~0.1, the cause is address scatter, not the write rule.
2. **`--target-levels 32`.** Each small-|t| threshold becomes about 3-4 neurons with independent addresses, so the hold sign is averaged.
   - The flag is global. Check the collective first: at T = 32 its first threshold is 0.016 pwm, against a hold label rms of 0.023. T = 20 would floor the collective hold to 0.
   - Expect: r up; transients about unchanged.
3. **`--rg-full-tilt-deg 15`.** This doubles the share of scorer-like hold states. Expect a small steady gain.
   - **Do not fly `--delta-max 0.3`** to gain-match: it removes the transient compensation (ct40: absolute lost).
   - The real fix for item 5.2 is a per-bank gain, which needs code.
