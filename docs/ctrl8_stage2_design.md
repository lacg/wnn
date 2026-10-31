# CTRL-8: stage-2 (horizontal) design, DRAFT for Luiz (30/10/2026)

Status: **draft. Nothing flies until Luiz approves it.** The prep is done:

- CTRL-17 gaps re-checked: G1–G4, G10a, G11 and G13 are fixed at ABI 31.
- 221/221 ram_controller tests pass.
- Rival bar CTRL-22 (411abc44).
- G6 label probe (964bfbac, d1f028fa).

Fitness = the decided one: S16noJM weights, zscore, alt 0.0, **no MAD floor** (CTRL-20 not adopted 30/10).

## 1. What CTRL-8 asks

Can the single-layer WNN learn to hold **position** when episodes start displaced horizontally, without selling attitude? The bundle is:

    --translation --xy-offset 0.5 --obs-pos-err-xy --obs-vel-xy

It sits on the `_op31` anchor argv: b24 n256, `--obs-pwm`, levels as banked, L4C, cf21, mpcof teacher, derived hover, 5 folds.

## 2. Measured inputs

**Rival bar (CTRL-22, estimator-fed, 5 report seeds, xy 0.5 m, mean start |xy| 0.385 m):**

| teacher | 2 s: err° / pos3d m | 6 s: err° / pos3d m |
|---|---|---|
| PID   | 2.08 / 0.349 | 0.97 / 0.277 |
| LQR   | 1.33 / 0.352 | 0.63 / 0.288 |
| MPC   | 1.37 / 0.355 | 0.65 / 0.314 |
| LQI   | 1.32 / 0.347 | 0.58 / 0.276 |
| MPCOF | 1.30 / 0.351 | 0.55 / 0.274 |

- All rivals are 100% stable in every cell.
- Without the horizontal loop (xy 0) they drift to 0.54–1.09 m by 6 s.
- The xy start costs the rivals about +0.3° of err at 2 s; that is G5, attitude measured against level.

**G6 label content (training-rollout trace, anchor recipe, about 1,040 episodes per arm):**

- The label's motor differential has SD 0.27, about 4 grid levels. 85–89% of motor-steps are ≥ 1/16 at xy 0 and at xy 0.5 alike, so the labels are not in the dead zone. They are busy with attitude under L4C.
- A linear fit on (e_x, e_y, v_x, v_y) explains **2–7%** of the label differential: under 1% in the first 0.2 s, 3–7% after 1 s.
- The lateral amplitude is about 0.02–0.03 per metre. At a typical 0.4 m start that is about 0.01, **one sixth of a grid level**.
- Caveat: position acts through the teacher's tilt reference, so part of the lateral content sits in the attitude terms, which this fit omits.
- **Risk: the student may learn to ignore the position features.** The twin control (§4) is there to detect exactly this.

## 3. Decisions (recommendation first; each is Luiz's call)

**D1. Position in the fitness.**

- **Recommend:** `--fit-weight-pos 0.10`, with the S16noJM weights scaled ×0.90. This is the same construction CTRL-19 used for alt.
- The DAgger gate λ_pos is left unset, so it is DERIVED (gate_lambdas.py: λ_pos = (w_pos/w_att)·θ²/s_xy²).
- Why not 0, as for alt? For altitude, the attitude terms were an acceptable proxy. For position they are **anti-correlated**: translating requires tilt, and tilt is scored as attitude error (G5). With w_pos = 0 the fitness would select genomes that refuse to translate.

**D2. Position metric (G10b).**

- `mean_position_error_m` is **3-D**. With w_pos > 0 it re-introduces altitude, about 0.16–0.20 m, which Luiz set to 0 on purpose.
- **Recommend:** add a horizontal-radial metric and use it for `--fit-weight-pos`.
  - This is a scorer change on CPU and Metal together, parity-pinned.
  - It changes nothing while w_pos = 0.
  - Keep reporting the 3-D figure beside it for the Molchanov comparison.
- Alternative: accept the 3-D metric and disclose it.

**D3. Attitude measured against level (G5).**

- **Recommend:** accept it and disclose it. The rivals are scored the same way, so WNN vs rival is fair.
- The alternative, measuring err against the cascade's tilt reference, changes the metric's meaning for every stage-2 row.

**D4. Episode length (G7).**

- `--steps` sets training and scoring together.
- At 2 s the rivals only get from 0.385 m to about 0.35 m, so position is transient-dominated. At 6 s they reach 0.27–0.31 m.
- 6 s costs about 3× per run, roughly 30 h instead of 10 h.
- **Recommend:** 2 s for the smoke. For the A/B, decide after the smoke reads criterion (d). If the WNN beats the twin at 2 s, keep 2 s.

**D5. Accelerometer is an ideal inclinometer (G12).**

- **Recommend:** disclose for CTRL-8, and fix before any paper claim on stage 2.
- The fix is specific force on CPU and Metal; it breaks lineage for every translation run.

**D6. MEMORY stage.** G2 is fixed, so it may run. **Recommend:** skip it in the smoke, and enable it in the A/B with a check that the recorded universe has non-zero horizontal bits.

## 4. Plan

**Step 1. Smoke (one seed, s31337002).** Pop 6, grid → neurons, MEMORY skipped, the D1/D2 choices. Two runs of about 1–2 h each:

- the arm;
- the **twin**: the same flags minus `--obs-pos-err-xy --obs-vel-xy`. Same translation and xy starts, but blind to position.

Pass criteria (audit §6):

- (a) no guard fires;
- (b) the fitter prints non-degenerate spans for all four xy features;
- (c) the gen line carries the position metric;
- (d) held-out horizontal error is below the start mean (about 0.38 m) **and below the twin's**;
- (e) the rival column starts off-origin.

**Step 2. A/B (only if the smoke passes (d)).**

- The arm `_s2p31` and the twin `_s2b31`, each on s2–s6, paired per seed.
- About 10 runs × ~10 h = **~100 h** at 2 s (or about 300 h at 6 s).
- PRE-REGISTERED RULE (to be fixed before launch): the position features are adopted iff all of the following hold:
  - mean paired horizontal error is lower than the twin's;
  - mean err is not worse than +0.10°;
  - mean stable is not worse than −0.5 pp.

  This is the same non-inferiority form as CTRL-20, plus a required position gain. Both arms are also printed against the CTRL-22 rival bar.

**Not in scope:** multi-axis (round 1 follows stage 2), state neurons, yaw_ref ≠ 0 (G8).
