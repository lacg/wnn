# CTRL-48: why `--obs-cascade-dhat-residual` (dres) beats `--obs-cascade-dhat` (dhat) — 09/11/2026

Investigation only (controller agent; scratch /tmp/dres_probe/). ABI 45, rf axis/cross 0.2 + c200. Probes:
- the production calibration sampler, 200 eps at s1's train seed (s1's exact ladders) + 60 disjoint eps as held-out;
- winner connections; per-unit held-out rows from the logs.

Not measured: teacher labels and EMPTY rates. No API exposes them, and the imitation_probe fixture drops the CTRL-38/48 flags.

## 1. Definitions (cascade_dhat.rs, cascade_obs.rs)
- **d̂** (`cdhat_*`): the teacher's observer, d̂ += 0.05·(ω̇ − c·u_a − d̂). Units rad/s², not normalised; 1 kHz;
  reset each episode. It is constant per unit (±20 % motor wear).
- **residual** (`cresid_*`): r = d̂/A + I/L. A = c·ki·L is the integrator authority, measured 95.7 / 95.2 / **3.65**
  rad/s²; the identity holds exactly (2e-7).
  - Equivalently r = (I − I_ss(d̂))/L: the integrator coordinate of the x − x_ss(d̂) MPCOF regulates (ctbr_axis_mpc.rs:152).
- Both are student inputs only (recorded and replayed). Labels are unchanged.

## 2. Feature behaviour (s1 quantile ladder, 8 thresholds)
| feature | between-ep var % all / hold | unit-ID share of hold code | held-out sat. 0-.1/.1-.3/.3-1/1-2 s |
|---|---|---|---|
| cdhat_x | 98.5 / 100 | **0.975** | .06/.21/.25/.23 |
| cresid_x | 60.5 / 89.6 | **0.60** | **.53/.70**/.33/.02 |
| rate_i_x | 91.8 / 99.9 | 0.97 | .00/.02/.19/.35 |
| cdhat_z / cresid_z | 97.5 / 97.0 | 0.82 / 0.78 | ≈ equal |
- **Roll/pitch:** in the hold, d̂ and I are each a per-unit constant (I → −d̂/A), and the residual cancels them.
  - Unabsorbed fraction r·A/d̂ by window: .77/.57/.25/.06.
  - Residual hold codes: 69 % in the middle 3 codes. d̂'s hold code is uniform and ~98 % set by the unit.
- **Yaw:** integrator τ ≈ 7 s > the 2 s episode, so the unabsorbed fraction stays 1.00 → 0.81; corr(cresid_z, cdhat_z)
  = +1.00 in every window. Quantile ladders are rank-invariant, so yaw codes are the same. **The arms differ only on
  roll/pitch.**
- **Transient blindness:** 85 % of residual samples are hold samples, so its ladder ends at ±0.1. 53-70 % of
  roll/pitch values saturate in the first 0.3 s, leaving sign only.
- Train/held-out code shift (TV .06-.21) is similar for both inputs, so it does not separate them.

## 3. WNN side
- **No selection:** every winner wires each input at the uniform within-family share (x/y .030-.034, z .047-.053);
  the connections stage is skipped.
- **Hold address split:** each neuron's own bits of the input, on 200 eps, t > 1 s.
  - roll/pitch: dhat 1.24-1.34 bits, unit-ID **0.98**; dres 0.99-1.09 bits, unit-ID **0.68-0.70**.
  - yaw: dhat 1.60-1.72 bits, 0.89; dres 1.56-1.64 bits, 0.87.
  - So d̂ splits hold addresses almost purely by unit, a split the roll/pitch hold label (sp_ss = 0) does not use.

## 4. Per-unit (35 units; asymmetry via disturbance_stream(fold_pool_seed(s, 0)), scale fitted)
- Paired dhat − dres err: +0.183° on s1 (dres better on 33/35 units), +0.068° on s2 (24/35).
- Paired steady: +0.201° on s1 (28/35), +0.010° on s2 (19/35).
- The gain tracks roll/pitch asymmetry: corr with max|d_roll, d_pitch| +.33/+.33 (err), +.44/+.24 (steady).
- It does not track yaw: −.02/−.31 (err), −.11/−.45 (steady). On s2, dhat wins on the 4 units beyond yaw authority 3.65.
- **dhat s2's 12 failures:** 1-2 each on units 12, 14, 16, 17, 18, 23, 31.
  - 12, 18, 23 have d̂ beyond the outer thresholds and are also the hardest units for c200 and MPCOF.
  - 14, 16, 17, 31 are mild units, and 6 of the 9 out-of-range units did not fail. **Not a range effect.**

## 5. Mechanisms ranked
1. **SUPPORTED:** on roll/pitch in the hold, d̂ is a unit ID the label does not use, and it fragments cell evidence
   by unit. The residual removes the absorbed part, leaving the deficit the teacher acts on. Q4 confirmed: d̂ varies
   across units, the residual within the episode.
2. **PLAUSIBLE, untested:** the RAM cannot form d̂/A + I/L itself (~0.75 bits per roll feature per neuron). rateI
   alone 1.84/1.87/1.76°, d̂ alone 1.67/1.73°, dres 1.49/1.66°; no arm feeds rateI + d̂ together.
3. **The main session's claim is half right:** the residual is ONE coordinate of x − x_ss(d̂). MPCOF also weights
   u_a − u_a,ss and Fh − Fh_ss; its gains K were not measured.
4. **REFUTED:** GA selection; a yaw difference; out-of-range d̂ as the failure cause; transient magnitude (it saturates).
5. **Caveats (experiment-design):** n=2 with s3 pending; s2 steady is a tie. Size confound: dres s1's headline has
   472 neurons, against 256 for dhat s1, which is a GRID genome.

## 6. Making it better (arms over rf_c200_dres, seeds s1..s3)
1. **`--threshold-outer-quantile 0.02`:** transient saturation falls .53/.70 → .15/.24, hold unit-ID stays .63,
   near-zero thresholds stay ±0.002-0.017. The flag is global, so this is a lineage break; a residual-only version
   needs code.
2. **`rf_c200_rateI_dhat`** (`--obs-rate-i --obs-cascade-dhat`), the mechanism test.
   - If it ≈ dres, the gain is information.
   - If it < dres, pre-summing is the lever: next derive (Fh + d̂/c)/out_limit, the next x − x_ss coordinate.
3. **`--drop-constant-features` on dres:** tgt_r/p/y are constant 0 and take ~12 % of the connections.
