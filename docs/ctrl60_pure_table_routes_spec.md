# CTRL-60: pure-table routes for the yaw trim — SPEC (19/11/2026)

Luiz, 19/11/2026 ~17:20 EST: the float yaw P-term is dropped; F (`--yaw-trim-ff`) stays as it is; spec the
routes that keep the table as the controller. His question: **"is there a way to make the table's view of the
trim less coarse?"**

Docs only. No code changed, nothing flown, nothing committed. New measurements are VAL (the 20 stage-select val
units of each run, 100 episodes each), replayed through the installed ABI 59 wheel and gated to the production
scorer; the trace tooling refuses the report units by design, so the TEST units were not traced. Scripts and
data: `logs/controller/ctrl48/yf_gap/` (`trim_coarse.py`, `trim_ana.py`, `ladders.py`, `trim_Y_s1_val.npz`,
`trim_YF_s1_val.npz`).

## 0. Short answer
- **Yes, the trim input can be given more cut points. The measurement says that would not buy what F buys.**
- The trim is a per-airframe constant. Each unit's trim sits in one bin of the `cresid_z` ladder and stays
  there. If the bin width were what hurt, the standing yaw error of the F-less student would follow the trim's
  position INSIDE its bin. It does not (R² 0.00). It follows the SIZE of the trim (R² 0.60).
- So F does not help by seeing the trim more finely. It helps by taking the trim off the table's hands: the
  yaw bank no longer has to hold a large constant setpoint, only a small remainder.
- The pure-table route that attacks that same mechanism is a longer memory in the yaw bank's own accumulator
  (Route C2). It is unproven and needs a pre-check before any build.

## 1. How coarse is the table's view of the trim? (measured unless marked)

### 1.1 The cuts today
`cresid_z` takes the plain quantile ladder (8 cuts, outer cuts at the 11% / 89% calibration quantiles). The
cuts are fitted per seed and are identical for Y and Y+F at the same seed.

```
run (Y and Y+F)  | cresid_z cuts (its own units)
-----------------+----------------------------------------------------------------------
s1               | -0.724  -0.492  -0.286  -0.106   0.064   0.238   0.439   0.741
s2               | -0.769  -0.488  -0.321  -0.141   0.064   0.271   0.479   0.733
s3               | -0.661  -0.410  -0.236  -0.029   0.167   0.351   0.558   0.792
```
- One bin is 0.17–0.30 units wide. Through F's gain (cT = −0.39927 rad/s per unit) that is **0.07–0.12 rad/s of
  yaw-rate setpoint per bin**, or 0.40–0.72° of yaw at the teacher's proportional gain (9.59 rad/s per rad).
- Each yaw-bank neuron reads about 5 of its 24 bits from `cresid_z` (4.9–5.4), the rest from `yaw_err`,
  `yaw_err_i` and `gyro_z` in equal parts.

### 1.2 Where the trim actually lands (s1, 20 VAL units, hold window t ≥ 1.6 s; Y and Y+F agree to 0.002)
```
quantity                                                        | value
----------------------------------------------------------------+-------------------------------
cresid_z in hold, 1 / 5 / 25 / 50 / 75 / 95 / 99 %              | -1.08 -0.90 -0.31 -0.03 0.42 0.88 1.03
spread ACROSS units (sd of the unit means)                      | 0.513
spread between episodes of one unit (sd of episode means)       | 0.045
spread inside one episode's hold (sd)                           | 0.059
unit means, sorted                                              | -0.96 -0.85 -0.49 -0.48 -0.40 -0.24 -0.20 -0.18 -0.12 -0.05
                                                                |  0.01  0.05  0.31  0.34  0.36  0.47  0.50  0.71  0.84  0.91
bins the 20 unit means fall in (9 bins)                         | 7 distinct; 4 units beyond the outer cuts
bins hit by one unit's 100 episode means                        | median 1, max 2
hold ticks beyond the outer cuts (saturated)                    | 19.9% (22% over the whole episode)
```
- The trim is a property of the airframe: 0.51 across units against 0.05 within a unit.
- Inferred: the 35 TEST units draw from the same motor-wear distribution, so they will look the same.

### 1.3 What F delivers against what nine bins can express
```
quantity (hold, s1)                                             | rad/s            | yaw-equivalent
----------------------------------------------------------------+------------------+---------------
F = cT·cresid_z: spread (sd), 1–99% range                       | 0.207, ±0.42     | —
per-unit standing trim (mean |cT·cresid_z|)                     | 0.169            | —
best possible 9-level version: error mean / p95                 | 0.023 / 0.056    | 0.14° / 0.33°
  inside the ladder (bins 1–7): error sd per bin                | 0.019–0.036      | 0.11–0.22°
  beyond the outer cuts (bins 0, 8): error sd                   | 0.039–0.046      | 0.23–0.27°
cuts needed for a uniform ladder over the hold range ±1.1       | 21 / 43 / 87     | for 0.12° / 0.06° / 0.03°
```
So "coarse" is real in the sense Luiz means: nine levels reproduce F's setpoint to about 0.02 rad/s on average
and 0.06 at the 95th percentile, where F is exact.

### 1.4 But the coarseness is not what the F-less student pays for (Y s1, 2000 VAL episodes)
- The Y student's standing yaw error in hold is 0.295° per episode (Y+F: 0.101°).
- Regress the episode's hold yaw on (a) where its trim sits inside its bin and (b) the bin's own mean trim:
  - (a) the within-bin part explains **R² 0.00**, slope +0.17°/unit. If the table simply ignored the trim
    inside a bin, the slope would be +2.39°/unit (cT / K_θ).
  - (b) the bin-level part explains **R² 0.60**, slope −0.57°/unit: the yaw error grows with the trim itself.
  - Per bin, mean |yaw|: 0.14° at zero trim, 0.23–0.24° at ±0.2–0.35, 0.34–0.58° at ±0.4–0.6, 0.37–0.41° beyond
    the outer cuts. The saturated bins are not worse than their neighbours; only there does the within-bin
    slope turn positive (+0.6, +1.7°/unit), a partial quantisation signature.
- With F on, both dependences vanish (R² 0.07 together) and the per-unit standing error falls from 0.29° to 0.07°.
- Reading (inferred): the yaw bank under-delivers a setpoint in proportion to its size, and the yaw error
  settles where the proportional response makes up the shortfall. At the student's own measured yaw gain
  (about 4–5 /s) the shortfall is roughly a tenth of the trim. F removes 88% of what the bank must deliver in
  hold (`l2yp_investigation/ff_precheck_verdict.txt`: 0.139–0.172 → 0.015–0.021 rad/s), so the same
  proportional shortfall becomes negligible. This agrees with the 16/11 finding that the offset scales with
  the "need", and with the 16/11 NO-GO on a finer yaw ladder for the hold offset.

## 2. ROUTE A — finer cut points on the trim input, no F
**Mechanism.** More levels on `cresid_z`, so the table can learn the trim setpoint to F's resolution.

**Is this L2y / L2y′ again?** No, but it is next door. L2y (`--obs-yaw-trim-err`) and L2y′
(`--obs-yaw-trim-err-p`) changed the yaw-ERROR input: they subtracted κ·cresid_z from `yaw_err` and gave that
shifted input its own dense ladder (±0.25, ±0.75, ±1.50°). The `cresid_z` ladder was never touched. Route A
leaves `yaw_err` alone and changes only the `cresid_z` ladder. What carries over from L2y′ is its diagnosis:
the yaw neurons already address `yaw_err` × `cresid_z` jointly, so re-presenting the trim added no information.
Section 1.4 says the same about resolution.

**What exists.**
- There is NO dense / centred option for the residual inputs. The fixed-cut machinery
  (`evaluator._dense_inner_for`) covers `yaw_err`, `yaw_err_i`, the trim-shifted yaw error and the tilt errors
  only. `--calib-ladder centred` would re-shape `cresid_z` (it is in `ZERO_REFERENCED`) but also every other
  zero-referenced input, with the same 8 cuts.
- The cut COUNT is one global number (`--bits-per-feature`, default 8). The address layout is
  `feature · bits_per_feature + bit` in Python, in the Rust controller and in the Metal kernel.

**Three ways to do it, by cost** (T0–T2, named so they are not confused with the CTRL-58 A2 block)**.**
```
variant                                   | what changes                         | trim levels | cost
------------------------------------------+--------------------------------------+-------------+-----------------------------
T0  --bits-per-feature 16 (flag exists)   | EVERY input gets 16 cuts             | 17          | no code; needs a smoke on the
                                          |                                      |             | current stack; confounded
T1  cresid ladder that covers the range   | 8 cuts re-placed to span ±1.1 units  | 9           | Python only (threshold fit);
    (new `--trim-ladder cover`)           | (kills the 20% saturation)           |             | bins get WIDER inside
T2  per-feature cut count                 | cresid_z alone gets 32–48 cuts,      | 33–49       | Rust + Metal + Python layout
    (new `--bits-feature cresid_z=N`)     | uniform over ±1.1 units              |             | change, ABI bump
```
- Proposed fixed cuts for T2: uniform, step 0.05 units (0.02 rad/s, 0.12° yaw-equivalent) over ±1.1 = 44 cuts;
  or step 0.07 = 32 cuts. For T1: ±0.15, ±0.45, ±0.75, ±1.05.
- Bits: T0 doubles the input code (27 × 8 → 27 × 16 bits); T2 adds 24–40 bits to one feature. Neurons keep
  24 address bits either way, so the per-neuron address space is unchanged.
- Memory (inferred, not measured): cells are stored sparsely per visited address. A finer trim ladder splits
  the yaw bank's visited addresses by airframe, so its cell count grows, bounded by the ratio of levels
  (about ×2 for T0 across all banks, up to ×4–5 for the yaw bank under T2). Today's winners are about 1.2 MB.
- T2 is a feature-layout change: every consumer of the layout must change together (`feature_layout.py`,
  `evaluator.arch_shape_from_spec` and the receptive-field families, `compute_features`, the kernel's
  thermometer block, the connectivity samplers, checkpoint load). This is the class of change that once left
  the controller memoryless while it appeared to train.
- All three are new lineages (thresholds and addresses change).

**A/B (if flown).** Base Y, no F. Arm T = Y + the finer trim ladder. Compared against BOTH existing rows of the
same seed: Y (dmy Y s1-s5 — "does the table now learn the trim?") and Y+F (`…_ffF_full_s1..s5` — "does it match
F?"). s1-s5, TEST over 35 units, paired by seed.
- **Expected (from 1.4): steady 0 to −0.03° against Y; it will not match Y+F** (Y+F − Y is −0.22° steady and
  −0.20° err on s1-s5). err flat. pos flat.
- **Rule:** "the table learns the trim" = steady better than Y by ≥ 0.10° AND within 0.03° of Y+F, err not
  worse than Y+F by more than 0.03°, pos not worse by more than 0.010 m, fails not above either row. A gain
  under 0.05° against Y on the first three seeds is a not-adopt candidate (expectation-consistency); Luiz
  decides on stopping.
- **Check on the winner:** the 1.4 regression. If Route A works, the bin-level slope (−0.57°/unit today) must
  shrink towards zero.

**Recommendation.** Do not build T2. If Luiz wants the direct experimental answer to his question, T0 is the
only variant worth flying (no code), as a 3-seed look, knowing it also refines every other input.

## 3. ROUTE B — F without a multiplier
F is one multiplication by a constant fixed per airframe (cT = −0.39927 on cf21).

**Shift-add.** 0.39927 ≈ 2⁻² + 2⁻³ + 2⁻⁶ + 2⁻⁷ + 2⁻¹⁰ = 0.399414 (relative error 0.036%): four additions of
shifted copies of a fixed-point `cresid_z`, then a negation. Worst-case setpoint error over the measured range
(|cresid_z| ≤ 1.2): 0.00017 rad/s.

**Lookup.** A one-input table `cresid_z → setpoint offset`. From 1.2–1.3: 128 entries over ±1.28 units (step
0.02 units = 0.008 rad/s per entry, maximum error 0.004 rad/s, 0.02° yaw-equivalent), clamped outside; 256 bytes
at 16 bits per entry. 64 entries give 0.008 rad/s / 0.05°.

**How fine is fine enough.** There is no setpoint grid in the simulation; the yaw-rate setpoint is a float
handed to the rate loop. The natural yardstick is the yaw bank's own smallest step: with 64–68 neurons per
bank, γ = 2 and `--delta-max-yaw 0.5`, the smallest increment is about 0.001 rad/s, or 0.003 rad/s of held
setpoint through the 0.7 leak (inferred from the decode formula, not measured). Shift-add is 20 times finer
than that. A 128-entry lookup is at that level.

**Does it need a flight?**
- Shift-add: no. It is not bit-identical to the float product, but its error is bounded far below the bank's
  own step and below anything the scorer resolves. It is a firmware / MCU-export matter. If wanted, one Rust
  unit test can pin the bound.
- Lookup: its error is at the bank's step, and the DAgger remainder label would change with it, so "equivalent"
  would have to be flown (a quantised-F flag, Rust + Metal + ABI, non-inferiority against Y+F).

**Is it worth doing now? No.** The multiplication in F is one among many outside the network: the Mahony
estimator, the firmware rate PID, the mixer, the residual observer that produces `cresid_z`, and the 0.7 leak
of the output law are all multiplications. The survey's ruling stands: "multiplier-free" is claimed for the
network only. Route B is a paragraph for the paper:
> The yaw-rate setpoint adds a fixed feed-forward of the rate loop's unabsorbed trim (one constant gain taken
> from the teacher's gain row). On the target it is four shift-adds; it is not learned and is not part of the
> network. An ablation row reports the controller without it.

One variant does change the claim and is noted for completeness: a LEARNED one-input lookup bank (a weightless
bank that reads only a fine ladder of `cresid_z` and adds to the yaw setpoint). Trained on the teacher's labels
it converges to quantised F, and it would put F inside the network. It needs the per-feature cut count of T2
plus an absolute-decode bank, and a flown non-inferiority check. Not recommended unless Luiz wants F inside the
table for the claim.

## 4. ROUTE C — other pure-table ways
### C1. Dense yaw ladder WITHOUT F (flag exists, no code)
- Arm = Y + `--yaw-centred-ladder dense`, no F; against Y and Y+F, same design as Route A.
- Mechanism: the F-less yaw error parks inside the ±0.39° zone where the yaw bank is a relay; with cuts at
  ±0.10° / ±0.25° the proportional response can make up the trim shortfall at a smaller error.
- Evidence against: the 16/11 pre-check put the gain of a perfect yaw ladder on the hold offset at about 0.02°.
- Expected: steady −0.02 to −0.08° against Y; not F's −0.22°. Worth flying ONLY if tonight's ladder block
  (`docs/ctrl48_yawladder_prereg.md`, on Y+F) shows the relay zone is fixed; if that block is flat, C1 is dead.

### C2. A longer memory in the yaw bank's accumulator (yaw-only leak) — needs a pre-check, then Rust + ABI
- The output law is `setpoint ← leak · previous + δ`, leak 0.7 on every bank. To HOLD a trim setpoint s the bank
  must emit δ = 0.3·s on every decision, forever. That standing output is what 1.4 shows it under-delivers.
- With a yaw-bank leak of 1 − 2⁻⁴ = 0.9375 the standing output is 0.0625·s: 79% less to hold (F: 88% less),
  and the accumulator, not the table's cells, carries the trim. It is still the network's own output law, and
  a leak of 1 − 2⁻ᵏ is a shift and a subtraction.
- Risk, and why this is not a recommendation yet: a longer memory multiplies the bank's own step-to-step code
  noise by 1 / (1 − leak) (3.3 → 16), which can cost transient error and jitter. The leak family has a poor
  record here (global leak 0.90 closed on its altitude cost; the granularity pair refuted). Those were global
  changes that also moved the collective bank; a yaw-only leak avoids that cost the way `--delta-max-yaw`
  avoided D's, but that is an argument, not a measurement.
- Only `--delta-max-yaw` is per-bank today; a `--delta-leak-yaw` needs the same Rust work as ABI 57 did
  (decode, DAgger label encoding, Metal twin, parity, flag-off gate).
- **Pre-check before any build (VAL, no training):** (i) the rival probe's "label law through the student's
  output law" what-if (`rival_probe.rs`) at yaw leak 0.875 / 0.9375, for the alphabet cost; (ii) the
  calibrated hold-offset model of `l2yp_investigation/yaw_ff_calibrated.py` with the leak changed. GO only if
  the predicted hold yaw offset is within 0.03° of F's and the alphabet cost is under 0.01°.
- Expected if it passes: steady within 0.03–0.05° of Y+F without F. Uncertainty: high.

### Not proposed
A yaw-error-rate input (it is `gyro_z`, already read, 4.9 bits per neuron); L2y, L2y′, `--obs-yaw-trim-err`
variants and block 10b P / H (closed); the float P-term (dropped; stage 0: the float law alone flies yaw 0.540°
at scale 1.0 and 0.447° at 0.5 against the label law's 0.289°).

## 5. How the routes relate to tonight's ladder block and to A2
- The ladder block keeps F and attacks the TRANSIENT (err). Routes A, C1 and C2 ask whether F can be REMOVED;
  they attack the hold offset (steady) of the F-less student. They do not compete with the ladder block, and
  none of them is expected to improve on Y+F.
- A2 (refinement fitness with the mid-window term) is independent of all of them: it changes the search, not
  the representation, and stacks on whatever base is chosen.
- With F kept, the paper needs one sentence and the ablation row (Y against Y+F, already flown on s1-s5:
  err 1.03 → 0.83°, steady 0.43 → 0.21°).

## 6. Ranked recommendation: what to fly after the ladder block
1. **A2** on the chosen base, as planned.
2. **C1** (Y + dense yaw ladder, no F), n=3 first — only if the ladder block passes. Free, pure table, and it
   answers "how much of F can the table recover through resolution".
3. **C2 pre-check** (offline, VAL). Build and fly only on GO. It is the one pure-table route aimed at the
   mechanism F acts on.
4. **Route A, variant T0** (`--bits-per-feature 16`, no F), n=3 — only if Luiz wants his question answered by experiment
   and not by section 1.4. Expected to be flat.
5. **Route B**: the paper paragraph. No flight.
Do not build the per-feature layout change (Route A, variant T2) or the learned trim bank on today's evidence.

## 7. The `ctrl59-yaw-p-ff` branch (commit 5ba2af06, ABI 60, not merged, not installed)
It carries two things: the dropped `--yaw-p-ff` and the mid-window scorer row A2 still wants.
- **Merge as is, flag default-off.** One build, one round of gates; the flag-off determinism gate covers the
  P-term paths. Cost: a dropped lever stays in mainline in the controller, the DAgger trainer, the Metal
  kernel's uniforms and about a dozen Python forwarding sites — surface that every later parity change has to
  carry, and a flag someone can switch on by accident.
- **Strip the P-term, merge only the mid-window row as ABI 60.** The cleaner mainline, in line with the
  project's minimal-design rule; the P-term commit stays on the branch as the record. Cost: a second pass by
  `rust-code`, and the cargo suite, parity sweeps and the determinism gate run again on the stripped build.
- **Leave the branch and fly A2 on the Python-only `pre` fallback at ABI 59.** No wheel at all, soonest; the
  term is blunter (it includes the first 0.3 s).
- My recommendation is to strip: stage 0 showed the P-term is weak on its own, so it is unlikely to come back,
  and if a per-bank yaw leak (C2) is ever built it would be a natural companion for the same ABI bump.
  Luiz decides.

## 8. Open questions for Luiz
1. Is section 1.4 enough of an answer to the trim-coarseness question, or should Route A's T0 fly (3 seeds, about 4 h)?
2. C2: run the pre-check? It needs a rival-probe cargo test on a busy box (nice 15) and no flight.
3. The branch: strip the P-term, merge as is, or use the fallback.
4. Does the claim need F inside the network (the learned trim bank), or is one sentence plus the ablation row
   final?
