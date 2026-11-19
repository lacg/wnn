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

---

## C2 pre-check result (19/11/2026 evening; Luiz's go 19/11 ~17:50 EST)

**Verdict: NO-GO for a longer yaw-bank leak. The pre-check also showed that the premise of section 4 C2 was
wrong.** VAL only (the 20 stage-select val units 163226054..163226073 × 100 episodes). No flight, no training,
no install; the installed ABI 59 wheel and the main checkout's `src/` were not touched.

### Correction to section 4 C2
I wrote that the yaw bank "must emit δ = 0.3·s on every decision" to hold a setpoint s. That is the true-delta
label (`--dagger-label-delta`), which this recipe does NOT use. As flown, the bank is trained on the teacher's
ABSOLUTE yaw-rate setpoint and the delta law integrates what it says:
`setpoint ← leak·setpoint + delta_max·B`, so at rest the held setpoint is **G·B, with G = delta_max / (1 − leak)**
(`l2yp_investigation/yaw_g_aware.py`, decode identity fitted at 1.97–2.01 for G 2). With `--delta-max-yaw 0.5` and
leak 0.7, G = 1.67: a table that says exactly the teacher's label holds 1.67 times the trim, and the yaw error
settles where the proportional response removes the excess. A longer leak RAISES G (4 at 0.875, 8 at 0.9375),
so it makes the trim-proportional error larger, not smaller.

### What was run
1. **Rival probe (the spec's item i).** The label law (MPCOF>rate(h10), est + bias compensation) flown through the
   Y base's output law: leak 0.7, delta_max 0.6, yaw delta_max 0.5, 64 cells per bank, ranges 0.25 / 4.2 / 4.2 / 2,
   γ 1 / 2 / 2 / 2, with the YAW bank's leak changed. Two blocks:
   - *ideal delta*: the existing what-if (the setpoint the alphabet can reach) — the alphabet cost;
   - *absolute yaw label*: the yaw bank driven as the flown students are (a perfect table that says the absolute
     label; the other three banks stay ideal). Added because of the correction above.
   - Temporary Rust, **scratch branch `ctrl60-c2-precheck`, worktree `/Users/lacg/wnn-c2-precheck`, local commit
     9cf689c6, not pushed, NOT for merge**: per-bank yaw leak / delta_max and the absolute-label mode in
     `rival_probe.rs::StudentAlphabet`, a yaw-setpoint step RMS, and the `#[ignore]` test
     `rival_probe_tests::val_c2_yaw_leak`. Built with `CARGO_TARGET_DIR=logs/controller/ctrl48/yf_gap/target-c2`,
     `nice -n 15`, `-j 6`. Fixture: `yf_gap/c2_fixture` (VAL, written from the Y+F s1 run log). Self-check passed
     (the probe build reproduces the installed wheel's est-fed row on unit 163226054). Log: `yf_gap/c2_probe.log`.
2. **Calibrated hold-offset model (the spec's item ii).** `yf_gap/c2_model.py` = `yaw_ff_calibrated.py` with G
   changed, on the VAL replays already on disk (6 units × 100 episodes per seed): the M genomes re-scored at
   G 1.67 (s1-s5) and the flown Y genomes (s1, s2).

### Rival probe (VAL, 20 units × 100 episodes; no student in this table)
```
arm                                        | stable % | fails | err °  | steady ° | pos m | yaw whole ° | yaw hold ° | yaw sp step RMS all / hold rad/s | per-unit hold yaw vs trim: slope °/unit, R²
-------------------------------------------+----------+-------+--------+----------+-------+-------------+------------+----------------------------------+--------------------------------------------
label law, no alphabet                     | 100.00   | 0     | 0.693  | 0.171    | 0.355 | 0.289       | 0.075      | 0.071 / 0.067                    | −0.001, 0.00
IDEAL DELTA (alphabet cost)                |          |       |        |          |       |             |            |                                  |
  leak 0.70,   dmax 0.5  (G 1.67) control  | 100.00   | 0     | 0.693  | 0.172    | 0.355 | 0.289       | 0.075      | 0.071 / 0.067                    | −0.002, 0.00
  leak 0.875,  dmax 0.5  (G 4)             | 100.00   | 0     | 0.693  | 0.172    | 0.355 | 0.289       | 0.075      | 0.071 / 0.067                    | −0.001, 0.00
  leak 0.9375, dmax 0.5  (G 8)             | 100.00   | 0     | 0.693  | 0.172    | 0.355 | 0.289       | 0.075      | 0.071 / 0.067                    | −0.001, 0.00
ABSOLUTE YAW LABEL (as flown, perfect table)|         |       |        |          |       |             |            |                                  |
  leak 0.70,   dmax 0.5  (G 1.67) control  | 100.00   | 0     | 0.993  | 0.579    | 0.355 | 0.645       | 0.532      | 0.032 / 0.026                    | −1.25, 1.00
  leak 0.875,  dmax 0.5  (G 4)             | 100.00   | 0     | 1.379  | 1.035    | 0.356 | 1.053       | 1.000      | 0.048 / 0.027                    | −2.34, 1.00
  leak 0.9375, dmax 0.5  (G 8)             | 100.00   | 0     | 1.612  | 1.309    | 0.356 | 1.276       | 1.257      | 0.089 / 0.064                    | −2.72, 1.00
  leak 0.70,   dmax 0.3    (G 1) [extra]   | 100.00   | 0     | 0.727  | 0.173    | 0.355 | 0.342       | 0.077      | 0.020 / 0.016                    | −0.003, 0.00
  leak 0.875,  dmax 0.125  (G 1) [extra]   | 100.00   | 0     | 0.779  | 0.174    | 0.355 | 0.419       | 0.078      | 0.013 / 0.006                    | −0.002, 0.00
  leak 0.9375, dmax 0.0625 (G 1) [extra]   | 100.00   | 0     | 0.915  | 0.182    | 0.355 | 0.595       | 0.088      | 0.009 / 0.003                    | −0.015, 0.03
```
- "yaw sp step RMS" is the RMS decision-to-decision change of the flown yaw-rate setpoint (the noise measure).
- The last column regresses each unit's signed hold yaw on its mean trim (`cresid_z`, from `trim_YF_s1_val.npz`).
- **Alphabet cost of a longer leak: 0.000–0.001°.** Not the limit.
- **As flown, the trim-proportional error is built into the decode.** Even a perfect table holds 0.53° of yaw at
  G 1.67, entirely explained by the trim (R² 1.00). The real Y student holds 0.30° with slope −0.57°/unit
  (section 1.4): its table has learned to say LESS than the teacher's label and so cancels about half of the
  excess. That is the R² 0.60 dependence.
- **A longer leak at the same delta_max doubles it** (1.00° at 0.875, 1.26° at 0.9375), with a noisier setpoint.
- **G = 1 removes it**: hold yaw 0.077° against the label law's 0.075°, trim R² 0.00, and the smoothest setpoint.
  The cost is in the transient: whole-episode yaw 0.342° against 0.289° (err +0.034°) at leak 0.7, and much more
  with a longer leak at G 1 (0.42° / 0.60°), because the accumulator then low-passes the label.

### Calibrated model (VAL replays of real genomes; predicted per-episode hold |yaw|, mean over seeds)
```
arm                                        | M genomes at G 1.67, s1-s5 | trim R² | flown Y genomes, s1-s2 | trim R²
-------------------------------------------+----------------------------+---------+------------------------+--------
leak 0.70 (control, G 1.67), measured      | 0.235°                     | 0.49    | 0.259°                 | 0.50
leak 0.875 (G 4)                           | 0.395°                     | 0.77    | 0.482°                 | 0.77
leak 0.9375 (G 8)                          | 0.463°                     | 0.82    | 0.563°                 | 0.82
F at leak 0.70 (reference)                 | 0.206°                     | 0.00    | 0.146°                 | 0.00
[extra] G 1 = delta-max-yaw 0.3, leak 0.70 | 0.209°                     | 0.00    | 0.153°                 | 0.00
```
The dependence on the trim does not shrink under a longer leak; it grows (R² 0.49 → 0.77 → 0.82).

### Go / no-go against the written criterion
> "GO only if the predicted hold yaw offset is within 0.03° of F's and the alphabet cost is under 0.01°."

- Alphabet cost: 0.000–0.001°. **Passes.**
- Predicted hold yaw offset against F: +0.19° / +0.26° (M genomes) and +0.34° / +0.42° (Y genomes) at leak
  0.875 / 0.9375. **Fails by an order of magnitude, in the wrong direction.**
- **NO-GO.** No per-bank leak is to be built. Section 4 C2 and item 3 of section 6 are withdrawn.

### What the pre-check can and cannot show
- The probe rows are the TEACHER's law pushed through the student's output law. They show what that law costs a
  table that says exactly the label. They are not a student and say nothing about what a table would learn.
- The model rows re-predict the hold offset of genomes trained at leak 0.7. A changed leak was NOT flown through
  a frozen table: the cells were written for G 1.67 or 2, and a different G changes what they should hold, so
  such a replay would not be a controller anyone would train. A retrained student can adapt (the Y student
  already cancels about half of the built-in excess), so the real numbers would differ from both tables.
- The model was validated for one step only (G 2 → 1.67: predicted −0.049°, measured −0.054° in the mean, the
  sign right on 3 of 5 seeds). G 4, G 8 and G 1 are extrapolations.
- VAL, s1-s5 / s1-s2, 6 units per seed for the model. Directions, not results.

### A by-product that Luiz should see (not asked for; nothing was flown)
The same two tools point at the pure-table route this spec was looking for: **make the yaw decode gain 1**, i.e.
`--delta-max-yaw 0.3` with the leak left at 0.7. The flag exists (ABI 57); no code, no wheel.
- Probe: hold yaw 0.077° (label law 0.075°), no dependence on the trim, at +0.034° of err in the transient.
- Model: within 0.003–0.007° of F on real genomes.
- It is the continuation of a step already flown: Y is delta_max_yaw 0.6 → 0.5 (G 2 → 1.67), which gave about
  −0.04° of steady. G 1 is the end of that line, where a table that says the label holds exactly the label.
- If Luiz wants it tested: arm = base + `--delta-max-yaw 0.3`, WITHOUT F, s1-s5, against Y and against Y+F of the
  same seed (the design of section 2). Expected from the model: steady close to Y+F's; err is the open question
  (the transient cost above, against F's gain). It would answer whether F is needed at all.
- Caveats: an extrapolation of a model checked on one step; the probe's transient cost is for a perfect table;
  the yaw bank's largest step per decision falls from 1.0 to 0.6 rad/s, which may slow the first 0.3 s.

### Files
- `logs/controller/ctrl48/yf_gap/c2_probe.log`, `c2_model.py`, `c2_fixture/`, `c2_val_fixture.py` (the fixture
  writer as it stands on commit 5ba2af06, copied so the main checkout's script was not edited), `target-c2/`
  (178 MB build directory; safe to delete).
- Scratch branch `ctrl60-c2-precheck` (local commit 9cf689c6) in `/Users/lacg/wnn-c2-precheck`; delete both when
  this section is accepted.
