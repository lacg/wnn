# CTRL-59: float yaw P-term feed-forward (`--yaw-p-ff`) — implementation SPEC (19/11/2026)

Luiz, 19/11/2026 ~14:15 EST: build it now, in parallel with the dense-yaw-ladder block. This is a SPEC for the
`rust-code` agent. No code was changed to write it. Controller ABI **60**. `ram_controller` only (swap-free:
`maturin develop --release -m controller/Cargo.toml`); no `ram_core` change, no worker swap.

## 0. Why (measured, `logs/controller/ctrl48/yf_gap/`)
- On the Y+F base the whole-episode yaw error is 0.395° against the label law's 0.287° (TEST, n=3 seeds,
  +0.108 ± 0.016 across units, 35/35 units). Roll / pitch are +0.044 / +0.044 and seed-variable.
- `--yaw-trim-ff` (ABI 58) moved the TRIM part of the teacher's yaw law out of the table into floating point,
  and the hold yaw fell from 0.353° to 0.101° on the same seeds. What the yaw bank still has to produce is the
  PROPORTIONAL part, from a `yaw_err` input with no magnitude resolution inside ±0.39°. Measured on the s1
  winner (VAL): inside that zone the bank's output is flat (a relay), and the yaw error creeps to its floor over
  about 1 s.
- This lever moves the proportional part into floating point too. The bank then learns only what is left.

## 1. The law
All quantities BODY-frame yaw-rate setpoint, rad/s, at the outer (decision) rate; yaw angles in rad.

```
e_yaw   = target_yaw − yaw_heading          # the float the yaw_err feature is built from, before any ladder
FF_trim = cT · cresid_z                     # ABI 58, unchanged (cT = K_I·L = −0.39927 on cf21, n_o = 10)
FF_p    = K_P · e_yaw                       # NEW
sp_r    = clamp(bank_r + FF_trim + FF_p, −range_r, +range_r)        # ONE clamp, range_r = 2.0 rad/s
label_r = sp_T − (FF_trim + FF_p)           # what the yaw bank's DAgger label becomes (the remainder)
```

- **Which yaw error.** Exactly the value `compute_features` computes at `controller.rs:6483`
  (`let yaw_err = target_y - self.yaw_heading;`): the student's own anchored heading, which under
  `--obs-yaw-from-att-est` is its Mahony yaw (drift 0.017° measured). Not the true yaw, not the ladder code, not
  the L2y′-shifted value. No new sensor and no new information: it is the number the student already reads.
- **Sign.** The teacher's unconstrained law is `sp = K_θ·(θ − target) + …` with `K_θ < 0`
  (`yaw_trim_err.rs`: `k_theta < 0.0` is asserted; `CtbrAxisMpc::gain_row(...)[TH]`). With
  `e_yaw = target − θ` this is `sp = −K_θ · e_yaw`, so **`K_P = −K_θ · scale > 0`**.
- **Value.** `K_θ` comes from the SAME gain row `cT` and κ come from (`YawTrimErr::ff_gain` /
  `YawTrimErr::new`: the student's own rate loop and airframe, its decision hold, phase 0, lag-free,
  `CTBR_MPC_WEIGHTS`, `CTBR_MPC_HORIZON`, `CTBR_MPC_RATE_MAX[YAW]`). On cf21_brushless at n_o = 10 the existing
  pin (`ctrl58_yaw_trim_ff_tests.rs:37-38`) gives `K_θ = cT / κ = −0.39927 / 0.04165 ≈ −9.59 rad/s per rad`.
  **Never a hard-coded number**: the flag carries a dimensionless `scale`, the gain is derived.
- **The bank's own value is untouched.** `self.pwm[YAW_BANK]` stays the delta-law accumulator, as under ABI 58,
  so the leak (`--delta-leak 0.7`) never bleeds either FF. `--delta-max-yaw` keeps its meaning (the bank's
  alphabet); it does not scale the FF.
- **Clamp margin.** Largest |FF_p| in the regime: 5° × 9.59 = 0.84 rad/s; largest |FF_trim| seen: 0.51 rad/s;
  sum 1.35 < 2.0 rad/s. The clamp does not bind on the FF alone.
- **No angle wrap.** `e_yaw` is not wrapped in `compute_features` today and the FF must use the identical
  float. Fine in this regime (initial yaw ≤ 5°); a yaw-target or large-angle regime would need the wrap added
  in ONE place for both the feature and the FF.

## 2. The flag
- **`--yaw-p-ff [SCALE]`** — `nargs="?"`, `const=1.0`, `default=None`, type float. `None` = off = byte-identical
  to ABI 59. `SCALE` multiplies the derived `−K_θ`; it must be finite and in (0, 2].
- **Requires** (refuse loudly at launch, in `resolve_block10_args`, and again in Rust in `enable_yaw_p_ff`):
  - `--yaw-trim-ff` — one FF path, one label-remainder rule. A P-term without the trim FF is a different
    experiment and is not built;
  - `--obs-yaw-err` — the kernel reads `e_yaw` from the feature array (section 4.2);
  - everything `--yaw-trim-ff` already requires (`--output-mode ctbr`, `--obs-cascade-dhat-residual`);
  - NOT with `--obs-yaw-trim-err[-p]` (already refused by `--yaw-trim-ff`).
- Works with `--delta-max-yaw`, `--obs-yaw-from-att-est` and either yaw ladder.
- Provenance: the `yawref=` token (`block10_tokens`, `phased_ga.py:4505-4514`) gains `+pff<scale>`
  (e.g. `yawref=mahony+ff+pff1`). A new lineage: labels change.

## 3. Choosing the gain: stage 0, before any training (VAL only)
P at the teacher's full `K_θ` is the teacher's law with its other state terms removed (rate, actuator and
filter states); those terms now have to come from the bank, whose `gyro_z` ladder saturates at ±0.03 rad/s.
Whether full gain is well damped on this plant (motor settling time 0.20 s, τ = 0.05 s; 100 Hz hold) is not
known from disk. So:

- **Stage 0 (measurement, no student, no flight):** in the rival probe (`rival_probe.rs` /
  `rival_probe_tests.rs`, an `#[ignore]` cargo test, VAL fixture `scripts/ctrl_indi_val_fixture.py`, the 20 val
  units — never 99990101-99990400), fly the label law with ONLY its yaw setpoint replaced by
  `clamp(cT·cresid_z + scale·(−K_θ)·e_yaw, ±2.0)`, fed the estimate + bias compensation. Roll, pitch and
  collective stay the label law's. Arms: `scale` 1.0 and 0.5. Report stable, fails, err, steady, pos and the
  per-axis yaw err / hold against the unmodified law (VAL: yaw hold 0.075°).
- **Default = 1.0** if at scale 1.0 the yaw-only law is fail-free and its whole-episode yaw error is within
  1.3× the label law's. Otherwise the A/B flies 0.5, and the 1.0 row is reported as the reason.
- This row is also the honest answer to "how much of the yaw axis is the float law alone" (section 7) and must
  be kept.
- The A/B flies ONE scale. A second scale is a follow-up arm only if the first shows the expected sign.

## 4. Where it goes

### 4.1 Rust, CPU (one implementation; no second copy of the formula)
- `controller/yaw_trim_err.rs`: `YawTrimErr::p_gain(spec, dt, hold, scale) -> Result<f32, String>` beside
  `ff_gain`, built from the same `gain_row` call (factor the shared row construction so κ, cT and K_P read one
  row; do not paste the call a third time). Refuse a non-finite or non-negative `K_θ`.
- `controller/controller.rs`:
  - field `yaw_p_ff: Option<f32>` beside `yaw_trim_ff` (struct ~3902, default `None` at ~3057);
  - field `yaw_err_last: f32`, written where `yaw_err` is computed (6483), zeroed in `reset`;
  - `enable_yaw_p_ff(scale)` beside `enable_yaw_trim_ff` (2078): requires `yaw_trim_ff.is_some()` and
    `obs_yaw_err`, refuses a second call and L2y / L2y′;
  - `yaw_p_ff_gain() -> Option<f32>` beside `yaw_trim_ff_gain` (2109);
  - **`yaw_ff_value()` (2117) returns the TOTAL feed-forward** `FF_trim + FF_p` — the single formula decode and
    the recorder both read. With `yaw_p_ff == None` it must take the existing branch unchanged (no `+ 0.0`),
    so flag-off stays bit-identical;
  - `decode_ctbr` (6983): unchanged in shape — it already adds `yaw_ff_value()` and clamps once;
  - the two "alternatives" refusals (2027, 2063) need no change (P requires the trim FF, which they refuse);
  - PyO3 constructor: new keyword `yaw_p_ff: Option<f32> = None` after `yaw_trim_ff` (signature ~4090, args
    ~4159, body ~4268), getter `yaw_p_ff` beside `yaw_trim_ff` (4333).
- `controller/dagger_train.rs`:
  - the label site (2058-2061) needs NO new arithmetic: it subtracts `yaw_ff_value()`, which now carries both
    terms. Change only its guard/comment so the remainder rule is stated once: "bank 3's label = sp_T − the FF
    this tick's decode added". The flown command (`applied`) stays untouched;
  - `dagger_train_batch_inplace` keyword `yaw_p_ff` (signature 3290, args 3377, the controller build at 3486)
    and `dagger_train_inplace` likewise, forwarded to the controller it constructs.
- Every POSITIONAL constructor call must gain the argument: `imitation_probe.rs:720`,
  `ctrl51_flags_off_tests.rs:54`, and any other the compiler flags.
- `controller/lib.rs`: `ABI_VERSION = 60` (579), the ABI doc block (~564), `mod ctrl60_yaw_p_ff_tests;`.
- The classical teachers and rival scorers (`score_classical_baseline`, `ctbr_mpcof.rs`) do NOT change: the FF
  is a student-side law. The stage-0 what-if lives in `rival_probe.rs` behind its own option.

### 4.2 Metal twin
- `controller/shaders/controller_rollout.metal`: params struct (333-334) gains `uint ypf_on; float ypf_kp;
  uint ypf_idx;`. The decode site (1958-1959) becomes, for `m == 3`:
  `ff = P.yff_ct * sensors[<cresid_z>]; if (P.ypf_on != 0u) ff += P.ypf_kp * sensors[P.ypf_idx];
  ctbr_held[m] = clamp(pwm_acc[m] + ff, -r, r);` — same association order as the CPU
  (`trim`, then `+ p`, then `bank + ff`). With `ypf_on == 0` the emitted expression must be the ABI 59 one.
- `ypf_idx` = the index of the `yaw_err` feature in the kernel's float feature array, computed HOST-SIDE from the
  Rust feature layout (the function that places `yaw_err` in `compute_features`), never a literal. The feature
  layout is load-bearing (the 9-vs-10 feature bug): a unit test must assert that, on the CPU, the feature
  vector's entry at that index equals `yaw_err_last` bit for bit.
- `controller/metal_controller.rs`: the uniform struct (408-410), its fill (1261-1262), the struct-size comment
  (8376), and the batch guard (1714): refuse a batch that mixes `yaw_p_ff_gain()` values.
- Acceptance check for "every kernel site": every line that reads `yff_on` / `yff_ct` today has the P-term next
  to it. Today that is exactly the one decode site; if the search finds another by build time, it gets the
  term too.

### 4.3 Python — every forwarding site (mirror `yaw_trim_ff` hit for hit)
The "--fit-weight-alt was a no-op at 4 of 5 sites" and "the trainer never got the airframe" bugs were both a
missed site. **Acceptance: every file below that mentions `yaw_trim_ff` also mentions `yaw_p_ff`, with the same
number of forwarding sites.**
- `wnn/control/evaluator.py`: `ControllerSpec.yaw_p_ff: float | None = None` (beside 276); helper
  `yaw_p_ff_kw(spec)` (beside 745); the controller builds at 716 and 1634; the batch calls at 2245 and 2363.
- `wnn/control/dagger.py:412`, `wnn/control/reward_gated.py:511`, `wnn/control/ga_memory.py:170` and `:346`,
  `wnn/control/arch_adaptation.py:86` (plus the import lines 32 / 55 / 44 / 51 of those files).
- `wnn/control/controller_grid_search.py:211` (reads `args`).
- `wnn/control/phased_ga.py`: `_make_spec` parameter (550) and assignment (611); the argparse entry (beside
  3184); validation in `resolve_block10_args` (beside 4474); the provenance token (4512).
- `wnn/control/_accel.py`: `EXPECTED_ABI = 60` (181) and the ABI comment block (176).
- `wnn/ram/experiments/params.py`: `"controller_yaw_p_ff"` in `KNOWN_PARAMS` (beside 254).
- Checkpoints: `ControllerSpec` is serialised with the winner; an ABI 59 winner has no `yaw_p_ff` and must load
  as `None` (test it).
- Typed, explicit parameters only; no `**kwargs`, no module-level state.

## 5. Tests (all in `cargo test -p ram_controller --lib --no-default-features`, plus the Python suites)
New file `controller/ctrl60_yaw_p_ff_tests.rs`, in the style of `ctrl58_yaw_trim_ff_tests.rs`:
1. **Gain:** `K_P(scale = 1) = −K_θ`, in (9.0, 10.2) rad/s per rad on cf21 n_o = 10, and `= cT / κ` negated to
   1e-4; scales linearly.
2. **Refusals:** without the trim FF; without `obs_yaw_err`; with L2y / L2y′; twice; non-finite or
   out-of-range scale.
3. **Setpoint-only:** with the flag on, `bank_value(3)` is unchanged by the FF; the held setpoint equals
   `clamp(bank + cT·cresid_z + K_P·e_yaw)`; the leak acts on the bank only.
4. **Label remainder:** the recorded bank-3 label equals `sp_T − (FF_trim + FF_p)` with the FF of the SAME
   tick, on every label path (CPU split, bptt windows, GPU record / train, both walks); the flown command is
   untouched.
5. **CPU / GPU parity with the flag ON:** the score sweep (all rows the fitness ranks, plus steady and the
   hold columns) and the walk parity, as the ABI 58 suite does for the trim FF.
6. **Feature-index pin** (4.2).
7. **Flag off = ABI 59:** the existing flags-off pins (`ctrl51_flags_off_tests.rs`) extended with the new
   argument at `None`.

Outside cargo:
- **Determinism gate, flag off:** `scripts/ctrl_determinism_gate.sh` — ABI 60 ×2 against ABI 59 must be
  identical in gen lines, stage-select lines, checkpoint fingerprints and `[gen-digest]`s. Required before any
  ABI 60 run is paired with an ABI 59 row. It needs the box quiet (it refuses while a phased_ga runs).
- **Flag-on smoke:** one short phased_ga with `--yaw-p-ff` exits 0 and prints `yawref=…+pff…`.
- **Python:** the forwarding-site test of 4.3, the old-checkpoint load, the full controller pytest suite.
- If the mid-window scorer row of `docs/ctrl58_a2_prereg.md` (Amendment 19/11) rides this ABI, its tests are
  listed there; the two changes are independent and each must pass its own flag-off check.

## 6. The A/B (prereg to be written from this section once stage 0 has picked the scale)
- **Base:** Y+F. **B** = the existing rows `…_a59_…_P_both_L1_ffF_full_s1..s5`, not re-flown; valid only with
  the flag-off determinism gate 60 vs 59 passed and an argv that differs by the one flag plus paths.
- **Arm P** = Y+F + `--yaw-p-ff <scale>`, ABI 60, s1-s5 seed-major. Suggested tags
  `C-S2-d_wposx2_a60_rf_c200_dc_biasin_ms01_P_both_L1_ffF_ypf_full_sN`. About 7 h.
- **Expected:** err −0.06 to −0.09°, carried by the yaw axis (whole-episode yaw −0.08 to −0.11°, i.e. to the
  label law's 0.29°); steady −0.01 to −0.02°; pos flat; stable 100%, 0 fails. Roll / pitch unchanged.
- **Rule (TEST, 35 units, paired by seed, P − B; Luiz judges the whole vector):** ADOPT if mean err is better
  by ≥ 0.05° and better on at least 4 of 5 seeds, the per-axis split shows the gain on yaw, steady is not
  worse, pos not worse by more than 0.010 m, and fails are not above B's (B has 0 so far; any fail weighs
  heavily). Consistently flat or wrong-signed against the expectation = not-adopt candidate at any point.
- **Checks to report:** the per-axis split; the yaw gain curve and time profile of the s1 winner
  (`yf_gap/yaw_gain.py`, `yaw_profile.py`); jerk and effort (a float P on a noisy heading can add command
  noise); the share of ticks with the yaw setpoint at the clamp (0% today).
- **Relation to the dense yaw ladder** (`docs/ctrl48_yawladder_prereg.md`): the two attack the same share of
  the gap from the input side and the output side. They are tested separately against the same B. If both
  pass, a stacked arm decides whether they add; do not assume they do.

## 7. What this costs the paper claim (for Luiz)
- With `--yaw-trim-ff` and `--yaw-p-ff` the **yaw axis is a classical loop plus a table residual**: a fixed
  floating-point law (a trim feed-forward and a proportional term, both taken from the MPC teacher's own gain
  row) sets the yaw-rate setpoint, and the weightless table adds a learned correction. Roll, pitch and
  collective remain pure table outputs.
- It also adds one multiply-add per decision on the yaw axis, so "multiplier-free" no longer holds for yaw.
- Honest wording: "the yaw-rate setpoint is the sum of a fixed linear feed-forward (trim and proportional
  terms derived from the teacher's gains) and the table's output; roll, pitch and thrust setpoints are produced
  by the table alone." Report the stage-0 row (the float yaw law with no table) next to the WNN row so the
  reader sees what the table adds on yaw.
- Not acceptable: describing the controller as a lookup table on all axes, or quoting the yaw-axis result as
  evidence that the table learned the yaw law.
- The dense-ladder arm is the version of this fix that keeps yaw a pure table. If it delivers most of the
  effect, it is the one to prefer for the claim.

## 8. Open points
- **P only, or P + rate?** The teacher's row also has a rate gain (`gain_row[W]`). Feeding `K_ω·gyro_z` forward
  as well would leave the bank only the actuator-state terms, at the price of a fully classical PD yaw loop.
  Not approved and not in this build. Stage 0 says whether P alone is well damped; if it is not at scale 1.0,
  this is the alternative to halving the gain. Luiz's call.
- **Which base if the dense ladder is adopted first.** This spec assumes Y+F.
