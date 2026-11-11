# CTRL-48 "E": roll/pitch input representation for the cascade WNN (DESIGN)

Status: DESIGN ONLY, 08/11/2026. Nothing is built or queued. It stacks on "A" (the
`ThresholdCalibration` work in progress) and on CTRL-45 (axis receptive fields, ABI 44).
Sources read: `docs/ctrl44b_thermometer_encoding_audit.md`, `docs/ctrl44_dhat_investigation.md`,
`docs/ctrl45_axis_receptive_fields_design.md`, `src/wnn/control/feature_layout.py`,
`controller.rs::compute_features` (L5802-6015), `cascade_obs.rs`, `imitation_probe_ctbr.rs`,
`imitation_probe.rs` (`ridge`, `raw_rows`, `thermo_rows`), `episode_regime.rs::expert_ctbr` and
`set_horizontal_obs` (L146), `position_loop.rs::tilt_ref`, `att_err_ref.rs`, `ctbr_mpcof.rs` header,
`shaders/controller_rollout.metal` (feature twin ~L852, `tilt_ref_quat` L375), `mcu/wnn_cascade.c`
(features L171-262), `phased_ga.py` estimator resolution (L3750), and `experiments/ctrl22/ARGS_ma_anchor_lag020.txt`.

**Concurrent work (seen on disk while writing).** Uncommitted edits to `cascade_obs.rs` add a3
`obs_dhat_residual_plant` (`--obs-cascade-dhat-residual`), labelled "CTRL-48, ABI 45". That is CTRL-44's ranked
fix 2, the unabsorbed disturbance.
- **Label clash:** this doc's "E" should get its own ticket id, or the residual should be renamed.
- **ABI:** E1's ABI bump is 46, or E1 bundles into the same wheel.
- The residual is a derived per-axis feature in the cascade block. It is orthogonal to E1 (it adds 3 features; E1
  changes none) and belongs to branch 5 of the decision tree (hidden state). Do not fly both in one arm without a
  factorial.

Labels used below: **[M]** = measured (by a cited doc or probe). **[C]** = follows from reading the code.
**[I]** = my inference, not yet measured.

---

## 1. Problem and constraints

**[M] The gap (CTRL-44b, mean of s2–s4, ridge R² of the MPCOF CTBR label on held-out 30° teacher states):**

| | p (roll rate) | q (pitch rate) |
|---|---|---|
| production thermometer, 8 bits | 0.340 | 0.422 |
| best ladder: 30°-fitted, 16 bits | 0.534 | 0.608 |
| raw features, linear | 0.917 | 0.929 |

Yaw (r) and thrust (T) are close to raw once the ladders are fixed. A is expected to close about 40% of the
roll/pitch gap.

**Constraints.**
- IDS outranks this, so nothing here pauses IDS.
- Only the controller wheel changes, so no worker swap is needed.
- Every feature must exist in the Rust CPU path, the Metal twin and the C twin (`mcu/wnn_cascade.c`).
- No new RAM object. Learning stays connectivity plus memory writes.
- Rule 3: build the smallest thing that answers the question.

### 1.1 Three facts from the code that reframe the question

**F1 [C]: the thermometer-vs-raw ridge gap is NOT a missing interaction.**
- `thermo_rows` turns each feature into its own step bits, and `ridge` is a linear map on those bits. So the
  "thermo" model is an additive staircase: a generalized additive model with 8 (or 16) steps per feature.
- The "raw" model is additive too: one slope per feature.
- An 8-step staircase can approximate any monotone 1-D response, so it is a superset of the raw linear model
  except for quantization (bin resolution, placement, and saturation past the end bins).
- So the measured 0.534 → 0.917 gap is entirely quantization loss. An interaction can only explain the much
  smaller 0.917 → 1.0 remainder.

Question (1b) therefore needs its own measurement (section 2, D3). It cannot be read off the existing table.

**F2 [C]: the RAM is less additive than the ridge, so the ridge understates the problem.**
- Under the BINARY antagonist decode, each output neuron is trained on one level of its bank's label
  (`target bit = label > t_i`). The bank output counts the TRUE neurons.
- So neuron i must represent the half-space `{x : label(x) > t_i}` by itself, from its 24 address bits.
- The ridge can add contributions across features for free. A RAM neuron cannot: it has to co-observe every
  feature the label depends on, at enough thresholds to place that half-space boundary.
- With axis receptive fields at cross 0.2, a roll-bank neuron has 20 in-family connections across 7 ROLL features
  (about 2.9 thresholds per feature).
- [I] This is why the representation matters more than the R² table suggests. A derived feature that already
  equals the linear combination the label depends on turns each neuron's half-space into a threshold on one
  feature, which a single bit can express.

**F3 [C]: the student and the teacher use different attitude references.**
- The teacher plans on `tgt = PositionLoop::tilt_ref(e, v)` (`expert_ctbr`, `episode_regime.rs` L228-237).
- The scorer measures attitude error against the same reference (`--att-err-ref tilt` in the recipe).
- The student is stepped with `target = rig.target = [0,0,0]`. So:
  - `tgt_r/p/y` are dead (CTRL-44b finding 3);
  - `roll_err = target[0] − roll_est` is level-referenced (`compute_features` L5853);
  - `roll_i` integrates the level-referenced error.
- The quantity the teacher regulates, `θ_ref(e_y, v_y) − θ`, reaches the student only as three separate
  thermometers (roll_err, e_y, v_y), which a neuron would have to recombine.
- [I] In the 5° hold the scorer measures, the vehicle tracks a tilt reference of a few degrees (0.5 m offset,
  ωn = 1 → about 3°). So θ and θ_ref are similar in size and nearly cancel, and the label depends on their small
  difference. That is the regime where per-feature quantization error is amplified most.
- The audit's θ − τω test checked a rate/attitude combination, not the attitude/reference one, so this mechanism
  is untested.

**F4 [C], a confound to rule out first:** the student's attitude is a Mahony estimate. It is on by default under
the specific-force IMU (`phased_ga.py` L3753); the teacher uses the true quaternion. If the estimate is wrong by
an amount comparable to the hold signal, this is an estimator problem, not a representation problem, and E is the
wrong lever.

---

## 2. Diagnosis first (offline only: no GA, no DAgger)

### 2.1 One probe extension

Extend the existing CTBR imitation probe. Add a new function `diagnose` in `imitation_probe_ctbr.rs`, called from
`report`, gated by `WNN_IMIT_DIAG=1`. Reasons:
- `ImitCapture` already holds, per decision record:
  - the student's feature vector (`feats`);
  - the teacher's inputs, including the TRUE Euler angles and body rates (`tx`);
  - the teacher's hidden plan state, i.e. ua, shadow I, Fh, d̂ (`hidden`);
  - the student's real inner loop (`inner`);
  - the label;
  - the addresses and target bits.
- `ridge`, `raw_rows`, `thermo_rows` and `oracle_phys` already exist.

So every measurement below is a new row builder plus a call to `ls(...)`. No new capture, no new flight, no
Python. It runs on the same 51-run setup as CTRL-44b: s2–s4 winners, the A ladder once available, 30° pool plus
5° pool. CPU only. It is a diagnostic, so no GPU twin is needed. [C]

**Stratification.** Report every R² three ways:
- **ALL**: the CTRL-44b number.
- **TRANSIENT**: t < 300 ms, or |θ| > 10°.
- **HOLD**: 5° pool, t > 300 ms. This is what `steady` scores.

[I] R² is weighted by variance. On the 30° pool it is dominated by recovery transients, which the scorer barely
weighs, so the decision gates below use HOLD.

### 2.2 The measurements and what each one separates

| id | question | rows (ridge, train → held-out) | reading |
|---|---|---|---|
| D0 | ceiling at the decode grid | label vs `memoriser(label)` (exists) | R² ceiling; everything below is read relative to it |
| D1 | **(a) which feature's quantization costs** | hybrid rows: thermometer for every feature except f, which is raw; one row set per f in the roll/pitch families | ΔR²_f = R²(hybrid_f) − R²(thermo). Large ΔR² concentrated on 1–3 features means resolution/placement of those. Spread across many features means cancellation (D2) |
| D2 | **(a) vs (b): cancellation** | from the raw ridge: amplification A = Σ_f β_f²·Var(x_f) / Var(y), per stratum. Then thermo rows plus ONE extra feature, the thermometer-coded (8 bits, quantile on the train split) raw-ridge prediction ŷ | A ≫ 1 means the label is a small difference of large terms. If thermo+ŷ_thermo reaches about the raw R², one well-chosen linear combination recovers the information, and section 3's derived features are the right family |
| D3 | **(b) missing interaction** | raw rows + pairwise products within each family (roll: err×gyro, err×e_y, e_y×v_y, …) + clipped versions of the label's own clip (±4.2 rad/s) | if raw+products ≈ raw (< +2 pp), no interaction is needed; the remaining 0.08 is noise or hidden state |
| D4 | **(c) 24 bits too few** | level-set oracle on SYNTHETIC connectivities: draw 16 maps per policy (spread; axis-RF via `arch_project_receptive_fields`; axis-RF with the D2/E1 derived feature in the frame), compute addresses from the captured thermometer frames, then `oracle_phys` majority per address → bank R² | the same oracle at width 24 vs 48 (diagnostic only) separates "bits per neuron" from "the information isn't in the frame". If E1's frame raises the oracle at width 24, co-observation was the bottleneck |
| D5 | **(d) teacher hidden state** | raw+I, raw+ua, raw+Fh, raw+d̂, raw+all (exist); NEW: thermo + each hidden block thermometer-coded | raw+hidden − raw is the hidden-state share of the 0.08 remainder. ua matters here: motor lag 0.20 s and the chain's sed strips `--obs-pwm`, so the student does not see ua |
| D6 | **(e) estimator confound** | raw and thermo rows with roll_est/pitch_est replaced by true Euler from `tx` (re-thresholded on the train split) | if truth closes most of the HOLD gap, stop E and send the estimator to flight-dynamics |
| D7 | **E1 preview** | rows with `roll_err/pitch_err/tgt_r/tgt_p` recomputed on the tilt reference (section 3, E1) from the captured e, v, roll_est. Re-fit ladders on the train split, quantile 8 bits and the A ladder | the offline go/no-go for E1 |

**Reuse note.** D4's majority-per-address count is a diagnostic oracle. It is exactly what `oracle_phys` already
does, as an upper bound for a memoryless lookup with that connectivity. It is not a learner and never flies, so it
does not conflict with "what is NOT a RAM WNN".

**Caveat [M from CTRL-45 §2.2].** Offline predictability ranked closed-loop layouts at rho 0.3. These measurements
**diagnose the mechanism and gate the build**. They never pick the winner; only the closed-loop A/B in section 4
does.

### 2.3 Decision tree (pre-registered for the probe)

1. **D6 closes ≥ 50% of the HOLD thermo-vs-raw gap:** estimator problem. Stop E and open an estimator ticket.
2. Otherwise, if **D7 raises HOLD thermo R² by ≥ 1/3 of (raw − A-ladder thermo) on p and q (mean of seeds), or
   D4's E1 frame raises the oracle by ≥ 5 pp:** build E1 (section 4).
3. Otherwise, if **D2 shows A ≫ 1 and thermo+ŷ ≈ raw, but D7 does not:** the useful combination is not the tilt
   reference. Read the β of ŷ (which features carry it) and revisit E2 with Luiz (section 3).
4. Otherwise, if **D1 concentrates on 1–3 features:** that is per-feature resolution. A's centred ladder plus E3
   (targeted bits) is the answer, not a derived feature.
5. **D5 dominates the remaining gap:** the student is missing a state. The existing obs blocks (rate-I, obs-pwm/ua)
   are the lever, so no new representation is needed.

Cost: about 3 agent-hours, about 1 h of CPU probe time.

---

## 3. Candidate representations (all extend `compute_features` + the thermometer; none is a new neuron type)

| # | candidate | what changes | expected gain [I] | cost | risk | RAM-WNN model intact? |
|---|---|---|---|---|---|---|
| **E1** | **Tilt-referenced attitude target**: inside `compute_features`, when `att_target = tilt`, replace `target[0..1]` by `PositionLoop::tilt_ref(horiz_obs)` before `roll_err/pitch_err` are formed | `tgt_r/tgt_p` become the live reference (fixes 16 of the 24 dead bits); `roll_err/pitch_err` and `roll_i/pitch_i` become tilt-referenced, i.e. the teacher's and the scorer's coordinates | medium to high in HOLD, where θ ≈ θ_ref cancel; small in 30° transients | **low**: no new feature, so the layout, families, nf, Metal buffer sizes and MCU frame width are unchanged; a value change in 3 twins | the student inherits the classical position loop's shape (ωn, ζ, max_tilt); see flag 1 | **yes**: an input coordinate change; connectivity and writes unchanged |
| E2 | Per-axis linear combination z = θ_err + k·ω (+ hidden terms), with k from the firmware gains or the MPC's DARE gain | +2 features (ROLL, PITCH) appended in the cascade block | possibly high, but with the full DARE gain z ≈ the teacher's unclipped label | low to medium (appended block like `cascade_obs`) | **label leakage**: the student would be handed the teacher's answer, and the WNN degenerates to a quantizer of a computed controller. The partial PD version tested at 0.85× spread [M] | formally yes; scientifically doubtful, see flag 2 |
| E3 | More bits for roll/pitch features only (per-family bpf) | variable-width features | the 16-bit ladder gave +9 pp [M]. Under a fixed 24-connection budget, more thresholds per feature mostly add population DIVERSITY, not per-neuron resolution [C, F2] | **high**: `feature_of = idx / bpf` is assumed in `arch_ops`, `receptive_fields.rs`, Metal, MCU and the exporter | breaks the uniform-bpf invariant everywhere | yes |
| E4 | Two-stage / hierarchical: a first RAM stage computes the combination, the output stage reads it | sn > 0 (state layer as combiner), or a new layer | unknown | **high**: the recipe is sn = 0; state-layer training (BPTT GPU port) is still a plan | high | yes if it uses `RAMRecurrentNetwork`'s state layer; anything else is a new structure that **needs discussion** |
| E5 | Per-axis input subsets | — | — | — | — | already CTRL-45 (axis RF); E stacks on it |

**Ranking (gain/cost/risk):** E1 ≫ E2 (only with Luiz's sign-off) > E3 > E4.

E1 is the only one that:
- changes no layout or ABI-visible geometry;
- aligns the student's coordinates with the teacher and the scorer, which already use this exact reference;
- composes with CTRL-45 by construction: tgt_r and roll_err are already ROLL, so the bank's family now carries the
  e_y/v_y information in fewer bits.

---

## 4. Minimal first increment: E1, then a 2-arm A/B

### 4.1 Interfaces and placement

**Rust, crate `ram_controller` (controller wheel only).**
- `controller.rs`: new field `att_target: Option<PositionLoop>` (None = LEVEL = today, bit-identical) and setter
  `set_att_target(&mut self, pl: Option<PositionLoop>)`.
- In `compute_features`, immediately before L5853:
  ```rust
  let target = match self.att_target
  {
  	Some(pl) =>
  	{
  		let h = self.horiz_obs; // [err_x, err_y, vx, vy] = (−hx, −hy, hvx, hvy) (episode_regime L146)
  		let (r, p) = pl.tilt_ref(h[0] as f64, h[2] as f64, h[1] as f64, h[3] as f64);
  		[r as f32, p as f32, target[2]]
  	}
  	None => target,
  };
  ```
  The argument order matches `expert_ctbr`'s `tilt_ref(-hx, hvx, -hy, hvy)`. Pin this with a cargo test: student
  `tgt_r/tgt_p` equal the teacher's `tgt` on the same sim state.
- The 9-feature base block must also push the substituted target. Either move the `feats.extend` of `target` after
  the substitution, or substitute at the top of the function. The base-block parity anchor (`num_features ==
  NUM_FEATURES` early return) is unaffected, because `att_target` requires the horizontal channel.
- **One source of the loop's constants.** Build the `PositionLoop` from the same `(g, ωn, ζ, max_tilt)` that
  `OuterLoops::for_cfg` and `AttErrRef::tilt` use. No new constants.
- **Replay safety [C].** The value is derived inside `compute_features` from `horiz_obs`, which every host (live,
  DAgger record/replay, the calibration sampler, the Metal in-loop and buffer paths) already feeds. So there is no
  new record stream and no `replay_obs_guard` change.
- Pass the flag through the controller constructor/spec, next to the other obs flags.

**Metal (`shaders/controller_rollout.metal`).**
- Factor the roll/pitch part of the existing `tilt_ref_quat` (L375) into `inline float2 tilt_ref_rp(...)`, used by
  both the scorer (unchanged) and the feature twin at ~L852 (`P.target0/1` → `tilt_ref_rp(F.horiz_*)` when the new
  `P.att_target_tilt` flag is set).
- One function, two users: no duplicate law.
- Parity: add an `att_target` case to the existing CPU/GPU feature/rollout sweeps (`cargo test -p ram_controller
  --lib --no-default-features`). No sweep is run from Python.

**C twin (`mcu/wnn_cascade.c`).**
- `compute_attitude` (L192-193) reads `in->target`. Under `WNN_ATT_TARGET_TILT`, compute `target[0..1]` from
  `in->horiz` with `WNN_POS_BXY/OMEGA_N/ZETA/MAX_TILT` emitted into the model header.
- `scripts/export_controller_c.py` emits the macro and the four constants.
- Extend the `mcu/parity` fixture with an E1 case (Rust ↔ C feature vector on recorded ticks).
- Cost on the MCU: two multiply-adds, a clamp and one division per axis per tick. Negligible.

**Python (orchestration only).**
- `ControllerSpec.att_target: str = "level"`, serialized; old checkpoints load as level.
- CLI `--att-target {level,tilt}`, default `level`, next to `--att-err-ref` in `phased_ga.py`.
- Refuse `tilt` without `--translation`, `--obs-pos-err-xy` and `--obs-vel-xy` (SystemExit, mirroring
  `AttErrRef::require_horizontal`).
- Refuse `tilt` with `--obs-peraxis-yaw`. Not a correctness problem, but yaw keeps `target[2]`; this keeps the
  scope explicit.
- Spec builders: `phased_ga` (~L474/525) and `controller_grid_search._spec`. After CTRL-45 B2 there is one arch
  config builder.
- `feature_layout.py`: no change. Names and families are unchanged; optionally update the `tgt_r/tgt_p` comment.
- **Flows:** add `controller_att_target` to `wnn/ram/experiments/params.py` `KNOWN_PARAMS` and to
  `flow_adapter.py`'s cascade-only refusal tuple (same as `controller_obs_rate_i`). The adapter refuses CTBR, so
  there is no Svelte change.

**Provenance.**
- One startup line: `[att-target] tilt: tgt_r/p = PositionLoop(ωn=…, ζ=…, max_tilt=…)(e, v)`.
- The `[PLANT]` header gains `att_target=…`.
- The spec field lands in markers and checkpoints.

**ABI.** Controller ABI bump (a new spec/constructor field and a Metal Params field). No `ram_core` change, no
worker wheel.

### 4.2 Interactions to settle before building

1. **A's calibration.**
   - With E1, `tgt_r/tgt_p` and the tilt-referenced errors are zero-referenced, so A's `centred` ladder applies.
   - The calibration sampler must fly the horizontal channel. [C] e_x/e_y calibrate today with between-episode
     share 0.96 (CTRL-44b), so it does. Verify in the smoke that the `tgt_r` ladder is not degenerate.
2. **The constant-feature guard (CTRL-44b targeted fix 1)** must be DATA-driven (zero measured spread), not a
   hardcoded "drop tgt_*". Otherwise it would drop the very features E1 brings to life. `tgt_y` stays dead and the
   guard should still catch it.
3. **CTRL-45 RF:** no change. The families are unchanged and nf is unchanged.

### 4.3 The A/B (pre-registered)

**Recipe.**
- The CTRL-45 command line (MA args + chain sed + `--motor-lag-s 0.20 --output-mode ctbr --action-repeat 10
  --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off`).
- posx2 weights, `--test-seeds 99990101..99990135` (35 CF2.1 units), `--num-eval-folds 5`, pop 50.
- **Plus A's adopted calibration and CTRL-45's adopted RF setting.** If either has not been decided when E flies,
  use what the reference arm uses and record it. Both arms must use the same setting.
- `--att-err-ref tilt` (already in the recipe).

**Seeds.** A NEW set, s1..s3 = `--base-seed 31337001..31337003`. One wheel for both arms. Never paired with banked
rows from another wheel.

**Arms (2 × 3 = 6 runs, about 6 h serial):**

| arm | att target |
|---|---|
| ref | level (= the A + CTRL-45 baseline) |
| E1 | tilt |

**Saving 3 runs.** If E1 lands in the SAME wheel as A's own A/B, A's s1..s3 adopted-arm runs ARE `ref`. Only the 3
E1 runs are added. This is the strongest reason to bundle E1 into A's wheel (sequencing call for Luiz).

**Decision rule** (per seed: paired delta E1 − ref on the 35-unit headline; n = 3; mean ± t-CI with t = 4.30; the
attitude block judged as a whole):
- **WIN:** Δstable mean > 0 with 3/3 seeds ≥ 0 (or CI excluding 0), AND Δerr mean ≤ +0.10°, AND Δsteady mean ≤
  +0.10°.
- **ALSO WIN:** Δsteady mean ≤ −0.10° with 3/3 seeds ≤ 0, AND Δstable mean ≥ −1.0 pp, AND Δerr ≤ +0.10°. E1
  targets the hold, so steady is its primary lever.
- **LOSS:** Δstable ≤ −1.0 pp on 3/3 seeds, or err/steady worse by > 0.10° on 3/3 seeds.
- **TIE:** anything else. pos (pos3d) breaks the tie only when the attitude block is TIE and the deltas are of
  similar size (Luiz 05/11).
- **Adoption:** E1 becomes the cascade default only on WIN.
- **Kill:** LOSS closes E1. TIE closes E1 unless D7's HOLD gain was large (≥ 2/3 of the gap), in which case 3 more
  seeds (s4..s6) are run before closing. No dose arms; E1 has no dose.
- **Go/no-go after s1:** if E1's stable falls below 70% where ref is ≥ 80%, stop and report.
- **Per-bank secondary readout (not decisive):** the post-flight probe on both winners. HOLD p/q imitation R² of
  the RAM and the D4 oracle. Did the mechanism move as designed?

### 4.4 Build order and estimate (agent wall-clock)

| step | content | agent-h |
|---|---|---|
| 0 | probe extension D0–D7 + run on s2–s4 (section 2) | 3 |
| gate | decision tree 2.3; report to Luiz | — |
| 1 | Rust `att_target` + base-block substitution + cargo tests (teacher-equality pin, level-mode bit-identical pin) | 1.5 |
| 2 | Metal `tilt_ref_rp` factor + feature twin + parity sweep case | 1.5 |
| 3 | C twin + exporter constants + `mcu/parity` case | 1.5 |
| 4 | spec/CLI/refusals/KNOWN_PARAMS/adapter refusal/provenance lines + Python test (off-mode RNG/feature bit-identical) | 1 |
| 5 | pop-6 smoke (plumbing only), ladder-not-degenerate check, pre-deploy wheel review, chain script | 1 |
| | **total** | **about 9.5 agent-hours** + about 1 h probe CPU + 6 flight-hours (3 if bundled with A) |

---

## 5. Flags for Luiz (decide; do not let me decide)

1. **E1 puts the classical position loop's law into the input.** The student's attitude target becomes the
   PositionLoop tilt reference (the same law the teacher's cascade and the G5 scorer already use). The WNN still
   learns the whole attitude-to-rate mapping, disturbance handling included. But the paper must say the position→tilt
   map is computed, not learned. This is the "computed operation" pattern CLAUDE.md already names. Acceptable?
   (Paper-is-spec: if the unsubmitted paper says the WNN learns position control, this conflicts.)
2. **E2 with the DARE gain is label leakage in all but name.** I do not recommend it. If D2/D7 point to a combination
   other than the tilt reference, the choice of combination comes back to you, not to the build.
3. **E4 (hierarchical)** needs an architecture-integrity discussion before any design if it means anything other
   than the existing state layer.
4. **Sequencing:** bundling E1 into A's wheel saves 3 flight-hours and gives a shared baseline. It does mean A's wheel
   ships with one more default-off flag.
5. **Constant-feature guard ordering** (4.2 item 2): it must be data-driven, or it erases E1's features.
6. **The R² table is a 30° pool.** The scorer holds at 5°. I recommend that every future representation number is
   reported with the HOLD stratum alongside.

## 6. CLAUDE.md compliance

- No new RAM neuron or memory format. `Memory`, cells, decode and connectivity are unchanged. E1 is an input
  coordinate change in `compute_features`.
- One law, one implementation: `PositionLoop::tilt_ref`, plus its existing Metal twin factored into a shared inline
  and a C twin under parity.
- Every loop is in Rust. Python only plumbs a typed spec field. No kwargs, no globals.
- The full stack (Rust ↔ Metal ↔ C ↔ Python ↔ KNOWN_PARAMS/adapter) is specified. Svelte is unaffected because the
  adapter refuses CTBR.
- Both CPU and GPU paths are covered, and the parity test is in `cargo test`.
- K-fold 5 is kept.

## 7. Rejected alternatives

- **Treat the R² gap as an interaction problem without D3:** F1 shows the measured gap lies inside the additive class.
- **A tilt reference added as a NEW feature pair (keeping level-ref roll_err):** it adds features (dilution under
  spread), changes nf/layout/families/Metal sizes/MCU width, and leaves the 16 dead tgt bits dead. Substituting into
  the existing target slot gets the same information for free.
- **The host passing `target = tilt_ref` at every call site:** many hosts (CPU, Metal in-loop, Metal buffer, DAgger,
  calibration, MCU), so drift is likely. Deriving it inside `compute_features` from `horiz_obs` covers all of them.
- **Global `--bits-per-feature 16`:** it doubles the frame for every feature (dilution) and buys population
  diversity, not per-neuron resolution (F2).
- **Selecting a representation by offline R² alone:** refuted as a ranker (rho 0.3). Offline numbers only gate the
  build.
- **Python-side feature computation for the probe:** forbidden (no Python shortcuts). The probe is Rust.
