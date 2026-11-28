//! ram_controller — drone attitude-controller hot-path (paper #1).
//!
//! Split out of ram_accelerator on 2026-06-19 into its own wheel so a
//! controller-only change rebuilds ONLY this cdylib; the IDS/LM worker wheel
//! (ram_accelerator) keeps running untouched, so no worker swap is needed.
//! Links `ram_core` for the shared substrate (sparse Memory + GPU forward,
//! cell semantics, cooperative cancellation).

use pyo3::prelude::*;
use pyo3::types::PyModule;

mod actuator_lag; // ABI 34: the lag-aware teachers' actuator-state estimate
mod actuator_spec; // CTRL-86 S1/S2 (ABI 65): the plant actuator — asymmetric lag + idle floor
mod design_model; // CTRL-83 D1 (ABI 66): the controllers' design-time actuator model, held apart from the plant
mod payload_step; // CTRL-83 D3 (ABI 66): mid-episode payload (mass) step
mod stress_opts; // CTRL-83 (ABI 66): D1 + D3 as ONE scorer-side value on Stage1Cfg
mod train_lag_range; // CTRL-87 (ABI 68): per-episode TRAINING motor-lag spread (trainer + search scorers)
mod altitude_integrator; // L3 (ABI 51): altitude integrator in front of the altitude PD (--alt-integral)
mod altitude_pd; // scope C stage 1: outer altitude loop handing a collective to a teacher
mod box_qp; // exact box-QP solver for the MPC teachers (PDAS + primal active-set)
mod arch_ops; // architecture (connectivity) operators (counter_rng, Rust-first)
mod receptive_fields; // ABI 44 (CTRL-45): axis receptive-field projection (GA connectivity operator)
mod cell_mode;
mod vote_tally; // CTRL-46 (ABI 46): per-round DAgger vote tally storage (rule = ram_core WriteRule::Vote)
mod bptt_plan; // ABI 35: a bptt window's memory-independent inputs (CPU + GPU walk)
mod solver_view; // ABI 35 B: window-lived solver view (no per-record entry clone)
mod cell_remap; // cell ADDRESS remaps on architecture change (bit-exact port)
mod controller;
mod controller_split;
mod ctbr; // ABI 38 (CTRL-27): the cascade controller's output contract + inner stage
mod ctbr_axis_mpc; // ABI 39 (CTRL-33): the cascade outer MPC, one axis — model = the closed rate loop
mod ctbr_mpcof; // ABI 39 (CTRL-33): the cascade's MPCOF outer teacher (shadow rate loop + observer)
mod cascade_dhat; // CTRL-38 (ABI 41): the cascade MPCOF's offset-free observer, shared teacher/student
mod cascade_obs; // CTRL-38 (ABI 41): the cascade student's rate-I / d̂ input blocks
mod rate_err_lpf; // CTRL-69 (ABI 62): the cascade student's model-free rate-error input (one-pole low-pass)
mod controller_training;
mod cpu_score; // CPU (rayon) batch scorer — twin of score_controllers_metal
mod dagger_train;
mod episode_regime; // CTRL-17: ONE definition of a training episode (rollout, eval, calib, recorder)
mod calib_sampler; // CTRL-17 G1: threshold-calibration feature sampler on the training cascade
#[cfg(test)]
mod ctrl17_pins; // CTRL-17 byte-identity pins (attitude-only trainer, translation rollout)
#[cfg(test)]
mod ctrl17_tests; // CTRL-17 G1-G4 + G11 behaviour tests
mod estimator; // Mahony attitude estimator — Rust twin of wnn/control/estimator.py
mod genome_cells; // opaque Rust-side cell store (Stage B: cells never cross FFI hot paths)
mod memory_ops; // GA-Memory cell-value operators (counter_rng, Rust-first)
mod optimal; // LQR + MPC DAGGER teachers (hand-rolled, no deps)
mod overactuated;
mod pid_firmware; // firmware-sourced cascaded attitude PID (twin of wnn/control/pid_firmware.py)
mod position_loop; // scope C stage 2: outermost position loop handing a TILT REF to a teacher
mod position_score; // scope C stage 2: score the full-state cascade in METRES
mod record_ops; // reference-rollout recorders (address universe, input entropy)
#[cfg(test)]
mod abi32_tests; // ABI 32 behaviour + CPU/Metal parity (plant, G5, dx/dy, estimator)
#[cfg(test)]
mod abi35_tests; // ABI 35: bptt walk fixtures, CPU pins, CPU/GPU walk parity
#[cfg(test)]
mod lag_student_law_tests; // diagnostic: student output law under motor lag
mod imitation_probe; // diagnostic (WNN_IMIT_PROBE): imitation vs closed loop under motor lag
mod imitation_probe_ctbr; // diagnostic: the cascade (ctbr) half of imitation_probe
#[cfg(test)]
mod imitation_diag_ctbr; // CTRL-49 "E" diagnosis (test-only, WNN_IMIT_DIAG=1)
#[cfg(test)]
mod abi38_pins; // ABI 38: motor-level byte-identity pins captured on the ABI-37 build
#[cfg(test)]
mod train_asym_tests; // ABI 38: per-episode training motor asymmetry
#[cfg(test)]
mod ctrl38_pins; // CTRL-38: cascade byte-identity pins with both new blocks OFF (captured on ABI 40)
#[cfg(test)]
mod ctrl38_tests; // CTRL-38: rate-I / d̂ blocks — physics, train = deploy, CPU/GPU parity
#[cfg(test)]
mod ctrl39_tests; // CTRL-39: the MEMORY recorder reads the FLOWN command (train = deploy)
#[cfg(test)]
mod ctrl40_tests; // CTRL-40: Metal split train/record for the applied-actuator inputs (CPU parity)
#[cfg(test)]
mod ctrl50_yaw_tests; // ARM 3 (12/11/2026): --yaw-integral-leak (CPU features, CPU/GPU score + split parity)
#[cfg(test)]
mod ctrl51_flags_off_tests; // ABI 51: every new argument at its default is the ABI-50 path
#[cfg(test)]
mod ctrl51_hold_tests; // ABI 51: hold-window position column — recomputation, CPU/GPU rows
#[cfg(test)]
mod ctrl51_tilt_ref_tests; // L1 of block 10 (ABI 51): tilt-reference roll/pitch error — definition, replay, CPU/GPU score/split/walk parity
#[cfg(test)]
mod ctrl51_yaw_trim_tests; // L2y of block 10 (ABI 51): trim-shifted yaw error — definition, replay, CPU/GPU score/split/walk parity
#[cfg(test)]
mod ctrl52_yaw_trim_p_tests; // L2y' (ABI 52): proportional-only yaw trim shift, raw integral — definition, replay, CPU/GPU score/split/walk parity
#[cfg(test)]
mod ctrl54_yaw_heading_tests; // lever 1 (ABI 54): yaw heading from the student's own Mahony estimate
#[cfg(test)]
mod ctrl61_f32_estimator_tests; // ABI 61: f32 Mahony + heading — CPU == Metal bit for bit
#[cfg(test)]
mod ctrl62_rate_err_tests; // ABI 62 (CTRL-69): rate-loop inputs — rate error LPF + rate-I axis subsets
#[cfg(test)]
mod ctrl54_e2e_parity_tests; // CTRL-54: end-to-end CPU/GPU scorer parity on trained genomes
#[cfg(test)]
mod ctrl60_mid_window_tests; // ABI 60: the `mid` scorer row — window definition, err decomposition, diverged fallback, CPU/GPU parity
#[cfg(test)]
mod ctrl59_tail_gate_tests; // ABI 59 (CTRL-58): tail sink inert + capped + CPU==GPU; gate tests in ram_core::fitness
#[cfg(test)]
mod ctrl58_yaw_trim_ff_tests; // ABI 58: --yaw-trim-ff — cT, refusals, setpoint-only FF, label remainder, CPU/GPU score + walk parity
#[cfg(test)]
mod ctrl74_delta_leak_yaw_tests; // ABI 63 (CTRL-74): --delta-leak-yaw — refusals, bit-identity, yaw-only leak, decoder/label agreement, CPU/GPU score + walk parity
#[cfg(test)]
mod ctrl80_yaw_lead_tests; // ABI 64 (CTRL-80): --obs-yaw-lead — refusals, layout + definition, flag-off identity, replay == live, CPU/GPU score + split + walk parity
#[cfg(test)]
mod ctrl86_flag_off_pins; // CTRL-86 S1/S2: cf21 flag-off byte-identity pins (captured on ABI 64)
#[cfg(test)]
mod ctrl86_actuator_tests; // CTRL-86 S1/S2 (ABI 65): asymmetric lag + idle floor — hand steps, CPU/GPU parity, reach, refusals
#[cfg(test)]
mod ctrl83_flag_off_pins; // CTRL-83 (ABI 66): every stress/info flag at its default is the ABI-65 cf21 rollout
#[cfg(test)]
mod ctrl83_all_cell_tests; // CTRL-83 cell `all` (ABI 69): rivals fly the WNN scorers' per-episode lag — pins, semantics, CPU/GPU parity
#[cfg(test)]
mod ctrl87_train_spread_tests; // CTRL-87 (ABI 68): training lag spread — [0.20,0.20] pin, hosts, CPU/GPU parity, draw, nominal twin
#[cfg(test)]
mod ctrl83_stress_tests; // CTRL-83 (ABI 66): D1 mismatch keeps the models nominal, D2 anchor, D3 payload CPU/GPU parity
#[cfg(test)]
mod ctrl57_delta_max_yaw_tests; // ABI 57: --delta-max-yaw — refusals, bit-identity, yaw-only decode, label round trip, CPU/GPU score + walk parity
#[cfg(test)]
mod ctrl56_axis_split_tests; // ABI 56: per-axis split on the WNN CPU scorer — row bit-identity, err/steady decomposition, L3 pole3
#[cfg(test)]
mod ctrl55_yaw_heading_replay_tests; // lever 1 plumbing (ABI 55): replay stream, Metal score/train/record, CPU/GPU parity
#[cfg(test)]
mod ctrl51_alt_integral_tests; // L3 of block 10 (ABI 51): altitude integrator — law, anti-windup, droop, hosts, CPU/GPU score parity
#[cfg(test)]
mod ctrl46_tests; // CTRL-46 (ABI 46): vote write rule (CPU/GPU, order independence) + estimate-built labels
#[cfg(test)]
mod init_vxy_tests; // ABI 48: training-only initial horizontal velocity (draw order, validation)
#[cfg(test)]
mod scorer_chunk_tests; // 09/11: CPU scorer vs Metal across score chunks on the rf_c200_dres recipe (ep_vert global index)
#[cfg(test)]
mod ctrl49_tests; // ABI 49: the student feature sampler (--threshold-refit-from-student)
#[cfg(test)]
mod ctrl48_tests; // CTRL-48: the cascade unabsorbed-disturbance block — physics, replay, CPU/GPU parity
#[cfg(test)]
mod abi38_tests; // ABI 38: the cascade controller — identity, rivals, CPU/GPU parity, trainers
#[cfg(test)]
mod abi34_tests; // ABI 34: lag-aware teachers, one pwm convention under motor lag, T=0 pins
mod yaw_trim_err; // L2y (ABI 51): yaw error referenced to the trim the rate loop rides on (--obs-yaw-trim-err)
mod yaw_lead; // CTRL-80 (ABI 64): the yaw LEAD input yaw_err − τ·gyro_z (--obs-yaw-lead)
mod tilt_ref_err; // L1 (ABI 51): roll/pitch error vs the tilt reference, bias-corrected (--obs-tilt-ref-err)
mod att_bias_pred; // CTRL-46 (ABI 46): onboard predictor of the Mahony specific-force bias (--obs-att-bias-pred)
mod episode_tail; // ABI 59 (CTRL-58): the GA's near-fail tail (p99 of per-episode err, inverted_cdf)
mod axis_split; // ABI 53/56: per-axis split of a scorer's attitude error (rival + WNN scorers)
mod rival_probe; // opt-in probe on the classical-rival scorer (INDI spec §10: observer gain, bias compensation)
#[cfg(test)]
mod rival_probe_tests; // probe-off bit-identity + the #[ignore] val-unit sweeps of the INDI spec §10
#[cfg(test)]
mod label_replay_tests; // docs/ctrl_gap_after_block9.md §B.3: the teacher's label along a recorded STUDENT flight (#[ignore], val units)
mod att_estimator; // ABI 32: the WNN's Mahony attitude estimate (obs_att_estimator)
mod mahony_f32; // ABI 61: the WNN's Mahony filter + heading in f32, bit-identical to the Metal kernel
mod att_err_ref; // CTRL-17 G5 (ABI 32): attitude error vs the position loop tilt reference
mod plant_opts; // ABI 32: specific-force IMU + rotor drag as ONE value every sim host applies
mod stage1; // scope C stage 1 vertical-channel config (scorer parameter object) // Phase-0 N-rotor allocation substrate (not wired; docs/OVERACTUATED_RESIDUAL_DESIGN.md)

// ABI 35: GPU-resident sn>0 bptt walk host (macOS/Metal only).
#[cfg(target_os = "macos")]
mod bptt_walk_gpu;
// GPU-batched closed-loop controller eval (macOS/Metal only).
#[cfg(target_os = "macos")]
#[path = "metal_controller.rs"]
mod metal_controller;

/// ABI version of the controller wheel's Python surface. Mirrors
/// ram_accelerator's contract; wnn/control/_accel.py asserts it at import.
/// 3 = W2 disturbances (set_disturbance / disturbance_episode_seed /
///     score_controllers_metal + eval_ensemble_closed_loop dist args).
/// 4 = overactuated Phase 1: AttitudeSim.set_geometry/step_n/perturb_geometry/
///     set_rotor_asym + geometry=/rotor_asym= kwargs on
///     score_controllers_metal AND score_controllers_cpu (None = legacy quad).
/// 5 = overactuated Phase 2 step 1: AllocLqrRs (allocator-aware LQR teacher,
///     the N-rotor residual baseline / DAGGER label generator).
/// 6 = mono/jerk semantics UNIFIED in score_controllers_cpu (12/07/2026, Luiz
///     order): mono = last decision step per episode, jerk = per-episode mean
///     — the GPU kernel's aggregation. Fitness ranks differently than ≤5.
/// 7 = overactuated Phase 2 step 2: allocator-LQR residual baseline —
///     alloc_* kwargs on BOTH scorers (in-kernel alloc_step buffer 28 /
///     rollout_one composition), AllocBaseline precomputed-pinv path.
/// 8 = overactuated Phase 2 step 3: AttitudeSim.geometry_rows() exporter
///     (presets/perturbation built in Rust, table read back by Python).
/// 9 = allocation-effort metric (Phase 3 Σu² fitness input): scorer rows grow
///     12 → 13 ([.., ise, effort]); rollout floats bit-identical to 8.
/// 10 = effort SEMANTICS on alloc-residual runs: EXCESS thrust-effort vs the
///     pinv optimum for the same realized wrench (raw Σ pwm² was gameable by
///     collective shedding on the attitude-only sim). Raw metric unchanged
///     on non-alloc runs.
/// 11 = residual anchor = NEUTRAL_DECODE derived from cell semantics (QUAD
///     empty→0.75; ternary would give 0.5): untrained residual is EXACTLY 0.
///     Pre-11 anchored at 0.5 → hidden +clamp offset (E5 runs included).
/// 12 = memory-mode-aware controller (granularity ablation, Luiz 12/07/2026):
///     WnnController(memory_mode=) — TERNARY (empty_value=0.5, PLN convention)
///     + BINARY (classical WiSARD, antagonist-pair E/I output halves, decoded
///     = 0.5 + (ΣE−ΣI)/levels). Mode-derived neutral threads through decode /
///     delta / residual / DAGGER-bptt nudges on CPU AND the rollout+train
///     kernels (Params/TrainParams +memory_mode). split_train[_loop] is
///     QUAD-only (loud guard). Exports neutral_decode_for_mode(); QUAD paths
///     bit-identical to 11.
/// ABI 13 (18/07/2026, Phase-4 state-pressure): two STATEFUL DAGGER teachers —
///     lqi (id 3, integral-augmented LQR) and mpcof (id 4, offset-free MPC with
///     an input-disturbance observer fed by Teacher::observe in the rollout
///     loop); both expose integrals()/i_clamps() for the Option-A target. Plus
///     three new disturbance levers on Disturbance/AttitudeSim + the Metal twin:
///     D5 sensor dropout/freeze, D6 observation latency, D7 per-episode
///     torque-scale jitter (channels 5/6 appended; zero-default = bit-identical
///     to 12). RewardGatedConfigPacked + scorers gain the 4 dist fields.
/// ABI 14 (19/07/2026, single-layer promotion): state_neurons=0 is a first-class
///     config — bptt_train_window skips the state-serving QSR solves (direct
///     supervised output writes = the classic RAMLayer trainer), split_train_loop
///     no-ops (dagger falls back to the non-split path). RewardGatedConfigPacked
///     gains `expert_drives` (pure behavior cloning: the teacher's pwm drives the
///     sim; default false = bit-identical DAGGER). sn>0 paths bit-identical to 13.
/// ABI 15 (20/07/2026, memory): dagger_train_batch_inplace takes `fold_seeds:
///     Vec<Vec<u64>>` (was `seeds: Vec<u64>`) and runs the WHOLE K-fold accumulate
///     chain inside one rayon task, so cells never cross the FFI boundary between
///     folds. Adds WnnController::load_cells (bulk warm-start with exact
///     write_*_cell semantics — canonicalising, masked, bounds-checked; NOT
///     restore_cells, whose raw import stores default-valued cells) and
///     cell_fill_counts (per-neuron distinct-address tallies in Rust).
///     split_record emits state_ins_flat bit-packed in the Metal word layout.
///     All bit-identical to 14.
/// ABI 16 (20/07/2026, Rust-first): neighbor_search promoted to ram_core so BOTH
///     wheels can use it (the controller previously could not and grew a parallel
///     Python GA). Exposes ram_core::counter_rng — a counter-based, order-
///     independent RNG shared by both substrates and mirrored bit-for-bit in
///     wnn/ram/counter_rng.py. Nothing CONSUMES it yet, so 16 is bit-identical to
///     15; adopting it for the genome operators is a separate, versioned break.
/// ABI 17 (20/07/2026, LINEAGE BREAK): the controller MEMORY-cell operators moved
///     to Rust and the Python per-cell loops were DELETED (ga_memory mutate/
///     crossover, recurrent_genome _mutate_memory/crossover_memory). Per-cell
///     draws now come from ram_core::counter_rng instead of numpy PCG64, so
///     genome lineage is RE-BASED — results before and after are not comparable.
///     Adds memory_mutate_values / memory_crossover_values / memory_crossover_keyed
///     and LAYER_STATE / LAYER_OUTPUT.
/// ABI 18 (20/07/2026): reference-rollout recorders in Rust —
///     record_address_universe / record_input_entropy run the whole PID rollout
///     natively with Python-injected episode ICs. BIT-EXACT (ICs, sim, controller
///     and accumulation order all unchanged). The overactuated (allocator-LQR)
///     branch of the universe recorder is still Python.
/// ABI 19 (20/07/2026): the controller operator stack is 100%% Rust — the
///     overactuated (allocator-LQR) rollout driver, the five scalar per-genome
///     gates, and MEMORY genesis all moved. No Python code draws a random number
///     for a controller genome any more; numpy supplies per-call SEEDS only.
/// ABI 20 (21/07/2026, Stage B — cells live in Rust): GenomeCells, the opaque
///     cell store MemoryPayload now wraps. Cell remaps/mutation/crossover/
///     digest/validate run in place on the handle (cell_remap_* pyfunctions
///     from Stage A remain for parity tests). WnnController gains
///     load_cells_handle / export_cells_handle. BREAKING:
///     dagger_train_batch_inplace's init_state_cells_per_genome +
///     init_output_cells_per_genome (Vec<Vec<triple>>) are replaced by
///     init_cells_per_genome (Vec<GenomeCells>; empty handle = no warm-start).
///     Cell CONTENT and operator draws are bit-identical to 19 — only the
///     container moved, so lineage is preserved.
/// ABI 21 (21/07/2026): score_classical_baseline — held-out rollout of ONE
///     classical controller (PID/LQR/MPC/LQI/MPCOF via Teacher::from_id) under
///     the same sim + W2/W2.4 disturbance the WNN scorer uses, returning
///     (stable_rate, mean_err_deg, steady_deg). Additive; all else bit-identical
///     to 20. Lets a published comparison table come from ONE physics engine.
/// ABI 22 (06/08/2026): L1 `obs_dhat` — WnnController::new gains `dhat_b`
///     (Option<[f64;3]>, the plant's control effectiveness from
///     calibrate_control_gains_rs) and `dhat_l_gain`. Some(b) appends 3 features
///     (the mpcof teacher's disturbance estimate d̂, computed from the student's
///     OWN throttle accumulator and gyro finite-difference). None reproduces every
///     pre-L1 run bit-for-bit — the ctor CHANGED, hence the bump, but no existing
///     behaviour did. Also exposes `calibrate_control_gains` so Python never
///     re-derives b. Motivation: the D2 decomposition
///     (docs/l4_teacher_screen_results.md) showed the student's error is dominated
///     by holding attitude against an unobservable torque.
/// ABI 24 (21/08/2026): `gated_fitness_combine` — viability gate before the
///     weighted combine (Deb's rules; ram_core::fitness::gated_combine_flat).
///     Additive-only: fitness_combine and every other export are untouched, so
///     a facade pinned to >= 23 keeps working and banked recipes reproduce
///     bit-identically. The bump exists so the Python gate path can ASSERT the
///     wheel has the function instead of discovering mid-run that it does not.
/// ABI 26 (29/08/2026): address naming above 64 bits — ram_core names wide
///     tuples by a 64-bit mix of their words instead of the wrapping shift that
///     OR-folded slots i and i+64 (project_bits_above_64_or_fold). Bit-identical
///     at <= 64 bits (identity path), so every banked controller run reproduces;
///     the bump marks that the shared substrate changed and the Metal twin
///     (common.metal wnn_addr_*) must match the installed core.
/// ABI 27 (11/09/2026): D0 teacher hover anchoring —
///     `RewardGatedConfigPacked.teacher_hover_mode` (0 = legacy 0.5, 1 = derived
///     nominal hover √(m·g/4k), the scorer's own derivation) and the label
///     re-base label = neutral + (p − hover_teacher) it switches on
///     (dagger_train::teacher_label_f32 / rebased_label_base). Default 0 is
///     bit-identical to ABI 26 for every banked run; the bump lets the Python
///     side ASSERT the ctor accepts the new key instead of a TypeError mid-run.
/// ABI 29 (13/09/2026): obs_pwm fixed at THREE defects — the feature is now the
///     accumulator's DEVIATION from its bank anchor (raw pwm encoded the
///     per-episode collective jitter and split the memory by episode);
///     `ReplayObs` gains the per-step `pwm_dev` stream (the feature value
///     compute_features produced at rollout; recorded in TrajectoryRs.pwm_dev,
///     applied by every replay trainer: CPU bptt, CPU split, Metal train/record
///     buffers 25/22) and `WnnController.set_pwm_accumulator` (the threshold
///     fitter's hook, whose untrained feature controller made the ladder degenerate). Every
///     obs_pwm-off run is bit-identical to ABI 28; obs_pwm-on runs before this
///     (c2k, bit_sweep, e5, frame_fix, low_edge, arm B `_bd` s2) trained on a
///     frozen accumulator and a degenerate ladder and are VOID.
/// ABI 30 (19/09/2026): AXIS F actuator lag reaches EVERY place the plant is
/// stepped, not only score_controllers_metal (which had `motor_lag_s` since
/// 12/08 while nothing else did — a lagged cohort would have TRAINED lag-free
/// and SCORED lagged, the stage-1 trainer-gap pattern). New `motor_lag_s`
/// (Molchanov eq. 7 2% settling time T, s; 0.0 = OFF) on:
///   RewardGatedConfigPacked (training rollout + per-round eval, via
///   AirframeRs::sim), score_controllers_cpu, trace_controller_cpu,
///   score_classical_baseline, trace_classical_baseline,
///   eval_ensemble_closed_loop, score_position_teacher, record_address_universe.
/// Every default is 0.0 and the 0.0 path never calls set_motor_lag, so ABI 29
/// results reproduce bit-for-bit (pinned by the motor_lag_zero_* tests). The
/// bump exists so the Python facade can ASSERT the ctor accepts the key.
/// ABI 31 (26/09/2026): CTRL-17 stage-2 trainer fixes — RESULTS-CHANGING for
/// every translation run (a new lineage; attitude-only runs are byte-identical).
///   G1 new export `sample_calibration_features` (the thermometer fitter flies
///      the training cascade under translation);
///   G2 `record_address_universe(reference_cfg=)` REQUIRED with stage-1 draws
///      (recorder drives the training teacher + outer loops, applies xy starts);
///   G3 per-round checkpoint eval flies training episodes with live features and
///      ranks on the gate's weighted reward;
///   G4 RewardGatedConfigPacked.lambda_alt (new) and lambda_pos are READ by the
///      DAgger gate and the checkpoint (episode_regime::step_reward).
/// ABI 32 (31/10/2026): the real-life plant + stage-2 scoring (Luiz option A).
///   - `imu_specific_force` / `rotor_drag` (PlantOpts) on every host that builds a
///     sim: RewardGatedConfigPacked (→ AirframeRs: trainer, eval, calibration,
///     recorder), score_controllers_cpu/metal (Stage1Cfg.plant + the Metal twin),
///     score_classical_baseline, score_position_teacher. Translation-only.
///   - G5 `att_err_tilt_ref` (+ pos_omega/zeta/max_tilt on the scorers): per-step
///     attitude error vs the position loop's tilt reference (att_err_ref.rs) on
///     both scorers, the rivals, the position scorer and the DAgger gate/checkpoint.
///   - WnnController(obs_att_estimator=, est_kp=, est_ki=) + seed_att_estimator /
///     att_up_obs: the attitude features read a Mahony estimate; ReplayObs.att_up
///     (TrajectoryRs.att_up; Metal train/record buffers 26/23) replays it;
///     dagger_train_batch_inplace(obs_att_estimator=).
///   - scorer rows grow 15 → 17: [.., pos_err_m, alt_err_m, dx_m, dy_m]; the Metal
///     per-episode rows gain dx_m, dy_m at the end.
/// Every new option defaults OFF and the OFF path is bit-identical to ABI 31
/// (ctrl17_pins + every parity anchor unchanged); only the row width moved.
/// ABI 33 (01/11/2026): the mpc / mpcof teachers solve their box-constrained QP
///   EXACTLY (box_qp.rs: primal-dual active set + primal active-set fallback),
///   replacing projected FISTA capped at 200 iterations that exited AT the cap on
///   93-95% of constrained solves (cond(H) ≈ 499). RESULTS-CHANGING for every
///   mpc/mpcof run whose box binds (student-driven: ~99% of steps); label change
///   |Δpwm| mean 6e-5, p99 7.5e-4, max 5.3e-3. Unconstrained-path labels are
///   byte-identical (mpc_unconstrained_labels_bit_identical_to_fista_build); PID,
///   LQR, LQI unaffected. Python surface unchanged. ~16x faster per solve.
/// ABI 34 (05/11/2026): motor lag done right on CPU AND GPU (Luiz, 05/11).
///   - ONE pwm convention under --motor-lag-s: the observer's u (pwm_applied) is
///     the COMMANDED pwm on both sides — the Metal kernel stored it after the lag
///     filter while the CPU held the command (parity_lag_x_obs_pwm_dhat_cpu_vs_gpu).
///   - LAG-AWARE teachers on the TRUE T (ids 1-4; PID unchanged): LQR/LQI by
///     CARE on the actuator-augmented continuous model (Newton–Kleinman), MPC/MPCOF
///     on the augmented discrete model (doubling DARE terminal cost); the
///     actuator is estimated by filtering the applied pwm (actuator_lag.rs); the
///     episode's first decision is the law's fixed point under the plant's
///     seed-at-first-command filter.
///   - MPCOF observer subtracts b·(filtered actuator), not b·u_commanded.
///   RESULTS-CHANGING for every lagged run (teacher labels, rivals, GPU d̂
///   observer). T = 0 is bit-identical to ABI 33 (abi34_tests digests captured
///   on e56d6084; t0_gpu_lag_obs_rows_bit_identical_to_abi33). Python surface
///   unchanged.
///
/// ABI 35 (05/11/2026):
///   - GPU-resident sn>0 EDRA-BPTT walk (controller_bptt_walk): a genome's gated
///     round walks in-kernel over resident writable tables, bit-exact vs the CPU walk
///     (abi35_tests). Selected per process by WNN_CONTROLLER_GPU_BPTT ("1"/"0"; unset
///     ⇒ BPTT_DEVICE_DEFAULT) — phased_ga --bptt-device sets it.
///   - CPU walk quick wins (window-lived solver view, concurrent motor/state solves),
///     bit-identical to ABI 34 (digests captured on 72b8535e).
///   - The per-record GPU solve (WNN_CONTROLLER_GPU_SOLVE) is retired.
///   - Lag-aware LQR/LQI/MPC/MPCOF price the DELIVERED actuator + a fixed command
///     share (LAG_CMD_SHARE); student d-hat observer subtracts the LAGGED actuator
///     (new Python method set_observer_motor_lag). PLN fire-bit GPU twin fixed.
///   RESULTS-CHANGING for lagged runs only; T = 0 bit-identical to ABI 34.
///
/// ABI 36 (06/11/2026):
///   - Under motor lag the --obs-pwm feature reports the student's DELIVERED
///     actuator estimate u_a − anchor (WnnController::act_twin_next: the plant
///     filter's twin over the commands that FLEW, plant coefficient, seeded at the
///     first command), CPU and Metal score kernel; the twin advances whenever the
///     lag is on (it used to advance only for d-hat). Fixes the sn=0 collapse under
///     the airframe lag (lag_student_law_tests; metal abi36_* parity + pins).
///   RESULTS-CHANGING for lagged obs_pwm runs only; T = 0 and d-hat-only lagged
///   rows bit-identical to ABI 35 (abi36_t0_obs_pwm_rows_bit_identical_to_abi35).
///
/// ABI 37 (06/11/2026): two GPU SCORER bugs (controller_rollout.metal), present
/// since stage 1/2 — RESULTS-CHANGING for every GPU-scored --translation run:
///   - MAX_FEATURES was 21 while the production vector is 26: sensors[21..25]
///     were written past the thread-private array (silent UB) in the score,
///     train and record kernels. Now 40 (+ a 168-float ring cap), host-guarded
///     (kernel_feature_dims_ok).
///   - the score kernel fed --obs-collective-cmd the hover-relative FRACTION;
///     the trainer and the CPU scorer feed the ABSOLUTE collective pwm (since
///     ed921bac, 13/08/2026). Now the anchor, as on the CPU.
///   Evidence: pop-6 grid, same trained genomes, lag 0: GPU 6.58°/49% vs CPU
///   3.10°/94%; production-shaped fixture CPU/GPU rel 0.3-0.6 → 1e-6
///   (abi32_tests::parity_production_regime_cpu_vs_gpu). ≤ 21-feature, no
///   collective_cmd controllers are bit-identical to ABI 36.
///
/// ABI 38 (06/11/2026, CTRL-27):
///   - PER-EPISODE TRAINING MOTOR ASYMMETRY (Luiz, "train the same way,
///     asymmetric"): RewardGatedConfigPacked(dist_motor_asym_mag=) — every
///     training episode (DAgger rollouts, per-round eval, calibration sampler; the
///     MEMORY recorder flies the training weather when it is on) draws
///     dist_motor_asym·(1 + U(−mag, +mag)) per motor (controller::train_motor_asym,
///     ram_core::counter_rng). mag 0 ⇒ bit-identical to ABI 37 (abi38_pins,
///     train_asym_tests); mag > 0 is a LINEAGE BREAK for every L4C run
///     (provenance train_asym=per_episode:<mag>).
///   - THE CASCADE CONTROLLER (CTRL-27, Luiz 06/11; docs/ctrl_cascade_design.md):
///     WnnController(ctbr=CtbrSpec) — the 4 banks are [collective deviation, p, q,
///     r], decoded absolutely on a per-bank range/γ, held across action-repeat and
///     flown every tick through the firmware RATE PID (pid_firmware::RateStageFw,
///     sourced cf21bl gains) on the measured gyro; step() still returns 4 pwm.
///     Teachers: Teacher::ctbr_label (PID: the firmware attitude stage; mpcof: the
///     plan's body rate one outer period ahead). Every trainer (sn=0 direct write,
///     sn>0 EDRA-BPTT CPU + GPU walk, the state-split trainer CPU + GPU), the
///     calibration sampler, the recorder, expert_drives, the CPU and Metal scorers
///     and the rivals (score_classical_baseline(ctbr_outer_hold=)) support it.
///     dagger_train_batch_inplace(ctbr=). output_mode "pwm" is bit-identical to
///     ABI 37 (abi38_pins, abi38_gpu_pidfw_residual_rows pin).
///   - Bugs fixed alongside: the wheel did not build at 15d5bbf0 (imitation_probe
///     test-only loader); the GPU split ignored --delta-label-scale (trained the
///     unscaled label); two split-parity sweep cases planted nothing (vacuous).
///
/// ABI 39 (06/11/2026, CTRL-33): the cascade's MPCOF OUTER teacher is re-derived —
///   its prediction model IS the closed inner loop (firmware rate PID with the sourced
///   gains + the airframe's resolved motor lag + the rigid body), discretised at the
///   outer period (action_repeat ticks); its decision is the rate SETPOINT (per-axis
///   condensed MPC, DARE terminal, box = the bank rate range; ctbr_axis_mpc.rs) with
///   an offset-free observer on ω' and the plan regulated about the d̂ rest point;
///   state from a shadow of the rate loop + the measured lagged actuator
///   (ctbr_mpcof.rs). Replaces the ABI-38 label (the motor-level plan's rate N_o
///   ahead: MPCOF>rate 18.8% stable at lag 0.20). Python surface unchanged; the bump
///   separates the cascade+mpcof lineage. Motor-level teachers bit-identical to 38.
/// ABI 40 (06/11/2026, CTRL-34): under DAgger the cascade MPCOF teacher plans from the
///   FLOWN rate loop (Teacher::ctbr_sync_inner copies the student's real inner-loop state
///   before each label — the shadow had drifted 0.16–0.40 vs a real integrator ~0.15) and
///   the reward gate is selectable (--rg-gate improvement|quantile|off; default improvement
///   = 39). Cascade+mpcof labels change → new lineage; motor-level path bit-identical to 39.
/// ABI 41 (07/11/2026, CTRL-38): two default-OFF input blocks for the CASCADE student,
///   carried on CtbrSpec (obs_rate_i=, obs_dhat_plant=[arm, k_drag, Ixx, Iyy, Izz]) so
///   every host that builds a cascade gets them with no extra plumbing: a1 the inner
///   rate PID's integrator I/i_limit ×3, a2 the cascade MPCOF teacher's own d̂ observer
///   (cascade_dhat.rs, factored out of ctbr_mpcof.rs) on the applied command ×3 —
///   appended LAST in compute_features, mirrored in controller_rollout.metal. The
///   thermometer rows ride at the TAIL of `thresholds`. ReplayObs gains `cascade=`
///   (the per-step values a rollout recorded; the replay re-applies them). Both OFF
///   ⇒ the ABI-40 cascade and the motor path bit-identical (ctrl38_pins, abi38_pins).
/// ABI 42 (07/11/2026, CTRL-39): the MEMORY recorder (record_ops::run_episode) hands
///   the student the command that FLEW every step — observe_applied (every driver) and,
///   under obs_pwm, the accumulator proxy (the calibration sampler's protocol); under
///   expert_drives an obs_pwm student's accumulator is handed the flown command too.
///   Before, the motor-level obs_dhat observer and the obs_pwm feature (lagged twin and
///   lag-off accumulator) read the student's own discarded command (train ≠ deploy).
///   Motor-level obs_pwm/obs_dhat lineages MOVE; cascade (CTRL-38 off) and motor
///   students without those inputs are bit-identical (ctrl39_tests, ctrl38_pins,
///   abi38_pins). New Python method WnnController.observe_applied; module constant
///   RECORDER_INPUTS = "flown" (provenance recorder=).
/// ABI 43 (07/11/2026, CTRL-40): the Metal state-split train/record kernels carry the
///   student's applied-actuator inputs — the motor-level d̂ (TrainBatch.dhat_rec,
///   injected as the observer state with have_last = false) and the CTRL-38 cascade
///   blocks (cas_rec, written into the sensors tail) — re-applying the RECORDED values
///   like the CPU replay (ReplayObs.dhat, new; ReplayObs.cascade). obs_dhat now also
///   trains through the CPU state-split (the Fix-A refusal is gone), and dagger's
///   WNN_CONTROLLER_GPU_TRAIN gate no longer steps these lineages aside. The bptt
///   replay re-applies the recorded d̂ — equal to the Fix-A re-derivation bit for bit
///   (ctrl40_tests), so bptt lineages do not move. Flags off: bit-identical (pins).
/// ABI 44 (08/11/2026, CTRL-45 + increment-0 fixes): (a) BANK-AWARE output
///   neurogenesis — arch_output_segment_map (old -> new index, the output layer is
///   output_quantum equal segments = bank or E/I half; each resizes at its own tail)
///   and GenomeCells.reindex_output_neurons (cells follow their neuron). Any run whose
///   output level count changes (NEURONS stage, different-N crossover) MOVES; NEURO_LAYOUT
///   = "segment" (provenance neuro=). (b) Axis receptive fields: arch_project_receptive_fields
///   (receptive_fields.rs) — a host-side GA connectivity operator, no kernel change;
///   off (the default) draws nothing. Every runtime path is unchanged (pins).
/// ABI 45 (08/11/2026, CTRL-48, every item default-OFF ⇒ ABI 44 bit-identical):
///   (a) CtbrSpec.obs_dhat_residual_plant (`--obs-cascade-dhat-residual`): the cascade
///   block r = (d̂ + c·ki·I)/(c·ki·i_limit) per axis (cascade_obs.rs a3), computed from
///   the same recorded six values as rate-I / d̂ (record format unchanged), CPU + Metal
///   (score kernel cas_step, replay_applied_inputs; RolloutParams +7 / TrainParams +7);
///   (b) arch_project_receptive_fields treats a feature mask of 0 as EXCLUDED (never
///   kept, not even as cross) — the connection-space mechanism of `--drop-features` /
///   `--drop-constant-features`; the frame layout and every kernel are unchanged.
///   Calibration (`--calib-*`) is Python orchestration over the existing sampler.
/// ABI 46 (09/11/2026, CTRL-46, every item default-OFF ⇒ ABI 45 bit-identical):
///   (a) RewardGatedConfigPacked(write_vote=, vote_share=) — `--write-rule vote
///   [--vote-share θ]`: the DAgger walk tallies every visit of a round per (neuron,
///   address) and commits once (ram_core WriteRule::Vote / VoteShare /
///   vote_net_to_cell — the IDS-56 rule, shared), CPU walk + GPU walk (vote buffers
///   29/30 of controller_bptt_walk); QUAD refused (ram_core resolve_controller);
///   (b) RewardGatedConfigPacked(dagger_label_estimate=) — `--dagger-label-attitude
///   estimate`: the DAgger LABEL is built from the student's Mahony estimate, not the
///   true quaternion (labels are CPU-only);
///   (c) WnnController(obs_att_bias_pred=) + dagger_train_batch_inplace(obs_att_bias_pred=)
///   + ReplayObs(att_bias=) — `--obs-att-bias-pred`: +2 features at the layout tail,
///   the onboard predictor of the Mahony specific-force bias (att_bias_pred.rs), CPU +
///   Metal score (FC_ATT_BIAS_PRED, RolloutParams.abp_on) + Metal split replay
///   (TrainParams.abp_on, abp_rec buffers 29/26).
/// ABI 47 (09/11/2026, score-kernel FIX, NOT default-off): the Metal score kernel read
///   the per-episode stage-1/2 blob (buffer 30: z0, vz0, mass, collective, x0, y0) by
///   the CHUNK-local episode while that buffer is bound whole, so with translation on
///   and > episodes_per_chunk() (25) episodes every later chunk replayed episodes
///   0..24's translational draws (the CPU scorer never did). Fixed to the global index
///   (dist_ep_offset + e). Every GPU-scored translation row over > 25 episodes MOVES
///   to the CPU scorer's value; attitude-only runs and <= 25-episode calls are
///   bit-identical. No Python surface change. Pin: scorer_chunk_tests.rs.
/// ABI 48 (09/11/2026, docs/ctrl_dres_failures.md idea #4, default-OFF ⇒ ABI 47
///   bit-identical): RewardGatedConfigPacked(init_vxy=) — per-axis bound of a
///   U(-v, v) initial horizontal velocity, drawn in episode_regime::
///   draw_translation_episode right after x0/y0 (0 draws nothing). Python sets it on
///   the DAgger TRAINING config only (`--train-init-vxy`); scorers, calibration
///   sampler and recorder are untouched. Pin: init_vxy_tests.rs.
/// ABI 49 (09/11/2026, additive export, every existing path bit-identical):
///   sample_student_features(controller, cfg, num_episodes, seed) — the TRAINED
///   student's own feature values on the training episodes, flown through the
///   DAgger student rollout (calib_sampler::sample_student_features_rs). Replaces the
///   Python attitude-only collector for `--threshold-refit-from-student` on every
///   run whose teacher ladder comes from the training-cascade sampler (translation
///   + airframe): that collector refused the ABI-32 plant and the refit silently
///   fell back to the teacher ladder. Pin: ctrl49_tests.rs.
/// ABI 50 (12/11/2026, write-rule consolidation; NO Python surface change): the
///   controller's cell semantics (fire bit, nudge step / toward-value, plants,
///   nudge distance, coin prob, unwritten cell) are the shared ram_core cell_write.rs
///   primitives (+ common.metal wnn_* twins); controller/cell_mode.rs keeps only
///   decode. Default flags bit-identical (cargo parity sweeps + phased_ga goldens).
///   ONE behaviour change, GPU-split-train only (WNN_CONTROLLER_GPU_TRAIN=1) with
///   PLN: controller_train sent PLN through the QUAD nudge; it now LastWrites like
///   the CPU (parity_controller_train now sweeps TERNARY/BINARY/PLN).
///   Same ABI (nothing deployed in between): ARM 3 (Luiz 12/11/2026) additive kwarg
///   `yaw_integral_leak=None` on WnnController and dagger_train_batch_inplace (+ getter)
///   — a separate leak for the yaw integral features (yaw_err_i, yaw_i_pa), CPU and
///   Metal (Params/TrainParams/FwdParams gain a trailing float). None ⇒ integral_leak,
///   bit-identical. Pin: ctrl50_yaw_tests.rs. (ARM 2 --yaw-centred-ladder is Python-only:
///   thresholds reach Rust as data.)
/// ABI 51 (14/11/2026, CTRL-48 block 10 levers — docs/ctrl48_block10_levers_spec.md;
///   every addition default-OFF and bit-identical with the flags off: the pin suites
///   run through the new arguments at their defaults, + ctrl51_flags_off_tests.rs):
///   * L1 `obs_tilt_ref_err=None` on WnnController and dagger_train_batch_inplace
///     (+ getter): Some((gravity, pos_omega_n, pos_zeta, pos_max_tilt_rad)) measures
///     roll_err / pitch_err (and their integrals) against the position loop's tilt
///     reference, corrected by the predicted Mahony bias — the VALUE of two existing
///     inputs, no layout change (tilt_ref_err.rs). Needs obs_peraxis_p,
///     obs_att_bias_pred, obs_pos_err_xy, obs_vel_xy. CPU + the Metal rollout / train /
///     record kernels (Params / TrainParams gain 5 trailing words); a replay recomputes
///     it from the streams it already re-applies. Pin: ctrl51_tilt_ref_tests.rs.
///   * L3 `alt_integral=0` (0 off, 1 pi, 2 pole3) on RewardGatedConfigPacked (+ get),
///     score_classical_baseline, score_position_teacher, and — with `alt_pd_omega`,
///     `alt_pd_zeta`, `alt_pd_max_delta`, `alt_pd_mass` — on score_controllers_cpu /
///     score_controllers_metal: one altitude integrator per episode in front of the
///     unchanged altitude PD (altitude_integrator.rs), in the rivals, the DAgger label
///     and every teacher-flown rollout (OuterLoops), and as the student's `alt_err`
///     observation (CPU + the Metal rollout kernel; Params gains 8 trailing words).
///     Pin: ctrl51_alt_integral_tests.rs.
///   * Hold-window position column (output only): score_classical_baseline returns
///     10 values (+ hold 3-D position error, hold |dz|, m), score_position_teacher 7;
///     the WNN scorer rows grow 17 → 19 (rows 17, 18) and the Metal per-episode rows
///     13 → 15. Existing values and their order unchanged. Pin: ctrl51_hold_tests.rs.
/// ABI 52 (15/11/2026, Luiz, block 10 round 3; default-off, flags off = ABI 51):
///   * L2y' `obs_yaw_trim_err_p=False` on WnnController / dagger_train_batch_inplace: the
///     PROPORTIONAL yaw inputs carry yaw_err − κ·cresid_z, the yaw integrals keep the raw
///     heading error (yaw_trim_err.rs κ). Metal: FC_YAW_TRIM_P(4) specialization of the
///     cascade rollout and the controller_train / controller_record pipelines;
///     RolloutParams / TrainParams gain one trailing float (ytp_kappa).
///     Pin: ctrl52_yaw_trim_p_tests.rs.
///   * WNN hold rows (ABI 51 round 2): averaged over the episodes that reached the window.
/// ABI 53 (16/11/2026, CTRL-48 yaw floor; additive, default = ABI 52 bit-identical):
///   * `score_classical_baseline(axis_out=None)`: a Python list ⇒ the rival row is flown with
///     a measure-only RivalProbe and the per-axis split of its err / steady (rotation vector
///     of q_ref⁻¹·q_true, body frame; rival_probe::AxisSplit::flat, 14 values) is appended.
///     The returned 10-tuple is unchanged (rival_probe_tests pins probe-on == probe-off).
/// ABI 54 (16/11/2026, lever 1 heading estimate; additive, default = ABI 53 bit-identical):
///   * WnnController.set_yaw_from_att_est(bool) / .yaw_from_att_est / .yaw_heading_est: the
///     anchored yaw heading from the student's own Mahony quaternion instead of gyro_z·dt.
///     CPU only (score_controllers_metal refuses it). Pin: ctrl54_yaw_heading_tests.rs.
/// ABI 55 (16/11/2026, Luiz: lever 1 plumbing; additive, default = ABI 54 bit-identical):
///   * WnnController(obs_yaw_from_att_est=False) / dagger_train_batch_inplace(obs_yaw_from_att_est=):
///     the constructor / trainer twin of set_yaw_from_att_est.
///   * TrajectoryRs / ReplayObs.yaw_heading (+ ReplayObs(yaw_heading=)): the heading the rollout
///     used, applied verbatim by every replay (guarded). Metal: RolloutParams / TrainParams gain
///     a trailing u32 yfe_on (190 words), FwdParams yfe_on + yfe_yaw; controller_train /
///     controller_record buffer 30 / 27 = yh_rec. The ABI-54 Metal refusal is lifted.
///     Pin: ctrl55_yaw_heading_replay_tests.rs.
/// ABI 56 (17/11/2026, Luiz; additive, default = ABI 55 bit-identical):
///   * score_controllers_cpu(axis_out=None): a Python list ⇒ every controller's rollout also
///     accumulates the per-axis split (axis_split::AxisAcc, HoldPolicy::FallbackWhole = this
///     scorer's steady) and its 14 AxisSplit::flat values are appended in controller order
///     (classical_baseline.AXIS_FIELDS; slot 11 = the student's anchored heading drift, NaN
///     unanchored). CPU only — score_controllers_metal has no such argument. The ABI-53
///     rival probe now shares the same accumulator (rival_probe::RivalProbe.axes).
///     Pin: ctrl56_axis_split_tests.rs.
/// ABI 57 (17/11/2026, Luiz; additive, default = ABI 56 bit-identical):
///   * WnnController / dagger_train_batch_inplace(delta_max_yaw=None): the yaw bank's (m = 3)
///     own delta_max under the delta law (CTBR r / decouple τ_yaw); read through
///     delta_max_bank(m) by every encode (delta_label_target) and decode (decode_ctbr,
///     decode_outputs). Metal: RolloutParams + delta_max_yaw (191 words); the train/record
///     kernels read CPU-encoded labels. Pin: ctrl57_delta_max_yaw_tests.rs.
/// ABI 58 (17/11/2026, Luiz; additive, default = ABI 57 bit-identical):
///   * WnnController / dagger_train_batch_inplace(yaw_trim_ff=False): the yaw-rate FEED-FORWARD
///     sp_r = clamp(bank_r + cT·cresid_z, ±range_r) on the held setpoint (pwm[3] keeps the bank
///     value), cT = K_I·L (YawTrimErr::ff_gain; cf21 n_o 10: −0.39927); bank 3's DAgger label
///     recorded as sp_T − FF (one site: every split / bptt / GPU label path reads it). Metal:
///     RolloutParams + yff_on, yff_ct (193 words). Refused with L2y / L2y'.
///     Pin: ctrl58_yaw_trim_ff_tests.rs.
/// ABI 59 (18/11/2026, Luiz: CTRL-58 Rust part; additive, default = ABI 58 bit-identical):
///   * score_controllers_cpu / score_controllers_metal(episode_errs=False, tail_diverged_err_deg=8.0):
///     opt-in per-episode tail export — each genome row followed by [n, e_1..e_n] (deg, diverged =
///     the gate err limit); Metal builds it host-side from the per-episode export.
///   * episode_tail_p99_deg(errs): the GA tail column, numpy inverted_cdf p99 (episode_tail.rs).
///   * gated_fitness_combine(gate_steady=, gate_steady_max=, gate_pos=, gate_pos_max=): the four-input
///     gate, forwarded to THE gate ram_core::fitness::gated_combine_flat_bounds (one implementation;
///     gated_combine_flat is its no-extra-bound case, pinned by a golden over 2000 random pools).
///     Pin: ctrl59_tail_gate_tests.rs + episode_tail unit tests; ram_core fitness gate tests.
/// ABI 60 (19/11/2026, Luiz: the CTRL-58 A2 mid-window amendment; additive, default = ABI 59
/// bit-identical on every pre-existing row):
///   * score_controllers_cpu / score_controllers_metal rows: 19 → 20 metrics. Row 19 = `mid`,
///     the mean attitude err (rad) over ceil(0.15·steps) <= t < ceil(0.80·steps), per episode,
///     an episode that ended before the hold window contributing its whole-episode mean
///     (cpu_score::mid_window_start / mid_episode_mean; kernel out_effort slot 7, 8 slots).
///     The opt-in tail export still FOLLOWS the metric slots (now 20).
///     Pin: ctrl60_mid_window_tests.rs. Prereg: docs/ctrl58_a2_prereg.md (Amendment 19/11).
/// ABI 61 (22/11/2026, Luiz plan "A + B"; NOT bit-identical — a NEW LINEAGE for every run with
/// obs_att_estimator, i.e. every --translation run since ABI 32):
///   * the WNN's Mahony estimator (AttEstimator) is f32 (mahony_f32.rs MahonyF32), op for op the
///     Metal kernel's; the ABI-55 heading (obs_yaw_from_att_est, controller.rs mahony_heading) is
///     f32 through a deterministic atan2 (atan2_det, IEEE basic ops only). The kernel's
///     mahony_update / mahony_up / mahony_heading run in a strict-FP scope (safe math, contract
///     off) inside the fast-math rollout library, so the estimator's quaternion, up vector and
///     heading are CPU == GPU bit for bit (they were f64 vs f32 before: a CTRL-54 difference).
///     SYMMETRIC: every other simulated on-board estimate is the same MahonyF32 — the classical
///     rivals' estimator-fed rows (score/trace_classical_baseline, score_position_teacher via
///     mahony_f32::rival_estimator; hence the MPCOF eq-info bias predictor's q input) and the
///     Python-facing MahonyEstimatorRs — so WNN and rivals read bit-identical estimates (the
///     f64 MahonyFilter is test-only, the Python reference twin). Oracle-fed DAgger labels are
///     unchanged (they never read an estimate). New module constant
///     ATT_EST_PRECISION = "f32" (a wheel without it is f64); the MCU C port reads it through
///     the exporter (WNN_EST_F32). Pin: ctrl61_f32_estimator_tests.rs + mahony_f32 unit tests.
/// ABI 62 (22/11/2026, Luiz, Plane CTRL-69; additive, default = ABI 61 bit-identical):
///   * CtbrSpec(obs_rate_err_tau=, obs_rate_err_axes=): the MODEL-FREE cascade input a4 — the
///     flown firmware rate loop's tracking error e = rate_sp − gyro (pid_firmware.rs
///     PidCh::last_err, the P-term's error), / the CTBR bank range, through a one-pole
///     low-pass of time constant τ ticked per physical step (rate_err_lpf.rs). Emitted after
///     rate-I, before d̂ (the residual stays the cascade block's last slot). Recorded as its
///     own replay stream (ReplayObs.rate_err / TrajectoryRs.rate_err; on Metal appended to
///     the cas_rec device buffer at TrainParams.cas_rerr_off — the train kernel's 31 buffer
///     slots are all taken); the six-value cascade record is unchanged.
///   * CtbrSpec(obs_rate_i_axes=): rate-I on an axis subset (bitmask; 7 = all = ABI 61).
///   * Metal: both live in the FC_RATE_LOOP_OBS(5) specialization (score, train, record);
///     the LPF step is strict-FP (CPU == GPU == MCU C on equal inputs). A teacher-flown
///     rollout lends the flown loop when either rate-loop input is on
///     (cascade_reads_flown_rate). Pin: ctrl62_rate_err_tests.rs + rate_err_lpf unit tests.
/// ABI 63 (25/11/2026, Luiz, Plane CTRL-74; additive, default = ABI 62 bit-identical):
///   * WnnController / dagger_train_batch_inplace(delta_leak_yaw=None): the yaw bank's (m = 3)
///     own OUTPUT-accumulator leak under the delta law (CTBR r / decouple τ_yaw), so the WNN can
///     HOLD a per-unit yaw-rate trim instead of rebuilding it through --delta-leak every action
///     (docs/ctrl73_gap_analysis_gt.md L1). Read only through delta_leak_bank(m) inside
///     leaked_baseline, i.e. by every decode (decode_ctbr, decode_outputs) AND by the DAgger
///     label baseline (label_baselines) — one leak for decoder and label. Refused without the
///     delta law or a yaw bank, or outside (0, 1]. Metal: RolloutParams + delta_leak_yaw (200
///     words), kernel m==3 select at the CTBR and decouple accumulators; the population must
///     agree; the train/record kernels read CPU-encoded labels. MCU C: WNN_DELTA_LEAK_YAW.
///     Pin: ctrl74_delta_leak_yaw_tests.rs.
/// ABI 64 (26/11/2026, Luiz, Plane CTRL-80; additive, default = ABI 63 bit-identical):
///   * WnnController / dagger_train_batch_inplace(obs_yaw_lead=None): the yaw LEAD input
///     yaw_lead = yaw_err − τ·gyro_z (yaw_lead.rs; τ seconds, fixed) — where the proportional
///     yaw input will be in τ at the present rate. ONE feature directly after the yaw-anchor
///     pair (yaw_err, yaw_err_i), so its thermometer rows sit in the MIDDLE of the ladder and
///     every later feature moves up one slot. Needs obs_yaw_err. A fixed linear combination
///     of two inputs the student already reads: no observer, no plant model, no new replay
///     stream (gyro and the heading are already replayed). Metal: FC_YAW_LEAD(6)
///     specialization of the rollout (alone it selects the cascade-obs pipeline map) and of
///     controller_train / controller_record; RolloutParams (201 words) / TrainParams gain one
///     trailing float (ylead_tau); the kernel's yaw_lead_value is strict-FP (CPU == GPU on
///     equal inputs). Refused on the N-rotor geometry pipeline and on a mixed batch. The MCU
///     C port does not carry it (export_controller_c.py refuses). Pin: ctrl80_yaw_lead_tests.rs
///     + yaw_lead unit tests.
/// ABI 65 (27/11/2026, Luiz, Plane CTRL-86 S1+S2 — one bump for both; additive, default =
/// ABI 64 bit-identical):
///   * S1 ASYMMETRIC MOTOR LAG: motor_lag_down_s (s, the spin-down 2% settling time; 0 ⇒
///     symmetric) on RewardGatedConfigPacked, score_controllers_cpu/metal,
///     score_classical_baseline and score_position_teacher, carried by PlantOpts
///     (translation-only, like the rest of the real-life plant). Per motor per step the plant
///     filter uses T_up when the command is above the filtered value, T_down otherwise
///     (actuator_spec.rs). Every twin follows the plant's ActuatorSpec: the teachers'
///     ActuatorLag goes PER MOTOR when extended (up/down switching breaks the per-axis
///     linearity), the cascade d̂ and the student's actuator twin switch per motor too; the
///     linear teachers DESIGN at the mean of up and down (amendment 1).
///   * S2 IDLE FLOOR: pwm_min (0 ⇒ none) — the plant clamps every motor command to
///     [pwm_min, 1] before the lag; the twins clamp the command they filter the same way.
///   * Metal: RolloutParams (203 words) gains motor_settling_time_down_s + pwm_min; the
///     rollout's plant filter, actuator twin and cascade d̂ twin switch per motor.
///     Pins: ctrl86_flag_off_pins.rs (cf21 l2b-shaped regime, captured on ABI 64, both
///     spellings) + ctrl86_actuator_tests.rs (hand-computed steps, CPU/GPU parity).
/// ABI 66 (28/11/2026, Luiz, CTRL-83 stress-panel fairness — one bump for the batch;
/// additive, every default = ABI 65 bit-identical, pinned by ctrl83_flag_off_pins.rs):
///   * D1 DESIGN-MODEL OVERRIDE: model_motor_lag_s (None ⇒ the plant's lag) on
///     score_controllers_cpu/metal and score_classical_baseline — the lag every controller's
///     MODEL is built on (the rivals' lag-aware designs and observer twins; the WNN's
///     actuator twin), held apart from the plant's (design_model.rs). Metal: the twins read
///     Params.twin_settling_time_s.
///   * D2 NOMINAL-MASS ANCHOR (opt-in, default = the TRUE-mass anchor): anchor_mass on both
///     WNN scorers (Stage1Cfg::anchor_at_mass rebases the commanded fraction once, so CPU,
///     Metal and the recorder agree) and anchor_nominal_mass on RewardGatedConfigPacked (the
///     trainer + calib sampler + recorder via episode_regime). A lineage break when on.
///   * D3 PAYLOAD STEP: payload_step_s / payload_mass_scale on both WNN scorers and the
///     rivals — the plant mass becomes draw × scale at that step (payload_step.rs).
///     Metal: Params.payload_step_idx / payload_mass_scale (RolloutParams 203 → 206 words).
/// - 68 (28/11/2026, Luiz, CTRL-87 l2bnmw; additive, default = ABI 66 bit-identical; 67 is
///   claimed by the per-bank-width branch): the per-episode TRAINING motor-lag spread
///   (train_lag_range.rs). `train_motor_lag_range=None` on RewardGatedConfigPacked — each
///   training episode (DAgger rollouts, per-round eval, calibration sampler, recorder;
///   arm_cfg_disturbance) flies T_e ~ U(lo, hi) from its weather seed — and on
///   score_controllers_cpu / score_controllers_metal for the GA SEARCH pools (T_e drawn
///   host-side from disturbance_episode_seed(dist_seed, e), Stage1Cfg.ep_motor_lag; Metal:
///   buffer 30 widens to 7 floats/episode under the FC_EP_LAG(8) specialization; RolloutParams unchanged at 206
///   words). The teachers and the student's actuator twin stay at the nominal motor_lag_s.
///   TEST / VAL / report calls never pass it. Training wear spread (--train-motor-asym-mag)
///   is Python-only (dist_motor_asym_mag already takes any value in [0, 1)).
/// - 69 (28/11/2026, Luiz, CTRL-83 cell `all` = the single joint-OOD headline; additive, default
///   = ABI 68 bit-identical): `train_motor_lag_range=None` on score_classical_baseline too, so
///   the rivals (and the row-B cargo probe, fixture key `train_motor_lag_range`) fly the SAME
///   per-episode plant lag T_e ~ U(lo, hi) the WNN scorers fly — one draw function
///   (train_lag_range::scorer_episode_lags), episode e keyed on disturbance_episode_seed(dist_seed, e).
///   The rival's model keeps motor_lag_s / model_motor_lag_s (D1). Used at REPORT time by the
///   stress cell only; the trainer is untouched.
pub const ABI_VERSION: u32 = 69;

/// ABI 61: the precision of the WNN's own attitude estimator + heading ("f32"; a wheel
/// without this constant runs the f64 estimator). Read by scripts/export_controller_c.py.
pub const ATT_EST_PRECISION: &str = "f32";

/// ABI 44: how output neurogenesis lays out the output layer — "segment" = every
/// bank / E-I half resizes at its own tail (pre-44 Python resized the flat tail).
pub const NEURO_LAYOUT: &str = "segment";

/// CTRL-39: what the reference recorders feed the student's applied-actuator inputs.
/// "flown" (ABI ≥ 42) = the command the plant received; a wheel without the constant
/// predates the fix (its MEMORY universes read the student's own command).
pub const RECORDER_INPUTS: &str = "flown";

/// Where the sn>0 trainer walks when WNN_CONTROLLER_GPU_BPTT is unset ("cpu"|"gpu").
/// Decided by the item-E measurement rule (05/11/2026).
pub const BPTT_DEVICE_DEFAULT: &str = "cpu";

/// Mode-aware untrained-cell decode anchor (ABI 12): QUAD→0.75, TERNARY→0.5
/// (the fixed PLN empty_value), BINARY→0.5 (antagonist-pair effective neutral).
#[pyfunction]
fn neutral_decode_for_mode(memory_mode: u8) -> PyResult<f32>
{
	cell_mode::validate_mode(memory_mode).map_err(pyo3::exceptions::PyValueError::new_err)?;
	Ok(cell_mode::neutral_decode(memory_mode))
}

// ---- counter_rng bridge (ram_core) -----------------------------------------
// Exposed so the Python mirror (wnn/ram/counter_rng.py) can be proven identical
// draw-for-draw. These are NOT a Python draw API — operators belong in Rust; the
// mirror exists to verify that moving them there does not change what a draw is.

#[pyfunction]
fn counter_rng_draw_u64(
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
	index: u64,
	sub: u64,
) -> u64
{
	ram_core::counter_rng::draw_u64(seed, generation, genome, layer, index, sub)
}

#[pyfunction]
fn counter_rng_uniform(
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
	index: u64,
	sub: u64,
) -> f64
{
	ram_core::counter_rng::uniform(seed, generation, genome, layer, index, sub)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn counter_rng_below(
	n: u64,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
	index: u64,
	sub: u64,
) -> u64
{
	ram_core::counter_rng::below(n, seed, generation, genome, layer, index, sub)
}

/// GA-Memory value mutation, one FFI call for a whole layer. Replaces the
/// per-cell Python loop (~10^9 interpreter iterations per production run, each
/// with a numpy rng.random()). Uses the shared counter RNG, so results differ
/// from the numpy path BY DESIGN — this is the opt-in lineage break.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn memory_mutate_values(
	values: Vec<u8>,
	quad: bool,
	rate: f64,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> Vec<u8>
{
	let mut v = values;
	memory_ops::mutate_values(&mut v, quad, rate, seed, generation, genome, layer);
	v
}

// ---- cell address remaps (architecture change) -----------------------------
// Column form in, column form out: (neurons, addrs, values). Bit-exact ports of
// recurrent_genome._remap_* / _drop_*; see cell_remap.rs for why output ORDER
// and the majority tie-break are load-bearing. Overflow past u64 is raised as
// OverflowError to match what MemoryPayload does with the Python bigint result.

type PyCells = (Vec<u32>, Vec<u64>, Vec<u8>);

fn cells_or_overflow(r: Result<cell_remap::Cells, cell_remap::AddrOverflow>) -> PyResult<PyCells>
{
	r.map_err(|e| {
		pyo3::exceptions::PyOverflowError::new_err(format!(
			"cell address {} exceeds u64 after remap",
			e.0
		))
	})
}

#[pyfunction]
fn cell_remap_grow(neurons: Vec<u32>, addrs: Vec<u64>, values: Vec<u8>, d: u32)
	-> PyResult<PyCells>
{
	cells_or_overflow(cell_remap::remap_grow(&neurons, &addrs, &values, d))
}

#[pyfunction]
fn cell_remap_shrink(neurons: Vec<u32>, addrs: Vec<u64>, values: Vec<u8>, d: u32) -> PyCells
{
	cell_remap::remap_shrink(&neurons, &addrs, &values, d)
}

#[pyfunction]
fn cell_remap_prefix_grow(
	neurons: Vec<u32>,
	addrs: Vec<u64>,
	values: Vec<u8>,
	k: u32,
	w: u32,
	pf: u32,
) -> PyResult<PyCells>
{
	cells_or_overflow(cell_remap::remap_prefix_grow(
		&neurons, &addrs, &values, k, w, pf,
	))
}

#[pyfunction]
fn cell_remap_prefix_shrink(
	neurons: Vec<u32>,
	addrs: Vec<u64>,
	values: Vec<u8>,
	k: u32,
	w: u32,
	pf: u32,
) -> PyCells
{
	cell_remap::remap_prefix_shrink(&neurons, &addrs, &values, k, w, pf)
}

#[pyfunction]
fn cell_remap_delete_bit_window(
	neurons: Vec<u32>,
	addrs: Vec<u64>,
	values: Vec<u8>,
	p_lsb: u32,
	nbits: u32,
) -> PyCells
{
	cell_remap::remap_delete_bit_window(&neurons, &addrs, &values, p_lsb, nbits)
}

#[pyfunction]
fn cell_drop_neurons_ge(neurons: Vec<u32>, addrs: Vec<u64>, values: Vec<u8>, limit: u32)
	-> PyCells
{
	cell_remap::drop_neurons_ge(&neurons, &addrs, &values, limit)
}

#[pyfunction]
fn cell_drop_changed_neurons(
	neurons: Vec<u32>,
	addrs: Vec<u64>,
	values: Vec<u8>,
	changed: Vec<u32>,
) -> PyCells
{
	cell_remap::drop_changed_neurons(&neurons, &addrs, &values, &changed)
}

#[pyfunction]
fn cell_majority(values: Vec<u8>) -> u8
{
	cell_remap::majority(&values)
}

/// Uniform per-cell crossover over two index-aligned value vectors.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn memory_crossover_values(
	a: Vec<u8>,
	b: Vec<u8>,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> PyResult<Vec<u8>>
{
	if a.len() != b.len()
	{
		return Err(pyo3::exceptions::PyValueError::new_err(format!(
			"crossover needs index-aligned parents, got {} and {}",
			a.len(),
			b.len()
		)));
	}
	Ok(memory_ops::crossover_values(
		&a, &b, seed, generation, genome, layer,
	))
}

/// Random initial cell values for a MEMORY-phase genome (counter RNG).
#[pyfunction]
fn memory_random_values(
	n: usize,
	hi: u8,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> Vec<u8>
{
	memory_ops::random_values(n, hi, seed, generation, genome, layer)
}

/// Address-KEYED uniform crossover of cell values (MEMORY phase). Handles
/// different-shaped parents: the child keeps a's universe and adopts b's value
/// only where b holds the same (neuron, address).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn memory_crossover_keyed(
	a_neurons: Vec<u32>,
	a_addrs: Vec<u64>,
	a_values: Vec<u8>,
	b_neurons: Vec<u32>,
	b_addrs: Vec<u64>,
	b_values: Vec<u8>,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> PyResult<Vec<u8>>
{
	if a_neurons.len() != a_addrs.len() || a_neurons.len() != a_values.len()
	{
		return Err(pyo3::exceptions::PyValueError::new_err(
			"parent a arrays must be equal length",
		));
	}
	if b_neurons.len() != b_addrs.len() || b_neurons.len() != b_values.len()
	{
		return Err(pyo3::exceptions::PyValueError::new_err(
			"parent b arrays must be equal length",
		));
	}
	Ok(memory_ops::crossover_values_keyed(
		&a_neurons, &a_addrs, &a_values, &b_neurons, &b_addrs, &b_values, seed, generation, genome,
		layer,
	))
}

/// Per-entry resample of a sampled suffix, preserving distinctness (8-try retry).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn arch_resample_suffix(
	suffix: Vec<i64>,
	space: usize,
	rate: f64,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> Vec<i64>
{
	let mut s = suffix;
	arch_ops::resample_suffix(&mut s, space, rate, seed, generation, genome, layer);
	s
}

/// k distinct indices in [0, space) avoiding `exclude`, without replacement.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn arch_sample_distinct(
	space: usize,
	k: usize,
	exclude: Vec<i64>,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> Vec<i64>
{
	arch_ops::sample_distinct(space, k, &exclude, seed, generation, genome, layer)
}

/// Scoped axonogenesis: per-connection resample where the replacement stays in
/// the original bit's range. scope 0=free (legacy, bit-identical), 1=window,
/// 2=feature. See arch_ops::resample_suffix_scoped.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn arch_resample_suffix_scoped(
	suffix: Vec<i64>,
	space: usize,
	rate: f64,
	scope: u32,
	frame_bits: usize,
	bpf: usize,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> Vec<i64>
{
	let mut s = suffix;
	arch_ops::resample_suffix_scoped(
		&mut s, space, rate, scope, frame_bits, bpf, seed, generation, genome, layer,
	);
	s
}

/// One fresh suffix under MIN_PER_CLUSTER(m); m=1 = full feature coverage.
/// The unsatisfiable-request fallback to spread lives in Rust.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn arch_sample_min_per_cluster(
	space: usize,
	width: usize,
	bpf: usize,
	m: usize,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> Vec<i64>
{
	arch_ops::sample_min_per_cluster(space, width, bpf, m, seed, generation, genome, layer)
}

/// One fresh FRAMED1 suffix: frame-pure + min1 within the frame. slot < 0 =
/// recency-weighted draw (2^s, neurogenesis); slot >= 0 = caller-supplied quota.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn arch_sample_framed1(
	space: usize,
	width: usize,
	bpf: usize,
	k: usize,
	slot: i64,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> Vec<i64>
{
	arch_ops::sample_framed1(space, width, bpf, k, slot, seed, generation, genome, layer)
}

/// EXACT frame-slot quotas for a fresh framed1 population (largest-remainder
/// over 2^s per motor block, shuffled within block).
#[pyfunction]
fn arch_framed1_slot_schedule(
	n_neurons: usize,
	k: usize,
	quantum: usize,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> Vec<i64>
{
	arch_ops::framed1_slot_schedule(n_neurons, k, quantum, seed, generation, genome, layer)
}

/// Feature-balance cap over a flat `sampled` with per-neuron `offsets`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn arch_rebalance_features(
	sampled: Vec<i64>,
	offsets: Vec<usize>,
	space: usize,
	frame_bits: usize,
	bpf: usize,
	ratio: f64,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> Vec<i64>
{
	let mut s = sampled;
	arch_ops::rebalance_features(
		&mut s, &offsets, space, frame_bits, bpf, ratio, seed, generation, genome, layer,
	);
	s
}

/// Output-neurogenesis role map (bank-growth fix): old index -> new index (-1 =
/// dropped) when the output layer goes old_n -> new_n neurons in `segments`
/// (= output_quantum) equal blocks. See arch_ops::output_segment_map.
#[pyfunction]
fn arch_output_segment_map(old_n: usize, new_n: usize, segments: usize) -> PyResult<Vec<i64>>
{
	arch_ops::output_segment_map(old_n, new_n, segments).map_err(pyo3::exceptions::PyValueError::new_err)
}

/// CTRL-45 axis receptive fields over a flat output-suffix list (`offsets`, len =
/// neurons + 1): returns (projected flat suffixes, changed neuron indices). Neuron n
/// belongs to bank n / levels; see receptive_fields.rs.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn arch_project_receptive_fields(
	sampled: Vec<i64>,
	offsets: Vec<usize>,
	space: usize,
	frame_bits: usize,
	bpf: usize,
	levels: usize,
	bank_family: Vec<u8>,
	feature_family: Vec<u8>,
	max_cross: usize,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
) -> PyResult<(Vec<i64>, Vec<u32>)>
{
	let mut s = sampled;
	let layout = receptive_fields::RfLayout {
		space,
		frame_bits,
		bpf,
		levels,
		bank_family: &bank_family,
		feature_family: &feature_family,
		max_cross,
	};
	let draw = receptive_fields::RfDraw { seed, generation, genome, layer };
	let changed = receptive_fields::project_receptive_fields(&mut s, &offsets, &layout, draw)
		.map_err(pyo3::exceptions::PyValueError::new_err)?;
	Ok((s, changed))
}

/// n independent fair coins (per-position / per-block parent picks).
#[pyfunction]
fn arch_pick_mask(n: usize, seed: u64, generation: u64, genome: u64, layer: u64) -> Vec<bool>
{
	arch_ops::pick_mask(n, seed, generation, genome, layer)
}

/// Reference-rollout recorders. ICs are pre-drawn in Python (numpy PCG64) and
/// injected, the established parity convention for episode ICs — so these are
/// BIT-EXACT ports of the Python loops, not merely equivalent ones.
/// `geometry_rows` / `alloc_*` select the reference driver: None => the PID quad
/// path; Some => the allocator-LQR on the TRUE rotor table, driven through
/// step_n (the overactuated path, which used to be the last Python rollout).
///
/// `af_*` describe the aircraft the reference rollout flies. They default to the
/// legacy synthetic plant, so omitting them reproduces every universe recorded
/// before 13/08/2026 bit-for-bit. `s1_*` enable the SCOPE C STAGE 1 vertical
/// channel: without them the recorder flies a non-translating plant, the three
/// vertical features never move, and the recorded universe covers exactly one
/// degenerate slice of the space the controller will actually visit — the
/// defect that made the MEMORY phase meaningless for stage 1.
/// CTRL-17 G2 (26/09/2026): `reference_cfg` is the run's RewardGatedConfigPacked
/// and is REQUIRED whenever the stage-1 draws are passed. The recorder then flies
/// the TRAINING regime — the plant (airframe, motor lag) from that config, the
/// run's teacher inside the same outer altitude/position loops the DAgger labels
/// come from, the horizontal starts applied, live horizontal observations — rather
/// than the legacy 0.5-hover attitude PID (which does not hover a real airframe).
/// With reference_cfg the plant comes from it ALONE: af_*/motor_lag_s must be left
/// at their defaults (one owner per plant parameter).
#[pyfunction]
#[pyo3(signature = (controller, init_q, init_om, target, steps,
                    geometry_rows = None, nominal_rows = None, rotor_asym = None,
                    inertia = [0.0023, 0.0023, 0.0046],
                    q_att = 12.0, q_rate = 1.0, r_ctrl = 1.0, tau_max = 0.144,
                    f_hover = None, pinv_lambda = 1e-6,
                    af_dt = 0.001, af_arm_length = 0.075, af_k_thrust = 2.4,
                    af_k_drag = 0.05, af_inertia = [0.0023, 0.0023, 0.0046],
                    af_gravity = 9.81,
                    s1_target_altitude = None, s1_init_z = None, s1_init_vz = None,
                    s1_mass = None, s1_collective_frac = None,
                    s2_init_x = None, s2_init_y = None,
                    motor_lag_s = 0.0,
                    reference_cfg = None))]
#[allow(clippy::too_many_arguments)]
fn record_address_universe(
	mut controller: PyRefMut<'_, controller::WnnController>,
	init_q: Vec<[f32; 4]>,
	init_om: Vec<[f32; 3]>,
	target: [f32; 3],
	steps: usize,
	geometry_rows: Option<Vec<[f32; 9]>>,
	nominal_rows: Option<Vec<[f32; 9]>>,
	rotor_asym: Option<Vec<f32>>,
	inertia: [f32; 3],
	q_att: f64,
	q_rate: f64,
	r_ctrl: f64,
	tau_max: f64,
	f_hover: Option<f64>,
	pinv_lambda: f32,
	af_dt: f32,
	af_arm_length: f32,
	af_k_thrust: f32,
	af_k_drag: f32,
	af_inertia: [f32; 3],
	af_gravity: f32,
	s1_target_altitude: Option<f32>,
	s1_init_z: Option<Vec<f32>>,
	s1_init_vz: Option<Vec<f32>>,
	s1_mass: Option<Vec<f32>>,
	s1_collective_frac: Option<Vec<f32>>,
	s2_init_x: Option<Vec<f32>>,
	s2_init_y: Option<Vec<f32>>,
	// AXIS F (19/09/2026): the MEMORY-phase address universe is recorded on the
	// same lagged plant the run trains and scores on. 0.0 = OFF, bit-identical.
	motor_lag_s: f32,
	reference_cfg: Option<dagger_train::RewardGatedConfigPacked>,
) -> PyResult<(Vec<(usize, u64)>, Vec<(usize, u64)>)>
{
	let n_eps = init_q.len().min(init_om.len());
	let s1_cfg = recorder_stage1_cfg(
		n_eps,
		s1_target_altitude,
		s1_init_z,
		s1_init_vz,
		s1_mass,
		s1_collective_frac,
		s2_init_x,
		s2_init_y,
	)?;
	if let Some(cfg) = reference_cfg.as_ref()
	{
		let legacy_plant = RecorderLegacyPlant {
			af_dt,
			af_arm_length,
			af_k_thrust,
			af_k_drag,
			af_inertia,
			af_gravity,
			motor_lag_s,
		};
		check_reference_regime(
			&controller,
			cfg,
			s1_cfg.as_ref(),
			&legacy_plant,
			geometry_rows.is_some(),
		)?;
		let mut s1_cfg = s1_cfg.expect("checked: reference_cfg requires the stage-1 draws");
		// CTRL-83 D2: the recorder anchors where the lineage trained (the trainer's own flag).
		if cfg.anchor_nominal_mass
		{
			s1_cfg.anchor_at_mass(cfg.af_mass).map_err(pyo3::exceptions::PyValueError::new_err)?;
		}
		return Ok(record_on_training_cascade(
			&mut controller,
			cfg,
			&s1_cfg,
			&init_q,
			&init_om,
			target,
			steps,
		));
	}
	if s1_cfg.is_some()
	{
		return Err(pyo3::exceptions::PyValueError::new_err(
			"record_address_universe: stage-1 draws require reference_cfg (CTRL-17 G2) — the \
			 legacy 0.5-hover attitude PID does not hover a translating airframe, so the universe \
			 would be recorded on a climbing/falling vehicle. Pass the run's \
			 RewardGatedConfigPacked.",
		));
	}
	let mut sim = controller::AttitudeSim::new(
		af_dt,
		af_arm_length,
		af_k_thrust,
		af_k_drag,
		af_inertia,
		af_gravity,
	);
	if motor_lag_s > 0.0
	{
		sim.set_motor_lag(motor_lag_s);
	}
	match geometry_rows
	{
		None =>
		{
			let mut pid = controller::AttitudePidRs::new_default();
			let mut d = record_ops::Driver::Pid(&mut pid);
			Ok(record_ops::record_address_universe(
				&mut controller,
				&mut sim,
				&mut d,
				&init_q,
				&init_om,
				target,
				steps,
				None,
			))
		}
		Some(rows) =>
		{
			sim
				.set_geometry_core(rows.clone())
				.map_err(pyo3::exceptions::PyValueError::new_err)?;
			if let Some(a) = rotor_asym
			{
				sim
					.set_rotor_asym_core(Some(a))
					.map_err(pyo3::exceptions::PyValueError::new_err)?;
			}
			let nom = nominal_rows.unwrap_or(rows);
			let mut alloc = optimal::AllocLqrRs::build_core(
				&nom,
				inertia,
				q_att,
				q_rate,
				r_ctrl,
				tau_max,
				f_hover,
				pinv_lambda,
			)
			.map_err(pyo3::exceptions::PyValueError::new_err)?;
			let mut d = record_ops::Driver::Alloc(&mut alloc);
			Ok(record_ops::record_address_universe(
				&mut controller,
				&mut sim,
				&mut d,
				&init_q,
				&init_om,
				target,
				steps,
				None,
			))
		}
	}
}

/// Stage-1/2 recorder draws, all-or-nothing: a partial config would silently record
/// a half-vertical universe, which is the failure these parameters exist to end.
#[allow(clippy::too_many_arguments)]
fn recorder_stage1_cfg(
	n_eps: usize,
	target_altitude: Option<f32>,
	init_z: Option<Vec<f32>>,
	init_vz: Option<Vec<f32>>,
	mass: Option<Vec<f32>>,
	collective_frac: Option<Vec<f32>>,
	init_x: Option<Vec<f32>>,
	init_y: Option<Vec<f32>>,
) -> PyResult<Option<stage1::Stage1Cfg>>
{
	match (target_altitude, init_z, init_vz, mass, collective_frac)
	{
		(None, None, None, None, None) => Ok(None),
		(Some(t), Some(z), Some(vz), Some(m), Some(cf)) =>
		{
			let cfg = stage1::Stage1Cfg {
				target_altitude: t,
				lambda_alt: 0.0, // reward weight is unused when recording
				init_z: z,
				init_vz: vz,
				mass: m,
				collective_frac: cf,
				lambda_pos: 0.0,
				init_x: init_x.unwrap_or_default(),
				init_y: init_y.unwrap_or_default(),
				// ABI 32: the recorder's plant has ONE owner — reference_cfg
				// (AirframeRs::from_cfg arms imu_specific_force / rotor_drag on the
				// sim). The draws object carries neither; nothing is scored here.
				plant: plant_opts::PlantOpts::OFF,
				att_ref: att_err_ref::AttErrRef::LEVEL,
				alt_int: crate::altitude_integrator::AltitudeIntegrator::OFF,
				stress: crate::stress_opts::StressOpts::OFF,
				ep_motor_lag: Vec::new(),
			};
			cfg
				.validate(n_eps)
				.map_err(pyo3::exceptions::PyValueError::new_err)?;
			Ok(Some(cfg))
		}
		_ => Err(pyo3::exceptions::PyValueError::new_err(
			"record_address_universe: stage-1 args are all-or-nothing — pass every one of \
			 s1_target_altitude/s1_init_z/s1_init_vz/s1_mass/s1_collective_frac, or none",
		)),
	}
}

/// The recorder's legacy plant arguments — refused alongside reference_cfg.
struct RecorderLegacyPlant
{
	af_dt: f32,
	af_arm_length: f32,
	af_k_thrust: f32,
	af_k_drag: f32,
	af_inertia: [f32; 3],
	af_gravity: f32,
	motor_lag_s: f32,
}

impl RecorderLegacyPlant
{
	/// True when every field is at its signature default (i.e. nothing was passed).
	fn is_default(&self) -> bool
	{
		self.af_dt == 0.001
			&& self.af_arm_length == 0.075
			&& self.af_k_thrust == 2.4
			&& self.af_k_drag == 0.05
			&& self.af_inertia == [0.0023, 0.0023, 0.0046]
			&& self.af_gravity == 9.81
			&& self.motor_lag_s == 0.0
	}
}

/// Refusals for the training-cascade recorder: the plant has ONE owner, the draws
/// must match the armed channels, and features must not read an unflown channel.
fn check_reference_regime(
	controller: &controller::WnnController,
	cfg: &dagger_train::RewardGatedConfigPacked,
	s1: Option<&stage1::Stage1Cfg>,
	legacy: &RecorderLegacyPlant,
	has_geometry: bool,
) -> PyResult<()>
{
	let refuse = |m: String| {
		Err(pyo3::exceptions::PyValueError::new_err(format!(
			"record_address_universe: {m}"
		)))
	};
	if !cfg.translation
	{
		return refuse("reference_cfg must have translation on (it drives the translating recorder)".into());
	}
	if has_geometry
	{
		return refuse("reference_cfg (quad training cascade) cannot be combined with geometry_rows".into());
	}
	if !legacy.is_default()
	{
		return refuse(
			"pass the plant ONCE: with reference_cfg the airframe and motor lag come from it; \
			 leave af_*/motor_lag_s at their defaults"
				.into(),
		);
	}
	let Some(s1) = s1
	else
	{
		return refuse("reference_cfg requires the stage-1 draws (s1_*)".into());
	};
	if s1.target_altitude != cfg.target_altitude
	{
		return refuse(format!(
			"s1_target_altitude {} != reference_cfg.target_altitude {}",
			s1.target_altitude, cfg.target_altitude
		));
	}
	if s1.has_horizontal() != (cfg.xy_offset > 0.0)
	{
		return refuse(format!(
			"horizontal draws present={} but reference_cfg.xy_offset={} — the recorder's \
			 horizontal channel must be armed exactly when the training config's is",
			s1.has_horizontal(),
			cfg.xy_offset
		));
	}
	dagger_train::validate_train_regime(controller, cfg)
		.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("record_address_universe: {e}")))
}

/// Record on the training cascade: the config's plant and teacher (episode 0's
/// schedule slot), the injected ICs and draws.
fn record_on_training_cascade(
	controller: &mut controller::WnnController,
	cfg: &dagger_train::RewardGatedConfigPacked,
	s1_cfg: &stage1::Stage1Cfg,
	init_q: &[[f32; 4]],
	init_om: &[[f32; 3]],
	target: [f32; 3],
	steps: usize,
) -> (Vec<(usize, u64)>, Vec<(usize, u64)>)
{
	let af = dagger_train::AirframeRs::from_cfg(cfg);
	let mut sim = af.sim();
	let mut teacher = af.teacher(cfg.teacher_id_for(0, 0));
	let mut d = record_ops::Driver::Cascade(record_ops::CascadeDriver::new(&mut teacher, cfg, controller));
	let s1 = record_ops::RecorderStage1 {
		cfg: s1_cfg,
		gravity: cfg.af_gravity,
		k_thrust: cfg.af_k_thrust,
	};
	record_ops::record_address_universe(
		controller,
		&mut sim,
		&mut d,
		init_q,
		init_om,
		target,
		steps,
		Some(&s1),
	)
}

/// CTRL-17 G1 (26/09/2026): per-feature calibration samples from teacher-driven
/// TRAINING episodes of `cfg` (episode_regime: mass/vertical/horizontal draws,
/// collective, weather; the teacher inside the altitude + position loops). The
/// thermometer fitter consumes this under translation instead of flying the
/// attitude-only PID, whose ladders were degenerate on the xy features and fitted
/// an uncontrolled drift on the vertical ones. `controller` is the (untrained)
/// feature controller; its compute_features output is what is sampled.
#[pyfunction]
#[pyo3(signature = (controller, cfg, num_episodes, seed))]
fn sample_calibration_features(
	mut controller: PyRefMut<'_, controller::WnnController>,
	cfg: dagger_train::RewardGatedConfigPacked,
	num_episodes: usize,
	seed: u64,
) -> PyResult<Vec<Vec<f32>>>
{
	dagger_train::validate_train_regime(&controller, &cfg).map_err(|e| {
		pyo3::exceptions::PyValueError::new_err(format!("sample_calibration_features: {e}"))
	})?;
	if cfg.steps_per_episode == 0 || num_episodes == 0
	{
		return Err(pyo3::exceptions::PyValueError::new_err(
			"sample_calibration_features: steps_per_episode and num_episodes must be > 0",
		));
	}
	Ok(calib_sampler::sample_calibration_features_rs(
		&mut controller,
		&cfg,
		num_episodes,
		seed,
	))
}

/// ABI 49: the STUDENT's own visited feature values (per feature) on the same
/// training episodes `sample_calibration_features` flies the teacher through —
/// the `--threshold-refit-from-student` pool (calib_sampler::sample_student_features_rs).
/// `controller` is the TRAINED student (its cells are read, never written); `cfg` is
/// the reference config (expert_drives / explore_eps are overridden: the student flies).
#[pyfunction]
#[pyo3(signature = (controller, cfg, num_episodes, seed))]
fn sample_student_features(
	mut controller: PyRefMut<'_, controller::WnnController>,
	cfg: dagger_train::RewardGatedConfigPacked,
	num_episodes: usize,
	seed: u64,
) -> PyResult<Vec<Vec<f32>>>
{
	let flies = calib_sampler::student_flies(&cfg);
	dagger_train::validate_train_regime(&controller, &flies).map_err(|e| {
		pyo3::exceptions::PyValueError::new_err(format!("sample_student_features: {e}"))
	})?;
	if cfg.steps_per_episode == 0 || num_episodes == 0
	{
		return Err(pyo3::exceptions::PyValueError::new_err(
			"sample_student_features: steps_per_episode and num_episodes must be > 0",
		));
	}
	Ok(calib_sampler::sample_student_features_rs(&mut controller, &cfg, num_episodes, seed))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn record_input_entropy(
	mut controller: PyRefMut<'_, controller::WnnController>,
	init_q: Vec<[f32; 4]>,
	init_om: Vec<[f32; 3]>,
	target: [f32; 3],
	steps: usize,
	sensor_window: usize,
	sensor_frame: usize,
) -> (Vec<f64>, Vec<f64>)
{
	record_ops::record_input_entropy(
		&mut controller,
		&init_q,
		&init_om,
		target,
		steps,
		sensor_window,
		sensor_frame,
	)
}

/// Generic fitness combine (ram_core::fitness) — ABI 23. `values_flat` is
/// column-major (column c, candidate i at c*n+i); mode harmonic|arithmetic|
/// zscore; clamp read only by zscore. The SAME wrapper exists on the worker
/// wheel: the combine is results-determining logic shared by both substrates,
/// and Python adapters hold only the Metrics→columns mapping.
#[pyfunction]
#[pyo3(signature = (values_flat, num_candidates, weights, higher_is_better, mode, clamp, scale_floors=None))]
fn fitness_combine(
	values_flat: Vec<f64>,
	num_candidates: usize,
	weights: Vec<f64>,
	higher_is_better: Vec<bool>,
	mode: &str,
	clamp: f64,
	scale_floors: Option<Vec<f64>>,
) -> PyResult<Vec<f64>>
{
	ram_core::fitness::combine_flat(
		&values_flat, num_candidates, &weights, &higher_is_better, mode, clamp,
		scale_floors.as_deref())
		.map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Viability gate + base combine — ABI 24 (21/08/2026, approved gate
/// S_min=0.70 / E_max=8.0°; docs/CONTROLLER_FITNESS_GATE_SPEC.md). The gate
/// vectors are the PHYSICAL pair (stable_rate as a fraction, attitude error in
/// degrees), NOT the weighted columns: the fitness ranks reward, so "does it
/// fly" needs its own inputs. Additive export — fitness_combine is untouched,
/// so every banked recipe stays bit-identical.
#[pyfunction]
#[pyo3(signature = (values_flat, num_candidates, weights, higher_is_better, mode, clamp,
                    gate_stable, gate_err, gate_stable_min, gate_err_max, scale_floors=None,
                    gate_steady=None, gate_steady_max=None, gate_pos=None, gate_pos_max=None))]
fn gated_fitness_combine(
	values_flat: Vec<f64>,
	num_candidates: usize,
	weights: Vec<f64>,
	higher_is_better: Vec<bool>,
	mode: &str,
	clamp: f64,
	gate_stable: Vec<f64>,
	gate_err: Vec<f64>,
	gate_stable_min: f64,
	gate_err_max: f64,
	scale_floors: Option<Vec<f64>>,
	// ABI 59 (CTRL-58): the four-input gate's extra UPPER bounds, forwarded to THE gate
	// (ram_core::fitness::gated_combine_flat_bounds — one implementation, Luiz 18/11). Each is a
	// (values, bound) pair — both or neither; all unset ⇒ the two-input gate, bit for bit.
	gate_steady: Option<Vec<f64>>,
	gate_steady_max: Option<f64>,
	gate_pos: Option<Vec<f64>>,
	gate_pos_max: Option<f64>,
) -> PyResult<Vec<f64>>
{
	let verr = pyo3::exceptions::PyValueError::new_err;
	let mut extra = Vec::new();
	for (name, vals, max) in [("steady", &gate_steady, gate_steady_max), ("pos", &gate_pos, gate_pos_max)]
	{
		match (vals, max)
		{
			(Some(v), Some(m)) => extra.push(ram_core::fitness::GateUpperBound { values: v, max: m, name }),
			(None, None) => {}
			_ => return Err(verr(format!("gated_fitness_combine: gate_{name} and gate_{name}_max go together"))),
		}
	}
	ram_core::fitness::gated_combine_flat_bounds(
		&values_flat, num_candidates, &weights, &higher_is_better, mode, clamp,
		scale_floors.as_deref(),
		&gate_stable, &gate_err, gate_stable_min, gate_err_max, &extra)
		.map_err(verr)
}

/// Desirability combine — ABI 25 (26/08/2026, Luiz's redesign;
/// docs/DESIRABILITY_FITNESS_SHAPES.md, ram_core::fitness::
/// desirability_combine_flat). One continuous multiplicative utility: the
/// gate's job is the formula's limit behavior, the weights are never inert.
/// score = SUM w_c * h_c = weighted half-lives of desirability lost, lower =
/// better. `shapes[c]` in {"power" (higher-better fraction), "exp"
/// (lower-better cost)}; `half_anchors[c]` is where u = 0.5 (the retained
/// 0.70 / 8.0 gate calibration becomes these anchors). Additive export —
/// fitness_combine and gated_fitness_combine are untouched, so every banked
/// recipe stays bit-identical and the A/B's control arm is the shipped code.
#[pyfunction]
fn desirability_fitness_combine(
	values_flat: Vec<f64>,
	num_candidates: usize,
	weights: Vec<f64>,
	shapes: Vec<String>,
	half_anchors: Vec<f64>,
) -> PyResult<Vec<f64>>
{
	let shape_refs: Vec<&str> = shapes.iter().map(String::as_str).collect();
	ram_core::fitness::desirability_combine_flat(
		&values_flat, num_candidates, &weights, &shape_refs, &half_anchors)
		.map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pymodule]
fn ram_controller(m: &Bound<'_, PyModule>) -> PyResult<()>
{
	m.add("ABI_VERSION", ABI_VERSION)?;
	m.add("ATT_EST_PRECISION", ATT_EST_PRECISION)?;
	m.add("RECORDER_INPUTS", RECORDER_INPUTS)?;
	m.add("NEURO_LAYOUT", NEURO_LAYOUT)?;
	m.add("BPTT_DEVICE_DEFAULT", BPTT_DEVICE_DEFAULT)?;
	m.add_function(wrap_pyfunction!(fitness_combine, m)?)?;
	m.add_function(wrap_pyfunction!(gated_fitness_combine, m)?)?;
	m.add_function(wrap_pyfunction!(crate::episode_tail::episode_tail_p99_deg, m)?)?; // ABI 59
	m.add_function(wrap_pyfunction!(desirability_fitness_combine, m)?)?;
	m.add_function(wrap_pyfunction!(counter_rng_draw_u64, m)?)?;
	m.add_function(wrap_pyfunction!(counter_rng_uniform, m)?)?;
	m.add_function(wrap_pyfunction!(counter_rng_below, m)?)?;
	m.add_function(wrap_pyfunction!(memory_mutate_values, m)?)?;
	m.add_function(wrap_pyfunction!(memory_crossover_values, m)?)?;
	m.add_function(wrap_pyfunction!(memory_crossover_keyed, m)?)?;
	m.add_function(wrap_pyfunction!(memory_random_values, m)?)?;
	m.add_function(wrap_pyfunction!(cell_remap_grow, m)?)?;
	m.add_function(wrap_pyfunction!(cell_remap_shrink, m)?)?;
	m.add_function(wrap_pyfunction!(cell_remap_prefix_grow, m)?)?;
	m.add_function(wrap_pyfunction!(cell_remap_prefix_shrink, m)?)?;
	m.add_function(wrap_pyfunction!(cell_remap_delete_bit_window, m)?)?;
	m.add_function(wrap_pyfunction!(cell_drop_neurons_ge, m)?)?;
	m.add_function(wrap_pyfunction!(cell_drop_changed_neurons, m)?)?;
	m.add_function(wrap_pyfunction!(cell_majority, m)?)?;
	m.add_function(wrap_pyfunction!(arch_resample_suffix, m)?)?;
	m.add_function(wrap_pyfunction!(arch_resample_suffix_scoped, m)?)?;
	m.add_function(wrap_pyfunction!(arch_sample_distinct, m)?)?;
	m.add_function(wrap_pyfunction!(arch_sample_min_per_cluster, m)?)?;
	m.add_function(wrap_pyfunction!(arch_sample_framed1, m)?)?;
	m.add_function(wrap_pyfunction!(arch_framed1_slot_schedule, m)?)?;
	m.add_function(wrap_pyfunction!(arch_rebalance_features, m)?)?;
	m.add_function(wrap_pyfunction!(arch_pick_mask, m)?)?;
	m.add_function(wrap_pyfunction!(arch_output_segment_map, m)?)?;
	m.add_function(wrap_pyfunction!(arch_project_receptive_fields, m)?)?;
	m.add_function(wrap_pyfunction!(record_address_universe, m)?)?;
	m.add_function(wrap_pyfunction!(record_input_entropy, m)?)?;
	m.add_function(wrap_pyfunction!(sample_calibration_features, m)?)?;
	m.add_function(wrap_pyfunction!(sample_student_features, m)?)?;
	m.add("LAYER_STATE", memory_ops::LAYER_STATE)?;
	m.add("LAYER_OUTPUT", memory_ops::LAYER_OUTPUT)?;
	// Untrained-cell decode anchor (delta-control + residual neutral point),
	// derived from the active cell semantics — see controller::NEUTRAL_DECODE.
	// QUAD value; mode-aware callers use neutral_decode_for_mode (ABI 12).
	m.add("NEUTRAL_DECODE", controller::NEUTRAL_DECODE)?;
	m.add_function(wrap_pyfunction!(neutral_decode_for_mode, m)?)?;

	// Attitude sim + WNN controller + PID reference (paper #1 hot-path).
	m.add_class::<controller::AttitudeSim>()?;
	m.add_class::<controller::WnnController>()?;
	m.add_class::<ctbr::CtbrSpec>()?; // ABI 38: the cascade contract
	m.add_class::<genome_cells::GenomeCells>()?;
	m.add_class::<controller::AttitudePidRs>()?;
	// L1: plant control-effectiveness b, the d̂ observer's one plant constant.
	m.add_function(wrap_pyfunction!(controller::calibrate_control_gains, m)?)?;
	// Optimal-control DAGGER teachers (Rust port of control/optimal.py).
	m.add_class::<estimator::MahonyEstimatorRs>()?;
	m.add_class::<optimal::AttitudeLqrRs>()?;
	m.add_class::<optimal::AttitudeMpcRs>()?;
	m.add_class::<optimal::AttitudeLqiRs>()?;
	m.add_class::<optimal::AttitudeMpcOfRs>()?;
	// Overactuated Phase 2: allocator-aware LQR teacher (N-rotor residual baseline).
	m.add_class::<optimal::AllocLqrRs>()?;

	// DAGGER reward-gated training.
	m.add_class::<dagger_train::RewardGatedConfigPacked>()?;
	m.add_class::<dagger_train::TrainStats>()?;
	m.add_function(wrap_pyfunction!(dagger_train::dagger_train_inplace, m)?)?;
	m.add_function(wrap_pyfunction!(
		dagger_train::dagger_train_batch_inplace,
		m
	)?)?;
	// E4 committee scoring (rust-first hot loop; ICs pre-drawn in Python for numpy parity).
	m.add_function(wrap_pyfunction!(
		dagger_train::eval_ensemble_closed_loop,
		m
	)?)?;
	m.add_function(wrap_pyfunction!(dagger_train::score_classical_baseline, m)?)?;
	m.add_function(wrap_pyfunction!(position_score::score_position_teacher, m)?)?;
	// D1 diagnostic trace exports (06/08/2026). ADDITIVE ONLY — no scoring path is
	// touched and no ABI bump, deliberately: a live chain imports this wheel mid-run,
	// and additive-without-bump keeps old-source/new-wheel AND new-source/old-wheel
	// both consistent (the facade asserts strict ABI equality).
	m.add_function(wrap_pyfunction!(dagger_train::trace_classical_baseline, m)?)?;
	m.add_function(wrap_pyfunction!(cpu_score::trace_controller_cpu, m)?)?;

	// QSR decoders / monotonicity metric / reward.
	m.add_function(wrap_pyfunction!(controller::strategy_5_qsr_weighted, m)?)?;
	m.add_function(wrap_pyfunction!(controller::strategy_1_count_true, m)?)?;
	m.add_function(wrap_pyfunction!(controller::monotonicity_violations, m)?)?;
	m.add_function(wrap_pyfunction!(controller::compute_reward, m)?)?;
	m.add_function(wrap_pyfunction!(controller::yaw_from_quat, m)?)?;
	// W2 disturbances: per-episode seed derivation (the Metal kernel's twin).
	m.add_function(wrap_pyfunction!(controller::disturbance_episode_seed, m)?)?;

	// GPU-batched closed-loop scoring (macOS/Metal only).
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::score_controllers_metal,
		m
	)?)?;
	m.add_function(wrap_pyfunction!(cpu_score::score_controllers_cpu, m)?)?;
	// GPU controller training (split_retrain_output port) — bit-exact parity test.
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_train_parity_test,
		m
	)?)?;
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_train_seeded_parity_test,
		m
	)?)?;
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_split_train_loop_parity_test,
		m
	)?)?;
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_bptt_window_parity_test,
		m
	)?)?;
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_state_commit_parity_test,
		m
	)?)?;
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_nudge_distance_parity_test,
		m
	)?)?;
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_projected_address_parity_test,
		m
	)?)?;
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_candidate_rank_parity_test,
		m
	)?)?;
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_record_parity_test,
		m
	)?)?;
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_scan_parity_test,
		m
	)?)?;
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_sep_walk_parity_test,
		m
	)?)?;
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_accumulator_parity_test,
		m
	)?)?;
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_plant_latch_parity_test,
		m
	)?)?;
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_plant_counter_parity_test,
		m
	)?)?;
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_plant_bidir_parity_test,
		m
	)?)?;
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_mht_lookup_parity_test,
		m
	)?)?;
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_record_and_scan_parity_test,
		m
	)?)?;
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_record_search_parity_test,
		m
	)?)?;
	#[cfg(target_os = "macos")]
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_resolve_conflict_parity_test,
		m
	)?)?;

	// EDRA constraint solver (Rust port of Memory._solve_partial_connectivity).
	m.add_function(wrap_pyfunction!(
		controller_training::solve_partial_trinary_py,
		m
	)?)?;
	m.add_function(wrap_pyfunction!(
		controller_training::solve_partial_qsr_py,
		m
	)?)?;
	m.add_function(wrap_pyfunction!(
		controller_training::solve_partial_trinary_reachable_py,
		m
	)?)?;
	m.add_function(wrap_pyfunction!(
		controller_training::solve_partial_qsr_reachable_py,
		m
	)?)?;

	// Cooperative cancellation — the controller's OWN flag (separate process from
	// the worker, so ram_core::cancel's static is an independent copy here).
	m.add_function(wrap_pyfunction!(ram_core::cancel::set_cancel_flag, m)?)?;
	m.add_function(wrap_pyfunction!(ram_core::cancel::reset_cancel_flag, m)?)?;
	m.add_function(wrap_pyfunction!(ram_core::cancel::is_cancelled, m)?)?;

	Ok(())
}
