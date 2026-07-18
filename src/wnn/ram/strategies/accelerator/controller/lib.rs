//! ram_controller — drone attitude-controller hot-path (paper #1).
//!
//! Split out of ram_accelerator on 2026-06-19 into its own wheel so a
//! controller-only change rebuilds ONLY this cdylib; the IDS/LM worker wheel
//! (ram_accelerator) keeps running untouched, so no worker swap is needed.
//! Links `ram_core` for the shared substrate (sparse Memory + GPU forward,
//! cell semantics, cooperative cancellation).

use pyo3::prelude::*;
use pyo3::types::PyModule;

mod cell_mode;
mod controller;
mod controller_training;
mod controller_split;
mod dagger_train;
mod cpu_score;   // CPU (rayon) batch scorer — twin of score_controllers_metal
mod optimal;   // LQR + MPC DAGGER teachers (hand-rolled, no deps)
mod overactuated;   // Phase-0 N-rotor allocation substrate (not wired; docs/OVERACTUATED_RESIDUAL_DESIGN.md)

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
pub const ABI_VERSION: u32 = 13;

/// Mode-aware untrained-cell decode anchor (ABI 12): QUAD→0.75, TERNARY→0.5
/// (the fixed PLN empty_value), BINARY→0.5 (antagonist-pair effective neutral).
#[pyfunction]
fn neutral_decode_for_mode(memory_mode: u8) -> PyResult<f32> {
    cell_mode::validate_mode(memory_mode)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(cell_mode::neutral_decode(memory_mode))
}

#[pymodule]
fn ram_controller(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ABI_VERSION", ABI_VERSION)?;
    // Untrained-cell decode anchor (delta-control + residual neutral point),
    // derived from the active cell semantics — see controller::NEUTRAL_DECODE.
    // QUAD value; mode-aware callers use neutral_decode_for_mode (ABI 12).
    m.add("NEUTRAL_DECODE", controller::NEUTRAL_DECODE)?;
    m.add_function(wrap_pyfunction!(neutral_decode_for_mode, m)?)?;

    // Attitude sim + WNN controller + PID reference (paper #1 hot-path).
    m.add_class::<controller::AttitudeSim>()?;
    m.add_class::<controller::WnnController>()?;
    m.add_class::<controller::AttitudePidRs>()?;
    // Optimal-control DAGGER teachers (Rust port of control/optimal.py).
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
    m.add_function(wrap_pyfunction!(dagger_train::dagger_train_batch_inplace, m)?)?;
    // E4 committee scoring (rust-first hot loop; ICs pre-drawn in Python for numpy parity).
    m.add_function(wrap_pyfunction!(dagger_train::eval_ensemble_closed_loop, m)?)?;

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
    m.add_function(wrap_pyfunction!(metal_controller::score_controllers_metal, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_score::score_controllers_cpu, m)?)?;
    // GPU controller training (split_retrain_output port) — bit-exact parity test.
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_train_parity_test, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_train_seeded_parity_test, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_split_train_loop_parity_test, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_record_parity_test, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_scan_parity_test, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_sep_walk_parity_test, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_accumulator_parity_test, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_plant_latch_parity_test, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_plant_counter_parity_test, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_plant_bidir_parity_test, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_mht_lookup_parity_test, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_record_and_scan_parity_test, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_record_search_parity_test, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::run_controller_resolve_conflict_parity_test, m)?)?;

    // EDRA constraint solver (Rust port of Memory._solve_partial_connectivity).
    m.add_function(wrap_pyfunction!(controller_training::solve_partial_trinary_py, m)?)?;
    m.add_function(wrap_pyfunction!(controller_training::solve_partial_qsr_py, m)?)?;
    m.add_function(wrap_pyfunction!(controller_training::solve_partial_trinary_reachable_py, m)?)?;
    m.add_function(wrap_pyfunction!(controller_training::solve_partial_qsr_reachable_py, m)?)?;

    // Cooperative cancellation — the controller's OWN flag (separate process from
    // the worker, so ram_core::cancel's static is an independent copy here).
    m.add_function(wrap_pyfunction!(ram_core::cancel::set_cancel_flag, m)?)?;
    m.add_function(wrap_pyfunction!(ram_core::cancel::reset_cancel_flag, m)?)?;
    m.add_function(wrap_pyfunction!(ram_core::cancel::is_cancelled, m)?)?;

    Ok(())
}
