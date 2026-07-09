//! ram_controller — drone attitude-controller hot-path (paper #1).
//!
//! Split out of ram_accelerator on 2026-06-19 into its own wheel so a
//! controller-only change rebuilds ONLY this cdylib; the IDS/LM worker wheel
//! (ram_accelerator) keeps running untouched, so no worker swap is needed.
//! Links `ram_core` for the shared substrate (sparse Memory + GPU forward,
//! cell semantics, cooperative cancellation).

use pyo3::prelude::*;
use pyo3::types::PyModule;

mod controller;
mod controller_training;
mod controller_split;
mod dagger_train;
mod cpu_score;   // CPU (rayon) batch scorer — twin of score_controllers_metal
mod optimal;   // LQR + MPC DAGGER teachers (hand-rolled, no deps)

// GPU-batched closed-loop controller eval (macOS/Metal only).
#[cfg(target_os = "macos")]
#[path = "metal_controller.rs"]
mod metal_controller;

/// ABI version of the controller wheel's Python surface. Mirrors
/// ram_accelerator's contract; wnn/control/_accel.py asserts it at import.
/// 3 = W2 disturbances (set_disturbance / disturbance_episode_seed /
///     score_controllers_metal + eval_ensemble_closed_loop dist args).
pub const ABI_VERSION: u32 = 3;

#[pymodule]
fn ram_controller(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ABI_VERSION", ABI_VERSION)?;

    // Attitude sim + WNN controller + PID reference (paper #1 hot-path).
    m.add_class::<controller::AttitudeSim>()?;
    m.add_class::<controller::WnnController>()?;
    m.add_class::<controller::AttitudePidRs>()?;
    // Optimal-control DAGGER teachers (Rust port of control/optimal.py).
    m.add_class::<optimal::AttitudeLqrRs>()?;
    m.add_class::<optimal::AttitudeMpcRs>()?;

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
