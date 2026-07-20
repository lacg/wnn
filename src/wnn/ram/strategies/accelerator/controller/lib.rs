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
mod memory_ops;  // GA-Memory cell-value operators (counter_rng, Rust-first)
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
pub const ABI_VERSION: u32 = 17;

/// Mode-aware untrained-cell decode anchor (ABI 12): QUAD→0.75, TERNARY→0.5
/// (the fixed PLN empty_value), BINARY→0.5 (antagonist-pair effective neutral).
#[pyfunction]
fn neutral_decode_for_mode(memory_mode: u8) -> PyResult<f32> {
    cell_mode::validate_mode(memory_mode)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(cell_mode::neutral_decode(memory_mode))
}

// ---- counter_rng bridge (ram_core) -----------------------------------------
// Exposed so the Python mirror (wnn/ram/counter_rng.py) can be proven identical
// draw-for-draw. These are NOT a Python draw API — operators belong in Rust; the
// mirror exists to verify that moving them there does not change what a draw is.

#[pyfunction]
fn counter_rng_draw_u64(seed: u64, generation: u64, genome: u64, layer: u64, index: u64, sub: u64) -> u64 {
    ram_core::counter_rng::draw_u64(seed, generation, genome, layer, index, sub)
}

#[pyfunction]
fn counter_rng_uniform(seed: u64, generation: u64, genome: u64, layer: u64, index: u64, sub: u64) -> f64 {
    ram_core::counter_rng::uniform(seed, generation, genome, layer, index, sub)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn counter_rng_below(n: u64, seed: u64, generation: u64, genome: u64, layer: u64, index: u64, sub: u64) -> u64 {
    ram_core::counter_rng::below(n, seed, generation, genome, layer, index, sub)
}

/// GA-Memory value mutation, one FFI call for a whole layer. Replaces the
/// per-cell Python loop (~10^9 interpreter iterations per production run, each
/// with a numpy rng.random()). Uses the shared counter RNG, so results differ
/// from the numpy path BY DESIGN — this is the opt-in lineage break.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn memory_mutate_values(
    values: Vec<u8>, quad: bool, rate: f64,
    seed: u64, generation: u64, genome: u64, layer: u64,
) -> Vec<u8> {
    let mut v = values;
    memory_ops::mutate_values(&mut v, quad, rate, seed, generation, genome, layer);
    v
}

/// Uniform per-cell crossover over two index-aligned value vectors.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn memory_crossover_values(
    a: Vec<u8>, b: Vec<u8>,
    seed: u64, generation: u64, genome: u64, layer: u64,
) -> PyResult<Vec<u8>> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "crossover needs index-aligned parents, got {} and {}", a.len(), b.len())));
    }
    Ok(memory_ops::crossover_values(&a, &b, seed, generation, genome, layer))
}

/// Address-KEYED uniform crossover of cell values (MEMORY phase). Handles
/// different-shaped parents: the child keeps a's universe and adopts b's value
/// only where b holds the same (neuron, address).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn memory_crossover_keyed(
    a_neurons: Vec<u32>, a_addrs: Vec<u64>, a_values: Vec<u8>,
    b_neurons: Vec<u32>, b_addrs: Vec<u64>, b_values: Vec<u8>,
    seed: u64, generation: u64, genome: u64, layer: u64,
) -> PyResult<Vec<u8>> {
    if a_neurons.len() != a_addrs.len() || a_neurons.len() != a_values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("parent a arrays must be equal length"));
    }
    if b_neurons.len() != b_addrs.len() || b_neurons.len() != b_values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("parent b arrays must be equal length"));
    }
    Ok(memory_ops::crossover_values_keyed(
        &a_neurons, &a_addrs, &a_values, &b_neurons, &b_addrs, &b_values,
        seed, generation, genome, layer))
}

#[pymodule]
fn ram_controller(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ABI_VERSION", ABI_VERSION)?;
    m.add_function(wrap_pyfunction!(counter_rng_draw_u64, m)?)?;
    m.add_function(wrap_pyfunction!(counter_rng_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(counter_rng_below, m)?)?;
    m.add_function(wrap_pyfunction!(memory_mutate_values, m)?)?;
    m.add_function(wrap_pyfunction!(memory_crossover_values, m)?)?;
    m.add_function(wrap_pyfunction!(memory_crossover_keyed, m)?)?;
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
