//! ram_controller — drone attitude-controller hot-path (paper #1).
//!
//! Split out of ram_accelerator on 2026-06-19 into its own wheel so a
//! controller-only change rebuilds ONLY this cdylib; the IDS/LM worker wheel
//! (ram_accelerator) keeps running untouched, so no worker swap is needed.
//! Links `ram_core` for the shared substrate (sparse Memory + GPU forward,
//! cell semantics, cooperative cancellation).

use pyo3::prelude::*;
use pyo3::types::PyModule;

mod altitude_pd; // scope C stage 1: outer altitude loop handing a collective to a teacher
mod arch_ops; // architecture (connectivity) operators (counter_rng, Rust-first)
mod cell_mode;
mod cell_remap; // cell ADDRESS remaps on architecture change (bit-exact port)
mod controller;
mod controller_split;
mod controller_training;
mod cpu_score; // CPU (rayon) batch scorer — twin of score_controllers_metal
mod dagger_train;
mod estimator; // Mahony attitude estimator — Rust twin of wnn/control/estimator.py
mod genome_cells; // opaque Rust-side cell store (Stage B: cells never cross FFI hot paths)
mod memory_ops; // GA-Memory cell-value operators (counter_rng, Rust-first)
mod optimal; // LQR + MPC DAGGER teachers (hand-rolled, no deps)
mod overactuated;
mod pid_firmware; // firmware-sourced cascaded attitude PID (twin of wnn/control/pid_firmware.py)
mod position_loop; // scope C stage 2: outermost position loop handing a TILT REF to a teacher
mod position_score; // scope C stage 2: score the full-state cascade in METRES
mod record_ops; // reference-rollout recorders (address universe, input entropy)
mod stage1; // scope C stage 1 vertical-channel config (scorer parameter object) // Phase-0 N-rotor allocation substrate (not wired; docs/OVERACTUATED_RESIDUAL_DESIGN.md)

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
pub const ABI_VERSION: u32 = 25;

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
                    s2_init_x = None, s2_init_y = None))]
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
) -> PyResult<(Vec<(usize, u64)>, Vec<(usize, u64)>)>
{
	let mut sim = controller::AttitudeSim::new(
		af_dt,
		af_arm_length,
		af_k_thrust,
		af_k_drag,
		af_inertia,
		af_gravity,
	);
	// Stage 1 is all-or-nothing: a partial config would silently record a
	// half-vertical universe, which is the failure this parameter exists to end.
	let s1_cfg = match (
		s1_target_altitude,
		s1_init_z,
		s1_init_vz,
		s1_mass,
		s1_collective_frac,
	)
	{
		(None, None, None, None, None) => None,
		(Some(t), Some(z), Some(vz), Some(m), Some(cf)) =>
		{
			let cfg = stage1::Stage1Cfg {
				target_altitude: t,
				lambda_alt: 0.0, // reward weight is unused when recording
				init_z: z,
				init_vz: vz,
				mass: m,
				collective_frac: cf,
				// Stage-2 horizontal draws: not yet threaded to the recorder —
				// s2 runs must extend this BEFORE their MEMORY stage or the
				// horizontal universe is degenerate (the exact stage-1 lesson).
				lambda_pos: 0.0,
				init_x: s2_init_x.unwrap_or_default(),
				init_y: s2_init_y.unwrap_or_default(),
			};
			cfg
				.validate(init_q.len().min(init_om.len()))
				.map_err(pyo3::exceptions::PyValueError::new_err)?;
			Some(cfg)
		}
		_ =>
		{
			return Err(pyo3::exceptions::PyValueError::new_err(
				"record_address_universe: stage-1 args are all-or-nothing — pass every one of \
             s1_target_altitude/s1_init_z/s1_init_vz/s1_mass/s1_collective_frac, or none",
			))
		}
	};
	let s1 = s1_cfg.as_ref().map(|cfg| record_ops::RecorderStage1 {
		cfg,
		gravity: af_gravity,
		k_thrust: af_k_thrust,
	});
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
				s1.as_ref(),
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
				s1.as_ref(),
			))
		}
	}
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
fn fitness_combine(
	values_flat: Vec<f64>,
	num_candidates: usize,
	weights: Vec<f64>,
	higher_is_better: Vec<bool>,
	mode: &str,
	clamp: f64,
) -> PyResult<Vec<f64>>
{
	ram_core::fitness::combine_flat(
		&values_flat, num_candidates, &weights, &higher_is_better, mode, clamp)
		.map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Viability gate + base combine — ABI 24 (21/08/2026, approved gate
/// S_min=0.70 / E_max=8.0°; docs/CONTROLLER_FITNESS_GATE_SPEC.md). The gate
/// vectors are the PHYSICAL pair (stable_rate as a fraction, attitude error in
/// degrees), NOT the weighted columns: the fitness ranks reward, so "does it
/// fly" needs its own inputs. Additive export — fitness_combine is untouched,
/// so every banked recipe stays bit-identical.
#[pyfunction]
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
) -> PyResult<Vec<f64>>
{
	ram_core::fitness::gated_combine_flat(
		&values_flat, num_candidates, &weights, &higher_is_better, mode, clamp,
		&gate_stable, &gate_err, gate_stable_min, gate_err_max)
		.map_err(pyo3::exceptions::PyValueError::new_err)
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
	m.add_function(wrap_pyfunction!(fitness_combine, m)?)?;
	m.add_function(wrap_pyfunction!(gated_fitness_combine, m)?)?;
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
	m.add_function(wrap_pyfunction!(record_address_universe, m)?)?;
	m.add_function(wrap_pyfunction!(record_input_entropy, m)?)?;
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
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_phase1_topk_parity_test,
		m
	)?)?;
	m.add_function(wrap_pyfunction!(
		metal_controller::run_controller_beam_solve_parity_test,
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
