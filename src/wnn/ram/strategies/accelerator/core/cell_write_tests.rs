//! Exhaustive mode × cell × target pins for `cell_write.rs`.
//!
//! The `legacy` module is a VERBATIM copy of the pre-12/11/2026 implementations
//! the consolidation replaced (controller/cell_mode.rs, controller_training.rs
//! `nudge_toward*`, the IDS clamped `(cell ± 1).clamp(0, 3)` nudge, and the
//! two default-cell tables). Every shared primitive must agree with its oracle on
//! the whole input space — that is the bit-identity proof at the cell level.

use super::*;
use crate::unwritten_lean::UnwrittenLean;

const ALL_MODES: [u8; 6] = [0, 1, 2, 3, 4, 5];

mod legacy
{
	use crate::neuron_memory::{
		cell_to_weight, BINARY, EMPTY_U8, FALSE_U8, PLN, QSR, QUAD_BINARY, QUAD_WEAK_FALSE, QUAD_WEIGHTED,
		QUAD_WEIGHTS, TRUE_U8,
	};

	pub fn is_quad(mode: u8) -> bool
	{
		mode == QUAD_WEIGHTED || mode == QUAD_BINARY || mode == QSR
	}

	pub fn nudge_toward(current: u8, target_true: bool) -> u8
	{
		let cur = current & 0x3;
		if target_true
		{
			match cur
			{
				0 => 1,
				1 => 2,
				2 => 3,
				_ => 3,
			}
		}
		else
		{
			match cur
			{
				3 => 2,
				2 => 1,
				1 => 0,
				_ => 0,
			}
		}
	}

	pub fn nudge_toward_value(current: u8, target: u8) -> u8
	{
		let c = (current & 0x3) as i8;
		let t = (target & 0x3) as i8;
		if c < t
		{
			(c + 1) as u8
		}
		else if c > t
		{
			(c - 1) as u8
		}
		else
		{
			c as u8
		}
	}

	pub fn ids_clamp_nudge(cell: i64, target_true: bool) -> i64
	{
		(cell + 2 * (target_true as i64) - 1).clamp(0, 3)
	}

	pub fn nudge_cell(current: u8, target_true: bool, mode: u8) -> u8
	{
		if is_quad(mode)
		{
			nudge_toward(current, target_true)
		}
		else if target_true
		{
			TRUE_U8
		}
		else
		{
			FALSE_U8
		}
	}

	pub fn nudge_cell_value(current: u8, target: u8, mode: u8) -> u8
	{
		if is_quad(mode)
		{
			nudge_toward_value(current, target)
		}
		else
		{
			target
		}
	}

	pub fn nudge_distance(cell: u8, target_true: bool, mode: u8) -> u8
	{
		if is_quad(mode)
		{
			let c = (cell & 0x3) as i8;
			if target_true
			{
				(3 - c) as u8
			}
			else
			{
				c as u8
			}
		}
		else
		{
			let want = if target_true { TRUE_U8 } else { FALSE_U8 };
			if cell == want
			{
				0
			}
			else if cell == EMPTY_U8
			{
				1
			}
			else
			{
				2
			}
		}
	}

	pub fn cell_fire_bit(cell: u8, mode: u8) -> bool
	{
		if is_quad(mode)
		{
			(cell >> 1) & 1 != 0
		}
		else
		{
			cell == TRUE_U8
		}
	}

	pub fn true_cell(mode: u8) -> u8
	{
		if is_quad(mode)
		{
			3
		}
		else
		{
			TRUE_U8
		}
	}

	pub fn plant_cell(on: bool, mode: u8) -> u8
	{
		match (is_quad(mode), on)
		{
			(true, true) => 3,
			(true, false) => 1,
			(false, true) => TRUE_U8,
			(false, false) => FALSE_U8,
		}
	}

	pub fn cell_coin_prob(cell: u8, mode: u8) -> f32
	{
		match mode
		{
			QSR => QUAD_WEIGHTS[(cell & 0x3) as usize],
			PLN => match cell
			{
				FALSE_U8 => 0.0,
				TRUE_U8 => 1.0,
				_ => 0.5,
			},
			_ => cell_to_weight(cell as i64, mode, 0.5),
		}
	}

	/// controller/cell_mode.rs `canonical_default_cell`.
	pub fn controller_default_cell(mode: u8) -> u8
	{
		match mode
		{
			BINARY => FALSE_U8,
			_ => EMPTY_U8,
		}
	}

	/// core/cell_mode.rs `CellMode::default_cell` (the IDS table).
	pub fn ids_default_cell(mode: u8) -> u8
	{
		match mode
		{
			0 | 5 => EMPTY_U8,
			3 => FALSE_U8,
			_ => QUAD_WEAK_FALSE as u8,
		}
	}
}

#[test]
fn quad_step_matches_every_legacy_nudge()
{
	for cell in 0..=255u8
	{
		for t in [false, true]
		{
			assert_eq!(quad_step(cell, t), legacy::nudge_toward(cell, t), "cell {cell} t {t}");
		}
		for target in 0..=255u8
		{
			assert_eq!(quad_step_toward(cell, target), legacy::nudge_toward_value(cell, target), "{cell}->{target}");
		}
	}
	// The IDS trainers' clamped form agrees on every 2-bit cell.
	for cell in 0..=3i64
	{
		for t in [false, true]
		{
			assert_eq!(quad_step(cell as u8, t) as i64, legacy::ids_clamp_nudge(cell, t), "cell {cell} t {t}");
		}
	}
}

#[test]
fn every_mode_cell_target_matches_the_legacy_controller()
{
	for mode in ALL_MODES
	{
		let m = CellMode::of(mode);
		assert_eq!(m.is_quad_family(), legacy::is_quad(mode), "mode {mode}");
		assert_eq!(m.true_cell(), legacy::true_cell(mode), "mode {mode}");
		assert_eq!(m.false_cell(), FALSE_U8, "mode {mode}");
		for on in [false, true]
		{
			assert_eq!(m.plant_cell(on), legacy::plant_cell(on, mode), "mode {mode} on {on}");
		}
		for cell in 0..=255u8
		{
			assert_eq!(m.fire_bit(cell), legacy::cell_fire_bit(cell, mode), "mode {mode} cell {cell}");
			for t in [false, true]
			{
				assert_eq!(m.nudge_step(cell, t), legacy::nudge_cell(cell, t, mode), "mode {mode} cell {cell} t {t}");
				assert_eq!(m.nudge_distance(cell, t), legacy::nudge_distance(cell, t, mode), "mode {mode} cell {cell}");
			}
			for target in 0..=255u8
			{
				assert_eq!(m.nudge_toward_value(cell, target), legacy::nudge_cell_value(cell, target, mode));
			}
		}
		for cell in 0..=3u8
		{
			assert_eq!(m.coin_prob(cell, 0.5).to_bits(), legacy::cell_coin_prob(cell, mode).to_bits(), "mode {mode} cell {cell}");
		}
	}
}

/// The default-cell split: `False` is exactly the old IDS table, `True` exactly
/// the old controller table, and only the 4-state family distinguishes them.
#[test]
fn unwritten_lean_reproduces_both_legacy_default_tables()
{
	for mode in ALL_MODES
	{
		let m = CellMode::of(mode);
		assert_eq!(m.default_cell(UnwrittenLean::False), legacy::ids_default_cell(mode), "mode {mode}");
		assert_eq!(m.default_cell(UnwrittenLean::True), legacy::controller_default_cell(mode), "mode {mode}");
		let differs = m.default_cell(UnwrittenLean::False) != m.default_cell(UnwrittenLean::True);
		assert_eq!(differs, m.is_quad_family(), "mode {mode}: only 4-state modes lean");
	}
}

/// The sequential writer is named in the shared `WriteRule` vocabulary and is
/// exactly what the controller's resolver hands its DAgger walk with no vote.
#[test]
fn sequential_write_rule_is_the_controller_default_rule()
{
	for mode in ALL_MODES
	{
		let m = CellMode::of(mode);
		assert_eq!(Ok(m.sequential_write_rule()), WriteRule::resolve_controller(mode, None), "mode {mode}");
	}
}

#[test]
#[should_panic(expected = "unknown memory_mode 9")]
fn of_refuses_unknown_codes()
{
	let _ = CellMode::of(9);
}

/// common.metal carries the GPU twins under the shared names.
#[test]
fn common_metal_declares_the_write_twins()
{
	let src = include_str!("shaders/common.metal");
	for name in [
		"inline bool wnn_is_quad_family(uint mode)",
		"inline uint wnn_quad_step(uint cell, bool target_true)",
		"inline uint wnn_quad_step_toward(uint cell, uint target)",
		"inline uint wnn_true_cell(uint mode)",
		"inline bool wnn_fire_bit(uint cell, uint mode)",
		"inline uint wnn_nudge_step(uint cell, bool target_true, uint mode)",
		"inline uint wnn_nudge_toward_value(uint cell, uint target, uint mode)",
		"inline uint wnn_nudge_distance(uint cell, bool target_true, uint mode)",
	]
	{
		assert!(src.contains(name), "common.metal lacks `{name}`");
	}
	let decl = format!("constant float WNN_PLN_U_EXPECTED = {:.1}f;", crate::neuron_memory::PLN_U_EXPECTED);
	assert!(src.contains(&decl), "common.metal lacks `{decl}`");
}

/// GPU twin probe: one thread per (mode, cell, target, target_value) — the whole
/// 6 x 4 x 2 x 4 = 192-point space — evaluates every common.metal write twin plus
/// marker_slots `slot_nudge_cell` (the atomic sequential writer) and compares
/// against the Rust primitives. Exhaustive, not sampled.
#[cfg(target_os = "macos")]
const GPU_PROBE: &str = r#"
kernel void cell_write_probe(device uint *out [[buffer(0)]], device atomic_uint *slots [[buffer(1)]],
	uint i [[thread_position_in_grid]])
{
	if (i >= 192u)
		return;
	uint tv = i & 3u, t = (i >> 2) & 1u, cell = (i >> 3) & 3u, mode = i >> 5;
	bool tt = t != 0u;
	device uint *o = out + i * 8u;
	o[0] = wnn_fire_bit(cell, mode) ? 1u : 0u;
	o[1] = wnn_true_cell(mode);
	o[2] = wnn_nudge_step(cell, tt, mode);
	o[3] = wnn_nudge_toward_value(cell, tv, mode);
	o[4] = wnn_nudge_distance(cell, tt, mode);
	o[5] = wnn_quad_step(cell, tt);
	o[6] = wnn_quad_step_toward(cell, tv);
	atomic_store_explicit(&slots[i], cell, memory_order_relaxed);
	slot_nudge_cell(slots, i, tt, mode);
	o[7] = atomic_load_explicit(&slots[i], memory_order_relaxed);
}
"#;

#[cfg(target_os = "macos")]
fn run_gpu_probe() -> Option<Vec<u32>>
{
	use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
	let device = Device::system_default()?;
	let src = [include_str!("shaders/common.metal"), include_str!("shaders/marker_slots.metal"), GPU_PROBE].join("\n");
	let lib = device.new_library_with_source(&src, &CompileOptions::new()).expect("probe compiles with common.metal");
	let func = lib.get_function("cell_write_probe", None).expect("probe kernel");
	let pipe = device.new_compute_pipeline_state_with_function(&func).expect("probe pipeline");
	let out = device.new_buffer(192 * 8 * 4, MTLResourceOptions::StorageModeShared);
	let slots = device.new_buffer(192 * 4, MTLResourceOptions::StorageModeShared);
	let queue = device.new_command_queue();
	let cmd = queue.new_command_buffer();
	let enc = cmd.new_compute_command_encoder();
	enc.set_compute_pipeline_state(&pipe);
	enc.set_buffer(0, Some(&out), 0);
	enc.set_buffer(1, Some(&slots), 0);
	enc.dispatch_threads(MTLSize::new(192, 1, 1), MTLSize::new(64, 1, 1));
	enc.end_encoding();
	cmd.commit();
	cmd.wait_until_completed();
	let ptr = out.contents() as *const u32;
	Some(unsafe { std::slice::from_raw_parts(ptr, 192 * 8) }.to_vec())
}

#[cfg(target_os = "macos")]
#[test]
fn cell_write_gpu_parity()
{
	let Some(gpu) = run_gpu_probe()
	else
	{
		eprintln!("skipping: no Metal device");
		return;
	};
	for i in 0..192usize
	{
		let (tv, t, cell, mode) = ((i & 3) as u8, (i >> 2) & 1 == 1, ((i >> 3) & 3) as u8, (i >> 5) as u8);
		let m = CellMode::of(mode);
		let cpu = [
			m.fire_bit(cell) as u32,
			m.true_cell() as u32,
			m.nudge_step(cell, t) as u32,
			m.nudge_toward_value(cell, tv) as u32,
			m.nudge_distance(cell, t) as u32,
			quad_step(cell, t) as u32,
			quad_step_toward(cell, tv) as u32,
			m.nudge_step(cell, t) as u32,
		];
		assert_eq!(&gpu[i * 8..i * 8 + 8], &cpu, "mode {mode} cell {cell} target {t} value {tv}");
	}
}
