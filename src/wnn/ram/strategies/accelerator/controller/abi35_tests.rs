//! ABI 35: bptt walk fixtures, CPU byte-identity pins, and CPU/GPU walk parity.
//!
//! The CPU pins were CAPTURED on the pre-ABI-35 walk (commit 72b8535e, the per-record
//! neuron_entries + HashSet solver) by running this same file there; they hold the
//! item-B rewrite (window-lived SolverView, select-nth top-k, concurrent solves) to
//! bit-identity on every shape the GPU walk is later compared on.

use crate::controller::{ReplayObs, WnnController};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// FNV-1a over both layers' cells in canonical (layer, neuron, sorted addr) order.
pub(crate) fn cells_digest(c: &WnnController) -> u64
{
	let mut h: u64 = 0xcbf2_9ce4_8422_2325;
	let mut feed = |v: u64| {
		for b in v.to_le_bytes()
		{
			h ^= b as u64;
			h = h.wrapping_mul(0x1000_0000_01b3);
		}
	};
	let n_state = c.state_neurons_pub();
	let num_out = c.num_motors() * c.levels_per_motor();
	for (layer, nn) in [(0u64, n_state), (1u64, num_out)]
	{
		for n in 0..nn
		{
			let mut e = if layer == 0 { c.state_entries(n) } else { c.output_entries(n) };
			e.sort_unstable();
			feed(layer);
			feed(n as u64);
			feed(e.len() as u64);
			for (a, v) in e
			{
				feed(a);
				feed(v as u64);
			}
		}
	}
	h
}

/// One bptt walk shape. `prod` = the production feature set (per-axis p/i without
/// yaw, yaw err + integral, obs_pwm, vertical + horizontal channels: 26 features).
#[derive(Clone, Copy, Debug)]
pub(crate) struct WalkShape
{
	pub mode: u8,
	pub sn: usize,
	pub sbpn: usize,
	pub obpn: usize,
	pub levels: usize,
	pub bpf: usize,
	pub window_k: usize,
	pub prod: bool,
	pub seed: u64,
	/// Probability a connection lands on a state bit (the solves' mutable bits).
	pub tail: f64,
}

pub(crate) fn walk_controller(s: &WalkShape) -> WnnController
{
	let num_motors = 4usize;
	let nf = if s.prod { 26 } else { 9 };
	let frame_bits = nf * s.bpf;
	let mut rng = SmallRng::seed_from_u64(s.seed);
	let thresholds: Vec<f32> = (0..frame_bits).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
	let state_in = s.window_k * frame_bits + s.sn;
	let out_in = frame_bits + s.sn;
	// Bias half the connections onto the state bits so the solves have work to do.
	let pick = |rng: &mut SmallRng, width: usize, tail: usize| -> i64 {
		if tail > 0 && rng.gen_bool(s.tail)
		{
			(width - tail + rng.gen_range(0..tail)) as i64
		}
		else
		{
			rng.gen_range(0..width) as i64
		}
	};
	let sconn: Vec<i64> = (0..s.sn * s.sbpn).map(|_| pick(&mut rng, state_in, s.sn)).collect();
	let oconn: Vec<i64> = (0..num_motors * s.levels * s.obpn).map(|_| pick(&mut rng, out_in, s.sn)).collect();
	let decode = if s.mode == ram_core::neuron_memory::BINARY { Some(crate::cell_mode::DECODE_ANTAGONIST) } else { None };
	WnnController::new_core(
		num_motors, s.levels, s.bpf, s.window_k, s.sn, s.sbpn, s.obpn, thresholds, sconn, oconn,
		true, 0.1, 0.95, 1.0,
		false, false, s.prod, s.prod, false, s.prod, s.prod, s.prod,
		0.99, 1.0, 0.001, false, 1,
		s.mode, decode, None, 0.05, false, 0.30,
		s.prod, s.prod, s.prod, s.prod, s.prod, false, 1,
	)
	.expect("walk fixture controller must construct")
}

/// One synthetic episode's replay streams.
pub(crate) struct WalkEpisode
{
	pub g: Vec<[f32; 3]>,
	pub a: Vec<[f32; 3]>,
	pub t: Vec<[f32; 3]>,
	pub p: Vec<[f32; 4]>,
	pub obs: ReplayObs,
}

pub(crate) fn walk_episodes(seed: u64, eps: usize, steps: usize) -> Vec<WalkEpisode>
{
	let mut rng = SmallRng::seed_from_u64(seed ^ 0xE915_0DE5);
	let v3 = |rng: &mut SmallRng, s: f32| [rng.gen_range(-s..s), rng.gen_range(-s..s), rng.gen_range(-s..s)];
	(0..eps)
		.map(|_| {
			let mut e = WalkEpisode {
				g: vec![],
				a: vec![],
				t: vec![],
				p: vec![],
				obs: ReplayObs { vert: vec![], horiz: vec![], pwm_dev: vec![], att_up: vec![], cascade: vec![], dhat: vec![], att_bias: vec![] },
			};
			for _ in 0..steps
			{
				e.g.push(v3(&mut rng, 1.0));
				let mut ac = v3(&mut rng, 0.5);
				ac[2] += 9.81;
				e.a.push(ac);
				e.t.push(v3(&mut rng, 0.1));
				e.p.push([rng.gen_range(0.2..0.8), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]);
				e.obs.vert.push(v3(&mut rng, 0.5));
				e.obs.horiz.push([rng.gen_range(-0.5..0.5), rng.gen_range(-0.5..0.5), rng.gen_range(-0.5..0.5), rng.gen_range(-0.5..0.5)]);
				e.obs.pwm_dev.push([rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1)]);
			}
			e
		})
		.collect()
}

fn slice_obs(o: &ReplayObs, s: usize, e: usize) -> ReplayObs
{
	ReplayObs {
		vert: o.vert[s..e].to_vec(),
		horiz: o.horiz[s..e].to_vec(),
		pwm_dev: o.pwm_dev[s..e].to_vec(),
		att_up: vec![],
		cascade: vec![],
		dhat: vec![],
		att_bias: vec![],
	}
}

/// Walk every episode in `win`-step chunks through the public bptt_train_window,
/// exactly like train_on_trajectory_rs (reset only on each episode's first chunk).
pub(crate) fn walk_cpu_public(c: &mut WnnController, eps: &[WalkEpisode], win: usize, topk: usize, protect: bool) -> (usize, usize)
{
	let (mut sw, mut ow) = (0, 0);
	for e in eps
	{
		let n = e.g.len();
		let mut start = 0;
		while start < n
		{
			let end = (start + win).min(n);
			let (a, b) = c.bptt_train_window(
				e.g[start..end].to_vec(),
				e.a[start..end].to_vec(),
				e.t[start..end].to_vec(),
				e.p[start..end].to_vec(),
				topk,
				start == 0,
				protect,
				None,
				0.0,
				None,
				false,
				0.0,
				None,
				None,
				Some(slice_obs(&e.obs, start, end)),
			);
			sw += a;
			ow += b;
			start = end;
		}
	}
	(sw, ow)
}

/// (shape, eps, steps, window, protect) — the pinned CPU walk cases.
pub(crate) fn pin_cases() -> Vec<(WalkShape, usize, usize, usize, bool)>
{
	let mut v = Vec::new();
	for (i, &mode) in [2u8, 0, 3].iter().enumerate()
	{
		let small = WalkShape { mode, sn: 4, sbpn: 8, obpn: 8, levels: 8, bpf: 3, window_k: 2, prod: false, seed: 0xA35 + i as u64, tail: 0.3 };
		v.push((small, 3, 70, 32, false));
		v.push((small, 2, 50, 16, true));
		for &sn in &[4usize, 8]
		{
			let prod = WalkShape { mode, sn, sbpn: 24, obpn: 24, levels: 64, bpf: 16, window_k: 2, prod: true, seed: 0xB35 + i as u64, tail: 0.3 };
			v.push((prod, 2, 48, 32, false));
		}
	}
	v
}

/// sn>0 student for the whole-dagger_train pins: vertical channel on (matches the
/// cf21 translation regime of ctrl17_pins::pin_translation_cfg), 12 features.
pub(crate) fn dagger_student(mode: u8, sn: usize) -> WnnController
{
	let (levels, bpf, obpn, sbpn, k) = (8usize, 3usize, 10usize, 10usize, 2usize);
	let nf = 12usize;
	let frame_bits = nf * bpf;
	let mut rng = SmallRng::seed_from_u64(0xDA35 ^ mode as u64);
	let thresholds: Vec<f32> = (0..frame_bits).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
	let sconn: Vec<i64> = (0..sn * sbpn).map(|_| rng.gen_range(0..k * frame_bits + sn) as i64).collect();
	let oconn: Vec<i64> = (0..4 * levels * obpn).map(|_| rng.gen_range(0..frame_bits + sn) as i64).collect();
	let decode = if mode == ram_core::neuron_memory::BINARY { Some(crate::cell_mode::DECODE_ANTAGONIST) } else { None };
	WnnController::new_core(
		4, levels, bpf, k, sn, sbpn, obpn, thresholds, sconn, oconn,
		true, 0.1, 0.9, 1.0,
		false, false, false, false, false, false, false, false,
		0.99, 1.0, 0.001, false, 1,
		mode, decode, None, 0.05, false, 0.30,
		true, true, true, false, false, false, 1,
	)
	.expect("dagger student must construct")
}

/// Whole reward-gated DAgger on the cf21 translation pin config: (cells digest,
/// per-round cells written).
pub(crate) fn dagger_digest(mode: u8, sn: usize) -> (u64, Vec<usize>)
{
	dagger_digest_with(mode, sn, false).0
}

/// (cells digest, per-round writes), full TrainStats fingerprint — CPU or GPU walk.
pub(crate) fn dagger_digest_with(mode: u8, sn: usize, gpu: bool) -> ((u64, Vec<usize>), String)
{
	let mut c = dagger_student(mode, sn);
	let cfg = crate::ctrl17_pins::pin_translation_cfg(0.0);
	let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 0x5EED, gpu);
	let fp = format!(
		"{:?} {:?} {:?} {:?} {} {:?}",
		st.iter_fitness, st.iter_mean_err_deg, st.iter_stable_rate, st.iter_n_trained, st.train_steps, st.iter_motor_jerk_mean
	);
	// The Lamarckian egress ORDER is part of the contract: GA-Memory's mutate_values
	// keys its draws by universe position, so two memories with identical cells but a
	// different export order mutate different cells (the 05/11 MEMORY-stage smoke split).
	let ex = c.export_cells();
	let order = cells_order_digest(&ex.0, &ex.1);
	((cells_digest(&c), st.iter_cells_written.clone()), format!("{fp} egress={order:#x}"))
}

/// FNV over the exported triples IN ORDER (position-sensitive, unlike cells_digest).
pub(crate) fn cells_order_digest(st: &[(usize, u64, u8)], ot: &[(usize, u64, u8)]) -> u64
{
	let mut h: u64 = 0xcbf29ce484222325;
	for &(n, a, v) in st.iter().chain(ot.iter())
	{
		for x in [n as u64, a, v as u64]
		{
			h ^= x;
			h = h.wrapping_mul(0x100000001b3);
		}
	}
	h
}

/// Prepare every window of `eps` (train_on_trajectory_rs slicing) on a clone of `base`.
pub(crate) fn plan_episodes(base: &WnnController, eps: &[WalkEpisode], win: usize) -> Vec<crate::bptt_plan::BpttWindowPlan>
{
	let mut c = base.clone();
	let mut plans = Vec::new();
	for e in eps
	{
		let n = e.g.len();
		let mut start = 0;
		while start < n
		{
			let end = (start + win).min(n);
			let ro = slice_obs(&e.obs, start, end);
			if let Some(p) = c.bptt_prepare_checked(
				&e.g[start..end],
				&e.a[start..end],
				&e.t[start..end],
				&e.p[start..end],
				start == 0,
				None,
				0.0,
				None,
				None,
				Some(&ro),
			)
			{
				plans.push(p);
			}
			start = end;
		}
	}
	plans
}

/// CPU walk over a plan list: (cells digest, s_writes, o_writes, prev_state).
pub(crate) fn walk_plans_cpu(base: &WnnController, plans: &[crate::bptt_plan::BpttWindowPlan], topk: usize, protect: bool) -> (u64, usize, usize, Vec<u8>)
{
	let mut c = base.clone();
	let (mut sw, mut ow) = (0, 0);
	for p in plans
	{
		let (a, b) = c.bptt_walk_cpu(p, topk, protect, &crate::bptt_plan::BpttL4::off());
		sw += a;
		ow += b;
	}
	(cells_digest(&c), sw, ow, c.bptt_prev_state().to_vec())
}

#[cfg(test)]
mod tests
{
	use super::*;

	/// Digests of the pre-ABI-35 walk (72b8535e) on pin_cases(), in order: (cells, s_writes, o_writes).
	const PRE_ABI35: &[(u64, usize, usize)] = &[
		(0x369380d8621e9bf8, 0x2b, 0x9ce),
		(0x3c207e8241c0236f, 0x26, 0xf7),
		(0x2385d4f292633fe5, 0x3c, 0x3d91),
		(0xdd4087fe3247d6d4, 0xe3, 0x3ecd),
		(0xc54a0bb1d1d3e8e5, 0x15, 0x7c0),
		(0x6f37887060c6bc0a, 0x13, 0x9f),
		(0xf2ae391d25e8662e, 0x61, 0x2f90),
		(0x89764cc41e27d84d, 0x103, 0x2ec2),
		(0x397a00fbba927ba2, 0x348, 0x1670),
		(0x2c7431c5cc3546fc, 0x190, 0x444),
		(0x274b19851591a17d, 0x180, 0x531e),
		(0x6a93ce519fc01c8e, 0x300, 0x5315),
	];


	/// Random Option-A overrides on a plan list (the GPU path ports the integral target).
	fn with_overrides(plans: &mut [crate::bptt_plan::BpttWindowPlan], sn: usize, mode: u8, seed: u64)
	{
		let mut rng = SmallRng::seed_from_u64(seed);
		let (t, f) = (crate::cell_mode::true_cell(mode), crate::cell_mode::false_cell(mode));
		for p in plans.iter_mut()
		{
			p.state_target = Some(
				(0..p.n_rec * sn)
					.map(|_| match rng.gen_range(0..3) { 0 => t, 1 => f, _ => crate::bptt_plan::NO_STATE_OVERRIDE })
					.collect(),
			);
		}
	}

	#[cfg(target_os = "macos")]
	fn gpu_case(shape: &WalkShape, eps: usize, steps: usize, win: usize, protect: bool, overrides: bool, budget: usize, topk: usize) -> Result<String, String>
	{
		let base = walk_controller(shape);
		let ep = walk_episodes(shape.seed, eps, steps);
		let mut plans = plan_episodes(&base, &ep, win);
		if overrides
		{
			with_overrides(&mut plans, shape.sn, shape.mode, shape.seed);
		}
		let cpu = walk_plans_cpu(&base, &plans, topk, protect);
		let mut g = base.clone();
		let (sw, ow) = crate::bptt_walk_gpu::walk_plans_gpu_with(&mut g, &plans, topk, protect, budget)?;
		let gpu = (cells_digest(&g), sw, ow, g.bptt_prev_state().to_vec());
		if cpu != gpu
		{
			return Err(format!("{shape:?} eps={eps} win={win} protect={protect} ov={overrides} budget={budget} topk={topk}: cpu {cpu:x?} != gpu {gpu:x?}"));
		}
		if cpu.2 == 0 || cpu.1 == 0
		{
			return Err(format!("{shape:?}: vacuous walk (writes s={} o={})", cpu.1, cpu.2));
		}
		Ok(format!("{shape:?} ok digest {:#x} s={} o={}", cpu.0, cpu.1, cpu.2))
	}

	/// THE ACCEPTANCE GATE: the GPU walk reproduces the CPU walk bit-for-bit — whole
	/// cell function of both layers, write counts and the carried recurrent state —
	/// over multi-window, multi-episode resident walks at production shape (sn 4/8,
	/// 24-bit neurons, 64 levels, obs-pwm + vertical/horizontal features) and a dense
	/// small shape, in QUAD/TERNARY/BINARY, with protect_learned, Option-A overrides,
	/// one-window dispatches + table growth, and topk 1/4/8.
	#[cfg(target_os = "macos")]
	#[test]
	fn cpu_walk_matches_gpu()
	{
		if metal::Device::system_default().is_none()
		{
			return;
		}
		let mut fails = Vec::new();
		let mut n = 0;
		let mut cases = pin_cases();
		// Dense tie-heavy shapes: few bits, most connections on the 8 state bits, so
		// beam ties and the transition solve decide many records.
		for (i, &mode) in [2u8, 0, 3].iter().enumerate()
		{
			let dense = WalkShape { mode, sn: 8, sbpn: 5, obpn: 5, levels: 8, bpf: 2, window_k: 2, prod: false, seed: 0xC35 + i as u64, tail: 0.8 };
			cases.push((dense, 3, 80, 32, false));
			cases.push((dense, 2, 64, 8, true));
		}
		for (shape, eps, steps, win, protect) in cases
		{
			for &(ov, budget, topk) in &[(false, 256usize, 4usize), (true, 1, 4), (false, 64, 8), (false, 256, 1)]
			{
				n += 1;
				match gpu_case(&shape, eps, steps, win, protect, ov, budget, topk)
				{
					Ok(_) => {}
					Err(e) => fails.push(e),
				}
			}
		}
		assert!(fails.is_empty(), "{}/{} GPU walk cases diverged: {fails:#?}", fails.len(), n);
	}

	/// Whole reward-gated DAgger, CPU walk vs GPU walk: identical cells, per-round
	/// writes, and every closed-loop eval statistic (rounds 2+ roll out on the
	/// GPU-trained memory, so any divergence compounds into these).
	#[cfg(target_os = "macos")]
	#[test]
	fn dagger_train_cpu_matches_gpu()
	{
		if metal::Device::system_default().is_none()
		{
			return;
		}
		for &(mode, sn) in &[(2u8, 4usize), (0u8, 4usize), (3u8, 4usize), (2u8, 8usize)]
		{
			let cpu = dagger_digest_with(mode, sn, false);
			let before = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
			let gpu = dagger_digest_with(mode, sn, true);
			let after = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
			assert!(after > before, "mode {mode} sn {sn}: the GPU walk never ran (fell back to the CPU)");
			assert!(cpu.0 .1.iter().sum::<usize>() > 0, "mode {mode} sn {sn}: vacuous dagger_train");
			assert_eq!(cpu, gpu, "mode {mode} sn {sn}: GPU dagger_train diverged from the CPU trainer");
		}
	}

	/// Wall-time of the CPU vs GPU walk at production shape (prints; a human reads it).
	/// cargo test --release ... bench_bptt_walk -- --ignored --nocapture
	#[cfg(target_os = "macos")]
	#[test]
	#[ignore]
	fn bench_bptt_walk()
	{
		for &sn in &[4usize, 8]
		{
			let shape = WalkShape { mode: 2, sn, sbpn: 24, obpn: 24, levels: 64, bpf: 16, window_k: 2, prod: true, seed: 0xBE7C, tail: 0.3 };
			let mut base = walk_controller(&shape);
			// Warm memory: two passes so the solves scan real tables.
			let warm = walk_episodes(shape.seed ^ 1, 8, 1000);
			walk_cpu_public(&mut base, &warm, 32, 4, false);
			let ep = walk_episodes(shape.seed, 4, 400);
			let plans = plan_episodes(&base, &ep, 32);
			let recs: usize = plans.iter().map(|p| p.n_rec).sum();
			let t0 = std::time::Instant::now();
			let cpu = walk_plans_cpu(&base, &plans, 4, false);
			let tc = t0.elapsed().as_secs_f64();
			let mut g = base.clone();
			let t1 = std::time::Instant::now();
			let r = crate::bptt_walk_gpu::walk_plans_gpu(&mut g, &plans, 4, false).expect("gpu");
			let tg = t1.elapsed().as_secs_f64();
			assert_eq!((cpu.1, cpu.2), r);
			println!(
				"sn={sn} records={recs} cells={} cpu {:.3}s ({:.3} ms/rec) gpu {:.3}s ({:.3} ms/rec) speedup {:.2}x",
				base.total_cells(), tc, tc * 1e3 / recs as f64, tg, tg * 1e3 / recs as f64, tc / tg
			);
		}
	}

	/// Lamarckian egress is CANONICAL: two controllers holding the same cells export
	/// them in the same order whatever their map's capacity history (the CPU walk
	/// inserts-then-deletes cells the GPU apply-back never touches, so their tables
	/// grow differently). GA-Memory's index-keyed mutate_values depends on it.
	#[test]
	fn egress_order_independent_of_capacity_history()
	{
		let shape = WalkShape { mode: 3, sn: 4, sbpn: 12, obpn: 12, levels: 8, bpf: 4, window_k: 2, prod: false, seed: 0xE9, tail: 0.3 };
		let base = walk_controller(&shape);
		let default = crate::cell_mode::canonical_default_cell(shape.mode);
		let other = if default == 0 { 1 } else { 0 };
		let cells: Vec<(usize, u64)> = (0..3000u64).map(|i| ((i % 32) as usize, i.wrapping_mul(0x9E37_79B9) & 0xFFF)).collect();
		let (a, b) = (base.clone(), base.clone());
		for &(n, ad) in &cells
		{
			a.bptt_layer(false).write_cell(n, ad, other, true);
		}
		// b: grow every map far past a's capacity, then delete the extras.
		for n in 0..32usize
		{
			for ad in 0x1000u64..0x1000 + 4096
			{
				b.bptt_layer(false).write_cell(n, ad, other, true);
			}
		}
		for &(n, ad) in cells.iter().rev()
		{
			b.bptt_layer(false).write_cell(n, ad, other, true);
		}
		for n in 0..32usize
		{
			for ad in 0x1000u64..0x1000 + 4096
			{
				b.bptt_layer(false).write_cell(n, ad, default, true);
			}
		}
		assert_eq!(cells_digest(&a), cells_digest(&b), "fixture: same cell function");
		assert_eq!(a.export_cells(), b.export_cells(), "export_cells order depends on map history");
	}

	/// The plan-list CPU walk IS the public bptt_train_window walk (reset handling).
	#[test]
	fn plan_list_walk_equals_public_walk()
	{
		for (shape, eps, steps, win, protect) in pin_cases().into_iter().take(4)
		{
			let base = walk_controller(&shape);
			let ep = walk_episodes(shape.seed, eps, steps);
			let mut c = base.clone();
			let (sw, ow) = walk_cpu_public(&mut c, &ep, win, 4, protect);
			let plans = plan_episodes(&base, &ep, win);
			let (d, s2, o2, prev) = walk_plans_cpu(&base, &plans, 4, protect);
			assert_eq!((cells_digest(&c), sw, ow, c.bptt_prev_state().to_vec()), (d, s2, o2, prev), "{shape:?}");
		}
	}

	#[test]
	fn cpu_dagger_bit_identical_to_pre_abi35()
	{
		let got: Vec<(u64, Vec<usize>)> = [(2u8, 4usize), (3u8, 4usize)].iter().map(|&(m, sn)| dagger_digest(m, sn)).collect();
		println!("PRE_ABI35_DAGGER = {got:#x?}");
		assert!(got.iter().all(|g| g.1.iter().sum::<usize>() > 0), "a pinned dagger_train wrote nothing: {got:?}");
		let want: Vec<(u64, Vec<usize>)> = PRE_ABI35_DAGGER.iter().map(|(d, w)| (*d, w.to_vec())).collect();
		assert_eq!(got, want, "the ABI 35 CPU dagger_train drifted from the pre-ABI-35 trainer");
	}

	/// Pre-ABI-35 (72b8535e) whole-dagger_train digests: (cells, per-round writes).
	const PRE_ABI35_DAGGER: &[(u64, &[usize])] = &[(0x2cfa91b6962aaf53, &[0x296, 0x119]), (0x6922b3ae1abbbb6d, &[0x2927, 0x0])];

	#[test]
	fn cpu_walk_bit_identical_to_pre_abi35()
	{
		let cases = pin_cases();
		let mut got = Vec::new();
		for (shape, eps, steps, win, protect) in &cases
		{
			let mut c = walk_controller(shape);
			let ep = walk_episodes(shape.seed, *eps, *steps);
			let (sw, ow) = walk_cpu_public(&mut c, &ep, *win, 4, *protect);
			got.push((cells_digest(&c), sw, ow));
		}
		println!("PRE_ABI35 = {got:#x?}");
		assert!(got.iter().all(|g| g.2 > 0), "a pinned walk wrote no output cells: {got:?}");
		assert_eq!(got.len(), cases.len());
		assert_eq!(got.as_slice(), PRE_ABI35, "the ABI 35 CPU walk drifted from the pre-ABI-35 walk");
	}
}
