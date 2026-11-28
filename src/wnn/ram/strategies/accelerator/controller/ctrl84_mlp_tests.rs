//! CTRL-84 (ABI 71): the MLP baseline head — flag-off pin, CPU/GPU parity, determinism,
//! the DAgger loop, and the per-decision cost bench.
//!
//! * FLAG-OFF: no head ⇒ the cf21 lineage is the ABI-68 one. The CTRL-86 digests (trainer
//!   cells, feature stream, CPU rows, Metal rows, rivals, calibration — captured on ABI 64,
//!   held through 65-68) must not move; and a head installed then cleared leaves the
//!   controller's rollout bit-identical to one that never had a head.
//! * FORWARD PARITY: the kernel's `mlp_forward` == `MlpHead::forward` BIT FOR BIT (ReLU; both
//!   are the same fma chain), tanh within 1e-6.
//! * ROLLOUT PARITY: the CPU scorer and the Metal scorer (FC_MLP specialization) on the same MLP
//!   controller and episodes, within the CTRL-54 aggregate tolerance (the scorers differ in the
//!   feature pipeline's precision, not in the head).
//! * DETERMINISM: the same head scores bit-identically twice; dagger_train_mlp_with with a
//!   deterministic fit is a pure function of (controller, cfg, seed).

use crate::abi38_tests::ctbr_student;
use crate::ctrl86_flag_off_pins::{cpu_rows_on, digests, gpu_rows_on, l2b_cfg, real_life_plant, stage2_with, Spelling, ABI64_CF21};
use crate::mlp_dagger::{dagger_train_mlp_with, MlpRecords};
use crate::mlp_head::{param_count, MlpAct, MlpHead};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn random_head(in_dim: usize, hidden: &[usize], act: MlpAct, w_scale: f32, bias_out: f32, seed: u64) -> MlpHead
{
	let mut rng = SmallRng::seed_from_u64(seed);
	let n = param_count(in_dim, hidden, 4);
	let mut params: Vec<f32> = (0..n).map(|_| rng.gen_range(-w_scale..w_scale)).collect();
	// The output layer's biases are the last 4 params: centre the policy on `bias_out`.
	for b in params[n - 4..].iter_mut()
	{
		*b = bias_out;
	}
	let shift: Vec<f32> = (0..in_dim).map(|_| rng.gen_range(-0.1f32..0.1)).collect();
	let scale: Vec<f32> = (0..in_dim).map(|i| if i % 7 == 3 { 0.0 } else { rng.gen_range(0.5f32..2.0) }).collect();
	MlpHead::new(in_dim, hidden.to_vec(), 4, act, shift, scale, params).expect("random head")
}

/// A cascade student flying a gentle MLP head (small weights around the neutral decode).
fn mlp_student(hidden: &[usize], seed: u64) -> crate::controller::WnnController
{
	let mut c = ctbr_student(0, 10, 0x8400 + seed);
	let h = random_head(c.num_features(), hidden, MlpAct::Relu, 0.02, c.neutral_pub(), seed);
	c.set_mlp_head_core(h).expect("mlp student");
	c
}

#[test]
fn ctrl84_cf21_flag_off_bit_identical_to_abi68()
{
	let d = digests(Spelling::Legacy);
	eprintln!("[ctrl84-pin] {d:#018x?}");
	assert_eq!(d, ABI64_CF21, "the cf21 lineage moved with no MLP head installed");
}

#[test]
fn ctrl84_head_set_then_cleared_is_the_wnn()
{
	let (c, _) = crate::ctrl86_flag_off_pins::trained_on(&l2b_cfg(4, Spelling::Legacy));
	let s1 = stage2_with(real_life_plant(Spelling::Legacy));
	let plain = cpu_rows_on(&c, &s1, 400);
	let mut d = c.clone();
	d.set_mlp_head_core(random_head(d.num_features(), &[8], MlpAct::Relu, 0.1, 0.5, 3)).unwrap();
	let with_head = cpu_rows_on(&d, &s1, 400);
	d.clear_mlp_head();
	let cleared = cpu_rows_on(&d, &s1, 400);
	assert_eq!(plain.map(f64::to_bits), cleared.map(f64::to_bits), "clearing the head did not restore the WNN");
	assert_ne!(plain.map(f64::to_bits), with_head.map(f64::to_bits), "vacuous: the head did not change the rollout");
	if let (Some(g0), Some(g1)) = (gpu_rows_on(&c, &s1, 400), gpu_rows_on(&d, &s1, 400))
	{
		assert_eq!(
			g0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
			g1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
			"Metal: a cleared head is not the default pipeline"
		);
	}
}

#[test]
fn ctrl84_mlp_forward_cpu_gpu_bit_identical()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok()
	{
		return;
	}
	let Ok(ev) = crate::metal_controller::ControllerRolloutEvaluator::new()
	else
	{
		assert!(metal::Device::system_default().is_none(), "GPU present but the evaluator failed to build");
		return;
	};
	let n = 384usize;
	for (hidden, act, seed) in [(vec![64usize, 64], MlpAct::Relu, 1u64), (vec![576, 576], MlpAct::Relu, 2), (vec![32], MlpAct::Relu, 3), (vec![64, 64], MlpAct::Tanh, 4)]
	{
		let w = 0.6 / (*hidden.iter().max().unwrap() as f32).sqrt();
		let h = random_head(24, &hidden, act, w, 0.5, seed);
		let mut rng = SmallRng::seed_from_u64(100 + seed);
		let feats: Vec<f32> = (0..n * 24).map(|_| rng.gen_range(-3.0f32..3.0)).collect();
		let gpu = ev.mlp_forward_gpu(&h, &feats, n).expect("gpu forward");
		let mut worst = 0.0f32;
		let mut moved = 0usize;
		for r in 0..n
		{
			let mut cpu = [0.0f32; 4];
			h.forward(&feats[r * 24..(r + 1) * 24], &mut cpu);
			for k in 0..4
			{
				let g = gpu[r * 4 + k];
				worst = worst.max((g - cpu[k]).abs());
				moved += usize::from(cpu[k] > 0.0 && cpu[k] < 1.0);
				if act == MlpAct::Relu
				{
					assert_eq!(g.to_bits(), cpu[k].to_bits(), "{hidden:?} relu row {r} out {k}: gpu {g} cpu {}", cpu[k]);
				}
			}
		}
		eprintln!("[ctrl84] forward {hidden:?} {act:?}: worst |gpu-cpu| {worst:.3e}, {moved}/{} unclamped outputs", n * 4);
		assert!(worst <= 1e-6, "{hidden:?} {act:?}: worst |gpu-cpu| {worst}");
		assert!(moved > n, "vacuous: the outputs are mostly clamped");
	}
}

fn rel(a: f64, b: f64) -> f64
{
	(a - b).abs() / a.abs().max(b.abs()).max(1e-12)
}

#[test]
fn ctrl84_mlp_rollout_cpu_matches_gpu()
{
	let s1 = stage2_with(real_life_plant(Spelling::Legacy));
	for (hidden, seed) in [(vec![64usize, 64], 11u64), (vec![96], 12)]
	{
		let c = mlp_student(&hidden, seed);
		let cpu = cpu_rows_on(&c, &s1, 600);
		let Some(gpu) = gpu_rows_on(&c, &s1, 600)
		else
		{
			return;
		};
		eprintln!("[ctrl84] rollout {hidden:?}: cpu err {:.6e} stable {} steady {:.6e} | gpu err {:.6e} stable {} steady {:.6e}", cpu[1], cpu[2], cpu[5], gpu[1], gpu[2], gpu[5]);
		assert_eq!(gpu[2], cpu[2], "{hidden:?}: stable rate");
		for (i, name) in [(0usize, "reward"), (1, "err"), (5, "steady")]
		{
			let r = rel(gpu[i], cpu[i]);
			assert!(r <= 2e-3, "{hidden:?} {name}: gpu {} cpu {} rel {r:.3e} > 2e-3 (CTRL-54 aggregate tolerance)", gpu[i], cpu[i]);
		}
	}
}

#[test]
fn ctrl84_mlp_scoring_is_deterministic()
{
	let s1 = stage2_with(real_life_plant(Spelling::Legacy));
	let c = mlp_student(&[64, 64], 21);
	let a = cpu_rows_on(&c, &s1, 500);
	let b = cpu_rows_on(&c.clone(), &s1, 500);
	assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
}

/// A deterministic stand-in for the PyTorch fit: the per-bank mean target of the aggregated
/// records (zero weights), so the loop's data flow is testable in cargo.
fn mean_fit(agg: &mut MlpRecords, rec: &MlpRecords, in_dim: usize) -> Result<Option<MlpHead>, String>
{
	agg.feats.extend_from_slice(&rec.feats);
	agg.targets.extend_from_slice(&rec.targets);
	agg.n += rec.n;
	let mut n = param_count(in_dim, &[8], 4);
	let mut params = vec![0.0f32; n];
	for m in 0..4
	{
		let s: f64 = (0..agg.n).map(|r| agg.targets[r * 4 + m] as f64).sum();
		n = params.len();
		params[n - 4 + m] = (s / agg.n.max(1) as f64) as f32;
	}
	MlpHead::new(in_dim, vec![8], 4, MlpAct::Relu, vec![0.0; in_dim], vec![1.0; in_dim], params).map(Some)
}

fn smoke_cfg() -> crate::dagger_train::RewardGatedConfigPacked
{
	let mut cfg = l2b_cfg(4, Spelling::Legacy);
	cfg.num_rounds = 3;
	cfg.episodes_per_round = 3;
	cfg.eval_episodes = 2;
	cfg
}

fn run_smoke(seed: u64) -> (crate::dagger_train::TrainStats, MlpRecords, Vec<u32>)
{
	let cfg = smoke_cfg();
	let mut c = mlp_student(&[8], 31);
	let in_dim = c.num_features();
	let mut agg = MlpRecords::default();
	let st = dagger_train_mlp_with(&mut c, &cfg, [0.0; 3], seed, &mut |rec, _round| {
		assert_eq!(rec.feats.len(), rec.n * in_dim, "feature records misaligned");
		assert_eq!(rec.targets.len(), rec.n * 4, "target records misaligned");
		mean_fit(&mut agg, rec, in_dim)
	})
	.expect("mlp dagger");
	let p = c.mlp_head_ref().expect("head kept").params().iter().map(|v| v.to_bits()).collect();
	(st, agg, p)
}

#[test]
fn ctrl84_mlp_dagger_records_and_determinism()
{
	let (st, agg, p) = run_smoke(84);
	let cfg = smoke_cfg();
	let decisions_per_ep = cfg.steps_per_episode.div_ceil(10);
	assert_eq!(st.iter_cells_written.len(), cfg.num_rounds);
	assert!(agg.n > 0, "no records");
	// Every gated episode replays to its decision steps (a diverged episode is shorter).
	let gated: usize = st.iter_n_trained.iter().sum();
	assert!(agg.n <= gated * decisions_per_ep, "{} records > {gated} episodes × {decisions_per_ep} decisions", agg.n);
	assert!(agg.targets.iter().all(|t| (0.0..=1.0).contains(t)), "a raw-decode target outside [0, 1]");
	assert!(agg.feats.iter().all(|v| v.is_finite()));
	let spread = (0..4).any(|m| {
		let col: Vec<f32> = (0..agg.n).map(|r| agg.targets[r * 4 + m]).collect();
		col.iter().cloned().fold(f32::MIN, f32::max) - col.iter().cloned().fold(f32::MAX, f32::min) > 1e-3
	});
	assert!(spread, "vacuous: the teacher's targets are constant");
	let (st2, agg2, p2) = run_smoke(84);
	assert_eq!(st.iter_fitness.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), st2.iter_fitness.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
	assert_eq!(agg.feats.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), agg2.feats.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
	assert_eq!(p, p2, "the kept head differs between identical runs");
	let (_, agg3, _) = run_smoke(85);
	assert!(agg.feats != agg3.feats, "the seed does not reach the episodes");
}

#[test]
fn ctrl84_refusals()
{
	let mut c = ctbr_student(2, 10, 7);
	let h = random_head(c.num_features(), &[8], MlpAct::Relu, 0.1, 0.5, 1);
	assert!(c.set_mlp_head_core(h).unwrap_err().contains("state_neurons"));
	let mut c = ctbr_student(0, 10, 7);
	let h = random_head(c.num_features() + 1, &[8], MlpAct::Relu, 0.1, 0.5, 1);
	assert!(c.set_mlp_head_core(h).unwrap_err().contains("num_features"));
	let a = mlp_student(&[8], 1);
	let b = ctbr_student(0, 10, 9);
	assert!(crate::metal_controller::mlp_population(&[&a, &b]).is_err(), "a mixed MLP/WNN population must be refused");
	let d = mlp_student(&[9], 2);
	assert!(crate::metal_controller::mlp_population(&[&a, &d]).is_err(), "mixed head shapes must be refused");
	let mut cfg = smoke_cfg();
	cfg.num_rounds = 1;
	let mut plain = ctbr_student(0, 10, 9);
	assert!(dagger_train_mlp_with(&mut plain, &cfg, [0.0; 3], 1, &mut |_, _| Ok(None)).is_err(), "no initial head must be refused");
}

/// The production feature set (abi38_pins::prod_student, no obs_pwm) at an arbitrary output
/// shape — the bench's WNN.
fn wnn_like(levels: usize, bpf: usize, obpn: usize, seed: u64) -> crate::controller::WnnController
{
	let nf = 9 + 2 + 2 + 2 + 3 + 4;
	let frame_bits = nf * bpf;
	let mut rng = SmallRng::seed_from_u64(seed);
	let thresholds: Vec<f32> = (0..frame_bits).map(|_| rng.gen_range(-0.4f32..0.4)).collect();
	let oconn: Vec<i64> = (0..4 * levels * obpn).map(|_| rng.gen_range(0..frame_bits) as i64).collect();
	crate::controller::WnnController::new_core(
		4, levels, bpf, 1, 0, 0, obpn, thresholds, Vec::new(), oconn,
		true, 0.1, 0.95, 1.0,
		false, false, true, true, false, false, true, true,
		0.99, 1.0, crate::abi38_pins::DT, false, 10,
		ram_core::neuron_memory::BINARY, Some(crate::cell_mode::DECODE_ANTAGONIST), None, 0.05, false, 0.30,
		true, true, true, true, true, false, 1,
	)
	.expect("bench student")
}

/// Per-decision CPU cost (µs), release build only:
///   cargo test --release -p ram_controller --lib --no-default-features ctrl84_bench -- --ignored --nocapture
/// The MLP forward alone, and one DECISION step of the whole controller (features + head /
/// features + 256-neuron lookup over a memory of `WNN_KEYS` TRUE keys).
#[test]
#[ignore]
fn ctrl84_bench_per_decision_cost()
{
	const WNN_KEYS: usize = 346_000; // l2bnm s1-s3 mean TRUE keys (296,260 / 386,398 / 356,077)
	let reps = 2000usize;
	let mut rng = SmallRng::seed_from_u64(9);
	let x: Vec<f32> = (0..24).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
	for (name, hidden) in [("a 2x64", vec![64usize, 64]), ("b 2x576", vec![576, 576])]
	{
		let h = random_head(24, &hidden, MlpAct::Relu, 0.1, 0.5, 5);
		let mut out = [0.0f32; 4];
		let t = std::time::Instant::now();
		for _ in 0..reps
		{
			h.forward(std::hint::black_box(&x), &mut out);
		}
		let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
		let (macs, acts, floats) = h.cost();
		eprintln!("[ctrl84-bench] MLP {name}: forward {us:.2} µs/decision | {macs} MAC, {acts} activations, {} B params read (fp32)", floats * 4);
	}
	// Whole decision step: the l2bnm output layer's shape (256 neurons = 4 banks x 64 levels,
	// 24-bit addresses, 16 thresholds per feature) holding WNN_KEYS random TRUE keys, vs the
	// same controller with each head (features, decode and the inner rate loop included).
	let mut c = wnn_like(64, 16, 24, 77);
	c.enable_ctbr(crate::abi38_tests::spec()).unwrap();
	let n_out = c.num_motors() * c.levels_per_motor();
	for k in 0..WNN_KEYS
	{
		let _ = c.write_output_cell_internal(k % n_out, rng.gen_range(0..(1u64 << 24)), 1);
	}
	let mut time_decisions = |c: &mut crate::controller::WnnController, label: &str| {
		c.reset(0.0);
		let t = std::time::Instant::now();
		for i in 0..reps * 10
		{
			let g = [0.01 * (i % 7) as f32, -0.02, 0.005];
			let _ = c.step(g, [0.0, 0.0, 9.81], [0.0; 3]);
		}
		let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
		eprintln!("[ctrl84-bench] {label}: {us:.2} µs per 10-tick decision window (1 decision + 9 holds)");
	};
	time_decisions(&mut c, &format!("WNN lookup (256 neurons x 24 bits, {WNN_KEYS} keys)"));
	for hidden in [vec![64usize, 64], vec![576, 576]]
	{
		let mut m = c.clone();
		m.set_mlp_head_core(random_head(m.num_features(), &hidden, MlpAct::Relu, 0.1, 0.5, 6)).unwrap();
		time_decisions(&mut m, &format!("MLP {hidden:?} head"));
	}
}
