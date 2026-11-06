//! MPC teacher QP solver: convergence measurement + correctness pins (01/11/2026).
//!
//! The mpc/mpcof teachers solve, every step, the condensed box-constrained QP
//!     min ½UᵀHU + gᵀU   s.t.  |U_i| ≤ a        (H 3N×3N, N = 25 → 75×75)
//! by projected FISTA (cap 200 iterations, stop when max |ΔU| < 1e-9). These
//! tests measure — on rollouts of the live stage-1/stage-2 regimes — how often
//! the box is active, how often FISTA hits its cap, and how far its exit point
//! is from the EXACT optimum (an active-set solve verified by KKT).
//!
//! Child module of `optimal` so it can read the solver's private matrices.

use super::*;
use crate::controller::{AttitudeSim, WnnController};
use crate::dagger_train::{
	rollout_and_label_rs, train_on_trajectory_rs, AirframeRs, RewardGatedConfigPacked, TEACHER_HOVER_DERIVED,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};

const CF21_ATT: [f64; 12] = [6.0, 3.0, 0.0, 20.0, 6.0, 3.0, 0.0, 20.0, 6.0, 1.0, 0.35, 360.0];
const CF21_RATE: [f64; 12] = [200.0, 400.0, 2.5, 33.3, 200.0, 400.0, 2.5, 33.3, 120.0, 16.7, 0.0, 166.7];
const TILT_RAD: f64 = 5.0 * std::f64::consts::PI / 180.0;

/// The live recipe's plant: cf21_brushless, L4C (torque jitter 0.20, motor asym
/// ±0.20 — one fixed draw, ADIS16448 noise), translation, specific-force IMU,
/// rotor drag 0.78, mpcof at the derived hover. `xy` = 0.5 (stage 2, tilt-ref
/// attitude error) or 0.0 (stage 1).
fn regime(xy: f32, steps: usize, expert_drives: bool) -> RewardGatedConfigPacked
{
	RewardGatedConfigPacked {
		num_rounds: 1, episodes_per_round: 1, steps_per_episode: steps,
		bptt_window: 32, topk_per_neuron: 4, protect_learned: false,
		gate_mode: 0, gate_use_best: false, gate_window: 0,
		gate_quantile: 0.5, gate_running: true, target_source: 0,
		teacher: 4, teacher_schedule: vec![], teacher_blend: vec![],
		keep_best_checkpoint: true, explore_eps: 0.0, explore_scale: 0.1,
		curriculum: false, easy_tilt_deg: 5.0, full_tilt_deg: 5.0,
		dt: 0.001, max_initial_yaw_rad: 0.0,
		max_initial_body_rate: 0.0, max_initial_yaw_rate: 0.0,
		eval_episodes: 1,
		split_tau: 0.1, split_clean_gain: 0.999, split_accum_corr: 0.9,
		split_max_rounds: 5, split_k_start: 1, split_coarse_target: 32,
		split_selective_output: true,
		active_roll: true, active_pitch: true, active_yaw: true,
		dist_enabled: true, dist_tau_bias: [0.0; 3],
		dist_gust_sigma: 0.0, dist_gust_tau_c: 0.1,
		dist_motor_asym: [1.13, 0.86, 1.07, 0.91],
		dist_gyro_sigma: 3.394e-4 * 31.6228, dist_gyro_bias_walk: 3.8785e-5,
		dist_accel_sigma: 4.0e-3 * 31.6228,
		dist_dropout_prob: 0.0, dist_dropout_len_steps: 0,
		dist_obs_delay_steps: 0, dist_torque_scale_jitter: 0.20,
		expert_drives,
		af_arm_length: 0.070710678, af_k_thrust: 0.2, af_k_drag: 0.00569278844371417,
		af_inertia: [3.004e-5, 3.019e-5, 5.304e-5], af_gravity: 9.81,
		af_pid_att: CF21_ATT, af_pid_rate: CF21_RATE,
		af_pid_out_limit_n: 0.09999847409781033, af_pid_hover_n: 0.09638325,
		af_pid_attitude_hz: 500.0, af_pid_lpf_hz: 30.0,
		write_priority_err: false, write_err_floor_deg: 0.0,
		translation: true, af_mass: 0.0393, mass_jitter: 0.0,
		alt_offset: 0.0, init_vz: 0.0, collective_jitter: 0.0,
		target_altitude: 0.0,
		alt_pd_omega: 2.0, alt_pd_zeta: 1.0, alt_pd_max_delta: 0.25,
		xy_offset: xy, lambda_pos: 0.0, lambda_alt: 0.0,
		pos_omega: 1.0, pos_zeta: 1.0, pos_max_tilt_rad: 0.5236,
		teacher_hover_mode: TEACHER_HOVER_DERIVED,
		motor_lag_s: 0.0,
		imu_specific_force: true, rotor_drag: 0.78, att_err_tilt_ref: xy > 0.0,
		dist_motor_asym_mag: 0.0,
	}
}

/// sn=0 BINARY delta student at the recipe width (16 levels, 24 bits/neuron)
/// with the stage-1 vertical + stage-2 horizontal channels and the estimator.
fn student(seed: u64) -> WnnController
{
	let (levels, bpf, obpn, nf) = (16usize, 3usize, 24usize, 18usize);
	let frame_bits = nf * bpf;
	let mut rng = SmallRng::seed_from_u64(seed);
	let thresholds: Vec<f32> = (0..frame_bits)
		.map(|i| match i / bpf
		{
			9 | 10 => rng.gen_range(-0.3f32..0.3),
			f if f >= 11 => rng.gen_range(-0.6f32..0.6),
			_ => rng.gen_range(-5.0f32..5.0),
		})
		.collect();
	let out_conn: Vec<i64> = (0..4 * levels * obpn).map(|_| rng.gen_range(0..frame_bits) as i64).collect();
	let mut c = WnnController::new_core(
		4, levels, bpf, 2, 0, 0, obpn, thresholds, Vec::new(), out_conn,
		true, 0.05, 0.95, 1.0,
		false, false, true, false, false, false, false, false,
		0.99, 1.0, 0.001, false, 1, ram_core::neuron_memory::BINARY, None, None, 0.05, false, 0.30,
		true, true, true,
		true, true,
		false, 1,
	)
	.expect("student must construct");
	c.enable_att_estimator(2.0, 0.1).expect("estimator on");
	c
}

fn mpc_of(t: &mut Teacher) -> &mut AttitudeMpcRs
{
	match t
	{
		Teacher::MpcOf(m) => &mut m.mpc,
		Teacher::Mpc(m) => m,
		_ => panic!("not an MPC teacher"),
	}
}

/// Fly `episodes` rollouts of `cfg`; DAgger-train the student between them when
/// the student drives. Returns every solve the teacher made.
fn collect(cfg: &RewardGatedConfigPacked, episodes: usize, seed: u64) -> Vec<MpcProbe>
{
	let af = AirframeRs::from_cfg(cfg);
	let mut teacher = af.teacher(4);
	let mut sim: AttitudeSim = af.sim();
	let mut c = student(seed ^ 0x5EED);
	let mut rng = SmallRng::seed_from_u64(seed);
	mpc_of(&mut teacher).probe = Some(Vec::new());
	for _ in 0..episodes
	{
		let traj = rollout_and_label_rs(&mut c, &mut teacher, &mut sim, cfg, TILT_RAD, &mut rng, [0.0; 3]);
		if !cfg.expert_drives
		{
			let _ = train_on_trajectory_rs(&mut c, &traj, cfg);
		}
	}
	mpc_of(&mut teacher).probe.take().unwrap()
}

fn objective(h: &Mat, g: &[f64], u: &[f64]) -> f64
{
	let hu = h.mul_vec(u);
	(0..u.len()).map(|i| 0.5 * u[i] * hu[i] + g[i] * u[i]).sum()
}

/// Natural (projected-gradient) residual max_i |u_i − P(u_i − ∇f_i)|, ∇f = Hu+g.
fn kkt_residual(h: &Mat, g: &[f64], a: f64, u: &[f64]) -> f64
{
	let hu = h.mul_vec(u);
	(0..u.len()).map(|i| (u[i] - clamp_f64(u[i] - (hu[i] + g[i]), -a, a)).abs()).fold(0.0, f64::max)
}

/// EXACT box-QP optimum by primal active-set iteration from `start`: fix the
/// clamped coordinates, solve the free block exactly, move violators between
/// sets until the KKT conditions hold. Returns None if it does not settle.
fn exact_box_qp(h: &Mat, g: &[f64], a: f64, start: &[f64]) -> Option<Vec<f64>>
{
	let n = g.len();
	// bound[i]: 0 free, +1 at +a, −1 at −a
	let mut bound: Vec<i8> = start
		.iter()
		.map(|&u| if u >= a { 1 } else if u <= -a { -1 } else { 0 })
		.collect();
	for _ in 0..500
	{
		let free: Vec<usize> = (0..n).filter(|&i| bound[i] == 0).collect();
		let mut u = vec![0.0; n];
		for i in 0..n
		{
			u[i] = bound[i] as f64 * a;
		}
		if !free.is_empty()
		{
			let nf = free.len();
			let mut hff = Mat::zeros(nf, nf);
			let mut rhs = vec![0.0; nf];
			for (r, &i) in free.iter().enumerate()
			{
				let mut acc = -g[i];
				for j in 0..n
				{
					if bound[j] != 0
					{
						acc -= h.get(i, j) * u[j];
					}
				}
				rhs[r] = acc;
				for (cc, &j) in free.iter().enumerate()
				{
					hff.set(r, cc, h.get(i, j));
				}
			}
			let sol = inverse(&hff)?.mul_vec(&rhs);
			for (r, &i) in free.iter().enumerate()
			{
				u[i] = sol[r];
			}
		}
		// Primal violators → bind the worst; else dual violators → release the worst.
		let mut worst_p = (0.0f64, usize::MAX);
		for i in 0..n
		{
			if bound[i] == 0 && u[i].abs() - a > worst_p.0
			{
				worst_p = (u[i].abs() - a, i);
			}
		}
		if worst_p.1 != usize::MAX && worst_p.0 > 1e-13
		{
			let i = worst_p.1;
			bound[i] = if u[i] > 0.0 { 1 } else { -1 };
			continue;
		}
		let hu = h.mul_vec(&u);
		let mut worst_d = (0.0f64, usize::MAX);
		for i in 0..n
		{
			// at +a the gradient must be ≤ 0; at −a it must be ≥ 0
			let viol = bound[i] as f64 * (hu[i] + g[i]);
			if bound[i] != 0 && viol > worst_d.0
			{
				worst_d = (viol, i);
			}
		}
		if worst_d.1 != usize::MAX && worst_d.0 > 1e-12
		{
			bound[worst_d.1] = 0;
			continue;
		}
		// Clamp round-off on the free block, then return the KKT point.
		for i in 0..n
		{
			u[i] = clamp_f64(u[i], -a, a);
		}
		return Some(u);
	}
	None
}

/// (λmax, λmin) of SPD H by power iteration on H and on H⁻¹.
fn extreme_eigs(h: &Mat) -> (f64, f64)
{
	let power = |m: &Mat| -> f64 {
		let mut v = vec![1.0f64; m.r];
		let mut l = 0.0;
		for _ in 0..20000
		{
			let w = m.mul_vec(&v);
			let n = w.iter().map(|x| x * x).sum::<f64>().sqrt();
			l = n;
			v = w.iter().map(|x| x / n).collect();
		}
		l
	};
	(power(h), 1.0 / power(&inverse(h).unwrap()))
}

#[derive(Default)]
struct Report
{
	steps: usize,
	constrained: usize,
	cap_hits: usize,
	iters: Vec<usize>,
	max_resid_exit: f64,
	max_resid_exit_cap: f64,
	max_obj_gap: f64,
	max_du_first: f64,
	max_du_all: f64,
	n_first_gt_1e6: usize,
	n_first_gt_1e4: usize,
	max_resid_ref: f64,
	ref_failed: usize,
	/// per constrained step: max over motors of |Δpwm| the first control moves
	/// the '+'-mixed label (pre-clamp), exit point vs exact optimum
	dpwm: Vec<f64>,
}

/// Max per-motor pwm change the '+' mixer turns a first-control change into.
fn pwm_delta(du: [f64; 3]) -> f64
{
	let [r, p, y] = du;
	[(-p + y).abs(), (-r - y).abs(), (p + y).abs(), (r - y).abs()].into_iter().fold(0.0, f64::max)
}

fn analyse(m: &AttitudeMpcRs, probes: &[MpcProbe]) -> Report
{
	let mut r = Report { steps: probes.len(), ..Default::default() };
	let a = m.authority;
	for p in probes.iter().filter(|p| p.constrained)
	{
		r.constrained += 1;
		r.iters.push(p.iters);
		if !p.converged
		{
			r.cap_hits += 1;
		}
		let g = m.g_map.mul_vec(&p.x);
		let res = kkt_residual(&m.h, &g, a, &p.u);
		r.max_resid_exit = r.max_resid_exit.max(res);
		if !p.converged
		{
			r.max_resid_exit_cap = r.max_resid_exit_cap.max(res);
		}
		let Some(us) = exact_box_qp(&m.h, &g, a, &p.u)
		else
		{
			r.ref_failed += 1;
			continue;
		};
		r.max_resid_ref = r.max_resid_ref.max(kkt_residual(&m.h, &g, a, &us));
		let gap = objective(&m.h, &g, &p.u) - objective(&m.h, &g, &us);
		r.max_obj_gap = r.max_obj_gap.max(gap);
		let du_first = (0..3).map(|i| (p.u[i] - us[i]).abs()).fold(0.0, f64::max);
		let du_all = (0..us.len()).map(|i| (p.u[i] - us[i]).abs()).fold(0.0, f64::max);
		r.dpwm.push(pwm_delta([p.u[0] - us[0], p.u[1] - us[1], p.u[2] - us[2]]));
		r.max_du_first = r.max_du_first.max(du_first);
		r.max_du_all = r.max_du_all.max(du_all);
		if du_first > 1e-6
		{
			r.n_first_gt_1e6 += 1;
		}
		if du_first > 1e-4
		{
			r.n_first_gt_1e4 += 1;
		}
	}
	r
}

fn print_report(name: &str, r: &mut Report)
{
	r.iters.sort_unstable();
	let pct = |q: f64| r.iters.get(((r.iters.len() as f64 - 1.0) * q).round() as usize).copied().unwrap_or(0);
	let mean = r.iters.iter().sum::<usize>() as f64 / r.iters.len().max(1) as f64;
	r.dpwm.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let dq = |q: f64| r.dpwm.get(((r.dpwm.len() as f64 - 1.0) * q).round() as usize).copied().unwrap_or(0.0);
	let dmean = r.dpwm.iter().sum::<f64>() / r.dpwm.len().max(1) as f64;
	let over = |t: f64| r.dpwm.iter().filter(|&&d| d > t).count();
	eprintln!(
		"[mpc-qp {name}] label |Δpwm| exit-vs-exact: mean={dmean:.3e} p50={:.3e} p90={:.3e} p99={:.3e} max={:.3e} \
		 #>1e-3={} #>1/64={} of {}",
		dq(0.5), dq(0.9), dq(0.99), dq(1.0), over(1e-3), over(1.0 / 64.0), r.dpwm.len()
	);
	eprintln!(
		"[mpc-qp {name}] steps={} constrained={} ({:.2}%) cap_hits={} ({:.2}% of constrained) \
		 iters mean={mean:.1} p50={} p90={} p99={} max={} | exit KKT resid max={:.3e} (cap-hit max {:.3e}) \
		 | vs exact: obj gap max={:.3e} |dU0| max={:.3e} |dU| max={:.3e} #|dU0|>1e-6={} #|dU0|>1e-4={} \
		 | ref KKT max={:.3e} ref_failed={}",
		r.steps,
		r.constrained,
		100.0 * r.constrained as f64 / r.steps.max(1) as f64,
		r.cap_hits,
		100.0 * r.cap_hits as f64 / r.constrained.max(1) as f64,
		pct(0.5),
		pct(0.9),
		pct(0.99),
		r.iters.last().copied().unwrap_or(0),
		r.max_resid_exit,
		r.max_resid_exit_cap,
		r.max_obj_gap,
		r.max_du_first,
		r.max_du_all,
		r.n_first_gt_1e6,
		r.n_first_gt_1e4,
		r.max_resid_ref,
		r.ref_failed,
	);
}

/// STEP A MEASUREMENT (ignored — prints). Run with
/// `cargo test -p ram_controller --lib --no-default-features mpc_qp_convergence_census -- --ignored --nocapture`.
#[test]
#[ignore]
fn mpc_qp_convergence_census()
{
	let eps: usize = std::env::var("MPC_CENSUS_EPS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
	{
		let af = AirframeRs::from_cfg(&regime(0.5, 10, false));
		let mut t = af.teacher(4);
		let m = mpc_of(&mut t);
		let (lmax, lmin) = extreme_eigs(&m.h);
		eprintln!("[mpc-qp H] n={} lambda_max={lmax:.4e} lambda_min={lmin:.4e} cond={:.3e}", m.h.r, lmax / lmin);
	}
	for (name, xy, expert) in [
		("stage2-xy0.5-student", 0.5f32, false),
		("stage2-xy0.5-expert", 0.5, true),
		("stage1-xy0-student", 0.0, false),
		("stage1-xy0-expert", 0.0, true),
	]
	{
		let cfg = regime(xy, 2000, expert);
		let probes = collect(&cfg, eps, 0xC0FFEE ^ (xy.to_bits() as u64) ^ expert as u64);
		let af = AirframeRs::from_cfg(&cfg);
		let mut t = af.teacher(4);
		let mut r = analyse(mpc_of(&mut t), &probes);
		print_report(name, &mut r);
	}
}

/// FNV-1a over the f32 bits of every recorded label (pid_pwms) — the teacher
/// output the student is trained toward.
fn label_digest(cfg: &RewardGatedConfigPacked, episodes: usize, seed: u64) -> (u64, usize)
{
	let af = AirframeRs::from_cfg(cfg);
	let mut teacher = af.teacher(4);
	let mut sim: AttitudeSim = af.sim();
	let mut c = student(seed ^ 0x5EED);
	let mut rng = SmallRng::seed_from_u64(seed);
	let (mut h, mut n) = (0xcbf29ce484222325u64, 0usize);
	for _ in 0..episodes
	{
		let traj = rollout_and_label_rs(&mut c, &mut teacher, &mut sim, cfg, TILT_RAD, &mut rng, [0.0; 3]);
		for l in &traj.pid_pwms
		{
			for v in l
			{
				for b in v.to_bits().to_le_bytes()
				{
					h = (h ^ b as u64).wrapping_mul(0x100000001b3);
				}
			}
			n += 1;
		}
		if !cfg.expert_drives
		{
			let _ = train_on_trajectory_rs(&mut c, &traj, cfg);
		}
	}
	(h, n)
}

/// Prints the label digests (ignored). Used to show the unconstrained fast path is
/// bit-identical across the FISTA → exact-solver change and to size the
/// constrained-path label change.
#[test]
#[ignore]
fn mpc_label_digest_print()
{
	for (name, xy, expert) in [
		("stage2-student", 0.5f32, false),
		("stage2-expert", 0.5, true),
		("stage1-student", 0.0, false),
		("stage1-expert", 0.0, true),
	]
	{
		let (h, n) = label_digest(&regime(xy, 2000, expert), 4, 0xD16E57);
		eprintln!("[mpc-digest {name}] labels={n} digest={h:#018x}");
	}
}

/// Pin: on expert-driven rollouts the box never binds, so the labels come from
/// the unconstrained fast path and must be byte-identical to the pre-01/11
/// FISTA build (digests recorded from commit bec92960).
#[test]
fn mpc_unconstrained_labels_bit_identical_to_fista_build()
{
	assert_eq!(label_digest(&regime(0.5, 2000, true), 4, 0xD16E57), (STAGE2_EXPERT_DIGEST, STAGE2_EXPERT_LABELS));
	assert_eq!(label_digest(&regime(0.0, 2000, true), 4, 0xD16E57), (STAGE1_EXPERT_DIGEST, STAGE1_EXPERT_LABELS));
}

/// Pin: every constrained solve on student-driven rollouts (both regimes) is a
/// KKT point and equals an independent exact active-set solve; PDAS settles
/// without the fallback and in a handful of linear solves.
#[test]
fn mpc_constrained_solves_are_exact_on_rollouts()
{
	for xy in [0.5f32, 0.0]
	{
		let cfg = regime(xy, 2000, false);
		let probes = collect(&cfg, 3, 0xE8AC7 ^ xy.to_bits() as u64);
		let af = AirframeRs::from_cfg(&cfg);
		let mut t = af.teacher(4);
		let m = mpc_of(&mut t);
		let r = analyse(m, &probes);
		assert!(r.constrained > 1000, "xy={xy}: too few constrained solves to pin ({})", r.constrained);
		assert_eq!(r.cap_hits, 0, "xy={xy}: {} solves needed the primal fallback", r.cap_hits);
		assert_eq!(r.ref_failed, 0);
		assert!(r.max_resid_exit < 1e-9, "xy={xy}: exit KKT residual {:.3e}", r.max_resid_exit);
		assert!(r.max_du_all < 1e-9, "xy={xy}: |U − U*| {:.3e}", r.max_du_all);
		assert!(r.max_obj_gap < 1e-9, "xy={xy}: objective gap {:.3e}", r.max_obj_gap);
		assert!(*r.iters.iter().max().unwrap() <= 30, "xy={xy}: PDAS took {:?} solves", r.iters.iter().max());
	}
}

/// Pin: the solver against brute force on small random box-QPs (n = 6, all
/// 3^6 = 729 active-set patterns enumerated, the best KKT-feasible one wins) —
/// a reference that shares no code with the active-set searches — and with
/// cold, hostile warm starts that force the primal fallback path too.
#[test]
fn box_qp_matches_enumeration_on_random_qps()
{
	use crate::box_qp::BoxQpSolver;
	let mut rng = SmallRng::seed_from_u64(0xB0C5);
	let n = 6usize;
	let a = 0.4;
	let mut fallbacks = 0;
	for trial in 0..400
	{
		// SPD H = MᵀM + δI with a wide spectrum.
		let mut m = Mat::zeros(n, n);
		for i in 0..n
		{
			for j in 0..n
			{
				m.set(i, j, rng.gen_range(-1.0..1.0) * if j == 0 { 30.0 } else { 1.0 });
			}
		}
		let h = m.transpose().matmul(&m).add(&Mat::identity(n).scale(0.05));
		let g: Vec<f64> = (0..n).map(|_| rng.gen_range(-20.0..20.0)).collect();
		let warm: Vec<f64> = (0..n).map(|_| [-1.0, 0.0, 1.0][rng.gen_range(0..3)]).collect();
		let mut qp = BoxQpSolver::new(n, h.d.clone(), a);
		let u = qp.solve(&g, &warm).to_vec();
		fallbacks += qp.last_fallback as usize;
		let best = enumerate_box_qp(&h, &g, a);
		let du = (0..n).map(|i| (u[i] - best[i]).abs()).fold(0.0, f64::max);
		assert!(du < 1e-9, "trial {trial}: |u − u_enum| = {du:.3e}");
		assert!(kkt_residual(&h, &g, a, &u) < 1e-9);
	}
	eprintln!("[box-qp enumeration] 400 trials, {fallbacks} via primal fallback");
	assert!(fallbacks > 0, "the primal active-set fallback must be exercised by this seed");
}

/// Brute force: every assignment of {−a, free, +a}, keep the feasible
/// stationary point with the lowest objective.
fn enumerate_box_qp(h: &Mat, g: &[f64], a: f64) -> Vec<f64>
{
	let n = g.len();
	let (mut best, mut best_f) = (vec![0.0; n], f64::INFINITY);
	for code in 0..3usize.pow(n as u32)
	{
		let mut st = vec![0i8; n];
		let mut c = code;
		for s in st.iter_mut()
		{
			*s = (c % 3) as i8 - 1;
			c /= 3;
		}
		let free: Vec<usize> = (0..n).filter(|&i| st[i] == 0).collect();
		let mut u: Vec<f64> = st.iter().map(|&s| s as f64 * a).collect();
		if !free.is_empty()
		{
			let mut hff = Mat::zeros(free.len(), free.len());
			let mut rhs = vec![0.0; free.len()];
			for (r, &i) in free.iter().enumerate()
			{
				rhs[r] = -g[i] - (0..n).filter(|&j| st[j] != 0).map(|j| h.get(i, j) * u[j]).sum::<f64>();
				for (cc, &j) in free.iter().enumerate()
				{
					hff.set(r, cc, h.get(i, j));
				}
			}
			let sol = inverse(&hff).unwrap().mul_vec(&rhs);
			for (r, &i) in free.iter().enumerate()
			{
				u[i] = sol[r];
			}
		}
		if u.iter().any(|v| v.abs() > a + 1e-12)
		{
			continue;
		}
		let f = objective(h, g, &u);
		if f < best_f
		{
			(best, best_f) = (u, f);
		}
	}
	best
}

// Label digests of the expert-driven (unconstrained-only) regimes, recorded on
// the FISTA build (commit bec92960) by mpc_label_digest_print.
const STAGE2_EXPERT_DIGEST: u64 = 0x275a32c8db2e539a;
const STAGE2_EXPERT_LABELS: usize = 8000;
const STAGE1_EXPERT_DIGEST: u64 = 0x742d27b2b0d20431;
const STAGE1_EXPERT_LABELS: usize = 8000;
// The student-driven digests on that build were 0x40d37f4d641d0bc5 (stage 2,
// 5181 labels) and 0x3260c5b8e295378e (stage 1, 5731 labels); they CHANGE with
// the exact solver by design (constrained-path labels).

/// Teacher solve throughput (ignored — prints): replay the states a student-driven
/// stage-2 rollout visited through a fresh teacher's step_axes_rs (same warm-start
/// chain), best of 5. Run on both builds to compare.
#[test]
#[ignore]
fn mpc_solver_timing()
{
	for (name, xy) in [("stage2", 0.5f32), ("stage1", 0.0)]
	{
		let cfg = regime(xy, 2000, false);
		let xs: Vec<[f64; 6]> = collect(&cfg, 8, 0x71AE ^ xy.to_bits() as u64).iter().map(|p| p.x).collect();
		let af = AirframeRs::from_cfg(&cfg);
		let mut best = f64::INFINITY;
		let mut sink = 0.0;
		for _ in 0..5
		{
			let mut t = af.teacher(4);
			let m = mpc_of(&mut t);
			let t0 = std::time::Instant::now();
			for x in &xs
			{
				sink += m.step_axes_rs(x)[0];
			}
			best = best.min(t0.elapsed().as_secs_f64());
		}
		eprintln!("[mpc-timing {name}] {} solves, best {:.1} us/solve (sink {sink:.3})", xs.len(), 1e6 * best / xs.len() as f64);
	}
}
