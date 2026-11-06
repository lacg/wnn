//! ABI 38 (CTRL-27, the cascade controller): MOTOR-LEVEL BYTE-IDENTITY PINS.
//!
//! The cascade (`output=ctbr`) is a new lineage; the motor-level path (`output=pwm`)
//! must not move by one bit. Every digest below was CAPTURED on the ABI-37 build
//! (commit 15d5bbf0, before any ABI-38 line was written) by running this very file
//! there. They cover the production-shaped regime end to end:
//!
//!   * the whole DAgger trainer (dagger_train_inplace_rs: rollouts, mpcof labels,
//!     gate, training, per-round eval, checkpoint restore) on a cf21 stage-2 config
//!     — BINARY delta student with obs_pwm + vertical + horizontal channels, mpcof
//!     teacher at the derived hover, L4C-like weather — at lag 0 and 0.20,
//!     action_repeat 1 and 10, sn = 0 and sn = 4 (the CPU EDRA-BPTT walk);
//!   * the CPU scorer (rollout_one) and the Metal scorer on the trained genomes;
//!   * the classical rivals (score_classical_baseline, estimator-fed, stage 2);
//!   * the CPU state-split trainer (split_train_loop) on recorded rollouts;
//!   * the firmware PID cascade (AttitudePidFirmwareRs::step_rs) on a random
//!     trajectory — the composition the ABI-38 split into attitude + rate stages
//!     must reproduce.
//!
//! A digest moving means the motor-level lineage changed. Do not re-pin without a
//! lineage decision from Luiz.

use crate::controller::{AttitudeSim, WnnController};
use crate::dagger_train::{dagger_train_inplace_rs, rollout_and_label_rs, AirframeRs, RewardGatedConfigPacked};
use crate::stage1::Stage1Cfg;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub(crate) struct Fnv(pub u64);

impl Fnv
{
	pub(crate) fn new() -> Self
	{
		Fnv(0xcbf2_9ce4_8422_2325)
	}
	pub(crate) fn word(&mut self, w: u64)
	{
		for b in w.to_le_bytes()
		{
			self.0 ^= b as u64;
			self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
		}
	}
	pub(crate) fn f64s(&mut self, xs: &[f64])
	{
		for x in xs
		{
			self.word(x.to_bits());
		}
	}
}

pub(crate) const DT: f32 = 0.001;
pub(crate) const ARM: f32 = 0.070710678;
pub(crate) const KT: f32 = 0.2;
pub(crate) const KD: f32 = 0.00569278844371417;
pub(crate) const INERTIA: [f32; 3] = [3.004e-5, 3.019e-5, 5.304e-5];
pub(crate) const G: f32 = 9.81;
pub(crate) const MASS: f32 = 0.0393;

/// The cf21_brushless firmware cascade in SI (pin_translation_cfg's arrays).
pub(crate) const PID_ATT: [f64; 12] =
	[6.0, 3.0, 0.0, 0.3490658503988659, 6.0, 3.0, 0.0, 0.3490658503988659, 6.0, 1.0, 0.35, 6.283185307179586];
pub(crate) const PID_RATE: [f64; 12] = [
	0.03497110216713654, 0.06994220433427308, 0.00043713877708920673, 0.5811946409141117,
	0.03497110216713654, 0.06994220433427308, 0.00043713877708920673, 0.5811946409141117,
	0.02098266130028192, 0.002920087030955901, 0.0, 2.909463863074547,
];
pub(crate) const PID_OUT_LIMIT_N: f64 = 0.09999847409781033;
pub(crate) const PID_HOVER_N: f64 = 0.09638325;
pub(crate) const PID_ATT_HZ: f64 = 500.0;
pub(crate) const PID_LPF_HZ: f64 = 30.0;

/// The production-shaped feature set at a small width: per-axis p+i (yaw off),
/// yaw err + integral, obs_pwm (motor path only), vertical + horizontal channels.
pub(crate) fn prod_student(sn: usize, action_repeat: usize, obs_pwm: bool, seed: u64) -> WnnController
{
	let (levels, bpf, window, obpn, sbpn) = (8usize, 3usize, 2usize, 10usize, 6usize);
	let nf = 9 + 2 + 2 + if obs_pwm { 4 } else { 0 } + 2 + 3 + 4;
	let frame_bits = nf * bpf;
	let mut rng = SmallRng::seed_from_u64(seed);
	let thresholds: Vec<f32> = (0..frame_bits).map(|_| rng.gen_range(-0.4f32..0.4)).collect();
	let state_in = window * frame_bits + sn;
	let out_in = frame_bits + sn;
	let sconn: Vec<i64> = (0..sn * sbpn).map(|_| rng.gen_range(0..state_in) as i64).collect();
	let oconn: Vec<i64> = (0..4 * levels * obpn).map(|_| rng.gen_range(0..out_in) as i64).collect();
	let c = WnnController::new_core(
		4, levels, bpf, window, sn, if sn > 0 { sbpn } else { 0 }, obpn, thresholds, sconn, oconn,
		true, 0.1, 0.95, 1.0,
		false, false, true, true, false, obs_pwm, true, true,
		0.99, 1.0, DT, false, action_repeat,
		ram_core::neuron_memory::BINARY, Some(crate::cell_mode::DECODE_ANTAGONIST), None, 0.05, false, 0.30,
		true, true, true, true, true, false, 1,
	)
	.expect("abi38 pin student must construct");
	assert_eq!(c.num_features(), nf);
	c
}

/// cf21 stage-2 trainer config: mpcof teacher at the derived hover, L4C-like
/// weather (motor asymmetry + gyro noise + bias), tilt reference.
pub(crate) fn prod_cfg(lag: f32, teacher: u8) -> RewardGatedConfigPacked
{
	let mut c = crate::ctrl17_pins::pin_translation_cfg(0.5);
	c.teacher = teacher;
	c.teacher_hover_mode = 1;
	c.motor_lag_s = lag;
	c.dist_enabled = true;
	c.dist_motor_asym = [1.12, 0.9, 1.05, 0.95];
	c.dist_gyro_sigma = 0.0107;
	c.dist_torque_scale_jitter = 0.2;
	c.att_err_tilt_ref = true;
	c.steps_per_episode = 300;
	c
}

pub(crate) fn cells_and_stats_digest(c: &WnnController, st: Option<&crate::dagger_train::TrainStats>) -> u64
{
	let mut h = Fnv::new();
	if let Some(st) = st
	{
		h.f64s(&st.iter_fitness);
		h.f64s(&st.iter_mean_err_deg);
		h.f64s(&st.iter_stable_rate);
		h.f64s(&st.iter_mean_episode_reward);
		for n in st.iter_n_trained.iter().chain(&st.iter_cells_written)
		{
			h.word(*n as u64);
		}
	}
	h.word(crate::abi35_tests::cells_digest(c));
	h.0
}

pub(crate) fn train(sn: usize, ar: usize, lag: f32, teacher: u8, obs_pwm: bool) -> (WnnController, u64)
{
	let mut c = prod_student(sn, ar, obs_pwm, 0xC738 + sn as u64 * 7 + ar as u64);
	let cfg = prod_cfg(lag, teacher);
	let st = dagger_train_inplace_rs(&mut c, &cfg, [0.0; 3], 4243);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "pin trainer wrote no cells (sn={sn} ar={ar} lag={lag})");
	let d = cells_and_stats_digest(&c, Some(&st));
	(c, d)
}

pub(crate) const EPS: usize = 4;

pub(crate) fn ics() -> (Vec<f32>, Vec<f32>)
{
	let mut rng = SmallRng::seed_from_u64(0x38_1C5);
	let (mut q0, mut w0) = (Vec::new(), Vec::new());
	for _ in 0..EPS
	{
		let ang: f32 = rng.gen_range(-0.09..0.09);
		let (hx, hy): (f32, f32) = (rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
		let n = (hx * hx + hy * hy).sqrt().max(1e-6);
		let h = ang / 2.0;
		q0.extend_from_slice(&[h.cos(), h.sin() * hx / n, h.sin() * hy / n, 0.0]);
		w0.extend_from_slice(&[rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), 0.0]);
	}
	(q0, w0)
}

pub(crate) fn stage2() -> Stage1Cfg
{
	let c = Stage1Cfg {
		target_altitude: 0.0,
		lambda_alt: 1.0,
		init_z: (0..EPS).map(|e| -0.1 + 0.05 * e as f32).collect(),
		init_vz: vec![0.0; EPS],
		mass: (0..EPS).map(|e| MASS * (0.95 + 0.03 * e as f32)).collect(),
		collective_frac: (0..EPS).map(|e| 0.02 * e as f32 - 0.03).collect(),
		lambda_pos: 2.0,
		init_x: (0..EPS).map(|e| 0.4 - 0.25 * e as f32).collect(),
		init_y: (0..EPS).map(|e| -0.45 + 0.3 * e as f32).collect(),
		plant: crate::plant_opts::PlantOpts::OFF,
		att_ref: crate::att_err_ref::AttErrRef::tilt(G as f64, 1.0, 1.0, 0.5236).unwrap(),
	};
	c.validate(EPS).expect("abi38 stage-2 fixture must validate");
	c
}

pub(crate) const ASYM: [f32; 4] = [1.12, 0.9, 1.05, 0.95];

pub(crate) fn cpu_rows(c: &WnnController, steps: usize, lag: f32) -> [f64; 17]
{
	let (q0, w0) = ics();
	let s1 = stage2();
	let mut cc = c.clone();
	crate::cpu_score::rollout_one(
		&mut cc, &q0, &w0, EPS, steps, DT, ARM, KT, KD, INERTIA, G, lag, [0.0; 3],
		true, [0.0; 3], 0.0, 0.1, ASYM, 0.0107, 0.0, 0.0, 777,
		0.0, 0, 0, 0.2, c.levels_per_motor(), 4, None, None, None, 1.0, 0.4, None, Some(&s1),
	)
}

pub(crate) fn gpu_rows(c: &WnnController, steps: usize, lag: f32) -> Option<Vec<f64>>
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok()
	{
		return None;
	}
	let ev = crate::metal_controller::ControllerRolloutEvaluator::new().ok()?;
	let (q0, w0) = ics();
	let s1 = stage2();
	let d = crate::controller::Disturbance {
		tau_bias: [0.0; 3],
		gust_sigma: 0.0,
		gust_tau_c: 0.1,
		motor_asym: ASYM,
		gyro_sigma: 0.0107,
		gyro_bias_walk: 0.0,
		accel_sigma: 0.0,
		dropout_prob: 0.0,
		dropout_len_steps: 0,
		obs_delay_steps: 0,
		torque_scale_jitter: 0.2,
		seed: 777,
	};
	let rows = ev
		.score(&[c], &q0, &w0, EPS, steps, (DT, ARM, KT, INERTIA, G), KD, lag, [0.0; 3], Some(d), None, None, None, Some(&s1))
		.expect("gpu score");
	Some(rows[0].clone())
}

fn rows_digest(r: &[f64]) -> u64
{
	let mut h = Fnv::new();
	h.f64s(r);
	h.0
}

/// One classical rival on the stage-2 episode set (estimator-fed, L4C-like).
pub(crate) fn rival(teacher: u8, lag: f32) -> (f64, f64, f64, f64, f64, f64, f64, f64)
{
	let (q0, w0) = ics();
	let s1 = stage2();
	crate::dagger_train::score_classical_baseline(
		teacher, q0, w0, 600, 5.0,
		true, [0.0; 3], 0.0, 0.1, ASYM, 0.0107, 0.0, 0.0, 777, 0.0, 0, 0, 0.2,
		ARM, KT, KD, INERTIA, G, DT,
		PID_ATT, PID_RATE, PID_OUT_LIMIT_N, PID_HOVER_N, PID_ATT_HZ, PID_LPF_HZ,
		true, 2.0, 0.1,
		true, 0.0, s1.init_z.clone(), s1.init_vz.clone(), s1.mass.clone(), s1.collective_frac.clone(),
		2.0, 1.0, 0.25,
		s1.init_x.clone(), s1.init_y.clone(), 1.0, 1.0, 0.5236,
		lag, false, 0.0, true,
	)
	.expect("rival must score")
}

fn rival_digest(r: (f64, f64, f64, f64, f64, f64, f64, f64)) -> u64
{
	rows_digest(&[r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7])
}

/// The CPU state-split trainer on three recorded training rollouts (sn = 4).
fn split_digest(lag: f32) -> u64
{
	let cfg = prod_cfg(lag, 4);
	let af = AirframeRs::from_cfg(&cfg);
	let mut teacher = af.teacher(cfg.teacher);
	let mut sim: AttitudeSim = af.sim();
	let mut rec = prod_student(4, 1, true, 0x5111);
	let mut rng = SmallRng::seed_from_u64(91);
	let (mut g, mut a, mut t, mut p, mut iy, mut ro) = (vec![], vec![], vec![], vec![], vec![], vec![]);
	for _ in 0..3
	{
		let tr = rollout_and_label_rs(&mut rec, &mut teacher, &mut sim, &cfg, 0.1, &mut rng, [0.0; 3]);
		g.push(tr.gyros.clone());
		a.push(tr.accels.clone());
		t.push(tr.targets.clone());
		p.push(tr.pid_pwms.clone());
		iy.push(tr.init_yaw);
		ro.push(crate::controller::ReplayObs::slice(&tr.vert_obs, &tr.horiz_obs, &tr.pwm_dev, &tr.att_up, 0, tr.steps).expect("obs"));
	}
	let mut c = prod_student(4, 1, true, 0x5112);
	let (rounds, conf, planted, per, sat, wish) = c.split_train_loop(g, a, t, p, 0.1, 0.999, 0.9, 3, 1, 32, true, iy, ro);
	let mut h = Fnv::new();
	for v in [rounds, conf, planted, sat]
	{
		h.word(v as u64);
	}
	for v in per.iter().chain(&wish)
	{
		h.word(*v as u64);
	}
	h.word(crate::abi35_tests::cells_digest(&c));
	h.0
}

/// The firmware cascade on a random trajectory (no plant: random q / gyro / target).
pub(crate) fn pidfw_digest() -> u64
{
	let mut p = crate::pid_firmware::AttitudePidFirmwareRs::from_si_arrays(
		PID_ATT, PID_RATE, PID_OUT_LIMIT_N, PID_HOVER_N, KT as f64, 1000, PID_ATT_HZ, PID_LPF_HZ,
	)
	.expect("cascade gains");
	let mut rng = SmallRng::seed_from_u64(0xF1A7);
	let mut h = Fnv::new();
	for ep in 0..3
	{
		p.reset();
		for _ in 0..700
		{
			let ang: f64 = rng.gen_range(-0.6..0.6);
			let (hx, hy, hz): (f64, f64, f64) = (rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-0.3..0.3));
			let n = (hx * hx + hy * hy + hz * hz).sqrt().max(1e-9);
			let q = [(ang / 2.0).cos(), (ang / 2.0).sin() * hx / n, (ang / 2.0).sin() * hy / n, (ang / 2.0).sin() * hz / n];
			let gyro = [rng.gen_range(-3.0..3.0), rng.gen_range(-3.0..3.0), rng.gen_range(-1.0..1.0)];
			let tgt = [rng.gen_range(-0.2..0.2), rng.gen_range(-0.2..0.2), if ep == 2 { 3.0 } else { 0.0 }];
			h.f64s(&p.step_rs(q, gyro, tgt));
		}
	}
	h.0
}

/// [trainer ×4, cpu rows ×4, gpu rows ×4] — see the module note. Rivals, split and
/// the firmware cascade are pinned separately (they need no trained genome).
const ABI37_TRAIN_SCORE: [u64; 12] = [
	0xcb29_5ab9_f901_9129, 0xdd38_405f_3c3b_2b9e, 0xa413_ea1d_e54d_cb16, 0x0670_a569_ca31_ef07,
	0x322a_cc0b_d7a8_7301, 0xfae3_2599_6075_c40a, 0x8aa7_8178_50c0_5c9a, 0xeb85_75e5_a948_7976,
	0xdf79_94a9_b2ea_2a8d, 0xc4d3_bce3_8662_d09a, 0x3ee9_fef6_3492_9930, 0x6950_51a2_48b3_681c,
];
const ABI37_RIVALS: [u64; 4] = [0x1efc_ac56_65f8_81c3, 0xc51d_b21b_b380_bef5, 0xe2bf_f48c_969e_3f12, 0x8a6f_c5c4_348d_ce3a];
const ABI37_SPLIT: [u64; 2] = [0x41ea_9fe1_b773_8472, 0x2f82_c765_db5c_0439];
const ABI37_PIDFW: u64 = 0xcbfa_fa3a_5d95_1585;

/// (sn, action_repeat, lag) of the four pinned trainer cells.
const TRAIN_CASES: [(usize, usize, f32); 4] = [(0, 1, 0.0), (0, 10, 0.20), (4, 1, 0.20), (0, 1, 0.20)];

#[test]
fn abi38_motor_path_bit_identical_to_abi37()
{
	let mut got = Vec::new();
	let mut ctrls = Vec::new();
	for &(sn, ar, lag) in &TRAIN_CASES
	{
		let (c, d) = train(sn, ar, lag, 4, true);
		got.push(d);
		ctrls.push((c, lag));
	}
	for (c, lag) in &ctrls
	{
		got.push(rows_digest(&cpu_rows(c, 500, *lag)));
	}
	for (c, lag) in &ctrls
	{
		match gpu_rows(c, 500, *lag)
		{
			Some(r) => got.push(rows_digest(&r)),
			None => got.push(ABI37_TRAIN_SCORE[got.len()]),
		}
	}
	eprintln!("[abi38-pin] train/score {got:#018x?}");
	let rivals: Vec<u64> = [(0u8, 0.0f32), (0, 0.20), (4, 0.0), (4, 0.20)].iter().map(|&(t, l)| rival_digest(rival(t, l))).collect();
	eprintln!("[abi38-pin] rivals {rivals:#018x?}");
	let split = [split_digest(0.0), split_digest(0.20)];
	eprintln!("[abi38-pin] split {split:#018x?}");
	let pidfw = pidfw_digest();
	eprintln!("[abi38-pin] pidfw {pidfw:#018x}");
	assert_eq!(got[..], ABI37_TRAIN_SCORE[..], "motor-level trainer/scorer moved vs ABI 37");
	assert_eq!(rivals[..], ABI37_RIVALS[..], "classical rivals moved vs ABI 37");
	assert_eq!(split, ABI37_SPLIT, "split trainer moved vs ABI 37");
	assert_eq!(pidfw, ABI37_PIDFW, "firmware cascade moved vs ABI 37");
}
