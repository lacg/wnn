//! CTRL-39 (ABI 42): the MEMORY recorder's train ≠ deploy gap, fixed at the source.
//!
//! THE BUG (ABI ≤ 41). The recorder flies a REFERENCE driver (the training cascade's
//! teacher, the legacy PID, or the allocator) and steps the student alongside only
//! to read the addresses it visits. Every student input that carries state from the
//! APPLIED actuator read the student's OWN discarded command instead of the one that
//! flew: the motor-level d̂ observer (`obs_dhat`, its model term reads pwm_applied or,
//! under motor lag, the actuator twin built from it), the lagged `obs_pwm` feature
//! (the delivered-actuator twin, ABI 36) and the lag-off `obs_pwm` feature (the
//! student's free-running accumulator, which in deploy IS the command that flies).
//! The recorded universe was therefore addressed on feature values deploy never
//! produces — the 12/08 DOB bug class, in the one host DOB Fix A never reached.
//!
//! THE CONTRACT the tests pin: at every recorder step the student's applied-actuator
//! features equal the closed-form value computed from the FLOWN command stream (the
//! plant's own lag filter, the observer law) — the value a student that flew those
//! commands would compute in deploy. A pre-fix emulation (the student stepped alone,
//! as the ABI-41 recorder did) is measured against the same oracle and must fail it.

use crate::abi38_pins::{ARM, DT, G, INERTIA, KD, KT};
use crate::abi38_tests::l4c_cfg;
use crate::controller::WnnController;
use crate::dagger_train::{rollout_and_label_rs, AirframeRs, RewardGatedConfigPacked};
use crate::record_ops::{record_feature_stream, CascadeDriver, Driver};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const PWM0: usize = 9; // obs_pwm features [9, 13)
const ANCHOR: f32 = 0.5; // attitude-only: the legacy collective anchor

/// A motor-level delta student with exactly obs_pwm (4) + the L1 d̂ observer (3).
fn motor_student(seed: u64) -> WnnController
{
	let (levels, bpf, window, obpn, nf) = (8usize, 3usize, 2usize, 10usize, 16usize);
	let mut rng = SmallRng::seed_from_u64(seed);
	let thresholds: Vec<f32> = (0..nf * bpf).map(|_| rng.gen_range(-0.4f32..0.4)).collect();
	let oconn: Vec<i64> = (0..4 * levels * obpn).map(|_| rng.gen_range(0..nf * bpf) as i64).collect();
	let b = crate::controller::calibrate_control_gains_rs(DT, ARM, KT, KD, INERTIA, G, 0.5, 0.05);
	let c = WnnController::new_core(
		4, levels, bpf, window, 0, 0, obpn, thresholds, vec![], oconn,
		true, 0.1, 0.95, 1.0,
		false, false, false, false, true, true, false, false,
		0.99, 1.0, DT, false, 1,
		ram_core::neuron_memory::BINARY, Some(crate::cell_mode::DECODE_ANTAGONIST), Some(b), 0.05, false, 0.30,
		false, false, false, false, false, false, 1,
	)
	.expect("ctrl39 student");
	assert_eq!(c.num_features(), nf);
	c
}

/// The training cascade's driver on an attitude-only cf21 plant (lag `lag`, the L4C
/// weather — per-episode asymmetry, noise — armed by the recorder itself).
fn attitude_cfg(lag: f32) -> RewardGatedConfigPacked
{
	let mut cfg = l4c_cfg(lag, 4);
	cfg.translation = false;
	cfg.xy_offset = 0.0;
	cfg.att_err_tilt_ref = false;
	cfg
}

/// One recorder episode → (features, flown) per step.
fn record(c: &mut WnnController, lag: f32, steps: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>)
{
	let cfg = attitude_cfg(lag);
	let af = AirframeRs::from_cfg(&cfg);
	let mut sim = af.sim();
	let mut teacher = af.teacher(cfg.teacher);
	let mut d = Driver::Cascade(CascadeDriver::new(&mut teacher, &cfg, c));
	c.set_observer_motor_lag_core(sim.motor_lag());
	record_feature_stream(c, &mut sim, &mut d, [0.9990482, 0.0436194, 0.0, 0.0], [0.3, -0.2, 0.05], steps)
}

/// The deploy value of the applied-actuator features from the FLOWN stream: the
/// plant's lag (student twin: seed at the first command, 4dt/max(T, 4dt)), the
/// accumulator holding the last flown command (lag off), and the L1 observer law —
/// written out here independently of controller.rs, same f32 operation order.
fn oracle(c: &WnnController, feats: &[Vec<f32>], flown: &[Vec<f32>], lag: f32) -> Vec<[f32; 7]>
{
	let (b, l) = c.dhat_params().expect("d̂ on");
	let alpha = if lag > 0.0 { 4.0 * DT / lag.max(4.0 * DT) } else { 0.0 };
	let (mut ua, mut dh) = ([0.0f32; 4], [0.0f32; 3]);
	let mut out = Vec::with_capacity(feats.len());
	for t in 0..feats.len()
	{
		let mut v = [0.0f32; 7];
		if t >= 1
		{
			let f: [f32; 4] = std::array::from_fn(|m| flown[t - 1][m]);
			ua = if lag > 0.0 && t >= 2 { std::array::from_fn(|m| ua[m] + alpha * (f[m] - ua[m])) } else { f };
			let u = [(ua[3] - ua[1]) * 0.5, (ua[2] - ua[0]) * 0.5, ((ua[0] + ua[2]) - (ua[1] + ua[3])) * 0.25];
			for a in 0..3
			{
				let rate_dot = (feats[t][a] - feats[t - 1][a]) / DT;
				let residual = rate_dot - b[a] * u[a] - dh[a];
				dh[a] += l * residual;
			}
			for m in 0..4
			{
				v[m] = ua[m] - ANCHOR;
			}
		}
		v[4..].copy_from_slice(&dh);
		out.push(v);
	}
	out
}

/// The ABI-41 recorder, emulated: the student stepped on the recorded IMU alone —
/// no accumulator proxy, no observe_applied (its own discarded command flies in its
/// head). Returns its feature stream.
fn prefix_emulation(c: &WnnController, feats: &[Vec<f32>], lag: f32) -> Vec<Vec<f32>>
{
	let mut s = c.clone();
	s.set_observer_motor_lag_core(lag);
	s.reset(0.0);
	feats
		.iter()
		.map(|f| {
			let _ = s.step([f[0], f[1], f[2]], [f[3], f[4], f[5]], [f[6], f[7], f[8]]);
			s.last_feature_vector_ref().to_vec()
		})
		.collect()
}

fn max_dev(feats: &[Vec<f32>], want: &[[f32; 7]], cols: std::ops::Range<usize>) -> f32
{
	feats
		.iter()
		.zip(want)
		.map(|(f, w)| cols.clone().map(|i| (f[PWM0 + i] - w[i]).abs()).fold(0.0f32, f32::max))
		.fold(0.0f32, f32::max)
}

/// THE REGRESSION: recorded obs_pwm and d̂ features == the deploy value from the flown
/// stream, step for step, bit for bit — lag 0.20 (delivered twin) and lag 0
/// (accumulator). The ABI-41 recorder (emulated) is measured on the same oracle and
/// must diverge; the size of that divergence is printed (the CTRL-39 report figure).
#[test]
fn recorder_applied_actuator_features_equal_deploy_values()
{
	for lag in [0.20f32, 0.0]
	{
		let mut c = motor_student(0x39 + (lag * 100.0) as u64);
		let fresh = c.clone();
		let (feats, flown) = record(&mut c, lag, 600);
		let want = oracle(&fresh, &feats, &flown, lag);
		for t in 0..feats.len()
		{
			for i in 0..7
			{
				assert_eq!(feats[t][PWM0 + i].to_bits(), want[t][i].to_bits(), "lag {lag} step {t} feature {}: recorded {} deploy {}", PWM0 + i, feats[t][PWM0 + i], want[t][i]);
			}
		}
		let pre = prefix_emulation(&fresh, &feats, lag);
		let (dp, dd) = (max_dev(&pre, &want, 0..4), max_dev(&pre, &want, 4..7));
		let scale = want.iter().map(|w| w[4..].iter().fold(0.0f32, |m, v| m.max(v.abs()))).fold(0.0f32, f32::max);
		eprintln!("[ctrl39] lag {lag}: ABI-41 recorder vs deploy — obs_pwm max |Δ| {dp:.4} pwm, d̂ max |Δ| {dd:.3} (deploy |d̂| max {scale:.3})");
		assert!(dp > 1e-3 && dd > 1e-3, "lag {lag}: the pre-fix emulation must FAIL the contract (Δ pwm {dp}, Δ d̂ {dd})");
	}
}

/// expert_drives: the TEACHER flies, so an obs_pwm student's accumulator is handed the
/// command that flew — the recorded pwm feature is (last applied − anchor), as deploy.
#[test]
fn expert_drives_obs_pwm_reads_the_flown_command()
{
	let mut cfg = attitude_cfg(0.0);
	cfg.expert_drives = true;
	cfg.explore_eps = 0.0;
	cfg.steps_per_episode = 300;
	let af = AirframeRs::from_cfg(&cfg);
	let (mut teacher, mut sim) = (af.teacher(cfg.teacher), af.sim());
	let mut c = motor_student(0x3911);
	c.set_observer_motor_lag_core(sim.motor_lag());
	let mut rng = SmallRng::seed_from_u64(39);
	let tr = rollout_and_label_rs(&mut c, &mut teacher, &mut sim, &cfg, 0.05, &mut rng, [0.0; 3]);
	for t in 0..tr.steps
	{
		let want: [f32; 4] = if t == 0 { [0.0; 4] } else { std::array::from_fn(|m| tr.student_pwms[t - 1][m] - ANCHOR) };
		assert_eq!(tr.pwm_dev[t], want, "step {t}: the pwm feature must be the flown command − anchor");
	}
}

/// The training-cascade recorder's output universe (ctrl17_tests::record_universe_for)
/// for a student, hashed.
fn universe_digest(student: &WnnController) -> u64
{
	let u = crate::ctrl17_tests::record_universe_for(&l4c_cfg(0.20, 4), student);
	let mut h = crate::abi38_pins::Fnv::new();
	for (n, a) in u
	{
		h.word(n as u64);
		h.word(a);
	}
	h.0
}

/// Captured on the ABI-40 snapshot (pre-CTRL-38/39 recorder) by running the same
/// three recordings there: [cascade, flags off], [motor, no applied-actuator input],
/// [motor, obs_pwm under lag 0.20].
const ABI40_RECORDER: [u64; 3] = [0xe329_9c89_2b8e_2d9e, 0x8936_84c8_3180_b9fd, 0xe628_332a_945a_c535];

/// Scope of the fix: the cascade with the CTRL-38 blocks off and a motor student with
/// no applied-actuator input record the SAME universe as before (bit for bit); an
/// obs_pwm student's universe MOVES (that is the bug being fixed).
#[test]
fn recorder_fix_moves_only_applied_actuator_lineages()
{
	let got = [
		universe_digest(&crate::abi38_tests::ctbr_student(0, 10, 0x3901)),
		universe_digest(&crate::abi38_pins::prod_student(0, 1, false, 0x3902)),
		universe_digest(&crate::abi38_pins::prod_student(0, 1, true, 0x3903)),
	];
	eprintln!("[ctrl39] recorder universes {got:#018x?}");
	assert_eq!(got[0], ABI40_RECORDER[0], "the cascade (CTRL-38 off) recorder moved");
	assert_eq!(got[1], ABI40_RECORDER[1], "a motor student without applied-actuator inputs moved");
	assert_ne!(got[2], ABI40_RECORDER[2], "the obs_pwm recorder did not change — the fix is not reaching it");
}
