//! DIAGNOSTIC (06/11/2026, CTRL-5 lag collapse): does the student's OUTPUT LAW
//! survive the airframe motor lag when it is handed the teacher's label EXACTLY?
//!
//! The live sn=0 student (dagger_label_delta off) learns the teacher's re-based
//! absolute label p(obs) and its decode integrates it through the leaky delta
//! accumulator: e ← leak·e + 2·dmax·Q_L(p − hover), e = command − anchor
//! (controller.rs decode_outputs / leaked_baseline, delta_to_decoded with
//! gamma 1; BINARY antagonist grid Q_L = trunc to 1/L). DC gain 2·dmax/(1−leak),
//! pole at leak. This "perfect student" removes every learning/observability
//! question: if it diverges under lag, the law itself is the cause.

use crate::abi34_tests::Rollout;
use crate::controller::AttitudeSim;
use crate::dagger_train::AirframeRs;

fn airframe(motor_lag_s: f32) -> AirframeRs
{
	AirframeRs {
		dt: 0.001,
		arm_length: 0.0707,
		k_thrust: 0.2,
		k_drag: 0.0057,
		inertia: [3.0e-5, 3.0e-5, 5.3e-5],
		gravity: 9.81,
		pid_fw: None,
		teacher_hover: Some(0.6942),
		motor_lag_s,
		plant: crate::plant_opts::PlantOpts::OFF,
	}
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Law
{
	/// the teacher flies its own command (the rivals' bar)
	Teacher,
	/// the delta accumulator fed the exact label: (levels, dmax, leak); levels 0 = unquantized
	Delta(usize, f64, f64),
}

fn q_grid(x: f64, levels: usize) -> f64
{
	if levels == 0 { x } else { (x * levels as f64).trunc() / levels as f64 }
}

/// Teacher-in-the-loop rollout, the student law between teacher and plant. The
/// teacher observes what FLEW (the DAgger protocol). Returns stats + stable %.
pub(crate) fn surrogate(id: u8, lag: f32, law: Law, episodes: usize, steps: usize, tilt0: f32) -> (Rollout, f64)
{
	let af = airframe(lag);
	let mut teacher = af.teacher(id);
	let hover = teacher.hover();
	let mut sim: AttitudeSim = af.sim();
	let tail_start = steps * 4 / 5;
	let (mut sum_err, mut sum_tail, mut n, mut n_tail, mut diverged, mut stable) = (0.0f64, 0.0f64, 0usize, 0usize, 0usize, 0usize);
	for ep in 0..episodes
	{
		let a = tilt0.to_radians() * (0.4 + 0.6 * (ep as f32 + 0.5) / episodes as f32);
		let ang = ep as f32 * 2.399;
		let (s, c) = (a * 0.5).sin_cos();
		let q = [c, s * ang.cos(), s * ang.sin(), 0.0];
		teacher.reset();
		sim.reset(Some(q), Some([0.2, -0.15, 0.05]));
		let mut last = [hover; 4];
		let mut e = [0.0f64; 4];
		let (mut ep_tail, mut ep_n_tail, mut ep_div) = (0.0f64, 0usize, false);
		for t in 0..steps
		{
			if sim.is_unstable()
			{
				ep_div = true;
				break;
			}
			let (gyro, _) = sim.read_imu();
			teacher.observe(gyro, if t == 0 { [0.5; 4] } else { last });
			let p = teacher.step_rs(sim.quaternion(), gyro, [0.0, 0.0, 0.0]);
			let cmd = match law
			{
				Law::Teacher => p,
				Law::Delta(levels, dmax, leak) =>
				{
					let mut out = [0.0f64; 4];
					for m in 0..4
					{
						let label = q_grid((p[m] - hover).clamp(-0.5, 0.5), levels);
						let delta = (2.0 * dmax * label).clamp(-dmax, dmax);
						e[m] = leak * e[m] + delta;
						out[m] = (hover + e[m]).clamp(0.0, 1.0);
						e[m] = out[m] - hover;
					}
					out
				}
			};
			sim.step(cmd.map(|v| v as f32));
			last = cmd;
			let err = sim.attitude_error(None) as f64;
			sum_err += err;
			n += 1;
			if t >= tail_start
			{
				sum_tail += err;
				n_tail += 1;
				ep_tail += err;
				ep_n_tail += 1;
			}
		}
		if ep_div
		{
			diverged += 1;
		}
		else if ep_n_tail > 0 && (ep_tail / ep_n_tail as f64).to_degrees() < 5.0
		{
			stable += 1;
		}
	}
	(
		Rollout {
			digest: 0,
			mean_err: sum_err / n.max(1) as f64,
			steady_err: sum_tail / n_tail.max(1) as f64,
			mean_abs_dhat: 0.0,
			final_dhat: [0.0; 3],
			diverged,
			cost: 0.0,
		},
		100.0 * stable as f64 / episodes as f64,
	)
}

/// Prints the dose table; asserts nothing (run with --ignored --nocapture).
#[test]
#[ignore]
fn diag_student_law_under_lag()
{
	let laws = [
		("teacher", Law::Teacher),
		("delta L64 dmax.1 leak.95 (live)", Law::Delta(64, 0.1, 0.95)),
		("delta unquantized dmax.1 leak.95", Law::Delta(0, 0.1, 0.95)),
		("delta unq dmax.05 leak.95 (G=2)", Law::Delta(0, 0.05, 0.95)),
		("delta unq dmax.025 leak.95 (G=1)", Law::Delta(0, 0.025, 0.95)),
		("delta unq dmax.5 leak.0 (G=1,no pole)", Law::Delta(0, 0.5, 0.0)),
	];
	for id in [0u8, 1, 4]
	{
		for lag in [0.0f32, 0.05, 0.10, 0.20]
		{
			for (name, law) in laws
			{
				let (r, st) = surrogate(id, lag, law, 20, 2000, 20.0);
				eprintln!(
					"[lagdiag] id={id} T={lag:.2} {name:40} stable {st:5.1}% err {:7.2}° steady {:7.2}° div {}",
					r.mean_err.to_degrees(), r.steady_err.to_degrees(), r.diverged
				);
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Observability probe: the best LINEAR MEMORYLESS student. Features (control
// space, BODY frame): attitude error e (3, euler vs level), body rates (3), the
// student's own last command deviation c = unmix(cmd − hover) (3 — what
// --obs-pwm shows), bias. Optionally + the delivered-actuator estimate ua (3,
// ActuatorLag over the commands that flew — what the teacher plans on). Fit by
// ridge LS to the teacher's control-space label unmix(p − hover) along the
// student's own trajectories (DAgger: iterate fly → relabel → refit), then fly
// the fitted policy through the live output law.
// ---------------------------------------------------------------------------

use crate::actuator_lag::{unmix_plus, ActuatorLag};

fn mix_plus(h: f64, u: [f64; 3]) -> [f64; 4]
{
	[h - u[1] + u[2], h - u[0] - u[2], h + u[1] + u[2], h + u[0] - u[2]]
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64>
{
	let n = b.len();
	for c in 0..n
	{
		let piv = (c..n).max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap()).unwrap();
		a.swap(c, piv);
		b.swap(c, piv);
		for r in 0..n
		{
			if r != c
			{
				let f = a[r][c] / a[c][c];
				for k in c..n
				{
					a[r][k] -= f * a[c][k];
				}
				b[r] -= f * b[c];
			}
		}
	}
	(0..n).map(|i| b[i] / a[i][i]).collect()
}

struct Fit
{
	w: Vec<[f64; 3]>, // per feature, per output axis
}

/// with_ua: 0 = (e, c) [today's obs], 1 = (e, c, ua), 2 = (e, ua) [c replaced]
fn features(e: [f64; 6], c: [f64; 3], ua: [f64; 3], with_ua: u8) -> Vec<f64>
{
	let mut f = e.to_vec();
	if with_ua != 2
	{
		f.extend_from_slice(&c);
	}
	if with_ua >= 1
	{
		f.extend_from_slice(&ua);
	}
	f.push(1.0);
	f
}

/// One DAgger pass: fly `policy` (None = the exact teacher label) through the
/// live law, record (features, teacher label). Returns (rows, stable %, err°, steady°).
fn fly(id: u8, lag: f32, policy: Option<&Fit>, with_ua: u8, episodes: usize, steps: usize, seed_off: usize)
	-> (Vec<(Vec<f64>, [f64; 3])>, f64, f64, f64)
{
	match policy
	{
		None => fly_inner(id, lag, None, with_ua, episodes, steps, seed_off),
		Some(fit) => fly_inner(id, lag, Some(&|f: &[f64]| {
			let mut u = [0.0f64; 3];
			for (j, fj) in f.iter().enumerate()
			{
				for ax in 0..3
				{
					u[ax] += fit.w[j][ax] * fj;
				}
			}
			u
		}), with_ua, episodes, steps, seed_off),
	}
}

fn fly_with(id: u8, lag: f32, policy: &dyn Fn(&[f64]) -> [f64; 3], with_ua: u8, episodes: usize, steps: usize, seed_off: usize)
	-> (Vec<(Vec<f64>, [f64; 3])>, f64, f64, f64)
{
	fly_inner(id, lag, Some(policy), with_ua, episodes, steps, seed_off)
}

fn fly_inner(id: u8, lag: f32, policy: Option<&dyn Fn(&[f64]) -> [f64; 3]>, with_ua: u8, episodes: usize, steps: usize, seed_off: usize)
	-> (Vec<(Vec<f64>, [f64; 3])>, f64, f64, f64)
{
	let af = airframe(lag);
	let mut teacher = af.teacher(id);
	let hover = teacher.hover();
	let mut sim: AttitudeSim = af.sim();
	let (levels, dmax, leak) = (64usize, 0.1f64, 0.95f64);
	let mut rows = Vec::new();
	let tail_start = steps * 4 / 5;
	let (mut se, mut st, mut n, mut nt, mut stable) = (0.0, 0.0, 0usize, 0usize, 0usize);
	for ep in 0..episodes
	{
		let k = ep + seed_off;
		let a = 20f32.to_radians() * (0.4 + 0.6 * ((k * 7919) % 97) as f32 / 97.0);
		let ang = k as f32 * 2.399;
		let (s, cq) = (a * 0.5).sin_cos();
		teacher.reset();
		sim.reset(Some([cq, s * ang.cos(), s * ang.sin(), 0.0]), Some([0.2, -0.15, 0.05]));
		let mut act = ActuatorLag::new(lag, af.dt);
		act.observe([0.5; 4]);
		let mut last = [hover; 4];
		let mut e_acc = [0.0f64; 4];
		let (mut ept, mut epn, mut div) = (0.0, 0usize, false);
		for t in 0..steps
		{
			if sim.is_unstable()
			{
				div = true;
				break;
			}
			let (gyro, _) = sim.read_imu();
			teacher.observe(gyro, if t == 0 { [0.5; 4] } else { last });
			let p = teacher.step_rs(sim.quaternion(), gyro, [0.0, 0.0, 0.0]);
			let (r, pi, y) = crate::controller::quat_to_euler_f64(sim.quaternion());
			let ex = [r, pi, crate::controller::wrap_angle_f64(y), gyro[0] as f64, gyro[1] as f64, gyro[2] as f64];
			let c = unmix_plus(last.map(|v| v - hover));
			let ua = if t == 0 { c } else { act.state() };
			let f = features(ex, c, ua, with_ua);
			let lab = unmix_plus(p.map(|v| v - hover));
			rows.push((f.clone(), lab));
			let target = match policy
			{
				None => p,
				Some(pol) => mix_plus(hover, pol(&f)),
			};
			let mut cmd = [0.0f64; 4];
			for m in 0..4
			{
				let label = q_grid((target[m] - hover).clamp(-0.5, 0.5), levels);
				e_acc[m] = leak * e_acc[m] + (2.0 * dmax * label).clamp(-dmax, dmax);
				cmd[m] = (hover + e_acc[m]).clamp(0.0, 1.0);
				e_acc[m] = cmd[m] - hover;
			}
			sim.step(cmd.map(|v| v as f32));
			act.observe(cmd);
			last = cmd;
			let err = sim.attitude_error(None) as f64;
			se += err;
			n += 1;
			if t >= tail_start
			{
				st += err;
				nt += 1;
				ept += err;
				epn += 1;
			}
		}
		if !div && epn > 0 && (ept / epn as f64).to_degrees() < 5.0
		{
			stable += 1;
		}
	}
	(rows, 100.0 * stable as f64 / episodes as f64, (se / n.max(1) as f64).to_degrees(), (st / nt.max(1) as f64).to_degrees())
}

/// Ridge LS fit; returns the fit and the label R² per axis.
fn fit(rows: &[(Vec<f64>, [f64; 3])]) -> (Fit, [f64; 3])
{
	let nf = rows[0].0.len();
	let mut w = vec![[0.0f64; 3]; nf];
	let mut r2 = [0.0f64; 3];
	for ax in 0..3
	{
		let mut a = vec![vec![0.0f64; nf]; nf];
		let mut b = vec![0.0f64; nf];
		for (f, y) in rows
		{
			for i in 0..nf
			{
				b[i] += f[i] * y[ax];
				for j in 0..nf
				{
					a[i][j] += f[i] * f[j];
				}
			}
		}
		for i in 0..nf
		{
			a[i][i] += 1e-9 * rows.len() as f64;
		}
		let x = solve(a, b);
		let mean = rows.iter().map(|r| r.1[ax]).sum::<f64>() / rows.len() as f64;
		let (mut ss_res, mut ss_tot) = (0.0, 0.0);
		for (f, y) in rows
		{
			let yh: f64 = f.iter().zip(&x).map(|(a, b)| a * b).sum();
			ss_res += (y[ax] - yh).powi(2);
			ss_tot += (y[ax] - mean).powi(2);
		}
		r2[ax] = 1.0 - ss_res / ss_tot.max(1e-30);
		for i in 0..nf
		{
			w[i][ax] = x[i];
		}
	}
	(Fit { w }, r2)
}

#[test]
#[ignore]
fn diag_memoryless_student_observability()
{
	for id in [4u8, 1]
	{
		for lag in [0.0f32, 0.05, 0.10, 0.15, 0.20]
		{
			for with_ua in [0u8, 1, 2]
			{
				let (mut rows, _, _, _) = fly(id, lag, None, with_ua, 10, 2000, 0);
				let mut line = String::new();
				for it in 0..4
				{
					let (f, r2) = fit(&rows);
					let (new_rows, st, err, steady) = fly(id, lag, Some(&f), with_ua, 10, 2000, 100 + it * 10);
					line += &format!(" | it{it} R2 {:.3}/{:.3}/{:.3} fly {st:5.1}% {err:6.2}° {steady:6.2}°", r2[0], r2[1], r2[2]);
					rows.extend(new_rows);
				}
				eprintln!("[obsdiag] id={id} T={lag:.2} ua={}{line}", ["e,c", "e,c,ua", "e,ua"][with_ua as usize]);
			}
		}
	}
}

/// ABI 36 pin (the 06/11/2026 fix, in probe form): at the cf21 airframe lag
/// T = 0.20 the best linear memoryless student flown through the live output
/// law STAYS STABLE when it observes the delivered actuator u_a in place of its
/// command (the ABI-36 --obs-pwm convention), and COLLAPSES on the ABI-35
/// command convention — the contrast keeps the test from being vacuous.
#[test]
fn abi36_linear_student_with_delivered_actuator_holds_lag_020()
{
	let run = |with_ua: u8| {
		let (mut rows, _, _, _) = fly(4, 0.20, None, with_ua, 10, 2000, 0);
		let mut last = (0.0, 0.0, 0.0);
		for it in 0..4
		{
			let (f, _) = fit(&rows);
			let (new_rows, st, err, steady) = fly(4, 0.20, Some(&f), with_ua, 10, 2000, 100 + it * 10);
			last = (st, err, steady);
			rows.extend(new_rows);
		}
		last
	};
	let delivered = run(2);
	let command = run(0);
	eprintln!("[abi36] T=0.20 linear student: delivered {delivered:?} command {command:?}");
	assert!(delivered.0 >= 99.0 && delivered.1 < 4.0, "delivered-actuator student lost the lagged plant: {delivered:?}");
	assert!(command.0 < 50.0, "command-convention student no longer collapses — the probe lost its teeth: {command:?}");
}

// ---------------------------------------------------------------------------
// Probe 3 (06/11/2026, after ABI 36's grid still collapsed): the same linear
// student, but on THERMOMETER-ENCODED features (8 quantile thresholds per
// feature fitted on the teacher-label pass, the recipe's --bits-per-feature 8).
// Isolates input RESOLUTION from observability.
// ---------------------------------------------------------------------------

fn thermo_fit_thresholds(rows: &[(Vec<f64>, [f64; 3])], bpf: usize) -> Vec<Vec<f64>>
{
	let nf = rows[0].0.len() - 1; // last is the bias
	(0..nf)
		.map(|j| {
			let mut v: Vec<f64> = rows.iter().map(|r| r.0[j]).collect();
			v.sort_by(|a, b| a.partial_cmp(b).unwrap());
			(0..bpf).map(|k| v[((k as f64 + 0.5) / bpf as f64 * (v.len() - 1) as f64) as usize]).collect()
		})
		.collect()
}

fn thermo(f: &[f64], thr: &[Vec<f64>]) -> Vec<f64>
{
	let mut out = Vec::new();
	for (j, t) in thr.iter().enumerate()
	{
		for &x in t
		{
			out.push(if f[j] >= x { 1.0 } else { 0.0 });
		}
	}
	out.push(1.0);
	out
}

#[test]
#[ignore]
fn diag_thermometer_student_resolution()
{
	for bpf in [8usize, 32]
	{
		for lag in [0.0f32, 0.20]
		{
			for with_ua in [0u8, 2]
			{
				let (rows0, _, _, _) = fly(4, lag, None, with_ua, 10, 2000, 0);
				let thr = thermo_fit_thresholds(&rows0, bpf);
				let enc = |rows: Vec<(Vec<f64>, [f64; 3])>| -> Vec<(Vec<f64>, [f64; 3])> {
					rows.into_iter().map(|(f, y)| (thermo(&f, &thr), y)).collect()
				};
				let mut rows = enc(rows0.clone());
				let mut line = String::new();
				for it in 0..4
				{
					let (fit_t, r2) = fit(&rows);
					let (new_rows, st, err, steady) = fly_thermo(4, lag, &fit_t, &thr, with_ua, 10, 2000, 100 + it * 10);
					line += &format!(" | it{it} R2 {:.2}/{:.2} fly {st:5.1}% {err:6.2}° {steady:6.2}°", r2[0], r2[1]);
					rows.extend(enc(new_rows));
				}
				eprintln!("[thermodiag] bpf={bpf} T={lag:.2} feats={}{line}", ["e,c", "e,c,ua", "e,ua"][with_ua as usize]);
			}
		}
	}
}

/// fly() with the policy reading thermometer-encoded features.
fn fly_thermo(id: u8, lag: f32, fit_t: &Fit, thr: &[Vec<f64>], with_ua: u8, episodes: usize, steps: usize, seed_off: usize)
	-> (Vec<(Vec<f64>, [f64; 3])>, f64, f64, f64)
{
	fly_with(id, lag, &|f: &[f64]| {
		let b = thermo(f, thr);
		let mut u = [0.0f64; 3];
		for (j, bj) in b.iter().enumerate()
		{
			for ax in 0..3
			{
				u[ax] += fit_t.w[j][ax] * bj;
			}
		}
		u
	}, with_ua, episodes, steps, seed_off)
}

// ---------------------------------------------------------------------------
// Probe 4: the REAL trainer + REAL RAM student (dagger_train_inplace_rs, sn=0,
// BINARY, delta law, 64 levels/motor, 24-bit output neurons, thresholds fitted
// by the G1 calibration sampler at 8 per feature) on the cf21 airframe,
// attitude-only (no translation), mpcof teacher. A/B: T=0 vs T=0.20 delivered
// (ABI 36) vs T=0.20 command (ABI 35 convention, test toggle).
// ---------------------------------------------------------------------------

fn ram_student(nf: usize, thresholds: Vec<f32>, seed: u64, dmax: f32, leak: f32) -> crate::controller::WnnController
{
	use rand::{Rng, SeedableRng};
	let (levels, bpf, obpn) = (64usize, 8usize, 24usize);
	let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
	let frame_bits = nf * bpf;
	let thr = if thresholds.is_empty() { vec![0.0f32; frame_bits] } else { thresholds };
	let out_conn: Vec<i64> = (0..4 * levels * obpn).map(|_| rng.gen_range(0..frame_bits) as i64).collect();
	crate::controller::WnnController::new_core(
		4, levels, bpf, 1, 0, 0, obpn, thr, Vec::new(), out_conn,
		true, dmax, leak, 1.0,
		false, false, true, true, true, true, false, false,
		0.99, 1.0, 0.001, false, 1,
		ram_core::neuron_memory::BINARY, None, None, 0.05, false, 0.30,
		false, false, false, false, false, false, 1,
	)
	.expect("ram student")
}

fn probe_cfg(lag: f32) -> crate::dagger_train::RewardGatedConfigPacked
{
	let mut c = crate::ctrl17_pins::pin_translation_cfg(0.0);
	c.translation = false;
	c.teacher = 4;
	c.motor_lag_s = lag;
	c.steps_per_episode = 2000;
	c.num_rounds = 8;
	c.episodes_per_round = 24;
	c.eval_episodes = 20;
	c.easy_tilt_deg = 8.0;
	c.full_tilt_deg = 20.0;
	c.dist_enabled = false;
	c
}

fn ram_probe(lag: f32, force_command: bool, seed: u64, dmax: f32, leak: f32) -> (f64, f64, Vec<f64>)
{
	use rand::SeedableRng;
	let cfg = probe_cfg(lag);
	let mut c0 = ram_student(9 + 6 + 4, Vec::new(), seed, dmax, leak);
	let nf = c0.num_features();
	let samples = crate::calib_sampler::sample_calibration_features_rs(&mut c0, &cfg, 24, seed ^ 0xCA1);
	let mut thr = Vec::with_capacity(nf * 8);
	for col in samples.iter()
	{
		let mut v = col.clone();
		v.sort_by(|a, b| a.partial_cmp(b).unwrap());
		for k in 0..8
		{
			thr.push(v[((k as f64 + 0.5) / 8.0 * (v.len() - 1) as f64) as usize]);
		}
	}
	let mut c = ram_student(nf, thr, seed, dmax, leak);
	c.set_obs_pwm_force_command(force_command);
	let st = crate::dagger_train::dagger_train_inplace_rs(&mut c, &cfg, [0.0; 3], seed);
	let mut sim = crate::dagger_train::AirframeRs::from_cfg(&cfg).sim();
	c.set_observer_motor_lag_core(sim.motor_lag());
	let mut rng = rand::rngs::SmallRng::seed_from_u64(seed ^ 0xE7A1);
	let (_f, err, stable, _j, _m) = crate::dagger_train::eval_closed_loop_rs(&mut c, &mut sim, &cfg, &mut rng, [0.0; 3], 5f64.to_radians(), 4, 64);
	(stable * 100.0, err.to_degrees(), st.iter_stable_rate.clone())
}

#[test]
#[ignore]
fn diag_ram_student_real_trainer_under_lag()
{
	let laws: Vec<(f32, f32)> = std::env::var("LAGDIAG_LAWS")
		.ok()
		.map(|v| v.split(';').map(|p| { let mut it = p.split(','); (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap()) }).collect())
		.unwrap_or(vec![(0.1, 0.95)]);
	for (dmax, leak, seed) in laws.iter().flat_map(|&(d, l)| [1u64, 2].map(|s| (d, l, s)))
	{
		for (lag, force) in [(0.0f32, false), (0.20, false), (0.20, true)]
		{
			let (st, err, rounds) = ram_probe(lag, force, seed, dmax, leak);
			eprintln!(
				"[ramdiag] dmax={dmax} leak={leak} seed={seed} T={lag:.2} obs_pwm={} eval@5°: stable {st:5.1}% err {err:6.2}° | rounds stable {:?}",
				if lag > 0.0 && !force { "delivered" } else { "command" },
				rounds.iter().map(|v| (v * 100.0).round()).collect::<Vec<_>>()
			);
		}
	}
}
