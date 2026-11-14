//! docs/ctrl_gap_after_block9.md §B.3 — LABEL-REPLAY ERROR on the student's own visited
//! states: does the trained table reproduce its label where it actually flies?
//!
//! Input: a per-tick trace of a trained student's flight on VAL units, written by
//! scripts/ctrl_gap_student_trace.py through the installed wheel (26 f32 per tick:
//! [unit_idx, ep, t, gyro ×3, q_true ×4, x, y, vx, vy, alt, vz, sp ×4, pwm ×4,
//! collective anchor, target altitude]), plus the unit fixture of rival_probe_tests
//! (airframe, firmware cascade, outer-loop shapes).
//!
//! What runs here is the DAgger labelling order of dagger_train.rs, on the recorded
//! states instead of a live sim:
//!   teacher.observe(gyro_t, applied_{t−1}) → ctbr_sync_inner(the REAL rate loop, before
//!   this tick's update) → ctbr_label(q_TRUE, gyro, tilt reference, hold, altitude-PD
//!   collective) — the label a `label=true` run records at this state.
//! The real rate loop is not in the trace: it is REBUILT by stepping the firmware rate
//! stage on the student's recorded setpoints and gyro, and GATED on reproducing the
//! recorded motor command every tick. If that gate holds, the inner-loop state the
//! teacher plans from is the vehicle's.
//!
//!   WNN_INDI_PROBE_FIXTURE=/tmp/ctrl_indi_probe/fixture WNN_GAP_TRACE=/tmp/ctrl_gap/both_s2.trace.f32 \
//!     RAYON_NUM_THREADS=4 nice -n 10 cargo test -p ram_controller --lib --no-default-features --release \
//!     label_replay_tests -- --ignored --nocapture --test-threads 1

use crate::dagger_train::{nominal_hover_pwm, nominal_mass_kg, AirframeRs};
use crate::rival_probe_tests::{units, Unit};

const W: usize = 26;
const HOLD: usize = 10;
const BANKS: [&str; 4] = ["collective (pwm)", "p (rad/s)", "q (rad/s)", "r (rad/s)"];

struct Rig
{
	af: AirframeRs,
	alt_pd: crate::altitude_pd::AltitudePd,
	pos: crate::position_loop::PositionLoop,
}

/// The teacher's world, built as score_classical_baseline builds it.
fn rig(u: &Unit) -> Rig
{
	let (g, kt, hover_n) = (u.s("af_gravity") as f32, u.s("af_k_thrust") as f32, u.s("af_pid_hover_n"));
	let (plant, _) = crate::stage1::stage1_extras(
		u.b("imu_specific_force"),
		u.s("rotor_drag") as f32,
		u.b("att_err_tilt_ref"),
		g,
		u.s("pos_omega_n") as f32,
		u.s("pos_zeta") as f32,
		u.s("pos_max_tilt_rad") as f32,
	)
	.unwrap();
	let masses = u.v32("s1_mass");
	let mean_mass = masses.iter().sum::<f32>() / masses.len() as f32;
	let m_nom = nominal_mass_kg(hover_n, mean_mass, g);
	Rig {
		af: AirframeRs {
			dt: u.s("af_dt") as f32,
			arm_length: u.s("af_arm_length") as f32,
			k_thrust: kt,
			k_drag: u.s("af_k_drag") as f32,
			inertia: u.a32("af_inertia"),
			gravity: g,
			pid_fw: crate::pid_firmware::AttitudePidFirmwareRs::from_si_arrays(
				u.a64("af_pid_att"),
				u.a64("af_pid_rate"),
				u.s("af_pid_out_limit_n"),
				hover_n,
				kt as f64,
				(1.0 / u.s("af_dt")).round() as u32,
				u.s("af_pid_attitude_hz"),
				u.s("af_pid_lpf_hz"),
			),
			teacher_hover: Some(nominal_hover_pwm(hover_n, mean_mass, g, kt)),
			motor_lag_s: u.s("motor_lag_s") as f32,
			plant,
		},
		alt_pd: crate::altitude_pd::AltitudePd::from_plant(m_nom as f64, g as f64, kt as f64, u.s("alt_pd_omega"), u.s("alt_pd_zeta"), u.s("alt_pd_max_delta")).unwrap(),
		pos: crate::position_loop::PositionLoop::from_plant(g as f64, u.s("pos_omega_n"), u.s("pos_zeta"), u.s("pos_max_tilt_rad")).unwrap(),
	}
}

const WINDOWS: [&str; 3] = ["whole episode", "first 0.4 s", "hold (t ≥ 1.6 s)"];

/// Per bank, over decision ticks: sums for the three WINDOWS.
#[derive(Clone, Default)]
struct Acc
{
	n: [f64; 3],
	label_sq: [[f64; 4]; 3],
	diff_sq: [[f64; 4]; 3],
	diff_abs: [[f64; 4]; 3],
	/// For the slope / R² of student on label, per window.
	sl: [[f64; 4]; 3],
	ss: [[f64; 4]; 3],
	sss: [[f64; 4]; 3],
	sls: [[f64; 4]; 3],
	/// Sum over episodes of |episode-mean hold (student − label)|, and the episode count.
	ep_bias_abs: [f64; 4],
	ep_bias_signed: [f64; 4],
	episodes: f64,
	max_pwm_gap: f64,
}

impl Acc
{
	fn merge(mut self, o: Acc) -> Acc
	{
		for w in 0..3
		{
			self.n[w] += o.n[w];
			for m in 0..4
			{
				self.label_sq[w][m] += o.label_sq[w][m];
				self.diff_sq[w][m] += o.diff_sq[w][m];
				self.diff_abs[w][m] += o.diff_abs[w][m];
				self.sl[w][m] += o.sl[w][m];
				self.ss[w][m] += o.ss[w][m];
				self.sss[w][m] += o.sss[w][m];
				self.sls[w][m] += o.sls[w][m];
			}
		}
		for m in 0..4
		{
			self.ep_bias_abs[m] += o.ep_bias_abs[m];
			self.ep_bias_signed[m] += o.ep_bias_signed[m];
		}
		self.episodes += o.episodes;
		self.max_pwm_gap = self.max_pwm_gap.max(o.max_pwm_gap);
		self
	}
}

/// One recorded episode: rebuild the inner loop, label every tick, accumulate.
fn replay_episode(r: &Rig, ticks: &[f32], acc: &mut Acc)
{
	let mut teacher = r.af.teacher(4);
	teacher.reset();
	let mut real = r.af.pid_fw.as_ref().unwrap().rate_stage();
	real.reset();
	let steps_total = 2000usize;
	let tail = (steps_total as f64 * 0.8).ceil() as usize;
	let mut last = [0.5f64; 4];
	let (mut hold_sum, mut hold_n) = ([0.0f64; 4], 0.0f64);
	for rec in ticks.chunks_exact(W)
	{
		let t = rec[2] as usize;
		let gyro = [rec[3], rec[4], rec[5]];
		let q = [rec[6], rec[7], rec[8], rec[9]];
		let (x, y, vx, vy, alt, vz) = (rec[10] as f64, rec[11] as f64, rec[12] as f64, rec[13] as f64, rec[14], rec[15]);
		let sp = [rec[16] as f64, rec[17] as f64, rec[18] as f64, rec[19] as f64];
		let pwm = [rec[20] as f64, rec[21] as f64, rec[22] as f64, rec[23] as f64];
		let (anchor, talt) = (rec[24] as f64, rec[25]);
		teacher.observe(gyro, last);
		let pre = real;
		teacher.ctbr_sync_inner(&pre);
		let coll = r.alt_pd.delta((talt - alt) as f64, vz as f64);
		let (rr, rp) = r.pos.tilt_ref(-x, vx, -y, vy);
		let label = teacher.ctbr_label(q, gyro, [rr as f32, rp as f32, 0.0], HOLD, coll).unwrap();
		// The vehicle's own inner loop on the student's setpoint (ctbr.rs CtbrInner::fly).
		let flown = real.step([sp[1], sp[2], sp[3]], [gyro[0] as f64, gyro[1] as f64, gyro[2] as f64], (anchor - real.hover_pwm()) + sp[0]);
		for m in 0..4
		{
			acc.max_pwm_gap = acc.max_pwm_gap.max((flown[m].clamp(0.0, 1.0) - pwm[m]).abs());
		}
		last = pwm;
		if t % HOLD != 0
		{
			continue;
		}
		for w in 0..3
		{
			if (w == 1 && t >= 400) || (w == 2 && t < tail)
			{
				continue;
			}
			acc.n[w] += 1.0;
			for m in 0..4
			{
				let d = sp[m] - label[m];
				acc.label_sq[w][m] += label[m] * label[m];
				acc.diff_sq[w][m] += d * d;
				acc.diff_abs[w][m] += d.abs();
				acc.sl[w][m] += label[m];
				acc.ss[w][m] += sp[m];
				acc.sss[w][m] += sp[m] * sp[m];
				acc.sls[w][m] += label[m] * sp[m];
			}
		}
		if t >= tail
		{
			hold_n += 1.0;
			for m in 0..4
			{
				hold_sum[m] += sp[m] - label[m];
			}
		}
	}
	if hold_n > 0.0
	{
		for m in 0..4
		{
			acc.ep_bias_abs[m] += (hold_sum[m] / hold_n).abs();
			acc.ep_bias_signed[m] += hold_sum[m] / hold_n;
		}
		acc.episodes += 1.0;
	}
}

#[test]
#[ignore = "needs WNN_GAP_TRACE (a recorded student flight on VAL units) and WNN_INDI_PROBE_FIXTURE"]
fn label_replay_on_student_states()
{
	use rayon::prelude::*;
	let units = units();
	let path = std::env::var("WNN_GAP_TRACE").expect("set WNN_GAP_TRACE to the student trace");
	let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
	assert_eq!(bytes.len() % (4 * W), 0, "trace is not a whole number of {W}-float records");
	let data: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
	// Episode boundaries: a change of (unit_idx, ep).
	let mut spans: Vec<(usize, usize, usize)> = Vec::new(); // (unit_idx, start, end) in floats
	let mut start = 0usize;
	let n = data.len() / W;
	for i in 1..=n
	{
		if i == n || data[i * W] != data[start] || data[i * W + 1] != data[start + 1]
		{
			spans.push((data[start] as usize, start, i * W));
			start = i * W;
		}
	}
	let rigs: Vec<Rig> = units.iter().map(rig).collect();
	let acc = spans
		.par_iter()
		.map(|&(ui, a, b)| {
			let mut acc = Acc::default();
			replay_episode(&rigs[ui], &data[a..b], &mut acc);
			acc
		})
		.reduce(Acc::default, Acc::merge);
	// GATE: the rebuilt inner loop IS the vehicle's (f32 trace ⇒ ~1e-7 rounding).
	assert!(acc.max_pwm_gap < 5e-5, "the rebuilt rate loop does not reproduce the recorded motor command (max gap {}) — labels would be planned from the wrong inner state", acc.max_pwm_gap);
	println!("\n[label-replay] {path}: {} episodes, {} decision ticks ({} in the hold); rebuilt inner loop vs recorded pwm max gap {:.1e}", acc.episodes, acc.n[0], acc.n[2], acc.max_pwm_gap);
	println!("| bank | window | label RMS | student RMS | (student − label) RMS | mean abs | error / label RMS | slope student on label | R² | mean abs episode-mean hold bias | mean signed hold bias |");
	println!("|---|---|---|---|---|---|---|---|---|---|---|");
	for m in 0..4
	{
		for (w, name) in WINDOWS.iter().enumerate()
		{
			let n = acc.n[w];
			let (lr, dr, sr) = ((acc.label_sq[w][m] / n).sqrt(), (acc.diff_sq[w][m] / n).sqrt(), (acc.sss[w][m] / n).sqrt());
			let (ml, ms) = (acc.sl[w][m] / n, acc.ss[w][m] / n);
			let (vl, vs, cov) = (acc.label_sq[w][m] / n - ml * ml, acc.sss[w][m] / n - ms * ms, acc.sls[w][m] / n - ml * ms);
			let (eb, es) = if w == 2
			{
				(format!("{:.5}", acc.ep_bias_abs[m] / acc.episodes), format!("{:+.5}", acc.ep_bias_signed[m] / acc.episodes))
			}
			else
			{
				("-".into(), "-".into())
			};
			println!("| {} | {name} | {lr:.5} | {sr:.5} | {dr:.5} | {:.5} | {:.2} | {:.3} | {:.3} | {eb} | {es} |", BANKS[m], acc.diff_abs[w][m] / n, dr / lr, cov / vl, cov * cov / (vl * vs));
		}
	}
}
