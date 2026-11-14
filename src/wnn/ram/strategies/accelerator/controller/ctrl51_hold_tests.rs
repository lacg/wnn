//! Hold-window position column (ABI 51, docs/ctrl48_block10_levers_spec.md): the mean
//! 3-D position error and mean |dz| over t >= ceil(0.8·steps) — the `steady` window.
//! Additive output only: appended after the existing values of every scorer.
//!
//! Pinned here: the two new values equal a direct recomputation (rival scorer, position
//! teacher scorer, WNN CPU scorer); CPU rows == Metal rows; the per-episode Metal rows
//! carry them. That the values which existed before did not move is held by the
//! byte-identity pins (abi38_pins / ctrl38_pins / the metal_controller GPU pins), whose
//! digests cover exactly the pre-ABI-51 prefix of each row.

use crate::abi38_pins::{stage2, ARM, ASYM, DT, G, INERTIA, KD, KT, PID_ATT, PID_ATT_HZ, PID_HOVER_N, PID_LPF_HZ, PID_OUT_LIMIT_N, PID_RATE};
use crate::rival_probe::RivalProbe;

type Row10 = (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64);

fn rel_close(a: f64, b: f64, tol: f64, what: &str)
{
	assert!((a - b).abs() <= tol * b.abs().max(1e-12), "{what}: {a} vs {b}");
}

/// The abi38 stage-2 rival fixture (L4C-like weather, tilt reference), probed.
fn rival(teacher: u8, use_est: bool, hold: usize, steps: usize, translation: bool, probe: Option<&mut RivalProbe>) -> Row10
{
	let (q0, w0) = crate::abi38_pins::ics();
	let s1 = stage2();
	let v = |x: &Vec<f32>| if translation { x.clone() } else { Vec::new() };
	crate::dagger_train::score_classical_baseline_probed(
		teacher, q0, w0, steps, 5.0,
		true, [0.0; 3], 0.0, 0.1, ASYM, 0.0107, 0.0, 0.0, 777, 0.0, 0, 0, 0.2,
		ARM, KT, KD, INERTIA, G, DT,
		PID_ATT, PID_RATE, PID_OUT_LIMIT_N, PID_HOVER_N, PID_ATT_HZ, PID_LPF_HZ,
		use_est, 2.0, 0.1,
		translation, 0.0, v(&s1.init_z), v(&s1.init_vz), v(&s1.mass), v(&s1.collective_frac),
		2.0, 1.0, 0.25,
		v(&s1.init_x), v(&s1.init_y), 1.0, 1.0, 0.5236,
		0.20, false, 0.0, translation, hold,
		0,
		probe,
	)
	.expect("rival must score")
}

/// The rival scorer's 9th / 10th values equal the probe's own hold-window accounting
/// (rival_probe.rs record_position — an independent accumulation over the last fifth
/// of the episode), for the motor-level and the cascade rivals; NaN off translation.
#[test]
fn rival_hold_column_equals_direct_recomputation()
{
	for (teacher, hold) in [(0u8, 0usize), (4, 0), (0, 10), (4, 10)]
	{
		let mut probe = RivalProbe::measure_only();
		let r = rival(teacher, true, hold, 600, true, Some(&mut probe));
		let rep = probe.report();
		rel_close(r.8, rep.pos_fifths[4], 1e-12, &format!("teacher {teacher} hold {hold}: hold pos"));
		rel_close(r.9, rep.hold_abs_xyz[2], 1e-12, &format!("teacher {teacher} hold {hold}: hold |dz|"));
		// A different number from the whole-episode means (the window is the last fifth).
		assert!((r.8 - r.4).abs() > 1e-6 && (r.9 - r.3).abs() > 1e-6, "teacher {teacher} hold {hold}: hold == whole episode ({r:?})");
		// |dz| is one component of the 3-D error.
		assert!(r.9 > 0.0 && r.9 <= r.8, "teacher {teacher} hold {hold}: |dz| {} must be within the 3-D error {}", r.9, r.8);
	}
	let off = rival(1, false, 0, 600, false, None);
	assert!(off.8.is_nan() && off.9.is_nan() && off.4.is_nan(), "translation off: the position columns are NaN, got {off:?}");
}

/// Appending the column moved nothing that existed: the first eight values are the
/// pinned row (abi38_pins::rival feeds the ABI-37 digest test).
#[test]
fn rival_first_eight_values_are_the_pinned_row()
{
	for teacher in [0u8, 4]
	{
		let pinned = crate::abi38_pins::rival(teacher, 0.20);
		let (q0, w0) = crate::abi38_pins::ics();
		let s1 = stage2();
		let now = crate::dagger_train::score_classical_baseline(
			teacher, q0, w0, 600, 5.0,
			true, [0.0; 3], 0.0, 0.1, ASYM, 0.0107, 0.0, 0.0, 777, 0.0, 0, 0, 0.2,
			ARM, KT, KD, INERTIA, G, DT,
			PID_ATT, PID_RATE, PID_OUT_LIMIT_N, PID_HOVER_N, PID_ATT_HZ, PID_LPF_HZ,
			true, 2.0, 0.1,
			true, 0.0, s1.init_z.clone(), s1.init_vz.clone(), s1.mass.clone(), s1.collective_frac.clone(),
			2.0, 1.0, 0.25,
			s1.init_x.clone(), s1.init_y.clone(), 1.0, 1.0, 0.5236,
			0.20, false, 0.0, true, 0, 0,
		)
		.expect("rival must score");
		assert_eq!(format!("{:?}", crate::dagger_train::rival_first8(now)), format!("{pinned:?}"), "teacher {teacher}");
		assert!(now.8.is_finite() && now.9.is_finite());
	}
}

/// score_position_teacher: the two appended values equal a hand-flown recomputation
/// of the same cascade (which also reproduces the existing mean position error, so
/// the replica is the scorer's episode).
#[test]
fn position_teacher_hold_column_equals_direct_recomputation()
{
	let (steps, mass, p0) = (1500usize, 0.0393f32, [0.4f32, -0.3, 0.15]);
	let (q0, w0) = ([0.999f32, 0.03, -0.02, 0.0], [0.1f32, -0.1, 0.0]);
	let got = crate::position_score::score_position_teacher(
		1, q0.to_vec(), w0.to_vec(), p0.to_vec(), mass, steps, 5.0,
		1.0, 1.0, 30.0, 2.0, 1.0, 0.25,
		false, [0.0; 3], 0.0, 0.1, [1.0; 4], 0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0.0,
		ARM, KT, KD, INERTIA, G, DT,
		[0.0; 12], [0.0; 12], 0.0, 0.0, 0.0, 0.0,
		false, 2.0, 0.1, 0.0, false, 0.0, false, 0,
	)
	.expect("position teacher scores");
	// The same cascade, by hand.
	let pos = crate::position_loop::PositionLoop::from_plant(G as f64, 1.0, 1.0, 30.0f64.to_radians()).unwrap();
	let pd = crate::altitude_pd::AltitudePd::from_plant(mass as f64, G as f64, KT as f64, 2.0, 1.0, 0.25).unwrap();
	let hover = (mass as f64 * G as f64 / (4.0 * KT as f64)).sqrt();
	let mut teacher = crate::optimal::Teacher::from_id_lagged(1, DT, ARM, KT, KD, INERTIA, G, hover, 0.0);
	let mut sim = crate::controller::AttitudeSim::new(DT, ARM, KT, KD, INERTIA, G);
	teacher.reset();
	sim.reset(Some(q0), Some(w0));
	sim.set_translation_core(mass).unwrap();
	sim.set_vertical_state(p0[2], 0.0);
	sim.set_horizontal_state(p0[0], p0[1], 0.0, 0.0);
	let tail_start = ((steps as f64) * 0.80).ceil() as usize;
	let (mut sum, mut tail, mut n_tail) = (0.0f64, [0.0f64; 2], 0usize);
	let mut last = [0.5f64; 4];
	for t in 0..steps
	{
		assert!(!sim.is_unstable(), "the fixture must not diverge");
		let (gyro, _) = sim.read_imu();
		let (x, y) = sim.position_xy();
		let (vx, vy) = sim.velocity_xy();
		teacher.observe(gyro, last);
		let cmd = teacher.step_full_state(
			sim.quaternion(), gyro, 0.0, &pos, &pd,
			-(x as f64), vx as f64, -(y as f64), vy as f64,
			-(sim.altitude_rs() as f64), sim.vertical_velocity_rs() as f64,
		);
		sim.step(cmd.map(|v| v as f32));
		last = cmd;
		let (x, y) = sim.position_xy();
		let pe = ((x * x + y * y + sim.altitude_rs() * sim.altitude_rs()) as f64).sqrt();
		sum += pe;
		if t >= tail_start
		{
			tail[0] += pe;
			tail[1] += (sim.altitude_rs() as f64).abs();
			n_tail += 1;
		}
	}
	rel_close(got.0, sum / steps as f64, 1e-12, "the replica is not the scorer's episode (mean position error)");
	rel_close(got.5, tail[0] / n_tail as f64, 1e-12, "hold position error");
	rel_close(got.6, tail[1] / n_tail as f64, 1e-12, "hold |dz|");
	assert!(got.5 < got.0, "a converging hold must end closer than its episode mean");
}

/// WNN CPU scorer: rows 17 / 18 equal a hand-rolled recomputation of the same
/// episodes (which also reproduces rows 13 / 14, so it is the scorer's rollout).
#[test]
fn wnn_hold_rows_equal_direct_recomputation()
{
	use crate::abi32_tests::{cfg, controller, cpu, ics, ARM as A, DT as D, EPS, G as GR, INERTIA as IN, KD as KDR, KT as K};
	use crate::att_err_ref::AttErrRef;
	use crate::plant_opts::PlantOpts;
	let steps = 400usize;
	let c = controller(7, false);
	let s1 = cfg(PlantOpts::OFF, AttErrRef::LEVEL);
	let rows = cpu(&c, &s1, steps);
	let (q0, w0) = ics();
	let tail_start = ((steps as f64) * 0.80).ceil() as usize;
	let mut sim = crate::controller::AttitudeSim::new(D, A, K, KDR, IN, GR);
	let mut cc = c.clone();
	let (mut s_pos, mut s_alt, mut s_hold, mut s_hold_dz) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
	for ep in 0..EPS
	{
		let q = [q0[ep * 4], q0[ep * 4 + 1], q0[ep * 4 + 2], q0[ep * 4 + 3]];
		let om = [w0[ep * 3], w0[ep * 3 + 1], w0[ep * 3 + 2]];
		cc.reset(crate::controller::yaw_from_quat_rs(q));
		cc.set_decode_seed(crate::controller::disturbance_episode_seed(0, ep as u64));
		sim.reset(Some(q), Some(om));
		sim.set_translation_core(s1.mass[ep]).unwrap();
		sim.set_vertical_state(s1.init_z[ep], s1.init_vz[ep]);
		sim.set_horizontal_state(s1.init_x[ep], s1.init_y[ep], 0.0, 0.0);
		let coll = s1.collective_pwm(ep, GR, K);
		cc.set_collective_anchor(coll);
		let (mut pos, mut alt, mut tail, mut n_tail, mut done) = (0.0f64, 0.0f64, [0.0f64; 2], 0usize, 0usize);
		for t in 0..steps
		{
			if sim.is_unstable()
			{
				break;
			}
			let (gyro, accel) = sim.read_imu();
			cc.set_vertical_obs(coll, s1.target_altitude - sim.altitude_rs(), sim.vertical_velocity_rs());
			let [hx, hy] = sim.position_xy_rs();
			let [hvx, hvy] = sim.velocity_xy_rs();
			cc.set_horizontal_obs(-hx, -hy, hvx, hvy);
			let pwm = cc.step(gyro, accel, [0.0; 3]);
			sim.step([pwm[0], pwm[1], pwm[2], pwm[3]]);
			let [hx, hy] = sim.position_xy_rs();
			let ae = (s1.target_altitude - sim.altitude_rs()) as f64;
			let p = ((hx * hx + hy * hy) as f64 + ae * ae).sqrt();
			pos += p;
			alt += ae.abs();
			if t >= tail_start
			{
				tail[0] += p;
				tail[1] += ae.abs();
				n_tail += 1;
			}
			done += 1;
		}
		assert!(n_tail > 0, "episode {ep} never reached the hold window — pick a calmer fixture");
		s_pos += pos / done as f64;
		s_alt += alt / done as f64;
		s_hold += tail[0] / n_tail as f64;
		s_hold_dz += tail[1] / n_tail as f64;
	}
	let n = EPS as f64;
	rel_close(rows[13], s_pos / n, 1e-12, "the replica is not the scorer's rollout (pos)");
	rel_close(rows[14], s_alt / n, 1e-12, "the replica is not the scorer's rollout (alt)");
	rel_close(rows[17], s_hold / n, 1e-12, "row 17: hold position error");
	rel_close(rows[18], s_hold_dz / n, 1e-12, "row 18: hold |dz|");
	assert!((rows[17] - rows[13]).abs() > 1e-6 && rows[18] > 0.0 && rows[18] <= rows[17], "vacuous hold rows: {rows:?}");
	// Translation off: the rows exist and are 0, like pos / alt.
	let mut cc = c.clone();
	let off = crate::cpu_score::rollout_one(
		&mut cc, &q0, &w0, EPS, 200, D, A, K, KDR, IN, GR, 0.0, [0.0; 3], false, [0.0; 3],
		0.0, 0.1, [1.0; 4], 0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0.0, 4, 4, None, None, None, 1.0, 0.4, None, None,
	);
	assert_eq!((off[17], off[18]), (0.0, 0.0));
}

/// Metal rows 17 / 18 == the CPU scorer's (abi32 parity helper, which now compares
/// them in EVERY score-parity test), and the per-episode rows carry the two values
/// at indices 13 / 14, averaging to the aggregate.
#[cfg(target_os = "macos")]
#[test]
fn wnn_hold_rows_cpu_match_gpu()
{
	use crate::abi32_tests::{assert_parity_steps, cfg, controller, ics, tilt_ref, ARM as A, DT as D, EPS, G as GR, INERTIA as IN, KD as KDR, KT as K};
	let plant = crate::plant_opts::PlantOpts::new(true, 0.78).unwrap();
	let c = controller(7, true);
	let s1 = cfg(plant, tilt_ref());
	let cr = assert_parity_steps(&c, &s1, 400, "hold column");
	assert!(cr[17] > 0.0 && cr[18] > 0.0, "vacuous hold rows");
	let Ok(ev) = crate::metal_controller::ControllerRolloutEvaluator::new()
	else
	{
		return;
	};
	let (q0, w0) = ics();
	let (agg, eps) = ev
		.score_with_episodes(&[&c], &q0, &w0, EPS, 400, (D, A, K, IN, GR), KDR, 0.0, [0.0; 3], None, None, None, None, Some(&s1))
		.expect("gpu score");
	assert_eq!(agg[0].len(), 19);
	assert_eq!(eps.len(), EPS);
	for (row, col) in [(17usize, 13usize), (18, 14)]
	{
		assert!(eps.iter().all(|e| e.len() == 15), "per-episode rows must carry the hold column (15 values)");
		let mean = eps.iter().map(|e| e[col]).sum::<f64>() / EPS as f64;
		rel_close(mean, agg[0][row], 1e-9, &format!("per-episode column {col} vs aggregate row {row}"));
	}
}
