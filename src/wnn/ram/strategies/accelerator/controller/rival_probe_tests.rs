//! rival_probe.rs — (1) the probe is inert when off (bit-identity with the shipped
//! scorer), (2) its two options do what they say, (3) the `#[ignore]` VAL-unit sweeps
//! pre-registered in docs/ctrl_indi_teacher_spec.md §10.
//!
//! The sweeps read a fixture written by scripts/ctrl_indi_val_fixture.py: one text file
//! per val unit holding the EXACT arguments the shipped Python samplers hand
//! `score_classical_baseline` for the MPCOF>rate(h10) rival row (one `key v v v ...` line
//! per argument). The episode set is therefore the scorer's own — nothing is re-sampled
//! here. Run (one sweep at a time, niced, capped threads):
//!
//!   WNN_INDI_PROBE_FIXTURE=/tmp/ctrl_indi_probe/fixture RAYON_NUM_THREADS=4 nice -n 10 \
//!     cargo test -p ram_controller --lib --no-default-features --release \
//!     rival_probe_tests::val_sweep -- --ignored --nocapture --test-threads 1

use crate::rival_probe::{RivalProbe, RivalProbeReport, StudentAlphabet};
use std::collections::HashMap;

type Row = (f64, f64, f64, f64, f64, f64, f64, f64);

fn bits(t: Row) -> [u64; 8]
{
	[t.0, t.1, t.2, t.3, t.4, t.5, t.6, t.7].map(f64::to_bits)
}

/// The abi38 cascade-rival fixture (stage 2, L4C-like, est-fed, tilt reference).
fn small(teacher: u8, use_est: bool, probe: Option<&mut RivalProbe>) -> pyo3::PyResult<Row>
{
	use crate::abi38_pins::*;
	let (q0, w0) = ics();
	let s1 = stage2();
	crate::dagger_train::score_classical_baseline_probed(
		teacher, q0, w0, 600, 5.0,
		true, [0.0; 3], 0.0, 0.1, ASYM, 0.0107, 0.0, 0.0, 777, 0.0, 0, 0, 0.2,
		ARM, KT, KD, INERTIA, G, DT,
		PID_ATT, PID_RATE, PID_OUT_LIMIT_N, PID_HOVER_N, PID_ATT_HZ, PID_LPF_HZ,
		use_est, 2.0, 0.1,
		true, 0.0, s1.init_z.clone(), s1.init_vz.clone(), s1.mass.clone(), s1.collective_frac.clone(),
		2.0, 1.0, 0.25,
		s1.init_x.clone(), s1.init_y.clone(), 1.0, 1.0, 0.5236,
		0.20, true, 0.0, true, 10,
		0, (0.0, 0.0), // alt_integral off
		probe,
	)
	.map(crate::dagger_train::rival_first8)
}

/// Probe off ⇒ the shipped scorer, bit for bit — for both cascade rivals and both feeds.
#[test]
fn probe_off_is_the_shipped_scorer()
{
	for teacher in [0u8, 4]
	{
		for use_est in [false, true]
		{
			let shipped = small(teacher, use_est, None).unwrap();
			let mut p = RivalProbe::measure_only();
			let probed = small(teacher, use_est, Some(&mut p)).unwrap();
			assert_eq!(bits(shipped), bits(probed), "teacher {teacher} est {use_est}: a measure-only probe moved the score");
			let r = p.report();
			assert!(r.episodes > 0 && r.hold_abs.iter().all(|v| v.is_finite()), "{r:?}");
			if !use_est
			{
				assert_eq!(r.fed_bias_abs, [0.0; 2], "oracle-fed: the fed quaternion IS the truth");
			}
		}
	}
}

/// The shipped gain passed explicitly is the shipped scorer; another gain is not.
#[test]
fn observer_gain_override_reaches_the_teacher_and_only_when_asked()
{
	let shipped = small(4, true, None).unwrap();
	let mut same = RivalProbe::measure_only().with_observer_gain(crate::ctbr_mpcof::CTBR_MPC_L_GAIN);
	assert_eq!(bits(shipped), bits(small(4, true, Some(&mut same)).unwrap()));
	let mut fast = RivalProbe::measure_only().with_observer_gain(0.2);
	assert_ne!(bits(shipped), bits(small(4, true, Some(&mut fast)).unwrap()), "gain 0.2 changed nothing");
	// The firmware attitude stage has no observer: refused, not silently ignored.
	let mut pid = RivalProbe::measure_only().with_observer_gain(0.2);
	assert!(small(0, true, Some(&mut pid)).is_err());
	let mut bad = RivalProbe::measure_only().with_observer_gain(0.0);
	assert!(small(4, true, Some(&mut bad)).is_err());
}

/// Bias compensation needs an estimator-fed row, changes the flight, and its
/// prediction explains part of the fed attitude's bias.
#[test]
fn bias_compensation_is_est_only_and_tracks_the_bias()
{
	let mut oracle = RivalProbe::measure_only().with_bias_comp();
	assert!(small(4, false, Some(&mut oracle)).is_err(), "bias_comp on an oracle-fed row must be refused");
	let shipped = small(4, true, None).unwrap();
	let mut p = RivalProbe::measure_only().with_bias_comp();
	let comp = small(4, true, Some(&mut p)).unwrap();
	assert_ne!(bits(shipped), bits(comp));
	let r = p.report();
	assert!(
		r.comp_resid_abs[0] < r.fed_bias_abs[0] && r.comp_resid_abs[1] < r.fed_bias_abs[1],
		"the predictor does not reduce the fed bias: {r:?}"
	);
}

// ---------------------------------------------------------------------------------
// The val-unit sweeps (ignored: minutes of CPU, need the fixture).
// ---------------------------------------------------------------------------------

pub(crate) struct Unit
{
	pub(crate) seed: u64,
	/// Parsed as an integer: a u64 seed does not survive an f64.
	dist_seed: u64,
	f: HashMap<String, Vec<f64>>,
}

impl Unit
{
	fn load(path: &std::path::Path, allow_test: bool) -> Unit
	{
		let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
		let mut f = HashMap::new();
		let mut dist_seed = None;
		for line in text.lines().filter(|l| !l.trim().is_empty())
		{
			let mut it = line.split_whitespace();
			let key = it.next().unwrap().to_string();
			if key == "dist_seed"
			{
				dist_seed = Some(it.next().unwrap().parse::<u64>().expect("dist_seed: not a u64"));
				continue;
			}
			let vals: Vec<f64> = it.map(|v| v.parse().unwrap_or_else(|_| panic!("{key}: bad number {v}"))).collect();
			f.insert(key, vals);
		}
		let seed = f["unit_seed"][0] as u64;
		// Report-only units never enter a tuning read (spec §10.1).
		assert!(allow_test || !(99990101..=99990400).contains(&seed), "unit {seed} is a TEST unit — refused");
		Unit { seed, dist_seed: dist_seed.expect("fixture has no dist_seed"), f }
	}

	pub(crate) fn v(&self, k: &str) -> &Vec<f64>
	{
		self.f.get(k).unwrap_or_else(|| panic!("fixture has no key {k}"))
	}

	pub(crate) fn s(&self, k: &str) -> f64
	{
		let v = self.v(k);
		assert_eq!(v.len(), 1, "{k} is not a scalar");
		v[0]
	}

	pub(crate) fn b(&self, k: &str) -> bool
	{
		self.s(k) != 0.0
	}

	pub(crate) fn v32(&self, k: &str) -> Vec<f32>
	{
		self.v(k).iter().map(|&x| x as f32).collect()
	}

	pub(crate) fn a32<const N: usize>(&self, k: &str) -> [f32; N]
	{
		let v = self.v32(k);
		v.try_into().unwrap_or_else(|_| panic!("{k}: wrong length"))
	}

	pub(crate) fn a64<const N: usize>(&self, k: &str) -> [f64; N]
	{
		self.v(k).clone().try_into().unwrap_or_else(|_| panic!("{k}: wrong length"))
	}

	fn episodes(&self) -> usize
	{
		self.v("init_qs").len() / 4
	}

	fn score(&self, use_est: bool, probe: Option<&mut RivalProbe>) -> Row
	{
		self.score_tweaked(use_est, Tweak::default(), probe)
	}

	/// The same row with the TASK changed (position-loop ω_n, episode length) — the
	/// what-ifs of docs/ctrl_gap_after_block9.md §C. Default = the fixture's own values.
	fn score_tweaked(&self, use_est: bool, tw: Tweak, probe: Option<&mut RivalProbe>) -> Row
	{
		crate::dagger_train::score_classical_baseline_probed(
			tw.teacher.unwrap_or(4),
			self.v32("init_qs"),
			self.v32("init_omegas"),
			tw.steps.unwrap_or(self.s("steps") as usize),
			self.s("stable_deg"),
			self.b("dist_enabled"),
			self.a32("dist_tau_bias"),
			self.s("dist_gust_sigma") as f32,
			self.s("dist_gust_tau_c") as f32,
			self.a32("dist_motor_asym"),
			self.s("dist_gyro_sigma") as f32,
			self.s("dist_gyro_bias_walk") as f32,
			self.s("dist_accel_sigma") as f32,
			self.dist_seed,
			self.s("dist_dropout_prob") as f32,
			self.s("dist_dropout_len_steps") as u32,
			self.s("dist_obs_delay_steps") as u32,
			self.s("dist_torque_scale_jitter") as f32,
			self.s("af_arm_length") as f32,
			self.s("af_k_thrust") as f32,
			self.s("af_k_drag") as f32,
			self.a32("af_inertia"),
			self.s("af_gravity") as f32,
			self.s("af_dt") as f32,
			self.a64("af_pid_att"),
			self.a64("af_pid_rate"),
			self.s("af_pid_out_limit_n"),
			self.s("af_pid_hover_n"),
			self.s("af_pid_attitude_hz"),
			self.s("af_pid_lpf_hz"),
			use_est,
			self.s("est_kp"),
			self.s("est_ki"),
			self.b("translation"),
			self.s("s1_target_altitude") as f32,
			self.v32("s1_init_z"),
			self.v32("s1_init_vz"),
			self.v32("s1_mass"),
			self.v32("s1_collective_frac"),
			self.s("alt_pd_omega"),
			self.s("alt_pd_zeta"),
			self.s("alt_pd_max_delta"),
			self.v32("s2_init_x"),
			self.v32("s2_init_y"),
			tw.pos_omega_n.unwrap_or(self.s("pos_omega_n")),
			self.s("pos_zeta"),
			self.s("pos_max_tilt_rad"),
			self.s("motor_lag_s") as f32,
			self.b("imu_specific_force"),
			self.s("rotor_drag") as f32,
			self.b("att_err_tilt_ref"),
			self.s("ctbr_outer_hold") as usize,
			tw.alt_integral, (0.0, 0.0),
			probe,
		)
		.map(crate::dagger_train::rival_first8)
		.unwrap_or_else(|e| panic!("unit {}: {e}", self.seed))
	}
}

#[derive(Clone, Copy, Default)]
struct Tweak
{
	pos_omega_n: Option<f64>,
	steps: Option<usize>,
	/// Block 10 L3: the altitude integrator mode (0 off, 1 pi, 2 pole3) and the teacher
	/// (None = MPCOF, Some(0) = the firmware PID cascade).
	alt_integral: u8,
	teacher: Option<u8>,
}

pub(crate) fn units() -> Vec<Unit>
{
	let u = units_from("WNN_INDI_PROBE_FIXTURE", false);
	assert_eq!(u.len(), 20, "the pre-registration names 20 val units, found {}", u.len());
	u
}

fn units_from(var: &str, allow_test: bool) -> Vec<Unit>
{
	let dir = std::env::var(var).unwrap_or_else(|_| panic!("set {var} to the fixture directory"));
	let mut paths: Vec<_> = std::fs::read_dir(&dir)
		.unwrap_or_else(|e| panic!("{dir}: {e}"))
		.map(|e| e.unwrap().path())
		.filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("unit_")))
		.collect();
	paths.sort();
	paths.iter().map(|p| Unit::load(p, allow_test)).collect()
}

/// One arm on every unit: per-unit (row, report), units in parallel.
fn fly(units: &[Unit], use_est: bool, make: &(dyn Fn() -> RivalProbe + Sync)) -> Vec<(Row, RivalProbeReport)>
{
	fly_tweaked(units, use_est, Tweak::default(), make)
}

fn fly_tweaked(units: &[Unit], use_est: bool, tw: Tweak, make: &(dyn Fn() -> RivalProbe + Sync)) -> Vec<(Row, RivalProbeReport)>
{
	use rayon::prelude::*;
	units
		.par_iter()
		.map(|u| {
			let mut p = make();
			let r = u.score_tweaked(use_est, tw, Some(&mut p));
			(r, p.report())
		})
		.collect()
}

fn print_position_table(title: &str, units: &[Unit], arms: &[Arm])
{
	println!("\n{title}");
	println!("| arm | stable % | err ° | steady ° | pos m | fails | pos at start m | fifth 1 | fifth 2 | fifth 3 | fifth 4 | fifth 5 (hold) | pos at end m | mean abs dx / dy / dz m | hold abs dx / dy / dz m | hold speed m/s |");
	println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
	for a in arms
	{
		let pu = &a.per_unit;
		let fails: usize = pu.iter().zip(units).map(|((r, _), u)| ((1.0 - r.0) * u.episodes() as f64).round() as usize).sum();
		let m = |f: &dyn Fn(&RivalProbeReport) -> f64| mean(pu.iter().map(|(_, p)| f(p)));
		println!(
			"| {} | {:.2} | {:.3} | {:.3} | {:.3} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} / {:.3} / {:.3} | {:.3} / {:.3} / {:.3} | {:.3} |",
			a.name,
			100.0 * mean(pu.iter().map(|(r, _)| r.0)),
			mean(pu.iter().map(|(r, _)| r.1)),
			mean(pu.iter().map(|(r, _)| r.2)),
			mean(pu.iter().map(|(r, _)| r.4)),
			fails,
			m(&|p| p.pos_first_last[0]),
			m(&|p| p.pos_fifths[0]),
			m(&|p| p.pos_fifths[1]),
			m(&|p| p.pos_fifths[2]),
			m(&|p| p.pos_fifths[3]),
			m(&|p| p.pos_fifths[4]),
			m(&|p| p.pos_first_last[1]),
			m(&|p| p.abs_xyz[0]),
			m(&|p| p.abs_xyz[1]),
			m(&|p| p.abs_xyz[2]),
			m(&|p| p.hold_abs_xyz[0]),
			m(&|p| p.hold_abs_xyz[1]),
			m(&|p| p.hold_abs_xyz[2]),
			m(&|p| p.hold_speed),
		);
	}
}

struct Arm
{
	name: String,
	per_unit: Vec<(Row, RivalProbeReport)>,
}

fn mean(v: impl Iterator<Item = f64>) -> f64
{
	let x: Vec<f64> = v.collect();
	x.iter().sum::<f64>() / x.len() as f64
}

fn print_table(title: &str, units: &[Unit], arms: &[Arm], control: usize)
{
	println!("\n{title}");
	println!("| arm | stable % | err ° | steady ° | pos m | fails /{} | Δ steady vs shipped ° | units better/worse | hold roll ° | hold pitch ° | hold yaw ° | fed bias r/p ° | after comp r/p ° |", units.iter().map(Unit::episodes).sum::<usize>());
	println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
	let ctl = &arms[control].per_unit;
	for a in arms
	{
		let pu = &a.per_unit;
		let fails: usize = pu.iter().zip(units).map(|((r, _), u)| ((1.0 - r.0) * u.episodes() as f64).round() as usize).sum();
		let d: Vec<f64> = pu.iter().zip(ctl).map(|((r, _), (c, _))| r.2 - c.2).collect();
		let (better, worse) = (d.iter().filter(|&&x| x < 0.0).count(), d.iter().filter(|&&x| x > 0.0).count());
		let deg = |f: &dyn Fn(&RivalProbeReport) -> f64| mean(pu.iter().map(|(_, p)| f(p))).to_degrees();
		println!(
			"| {} | {:.2} | {:.3} | {:.3} | {:.3} | {} | {:+.3} | {}/{} | {:.3} | {:.3} | {:.3} | {:.2} / {:.2} | {:.2} / {:.2} |",
			a.name,
			100.0 * mean(pu.iter().map(|(r, _)| r.0)),
			mean(pu.iter().map(|(r, _)| r.1)),
			mean(pu.iter().map(|(r, _)| r.2)),
			mean(pu.iter().map(|(r, _)| r.4)),
			fails,
			mean(d.iter().copied()),
			better,
			worse,
			deg(&|p| p.hold_abs[0]),
			deg(&|p| p.hold_abs[1]),
			deg(&|p| p.hold_abs[2]),
			deg(&|p| p.fed_bias_abs[0]),
			deg(&|p| p.fed_bias_abs[1]),
			deg(&|p| p.comp_resid_abs[0]),
			deg(&|p| p.comp_resid_abs[1]),
		);
	}
}

/// Gate before any reading (spec §10.1): on the first unit, (a) a measure-only probe is
/// the unprobed scorer bit for bit, (b) the unprobed est-fed row equals what the
/// INSTALLED wheel returned for the same arguments when the fixture was written.
fn self_check(units: &[Unit])
{
	let u = &units[0];
	let shipped = u.score(true, None);
	let mut p = RivalProbe::measure_only();
	assert_eq!(bits(shipped), bits(u.score(true, Some(&mut p))), "measure-only probe moved the score");
	let want = u.v("expect_est_row");
	let got = [shipped.0, shipped.1, shipped.2, shipped.4];
	for i in 0..4
	{
		assert!(
			(got[i] - want[i]).abs() <= 1e-9 * want[i].abs().max(1.0),
			"unit {}: this build {got:?} vs the installed wheel {want:?} — the fixture does not reproduce the shipped rival row",
			u.seed
		);
	}
	println!("[self-check] unit {} est-fed shipped row reproduces the installed wheel: stable {:.4} err {:.4}° steady {:.4}° pos {:.4} m", u.seed, got[0], got[1], got[2], got[3]);
}

/// Spec §10.2 + §10.3 in one pass (the shipped est-fed row is flown once and shared).
#[test]
#[ignore = "val-unit sweep: needs WNN_INDI_PROBE_FIXTURE; minutes of CPU"]
fn val_sweep()
{
	let units = units();
	self_check(&units);
	const GAINS: [f64; 7] = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5];
	assert_eq!(GAINS[2], crate::ctbr_mpcof::CTBR_MPC_L_GAIN, "the control arm must be the shipped gain");
	for (feed, use_est) in [("est-fed", true), ("oracle-fed", false)]
	{
		let arms: Vec<Arm> = GAINS
			.iter()
			.map(|&l| Arm {
				name: format!("l = {l} (τ = {:.0} ms){}", 1.0 / l, if l == GAINS[2] { " SHIPPED" } else { "" }),
				per_unit: fly(&units, use_est, &move || RivalProbe::measure_only().with_observer_gain(l)),
			})
			.collect();
		print_table(&format!("STAGE −1 — observer speed, MPCOF>rate(h10), {feed}, VAL (20 units × 100 episodes)"), &units, &arms, 2);
		if use_est
		{
			let arms = vec![
				Arm { name: "MPCOF>rate(h10) est-fed SHIPPED".into(), per_unit: fly(&units, true, &RivalProbe::measure_only) },
				Arm { name: "MPCOF>rate(h10) est-fed + bias compensation".into(), per_unit: fly(&units, true, &|| RivalProbe::measure_only().with_bias_comp()) },
				Arm { name: "MPCOF>rate(h10) oracle-fed (bound, not a rival)".into(), per_unit: fly(&units, false, &RivalProbe::measure_only) },
				// POST HOC (added after the pre-registered arms were read; spec §11.3).
				Arm { name: "POST HOC: est-fed + bias compensation + l = 0.05".into(), per_unit: fly(&units, true, &|| RivalProbe::measure_only().with_bias_comp().with_observer_gain(0.05)) },
			];
			print_table("F1 — bias-compensated MPCOF, VAL (20 units × 100 episodes)", &units, &arms, 0);
		}
	}
}

/// Spec §12 (N2, Luiz 14/11/2026): the ONE look at the 35 test units for the
/// bias-compensated MPCOF rival. Three arms, fixed; arm 1 must reproduce the published
/// row before anything else is read. Guarded by its own fixture variable so the val
/// sweep can never pick test units up.
#[test]
#[ignore = "REPORT-ONLY one-shot on the 35 test units (spec §12): needs WNN_INDI_TEST_CONFIRM_FIXTURE"]
fn test_confirmation_bias_compensated_mpcof()
{
	let units = units_from("WNN_INDI_TEST_CONFIRM_FIXTURE", true);
	let seeds: Vec<u64> = units.iter().map(|u| u.seed).collect();
	assert_eq!(seeds, (99990101..=99990135).collect::<Vec<u64>>(), "exactly the 35 published test units");
	self_check(&units);
	let shipped = Arm { name: "MPCOF>rate(h10) est-fed SHIPPED".into(), per_unit: fly(&units, true, &RivalProbe::measure_only) };
	let (err, steady, pos) = (
		mean(shipped.per_unit.iter().map(|(r, _)| r.1)),
		mean(shipped.per_unit.iter().map(|(r, _)| r.2)),
		mean(shipped.per_unit.iter().map(|(r, _)| r.4)),
	);
	let stable = mean(shipped.per_unit.iter().map(|(r, _)| r.0));
	// docs/ctrl48_fixed_scorer_table.md: 100.0 / 1.35 / 0.68 / 0.341 (0 fails).
	assert!(
		stable == 1.0 && (err - 1.35).abs() < 0.005 && (steady - 0.68).abs() < 0.005 && (pos - 0.341).abs() < 0.0005,
		"the shipped row does not reproduce the published rival: {stable} / {err:.4} / {steady:.4} / {pos:.4} — STOP"
	);
	let arms = vec![
		shipped,
		Arm { name: "MPCOF>rate(h10) est-fed + bias compensation".into(), per_unit: fly(&units, true, &|| RivalProbe::measure_only().with_bias_comp()) },
		Arm { name: "MPCOF>rate(h10) oracle-fed (bound, not a rival)".into(), per_unit: fly(&units, false, &RivalProbe::measure_only) },
	];
	print_table("N2 — bias-compensated MPCOF, TEST (35 units × 100 episodes), one shot", &units, &arms, 0);
}

/// docs/ctrl48_stress_test_plan.md, Result gap (Luiz 15/11/2026): the equal-information
/// MPCOF comparator (est-fed + bias compensation) on the SAME 300 test units × 100
/// episodes the stress test scored — REPORT-ONLY, one shot, nothing tuned. The fixture is
/// scripts/ctrl_indi_val_fixture.py --units test on a stress-test log (its --test-seeds
/// = 99990101..99990400). Arm 1 (no bias compensation) must reproduce the stress test's
/// MPCOF row (5/30000 fails, err 1.396°, steady 0.697°, pos 0.340 m) before anything else
/// is read.
#[test]
#[ignore = "REPORT-ONLY one-shot on the 300 stress-test units: needs WNN_STRESS_MPCOF_FIXTURE"]
fn stress_units_bias_compensated_mpcof()
{
	let units = units_from("WNN_STRESS_MPCOF_FIXTURE", true);
	let seeds: Vec<u64> = units.iter().map(|u| u.seed).collect();
	assert_eq!(seeds, (99990101..=99990400).collect::<Vec<u64>>(), "exactly the 300 stress-test units");
	assert!(units.iter().all(|u| u.episodes() == 100), "100 episodes per unit");
	self_check(&units);
	let shipped = Arm { name: "MPCOF>rate(h10) est-fed, no bias comp (stress-test row)".into(), per_unit: fly(&units, true, &RivalProbe::measure_only) };
	let fails = |a: &Arm| -> usize { a.per_unit.iter().zip(&units).map(|((r, _), u)| ((1.0 - r.0) * u.episodes() as f64).round() as usize).sum() };
	let m = |a: &Arm, f: &dyn Fn(&Row) -> f64| mean(a.per_unit.iter().map(|(r, _)| f(r)));
	let (f0, err, steady, pos) = (fails(&shipped), m(&shipped, &|r| r.1), m(&shipped, &|r| r.2), m(&shipped, &|r| r.4));
	println!("[sanity] no-bias-comp row: {f0}/30000 fails, err {err:.4}, steady {steady:.4}, pos {pos:.4} (stress test: 5/30000, 1.396, 0.697, 0.340)");
	// The stress-test means average per-unit rows PRINTED at 0.01° / 0.001 m, so the
	// aggregate gate allows that rounding; the per-unit rows ([unit] lines) are checked
	// against the stress logs at their printed precision outside the test.
	for ((r, _), u) in shipped.per_unit.iter().zip(&units)
	{
		println!("[unit] {} {:.1} {:.2} {:.2} {:.3}", u.seed, 100.0 * r.0, r.1, r.2, r.4);
	}
	assert!(
		f0 == 5 && (err - 1.396).abs() < 0.003 && (steady - 0.697).abs() < 0.003 && (pos - 0.340).abs() < 0.001,
		"the no-bias-comp row does not reproduce the stress test — the units or episodes differ: STOP"
	);
	let arms = vec![shipped, Arm { name: "MPCOF>rate(h10) est-fed + bias compensation (COMPARATOR)".into(), per_unit: fly(&units, true, &|| RivalProbe::measure_only().with_bias_comp()) }];
	print_table("Stress gap — equal-information MPCOF on the 300 stress-test TEST units × 100 episodes, one shot", &units, &arms, 0);
	print_position_table("Stress gap — position (hold = the last fifth, t >= 0.8·steps)", &units, &arms);
	for a in &arms
	{
		println!("[result] {}: fails {}/{} stable {:.4} err {:.4} steady {:.4} pos {:.4} pos_hold {:.4} dz_hold {:.4}",
			a.name, fails(a), units.len() * 100, 100.0 * m(a, &|r| r.0), m(a, &|r| r.1), m(a, &|r| r.2), m(a, &|r| r.4),
			mean(a.per_unit.iter().map(|(_, p)| p.pos_fifths[4])), mean(a.per_unit.iter().map(|(_, p)| p.hold_abs_xyz[2])));
	}
}

/// docs/ctrl48_block10_2x2_prereg.md, "L3 reference rows" (Luiz 15/11/2026): the three
/// MPCOF reference rows re-scored ONCE with the altitude integrator (pole3) on the 35
/// standard TEST units — REPORT-ONLY context for the L3 arms, nothing tuned. Gate: with
/// the integrator off, all three rows reproduce the published ones before pole3 is read.
#[test]
#[ignore = "REPORT-ONLY one-shot on the 35 test units: needs WNN_L3_REF_FIXTURE"]
fn test_units_reference_rows_alt_integral_pole3()
{
	let units = units_from("WNN_L3_REF_FIXTURE", true);
	let seeds: Vec<u64> = units.iter().map(|u| u.seed).collect();
	assert_eq!(seeds, (99990101..=99990135).collect::<Vec<u64>>(), "exactly the 35 published test units");
	self_check(&units);
	let comp = || RivalProbe::measure_only().with_bias_comp();
	let rows: [(&str, bool, &(dyn Fn() -> RivalProbe + Sync), [f64; 4]); 3] = [
		("MPCOF>rate(h10) oracle-fed", false, &RivalProbe::measure_only, [1.0, 0.690, 0.163, 0.357]),
		("MPCOF>rate(h10) est + bias comp (COMPARATOR)", true, &comp, [1.0, 0.691, 0.165, 0.357]),
		("MPCOF>rate(h10) est-fed, no bias comp", true, &RivalProbe::measure_only, [1.0, 1.352, 0.684, 0.341]),
	];
	let fails = |a: &Arm| -> usize { a.per_unit.iter().zip(&units).map(|((r, _), u)| ((1.0 - r.0) * u.episodes() as f64).round() as usize).sum() };
	let m = |a: &Arm, f: &dyn Fn(&Row) -> f64| mean(a.per_unit.iter().map(|(r, _)| f(r)));
	let mut arms = Vec::new();
	for (mode, label) in [(0u8, "off"), (2u8, "pole3")]
	{
		for (name, use_est, make, published) in rows.iter()
		{
			let a = Arm { name: format!("{name}, alt-integral {label}"), per_unit: fly_tweaked(&units, *use_est, Tweak { alt_integral: mode, ..Tweak::default() }, *make) };
			let got = [m(&a, &|r| r.0), m(&a, &|r| r.1), m(&a, &|r| r.2), m(&a, &|r| r.4)];
			if mode == 0
			{
				// docs/ctrl_rival_row_ruling.md rows C / B / A, published at 0.001° / 0.001 m.
				assert!(
					got[0] == published[0] && fails(&a) == 0 && (got[1] - published[1]).abs() < 0.0005 && (got[2] - published[2]).abs() < 0.0005 && (got[3] - published[3]).abs() < 0.0005,
					"{name}, off: {got:?} does not reproduce the published row {published:?} — STOP"
				);
			}
			println!("[result] {}: stable {:.2} err {:.3} steady {:.3} pos {:.3} pos_hold {:.3} dz_hold {:.3} fails {}/{}",
				a.name, 100.0 * got[0], got[1], got[2], got[3],
				mean(a.per_unit.iter().map(|(_, p)| p.pos_fifths[4])), mean(a.per_unit.iter().map(|(_, p)| p.hold_abs_xyz[2])),
				fails(&a), units.len() * 100);
			arms.push(a);
		}
	}
	print_position_table("L3 reference rows — MPCOF, TEST (35 units × 100 episodes), alt-integral off vs pole3, one shot", &units, &arms);
}

/// CTRL-69 (Luiz 22/11/2026): rows A / B / C of docs/ctrl_rival_row_ruling.md §0 RE-SCORED
/// on the installed lineage (ABI 61/62: every on-board attitude estimate is the f32
/// MahonyF32). REPORT-ONLY, the same 35 TEST units × 100 episodes, nothing tuned. There is
/// deliberately NO published-value gate: the estimator changed, so the est-fed rows are
/// expected to move. The gate is self_check (this build == the installed wheel's est-fed
/// row on unit 1). The ABI-60 rows are printed beside for the old-vs-new table.
#[test]
#[ignore = "REPORT-ONLY re-score on the 35 test units (ABI-61 f32 lineage): needs WNN_RIVAL_RESCORE_FIXTURE"]
fn test_units_rival_rows_rescore_f32_lineage()
{
	let units = units_from("WNN_RIVAL_RESCORE_FIXTURE", true);
	let seeds: Vec<u64> = units.iter().map(|u| u.seed).collect();
	assert_eq!(seeds, (99990101..=99990135).collect::<Vec<u64>>(), "exactly the 35 published test units");
	assert!(units.iter().all(|u| u.episodes() == 100), "100 episodes per unit");
	self_check(&units);
	let comp = || RivalProbe::measure_only().with_bias_comp();
	// (row, name, est-fed, probe, ABI-60 published: stable, err, steady, pos)
	let rows: [(&str, &str, bool, &(dyn Fn() -> RivalProbe + Sync), [f64; 4]); 3] = [
		("A", "MPCOF>rate(h10) est-fed, no bias comp", true, &RivalProbe::measure_only, [1.0, 1.352, 0.684, 0.341]),
		("B", "MPCOF>rate(h10) est-fed + bias comp (eq-info COMPARATOR)", true, &comp, [1.0, 0.691, 0.165, 0.357]),
		("C", "MPCOF>rate(h10) oracle-fed (bound, label law)", false, &RivalProbe::measure_only, [1.0, 0.690, 0.163, 0.357]),
	];
	let fails = |a: &Arm| -> usize { a.per_unit.iter().zip(&units).map(|((r, _), u)| ((1.0 - r.0) * u.episodes() as f64).round() as usize).sum() };
	let m = |a: &Arm, f: &dyn Fn(&Row) -> f64| mean(a.per_unit.iter().map(|(r, _)| f(r)));
	let mut arms = Vec::new();
	for (row, name, use_est, make, old) in rows.iter()
	{
		let a = Arm { name: format!("{row} {name}"), per_unit: fly(&units, *use_est, *make) };
		println!("[rescore] row {row} NEW(f32): stable {:.2} err {:.3} steady {:.3} pos {:.3} fails {}/{} pos_hold {:.3} | OLD(ABI 60): stable {:.2} err {:.3} steady {:.3} pos {:.3} | {name}",
			100.0 * m(&a, &|r| r.0), m(&a, &|r| r.1), m(&a, &|r| r.2), m(&a, &|r| r.4), fails(&a), units.len() * 100,
			mean(a.per_unit.iter().map(|(_, p)| p.pos_fifths[4])), 100.0 * old[0], old[1], old[2], old[3]);
		arms.push(a);
	}
	print_table("CTRL-69 — MPCOF rows A/B/C re-scored on the f32 (ABI 61/62) lineage, TEST (35 units × 100 episodes)", &units, &arms, 0);
}

/// The cascade student's output law as flown in block 8/9 ("both": --ctbr-delta-law
/// --delta-leak 0.7 --delta-max 0.6, ranges 0.25 / 4.2 / 4.2 / 2, γ 1 / 2 / 2 / 2).
fn both_alphabet(levels: u32) -> StudentAlphabet
{
	StudentAlphabet { leak: 0.7, delta_max: 0.6, range: [0.25, 4.2, 4.2, 2.0], gamma: [1.0, 2.0, 2.0, 2.0], levels }
}

/// docs/ctrl_gap_after_block9.md §B.2 (quantisation) and §C (position) — VAL units only.
#[test]
#[ignore = "val-unit sweep: needs WNN_INDI_PROBE_FIXTURE; minutes of CPU"]
fn val_gap_and_position()
{
	let units = units();
	self_check(&units);
	let comp = || RivalProbe::measure_only().with_bias_comp();
	// §B.2 — the label law through the student's alphabet and 100 Hz hold.
	let mut arms = vec![
		Arm { name: "label law, est + bias comp (student-observable attitude)".into(), per_unit: fly(&units, true, &comp) },
	];
	for levels in [128u32, 64, 32, 16]
	{
		arms.push(Arm {
			name: format!("  + student alphabet, {levels} cells/bank"),
			per_unit: fly(&units, true, &move || RivalProbe::measure_only().with_bias_comp().with_student_alphabet(both_alphabet(levels))),
		});
	}
	arms.push(Arm { name: "label law, oracle-fed".into(), per_unit: fly(&units, false, &RivalProbe::measure_only) });
	arms.push(Arm {
		name: "  + student alphabet, 64 cells/bank".into(),
		per_unit: fly(&units, false, &|| RivalProbe::measure_only().with_student_alphabet(both_alphabet(64))),
	});
	print_table("B.2 — label law through the student's output alphabet, MPCOF>rate(h10), VAL (20 units × 100 episodes)", &units, &arms, 0);

	// §C — position budget on the shipped task, then the task what-ifs.
	let mut pos = vec![
		Arm { name: "est-fed SHIPPED".into(), per_unit: fly(&units, true, &RivalProbe::measure_only) },
		Arm { name: "est-fed + bias comp".into(), per_unit: fly(&units, true, &comp) },
		Arm { name: "oracle-fed".into(), per_unit: fly(&units, false, &RivalProbe::measure_only) },
	];
	for wn in [1.5f64, 2.0, 3.0]
	{
		pos.push(Arm {
			name: format!("TASK CHANGE ω_n = {wn} rad/s, est + bias comp"),
			per_unit: fly_tweaked(&units, true, Tweak { pos_omega_n: Some(wn), ..Tweak::default() }, &comp),
		});
	}
	for steps in [4000usize, 8000]
	{
		pos.push(Arm {
			name: format!("TASK CHANGE {} s episodes, est + bias comp", steps / 1000),
			per_unit: fly_tweaked(&units, true, Tweak { steps: Some(steps), ..Tweak::default() }, &comp),
		});
	}
	print_position_table("C — position budget, MPCOF>rate(h10), VAL (20 units × 100 episodes); fifths are consecutive fifths of the episode", &units, &pos);
}

/// docs/ctrl48_block10_levers_spec.md — L3 pre-check, VAL units only: the label law
/// (MPCOF>rate, est + bias compensation and oracle-fed) and the firmware PID cascade
/// rival, altitude integrator off / pi / pole3.
#[test]
#[ignore = "val-unit sweep: needs WNN_INDI_PROBE_FIXTURE; minutes of CPU"]
fn val_alt_integral()
{
	let units = units();
	self_check(&units);
	let comp = || RivalProbe::measure_only().with_bias_comp();
	let mut arms = Vec::new();
	for (who, teacher, use_est, make) in [
		("MPCOF>rate(h10) est + bias comp", None, true, &comp as &(dyn Fn() -> RivalProbe + Sync)),
		("MPCOF>rate(h10) oracle-fed", None, false, &RivalProbe::measure_only),
		("PID>rate(h10) est-fed", Some(0u8), true, &RivalProbe::measure_only),
	]
	{
		for (mode, name) in [(0u8, "off"), (1, "pi"), (2, "pole3")]
		{
			arms.push(Arm {
				name: format!("{who}, alt-integral {name}"),
				per_unit: fly_tweaked(&units, use_est, Tweak { alt_integral: mode, teacher, ..Tweak::default() }, make),
			});
		}
	}
	print_position_table("L3 — altitude integrator, VAL (20 units × 100 episodes); fifths are consecutive fifths of the episode", &units, &arms);
}

/// ABI 53 per-axis split: |v| of Log(q_ref⁻¹ q) reproduces the scorer's own err and steady
/// (geodesic angle, f32 acos vs f64 atan2 ⇒ only rounding apart), the axes are consistent
/// with the norm (each ≤ it, their root-sum-square ≥ it per sample ⇒ sum of means ≥ norm),
/// and an oracle-fed row has zero fed-heading drift while an est-fed one does not.
#[test]
fn axis_split_decomposes_the_scorer()
{
	for teacher in [0u8, 4]
	{
		for use_est in [false, true]
		{
			let mut p = RivalProbe::measure_only();
			let row = small(teacher, use_est, Some(&mut p)).unwrap();
			let a = p.report().axes;
			let (err, steady) = (row.1.to_radians(), row.2.to_radians());
			// The scorer's 2·acos(|q·t|) is f32: near dot ≈ 1 one ulp (6e-8) moves the angle by
			// 2·6e-8/sin(θ/2) ≈ 3e-5 rad at θ ≈ 0.5° — the scorer's own resolution, not a split error.
			assert!((a.err_norm - err).abs() < 1e-4, "t{teacher} est{use_est}: |v| {} vs err {}", a.err_norm, err);
			assert!((a.hold_norm - steady).abs() < 1e-4, "t{teacher} est{use_est}: hold |v| {} vs steady {}", a.hold_norm, steady);
			for i in 0..3
			{
				assert!(a.err_abs[i] <= a.err_norm + 1e-12 && a.hold_abs[i] <= a.hold_norm + 1e-12, "{a:?}");
				assert!(a.hold_signed[i].abs() <= a.hold_abs[i] + 1e-12, "{a:?}");
			}
			assert!(a.err_abs.iter().sum::<f64>() + 1e-12 >= a.err_norm, "{a:?}");
			assert!(a.episodes_all > 0 && a.episodes_hold > 0 && a.episodes_hold <= a.episodes_all, "{a:?}");
			if use_est
			{
				assert!(a.hold_fed_yaw_abs > 0.0, "est-fed: the Mahony heading must drift: {a:?}");
			}
			else
			{
				assert_eq!(a.hold_fed_yaw_abs, 0.0, "oracle-fed: the fed heading IS the truth");
			}
		}
	}
}

/// rotvec_rel on known rotations: a pure yaw about body z, a roll about a tilted reference,
/// and the w<0 double cover.
#[test]
fn rotvec_rel_known_rotations()
{
	use crate::rival_probe::rotvec_rel;
	let half = |a: f64| ((a * 0.5).cos() as f32, (a * 0.5).sin() as f32);
	let (c, s) = half(0.3);
	let yaw = [c, 0.0, 0.0, s];
	let v = rotvec_rel([1.0, 0.0, 0.0, 0.0], yaw);
	assert!((v[0]).abs() < 1e-7 && (v[1]).abs() < 1e-7 && (v[2] - 0.3).abs() < 1e-6, "{v:?}");
	let neg = yaw.map(|x| -x);
	let w = rotvec_rel([1.0, 0.0, 0.0, 0.0], neg);
	assert!((w[2] - 0.3).abs() < 1e-6, "double cover: {w:?}");
	// reference rolled 0.2, body rolled 0.25 ⇒ +0.05 about body x.
	let (c1, s1) = half(0.2);
	let (c2, s2) = half(0.25);
	let r = rotvec_rel([c1, s1, 0.0, 0.0], [c2, s2, 0.0, 0.0]);
	assert!((r[0] - 0.05).abs() < 1e-6 && r[1].abs() < 1e-7 && r[2].abs() < 1e-7, "{r:?}");
}
