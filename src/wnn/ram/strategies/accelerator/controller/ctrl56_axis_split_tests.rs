//! ABI 56: the OPT-IN per-axis split on the WNN CPU scorer (score_controllers_cpu(axis_out=),
//! rollout_one(axes=)), sharing axis_split::AxisAcc with the ABI-53 rival probe.
//!
//! Pinned: (1) measurement only — the scorer row is bit-identical with and without the
//! accumulator, on the attitude-only path, the stage-2 tilt-reference path and stage 2 with
//! the L3 pole3 altitude integrator; (2) the split DECOMPOSES the row — |v| reproduces err
//! (row[1]) and steady (row[5], HoldPolicy::FallbackWhole: an episode that never reached
//! the hold window contributes its whole-episode mean), including on diverging episodes;
//! (3) the heading-drift slot is the student's anchored heading drift (NaN unanchored).

use crate::abi38_pins::{ics, stage2, ARM, ASYM, DT, EPS, G, INERTIA, KD, KT, MASS};
use crate::axis_split::{AxisAcc, AxisSplit, HoldPolicy};
use crate::controller::WnnController;
use crate::ctrl51_yaw_trim_tests::{yaw_student, YawOpt};
use crate::stage1::Stage1Cfg;
use ram_core::neuron_memory::BINARY;

fn rows(c: &WnnController, steps: usize, s1: Option<&Stage1Cfg>, axes: Option<&mut AxisAcc>) -> [f64; crate::cpu_score::ROW_METRICS]
{
	let (q0, w0) = ics();
	let mut cc = c.clone();
	crate::cpu_score::rollout_one(
		&mut cc, &q0, &w0, EPS, steps, DT, ARM, KT, KD, INERTIA, G, 0.2, [0.0; 3],
		true, [0.0; 3], 0.0, 0.1, ASYM, 0.0107, 0.0, 0.0, 777,
		0.0, 0, 0, 0.2, c.levels_per_motor(), 4, None, None, None, 1.0, 0.4, None, s1, axes, None,
	)
}

fn pole3() -> Stage1Cfg
{
	let mut s = stage2();
	s.alt_int = crate::altitude_integrator::AltitudeIntegrator::from_plant(2, MASS as f64, G as f64, KT as f64, 2.0, 1.0, 0.25, DT)
		.expect("pole3 integrator");
	s
}

fn trained_student() -> WnnController
{
	let mut c = yaw_student(0, 10, 0x5601, YawOpt::Off, BINARY);
	c.enable_att_estimator(2.0, 0.1).unwrap();
	let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &crate::abi38_tests::l4c_cfg(0.20, 4), [0.0; 3], 23, false);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "vacuous fixture");
	c
}

fn split(c: &WnnController, steps: usize, s1: Option<&Stage1Cfg>) -> ([f64; crate::cpu_score::ROW_METRICS], AxisSplit)
{
	let mut acc = AxisAcc::default();
	let r = rows(c, steps, s1, Some(&mut acc));
	(r, acc.split(HoldPolicy::FallbackWhole))
}

fn bits(r: &[f64; crate::cpu_score::ROW_METRICS]) -> Vec<u64>
{
	r.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn axes_off_is_bit_identical()
{
	let c = trained_student();
	let (s2, p3) = (stage2(), pole3());
	for (name, s1) in [("attitude-only", None), ("stage-2 tilt ref", Some(&s2)), ("stage-2 + L3 pole3", Some(&p3))]
	{
		let off = rows(&c, 600, s1, None);
		let (on, _) = split(&c, 600, s1);
		assert_eq!(bits(&off), bits(&on), "{name}: the axis accumulator moved the scorer row");
	}
	// The L3 case with a student that READS alt_err (the yaw fixture does not, so for it
	// pole3 is inert): the integrator is in the loop and the split is still inert.
	let v = crate::ctrl51_alt_integral_tests::alt_student_cells(0x5604, 1 << 8);
	let n = 200; // short horizon: the every-address random policy is chaotic beyond it
	let (off, on) = (rows(&v, n, Some(&p3), None), split(&v, n, Some(&p3)).0);
	assert_eq!(bits(&off), bits(&on), "L3 pole3 + alt_err student: the accumulator moved the row");
	assert_ne!(bits(&rows(&v, n, Some(&s2), None)), bits(&off), "pole3 did not change the rollout (vacuous)");
	let (r, a) = split(&v, n, Some(&p3));
	assert!((a.err_norm - r[1]).abs() < 1e-4 && (a.hold_norm - r[5]).abs() < 1e-4, "L3: {a:?} vs err {} steady {}", r[1], r[5]);
}

#[test]
fn axes_reproduce_err_and_steady()
{
	let c = trained_student();
	let (s2, p3) = (stage2(), pole3());
	for (name, s1) in [("attitude-only", None), ("stage-2 tilt ref", Some(&s2)), ("stage-2 + L3 pole3", Some(&p3))]
	{
		let (r, a) = split(&c, 600, s1);
		eprintln!("[axis56] {name}: err {:.6} vs |v| {:.6} | steady {:.6} vs {:.6} | axes err {:?} hold {:?} drift {:.6}",
		          r[1], a.err_norm, r[5], a.hold_norm, a.err_abs, a.hold_abs, a.hold_fed_yaw_abs);
		// f32 acos resolution of the scorer's angle (rival_probe_tests): ≤ ~3e-5 rad per sample.
		assert!((a.err_norm - r[1]).abs() < 1e-4, "{name}: |v| {} vs err {}", a.err_norm, r[1]);
		assert!((a.hold_norm - r[5]).abs() < 1e-4, "{name}: hold |v| {} vs steady {}", a.hold_norm, r[5]);
		for i in 0..3
		{
			assert!(a.err_abs[i] <= a.err_norm + 1e-12 && a.hold_abs[i] <= a.hold_norm + 1e-12, "{name}: {a:?}");
		}
		assert!(a.err_abs.iter().sum::<f64>() + 1e-12 >= a.err_norm, "{name}: {a:?}");
		assert_eq!(a.episodes_all, EPS);
		assert!(a.hold_fed_yaw_abs.is_finite() && a.hold_fed_yaw_abs > 0.0, "{name}: anchored heading drift {}", a.hold_fed_yaw_abs);
	}
}

/// HoldPolicy accounting on episodes that never reach the hold window (pure: the scorer
/// fixtures here do not diverge early). FallbackWhole = the WNN scorer's else-branch: the
/// episode's whole mean enters steady and the divisor is every episode; ReachedOnly = the
/// rival scorer's: the episode is left out of steady. err is the same under both.
#[test]
fn axes_hold_policies_on_unreached_episodes()
{
	use crate::axis_split::rotvec_rel;
	let (h1, h2) = ((0.01f32).sin(), (0.03f32).sin()); // ~0.02 / 0.06 rad roll
	let q1 = [(0.01f32).cos(), h1, 0.0, 0.0];
	let q2 = [(0.03f32).cos(), h2, 0.0, 0.0];
	let (v1, v2) = (rotvec_rel([1.0, 0.0, 0.0, 0.0], q1)[0], rotvec_rel([1.0, 0.0, 0.0, 0.0], q2)[0]);
	let fly = |pol: HoldPolicy| {
		let mut a = AxisAcc::default();
		// episode 0: 10 steps, the last 2 in the hold window at q2.
		a.begin_episode();
		for t in 0..10
		{
			a.record(t >= 8, None, if t >= 8 { q2 } else { q1 });
		}
		a.end_episode(pol);
		// episode 1: diverged after 3 steps — never reached the window.
		a.begin_episode();
		for _ in 0..3
		{
			a.record(false, None, q1);
		}
		a.end_episode(pol);
		a.split(pol)
	};
	let err = ((8.0 * v1 + 2.0 * v2) / 10.0 + v1) / 2.0;
	let (fw, ro) = (fly(HoldPolicy::FallbackWhole), fly(HoldPolicy::ReachedOnly));
	assert!((fw.err_norm - err).abs() < 1e-12 && (ro.err_norm - err).abs() < 1e-12);
	assert!((fw.hold_norm - (v2 + v1) / 2.0).abs() < 1e-12, "FallbackWhole steady {}", fw.hold_norm);
	assert!((ro.hold_norm - v2).abs() < 1e-12, "ReachedOnly steady {}", ro.hold_norm);
	assert_eq!((fw.episodes_all, fw.episodes_hold), (2, 1));
}

#[test]
fn axes_heading_slot_nan_without_anchored_yaw()
{
	let c = crate::abi32_tests::controller(7, true);
	assert!(!c.yaw_anchored());
	let (_, a) = split(&c, 300, Some(&stage2()));
	assert!(a.hold_fed_yaw_abs.is_nan(), "unanchored heading is not an angle: {}", a.hold_fed_yaw_abs);
}
