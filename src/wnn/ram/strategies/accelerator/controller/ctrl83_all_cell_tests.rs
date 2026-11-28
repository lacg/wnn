//! CTRL-83 cell `all` (ABI 69, Luiz 28/11): the single joint-OOD headline. Every controller
//! flies the SAME per-episode plant lag T_e ~ U(0.15, 0.30) s while its model stays at the
//! nominal 0.20 — the WNN scorers since ABI 68, score_classical_baseline from ABI 69.
//!
//!   * flag off — the rival with no range is the ABI-68 row: every existing pin passes None,
//!     and a degenerate range [0.20, 0.20] is that row bit for bit;
//!   * semantics — a point range [T, T] on a 0.20 plant IS the fixed T plant with the model
//!     held at 0.20 (D1), bit for bit, on the rival AND on the WNN CPU scorer: the two
//!     scorers read the per-episode lag the same way;
//!   * CPU == GPU on the WNN under the joint draw (per-episode lag + nominal design model);
//!   * the spread moves every rival row, and the rival refuses the hosts the WNN refuses.

use crate::abi38_pins::EPS;
use crate::actuator_spec::ActuatorSpec;
use crate::ctrl86_flag_off_pins::{cpu_rows_lag, gpu_rows_lag, l2b_cfg, real_life_plant, rival_kw_lag, stage2_with, trained_on, Spelling, LAG};
use crate::design_model::DesignModel;
use crate::payload_step::PayloadStep;
use crate::stage1::Stage1Cfg;
use crate::stress_opts::{StressKwargs, StressOpts};
use crate::train_lag_range::TrainLagRange;

const LO: f32 = 0.15;
const HI: f32 = 0.30;
const STEPS: usize = 500;
/// The fixture's weather seed (ctrl86_flag_off_pins cpu_rows_lag / gpu_rows_lag / rival_kw).
const DIST_SEED: u64 = 777;

type Row8 = (f64, f64, f64, f64, f64, f64, f64, f64);

fn rival(teacher: u8, lag: f32, kw: StressKwargs, range: Option<(f32, f32)>) -> Row8
{
	rival_kw_lag(teacher, &stage2_with(real_life_plant(Spelling::Legacy)), lag, kw, range).unwrap_or_else(|e| {
		pyo3::prepare_freethreaded_python();
		panic!("rival {teacher}: {e}")
	})
}

fn bits(r: Row8) -> String
{
	format!("{r:?}")
}

/// The stage-2 real-life episodes with per-episode lags `lags` and stress `stress`.
fn episodes(lags: Vec<f32>, stress: StressOpts) -> Stage1Cfg
{
	let mut s1 = stage2_with(real_life_plant(Spelling::Legacy));
	s1.ep_motor_lag = lags;
	s1.stress = stress;
	s1.validate(EPS).expect("fixture");
	s1
}

fn held_at_nominal(plant_lag: f32) -> StressOpts
{
	StressOpts { design: DesignModel::new(Some(LAG), ActuatorSpec::symmetric(plant_lag)).unwrap(), payload: PayloadStep::OFF }
}

fn nominal_kw() -> StressKwargs
{
	StressKwargs { model_motor_lag_s: Some(LAG), ..StressKwargs::OFF }
}

#[test]
fn ctrl83_all_rival_degenerate_range_is_the_fixed_lag()
{
	for teacher in [0u8, 3, 4]
	{
		let off = rival(teacher, LAG, StressKwargs::OFF, None);
		let deg = rival(teacher, LAG, StressKwargs::OFF, Some((LAG, LAG)));
		assert_eq!(bits(off), bits(deg), "teacher {teacher}: [0.20, 0.20] moved the in-range row");
		// The cell also spells the D1 hold explicitly (model_motor_lag_s 0.20): still the same row.
		assert_eq!(bits(off), bits(rival(teacher, LAG, nominal_kw(), Some((LAG, LAG)))), "teacher {teacher}: D1 spelled");
	}
}

/// A point range [T, T] on the nominal plant == the fixed-T plant with the model held at the
/// nominal 0.20 — on the rival and on the WNN CPU scorer, for T on both sides of nominal.
#[test]
fn ctrl83_all_point_range_is_the_stressed_plant_with_the_nominal_model()
{
	for t in [0.30f32, 0.15]
	{
		for teacher in [0u8, 3, 4]
		{
			let point = rival(teacher, LAG, nominal_kw(), Some((t, t)));
			let fixed = rival(teacher, t, nominal_kw(), None);
			assert_eq!(bits(point), bits(fixed), "rival {teacher}: [{t}, {t}] on a 0.20 plant != the fixed {t} plant, model 0.20");
		}
	}
	let (cascade, _) = trained_on(&l2b_cfg(4, Spelling::Legacy));
	for t in [0.30f32, 0.15]
	{
		let point = cpu_rows_lag(&mut cascade.clone(), &episodes(vec![t; EPS], held_at_nominal(LAG)), STEPS, LAG);
		let fixed = cpu_rows_lag(&mut cascade.clone(), &episodes(vec![], held_at_nominal(t)), STEPS, t);
		assert_eq!(format!("{point:?}"), format!("{fixed:?}"), "WNN CPU: [{t}, {t}] != the fixed {t} plant with the twin at 0.20");
	}
}

/// The joint draw on the WNN: CPU == GPU (the ctrl83 / ctrl87 tolerance policy), and it moves the row.
#[test]
fn cpu_fallback_matches_gpu_ctrl83_all_joint_lag()
{
	let (cascade, _) = trained_on(&l2b_cfg(4, Spelling::Legacy));
	let lags = TrainLagRange::new(LO, HI).unwrap().per_episode(DIST_SEED, EPS);
	let base = episodes(vec![], held_at_nominal(LAG));
	let joint = episodes(lags, held_at_nominal(LAG));
	let cr0 = cpu_rows_lag(&mut cascade.clone(), &base, STEPS, LAG);
	let cr = cpu_rows_lag(&mut cascade.clone(), &joint, STEPS, LAG);
	assert!(cr0.iter().zip(cr.iter()).any(|(a, b)| (a - b).abs() > 1e-9), "the per-episode lag did not move the WNN row");
	if let Some(gr) = gpu_rows_lag(&cascade, &joint, STEPS, LAG)
	{
		let close = |g: f64, c: f64, rel: f64, floor: f64, what: &str| {
			let tol = floor.max(c.abs() * rel);
			assert!((g - c).abs() <= tol, "{what}: gpu={g} cpu={c} tol={tol}");
		};
		close(gr[1], cr[1], 2e-2, 1e-4, "err");
		close(gr[0], cr[0], 5e-2, 1e-3, "reward");
		for (i, name) in [(13, "pos"), (14, "alt"), (15, "dx"), (16, "dy")]
		{
			close(gr[i], cr[i], 5e-2, 1e-3, name);
		}
		close(gr[19], cr[19], 2e-2, 1e-4, "mid");
		eprintln!("[ctrl83-all-parity] cpu err {:.6} gpu err {:.6} | cpu alt {:.5} gpu alt {:.5}", cr[1], gr[1], cr[14], gr[14]);
	}
}

#[test]
fn ctrl83_all_spread_moves_every_rival_and_refuses_bad_hosts()
{
	for teacher in [0u8, 3, 4]
	{
		let off = rival(teacher, LAG, nominal_kw(), None);
		let on = rival(teacher, LAG, nominal_kw(), Some((LO, HI)));
		assert_ne!(bits(off), bits(on), "rival {teacher} never saw the per-episode lag");
	}
	let s1 = stage2_with(real_life_plant(Spelling::Legacy));
	assert!(rival_kw_lag(4, &s1, LAG, StressKwargs::OFF, Some((HI, LO))).is_err(), "lo > hi");
	assert!(rival_kw_lag(4, &s1, 0.0, StressKwargs::OFF, Some((LO, HI))).is_err(), "lag off: no nominal design lag");
}
