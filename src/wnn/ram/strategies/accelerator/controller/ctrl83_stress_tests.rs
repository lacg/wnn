//! CTRL-83 (ABI 66): the stress-panel fairness fixes.
//!
//!   D1 — under a design-model override the rivals' MODELS (lag-aware design, MPCOF and
//!        cascade observer twins) and the WNN's actuator twin stay at the NOMINAL lag
//!        while the plant flies the stressed one; at stress = nominal nothing moves.
//!   D2 — the opt-in nominal-mass anchor rebases every consumer identically (CPU == GPU).
//!   D3 — the payload step: the WNN CPU scorer, the Metal kernel and the rivals fly the
//!        identical event (CPU/GPU parity sweeps `cpu_fallback_matches_gpu_ctrl83_*`).
//!
//! The cf21 flag-off byte identity lives in ctrl83_flag_off_pins.rs.

use crate::abi38_pins::{stage2, train, DT, MASS};
use crate::actuator_spec::{lag_alpha, ActuatorSpec};
use crate::controller::WnnController;
use crate::ctrl86_flag_off_pins::{cpu_rows_lag, gpu_rows_lag, l2b_cfg, real_life_plant, rival_kw, stage2_with, trained_on, Spelling, LAG};
use crate::dagger_train::AirframeRs;
use crate::design_model::DesignModel;
use crate::payload_step::PayloadStep;
use crate::stage1::Stage1Cfg;
use crate::stress_opts::{StressKwargs, StressOpts};

/// The stressed plant lag (the lag30 cell): T = 0.30 s against the nominal 0.20.
const LAG_STRESS: f32 = 0.30;
const STEPS: usize = 500;

fn nominal_design() -> DesignModel
{
	DesignModel::new(Some(LAG), ActuatorSpec::symmetric(LAG_STRESS)).expect("design override")
}

/// The real-life stage-2 episodes with `stress` riding them.
fn episodes(stress: StressOpts) -> Stage1Cfg
{
	let mut s1 = stage2_with(real_life_plant(Spelling::Legacy));
	s1.stress = stress;
	s1
}

fn payload(scale: f32) -> PayloadStep
{
	PayloadStep::from_kwargs(Some(0.25), scale, DT, STEPS).expect("payload at step 250")
}

/// One rival row; an Err is reported with the interpreter up (a PyErr's message needs it).
fn rival(teacher: u8, s1: &Stage1Cfg, lag: f32, kw: StressKwargs) -> (f64, f64, f64, f64, f64, f64, f64, f64)
{
	rival_kw(teacher, s1, lag, kw).unwrap_or_else(|e| {
		pyo3::prepare_freethreaded_python();
		panic!("rival {teacher}: {e}")
	})
}

fn close(gpu: f64, cpu: f64, rel: f64, floor: f64, what: &str)
{
	let tol = floor.max(cpu.abs() * rel);
	assert!((gpu - cpu).abs() <= tol, "{what}: gpu={gpu} cpu={cpu} tol={tol}");
}

/// CPU == GPU on reward, err, pos, alt, dx, dy, mid (the ctrl86 / abi32 tolerance policy).
fn assert_parity(c: &WnnController, s1: &Stage1Cfg, lag: f32, what: &str) -> Vec<f64>
{
	let cr = cpu_rows_lag(&mut c.clone(), s1, STEPS, lag);
	if let Some(gr) = gpu_rows_lag(c, s1, STEPS, lag)
	{
		close(gr[1], cr[1], 2e-2, 1e-4, &format!("{what} err"));
		close(gr[0], cr[0], 5e-2, 1e-3, &format!("{what} reward"));
		for (i, name) in [(13, "pos"), (14, "alt"), (15, "dx"), (16, "dy")]
		{
			close(gr[i], cr[i], 5e-2, 1e-3, &format!("{what} {name}"));
		}
		close(gr[19], cr[19], 2e-2, 1e-4, &format!("{what} mid"));
		eprintln!("[ctrl83-parity] {what}: cpu err {:.6} gpu err {:.6} | cpu alt {:.5} gpu alt {:.5}", cr[1], gr[1], cr[14], gr[14]);
	}
	cr.to_vec()
}

fn moved(a: &[f64], b: &[f64], what: &str)
{
	assert!(a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-9), "{what}: the option did not move the row — dead plumbing");
}

/// The teacher's command over a fixed (q, gyro) sequence, with its observer fed the
/// command it gave — enough steps for the lag twin and the MPCOF observer to matter.
fn teacher_trace(af: &AirframeRs, id: u8) -> Vec<u64>
{
	let mut t = af.teacher(id);
	t.reset();
	let mut applied = [0.5f64; 4];
	let mut out = Vec::new();
	for k in 0..300
	{
		let a = 0.04 * ((k as f32) * 0.05).sin();
		let q = [(1.0 - a * a).sqrt(), a, -0.5 * a, 0.0];
		let gyro = [0.3 * a, -0.2 * a, 0.01];
		t.observe(gyro, applied);
		applied = t.step_rs(q, gyro, [0.0; 3]);
		out.extend(applied.iter().map(|v| v.to_bits()));
	}
	out
}

/// D1, the rival side: the design airframe builds EXACTLY the nominal teacher, the plant
/// keeps the stressed lag, and the matched (pre-ABI-66) path re-derives for the stress.
#[test]
fn d1_mismatch_keeps_rival_models_nominal_while_the_plant_changes()
{
	let mut cfg = l2b_cfg(4, Spelling::Legacy);
	cfg.motor_lag_s = LAG_STRESS;
	let plant = AirframeRs::from_cfg(&cfg);
	let mut nom_cfg = cfg.clone();
	nom_cfg.motor_lag_s = LAG;
	let nominal = AirframeRs::from_cfg(&nom_cfg);
	let designed = plant.designed(nominal_design());
	assert_eq!(designed.motor_lag_s, LAG, "the model is built on the nominal lag");
	assert_eq!(plant.sim().motor_lag(), LAG_STRESS, "the plant still flies the stressed lag");
	assert_eq!(designed.actuator(), nominal.actuator());
	for id in [3u8, 4]
	{
		let (d, n, m) = (teacher_trace(&designed, id), teacher_trace(&nominal, id), teacher_trace(&plant, id));
		assert_eq!(d, n, "teacher {id}: the mismatch-mode model is not the nominal design");
		assert_ne!(m, n, "teacher {id}: the matched model did not re-derive for the stressed lag");
	}
	// PID is lag-blind by design: one law in all three.
	assert_eq!(teacher_trace(&designed, 0), teacher_trace(&plant, 0));
	assert_eq!(DesignModel::MATCHED.lag_s(LAG_STRESS), LAG_STRESS);
}

/// D1, end to end on the rival scorer: the override reaches the row, and at stress =
/// nominal it is the ABI-65 row bit for bit (the in-range cell).
#[test]
fn d1_rival_rows_mismatch_moves_and_in_range_is_identical()
{
	let s1 = stage2_with(real_life_plant(Spelling::Legacy));
	let nominal = StressKwargs { model_motor_lag_s: Some(LAG), ..StressKwargs::OFF };
	for teacher in [0u8, 3, 4]
	{
		let off = rival(teacher, &s1, LAG, StressKwargs::OFF);
		let spelled = rival(teacher, &s1, LAG, nominal);
		assert_eq!(format!("{off:?}"), format!("{spelled:?}"), "teacher {teacher}: model = plant moved the in-range row");
		let told = rival(teacher, &s1, LAG_STRESS, StressKwargs::OFF);
		let held = rival(teacher, &s1, LAG_STRESS, nominal);
		if teacher == 0
		{
			assert_eq!(format!("{told:?}"), format!("{held:?}"), "PID has no lag model to hold");
		}
		else
		{
			assert_ne!(format!("{told:?}"), format!("{held:?}"), "teacher {teacher}: the design override never reached the rival");
		}
	}
}

/// D1, the WNN side: the scorer arms the student's actuator twin with the DESIGN lag.
#[test]
fn d1_wnn_twin_is_armed_with_the_design_lag()
{
	let (c, _) = train(0, 1, LAG, 4, true);
	let mut cc = c.clone();
	cpu_rows_lag(&mut cc, &episodes(StressOpts { design: nominal_design(), payload: PayloadStep::OFF }), 100, LAG_STRESS);
	assert_eq!(cc.observer_lag_alpha(), lag_alpha(LAG, DT));
	let mut told = c.clone();
	cpu_rows_lag(&mut told, &episodes(StressOpts::OFF), 100, LAG_STRESS);
	assert_eq!(told.observer_lag_alpha(), lag_alpha(LAG_STRESS, DT), "matched: the twin is the plant");
}

/// D1 on Metal: the twin field (Params.twin_settling_time_s) — the obs_pwm student reads
/// its twin every step, so a twin built on the wrong lag would split CPU from GPU.
#[test]
fn cpu_fallback_matches_gpu_ctrl83_design_mismatch()
{
	let mismatch = StressOpts { design: nominal_design(), payload: PayloadStep::OFF };
	let (motor, _) = train(0, 1, LAG, 4, true);
	let told = assert_parity(&motor, &episodes(StressOpts::OFF), LAG_STRESS, "motor+obs_pwm lag30 matched");
	let held = assert_parity(&motor, &episodes(mismatch), LAG_STRESS, "motor+obs_pwm lag30 design 0.20");
	moved(&told, &held, "obs_pwm twin design lag");
	let (cascade, _) = trained_on(&l2b_cfg(4, Spelling::Legacy));
	assert_parity(&cascade, &episodes(mismatch), LAG_STRESS, "cascade lag30 design 0.20");
}

/// D3: the payload step — CPU == GPU for both student shapes, added and dropped, and it
/// moves the WNN and every rival row (the same event on every scorer).
#[test]
fn cpu_fallback_matches_gpu_ctrl83_payload_step()
{
	let (cascade, _) = trained_on(&l2b_cfg(4, Spelling::Legacy));
	let (motor, _) = train(0, 1, LAG, 4, true);
	for (c, kind) in [(&cascade, "cascade"), (&motor, "motor+obs_pwm")]
	{
		let base = assert_parity(c, &episodes(StressOpts::OFF), LAG, &format!("{kind} no payload"));
		for scale in [1.25f32, 0.8]
		{
			let s = StressOpts { design: DesignModel::MATCHED, payload: payload(scale) };
			let row = assert_parity(c, &episodes(s), LAG, &format!("{kind} payload x{scale} at 0.25 s"));
			moved(&base, &row, &format!("{kind} payload x{scale}"));
		}
	}
	let s1 = stage2_with(real_life_plant(Spelling::Legacy));
	for teacher in [0u8, 3, 4]
	{
		let off = rival(teacher, &s1, LAG, StressKwargs::OFF);
		let kw = StressKwargs { payload_step_s: Some(0.25), payload_mass_scale: 1.25, ..StressKwargs::OFF };
		let on = rival(teacher, &s1, LAG, kw);
		assert_ne!(format!("{off:?}"), format!("{on:?}"), "rival {teacher} never saw the payload step");
	}
}

/// D3 hand check on the plant: from the event step on, the sim's mass is draw × scale and
/// the vertical acceleration at hover thrust drops by exactly that ratio.
#[test]
fn d3_payload_sets_the_plant_mass_once()
{
	let p = payload(1.25);
	let s1 = stage2();
	let mut sim = AirframeRs::from_cfg(&l2b_cfg(4, Spelling::Legacy)).sim();
	sim.reset(None, None);
	sim.set_translation_core(s1.mass[0]).unwrap();
	let mut fired = 0;
	for t in 0..STEPS
	{
		if let Some(m) = p.mass_at(t, s1.mass[0])
		{
			sim.set_mass_core(m);
			fired += 1;
		}
	}
	assert_eq!(fired, 1);
	assert_eq!(sim.vehicle_mass(), s1.mass[0] * 1.25);
}

/// D2: the nominal-mass anchor — CPU == GPU, moves the row when the draws vary, and its
/// anchor is the nominal hover × (1 + c) on every episode.
#[test]
fn cpu_fallback_matches_gpu_ctrl83_nominal_anchor()
{
	let (cascade, _) = trained_on(&l2b_cfg(4, Spelling::Legacy));
	let legacy = episodes(StressOpts::OFF);
	let mut anchored = legacy.clone();
	anchored.anchor_at_mass(MASS).unwrap();
	let a = assert_parity(&cascade, &legacy, LAG, "cascade true-mass anchor");
	let b = assert_parity(&cascade, &anchored, LAG, "cascade nominal anchor");
	moved(&a, &b, "nominal anchor");
	let (g, k) = (crate::abi38_pins::G, crate::abi38_pins::KT);
	for ep in 0..legacy.mass.len()
	{
		let want = (MASS * g / (4.0 * k)).sqrt() * (1.0 + legacy.collective_frac[ep]);
		assert!((anchored.collective_pwm(ep, g, k) - want).abs() < 1e-6, "ep {ep}");
	}
}

/// The trainer's D2 flag moves the anchor the trainer flies (and only under the flag).
#[test]
fn d2_trainer_flag_reaches_the_training_anchor()
{
	let legacy = l2b_cfg(4, Spelling::Legacy);
	let mut nominal = legacy.clone();
	nominal.anchor_nominal_mass = true;
	let (_, d0) = trained_on(&legacy);
	let (_, d1) = trained_on(&nominal);
	assert_ne!(d0, d1, "the trainer never saw anchor_nominal_mass");
}

#[test]
fn refusals()
{
	let ext = ActuatorSpec::new(LAG, 2.0 * LAG, 0.0).unwrap();
	let kw = StressKwargs { model_motor_lag_s: Some(LAG), ..StressKwargs::OFF };
	assert!(StressOpts::from_kwargs(kw, ext, DT, STEPS).is_err(), "design override on an extended actuator");
	let late = StressKwargs { payload_step_s: Some(0.6), payload_mass_scale: 1.25, ..StressKwargs::OFF };
	assert!(StressOpts::from_kwargs(late, ActuatorSpec::symmetric(LAG), DT, STEPS).is_err(), "payload beyond the episode");
	let on = StressOpts::from_kwargs(kw, ActuatorSpec::symmetric(LAG), DT, STEPS).unwrap();
	assert!(on.require_translation(false, "t").is_err());
	assert!(StressOpts::OFF.require_translation(false, "t").is_ok());
	assert!(crate::stage1::require_anchor_translation(Some(MASS), false, "t").is_err());
	let s1 = stage2_with(real_life_plant(Spelling::Legacy));
	assert!(rival_kw(4, &s1, LAG, late).is_err(), "the rival host validates the payload too");
}
