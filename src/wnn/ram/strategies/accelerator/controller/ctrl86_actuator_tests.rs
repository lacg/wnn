//! CTRL-86 S1 + S2 (ABI 65): the extended plant actuator — asymmetric motor lag and the
//! armed-idle floor. Hand-computed plant steps, the CPU/GPU parity sweeps `asym_lag`,
//! `pwm_floor` and both together (a cascade student and a motor-path obs_pwm student, so
//! the plant filter, the actuator twin and the cascade d̂ twin are all exercised), the
//! reach checks (trainer, rivals, scorer move), and the refusals. The cf21 flag-off
//! identity lives in ctrl86_flag_off_pins.rs.

use crate::abi38_pins::train;
use crate::actuator_spec::ActuatorSpec;
use crate::controller::{AttitudeSim, WnnController};
use crate::ctrl86_flag_off_pins::{cpu_rows_on, gpu_rows_on, l2b_cfg, real_life_plant, rival_on, stage2_with, trained_on, Spelling, LAG};
use crate::plant_opts::PlantOpts;

/// T_down for the sweeps: twice T_up, the x500 ratio (gz timeConstantDown / Up = 2).
const LAG_DOWN: f32 = 2.0 * LAG;
/// A floor that binds on cf21 (hover ≈ 0.69): the x500 0.15 would rarely engage here.
const FLOOR: f32 = 0.62;

fn x500_sim() -> AttitudeSim
{
	AttitudeSim::new(0.001, 0.348, 8.54858, 0.016, [0.023839, 0.023942, 0.044], 9.81)
}

fn plant(down: f32, floor: f32) -> PlantOpts
{
	real_life_plant(Spelling::Legacy).with_actuator(LAG, down, floor).expect("extended actuator")
}

/// The plan's S1 hand step on the PLANT: seeded at 0.5, cmd 1.0 for 10 steps at dt 1 ms
/// with T_up 0.05 gives 0.5 + 0.5·(1 − 0.92^10); then cmd 0 moves by the down coefficient
/// 0.04 (T_down 0.10). With down == up the same command reverses at 0.08.
#[test]
fn plant_hand_computed_up_then_down()
{
	let mut sim = x500_sim();
	sim.set_motor_lag(0.05);
	sim.set_actuator_extras_core(0.10, 0.0);
	assert_eq!(sim.actuate_for_test([0.5; 4]), [0.5; 4], "first command seeds the filter");
	let mut f = [0.0f32; 4];
	for _ in 0..10
	{
		f = sim.actuate_for_test([1.0; 4]);
	}
	let want = 0.5 + 0.5 * (1.0 - 0.92f64.powi(10));
	assert!((f[0] as f64 - want).abs() < 1e-6, "up: {} vs {want}", f[0]);
	let g = sim.actuate_for_test([0.0; 4]);
	assert!((g[0] as f64 - f[0] as f64 * 0.96).abs() < 1e-6, "down coefficient 0.04: {} from {}", g[0], f[0]);
	// Symmetric twin: the same command reverses at the UP coefficient.
	let mut sym = x500_sim();
	sym.set_motor_lag(0.05);
	sym.actuate_for_test([0.5; 4]);
	let mut h = [0.0f32; 4];
	for _ in 0..10
	{
		h = sym.actuate_for_test([1.0; 4]);
	}
	assert_eq!(h, f, "spin-up is unchanged by T_down");
	let k = sym.actuate_for_test([0.0; 4]);
	assert!((k[0] as f64 - h[0] as f64 * 0.92).abs() < 1e-6);
}

/// S2 on the PLANT: every command is clamped to [pwm_min, 1] BEFORE the lag (the filter
/// seeds at the floor and never goes below it); 0 is the ABI-64 clamp [0, 1].
#[test]
fn plant_idle_floor_clamps_before_the_lag()
{
	let mut sim = x500_sim();
	sim.set_actuator_extras_core(0.0, 0.15);
	assert_eq!(sim.actuate_for_test([0.0, 0.1, 0.5, 1.3]), [0.15, 0.15, 0.5, 1.0]);
	let mut lagged = x500_sim();
	lagged.set_motor_lag(0.05);
	lagged.set_actuator_extras_core(0.10, 0.15);
	assert_eq!(lagged.actuate_for_test([0.05; 4]), [0.15; 4], "the filter seeds at the floor");
	for _ in 0..200
	{
		let p = lagged.actuate_for_test([0.0; 4]);
		assert!(p.iter().all(|&v| v >= 0.15), "{p:?}");
	}
	let mut off = x500_sim();
	assert_eq!(off.actuate_for_test([-0.2, 0.1, 0.5, 1.3]), [0.0, 0.1, 0.5, 1.0]);
	assert_eq!(lagged.actuator(), ActuatorSpec::new(0.05, 0.10, 0.15).unwrap());
}

fn close(gpu: f64, cpu: f64, rel: f64, floor: f64, what: &str)
{
	let tol = floor.max(cpu.abs() * rel);
	assert!((gpu - cpu).abs() <= tol, "{what}: gpu={gpu} cpu={cpu} tol={tol}");
}

/// CPU == GPU on reward, err, pos, alt, dx, dy, mid (the abi32 tolerance policy). Returns
/// the CPU row (and whether the GPU ran).
fn assert_parity(c: &WnnController, p: PlantOpts, what: &str) -> [f64; crate::cpu_score::ROW_METRICS]
{
	let s1 = stage2_with(p);
	let cr = cpu_rows_on(c, &s1, 500);
	if let Some(gr) = gpu_rows_on(c, &s1, 500)
	{
		close(gr[1], cr[1], 2e-2, 1e-4, &format!("{what} err"));
		close(gr[0], cr[0], 5e-2, 1e-3, &format!("{what} reward"));
		for (i, name) in [(13, "pos"), (14, "alt"), (15, "dx"), (16, "dy")]
		{
			close(gr[i], cr[i], 5e-2, 1e-3, &format!("{what} {name}"));
		}
		close(gr[19], cr[19], 2e-2, 1e-4, &format!("{what} mid"));
		eprintln!("[ctrl86-parity] {what}: cpu err {:.6} gpu err {:.6} | cpu pos {:.5} gpu pos {:.5}", cr[1], gr[1], cr[13], gr[13]);
	}
	cr
}

fn moved(a: &[f64], b: &[f64], what: &str)
{
	assert!(a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-9), "{what}: the option did not move the row — dead plumbing");
}

fn sweep(c: &WnnController, kind: &str)
{
	let base = assert_parity(c, real_life_plant(Spelling::Legacy), &format!("{kind} symmetric"));
	let asym = assert_parity(c, plant(LAG_DOWN, 0.0), &format!("{kind} asym_lag"));
	let floor = assert_parity(c, plant(0.0, FLOOR), &format!("{kind} pwm_floor"));
	let both = assert_parity(c, plant(LAG_DOWN, FLOOR), &format!("{kind} asym_lag+pwm_floor"));
	moved(&base, &asym, &format!("{kind} asym_lag"));
	moved(&base, &floor, &format!("{kind} pwm_floor"));
	moved(&asym, &both, &format!("{kind} floor on top of asym"));
}

/// Parity sweep on a cascade (output=ctbr) student — the l2b lineage's student shape.
#[test]
fn cpu_fallback_matches_gpu_ctrl86_cascade()
{
	let (c, _) = trained_on(&l2b_cfg(4, Spelling::Legacy));
	sweep(&c, "cascade");
}

/// Parity sweep on a motor-path student with obs_pwm: the actuator twin is what the
/// obs_pwm feature reports under lag, so this sweep pins the twin's switch + floor too.
#[test]
fn cpu_fallback_matches_gpu_ctrl86_motor_obs_pwm()
{
	let (c, _) = train(0, 1, LAG, 4, true);
	sweep(&c, "motor+obs_pwm");
}

/// The extended actuator reaches the trainer (its rollouts AND its lag-aware teachers'
/// twins) and the classical rivals.
#[test]
fn extended_actuator_reaches_trainer_and_rivals()
{
	let legacy = l2b_cfg(4, Spelling::Legacy);
	let mut ext = legacy.clone();
	ext.motor_lag_down_s = LAG_DOWN;
	ext.pwm_min = FLOOR;
	ext.validate_abi32().expect("valid extended cfg");
	let (_, d0) = trained_on(&legacy);
	let (_, d1) = trained_on(&ext);
	assert_ne!(d0, d1, "the trainer never saw the extended actuator");
	for teacher in [0u8, 4]
	{
		let r0 = rival_on(teacher, &stage2_with(real_life_plant(Spelling::Legacy)));
		let r1 = rival_on(teacher, &stage2_with(plant(LAG_DOWN, FLOOR)));
		assert_ne!(r0, r1, "rival {teacher} never saw the extended actuator");
	}
	// The trainer's teacher designs at the MEAN lag; its twin follows the plant.
	let af = crate::dagger_train::AirframeRs::from_cfg(&ext);
	assert_eq!(af.actuator(), ActuatorSpec::new(LAG, LAG_DOWN, FLOOR).unwrap());
	assert_eq!(af.actuator().design_lag_s(), 0.5 * (LAG + LAG_DOWN));
	assert_eq!(af.sim().actuator(), af.actuator(), "sim() flies the cfg's actuator");
}

#[test]
fn refusals()
{
	let mut c = l2b_cfg(4, Spelling::Legacy);
	c.motor_lag_s = 0.0;
	c.motor_lag_down_s = 0.1;
	assert!(c.validate_abi32().unwrap_err().contains("motor_lag_down_s"), "down without a lag");
	let mut c = l2b_cfg(4, Spelling::Legacy);
	c.pwm_min = 1.0;
	assert!(c.validate_abi32().is_err(), "floor >= 1");
	let mut c = l2b_cfg(4, Spelling::Legacy);
	c.translation = false;
	c.imu_specific_force = false;
	c.rotor_drag = 0.0;
	c.att_err_tilt_ref = false;
	c.pwm_min = 0.15;
	assert!(c.validate_abi32().unwrap_err().contains("translation"), "extras without translation");
}
