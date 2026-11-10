//! CTRL-51 (10/11/2026): reference export for the on-board MPCOF C port
//! (mcu/mpcof_cascade.c). NOT part of the repo's crate: mcu/ctrl51/export_mpcof.py
//! copies the controller workspace to /tmp, drops this file in as a #[cfg(test)]
//! module and runs it, so the repo's Rust is never edited.
//!
//! It writes (directory $CTRL51_OUT):
//!   tables.txt   the offline part of CtbrMpcOf, built by the SAME calls
//!                CtbrAxisMpc::build makes (outer_map -> dare_doubling ->
//!                condense_horizon) on the AxisPlant CtbrMpcOf::new builds, plus the
//!                observer and shadow-rate-stage constants — every f64 as its bits;
//!   traj_<regime>.bin  teacher-flown episodes (the cascade MPCOF over the firmware
//!                rate loop, cf21_brushless, lag 0.20, L4C weather, stage-2 position
//!                loop): per tick the inputs CtbrMpcOf reads and its output label.
//! The C port is checked against the traj labels (mcu/ctrl51/parity_mpcof.py), so a
//! table built here that differs from the teacher's own planner fails parity.

use std::fmt::Write as _;
use std::io::Write as _;

use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::ctbr_axis_mpc::{AxisPlant, NS};
use crate::ctbr_mpcof::{CTBR_MPC_HORIZON, CTBR_MPC_L_GAIN, CTBR_MPC_RATE_MAX, CTBR_MPC_WEIGHTS};
use crate::dagger_train::{AirframeRs, RewardGatedConfigPacked};
use crate::optimal::{condense_horizon, dare_doubling, Mat};
use crate::pid_firmware::RateStageFw;

const N_O: usize = 10;

fn bits(v: &[f64]) -> String
{
	v.iter().map(|x| format!("{:016x}", x.to_bits())).collect::<Vec<_>>().join(" ")
}

fn line(out: &mut String, key: &str, v: &[f64])
{
	writeln!(out, "{key} {} {}", v.len(), bits(v)).unwrap();
}

/// The production-shaped cascade config: cf21 stage 2, mpcof teacher, lag 0.20,
/// L4C weather, the a49 recipe's plant (specific-force IMU, rotor drag 0.78).
fn cfg(steps: usize) -> RewardGatedConfigPacked
{
	let mut c = crate::abi38_tests::l4c_cfg(0.20, 4);
	c.imu_specific_force = true;
	c.rotor_drag = 0.78;
	c.steps_per_episode = steps;
	c
}

/// CtbrMpcOf::new's per-axis plant, field for field.
fn axis_plants(rate: &RateStageFw, af: &AirframeRs) -> [AxisPlant; 3]
{
	let alpha = crate::actuator_lag::motor_lag_alpha(af.motor_lag_s, af.dt);
	let c = crate::cascade_dhat::torque_gains(af.arm_length, af.k_drag, af.inertia);
	std::array::from_fn(|i| {
		let g = rate.axis_gains(i);
		AxisPlant {
			kp: g.kp,
			ki: g.ki,
			kd: g.kd,
			lpf: rate.lpf_coeffs(),
			dt: af.dt as f64,
			dt_rate: rate.dt_rate(),
			decimation: rate.decimation(),
			c: c[i],
			alpha,
		}
	})
}

/// CtbrAxisMpc::build's offline matrices (H, g_map, ustar_map) for one axis/phase.
fn condensed(p: &AxisPlant, phase: u32) -> [Vec<f64>; 3]
{
	let w = CTBR_MPC_WEIGHTS;
	let (a, b) = p.outer_map(N_O, phase);
	let mut qm = Mat::zeros(NS, NS);
	qm.set(0, 0, w.q_att);
	qm.set(1, 1, w.q_rate);
	let mut rm = Mat::zeros(1, 1);
	rm.set(0, 0, w.r);
	let qf = dare_doubling(&a, &b, &qm, &rm);
	let ch = condense_horizon(&a, &b, &qm, &rm, &qf, CTBR_MPC_HORIZON);
	[ch.h.data().to_vec(), ch.g_map.data().to_vec(), ch.ustar_map.data().to_vec()]
}

fn write_tables(dir: &str, af: &AirframeRs, rate: &RateStageFw)
{
	let plants = axis_plants(rate, af);
	let mut s = String::new();
	writeln!(s, "n_o {N_O}\nhorizon {CTBR_MPC_HORIZON}\nns {NS}\ndecimation {}", rate.decimation()).unwrap();
	line(&mut s, "rate_max", &CTBR_MPC_RATE_MAX);
	line(&mut s, "kp", &plants.map(|p| p.kp));
	line(&mut s, "ki", &plants.map(|p| p.ki));
	line(&mut s, "c", &plants.map(|p| p.c));
	for phase in 0..rate.decimation()
	{
		for (a, p) in plants.iter().enumerate()
		{
			let [h, g, u] = condensed(p, phase);
			line(&mut s, &format!("H_{phase}_{a}"), &h);
			line(&mut s, &format!("G_{phase}_{a}"), &g);
			line(&mut s, &format!("U_{phase}_{a}"), &u);
		}
	}
	write_observer(&mut s, af, rate);
	write_rate_stage(&mut s, rate);
	std::fs::write(format!("{dir}/tables.txt"), s).unwrap();
}

/// CascadeDhat::new(torque_gains, k_thrust, dt, motor_lag, out_limit) constants.
fn write_observer(s: &mut String, af: &AirframeRs, rate: &RateStageFw)
{
	let c = crate::cascade_dhat::torque_gains(af.arm_length, af.k_drag, af.inertia);
	let out = rate.out_limit_n();
	line(s, "obs_c", &c);
	line(s, "obs_clamp", &[c[0] * out, c[1] * out, c[2] * out]);
	line(s, "obs_kt", &[af.k_thrust as f64]);
	line(s, "obs_dt", &[af.dt as f64]);
	line(s, "obs_alpha", &[crate::actuator_lag::motor_lag_alpha(af.motor_lag_s, af.dt)]);
	line(s, "obs_l", &[CTBR_MPC_L_GAIN]);
}

/// The shadow firmware rate stage (RateStageFw) as built: gains, limits, filter.
fn write_rate_stage(s: &mut String, rate: &RateStageFw)
{
	line(s, "rate_gains", &rate.gains_flat());
	line(s, "rate_out_limit", &[rate.out_limit_n()]);
	line(s, "rate_dt", &[rate.dt_rate()]);
	line(s, "rate_lpf", &rate.lpf_coeffs().unwrap_or([0.0; 5]));
	line(s, "rate_has_lpf", &[rate.lpf_coeffs().is_some() as u8 as f64]);
}

/// One teacher-flown episode; appends per tick [14 f32 inputs][3 f64 label][12 f64 hidden].
fn fly_episode(buf: &mut Vec<u8>, cfg: &RewardGatedConfigPacked, af: &AirframeRs, rng: &mut SmallRng, tilt: f64, rate0: f64) -> usize
{
	use crate::episode_regime::{alt_pd_for, draw_translation_episode, pos_loop_for};
	let mut sim = af.sim();
	let mut teacher = af.teacher(4);
	let mut pilot = crate::ctbr::CtbrOverRate::new(af.pid_fw.as_ref().unwrap().rate_stage(), N_O);
	let (pd, pl) = (alt_pd_for(cfg).unwrap(), pos_loop_for(cfg).unwrap());
	let (q0, w0) = crate::dagger_train::sample_initial_state(rng, tilt, cfg.max_initial_yaw_rad, rate0, cfg.max_initial_yaw_rate, [true; 3]);
	sim.reset(Some(q0), Some(w0));
	draw_translation_episode(&mut sim, cfg, rng);
	crate::dagger_train::apply_cfg_disturbance(&mut sim, cfg, rng);
	teacher.reset();
	let mut last = [0.5f32; 4];
	let mut ticks = 0usize;
	let mut rows: Vec<u8> = Vec::new();
	for _ in 0..cfg.steps_per_episode
	{
		if sim.is_unstable()
		{
			break;
		}
		let (gyro, _) = sim.read_imu();
		let q = sim.quaternion();
		teacher.observe(gyro, last.map(|v| v as f64));
		let coll = pd.delta((cfg.target_altitude - sim.altitude_rs()) as f64, sim.vertical_velocity_rs() as f64);
		let [hx, hy] = sim.position_xy_rs();
		let [hvx, hvy] = sim.velocity_xy_rs();
		let (r, p) = pl.tilt_ref((-hx) as f64, hvx as f64, (-hy) as f64, hvy as f64);
		let tgt = [r as f32, p as f32, 0.0f32];
		let label = teacher.ctbr_label(q, gyro, tgt, N_O, coll).unwrap();
		let hidden = teacher.diag_ctbr_hidden();
		for v in gyro.iter().chain(last.iter()).chain(q.iter()).chain(tgt.iter())
		{
			rows.extend_from_slice(&v.to_le_bytes());
		}
		for v in label[1..].iter().chain(hidden.iter())
		{
			rows.extend_from_slice(&v.to_le_bytes());
		}
		last = pilot.fly(label, gyro).map(|v| v as f32);
		sim.step(last);
		ticks += 1;
	}
	buf.extend_from_slice(&(ticks as u32).to_le_bytes());
	buf.extend_from_slice(&rows);
	ticks
}

fn write_regime(dir: &str, name: &str, eps: usize, tilt_deg: f64, rate0: f64, seed: u64)
{
	let cfg = cfg(2000);
	let af = AirframeRs::from_cfg(&cfg);
	let mut rng = SmallRng::seed_from_u64(seed);
	let mut buf = (eps as u32).to_le_bytes().to_vec();
	let ticks: Vec<usize> = (0..eps).map(|_| fly_episode(&mut buf, &cfg, &af, &mut rng, tilt_deg.to_radians(), rate0)).collect();
	std::fs::File::create(format!("{dir}/traj_{name}.bin")).unwrap().write_all(&buf).unwrap();
	eprintln!("[ctrl51] {name}: {eps} episodes, ticks {ticks:?}");
}

#[test]
fn ctrl51_export_mpcof_reference()
{
	let dir = std::env::var("CTRL51_OUT").expect("CTRL51_OUT");
	let cfg = cfg(2000);
	let af = AirframeRs::from_cfg(&cfg);
	let rate = af.pid_fw.as_ref().expect("cf21 firmware cascade").rate_stage();
	eprintln!("[ctrl51] airframe arm {} kt {} kd {} J {:?} lag {} rate dec {} dt_rate {}",
		af.arm_length, af.k_thrust, af.k_drag, af.inertia, af.motor_lag_s, rate.decimation(), rate.dt_rate());
	write_tables(&dir, &af, &rate);
	write_regime(&dir, "nominal", 20, 5.0, cfg.max_initial_body_rate, 0xC751_0001);
	write_regime(&dir, "stress", 10, 30.0, 2.0, 0xC751_0002);
	// Worst-case probe: large upsets drive the plan into the rate box, so the exact
	// box QP (PDAS / primal fallback) runs — the variable-cost path.
	write_regime(&dir, "upset", 20, 60.0, 8.0, 0xC751_0003);
}
