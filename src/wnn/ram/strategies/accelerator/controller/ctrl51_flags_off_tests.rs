//! ABI 51 FLAGS-OFF IDENTITY (docs/ctrl48_block10_levers_spec.md: "each lever is its
//! own default-OFF flag, and with every flag off every existing path is bit-identical").
//!
//! WHERE THE IDENTITY VS ABI 50 IS HELD. The pre-existing byte-identity pins were
//! captured on earlier builds and now run THROUGH the new arguments at their defaults:
//!   * ctrl17_pins — `RewardGatedConfigPacked::new(.., alt_integral = 0)`: the DAgger
//!     rollout, the trainer, calibration and the recorder fingerprints;
//!   * abi38_pins / ctrl38_pins — `score_classical_baseline(.., alt_integral = 0)`,
//!     the CPU scorer and the Metal kernel with `Stage1Cfg.alt_int = OFF` (digest over
//!     the pre-ABI-51 row prefix), the motor-level and cascade trainers;
//!   * the metal_controller GPU pins (ABI 33 / 35 / 37 rows).
//! Those pin values are untouched. What is added here is the explicit statement that
//! the new arguments AT THEIR DEFAULTS are the off path: the defaults construct the
//! off objects, the off objects are identities, and a controller built through the
//! Python constructor with `obs_tilt_ref_err = None` is the controller the ABI-50
//! construction path builds.

use crate::altitude_integrator::AltitudeIntegrator;
use crate::controller::WnnController;
use ram_core::neuron_memory::QUAD_WEIGHTED;

const NF: usize = 22; // 9 base + roll/pitch err + roll/pitch i + 3 vertical + 4 horizontal + 2 abp
const BPF: usize = 3;

fn shape() -> (Vec<f32>, Vec<i64>, Vec<i64>)
{
	use rand::{Rng, SeedableRng};
	let mut rng = rand::rngs::SmallRng::seed_from_u64(0x51);
	let thresholds: Vec<f32> = (0..NF * BPF).map(|_| rng.gen_range(-0.5f32..0.5)).collect();
	let sconn: Vec<i64> = (0..4 * 6).map(|_| rng.gen_range(0..2 * NF * BPF + 4) as i64).collect();
	let oconn: Vec<i64> = (0..16 * 8).map(|_| rng.gen_range(0..NF * BPF + 4) as i64).collect();
	(thresholds, sconn, oconn)
}

/// The Python constructor with every ABI-51 argument at its default.
fn via_ctor(tilt_ref_err: Option<(f32, f32, f32, f32)>) -> WnnController
{
	let (thresholds, sconn, oconn) = shape();
	WnnController::new(
		4, 4, BPF, 2, 4, 6, 8, thresholds, sconn, oconn,
		true, 0.05, 0.98, 1.0,
		false, false, true, true, false, false, false, false,
		0.99, 1.0, 0.001, false, 1, QUAD_WEIGHTED, None, None, 0.05, false, 0.30,
		true, true, true, true, true, false, 1,
		1.0, false,
		true, 2.0, 0.1,
		None,
		true, None,
		tilt_ref_err,
	)
	.expect("ctor")
}

/// The ABI-50 construction path: new_core + the two enables, no ABI-51 call at all.
fn via_abi50_path() -> WnnController
{
	let (mut thresholds, sconn, oconn) = shape();
	let abp = thresholds.split_off((NF - 2) * BPF);
	let mut c = WnnController::new_core(
		4, 4, BPF, 2, 4, 6, 8, thresholds, sconn, oconn,
		true, 0.05, 0.98, 1.0,
		false, false, true, true, false, false, false, false,
		0.99, 1.0, 0.001, false, 1, QUAD_WEIGHTED, None, None, 0.05, false, 0.30,
		true, true, true, true, true, false, 1,
	)
	.expect("core");
	c.enable_att_estimator(2.0, 0.1).unwrap();
	c.enable_att_bias_pred(abp).unwrap();
	c
}

fn trace(c: &mut WnnController, steps: usize) -> Vec<(Vec<u32>, Vec<u32>)>
{
	c.reset(0.0);
	c.seed_att_estimator([0.998, 0.04, -0.05, 0.0]);
	(0..steps)
		.map(|t| {
			let s = t as f32;
			c.set_vertical_obs(0.69, 0.1 * (0.01 * s).sin(), 0.05 * (0.02 * s).cos());
			c.set_horizontal_obs(0.4 - 0.001 * s, -0.3 + 0.0007 * s, 0.25 * (0.02 * s).sin(), -0.2 * (0.015 * s).cos());
			let pwm = c.step([0.02 * (0.01 * s).sin(), -0.015 * (0.013 * s).cos(), 0.003], [0.35 * (0.004 * s).sin(), -0.25 * (0.005 * s).cos(), 9.81], [0.02, -0.01, 0.0]);
			(c.last_feature_vector_ref().iter().map(|v| v.to_bits()).collect(), pwm.iter().map(|v| v.to_bits()).collect())
		})
		.collect()
}

/// `obs_tilt_ref_err = None` (the default) is the ABI-50 controller: every feature
/// and every command, bit for bit; the option, when given, is what moves them.
#[test]
fn controller_default_is_the_abi50_controller()
{
	let (mut a, mut b) = (via_ctor(None), via_abi50_path());
	assert_eq!(a.obs_tilt_ref_err(), None);
	assert_eq!(a.tilt_ref_err_shape(), None);
	assert_eq!(trace(&mut a, 500), trace(&mut b, 500), "the default-constructed controller is not the ABI-50 one");
	let mut on = via_ctor(Some((9.81, 1.0, 1.0, 0.5236)));
	assert_eq!(on.obs_tilt_ref_err(), Some((9.81, 1.0, 1.0, 0.5236)));
	assert_ne!(trace(&mut on, 500), trace(&mut b, 500), "the option is dead");
}

/// `alt_integral = 0` (the default) builds the OFF integrator in every host, and OFF
/// hands each host back exactly the operands it computed before ABI 51.
#[test]
fn alt_integral_default_is_off_in_every_host()
{
	use crate::episode_regime::{alt_integrator_for, sense_altitude};
	for cfg in [crate::ctrl17_pins::pin_translation_cfg(0.5), crate::ctrl17_pins::pin_translation_cfg(0.0), crate::ctrl17_pins::pin_attitude_cfg()]
	{
		assert_eq!(cfg.alt_integral, 0);
		let mut it = alt_integrator_for(&cfg);
		assert!(!it.is_on());
		let af = crate::dagger_train::AirframeRs::from_cfg(&cfg);
		let mut sim = af.sim();
		sim.reset(Some([1.0, 0.0, 0.0, 0.0]), Some([0.0; 3]));
		if cfg.translation
		{
			sim.set_translation_core(cfg.af_mass).unwrap();
			sim.set_vertical_state(-0.37, 0.21);
		}
		for _ in 0..20
		{
			let (e, v) = sense_altitude(&mut it, &sim, cfg.target_altitude);
			assert_eq!(e.to_bits(), (cfg.target_altitude - sim.altitude_rs()).to_bits(), "the trainer's alt_err operand moved");
			assert_eq!(v.to_bits(), sim.vertical_velocity_rs().to_bits(), "the trainer's vz operand moved");
			sim.step([0.6; 4]);
		}
	}
	// The scorers' kwargs at their defaults (mode 0; the PD kwargs are then not read).
	let scorer = AltitudeIntegrator::from_plant(0, 0.0, 9.81, 0.2, 2.0, 1.0, 0.25, 0.001).expect("defaults");
	assert!(!scorer.is_on());
	// A Stage1Cfg carrying that default scores exactly like one carrying OFF.
	let c = crate::abi32_tests::controller(7, false);
	let mut s1 = crate::abi32_tests::cfg(crate::plant_opts::PlantOpts::OFF, crate::att_err_ref::AttErrRef::LEVEL);
	let base = crate::abi32_tests::cpu(&c, &s1, 300);
	s1.alt_int = scorer;
	assert_eq!(base.map(f64::to_bits), crate::abi32_tests::cpu(&c, &s1, 300).map(f64::to_bits));
}
