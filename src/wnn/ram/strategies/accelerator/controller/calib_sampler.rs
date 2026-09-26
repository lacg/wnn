//! CTRL-17 G1 (26/09/2026): the threshold-calibration feature sampler, in Rust.
//!
//! THE DEFECT. `fit_thresholds_from_pid_rollouts` (wnn/control/evaluator.py) fitted
//! the thermometer ladder on rollouts driven by the ATTITUDE-ONLY firmware PID with
//! target (0,0,0) and no outer loops. Under translation that plant is not the one
//! the student is trained in:
//!   * horizontal: it never called set_horizontal_state/set_horizontal_obs, so all
//!     four xy features were sampled as the constructor's 0.0 — every threshold 0.0,
//!     a sign-only code (32 address bits carrying 4 bits of information). Drawing
//!     x0/y0 alone would not fix it: with no position loop e stays at x0 and v ≈ 0,
//!     so the velocity ladder stays degenerate.
//!   * vertical: no altitude loop and no mass draw, so alt_err/vz sampled an
//!     uncontrolled constant-velocity DRIFT rather than the teacher's return to
//!     target — a ladder spent on states the trained student never visits.
//!
//! THE FIX. Fly the TRAINING episode (episode_regime: mass draw, vertical and
//! horizontal starts, commanded collective, weather) with the teacher inside the
//! same outer altitude + position loops the DAgger labels come from, the teacher
//! driving the plant, and record the controller's OWN compute_features output at
//! every step — the values step() thermometer-encodes, no Python re-derivation.
//!
//! The obs_pwm proxy of the Python fitter (13/09/2026) is kept: the feature
//! controller is untrained, so its accumulator is set to the teacher's PREVIOUS
//! action before each step or the pwm ladder is degenerate again.

use crate::controller::{yaw_from_quat_rs, AttitudeSim, WnnController};
use crate::dagger_train::{
	apply_cfg_disturbance, sample_initial_state, AirframeRs, RewardGatedConfigPacked, TeacherBank,
};
use crate::episode_regime::{anchor_controller, draw_translation_episode, feed_live_obs, OuterLoops};
use crate::optimal::Teacher;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Per-feature samples (outer index = feature, length `controller.num_features()`)
/// from `num_episodes` teacher-driven training episodes of `cfg`. The initial tilt
/// bound is `cfg.full_tilt_deg` (the caller passes its calibration regime there);
/// the teacher is `cfg.teacher_id_for(0, ep)`.
pub fn sample_calibration_features_rs(
	controller: &mut WnnController,
	cfg: &RewardGatedConfigPacked,
	num_episodes: usize,
	seed: u64,
) -> Vec<Vec<f32>>
{
	let nf = controller.num_features();
	let cap = num_episodes * cfg.steps_per_episode;
	let mut samples: Vec<Vec<f32>> = (0..nf).map(|_| Vec::with_capacity(cap)).collect();
	let mut rng = SmallRng::seed_from_u64(seed);
	let af = AirframeRs::from_cfg(cfg);
	let mut bank = TeacherBank::new(af);
	let mut sim = af.sim();
	let loops = OuterLoops::for_cfg(cfg);
	let mut ep_ctx = CalibEpisode {
		cfg,
		loops: &loops,
		tilt_rad: cfg.full_tilt_deg.to_radians(),
	};
	for ep in 0..num_episodes
	{
		if ram_core::cancel::check_cancel()
		{
			break;
		}
		let teacher = bank.get_mut(cfg.teacher_id_for(0, ep));
		ep_ctx.run(controller, teacher, &mut sim, &mut rng, &mut samples);
	}
	samples
}

/// What one calibration episode needs besides the mutable actors.
struct CalibEpisode<'a>
{
	cfg: &'a RewardGatedConfigPacked,
	loops: &'a OuterLoops,
	tilt_rad: f64,
}

impl CalibEpisode<'_>
{
	/// One teacher-driven training episode, appending every step's feature vector.
	fn run(
		&mut self,
		controller: &mut WnnController,
		teacher: &mut Teacher,
		sim: &mut AttitudeSim,
		rng: &mut SmallRng,
		samples: &mut [Vec<f32>],
	)
	{
		let collective = self.begin(controller, teacher, sim, rng);
		let proxy_pwm = controller.obs_pwm_flag();
		let mut prev_applied = [controller.collective_anchor(); 4];
		let mut last_applied = [0.5f32; 4];
		let target = [0.0f32; 3];
		for _t in 0..self.cfg.steps_per_episode
		{
			if sim.is_unstable()
			{
				break;
			}
			let (gyro, accel) = sim.read_imu();
			let q = sim.quaternion();
			teacher.observe(gyro, last_applied.map(|v| v as f64));
			feed_live_obs(controller, sim, self.cfg, collective);
			if proxy_pwm
			{
				controller.set_pwm_accumulator(prev_applied);
			}
			let _ = controller.step(gyro, accel, target);
			for (f, col) in samples.iter_mut().enumerate()
			{
				col.push(controller.last_feature_vector_ref()[f]);
			}
			let expert = self
				.loops
				.expert(teacher, sim, q, gyro, target, self.cfg.target_altitude);
			let applied = expert.map(|v| v as f32);
			controller.observe_applied(applied);
			sim.step(applied);
			prev_applied = applied;
			last_applied = applied;
		}
	}

	/// The training episode's setup, in the trainer's draw order. Returns the
	/// episode's commanded collective (0.0 without translation).
	fn begin(
		&self,
		controller: &mut WnnController,
		teacher: &mut Teacher,
		sim: &mut AttitudeSim,
		rng: &mut SmallRng,
	) -> f32
	{
		let cfg = self.cfg;
		let (init_q, init_omega) = sample_initial_state(
			rng,
			self.tilt_rad,
			cfg.max_initial_yaw_rad,
			cfg.max_initial_body_rate,
			cfg.max_initial_yaw_rate,
			[cfg.active_roll, cfg.active_pitch, cfg.active_yaw],
		);
		sim.reset(Some(init_q), Some(init_omega));
		let collective = draw_translation_episode(sim, cfg, rng);
		apply_cfg_disturbance(sim, cfg, rng);
		teacher.reset();
		controller.reset(yaw_from_quat_rs(init_q));
		anchor_controller(controller, cfg, collective);
		collective
	}
}
