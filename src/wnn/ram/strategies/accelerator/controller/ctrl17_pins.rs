//! CTRL-17 BYTE-IDENTITY PINS (26/09/2026).
//!
//! The stage-2 trainer fixes change the per-round checkpoint eval, the threshold
//! fitter and the MEMORY recorder for TRANSLATION runs (a deliberate new lineage).
//! They must NOT move anything else. These fingerprints were computed on the
//! pre-fix tree (commit b7b0f422) with this very file and are pinned here:
//!
//!   * attitude_only_trainer — the WHOLE dagger_train_inplace_rs (rollouts, gate,
//!     training, per-round eval, checkpoint restore) on the synthetic plant, with
//!     disturbances on. Every attitude-only run is byte-identical.
//!   * translation_rollout_{stage1,stage2} — rollout_and_label_rs alone under
//!     translation with λ_alt = λ_pos = 0: the TRAINING rollout (labels, obs
//!     streams, reward) is unchanged by the refactor onto episode_regime; only the
//!     eval / fitter / recorder around it moved.
//!
//! If one of these changes, the change is results-moving for runs it should not
//! touch — do not re-pin without a lineage decision.

use crate::controller::{AttitudeSim, WnnController};
use crate::dagger_train::{dagger_train_inplace_rs, rollout_and_label_rs, RewardGatedConfigPacked};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// FNV-1a over a stream of 64-bit words.
struct Fnv(u64);

impl Fnv
{
	fn new() -> Self
	{
		Fnv(0xcbf2_9ce4_8422_2325)
	}
	fn word(&mut self, w: u64)
	{
		for b in w.to_le_bytes()
		{
			self.0 ^= b as u64;
			self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
		}
	}
	fn f32s(&mut self, xs: &[f32])
	{
		for x in xs
		{
			self.word(x.to_bits() as u64);
		}
	}
}

/// A delta-control sn=0 BINARY student with `nf` features.
pub(crate) fn pin_student(vert: bool, horiz: bool) -> WnnController
{
	let nf = 9 + if vert { 3 } else { 0 } + if horiz { 4 } else { 0 };
	let (levels, bpf, obpn) = (8usize, 3usize, 8usize);
	let mut rng = SmallRng::seed_from_u64(0xC717);
	let frame_bits = nf * bpf;
	let thresholds: Vec<f32> = (0..frame_bits).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
	let out_conn: Vec<i64> = (0..4 * levels * obpn)
		.map(|_| rng.gen_range(0..frame_bits) as i64)
		.collect();
	WnnController::new_core(
		4, levels, bpf, 1, 0, 0, obpn, thresholds, Vec::new(), out_conn,
		true, 0.1, 0.9, 1.0,
		false, false, false, false, false, false, false, false,
		0.99, 1.0, 0.001, false, 1,
		ram_core::neuron_memory::BINARY, None, None, 0.05, false, 0.30,
		vert, vert, vert, horiz, horiz, false, 1,
	)
	.expect("pin student must construct")
}

/// The synthetic-plant attitude-only config, disturbances ON.
pub(crate) fn pin_attitude_cfg() -> RewardGatedConfigPacked
{
	let mut c = pin_translation_cfg(0.0);
	c.translation = false;
	c.af_mass = 0.0;
	c.af_arm_length = 0.075;
	c.af_k_thrust = 2.4;
	c.af_k_drag = 0.05;
	c.af_inertia = [0.0023, 0.0023, 0.0046];
	c.af_pid_att = [0.0; 12];
	c.af_pid_rate = [0.0; 12];
	c.af_pid_out_limit_n = 0.0;
	c.af_pid_hover_n = 0.0;
	c.af_pid_attitude_hz = 0.0;
	c.af_pid_lpf_hz = 0.0;
	c
}

/// A cf21 translation config (λ_alt = λ_pos = 0), xy armed iff xy > 0.
pub(crate) fn pin_translation_cfg(xy: f32) -> RewardGatedConfigPacked
{
	let mut c = RewardGatedConfigPacked::new(
		2, 3, 300, 32, 4, false, 0, false, 0, 0.5, true, 0, 0, vec![], vec![], true, 0.0, 0.1,
		true, 4.0, 8.0, 0.001, 0.3, 0.5, 0.3, 2, 0.1, 0.999, 0.9, 5, 1, 32, true, true, true,
		true, true, [0.01, -0.01, 0.0], 0.02, 0.1, [1.0, 1.0, 1.0, 1.0], 0.01, 0.0, 0.05, 0.0,
		0, 0, 0.0, false, 0.070710678, 0.2, 0.00569278844371417, [3.004e-5, 3.019e-5, 5.304e-5],
		9.81,
		[6.0, 3.0, 0.0, 0.3490658503988659, 6.0, 3.0, 0.0, 0.3490658503988659, 6.0, 1.0, 0.35, 6.283185307179586],
		[
			0.03497110216713654, 0.06994220433427308, 0.00043713877708920673, 0.5811946409141117,
			0.03497110216713654, 0.06994220433427308, 0.00043713877708920673, 0.5811946409141117,
			0.02098266130028192, 0.002920087030955901, 0.0, 2.909463863074547,
		],
		0.09999847409781033, 0.09638325, 500.0, 30.0, false, 0.0, true, 0.0393, 0.15, 0.3, 0.2,
		0.1, 0.0, 2.0, 1.0, 0.25, xy, 0.0, 0.0, 1.0, 1.0, 0.5236, 1, 0.0,
	);
	c.teacher = 0;
	c
}

fn hash_traj(h: &mut Fnv, t: &crate::dagger_train::TrajectoryRs)
{
	h.word(t.steps as u64);
	h.word(t.cumulative_reward.to_bits());
	h.word(t.mean_attitude_error_rad.to_bits());
	for i in 0..t.steps
	{
		h.f32s(&t.pid_pwms[i]);
		h.f32s(&t.student_pwms[i]);
		h.f32s(&t.vert_obs[i]);
		h.f32s(&t.horiz_obs[i]);
		h.f32s(&t.gyros[i]);
	}
}

fn rollout_fingerprint(cfg: &RewardGatedConfigPacked, vert: bool, horiz: bool) -> u64
{
	let mut c = pin_student(vert, horiz);
	let af = crate::dagger_train::AirframeRs::from_cfg(cfg);
	let mut teacher = af.teacher(cfg.teacher);
	let mut sim: AttitudeSim = af.sim();
	let mut rng = SmallRng::seed_from_u64(77);
	let mut h = Fnv::new();
	for _ in 0..3
	{
		let t = rollout_and_label_rs(&mut c, &mut teacher, &mut sim, cfg, 0.1, &mut rng, [0.0; 3]);
		hash_traj(&mut h, &t);
	}
	h.0
}

fn trainer_fingerprint(cfg: &RewardGatedConfigPacked) -> u64
{
	let mut c = pin_student(false, false);
	let st = dagger_train_inplace_rs(&mut c, cfg, [0.0; 3], 4242);
	let mut h = Fnv::new();
	for v in st
		.iter_fitness
		.iter()
		.chain(&st.iter_mean_err_deg)
		.chain(&st.iter_stable_rate)
		.chain(&st.iter_motor_jerk_mean)
		.chain(&st.iter_mono_violations)
		.chain(&st.iter_mean_episode_reward)
	{
		h.word(v.to_bits());
	}
	for n in st.iter_n_trained.iter().chain(&st.iter_cells_written)
	{
		h.word(*n as u64);
	}
	let (s_cells, o_cells) = c.export_cells();
	for (n, a, v) in s_cells.iter().chain(&o_cells)
	{
		h.word(*n as u64);
		h.word(*a);
		h.word(*v as u64);
	}
	h.0
}

#[test]
fn ctrl17_pin_fingerprints()
{
	let att = trainer_fingerprint(&pin_attitude_cfg());
	let s1 = rollout_fingerprint(&pin_translation_cfg(0.0), true, false);
	let s2 = rollout_fingerprint(&pin_translation_cfg(0.5), true, true);
	eprintln!("[ctrl17-pin] attitude_only_trainer={att:#018x} rollout_stage1={s1:#018x} rollout_stage2={s2:#018x}");
	// Computed on the pre-fix tree b7b0f422 (see the module note).
	assert_eq!(att, 0xcea4_5b5b_83ba_f1fc, "attitude-only trainer moved");
	assert_eq!(s1, 0x71d1_101d_f481_3abb, "stage-1 training rollout moved");
	assert_eq!(s2, 0xd9bb_cf0f_4855_543f, "stage-2 training rollout moved");
}
