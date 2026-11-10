//! ABI 48 (09/11/2026): the training-only initial horizontal velocity
//! (`RewardGatedConfigPacked.init_vxy`, `--train-init-vxy`; docs/ctrl_dres_failures.md
//! idea #4).
//!
//!   * init_vxy = 0 draws NOTHING: the episode starts at rest and the rng sits exactly
//!     where the ABI-47 draw left it (the ctrl17 pins cover the whole rollout).
//!   * init_vxy > 0 keeps x0/y0 (drawn first), starts the plant moving with
//!     |v_axis| <= init_vxy, and consumes exactly two extra draws.
//!   * the ctor refuses a negative / non-finite bound, and a bound without the
//!     horizontal channel (it would be silently ignored).

use crate::ctrl17_pins::pin_translation_cfg;
use crate::dagger_train::{AirframeRs, RewardGatedConfigPacked};
use crate::episode_regime::draw_translation_episode;
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};

/// One draw from seed `seed`: ((x, y), (vx, vy), the rng's next word afterwards).
fn draw(cfg: &RewardGatedConfigPacked, seed: u64) -> ((f32, f32), (f32, f32), u64)
{
	let mut sim = AirframeRs::from_cfg(cfg).sim();
	let mut rng = SmallRng::seed_from_u64(seed);
	draw_translation_episode(&mut sim, cfg, &mut rng);
	(sim.position_xy(), sim.velocity_xy(), rng.next_u64())
}

fn with_vxy(v: f32) -> RewardGatedConfigPacked
{
	let mut c = pin_translation_cfg(0.5);
	c.init_vxy = v;
	c
}

#[test]
fn zero_bound_starts_at_rest_and_draws_nothing()
{
	let base = pin_translation_cfg(0.5);
	assert_eq!(base.init_vxy, 0.0, "the default must be at rest");
	for seed in 0..32
	{
		let (_, v, _) = draw(&base, seed);
		assert_eq!(v, (0.0, 0.0), "seed {seed}: init_vxy 0 must start at rest");
	}
}

#[test]
fn positive_bound_moves_only_the_velocity()
{
	let (base, wide) = (pin_translation_cfg(0.5), with_vxy(0.4));
	let mut moving = 0;
	for seed in 0..64
	{
		let (p0, _, next0) = draw(&base, seed);
		let (p1, v1, next1) = draw(&wide, seed);
		assert_eq!(p0, p1, "seed {seed}: x0/y0 are drawn before the velocity and must not move");
		assert!(v1.0.abs() <= 0.4 && v1.1.abs() <= 0.4, "seed {seed}: {v1:?} outside the bound");
		assert_ne!(next0, next1, "seed {seed}: the velocity draw must consume the rng");
		moving += (v1.0.hypot(v1.1) > 0.2) as usize;
	}
	assert!(moving > 16, "only {moving}/64 episodes start above 0.2 m/s at a 0.4 bound");
}

#[test]
fn positive_bound_consumes_exactly_two_draws()
{
	use rand::Rng;
	let wide = with_vxy(0.4);
	let (_, _, next_wide) = draw(&wide, 7);
	let mut sim = AirframeRs::from_cfg(&wide).sim();
	let mut rng = SmallRng::seed_from_u64(7);
	draw_translation_episode(&mut sim, &pin_translation_cfg(0.5), &mut rng);
	// The ABI-47 draw then the two velocity draws, in the order episode_regime takes them.
	let _: f32 = rng.gen_range(-0.4..0.4);
	let _: f32 = rng.gen_range(-0.4..0.4);
	assert_eq!(rng.next_u64(), next_wide, "init_vxy must add exactly two draws after x0/y0");
}

#[test]
fn ctor_validation_refuses_bad_bounds()
{
	for bad in [-0.1f32, f32::NAN, f32::INFINITY]
	{
		assert!(with_vxy(bad).validate_init_vxy().is_err(), "init_vxy {bad} accepted");
	}
	let mut no_xy = pin_translation_cfg(0.0);
	no_xy.init_vxy = 0.3;
	assert!(no_xy.validate_init_vxy().is_err(), "init_vxy without the horizontal channel accepted");
	assert!(with_vxy(0.4).validate_init_vxy().is_ok());
	assert!(pin_translation_cfg(0.0).validate_init_vxy().is_ok(), "0 must be valid everywhere");
}
