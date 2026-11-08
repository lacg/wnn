//! CTRL-48 (ABI 45): the cascade student's unabsorbed-disturbance block (a3,
//! `--obs-cascade-dhat-residual`), r = (d̂ + c·ki·I)/(c·ki·i_limit) per axis.
//!
//!   * contract: +3 features after rate-I and d̂, independent of both (a3 alone runs
//!     the observer and the rate-I value without emitting them), value = the formula
//!     on the recorded six, provenance names it, plant must be the trainer's;
//!   * physics: under motor asymmetry the untrained cascade's rate loop absorbs the
//!     roll/pitch disturbance, so r → 0 there while d̂ stays large — the hold bias the
//!     plain d̂ block carries is gone;
//!   * train = deploy: the replay rebuilds r from the recorded six bit for bit;
//!   * CPU scorer vs Metal kernel, and the Metal split record/train vs the CPU split
//!     (CTRL-40 discipline: no CPU-only gate for the new block).

use crate::abi38_pins::{cpu_rows, gpu_rows, ARM, INERTIA, KD};
use crate::abi38_tests::{l4c_cfg, spec};
use crate::controller::ReplayObs;
use crate::ctrl38_tests::{arm_plant, asym_cfg, cf21_sim, fly, obs_spec3, obs_student3};
use crate::ctrl40_tests::{check, parity, rollouts, zero_yaw};
use crate::dagger_train::{dagger_train_inplace_with, rollout_and_label_rs, AirframeRs};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// r recomputed from a recorded six with the student's own scale.
fn resid_of(c: &crate::controller::WnnController, v: &[f32; 6]) -> [f32; 3]
{
	c.cascade_obs_ref().and_then(|o| o.residual()).expect("residual on").of(v)
}

#[test]
fn residual_contract_and_independence()
{
	let base = obs_student3(0, 10, 2, false, false, false).num_features();
	for (ri, dh, rs, add) in [(false, false, true, 3usize), (true, false, true, 6), (true, true, true, 9), (false, true, true, 6)]
	{
		let c = obs_student3(0, 10, 2, ri, dh, rs);
		assert_eq!(c.num_features(), base + add, "rate_i={ri} dhat={dh} resid={rs}");
		assert_eq!(c.ctbr_spec().unwrap().cascade_feature_count(), add);
	}
	// a3 alone: the six recorded values are LIVE (rate-I + d̂ both run), and the
	// three appended features are exactly r of them.
	let mut c = obs_student3(0, 1, 3, false, false, true);
	let mut sim = cf21_sim(0.0);
	arm_plant(&mut sim, [0.0; 3], [1.06, 0.95, 1.03, 0.97]);
	let _ = fly(&mut c, &mut sim, 400);
	let v = c.cascade_obs_last();
	assert!(v[0] != 0.0 && v[3] != 0.0, "a3 alone must still produce rate-I and d̂: {v:?}");
	let f = c.last_feature_vector_ref();
	assert_eq!(&f[f.len() - 3..], &resid_of(&c, &v)[..], "the tail must be r(six)");
	// all three: rate-I, d̂, r in that order.
	let mut c = obs_student3(0, 1, 3, true, true, true);
	let _ = fly(&mut c, &mut sim, 200);
	let (f, v) = (c.last_feature_vector_ref().to_vec(), c.cascade_obs_last());
	assert_eq!(&f[f.len() - 9..f.len() - 3], &v[..]);
	assert_eq!(&f[f.len() - 3..], &resid_of(&c, &v)[..]);
	// provenance and the plant checks
	assert!(!spec().provenance_line().contains("dhat_resid"));
	assert!(obs_spec3(false, false, true).provenance_line().contains("obs=dhat_resid"));
	let mut mixed = obs_spec3(false, true, true);
	mixed.obs_dhat_residual_plant = Some([ARM as f64 * 1.1, KD as f64, INERTIA[0] as f64, INERTIA[1] as f64, INERTIA[2] as f64]);
	assert!(mixed.validate().is_err(), "a2 and a3 must name the same airframe");
	assert!(crate::dagger_train::validate_train_regime(&obs_student3(0, 1, 6, false, false, true), &l4c_cfg(0.2, 4)).is_ok());
}

/// The rate loop absorbs roll/pitch asymmetry within ~1-2 s: r → 0 there while d̂
/// stays large. Yaw (authority 3.65 rad/s², τ ≈ 7 s) keeps a deficit at 6 s.
#[test]
fn residual_decays_as_the_rate_loop_absorbs()
{
	let mut c = obs_student3(0, 10, 8, false, false, true);
	let mut sim = cf21_sim(0.0);
	arm_plant(&mut sim, [0.0; 3], [1.06, 0.95, 1.03, 0.97]);
	let (rec, _) = fly(&mut c, &mut sim, 6000);
	let tail = |k: usize, f: &dyn Fn(&[f32; 6]) -> f64| rec[rec.len() - k..].iter().map(f).sum::<f64>() / k as f64;
	let scale = c.cascade_obs_ref().unwrap().residual().unwrap();
	for a in 0..2
	{
		let r = tail(300, &|v| resid_of(&c, v)[a] as f64);
		let d_frac = tail(300, &|v| (v[3 + a] * scale.inv_a[a]) as f64);
		eprintln!("[ctrl48] axis {a}: residual {r:.4} vs d̂/authority {d_frac:.4}");
		assert!(d_frac.abs() > 0.02, "fixture: axis {a} disturbance too small");
		assert!(r.abs() < 0.1 * d_frac.abs(), "axis {a}: residual {r} not absorbed (d̂ share {d_frac})");
	}
	let ry = tail(300, &|v| resid_of(&c, v)[2] as f64);
	let dy = tail(300, &|v| (v[5] * scale.inv_a[2]) as f64);
	eprintln!("[ctrl48] yaw: residual {ry:.4} vs d̂/authority {dy:.4}");
	assert!(ry.abs() > 0.2 * dy.abs(), "yaw keeps an unabsorbed deficit at 6 s");
}

/// TRAIN = DEPLOY: the replay rebuilds the residual from the recorded six.
#[test]
fn residual_replay_reapplies_the_recorded_stream()
{
	let cfg = asym_cfg([1.06, 0.95, 1.03, 0.97], 0.20);
	let af = AirframeRs::from_cfg(&cfg);
	let (mut teacher, mut sim) = (af.teacher(cfg.teacher), af.sim());
	let mut student = obs_student3(0, 10, 9, false, false, true);
	student.set_observer_motor_lag_core(sim.motor_lag());
	let mut rng = SmallRng::seed_from_u64(48);
	let mut short = cfg.clone();
	short.steps_per_episode = 400;
	let tr = rollout_and_label_rs(&mut student, &mut teacher, &mut sim, &short, 0.05, &mut rng, [0.0; 3]);
	let n = tr.steps;
	let ro = ReplayObs::slice(&tr.vert_obs, &tr.horiz_obs, &tr.pwm_dev, &tr.att_up, 0, n)
		.expect("obs")
		.with_cascade(&tr.cascade_obs, 0, n);
	let mut rep = obs_student3(0, 10, 9, false, false, true);
	rep.replay_obs_guard(Some(&ro), n, "test");
	rep.reset(tr.init_yaw);
	for t in 0..n
	{
		rep.apply_replay_obs(Some(&ro), t);
		let f = rep.compute_features(tr.gyros[t], tr.accels[t], tr.targets[t]);
		assert_eq!(&f[f.len() - 3..], &resid_of(&rep, &tr.cascade_obs[t])[..], "step {t}: replayed residual");
	}
	assert!(tr.cascade_obs.iter().any(|v| v[0] != 0.0 && v[3] != 0.0), "stream is trivially zero");
}

/// CPU scorer vs Metal kernel on trained students: a3 alone and with a1 + a2.
#[test]
fn residual_cpu_scorer_matches_gpu()
{
	for (ri, dh) in [(false, false), (true, true)]
	{
		for (ar, lag) in [(1usize, 0.0f32), (10, 0.20)]
		{
			let mut c = obs_student3(0, ar, 0x4811 + ar as u64, ri, dh, true);
			let st = dagger_train_inplace_with(&mut c, &l4c_cfg(lag, 4), [0.0; 3], 23, false);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "trainer wrote nothing");
			let cpu = cpu_rows(&c, 800, lag);
			let Some(gpu) = gpu_rows(&c, 800, lag)
			else
			{
				return;
			};
			eprintln!("[ctrl48] resid rate_i={ri} dhat={dh} ar={ar} T={lag}: err cpu {:.6} gpu {:.6}", cpu[1], gpu[1]);
			for (i, name) in [(1usize, "err"), (0, "reward"), (2, "stable"), (3, "jerk"), (13, "pos"), (14, "alt")]
			{
				let tol = 2e-3 * cpu[i].abs().max(1e-3);
				assert!((gpu[i] - cpu[i]).abs() <= tol, "resid ri={ri} dh={dh} ar={ar} T={lag} {name}: gpu {} cpu {}", gpu[i], cpu[i]);
			}
		}
	}
}

/// The Metal split record/train rebuilds the residual from cas_rec like the CPU split.
#[test]
fn residual_gpu_split_matches_cpu()
{
	for (ri, dh) in [(false, false), (true, true)]
	{
		let cfg = zero_yaw(l4c_cfg(0.20, 4), 0.20);
		let mut c = obs_student3(4, 1, 0x4802, ri, dh, true);
		let trs = rollouts(&mut c, &cfg, 3);
		let fresh = obs_student3(4, 1, 0x4802, ri, dh, true);
		let blank = |o: &mut ReplayObs| o.cascade.iter_mut().for_each(|v| *v = [0.0; 6]);
		check(&format!("residual rate_i={ri} dhat={dh}"), parity(&fresh, &trs, blank));
	}
}
