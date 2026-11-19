//! ABI 60 (docs/ctrl58_a2_prereg.md, Amendment 19/11): scorer row 19 = `mid`, the mean attitude
//! err (rad) over mid_window_start(steps) <= t < tail_start, per episode; an episode that ended
//! before the hold window contributes its whole-episode mean (the fallback `steady` uses).
//!
//! Pinned: the window's start is ceil(0.15·steps) exactly (integer arithmetic, every step
//! count); row 19 equals a direct recomputation from the per-step err trace; on completed
//! episodes 0.15·first + 0.65·mid + 0.20·steady == err; a diverged-before-the-hold episode
//! (which DID enter the mid window) carries its whole-episode mean; the trace sink leaves the
//! row untouched; CPU == Metal on row 19, with and without divergences in the batch.

use crate::abi38_pins::{gpu_rows, ics, stage2, ARM, ASYM, DT, EPS, G, INERTIA, KD, KT};
use crate::controller::WnnController;
use crate::cpu_score::{mid_episode_mean, mid_window_start, ROW_METRICS};
use crate::ctrl51_yaw_trim_tests::{yaw_student, YawOpt};
use ram_core::neuron_memory::BINARY;

const MID: usize = 19;

fn tail_start(steps: usize) -> usize
{
	((steps as f64) * 0.80).ceil() as usize
}

/// The abi38 stage-2 scoring call (cpu_rows) with the per-step err trace.
fn cpu_traced(c: &WnnController, steps: usize, lag: f32, trace: Option<&mut Vec<Vec<f64>>>) -> [f64; ROW_METRICS]
{
	let (q0, w0) = ics();
	let s1 = stage2();
	let mut cc = c.clone();
	crate::cpu_score::rollout_one(
		&mut cc, &q0, &w0, EPS, steps, DT, ARM, KT, KD, INERTIA, G, lag, [0.0; 3],
		true, [0.0; 3], 0.0, 0.1, ASYM, 0.0107, 0.0, 0.0, 777,
		0.0, 0, 0, 0.2, c.levels_per_motor(), 4, None, None, None, 1.0, 0.4, trace, Some(&s1), None, None,
	)
}

fn trained() -> WnnController
{
	let mut c = yaw_student(0, 10, 0x6019, YawOpt::Off, BINARY);
	let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &crate::abi38_tests::l4c_cfg(0.20, 4), [0.0; 3], 23, false);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0);
	c
}

fn mean(v: &[f64]) -> f64
{
	v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// One episode's `mid` from its err trace, by the prereg's words.
fn mid_of_trace(ep: &[f64], steps: usize) -> f64
{
	let (a, b) = (mid_window_start(steps), tail_start(steps));
	if ep.len() > b && b > a { mean(&ep[a..b]) } else { mean(ep) }
}

#[test]
fn mid_window_start_is_ceil_of_15_percent()
{
	assert_eq!(mid_window_start(2000), 300, "0.3 s at 2000 steps of 1 ms");
	assert_eq!((mid_window_start(800), mid_window_start(400), mid_window_start(1), mid_window_start(0)), (120, 60, 1, 0));
	for steps in 0..6000usize
	{
		assert_eq!(mid_window_start(steps), (3 * steps).div_ceil(20), "steps {steps}");
		assert!(mid_window_start(steps) <= tail_start(steps), "steps {steps}: the mid window starts after the hold window");
	}
	// The fallback rule in isolation: complete window; ended before the hold; never ran.
	assert_eq!(mid_episode_mean(6.0, 3, 1, 100.0, 10), 2.0);
	assert_eq!(mid_episode_mean(6.0, 3, 0, 100.0, 10), 10.0);
	assert_eq!(mid_episode_mean(0.0, 0, 0, 0.0, 0), 0.0);
}

/// Row 19 is the mid-window mean of the very errs the scorer averages, and on completed
/// episodes the three windows recompose err: 0.15·first + 0.65·mid + 0.20·steady == err.
#[test]
fn mid_row_recomputes_from_the_trace_and_recomposes_err()
{
	let c = trained();
	for steps in [800usize, 2000]
	{
		let mut trace = Vec::new();
		let row = cpu_traced(&c, steps, 0.2, Some(&mut trace));
		let plain = cpu_traced(&c, steps, 0.2, None);
		assert_eq!(row.map(f64::to_bits), plain.map(f64::to_bits), "the trace sink moved the row");
		assert_eq!(trace.len(), EPS);
		assert!(trace.iter().all(|e| e.len() == steps), "fixture: every episode must complete ({:?})", trace.iter().map(Vec::len).collect::<Vec<_>>());
		let want = mean(&trace.iter().map(|e| mid_of_trace(e, steps)).collect::<Vec<_>>());
		assert!((row[MID] - want).abs() <= 1e-12 * want.abs(), "steps {steps}: row 19 {} vs the trace's mid mean {want}", row[MID]);
		let first = mean(&trace.iter().map(|e| mean(&e[..mid_window_start(steps)])).collect::<Vec<_>>());
		let recomposed = 0.15 * first + 0.65 * row[MID] + 0.20 * row[5];
		eprintln!("[mid] steps {steps}: err {:.6} = 0.15*{first:.6} + 0.65*{:.6} + 0.20*{:.6} (rad)", row[1], row[MID], row[5]);
		assert!((recomposed - row[1]).abs() <= 1e-9 * row[1].abs(), "steps {steps}: 0.15*first + 0.65*mid + 0.20*steady = {recomposed} vs err {}", row[1]);
		assert!((row[MID] - row[1]).abs() > 1e-9 && (row[MID] - row[5]).abs() > 1e-9, "vacuous: mid equals err or steady");
	}
}

/// The diverging fixture of ctrl51_hold_tests: episodes 0-1 hold, 2-3 diverge at steps ~255 /
/// ~267 of 400 — INSIDE the mid window (60..320) and before the hold window. They carry their
/// whole-episode mean, not the partial mid-window mean; CPU == Metal with them in the batch.
#[cfg(target_os = "macos")]
#[test]
fn mid_row_diverged_episode_falls_back_to_its_whole_mean_cpu_match_gpu()
{
	use crate::abi32_tests::{cfg, controller, ics as ics32, tilt_ref, ARM as A, DT as D, EPS as E32, G as GR, INERTIA as IN, KD as KDR, KT as K};
	use crate::ctrl51_hold_tests::DIVERGE_ASYM;
	let steps = 400usize;
	let c = controller(7, true);
	let mut s1 = cfg(crate::plant_opts::PlantOpts::new(true, 0.78).unwrap(), tilt_ref());
	s1.collective_frac[3] = 0.6;
	s1.validate(E32).unwrap();
	let (q0, w0) = ics32();
	let mut trace = Vec::new();
	let mut cc = c.clone();
	let row = crate::cpu_score::rollout_one(
		&mut cc, &q0, &w0, E32, steps, D, A, K, KDR, IN, GR, 0.0, [0.0; 3],
		true, [0.0; 3], 0.0, 0.1, DIVERGE_ASYM, 0.0, 0.0, 0.0, 5,
		0.0, 0, 0, 0.0, c.levels_per_motor(), 4, None, None, None, 1.0, 0.4, Some(&mut trace), Some(&s1), None, None,
	);
	let (a, b) = (mid_window_start(steps), tail_start(steps));
	let early: Vec<usize> = (0..E32).filter(|&e| trace[e].len() <= b).collect();
	assert_eq!(early, vec![2, 3], "fixture: episodes 2-3 end before the hold window ({:?})", trace.iter().map(Vec::len).collect::<Vec<_>>());
	assert!(early.iter().all(|&e| trace[e].len() > a + 20), "fixture: the diverged episodes must have ENTERED the mid window");
	let want = mean(&trace.iter().map(|e| mid_of_trace(e, steps)).collect::<Vec<_>>());
	assert!((row[MID] - want).abs() <= 1e-12 * want.abs(), "row 19 {} vs whole-mean fallback {want}", row[MID]);
	let partial = mean(&trace.iter().map(|e| mean(&e[a..e.len().min(b)])).collect::<Vec<_>>());
	assert!((partial - want).abs() > 1e-6 * want.abs(), "vacuous: the partial mid-window mean equals the fallback");
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok()
	{
		return;
	}
	let Ok(ev) = crate::metal_controller::ControllerRolloutEvaluator::new()
	else
	{
		return;
	};
	let d = crate::controller::Disturbance {
		tau_bias: [0.0; 3],
		gust_sigma: 0.0,
		gust_tau_c: 0.1,
		motor_asym: DIVERGE_ASYM,
		gyro_sigma: 0.0,
		gyro_bias_walk: 0.0,
		accel_sigma: 0.0,
		dropout_prob: 0.0,
		dropout_len_steps: 0,
		obs_delay_steps: 0,
		torque_scale_jitter: 0.0,
		seed: 5,
	};
	let gpu = ev.score(&[&c], &q0, &w0, E32, steps, (D, A, K, IN, GR), KDR, 0.0, [0.0; 3], Some(d), None, None, None, Some(&s1)).expect("gpu score");
	eprintln!("[mid] diverging batch: mid cpu {:.6} gpu {:.6} (err {:.6} / {:.6})", row[MID], gpu[0][MID], row[1], gpu[0][1]);
	assert_eq!(gpu[0].len(), ROW_METRICS);
	assert!((gpu[0][2] - row[2]).abs() < 1e-12, "stable: gpu {} cpu {}", gpu[0][2], row[2]);
	let tol = 1e-4f64.max(5e-2 * row[MID].abs());
	assert!((gpu[0][MID] - row[MID]).abs() <= tol, "row 19 with divergences: gpu {} cpu {}", gpu[0][MID], row[MID]);
}

/// CPU == Metal on row 19 (and on the two rows it sits between) on the production-shaped
/// cascade student, at a step count where float ceil(0.15·steps) would have disagreed (800:
/// f32 gives 121, the exact value is 120) and at the production 2000.
#[cfg(target_os = "macos")]
#[test]
fn mid_row_cpu_matches_gpu()
{
	let c = trained();
	for steps in [800usize, 2000]
	{
		let cpu = cpu_traced(&c, steps, 0.2, None);
		let Some(gpu) = gpu_rows(&c, steps, 0.2)
		else
		{
			return;
		};
		eprintln!("[mid] steps {steps}: mid cpu {:.6} gpu {:.6}; err {:.6}/{:.6}; steady {:.6}/{:.6}", cpu[MID], gpu[MID], cpu[1], gpu[1], cpu[5], gpu[5]);
		assert_eq!(gpu.len(), ROW_METRICS);
		for (i, name) in [(1usize, "err"), (5, "steady"), (MID, "mid")]
		{
			let tol = 2e-3 * cpu[i].abs().max(1e-3);
			assert!((gpu[i] - cpu[i]).abs() <= tol, "steps {steps} {name}: gpu {} cpu {}", gpu[i], cpu[i]);
		}
	}
}
