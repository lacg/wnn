//! CTRL-80 (ABI 64, `--obs-yaw-lead`, Luiz 26/11/2026): the yaw LEAD input
//! yaw_lead = yaw_err − τ·gyro_z (yaw_lead.rs), one feature directly after the yaw-anchor
//! pair (docs/ctrl80_yaw_lead_prereg.md).
//!
//! Pinned here: refusals and default off; the layout (the new slot, every later input
//! moved up by one, every earlier input and every moved input BIT-identical to the off
//! twin, the ladder = the off ladder with the lead rows spliced in) and the value (the
//! definition, bit for bit); the Python constructor lifts the rows out of the MIDDLE of
//! the ladder and `None` is the ABI-63 controller; replay == live; CPU scorer == the
//! FC_YAW_LEAD Metal rollout (cascade student, and a plain student where the lead ALONE
//! selects the specialization); the specialised Metal split record/train == the CPU
//! split; the whole trainer's GPU walk == its CPU walk (QUAD_WEIGHTED and BINARY).
//! Flag off = every pre-existing pin of this crate (they construct without the option).

use crate::abi38_pins::{cpu_rows, gpu_rows};
use crate::abi38_tests::l4c_cfg;
use crate::controller::{ReplayObs, WnnController};
use crate::ctrl51_yaw_trim_tests::{yaw_student, YawOpt, YAW_ERR, YAW_I};
use ram_core::neuron_memory::{BINARY, QUAD_WEIGHTED};

/// The lead slot on yaw_student's layout: directly after yaw_err (13) / yaw_err_i (14).
const LEAD: usize = YAW_I + 1;
const TAU: f32 = 0.10;
/// A second τ on the SAME layout — the non-vacuity twin (the off twin has another layout,
/// so "differs from off" would prove nothing about the value).
const TAU_ALT: f32 = 0.35;
/// Fine cuts (rad) so the lead's bits toggle inside the fixture's small yaw errors.
const LEAD_ROWS: [f32; 3] = [-0.006, 0.0, 0.006];

fn student(sn: usize, ar: usize, seed: u64, tau: Option<f32>, mode: u8) -> WnnController
{
	let mut c = yaw_student(sn, ar, seed, YawOpt::Off, mode);
	if let Some(t) = tau
	{
		c.enable_yaw_lead(t, LEAD_ROWS.to_vec()).expect("lead on");
	}
	c
}

/// yaw_student's layout without the cascade (decouple banks): the lead is then the ONLY
/// reason the rollout leaves the default Metal pipeline.
fn plain_student(ar: usize, seed: u64, tau: Option<f32>) -> WnnController
{
	use rand::{Rng, SeedableRng};
	let (levels, bpf, window, obpn, nf) = (8usize, 3usize, 2usize, 10usize, 22usize);
	let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
	let thresholds: Vec<f32> = (0..nf * bpf)
		.map(|i| match (i / bpf, i % bpf)
		{
			(YAW_ERR, k) => [-0.006f32, 0.0, 0.006][k],
			(YAW_I, k) => [-0.3f32, 0.0, 0.3][k],
			_ => rng.gen_range(-0.4f32..0.4),
		})
		.collect();
	// Half the output bits on yaw_err / yaw_err_i / the lead slot, the rest anywhere.
	let oconn: Vec<i64> = (0..4 * levels * obpn)
		.map(|k| if k % 2 == 0 { ((YAW_ERR + (k / 2) % 3) * bpf + rng.gen_range(0..bpf)) as i64 } else { rng.gen_range(0..nf * bpf) as i64 })
		.collect();
	let mut c = WnnController::new_core(
		4, levels, bpf, window, 0, 0, obpn, thresholds, Vec::new(), oconn,
		true, 0.1, 0.95, 1.0,
		false, false, true, true, false, false, true, true,
		0.99, 1.0, crate::abi38_pins::DT, true, ar,
		BINARY, Some(crate::cell_mode::DECODE_ANTAGONIST), None, 0.05, false, 0.30,
		true, true, true, true, true, false, 1,
	)
	.expect("plain student must construct");
	if let Some(t) = tau
	{
		c.enable_yaw_lead(t, LEAD_ROWS.to_vec()).expect("lead on");
	}
	c
}

fn drive(t: usize) -> ([f32; 3], [f32; 3], [f32; 3])
{
	let s = t as f32;
	let gyro = [0.02 * (0.01 * s).sin(), -0.015 * (0.013 * s).cos(), 0.25 * (0.03 * s).sin() + 0.002];
	let accel = [0.3 * (0.004 * s).sin(), -0.2 * (0.005 * s).cos(), 9.81];
	(gyro, accel, [0.0, 0.0, 0.003])
}

#[test]
fn lead_refusals_and_default_off()
{
	let mut c = student(0, 10, 1, None, BINARY);
	let nf = c.num_features();
	assert_eq!((c.yaw_lead_tau(), c.obs_yaw_lead()), (None, None), "default off");
	for bad in [0.0f32, -0.1, 1.5, f32::NAN]
	{
		assert!(c.enable_yaw_lead(bad, LEAD_ROWS.to_vec()).is_err(), "tau {bad} accepted");
	}
	assert!(c.enable_yaw_lead(TAU, vec![0.0; 2]).is_err(), "a short ladder row accepted");
	assert_eq!(c.num_features(), nf, "a refused enable changed the layout");
	c.enable_yaw_lead(TAU, LEAD_ROWS.to_vec()).unwrap();
	assert_eq!((c.obs_yaw_lead(), c.num_features()), (Some(TAU), nf + 1));
	assert!(c.enable_yaw_lead(TAU, LEAD_ROWS.to_vec()).is_err(), "enabled twice");
	// No proportional yaw input: a motor-path student without the yaw channel.
	let mut bare = WnnController::new_core(
		4, 4, 3, 2, 0, 0, 8, vec![0.0; (9 + 2) * 3], Vec::new(), vec![0; 16 * 8],
		true, 0.05, 0.98, 1.0,
		false, false, true, false, false, false, false, false,
		0.99, 1.0, 0.001, false, 1, BINARY, None, None, 0.05, false, 0.30,
		false, false, false, false, false, false, 1,
	)
	.expect("bare student");
	assert!(bare.enable_yaw_lead(TAU, LEAD_ROWS.to_vec()).is_err(), "needs obs_yaw_err");
}

/// The lead slot carries yaw_err − τ·gyro_z bit for bit; every input before it is the off
/// twin's, every input after it is the off twin's previous slot (bit-identical); the ladder
/// is the off ladder with the lead rows spliced in at the slot.
#[test]
fn lead_layout_and_definition()
{
	let (mut on, mut off) = (student(0, 1, 7, Some(TAU), BINARY), student(0, 1, 7, None, BINARY));
	let (th, tho, bpf) = (on.thresholds_ref().to_vec(), off.thresholds_ref().to_vec(), LEAD_ROWS.len());
	assert_eq!(th[..LEAD * bpf], tho[..LEAD * bpf]);
	assert_eq!(th[LEAD * bpf..(LEAD + 1) * bpf], LEAD_ROWS);
	assert_eq!(th[(LEAD + 1) * bpf..], tho[LEAD * bpf..]);
	for c in [&mut on, &mut off]
	{
		c.reset(0.01);
	}
	let (mut span, mut led) = (0.0f32, 0usize);
	for t in 0..600
	{
		let (gyro, accel, tg) = drive(t);
		let (f, fo) = (on.compute_features(gyro, accel, tg), off.compute_features(gyro, accel, tg));
		assert_eq!(f.len(), fo.len() + 1);
		for i in 0..LEAD
		{
			assert_eq!(f[i].to_bits(), fo[i].to_bits(), "step {t}: input {i} (before the lead) moved");
		}
		for i in LEAD + 1..f.len()
		{
			assert_eq!(f[i].to_bits(), fo[i - 1].to_bits(), "step {t}: input {i} is not the off twin's {}", i - 1);
		}
		assert_eq!(f[LEAD].to_bits(), (fo[YAW_ERR] - TAU * gyro[2]).to_bits(), "step {t}: lead {} vs yaw_err − τ·gyro_z", f[LEAD]);
		span = span.max((f[LEAD] - f[YAW_ERR]).abs());
		// It LEADS: closer to the yaw_err 100 steps (τ at dt = 1 ms) ahead than yaw_err is
		// — checked below on the recorded series.
		led += (f[LEAD] != f[YAW_ERR]) as usize;
	}
	assert!(span > 0.01 && led > 500, "vacuous fixture (lead − yaw_err span {span}, differing steps {led})");
}

/// On a heading driven by gyro_z alone the lead predicts yaw_err τ ahead: its error vs
/// yaw_err(t + τ) is far below yaw_err's own, and the wrong sign's is far above.
#[test]
fn lead_predicts_yaw_err_tau_ahead()
{
	let mut c = student(0, 1, 9, Some(TAU), BINARY);
	c.reset(0.0);
	let dt = crate::abi38_pins::DT;
	let h = (TAU / dt).round() as usize;
	let (mut ye, mut ld, mut gz) = (Vec::new(), Vec::new(), Vec::new());
	for t in 0..(1500 + h)
	{
		let g = 0.4 * (0.004 * t as f32).sin();
		let f = c.compute_features([0.0, 0.0, g], [0.0, 0.0, 9.81], [0.0; 3]);
		ye.push(f[YAW_ERR]);
		ld.push(f[LEAD]);
		gz.push(g);
	}
	let rms = |p: &dyn Fn(usize) -> f32| ((0..1500).map(|t| (p(t) - ye[t + h]).powi(2)).sum::<f32>() / 1500.0).sqrt();
	let (e_lead, e_hold, e_wrong) = (rms(&|t| ld[t]), rms(&|t| ye[t]), rms(&|t| ye[t] + TAU * gz[t]));
	eprintln!("[ctrl80] predicting yaw_err(t+τ): lead {e_lead:.5} | yaw_err {e_hold:.5} | wrong sign {e_wrong:.5} rad");
	assert!(e_lead < 0.25 * e_hold && e_wrong > 1.5 * e_hold, "the lead does not lead (lead {e_lead}, hold {e_hold}, wrong sign {e_wrong})");
}

const CT_NF: usize = 15; // 9 base + roll/pitch err + yaw_err + yaw_err_i + alt_err + vz
const CT_SLOT: usize = 13;

/// The Python constructor on a motor-path layout with inputs BEFORE and AFTER the lead.
fn via_ctor(thresholds: Vec<f32>, oconn: Vec<i64>, lead: Option<f32>) -> pyo3::PyResult<WnnController>
{
	WnnController::new(
		4, 4, 3, 2, 0, 0, 8, thresholds, Vec::new(), oconn,
		true, 0.05, 0.98, 1.0,
		false, false, true, false, false, false, true, true,
		0.99, 1.0, 0.001, false, 1, QUAD_WEIGHTED, None, None, 0.05, false, 0.30,
		false, true, true, false, false, false, 1,
		1.0, false,
		false, 2.0, 0.1,
		None,
		false, None,
		None,
		false, // obs_yaw_trim_err
		false, // obs_yaw_trim_err_p
		false, // obs_yaw_from_att_est
		None, // delta_max_yaw
		false, // yaw_trim_ff
		None, // delta_leak_yaw
		lead, // obs_yaw_lead (ABI 64)
	)
}

/// The constructor takes the FULL ladder in layout order (lead rows in the middle) and
/// builds the controller new + enable builds; `None` is the ABI-63 controller (same
/// ladder, same features as a build that never heard of the option).
#[test]
fn lead_ctor_lifts_the_rows_from_the_middle()
{
	use rand::{Rng, SeedableRng};
	let mut rng = rand::rngs::SmallRng::seed_from_u64(0x80);
	let base: Vec<f32> = (0..CT_NF * 3).map(|_| rng.gen_range(-0.5f32..0.5)).collect();
	let oconn: Vec<i64> = (0..16 * 8).map(|_| rng.gen_range(0..CT_NF * 3) as i64).collect();
	let mut full = base.clone();
	full.splice(CT_SLOT * 3..CT_SLOT * 3, LEAD_ROWS);
	let mut off = via_ctor(base.clone(), oconn.clone(), None).expect("off");
	let mut on = via_ctor(full.clone(), oconn.clone(), Some(TAU)).expect("on");
	assert_eq!((off.num_features(), on.num_features()), (CT_NF, CT_NF + 1));
	assert_eq!((off.obs_yaw_lead(), on.obs_yaw_lead()), (None, Some(TAU)));
	assert_eq!(off.thresholds_ref(), &base[..], "flag off changed the ladder");
	assert_eq!(on.thresholds_ref(), &full[..], "the rows did not go back where they were lifted from");
	let mut manual = via_ctor(base.clone(), oconn.clone(), None).unwrap();
	manual.enable_yaw_lead(TAU, LEAD_ROWS.to_vec()).unwrap();
	// A ladder of the wrong length is refused either way round.
	assert!(via_ctor(base.clone(), oconn.clone(), Some(TAU)).is_err(), "lead on with the lead-less ladder accepted");
	assert!(via_ctor(full.clone(), oconn.clone(), None).is_err(), "lead off with the lead ladder accepted");
	for c in [&mut off, &mut on, &mut manual]
	{
		c.reset(0.02);
		c.set_vertical_obs(0.5, 0.03, -0.02);
	}
	for t in 0..200
	{
		let (gyro, accel, tg) = drive(t);
		let (f, fm, fo) = (on.compute_features(gyro, accel, tg), manual.compute_features(gyro, accel, tg), off.compute_features(gyro, accel, tg));
		let b = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
		assert_eq!(b(&f), b(&fm), "step {t}: constructor != new + enable");
		assert_eq!(f[CT_SLOT].to_bits(), (fo[CT_SLOT - 2] - TAU * gyro[2]).to_bits());
		assert_eq!(b(&f[CT_SLOT + 1..]), b(&fo[CT_SLOT..]), "step {t}: the inputs after the lead are not the off twin's");
		assert!(fo[CT_SLOT] != 0.0 && fo[CT_SLOT + 1] != 0.0, "vacuous: the vertical inputs after the lead are zero");
	}
}

/// TRAIN = DEPLOY: the replay rebuilds every frame (the lead included) from the streams it
/// already re-applies — the lead needs no stream of its own.
#[test]
fn lead_replay_equals_live()
{
	use crate::dagger_train::{rollout_and_label_inner, AirframeRs};
	use rand::SeedableRng;
	let mut cfg = l4c_cfg(0.20, 4);
	cfg.steps_per_episode = 400;
	let af = AirframeRs::from_cfg(&cfg);
	let (mut teacher, mut sim) = (af.teacher(cfg.teacher), af.sim());
	let mut c = student(0, 10, 3, Some(TAU), BINARY);
	c.set_observer_motor_lag_core(sim.motor_lag());
	let mut rng = rand::rngs::SmallRng::seed_from_u64(80);
	let mut cap = crate::imitation_probe::ImitCapture::feature_sampler();
	let tr = rollout_and_label_inner(&mut c, &mut teacher, &mut sim, &cfg, 0.1, &mut rng, [0.0; 3], Some(&mut cap));
	let n = tr.steps;
	assert!(n >= 50 && cap.feats.len() == n);
	let ro = ReplayObs::slice(&tr.vert_obs, &tr.horiz_obs, &tr.pwm_dev, &tr.att_up, 0, n)
		.expect("obs")
		.with_cascade(&tr.cascade_obs, 0, n);
	let mut rep = student(0, 10, 3, Some(TAU), BINARY);
	rep.replay_obs_guard(Some(&ro), n, "test");
	rep.reset(tr.init_yaw);
	let mut moved = 0usize;
	for t in 0..n
	{
		rep.apply_replay_obs(Some(&ro), t);
		let f = rep.compute_features(tr.gyros[t], tr.accels[t], tr.targets[t]);
		let (a, b): (Vec<u32>, Vec<u32>) = (f.iter().map(|v| v.to_bits()).collect(), cap.feats[t].iter().map(|v| v.to_bits()).collect());
		assert_eq!(a, b, "step {t}: the replayed frame is not the live one");
		assert_eq!(f[LEAD].to_bits(), (f[YAW_ERR] - TAU * tr.gyros[t][2]).to_bits());
		moved += (f[LEAD] != f[YAW_ERR]) as usize;
	}
	assert!(moved > n / 2, "the lead barely differs from yaw_err ({moved}/{n})");
}

fn assert_gpu_close(tag: &str, cpu: &[f64], gpu: &[f64])
{
	for (i, name) in [(1usize, "err"), (0, "reward"), (2, "stable"), (3, "jerk"), (13, "pos"), (14, "alt")]
	{
		let tol = 2e-3 * cpu[i].abs().max(1e-3);
		assert!((gpu[i] - cpu[i]).abs() <= tol, "{tag} {name}: gpu {} cpu {}", gpu[i], cpu[i]);
	}
}

fn moved(a: &[f64], b: &[f64]) -> bool
{
	[0usize, 1, 13].iter().any(|&i| (a[i] - b[i]).abs() > 1e-9)
}

/// CPU scorer == the FC_YAW_LEAD Metal rollout on the cascade student, trained, both
/// holds; τ moves the result on BOTH sides (the GPU reads the lead, with its τ).
#[cfg(target_os = "macos")]
#[test]
fn lead_score_cpu_matches_gpu()
{
	use crate::dagger_train::dagger_train_inplace_with;
	for (ar, lag) in [(1usize, 0.0f32), (10, 0.20)]
	{
		let train = |tau: f32| {
			let mut c = student(0, ar, 0x8021 + ar as u64, Some(tau), BINARY);
			let st = dagger_train_inplace_with(&mut c, &l4c_cfg(lag, 4), [0.0; 3], 23, false);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "trainer wrote nothing");
			c
		};
		let (on, alt) = (train(TAU), train(TAU_ALT));
		let (cpu, cpu_alt) = (cpu_rows(&on, 800, lag), cpu_rows(&alt, 800, lag));
		assert!(moved(&cpu, &cpu_alt), "ar {ar}: τ did not move the CPU result (the lead is not read)");
		let Some(gpu) = gpu_rows(&on, 800, lag)
		else
		{
			return;
		};
		let gpu_alt = gpu_rows(&alt, 800, lag).unwrap();
		eprintln!("[ctrl80] cascade ar={ar} T={lag}: err cpu {:.6} gpu {:.6} (τ {TAU_ALT}: {:.6} / {:.6})", cpu[1], gpu[1], cpu_alt[1], gpu_alt[1]);
		assert_gpu_close(&format!("ar={ar} T={lag}"), &cpu, &gpu);
		assert_gpu_close(&format!("ar={ar} T={lag} τ={TAU_ALT}"), &cpu_alt, &gpu_alt);
		assert!(moved(&gpu, &gpu_alt), "ar {ar}: the GPU ignored obs_yaw_lead's τ");
	}
}

/// The lead ALONE (no cascade blocks, no bias predictor): the rollout must still leave the
/// default pipeline for the FC_YAW_LEAD specialization — else the kernel would never write
/// the slot and every later input would sit one slot early.
#[cfg(target_os = "macos")]
#[test]
fn lead_alone_score_cpu_matches_gpu()
{
	let lag = 0.20f32;
	let train = |tau: Option<f32>| {
		let mut c = plain_student(10, 0x80d1, tau);
		let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &l4c_cfg(lag, 4), [0.0; 3], 23, false);
		assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "vacuous plain trainer run");
		c
	};
	let (on, alt, off) = (train(Some(TAU)), train(Some(TAU_ALT)), train(None));
	let (cpu, cpu_alt, cpu_off) = (cpu_rows(&on, 800, lag), cpu_rows(&alt, 800, lag), cpu_rows(&off, 800, lag));
	assert!(moved(&cpu, &cpu_alt), "τ did not move the CPU result");
	let Some(gpu) = gpu_rows(&on, 800, lag)
	else
	{
		return;
	};
	let (gpu_alt, gpu_off) = (gpu_rows(&alt, 800, lag).unwrap(), gpu_rows(&off, 800, lag).unwrap());
	eprintln!("[ctrl80] plain: err cpu {:.6} gpu {:.6} | τ {TAU_ALT}: {:.6} / {:.6} | off {:.6} / {:.6}", cpu[1], gpu[1], cpu_alt[1], gpu_alt[1], cpu_off[1], gpu_off[1]);
	assert_gpu_close("plain", &cpu, &gpu);
	assert_gpu_close("plain alt", &cpu_alt, &gpu_alt);
	assert_gpu_close("plain off", &cpu_off, &gpu_off);
	assert!(moved(&gpu, &gpu_alt), "the GPU ignored obs_yaw_lead's τ on the plain student");
}

/// One RolloutParams per launch: a batch that mixes τ (or on/off) is refused by the
/// pymethod; the evaluator itself is given a uniform population here and must agree
/// with the single-controller dispatch.
#[cfg(target_os = "macos")]
#[test]
fn lead_population_is_one_lineage()
{
	let (a, b) = (student(0, 10, 0x8031, Some(TAU), BINARY), student(0, 10, 0x8032, Some(TAU), BINARY));
	let Some(many) = crate::abi38_pins::gpu_rows_many(&[&a, &b], 400, 0.2)
	else
	{
		return;
	};
	let (one_a, one_b) = (gpu_rows(&a, 400, 0.2).unwrap(), gpu_rows(&b, 400, 0.2).unwrap());
	let bits = |r: &[f64]| r.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
	assert_eq!(bits(&many[0]), bits(&one_a));
	assert_eq!(bits(&many[1]), bits(&one_b));
}

/// The FC_YAW_LEAD controller_record / controller_train == the CPU split replays; another
/// τ records different rows on the SAME streams (the kernels read the lead, with its τ).
#[cfg(target_os = "macos")]
#[test]
fn lead_gpu_split_matches_cpu()
{
	let cfg = crate::ctrl40_tests::zero_yaw(l4c_cfg(0.20, 4), 0.20);
	let mut c = student(4, 1, 0x8002, Some(TAU), BINARY);
	let trs = crate::ctrl40_tests::rollouts(&mut c, &cfg, 3);
	let fresh = student(4, 1, 0x8002, Some(TAU), BINARY);
	crate::ctrl40_tests::check("yaw lead", crate::ctrl40_tests::parity(&fresh, &trs, |o| o.cascade.iter_mut().for_each(|v| *v = [0.0; 6])));
	let alt = student(4, 1, 0x8002, Some(TAU_ALT), BINARY);
	crate::ctrl40_tests::check("yaw lead (second τ)", crate::ctrl40_tests::parity(&alt, &trs, |o| o.cascade.iter_mut().for_each(|v| *v = [0.0; 6])));
	let obs: Vec<ReplayObs> = trs.iter().map(crate::ctrl40_tests::replay_obs).collect();
	let (g, a, t, p) = crate::ctrl40_tests::streams(&trs);
	let (rows_on, ..) = fresh.clone().split_record_pub(g.clone(), a.clone(), t.clone(), p.clone(), Some(&obs));
	let (rows_alt, ..) = alt.clone().split_record_pub(g, a, t, p, Some(&obs));
	assert_ne!(rows_on, rows_alt, "the split replay records the same rows whatever τ is");
}

/// The whole DAgger trainer: GPU walk == CPU walk with the lead on, both memory modes,
/// and τ reaches the trainer.
#[cfg(target_os = "macos")]
#[test]
fn lead_trainer_cpu_matches_gpu_walk()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
	{
		return;
	}
	for mode in [QUAD_WEIGHTED, BINARY]
	{
		let cfg = l4c_cfg(0.20, 4);
		let run = |tau: f32, gpu: bool| {
			let mut c = student(4, 1, 0x8041, Some(tau), mode);
			let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 51, gpu);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "vacuous trainer run");
			let ex = c.export_cells();
			(crate::abi38_pins::cells_and_stats_digest(&c, Some(&st)), crate::abi35_tests::cells_order_digest(&ex.0, &ex.1))
		};
		let before = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
		let gpu = run(TAU, true);
		assert!(crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed) > before, "mode {mode}: GPU walk never ran");
		assert_eq!(run(TAU, false), gpu, "mode {mode}: GPU walk diverged from the CPU walk with obs_yaw_lead");
		assert_ne!(run(TAU_ALT, false), gpu, "mode {mode}: obs_yaw_lead's τ never reached the trainer");
	}
}
