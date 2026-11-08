//! DIAGNOSTIC (06/11/2026, CTRL-27 cascade): the CTBR half of imitation_probe.
//! Called from imitation_probe::run when the recipe controller is a cascade
//! (output_mode=ctbr). Everything here is read-only on the trained students; the
//! flights below fly the TEACHER over the student's own inner stage.
//!
//! Per student (BC / DAgger) it reports, in PHYSICAL bank units (collective pwm,
//! rate rad/s), on decision steps only (the records the sn=0 trainer writes):
//!   LABEL     — |label| quantiles per bank vs the BINARY decode grid's first step;
//!   IMITATION — RAM MAE and R² on held-out teacher states and on the student's own
//!               states; the perfect-memoriser floor (label → target bits → decode);
//!               the per-neuron MAJORITY oracle on the held-out addresses;
//!   REPRESENTATION — ridge R² of the label on the student's raw features,
//!               thermometer bits, and raw + each piece of the teacher's hidden plan
//!               state (MPCOF: measured ua, rate-loop integrator I, held force Fh, d̂);
//!   SHADOW    — MPCOF's shadow rate loop vs the student's REAL inner loop along the
//!               student's own states (the DAgger label is planned from the shadow).
//! Then CLOSED LOOP flights of the teacher's label through the student's inner
//! stage, transformed: exact, through the student's decode grid (a perfect
//! memoriser), and plus Gaussian noise the size of the measured RAM error.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::controller::WnnController;
use crate::dagger_train::RewardGatedConfigPacked;
use crate::imitation_probe::{raw_rows, ridge, thermo_rows, ImitCapture, Rig};

const BANK: [&str; 4] = ["T", "p", "q", "r"];

fn spec(c: &WnnController) -> &crate::ctbr::CtbrSpec
{
	&c.ctbr_ref().expect("imitation_probe_ctbr: a cascade controller").spec
}

/// A raw decode ∈ [0,1] in bank units.
pub(crate) fn phys(c: &WnnController, m: usize, dec: f32) -> f64
{
	spec(c).decode(m, dec, c.neutral_f32()) as f64
}

/// The label as the encoder sees it (clipped at ±range).
fn clipped(c: &WnnController, l: [f32; 4]) -> [f64; 4]
{
	let r = spec(c).range;
	std::array::from_fn(|m| l[m].clamp(-r[m], r[m]) as f64)
}

/// What a PERFECT memoriser emits for this label: target bits → cells → decode.
pub(crate) fn memoriser(c: &WnnController, l: [f32; 4]) -> [f64; 4]
{
	let (_d, bits) = c.diag_targets(l);
	let mode = c.memory_mode_u8();
	let (tc, fc) = (crate::cell_mode::true_cell(mode), crate::cell_mode::false_cell(mode));
	let lv = c.levels_per_motor();
	std::array::from_fn(|m| {
		let cells: Vec<u8> = bits[m * lv..(m + 1) * lv].iter().map(|&b| if b { tc } else { fc }).collect();
		phys(c, m, crate::cell_mode::decode_motor_cells(&cells, mode, c.output_decode_u8()))
	})
}

pub(crate) fn ys(c: &WnnController, cap: &ImitCapture) -> Vec<[f64; 4]>
{
	cap.label.iter().map(|&l| clipped(c, l)).collect()
}

fn ram(c: &WnnController, cap: &ImitCapture) -> Vec<[f64; 4]>
{
	cap.decoded.iter().map(|d| std::array::from_fn(|m| phys(c, m, d[m]))).collect()
}

/// Per bank: (MAE, RMSE, R²) of pred vs y.
pub(crate) fn score(pred: &[[f64; 4]], y: &[[f64; 4]]) -> [(f64, f64, f64); 4]
{
	let n = y.len().max(1) as f64;
	std::array::from_fn(|m| {
		let mu = y.iter().map(|v| v[m]).sum::<f64>() / n;
		let var = y.iter().map(|v| (v[m] - mu).powi(2)).sum::<f64>() / n;
		let (mut ae, mut se) = (0.0, 0.0);
		for (p, t) in pred.iter().zip(y)
		{
			ae += (p[m] - t[m]).abs();
			se += (p[m] - t[m]).powi(2);
		}
		(ae / n, (se / n).sqrt(), 1.0 - (se / n) / var.max(1e-30))
	})
}

fn fmt_r2(s: &[(f64, f64, f64); 4]) -> String
{
	(0..4).map(|m| format!("{}:{:.3}", BANK[m], s[m].2)).collect::<Vec<_>>().join(" ")
}

fn fmt_mae(s: &[(f64, f64, f64); 4]) -> String
{
	(0..4).map(|m| format!("{}:{:.4}", BANK[m], s[m].0)).collect::<Vec<_>>().join(" ")
}

/// Ridge on the physical labels (f32 rows the shared ridge takes), predicting unclamped.
pub(crate) fn ls(train_x: &[Vec<f64>], train_y: &[[f64; 4]], test_x: &[Vec<f64>]) -> Vec<[f64; 4]>
{
	let y32: Vec<[f32; 4]> = train_y.iter().map(|v| v.map(|x| x as f32)).collect();
	let w = ridge(train_x, &y32, 4);
	test_x
		.iter()
		.map(|x| std::array::from_fn(|m| x.iter().zip(&w[m]).map(|(a, b)| a * b).sum()))
		.collect()
}

/// Raw features plus a slice of the teacher's hidden plan state.
fn raw_plus(cap: &ImitCapture, pick: &[usize]) -> Vec<Vec<f64>>
{
	raw_rows(&cap.feats)
		.into_iter()
		.zip(&cap.hidden)
		.map(|(mut r, h)| {
			r.extend(pick.iter().map(|&i| h[i]));
			r
		})
		.collect()
}

/// The teacher's own inputs: Euler rpy, gyro, vertical obs + every hidden piece.
fn teacher_rows(cap: &ImitCapture) -> Vec<Vec<f64>>
{
	cap.tx
		.iter()
		.zip(&cap.hidden)
		.map(|(x, h)| {
			let mut r: Vec<f64> = x[..6].to_vec();
			r.extend_from_slice(&x[12..15]);
			r.extend_from_slice(h);
			r.push(1.0);
			r
		})
		.collect()
}

/// Per-neuron MAJORITY decode on the held-out set's own addresses (the aliasing
/// floor of ANY memoryless lookup on these address functions), in bank units, and
/// the conflict-cell rate.
fn oracle_phys(c: &WnnController, cap: &ImitCapture) -> (Vec<[f64; 4]>, f64)
{
	use std::collections::HashMap;
	let nn = cap.tbits[0].len();
	let mut maps: Vec<HashMap<u64, (u32, u32)>> = (0..nn).map(|_| HashMap::new()).collect();
	for (a, b) in cap.addrs.iter().zip(&cap.tbits)
	{
		for n in 0..nn
		{
			let e = maps[n].entry(a[n]).or_insert((0, 0));
			if b[n] { e.0 += 1 } else { e.1 += 1 }
		}
	}
	let (cells, conf) = maps.iter().flat_map(|m| m.values()).fold((0usize, 0usize), |(c0, c1), &(t, f)| (c0 + 1, c1 + (t > 0 && f > 0) as usize));
	let mode = c.memory_mode_u8();
	let (tc, fc) = (crate::cell_mode::true_cell(mode), crate::cell_mode::false_cell(mode));
	let lv = c.levels_per_motor();
	let pred = cap
		.addrs
		.iter()
		.map(|a| {
			let cellv: Vec<u8> = (0..nn).map(|n| { let (t, f) = maps[n][&a[n]]; if t >= f { tc } else { fc } }).collect();
			std::array::from_fn(|m| phys(c, m, crate::cell_mode::decode_motor_cells(&cellv[m * lv..(m + 1) * lv], mode, c.output_decode_u8())))
		})
		.collect();
	(pred, conf as f64 / cells.max(1) as f64)
}

fn quantiles(v: &mut [f64]) -> String
{
	v.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
	format!("p10 {:.3} p50 {:.3} p90 {:.3} p99 {:.3}", q(0.1), q(0.5), q(0.9), q(0.99))
}

/// LABEL vs DECODE GRID: the BINARY antagonist decode steps t by 2/levels, so the
/// first nonzero setpoint is range·(2/levels)^γ.
fn label_grid_line(tag: &str, c: &WnnController, cap: &ImitCapture)
{
	let s = spec(c);
	let lv = c.levels_per_motor() as f64;
	let parts: Vec<String> = (0..4)
		.map(|m| {
			let mut a: Vec<f64> = cap.label.iter().map(|l| (l[m] as f64).abs()).collect();
			let step = s.range[m] as f64 * (2.0 / lv).powf(s.gamma[m] as f64);
			let below = a.iter().filter(|&&v| v < 0.5 * step).count() as f64 / a.len() as f64;
			let sat = a.iter().filter(|&&v| v >= s.range[m] as f64).count() as f64 / a.len() as f64;
			format!("{}: |l| {} | step1 {:.4} below½ {:.3} sat {:.3}", BANK[m], quantiles(&mut a), step, below, sat)
		})
		.collect();
	eprintln!("[ctbr] {tag:6} LABEL n={} {}", cap.label.len(), parts.join(" || "));
}

/// Which axis carries the student's own-state error: mean |Euler angle| per axis
/// over all decision records and over the last 20% of each episode's records
/// (the own-state rollouts fly the train-time per-episode motor asymmetry).
fn axis_line(tag: &str, cap: &ImitCapture)
{
	let n = cap.tx.len().max(1) as f64;
	let all: Vec<f64> = (0..3).map(|a| cap.tx.iter().map(|x| x[a].abs()).sum::<f64>() / n).collect();
	eprintln!(
		"[ctbr] {tag:6} AXIS mean |roll| {:.2}° |pitch| {:.2}° |yaw| {:.2}°",
		all[0].to_degrees(), all[1].to_degrees(), all[2].to_degrees()
	);
}

/// MPCOF plans from a SHADOW rate loop driven by ITS OWN held setpoints; under DAgger
/// the real loop runs on the student's. How far apart are they along the student's states?
fn shadow_line(tag: &str, cap: &ImitCapture)
{
	let n = cap.hidden.len().max(1) as f64;
	let mut parts = Vec::new();
	for a in 0..3
	{
		let (mut di, mut si, mut ri, mut df) = (0.0, 0.0, 0.0, 0.0);
		for (h, r) in cap.hidden.iter().zip(&cap.inner)
		{
			di += (h[3 + a] - r[a]).abs();
			si += h[3 + a].abs();
			ri += r[a].abs();
			df += (h[6 + a] - r[3 + a]).abs();
		}
		parts.push(format!("{}: |I_sh−I| {:.4} (|I_sh| {:.4} |I| {:.4}) |Fh_sh−Fh| {:.5}", BANK[a + 1], di / n, si / n, ri / n, df / n));
	}
	eprintln!("[ctbr] {tag:6} SHADOW vs REAL inner loop: {}", parts.join(" | "));
}

/// The full report for one trained student.
pub(crate) fn report(rig: &mut Rig, tag: &str, cfg: &RewardGatedConfigPacked, c: &WnnController, train: &ImitCapture, ho: &ImitCapture, own: &ImitCapture)
{
	let (y_tr, y_ho, y_own) = (ys(c, train), ys(c, ho), ys(c, own));
	label_grid_line(&format!("{tag}/ho"), c, ho);
	label_grid_line(&format!("{tag}/own"), c, own);
	let s_ram = score(&ram(c, ho), &y_ho);
	let s_mem = score(&ho.label.iter().map(|&l| memoriser(c, l)).collect::<Vec<_>>(), &y_ho);
	let (or_pred, conf) = oracle_phys(c, ho);
	let s_or = score(&or_pred, &y_ho);
	let s_own = score(&ram(c, own), &y_own);
	eprintln!(
		"[ctbr] {tag:6} IMITATION MAE  teacher-states RAM {} | memoriser {} | ORACLE {} (conflict-cells {conf:.3}) | own-states RAM {}",
		fmt_mae(&s_ram), fmt_mae(&s_mem), fmt_mae(&s_or), fmt_mae(&s_own)
	);
	eprintln!("[ctbr] {tag:6} IMITATION R²   teacher-states RAM {} | ORACLE {} | own-states RAM {}", fmt_r2(&s_ram), fmt_r2(&s_or), fmt_r2(&s_own));
	let bpf = c.thresholds_ref().len() / c.num_features().max(1);
	let r2 = |xtr: Vec<Vec<f64>>, xho: Vec<Vec<f64>>| fmt_r2(&score(&ls(&xtr, &y_tr, &xho), &y_ho));
	eprintln!(
		"[ctbr] {tag:6} REPRESENTATION R² (ridge, train→held-out) raw {} | thermo {} | raw+ua {} | raw+I {} | raw+Fh {} | raw+d̂ {} | raw+all {} | teacher(rpy,ω,vert,hidden) {}",
		r2(raw_rows(&train.feats), raw_rows(&ho.feats)),
		r2(thermo_rows(&train.feats, c.thresholds_ref(), bpf), thermo_rows(&ho.feats, c.thresholds_ref(), bpf)),
		r2(raw_plus(train, &[0, 1, 2]), raw_plus(ho, &[0, 1, 2])),
		r2(raw_plus(train, &[3, 4, 5]), raw_plus(ho, &[3, 4, 5])),
		r2(raw_plus(train, &[6, 7, 8]), raw_plus(ho, &[6, 7, 8])),
		r2(raw_plus(train, &[9, 10, 11]), raw_plus(ho, &[9, 10, 11])),
		r2(raw_plus(train, &(0..12).collect::<Vec<_>>()), raw_plus(ho, &(0..12).collect::<Vec<_>>())),
		r2(teacher_rows(train), teacher_rows(ho)),
	);
	// The same linear map, applied on the student's OWN states (distribution shift).
	// The same linear maps fitted on the student's OWN states (first half of the
	// own-state records → second half): what predicts the label where it flies.
	let h = own.feats.len() / 2;
	let split = |rows: Vec<Vec<f64>>| -> String {
		let (a, b) = rows.split_at(h);
		fmt_r2(&score(&ls(a, &y_own[..h], b), &y_own[h..]))
	};
	let all: Vec<usize> = (0..12).collect();
	eprintln!(
		"[ctbr] {tag:6} REPRESENTATION R² OWN states (fit ½ → test ½): raw {} | raw+ua {} | raw+I {} | raw+Fh {} | raw+d̂ {} | raw+I+Fh {} | raw+all {}",
		split(raw_rows(&own.feats)),
		split(raw_plus(own, &[0, 1, 2])),
		split(raw_plus(own, &[3, 4, 5])),
		split(raw_plus(own, &[6, 7, 8])),
		split(raw_plus(own, &[9, 10, 11])),
		split(raw_plus(own, &[3, 4, 5, 6, 7, 8])),
		split(raw_plus(own, &all)),
	);
	axis_line(&format!("{tag}/own"), own);
	shadow_line(&format!("{tag}/ho"), ho);
	shadow_line(&format!("{tag}/own"), own);
	flights(rig, tag, cfg, c, [s_ram[0].1, s_ram[1].1, s_ram[2].1, s_ram[3].1], [s_own[0].1, s_own[1].1, s_own[2].1, s_own[3].1]);
}

/// The scorer's five ±20% asymmetry pools (the probe's draw, seed-fixed): per pool,
/// the STUDENT and the TEACHER (exact, over the same inner stage) — stable %, and
/// mean |roll|, |pitch|, |yaw| over the last 20% of each episode. Which axis fails?
pub(crate) fn pool_axes(rig: &mut Rig, cfg: &RewardGatedConfigPacked, c: &WnnController)
{
	let n: usize = std::env::var("WNN_IMIT_POOL_EPS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
	if n == 0
	{
		return;
	}
	let mut ar = SmallRng::seed_from_u64(0xA5A5);
	for _pool in 0..5
	{
		let a: [f32; 4] = std::array::from_fn(|m| cfg.dist_motor_asym[m] * (1.0 + ar.gen_range(-0.2f32..0.2)));
		let mut k = cfg.clone();
		k.dist_motor_asym = a;
		k.dist_motor_asym_mag = 0.0;
		let s = fly_axes(rig, &k, c, true, n);
		let t = fly_axes(rig, &k, c, false, n);
		eprintln!(
			"[ctbr] POOL asym [{:.2} {:.2} {:.2} {:.2}] student {:.0}% tail |r,p,y| {:.2} {:.2} {:.2}° I/ilim {:+.2} {:+.2} {:+.2} sp {:+.3} {:+.3} {:+.3} | teacher {:.0}% tail {:.2} {:.2} {:.2}°",
			a[0], a[1], a[2], a[3], s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], t[0], t[1], t[2], t[3]
		);
	}
}

/// One fixed-plant flight, student or teacher (exact, over the student's inner
/// stage); returns [stable %, tail |roll|°, tail |pitch|°, tail |yaw|°].
fn fly_axes(rig: &mut Rig, cfg: &RewardGatedConfigPacked, c: &WnnController, student: bool, n: usize) -> [f64; 10]
{
	use crate::episode_regime::{anchor_controller, draw_translation_episode, feed_live_obs, scored_att_err, OuterLoops};
	let hold = c.action_repeat_n().max(1);
	let loops = OuterLoops::for_cfg(cfg);
	let mut rng = SmallRng::seed_from_u64(0xA7E5);
	let mut cc = c.clone();
	let (mut n_st, mut tail) = (0usize, [0.0f64; 3]);
	// student only: tail mean of the inner integrator / i_limit and of the held setpoint (signed)
	let ilim: [f64; 3] = std::array::from_fn(|a| spec(c).rate_gains[4 * a + 3].max(1e-12));
	let (mut tail_i, mut tail_sp) = ([0.0f64; 3], [0.0f64; 3]);
	for ep in 0..n
	{
		let (q0, w0) = crate::dagger_train::sample_initial_state(
			&mut rng, 5f64.to_radians(), cfg.max_initial_yaw_rad, cfg.max_initial_body_rate, cfg.max_initial_yaw_rate,
			[cfg.active_roll, cfg.active_pitch, cfg.active_yaw],
		);
		rig.sim.reset(Some(q0), Some(w0));
		let coll = draw_translation_episode(&mut rig.sim, cfg, &mut rng);
		crate::dagger_train::apply_cfg_disturbance(&mut rig.sim, cfg, &mut rng);
		let teacher = rig.teachers.get_mut(cfg.teacher_id_for(0, ep));
		teacher.reset();
		cc.reset(crate::controller::yaw_from_quat_rs(q0));
		cc.seed_att_estimator(q0);
		anchor_controller(&mut cc, cfg, coll);
		let mut pilot = crate::ctbr::CtbrOverRate::new(spec(c).rate_stage(), hold);
		let mut last = [0.5f32; 4];
		let (mut errs, mut eul, mut div) = (Vec::new(), Vec::new(), false);
		let mut isp: Vec<[f64; 6]> = Vec::new();
		for _t in 0..cfg.steps_per_episode
		{
			if rig.sim.is_unstable()
			{
				div = true;
				break;
			}
			let (gyro, accel) = rig.sim.read_imu();
			let q = rig.sim.quaternion();
			let pwm = if student
			{
				feed_live_obs(&mut cc, &rig.sim, cfg, coll);
				let v = cc.step(gyro, accel, rig.target);
				let (inr, sp) = (cc.diag_ctbr_inner_state().unwrap(), cc.ctbr_held_setpoint().unwrap());
				isp.push([inr[0] / ilim[0], inr[1] / ilim[1], inr[2] / ilim[2], sp[1], sp[2], sp[3]]);
				[v[0], v[1], v[2], v[3]]
			}
			else
			{
				teacher.observe(gyro, last.map(|v| v as f64));
				let label = loops.expert_ctbr(teacher, &rig.sim, q, gyro, rig.target, cfg.target_altitude, hold);
				pilot.fly(label, gyro).map(|v| v as f32)
			};
			last = pwm;
			rig.sim.step(pwm);
			errs.push(scored_att_err(&rig.sim, cfg) as f64);
			let (r, p, y) = crate::controller::quat_to_euler_f64(rig.sim.quaternion());
			eul.push([r.abs(), p.abs(), crate::controller::wrap_angle_f64(y).abs()]);
		}
		let mean = errs.iter().sum::<f64>() / errs.len().max(1) as f64;
		if !div && mean <= 5f64.to_radians()
		{
			n_st += 1;
		}
		let t0 = eul.len() * 4 / 5;
		for a in 0..3
		{
			tail[a] += eul[t0..].iter().map(|e| e[a]).sum::<f64>() / (eul.len() - t0).max(1) as f64;
			if student && isp.len() > t0
			{
				tail_i[a] += isp[t0..].iter().map(|e| e[a]).sum::<f64>() / (isp.len() - t0) as f64;
				tail_sp[a] += isp[t0..].iter().map(|e| e[3 + a]).sum::<f64>() / (isp.len() - t0) as f64;
			}
		}
	}
	let nf = n as f64;
	[
		100.0 * n_st as f64 / nf,
		(tail[0] / nf).to_degrees(),
		(tail[1] / nf).to_degrees(),
		(tail[2] / nf).to_degrees(),
		tail_i[0] / nf,
		tail_i[1] / nf,
		tail_i[2] / nf,
		tail_sp[0] / nf,
		tail_sp[1] / nf,
		tail_sp[2] / nf,
	]
}

/// How a flight turns the teacher's decision into the setpoint the inner stage flies.
enum Xf
{
	Exact,
	Memoriser,
	Noise([f64; 4]),
}

fn flights(rig: &mut Rig, tag: &str, cfg: &RewardGatedConfigPacked, c: &WnnController, rmse_ho: [f64; 4], rmse_own: [f64; 4])
{
	let n: usize = std::env::var("WNN_IMIT_FLIGHT_EPS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
	if n == 0
	{
		return;
	}
	let tilt = 5f64.to_radians();
	for (name, xf) in [
		("exact", Xf::Exact),
		("memoriser", Xf::Memoriser),
		("noise=RAM rmse (teacher states)", Xf::Noise(rmse_ho)),
		("noise=RAM rmse (own states)", Xf::Noise(rmse_own)),
	]
	{
		let (st, err, steady) = fly(rig, cfg, c, &xf, n, tilt, 0xF1E5);
		eprintln!("[ctbr] {tag:6} FLIGHT teacher→{name}: stable {st:.1}% err {err:.2}° steady {steady:.2}° ({n} eps @5°)");
	}
}

fn gauss(rng: &mut SmallRng) -> f64
{
	let (u1, u2): (f64, f64) = (rng.gen::<f64>().max(1e-300), rng.gen());
	(-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Fly the teacher's CTBR decision (transformed) through the student's inner stage at
/// the student's outer rate — the training episode regime (ICs, translation draw,
/// per-episode weather incl. the train-time motor asymmetry). (stable %, err °, steady °).
fn fly(rig: &mut Rig, cfg: &RewardGatedConfigPacked, c: &WnnController, xf: &Xf, n: usize, tilt: f64, seed: u64) -> (f64, f64, f64)
{
	use crate::episode_regime::{draw_translation_episode, scored_att_err, OuterLoops};
	let hold = c.action_repeat_n().max(1);
	let loops = OuterLoops::for_cfg(cfg);
	let mut rng = SmallRng::seed_from_u64(seed);
	let mut nrng = SmallRng::seed_from_u64(seed ^ 0x9015E);
	let (mut n_st, mut sum_err, mut sum_steady, mut n_steady) = (0usize, 0.0f64, 0.0f64, 0usize);
	for ep in 0..n
	{
		let (q0, w0) = crate::dagger_train::sample_initial_state(
			&mut rng, tilt, cfg.max_initial_yaw_rad, cfg.max_initial_body_rate, cfg.max_initial_yaw_rate,
			[cfg.active_roll, cfg.active_pitch, cfg.active_yaw],
		);
		rig.sim.reset(Some(q0), Some(w0));
		draw_translation_episode(&mut rig.sim, cfg, &mut rng);
		crate::dagger_train::apply_cfg_disturbance(&mut rig.sim, cfg, &mut rng);
		let teacher = rig.teachers.get_mut(cfg.teacher_id_for(0, ep));
		teacher.reset();
		let mut pilot = crate::ctbr::CtbrOverRate::new(spec(c).rate_stage(), hold);
		let mut last = [0.5f32; 4];
		let mut held = [0.0f64; 4];
		let (mut errs, mut div) = (Vec::with_capacity(cfg.steps_per_episode), false);
		for t in 0..cfg.steps_per_episode
		{
			if rig.sim.is_unstable()
			{
				div = true;
				break;
			}
			let (gyro, _a) = rig.sim.read_imu();
			let q = rig.sim.quaternion();
			teacher.observe(gyro, last.map(|v| v as f64));
			let label = loops.expert_ctbr(teacher, &rig.sim, q, gyro, rig.target, cfg.target_altitude, hold);
			if t % hold == 0
			{
				held = match xf
				{
					Xf::Exact => label,
					Xf::Memoriser => memoriser(c, label.map(|v| v as f32)),
					Xf::Noise(sd) =>
					{
						let r = spec(c).range;
						std::array::from_fn(|m| (label[m] + sd[m] * gauss(&mut nrng)).clamp(-r[m] as f64, r[m] as f64))
					}
				};
			}
			let pwm = pilot.fly(held, gyro);
			last = pwm.map(|v| v as f32);
			rig.sim.step(last);
			errs.push(scored_att_err(&rig.sim, cfg) as f64);
		}
		let mean = errs.iter().sum::<f64>() / errs.len().max(1) as f64;
		if !div && mean <= 5f64.to_radians()
		{
			n_st += 1;
		}
		sum_err += mean;
		let tail0 = cfg.steps_per_episode * 4 / 5;
		if errs.len() > tail0
		{
			sum_steady += errs[tail0..].iter().sum::<f64>() / (errs.len() - tail0) as f64;
			n_steady += 1;
		}
	}
	(100.0 * n_st as f64 / n as f64, (sum_err / n as f64).to_degrees(), (sum_steady / n_steady.max(1) as f64).to_degrees())
}
