//! DIAGNOSTIC (06/11/2026, CTRL-5 lag): IMITATION vs CLOSED LOOP for the real
//! RAM student under motor lag. Env-gated (WNN_IMIT_PROBE=1 in
//! dagger_train_batch_inplace, genome 0 only) so it runs on the EXACT recipe
//! controller + packed config a phased_ga run builds; never set in production.
//!
//! Per run it reports, on HELD-OUT teacher-driven states:
//!   RAM    — the trained student's raw decode vs the trainer's decode target;
//!   LSraw  — ridge LS on the continuous feature vector (information ceiling of
//!            a linear memoryless policy on the SAME observation);
//!   LSthr  — ridge LS on the thermometer bits (encoding-resolution ceiling);
//!   ORACLE — per-neuron MAJORITY bit at each address over the held-out set
//!            itself (the best ANY memoryless lookup on these address
//!            functions can do — the aliasing floor), plus the conflict rate
//!            (fraction of visited cells that receive both TRUE and FALSE
//!            targets) and the minority-sample fraction.
//! for (A) a pure behaviour-cloning student (teacher drives, real write path,
//! no DAgger relabel) and (B) the production DAgger student; plus each one's
//! closed-loop eval and its imitation on its OWN states.

use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::controller::WnnController;
use crate::dagger_train::{
	eval_closed_loop_rs, rollout_and_label_inner, sim_default, train_on_trajectory_rs, AirframeRs, RewardGatedConfigPacked,
	TeacherBank,
};

/// Per-step capture (see rollout_and_label_inner).
pub(crate) struct ImitCapture
{
	/// expert_drives only: overwrite the student's accumulator with what flew, so
	/// the obs_pwm feature is the closed-loop invariant (own command = flown)
	/// instead of a free-running integral of its own unused decodes.
	pub sync_accumulator: bool,
	pub want_addresses: bool,
	pub feats: Vec<Vec<f32>>,
	pub decoded: Vec<[f32; 4]>,
	pub target: Vec<[f32; 4]>,
	pub empty: Vec<u16>,
	pub addrs: Vec<Vec<u64>>,
	pub tbits: Vec<Vec<bool>>,
	/// The TEACHER's own inputs at this decision: state error e (3), body rates
	/// (3), its actuator estimate ua (3, a twin fed the same applied stream in
	/// the same order) and its observer d̂ (3, Teacher::integrals — MPCOF's d̂).
	pub tx: Vec<[f64; 15]>,
	act: crate::actuator_lag::ActuatorLag,
}

impl ImitCapture
{
	pub(crate) fn new(sync_accumulator: bool, want_addresses: bool) -> Self
	{
		ImitCapture {
			sync_accumulator,
			want_addresses,
			feats: Vec::new(),
			decoded: Vec::new(),
			target: Vec::new(),
			empty: Vec::new(),
			addrs: Vec::new(),
			tbits: Vec::new(),
			tx: Vec::new(),
			act: crate::actuator_lag::ActuatorLag::new(0.0, 0.001),
		}
	}

	/// Episode start (rollout_and_label_inner): the twin re-seeds like the teacher's.
	pub(crate) fn begin_episode(&mut self, motor_lag_s: f32, dt: f32)
	{
		self.act = crate::actuator_lag::ActuatorLag::new(motor_lag_s, dt);
		self.act.reset();
	}

	/// Mirror of teacher.observe(gyro, last_applied) — called where the rollout calls it.
	pub(crate) fn observe_applied(&mut self, applied: [f32; 4])
	{
		self.act.observe(applied.map(|v| v as f64));
	}

	pub(crate) fn record_teacher_inputs(&mut self, q: [f32; 4], gyro: [f32; 3], dhat: [f32; 3], vert: [f32; 3])
	{
		let (r, p, y) = crate::controller::quat_to_euler_f64(q);
		let ua = self.act.state();
		self.tx.push([
			r, p, crate::controller::wrap_angle_f64(y), gyro[0] as f64, gyro[1] as f64, gyro[2] as f64,
			ua[0], ua[1], ua[2], dhat[0] as f64, dhat[1] as f64, dhat[2] as f64,
			vert[0] as f64, vert[1] as f64, vert[2] as f64,
		]);
	}

	/// `label` = the bank-space label the trajectory recorded this step.
	pub(crate) fn record(&mut self, c: &mut WnnController, label: [f32; 4], applied: [f32; 4])
	{
		let (d, bits) = c.diag_targets(label);
		self.feats.push(c.last_feature_vector_ref().to_vec());
		self.decoded.push(c.diag_raw_decode());
		self.target.push(d);
		self.empty.push(c.last_output_empty_count() as u16);
		if self.want_addresses
		{
			self.addrs.push(c.diag_output_addresses());
			self.tbits.push(bits);
		}
		if self.sync_accumulator
		{
			c.set_pwm_accumulator(applied);
		}
	}
}

/// Ridge LS per output bank: rows x → y[m]. Returns weights [m][feature].
fn ridge(xs: &[Vec<f64>], ys: &[[f32; 4]], nm: usize) -> Vec<Vec<f64>>
{
	let nf = xs[0].len();
	let mut a = vec![0.0f64; nf * nf];
	for x in xs
	{
		for i in 0..nf
		{
			if x[i] == 0.0
			{
				continue;
			}
			for j in 0..nf
			{
				a[i * nf + j] += x[i] * x[j];
			}
		}
	}
	for i in 0..nf
	{
		a[i * nf + i] += 1e-6 * xs.len() as f64 + 1e-9;
	}
	(0..nm)
		.map(|m| {
			let mut b = vec![0.0f64; nf];
			for (x, y) in xs.iter().zip(ys)
			{
				for i in 0..nf
				{
					b[i] += x[i] * y[m] as f64;
				}
			}
			solve(a.clone(), b, nf)
		})
		.collect()
}

fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64>
{
	for c in 0..n
	{
		let piv = (c..n).max_by(|&i, &j| a[i * n + c].abs().partial_cmp(&a[j * n + c].abs()).unwrap()).unwrap();
		if piv != c
		{
			for k in 0..n
			{
				a.swap(c * n + k, piv * n + k);
			}
			b.swap(c, piv);
		}
		let p = a[c * n + c];
		if p.abs() < 1e-300
		{
			continue;
		}
		for r in 0..n
		{
			if r != c
			{
				let f = a[r * n + c] / p;
				if f != 0.0
				{
					for k in c..n
					{
						a[r * n + k] -= f * a[c * n + k];
					}
					b[r] -= f * b[c];
				}
			}
		}
	}
	(0..n).map(|i| if a[i * n + i].abs() < 1e-300 { 0.0 } else { b[i] / a[i * n + i] }).collect()
}

fn raw_rows(feats: &[Vec<f32>]) -> Vec<Vec<f64>>
{
	feats.iter().map(|f| f.iter().map(|&v| v as f64).chain(std::iter::once(1.0)).collect()).collect()
}

fn thermo_rows(feats: &[Vec<f32>], thr: &[f32], bpf: usize) -> Vec<Vec<f64>>
{
	feats
		.iter()
		.map(|f| {
			let mut r = Vec::with_capacity(f.len() * bpf + 1);
			for (i, &v) in f.iter().enumerate()
			{
				for b in 0..bpf
				{
					r.push(if v >= thr[i * bpf + b] { 1.0 } else { 0.0 });
				}
			}
			r.push(1.0);
			r
		})
		.collect()
}

/// mean |ŷ − y| over banks [0, nm) on a held-out set.
fn ls_mae(w: &[Vec<f64>], xs: &[Vec<f64>], ys: &[[f32; 4]]) -> f64
{
	let nm = w.len();
	let mut s = 0.0;
	for (x, y) in xs.iter().zip(ys)
	{
		for m in 0..nm
		{
			let yh: f64 = x.iter().zip(&w[m]).map(|(a, b)| a * b).sum();
			s += (yh.clamp(0.0, 1.0) - y[m] as f64).abs();
		}
	}
	s / (xs.len() * nm).max(1) as f64
}

/// Spread of the target: mean |y − mean(y)| (the MAE of a constant predictor).
fn spread(ys: &[[f32; 4]], nm: usize) -> f64
{
	let mut s = 0.0;
	for m in 0..nm
	{
		let mu = ys.iter().map(|y| y[m] as f64).sum::<f64>() / ys.len().max(1) as f64;
		s += ys.iter().map(|y| (y[m] as f64 - mu).abs()).sum::<f64>();
	}
	s / (ys.len() * nm).max(1) as f64
}

fn ram_mae(cap: &ImitCapture, nm: usize) -> (f64, f64)
{
	let mut s = 0.0;
	for (d, t) in cap.decoded.iter().zip(&cap.target)
	{
		for m in 0..nm
		{
			s += (d[m] - t[m]).abs() as f64;
		}
	}
	let n = cap.decoded.len().max(1);
	let e = cap.empty.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
	(s / (n * nm) as f64, e)
}

/// Aliasing floor on the held-out set's own addresses: conflict-cell rate,
/// minority-sample fraction, and the MAE of the per-neuron-majority decode.
fn oracle(cap: &ImitCapture, c: &WnnController) -> (f64, f64, f64, f64)
{
	let nn = cap.tbits[0].len();
	let mut maps: Vec<HashMap<u64, (u32, u32)>> = (0..nn).map(|_| HashMap::new()).collect();
	for (a, b) in cap.addrs.iter().zip(&cap.tbits)
	{
		for n in 0..nn
		{
			let e = maps[n].entry(a[n]).or_insert((0, 0));
			if b[n]
			{
				e.0 += 1;
			}
			else
			{
				e.1 += 1;
			}
		}
	}
	let (mut cells, mut conf, mut minority, mut samples) = (0usize, 0usize, 0u64, 0u64);
	for mp in &maps
	{
		for &(t, f) in mp.values()
		{
			cells += 1;
			if t > 0 && f > 0
			{
				conf += 1;
			}
			minority += t.min(f) as u64;
			samples += (t + f) as u64;
		}
	}
	let mode = c.memory_mode_u8();
	let (tc, fc) = (crate::cell_mode::true_cell(mode), crate::cell_mode::false_cell(mode));
	let l = c.levels_per_motor();
	let nm = c.num_motors().min(4);
	let mut s = 0.0;
	let mut cells_buf = vec![0u8; nn];
	for (k, a) in cap.addrs.iter().enumerate()
	{
		for n in 0..nn
		{
			let (t, f) = maps[n][&a[n]];
			cells_buf[n] = if t >= f { tc } else { fc };
		}
		for m in 0..nm
		{
			let d = crate::cell_mode::decode_motor_cells(&cells_buf[m * l..(m + 1) * l], mode, c.output_decode_u8());
			s += (d - cap.target[k][m]).abs() as f64;
		}
	}
	let mean_visits = samples as f64 / cells.max(1) as f64;
	(
		conf as f64 / cells.max(1) as f64,
		minority as f64 / samples.max(1) as f64,
		s / (cap.addrs.len() * nm).max(1) as f64,
		mean_visits,
	)
}

struct Rig
{
	cfg_bc: RewardGatedConfigPacked,
	teachers: TeacherBank,
	sim: crate::controller::AttitudeSim,
	target: [f32; 3],
}

impl Rig
{
	fn roll(&mut self, c: &mut WnnController, cfg: &RewardGatedConfigPacked, it: usize, ep: usize, tilt: f64, rng: &mut SmallRng, cap: &mut ImitCapture)
		-> crate::dagger_train::TrajectoryRs
	{
		let teacher = self.teachers.get_mut(cfg.teacher_id_for(it, ep));
		rollout_and_label_inner(c, teacher, &mut self.sim, cfg, tilt, rng, self.target, Some(cap))
	}
}

/// Held-out teacher-driven set (and, separately, the student's own states).
fn held_out(rig: &mut Rig, c: &mut WnnController, own: bool, episodes: usize, tilt: f64, seed: u64) -> ImitCapture
{
	let cfg = if own
	{
		let mut k = rig.cfg_bc.clone();
		k.expert_drives = false;
		k
	}
	else
	{
		rig.cfg_bc.clone()
	};
	let mut rng = SmallRng::seed_from_u64(seed);
	let mut cap = ImitCapture::new(!own, !own);
	for ep in 0..episodes
	{
		rig.roll(c, &cfg, 0, ep, tilt, &mut rng, &mut cap);
	}
	cap
}

fn closed_loop(rig: &mut Rig, c: &mut WnnController, seed: u64) -> (f64, f64)
{
	let mut cfg = rig.cfg_bc.clone();
	cfg.expert_drives = false;
	cfg.eval_episodes = 40;
	let mut rng = SmallRng::seed_from_u64(seed);
	let (nm, l) = (c.num_motors(), c.levels_per_motor());
	let (_f, err, st, _j, _m) = eval_closed_loop_rs(c, &mut rig.sim, &cfg, &mut rng, rig.target, 5f64.to_radians(), nm, l);
	(st * 100.0, err.to_degrees())
}

/// The SAME controller through the production CPU scorer (cpu_score::rollout_one,
/// the GA-fitness twin of the Metal kernel), ICs drawn from the trainer's own
/// distributions at the scorer tilt (5°). Returns (stable %, err °, steady °).
fn closed_loop_scorer(cfg: &RewardGatedConfigPacked, c: &WnnController, n: usize, seed: u64, asym: [f32; 4]) -> (f64, f64, f64)
{
	use crate::episode_regime::jitter_sym;
	let mut rng = SmallRng::seed_from_u64(seed);
	let (mut q0, mut w0) = (Vec::with_capacity(4 * n), Vec::with_capacity(3 * n));
	let (mut z, mut vz, mut mass, mut coll) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
	for _ in 0..n
	{
		let (q, w) = crate::dagger_train::sample_initial_state(
			&mut rng,
			5f64.to_radians(),
			cfg.max_initial_yaw_rad,
			cfg.max_initial_body_rate,
			cfg.max_initial_yaw_rate,
			[cfg.active_roll, cfg.active_pitch, cfg.active_yaw],
		);
		q0.extend_from_slice(&q);
		w0.extend_from_slice(&w);
		mass.push(cfg.af_mass * (1.0 + jitter_sym(&mut rng, cfg.mass_jitter)));
		z.push(jitter_sym(&mut rng, cfg.alt_offset));
		vz.push(jitter_sym(&mut rng, cfg.init_vz));
		coll.push(jitter_sym(&mut rng, cfg.collective_jitter));
	}
	let s1 = crate::stage1::Stage1Cfg {
		target_altitude: cfg.target_altitude,
		lambda_alt: 0.0,
		init_z: z,
		init_vz: vz,
		mass,
		collective_frac: coll,
		lambda_pos: 0.0,
		init_x: Vec::new(),
		init_y: Vec::new(),
		plant: cfg.plant_opts(),
		att_ref: crate::att_err_ref::AttErrRef::LEVEL,
	};
	let mut cc = c.clone();
	let row = crate::cpu_score::rollout_one(
		&mut cc, &q0, &w0, n, cfg.steps_per_episode, cfg.dt as f32, cfg.af_arm_length, cfg.af_k_thrust, cfg.af_k_drag,
		cfg.af_inertia, cfg.af_gravity, cfg.motor_lag_s, [0.0; 3], cfg.dist_enabled, cfg.dist_tau_bias, cfg.dist_gust_sigma,
		cfg.dist_gust_tau_c, asym, cfg.dist_gyro_sigma, cfg.dist_gyro_bias_walk, cfg.dist_accel_sigma,
		seed ^ 0xD157, cfg.dist_dropout_prob, cfg.dist_dropout_len_steps, cfg.dist_obs_delay_steps,
		cfg.dist_torque_scale_jitter, c.levels_per_motor(), c.num_motors(), None, None, None, 0.0, 0.0, None,
		if cfg.translation { Some(&s1) } else { None },
	);
	(row[2] * 100.0, row[1].to_degrees(), row[5].to_degrees())
}

fn report(tag: &str, lag: f32, c: &WnnController, train: &ImitCapture, ho: &ImitCapture, own: Option<&ImitCapture>, cl: (f64, f64))
{
	let nm = c.num_motors().min(4);
	let (mae, empty) = ram_mae(ho, nm);
	let sp = spread(&ho.target, nm);
	let w_raw = ridge(&raw_rows(&train.feats), &train.target, nm);
	let ls_raw = ls_mae(&w_raw, &raw_rows(&ho.feats), &ho.target);
	let bpf = c.thresholds_ref().len() / c.num_features().max(1);
	let w_thr = ridge(&thermo_rows(&train.feats, c.thresholds_ref(), bpf), &train.target, nm);
	let ls_thr = ls_mae(&w_thr, &thermo_rows(&ho.feats, c.thresholds_ref(), bpf), &ho.target);
	let (conf, minority, or_mae, visits) = oracle(ho, c);
	let tx_rows = |cap: &ImitCapture, k: usize| -> Vec<Vec<f64>> { cap.tx.iter().map(|x| x[..k].iter().copied().chain(std::iter::once(1.0)).collect()).collect() };
	let ls_tx = |k: usize| { let w = ridge(&tx_rows(train, k), &train.target, nm); ls_mae(&w, &tx_rows(ho, k), &ho.target) };
	let (ls_x, ls_xu, ls_xud, ls_all) = (ls_tx(6), ls_tx(9), ls_tx(12), ls_tx(15));
	let rawplus = |cap: &ImitCapture| -> Vec<Vec<f64>> { raw_rows(&cap.feats).into_iter().zip(&cap.tx).map(|(mut r, x)| { r.extend_from_slice(&x[6..15]); r }).collect() };
	let w_rp = ridge(&rawplus(train), &train.target, nm);
	let ls_rp = ls_mae(&w_rp, &rawplus(ho), &ho.target);
	let rawsub = |cap: &ImitCapture, a: usize, b: usize| -> Vec<Vec<f64>> { raw_rows(&cap.feats).into_iter().zip(&cap.tx).map(|(mut r, x)| { r.extend_from_slice(&x[a..b]); r }).collect() };
	let ls_rs = |a: usize, b: usize| { let w = ridge(&rawsub(train, a, b), &train.target, nm); ls_mae(&w, &rawsub(ho, a, b), &ho.target) };
	let (ls_r_ua, ls_r_dh) = (ls_rs(6, 9), ls_rs(9, 12));
	// The student's own obs_pwm columns vs the teacher's ua: which feature columns carry it?
	let nfeat = train.feats[0].len();
	let corr = |j: usize, k: usize| {
		let n = ho.feats.len() as f64;
		let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
		for (f, x) in ho.feats.iter().zip(&ho.tx) { let (a, b) = (f[j] as f64, x[k]); sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; }
		let cov = sxy / n - sx * sy / (n * n);
		cov / ((sxx / n - (sx / n).powi(2)).max(1e-30) * (syy / n - (sy / n).powi(2)).max(1e-30)).sqrt()
	};
	let best: Vec<String> = (6..9).map(|k| { let (j, r) = (0..nfeat).map(|j| (j, corr(j, k))).max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap()).unwrap(); format!("ua[{}]~f{j} r={r:.3}", k - 6) }).collect();
	{
		// d̂ statistics on the held-out set (per axis): mean, sd, lag-1 / lag-50 autocorrelation.
		let mut parts = Vec::new();
		for k in 9..12
		{
			let v: Vec<f64> = ho.tx.iter().map(|x| x[k]).collect();
			let n = v.len() as f64;
			let mu = v.iter().sum::<f64>() / n;
			let var = v.iter().map(|a| (a - mu).powi(2)).sum::<f64>() / n;
			let ac = |l: usize| v.windows(l + 1).map(|w| (w[0] - mu) * (w[l] - mu)).sum::<f64>() / (n * var.max(1e-30));
			parts.push(format!("d{}: mean {mu:.3} sd {:.3} ac1 {:.3} ac50 {:.3}", k - 9, var.sqrt(), ac(1), ac(50)));
		}
		eprintln!("[imitprobe] lag={lag:.2} {tag:6} teacher d̂ (rad/s²) {}", parts.join(" | "));
	}
	eprintln!("[imitprobe] lag={lag:.2} {tag:6} LSraw+ua {ls_r_ua:.4} LSraw+dhat {ls_r_dh:.4} | best student column per teacher ua axis: {}", best.join(", "));
	let mut dev: Vec<f32> = ho.target.iter().flat_map(|t| t[..nm].iter().map(|v| (v - 0.5).abs()).collect::<Vec<_>>()).collect();
	dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let q = |p: f64| dev[((dev.len() - 1) as f64 * p) as usize];
	let lvl = 1.0 / c.levels_per_motor() as f32;
	let sub = dev.iter().filter(|&&v| v < lvl).count() as f64 / dev.len() as f64;
	eprintln!(
		"[imitprobe] lag={lag:.2} {tag:6} |target-0.5| p50 {:.4} p90 {:.4} p99 {:.4} max {:.4}; below one decode level (1/{}) {:.3}",
		q(0.5), q(0.9), q(0.99), q(1.0), c.levels_per_motor(), sub
	);
	let own_s = own.map(|o| { let (m, e) = ram_mae(o, nm); format!(" | own-states RAM {m:.4} empty {:.3}", e / (nm * c.levels_per_motor()) as f64) }).unwrap_or_default();
	eprintln!(
		"[imitprobe] lag={lag:.2} {tag:6} held-out teacher states n={}: target spread {sp:.4} | RAM {mae:.4} (empty {:.3}) | LSraw {ls_raw:.4} | LSthr {ls_thr:.4} | teacher-inputs LS e,w {ls_x:.4} +ua {ls_xu:.4} +ua+dhat {ls_xud:.4} +vert {ls_all:.4} | LSraw+ua+dhat+vert {ls_rp:.4} | ORACLE-majority {or_mae:.4} conflict-cells {conf:.3} minority {minority:.3} visits/cell {visits:.1}{own_s} | closed-loop@5° stable {:.1}% err {:.2}°",
		ho.target.len(),
		empty / (nm * c.levels_per_motor()) as f64,
		cl.0,
		cl.1
	);
}

/// The probe. `base` is the untrained (or warm-started) recipe controller.
pub(crate) fn run(base: &WnnController, cfg: &RewardGatedConfigPacked, target: [f32; 3], folds: &[u64])
{
	let seed = folds[0];
	let lag = cfg.motor_lag_s;
	let mut cfg_bc = cfg.clone();
	cfg_bc.expert_drives = true;
	let ho_eps: usize = std::env::var("WNN_IMIT_PROBE_HO").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
	let ho_tilt = cfg.full_tilt_deg.to_radians();
	let mut rig = Rig { cfg_bc, teachers: TeacherBank::new(AirframeRs::from_cfg(cfg)), sim: sim_default(cfg), target };
	eprintln!(
		"[imitprobe] lag={lag:.2} rounds={} eps/round={} steps={} teacher={} expert_drives(BC)=on held-out eps={ho_eps} @ {:.0}° seed={seed}",
		cfg.num_rounds, cfg.episodes_per_round, cfg.steps_per_episode, cfg.teacher, cfg.full_tilt_deg
	);

	// (A) BEHAVIOUR CLONING — teacher drives, the real sn=0 write path, no gate.
	let mut c = base.clone();
	c.set_observer_motor_lag_core(rig.sim.motor_lag());
	let mut train = ImitCapture::new(true, false);
	let cfg_bc = rig.cfg_bc.clone();
	// Same budget as the production trainer: every fold, every round, every episode.
	for &fs in folds
	{
		let mut rng = SmallRng::seed_from_u64(fs ^ 0xB0C);
		for it in 0..cfg.num_rounds
		{
			let tilt = cfg.round_tilt_rad(it);
			for ep in 0..cfg.episodes_per_round
			{
				let t = rig.roll(&mut c, &cfg_bc, it, ep, tilt, &mut rng, &mut train);
				train_on_trajectory_rs(&mut c, &t, cfg);
			}
		}
	}
	let ho = held_out(&mut rig, &mut c, false, ho_eps, ho_tilt, seed ^ 0x4E1D);
	let own = held_out(&mut rig, &mut c, true, ho_eps, 5f64.to_radians(), seed ^ 0x0E1D);
	let cl = closed_loop(&mut rig, &mut c, seed ^ 0xE7A1);
	report("BC", lag, &c, &train, &ho, Some(&own), cl);

	if std::env::var("WNN_IMIT_PROBE_BC_ONLY").is_ok()
	{
		eprintln!("[imitprobe] DONE lag={lag:.2}");
		return;
	}
	// (B) the PRODUCTION DAgger student (same seed) — imitation on teacher
	// states and on its own, and its closed loop.
	let mut d = base.clone();
	let mut st = crate::dagger_train::TrainStats::default();
	// WNN_IMIT_TRAIN_ASYM=<mag>: each fold trains on its OWN resolved motor
	// asymmetry (1 + U(±mag), the scorer's draw law) — the train/score plant match.
	let train_asym: Option<f32> = std::env::var("WNN_IMIT_TRAIN_ASYM").ok().and_then(|v| v.parse().ok());
	let mut tr_rng = SmallRng::seed_from_u64(seed ^ 0x7A5A);
	for &fs in folds
	{
		let mut cf = cfg.clone();
		if let Some(mag) = train_asym
		{
			use rand::Rng;
			cf.dist_motor_asym = std::array::from_fn(|m| cfg.dist_motor_asym[m] * (1.0 + tr_rng.gen_range(-mag..mag)));
			eprintln!("[imitprobe] train fold {fs}: motor asym {:?}", cf.dist_motor_asym);
		}
		st = crate::dagger_train::dagger_train_inplace_rs(&mut d, &cf, target, fs);
	}
	let ho_d = held_out(&mut rig, &mut d, false, ho_eps, ho_tilt, seed ^ 0x4E1D);
	let own_d = held_out(&mut rig, &mut d, true, ho_eps, 5f64.to_radians(), seed ^ 0x0E1D);
	let cl_d = closed_loop(&mut rig, &mut d, seed ^ 0xE7A1);
	report("DAgger", lag, &d, &train, &ho_d, Some(&own_d), cl_d);
	let sc = closed_loop_scorer(cfg, &d, 200, seed ^ 0x5C0E, cfg.dist_motor_asym);
	eprintln!("[imitprobe] lag={lag:.2} DAgger via PRODUCTION CPU scorer (rollout_one, 200 eps @5°): stable {:.1}% err {:.2}° steady {:.2}°", sc.0, sc.1, sc.2);
	// The production scorer's pools fly a RESOLVED per-pool motor asymmetry
	// (DisturbanceConfig.resolved_motor_asym: 1 + U(±mag), L4C mag 0.20) that the
	// trainer's packed cfg never carries (fixed multipliers). 5 pools, same draw law.
	let mag: f32 = std::env::var("WNN_IMIT_ASYM_MAG").ok().and_then(|v| v.parse().ok()).unwrap_or(0.20);
	let mut ar = SmallRng::seed_from_u64(seed ^ 0xA5A5);
	let mut acc = (0.0, 0.0, 0.0);
	let mut rows = Vec::new();
	for pool in 0..5u64
	{
		use rand::Rng;
		let a: [f32; 4] = std::array::from_fn(|m| cfg.dist_motor_asym[m] * (1.0 + ar.gen_range(-mag..mag)));
		let r = closed_loop_scorer(cfg, &d, 100, seed ^ 0x5C0E ^ (pool << 20), a);
		rows.push(format!("[{:.2} {:.2} {:.2} {:.2}] {:.0}%/{:.2}°", a[0], a[1], a[2], a[3], r.0, r.1));
		acc = (acc.0 + r.0 / 5.0, acc.1 + r.1 / 5.0, acc.2 + r.2 / 5.0);
	}
	eprintln!(
		"[imitprobe] lag={lag:.2} DAgger via PRODUCTION CPU scorer WITH ±{mag} motor asym (5 pools × 100): stable {:.1}% err {:.2}° steady {:.2}° | {}",
		acc.0, acc.1, acc.2, rows.join(" ")
	);
	eprintln!(
		"[imitprobe] lag={lag:.2} DAgger rounds stable {:?} err {:?}",
		st.iter_stable_rate.iter().map(|v| (v * 100.0).round()).collect::<Vec<_>>(),
		st.iter_mean_err_deg.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>()
	);
	eprintln!("[imitprobe] DONE lag={lag:.2}");
}

/// The recipe controller + cfg from a dump_recipe_fixture.py text fixture.
pub(crate) fn from_fixture(path: &str) -> (WnnController, RewardGatedConfigPacked, [f32; 3], Vec<u64>)
{
	let text = std::fs::read_to_string(path).expect("fixture");
	let mut cfg = crate::ctrl17_pins::pin_translation_cfg(0.0);
	let mut kw: HashMap<String, String> = HashMap::new();
	for line in text.lines()
	{
		let (k, v) = line.split_once('=').expect("fixture line");
		if let Some(f) = k.strip_prefix("cfg.")
		{
			cfg_set(&mut cfg, f, v);
		}
		else if let Some(f) = k.strip_prefix("kw.")
		{
			kw.insert(f.to_string(), v.to_string());
		}
	}
	let g = |k: &str| kw.get(k).unwrap_or_else(|| panic!("fixture: missing kw.{k}")).clone();
	let us = |k: &str| g(k).parse::<usize>().unwrap();
	let f = |k: &str| g(k).parse::<f32>().unwrap();
	let b = |k: &str| g(k) == "true";
	let vf = |k: &str| g(k).split_whitespace().map(|x| x.parse::<f32>().unwrap()).collect::<Vec<f32>>();
	let vi = |k: &str| g(k).split_whitespace().map(|x| x.parse::<i64>().unwrap()).collect::<Vec<i64>>();
	let opt_u8 = |k: &str| { let v = g(k); if v == "none" { None } else { Some(v.parse::<u8>().unwrap()) } };
	let dhat_b = { let v = g("dhat_b"); if v == "none" { None } else { let x: Vec<f64> = v.split_whitespace().map(|x| x.parse().unwrap()).collect(); Some([x[0], x[1], x[2]]) } };
	// ARM-D history variant: a fixture edited to input_window_k > 1 with
	// output_full_window needs output connections drawn over the WHOLE window
	// (the dumped ones only address frame bits). Uniform redraw, fixed seed.
	let mut out_conn = vi("output_connections_per_genome");
	if us("input_window_k") > 1 && b("output_full_window")
	{
		use rand::Rng;
		let span = us("input_window_k") * vf("thresholds").len();
		let mut r = SmallRng::seed_from_u64(0x415D);
		for v in out_conn.iter_mut()
		{
			*v = r.gen_range(0..span) as i64;
		}
	}
	let mut c = WnnController::new(
		us("num_motors"), us("levels_per_motor_per_genome"), us("bits_per_feature"), us("input_window_k"),
		us("state_neurons_per_genome"), us("state_bits_per_neuron_per_genome"), us("output_bits_per_neuron_per_genome"),
		vf("thresholds"), vi("state_connections_per_genome"), out_conn,
		b("delta_control"), f("delta_max"), f("delta_leak"), f("delta_gamma"),
		b("obs_tilt_p"), b("obs_tilt_i"), b("obs_peraxis_p"), b("obs_peraxis_i"), b("obs_peraxis_yaw"), b("obs_pwm"), b("obs_yaw_err"), b("obs_yaw_err_i"),
		f("integral_leak"), f("integral_scale"), f("dt"), b("decouple_outputs"), us("action_repeat"),
		g("memory_mode").parse().unwrap(), opt_u8("output_decode"),
		dhat_b, f("dhat_l_gain"), b("dhat_ff"), f("dhat_ff_clamp"),
		b("obs_collective_cmd"), b("obs_alt_err"), b("obs_vz"), b("obs_pos_err_xy"), b("obs_vel_xy"),
		b("output_full_window"), us("frame_stride"),
		kw.get("delta_label_scale").map(|v| v.parse().unwrap()).unwrap_or(1.0),
		kw.get("dagger_label_delta").map(|v| v == "true").unwrap_or(false),
		b("obs_att_estimator"), 2.0, 0.1,
	)
	.expect("fixture controller");
	c.set_target_levels_core(kw.get("target_levels").map(|v| v.parse().unwrap()).unwrap_or(0)).unwrap();
	let t: Vec<f32> = vf("target_rpy");
	let seeds: Vec<u64> = g("fold_seeds").split_whitespace().map(|x| x.parse().unwrap()).collect();
	(c, cfg, [t[0], t[1], t[2]], seeds)
}

#[cfg(test)]
mod tests
{
	/// The RECIPE probe: WNN_IMIT_FIXTURE=<dump_recipe_fixture.py output>.
	#[test]
	#[ignore]
	fn diag_imitation_probe_recipe()
	{
		let path = std::env::var("WNN_IMIT_FIXTURE").expect("WNN_IMIT_FIXTURE");
		let (c, cfg, target, seeds) = super::from_fixture(&path);
		let nf: usize = std::env::var("WNN_IMIT_FOLDS").ok().and_then(|v| v.parse().ok()).unwrap_or(seeds.len());
		super::run(&c, &cfg, target, &seeds[..nf.min(seeds.len())]);
	}

	/// Light probe on the lag_student_law_tests RAM student (not the recipe —
	/// the recipe run is WNN_IMIT_PROBE=1 on phased_ga). --ignored --nocapture.
	#[test]
	#[ignore]
	fn diag_imitation_probe_light()
	{
		for lag in [0.0f32, 0.20]
		{
			let (c, cfg) = crate::lag_student_law_tests::probe_student_and_cfg(lag, 1);
			super::run(&c, &cfg, [0.0; 3], &[1]);
		}
	}
}

/// Fixture loader (generated from RewardGatedConfigPacked's 77 fields by
/// scripts/diag/dump_recipe_fixture.py's sibling step; keep in sync).
fn cfg_set(c: &mut RewardGatedConfigPacked, k: &str, v: &str)
{
	fn vecp<T: std::str::FromStr>(v: &str) -> Vec<T>
	where
		T::Err: std::fmt::Debug,
	{
		v.split_whitespace().map(|x| x.parse().unwrap()).collect()
	}
	fn arr<T: std::str::FromStr + Copy + Default, const N: usize>(v: &str) -> [T; N]
	where
		T::Err: std::fmt::Debug,
	{
		let x: Vec<T> = vecp(v);
		assert_eq!(x.len(), N, "fixture array length");
		let mut a = [T::default(); N];
		a.copy_from_slice(&x);
		a
	}
	match k
	{
		"num_rounds" => c.num_rounds = v.parse().unwrap(),
		"episodes_per_round" => c.episodes_per_round = v.parse().unwrap(),
		"steps_per_episode" => c.steps_per_episode = v.parse().unwrap(),
		"bptt_window" => c.bptt_window = v.parse().unwrap(),
		"topk_per_neuron" => c.topk_per_neuron = v.parse().unwrap(),
		"protect_learned" => c.protect_learned = v == "true",
		"gate_mode" => c.gate_mode = v.parse().unwrap(),
		"gate_use_best" => c.gate_use_best = v == "true",
		"gate_window" => c.gate_window = v.parse().unwrap(),
		"gate_quantile" => c.gate_quantile = v.parse().unwrap(),
		"gate_running" => c.gate_running = v == "true",
		"target_source" => c.target_source = v.parse().unwrap(),
		"teacher" => c.teacher = v.parse().unwrap(),
		"teacher_schedule" => c.teacher_schedule = vecp(v),
		"teacher_blend" => c.teacher_blend = vecp(v),
		"keep_best_checkpoint" => c.keep_best_checkpoint = v == "true",
		"explore_eps" => c.explore_eps = v.parse().unwrap(),
		"explore_scale" => c.explore_scale = v.parse().unwrap(),
		"curriculum" => c.curriculum = v == "true",
		"easy_tilt_deg" => c.easy_tilt_deg = v.parse().unwrap(),
		"full_tilt_deg" => c.full_tilt_deg = v.parse().unwrap(),
		"dt" => c.dt = v.parse().unwrap(),
		"max_initial_yaw_rad" => c.max_initial_yaw_rad = v.parse().unwrap(),
		"max_initial_body_rate" => c.max_initial_body_rate = v.parse().unwrap(),
		"max_initial_yaw_rate" => c.max_initial_yaw_rate = v.parse().unwrap(),
		"eval_episodes" => c.eval_episodes = v.parse().unwrap(),
		"split_tau" => c.split_tau = v.parse().unwrap(),
		"split_clean_gain" => c.split_clean_gain = v.parse().unwrap(),
		"split_accum_corr" => c.split_accum_corr = v.parse().unwrap(),
		"split_max_rounds" => c.split_max_rounds = v.parse().unwrap(),
		"split_k_start" => c.split_k_start = v.parse().unwrap(),
		"split_coarse_target" => c.split_coarse_target = v.parse().unwrap(),
		"split_selective_output" => c.split_selective_output = v == "true",
		"active_roll" => c.active_roll = v == "true",
		"active_pitch" => c.active_pitch = v == "true",
		"active_yaw" => c.active_yaw = v == "true",
		"dist_enabled" => c.dist_enabled = v == "true",
		"dist_tau_bias" => c.dist_tau_bias = arr(v),
		"dist_gust_sigma" => c.dist_gust_sigma = v.parse().unwrap(),
		"dist_gust_tau_c" => c.dist_gust_tau_c = v.parse().unwrap(),
		"dist_motor_asym" => c.dist_motor_asym = arr(v),
		"dist_gyro_sigma" => c.dist_gyro_sigma = v.parse().unwrap(),
		"dist_gyro_bias_walk" => c.dist_gyro_bias_walk = v.parse().unwrap(),
		"dist_accel_sigma" => c.dist_accel_sigma = v.parse().unwrap(),
		"dist_dropout_prob" => c.dist_dropout_prob = v.parse().unwrap(),
		"dist_dropout_len_steps" => c.dist_dropout_len_steps = v.parse().unwrap(),
		"dist_obs_delay_steps" => c.dist_obs_delay_steps = v.parse().unwrap(),
		"dist_torque_scale_jitter" => c.dist_torque_scale_jitter = v.parse().unwrap(),
		"expert_drives" => c.expert_drives = v == "true",
		"af_arm_length" => c.af_arm_length = v.parse().unwrap(),
		"af_k_thrust" => c.af_k_thrust = v.parse().unwrap(),
		"af_k_drag" => c.af_k_drag = v.parse().unwrap(),
		"af_inertia" => c.af_inertia = arr(v),
		"af_gravity" => c.af_gravity = v.parse().unwrap(),
		"af_pid_att" => c.af_pid_att = arr(v),
		"af_pid_rate" => c.af_pid_rate = arr(v),
		"af_pid_out_limit_n" => c.af_pid_out_limit_n = v.parse().unwrap(),
		"af_pid_hover_n" => c.af_pid_hover_n = v.parse().unwrap(),
		"af_pid_attitude_hz" => c.af_pid_attitude_hz = v.parse().unwrap(),
		"af_pid_lpf_hz" => c.af_pid_lpf_hz = v.parse().unwrap(),
		"write_priority_err" => c.write_priority_err = v == "true",
		"write_err_floor_deg" => c.write_err_floor_deg = v.parse().unwrap(),
		"translation" => c.translation = v == "true",
		"af_mass" => c.af_mass = v.parse().unwrap(),
		"mass_jitter" => c.mass_jitter = v.parse().unwrap(),
		"alt_offset" => c.alt_offset = v.parse().unwrap(),
		"init_vz" => c.init_vz = v.parse().unwrap(),
		"collective_jitter" => c.collective_jitter = v.parse().unwrap(),
		"target_altitude" => c.target_altitude = v.parse().unwrap(),
		"alt_pd_omega" => c.alt_pd_omega = v.parse().unwrap(),
		"alt_pd_zeta" => c.alt_pd_zeta = v.parse().unwrap(),
		"alt_pd_max_delta" => c.alt_pd_max_delta = v.parse().unwrap(),
		"xy_offset" => c.xy_offset = v.parse().unwrap(),
		"lambda_pos" => c.lambda_pos = v.parse().unwrap(),
		"lambda_alt" => c.lambda_alt = v.parse().unwrap(),
		"pos_omega" => c.pos_omega = v.parse().unwrap(),
		"pos_zeta" => c.pos_zeta = v.parse().unwrap(),
		"pos_max_tilt_rad" => c.pos_max_tilt_rad = v.parse().unwrap(),
		"teacher_hover_mode" => c.teacher_hover_mode = v.parse().unwrap(),
		"motor_lag_s" => c.motor_lag_s = v.parse().unwrap(),
		"imu_specific_force" => c.imu_specific_force = v == "true",
		"rotor_drag" => c.rotor_drag = v.parse().unwrap(),
		"att_err_tilt_ref" => c.att_err_tilt_ref = v == "true",
		_ => panic!("fixture: unknown cfg field {k}"),
	}
}
