//! CTRL-49 "E" DIAGNOSIS (docs/ctrl49_input_representation_design.md §2, D0-D7).
//! TEST-ONLY (`#[cfg(test)]`, an `#[ignore]` test, `WNN_IMIT_DIAG=1`): it never reaches a
//! wheel and changes no default path. Offline only: no GA, no DAgger — the states are
//! TEACHER-driven (expert_drives), so the student's cells never matter and the recipe
//! controller from a dump_recipe_fixture.py fixture is used untrained.
//!
//! Pools (one fold of the recipe's own curriculum, plus scorer-tilt pools):
//!   train  — the BC curriculum states (rounds × episodes, 8→30°): the ridge fit set
//!   ho30   — 12 held-out episodes at the full tilt (the CTRL-44b test set)
//!   train5 / ho5 — two disjoint sets of 5° episodes (the scorer's tilt)
//! Strata (every R² three ways):
//!   ALL       train → ho30                       (the CTRL-44b number)
//!   TRANSIENT train → ho30 records with t < 300 ms or max(|roll|,|pitch|) > 10°
//!   HOLD      train5(t > 300 ms) → ho5(t > 300 ms)   (what `steady` scores)
//! The layout (names, axis families) is the ONE Python table, passed as a file
//! (WNN_IMIT_DIAG_LAYOUT, written from feature_layout.py) — never re-declared here.

use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::controller::WnnController;
use crate::dagger_train::RewardGatedConfigPacked;
use crate::imitation_probe::{ImitCapture, Rig};
use crate::imitation_probe_ctbr::{ls, memoriser, phys, score, ys};

const HOLD_T: f64 = 0.3;
const TRANS_TILT_DEG: f64 = 10.0;

/// The recipe's feature table (feature_layout.py): name + axis-family mask.
pub(crate) struct Layout
{
	names: Vec<String>,
	fam: Vec<u8>,
}

impl Layout
{
	pub(crate) fn load(path: &str) -> Self
	{
		let text = std::fs::read_to_string(path).expect("WNN_IMIT_DIAG_LAYOUT");
		let rows: Vec<(String, u8)> = text
			.lines()
			.filter(|l| !l.trim().is_empty())
			.map(|l| {
				let p: Vec<&str> = l.split_whitespace().collect();
				(p[0].to_string(), p[1].parse().expect("family mask"))
			})
			.collect();
		Layout { names: rows.iter().map(|r| r.0.clone()).collect(), fam: rows.iter().map(|r| r.1).collect() }
	}

	fn idx(&self, n: &str) -> usize
	{
		self.names.iter().position(|x| x == n).unwrap_or_else(|| panic!("layout has no {n}"))
	}

	/// Features of the roll (1) or pitch (2) family.
	fn family(&self, bit: u8) -> Vec<usize>
	{
		(0..self.fam.len()).filter(|&f| self.fam[f] & bit != 0).collect()
	}
}

/// The run's fixed geometry for the derived variants.
struct Ctx<'a>
{
	lay: &'a Layout,
	pos: crate::position_loop::PositionLoop,
	leak_ar: f64,  // λ^ar
	sum_ar: f64,   // Σ_{j<ar} λ^j
	scale: f64,
	t_rec: f64,    // seconds per decision record
}

/// A capture + per-record episode step and three feature variants.
struct Pool
{
	cap: ImitCapture,
	step: Vec<usize>,
	y: Vec<[f64; 4]>,
	base: Vec<Vec<f64>>,
	e1: Vec<Vec<f64>>,
	tru: Vec<Vec<f64>>,
}

#[derive(Clone, Copy, PartialEq)]
enum Var
{
	Base,
	E1,
	True,
}

impl Pool
{
	fn rows(&self, v: Var) -> &Vec<Vec<f64>>
	{
		match v
		{
			Var::Base => &self.base,
			Var::E1 => &self.e1,
			Var::True => &self.tru,
		}
	}

	fn tilt_deg(&self, i: usize) -> f64
	{
		self.cap.tx[i][0].abs().max(self.cap.tx[i][1].abs()).to_degrees()
	}
}

/// Fly `plan` = [(round, tilt rad)] teacher-driven episodes, recording each record's step.
fn capture(rig: &mut Rig, c: &mut WnnController, plan: &[(usize, f64)], seed: u64) -> (ImitCapture, Vec<usize>)
{
	let cfg = rig.cfg_bc.clone();
	let mut rng = SmallRng::seed_from_u64(seed);
	let (mut cap, mut step) = (ImitCapture::new(true, false), Vec::new());
	for (ep, &(it, tilt)) in plan.iter().enumerate()
	{
		let n0 = cap.label.len();
		rig.roll(c, &cfg, it, ep, tilt, &mut rng, &mut cap);
		step.extend(0..cap.label.len() - n0);
	}
	(cap, step)
}

/// Re-reference the roll/pitch attitude columns. E1: the tilt-reference target
/// (err' = err + ref, tgt_r/p = ref). TRUE: the estimate replaced by the true Euler
/// angle (err' = −true, target 0). The integrals add the same increment, summed at the
/// decision rate (λ^ar per record, Σλ^j per held tick) — the per-tick integral is not
/// recorded, so this is the decision-rate approximation of it.
fn derive(ctx: &Ctx, base: &[Vec<f64>], cap: &ImitCapture, step: &[usize], e1: bool) -> Vec<Vec<f64>>
{
	let l = ctx.lay;
	let (ix, iy, ivx, ivy) = (l.idx("e_x"), l.idx("e_y"), l.idx("v_x"), l.idx("v_y"));
	let (re, pe, ri, pi) = (l.idx("roll_err"), l.idx("pitch_err"), l.idx("roll_i"), l.idx("pitch_i"));
	let (tr, tp) = (l.idx("tgt_r"), l.idx("tgt_p"));
	let mut acc = [0.0f64; 2];
	let mut out = Vec::with_capacity(base.len());
	for (i, f) in base.iter().enumerate()
	{
		if step[i] == 0
		{
			acc = [0.0; 2];
		}
		let mut g = f.clone();
		let add = if e1
		{
			let (rr, pr) = ctx.pos.tilt_ref(f[ix], f[ivx], f[iy], f[ivy]);
			g[tr] = rr;
			g[tp] = pr;
			[rr, pr]
		}
		else
		{
			[-cap.tx[i][0] - f[re], -cap.tx[i][1] - f[pe]]
		};
		g[re] += add[0];
		g[pe] += add[1];
		for a in 0..2
		{
			acc[a] = ctx.leak_ar * acc[a] + add[a] * ctx.sum_ar;
		}
		g[ri] += ctx.scale * acc[0];
		g[pi] += ctx.scale * acc[1];
		out.push(g);
	}
	out
}

fn make_pool(ctx: &Ctx, c: &WnnController, (cap, step): (ImitCapture, Vec<usize>)) -> Pool
{
	let base: Vec<Vec<f64>> = cap.feats.iter().map(|f| f.iter().map(|&v| v as f64).collect()).collect();
	let e1 = derive(ctx, &base, &cap, &step, true);
	let tru = derive(ctx, &base, &cap, &step, false);
	let y = ys(c, &cap);
	Pool { cap, step, y, base, e1, tru }
}

// ---- ladders + row encoders ------------------------------------------------

/// Quantile ladder at 1/(b+1)..b/(b+1) (the production rule) of `vals`.
fn quantile_ladder(mut vals: Vec<f64>, bpf: usize) -> Vec<f64>
{
	vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let n = vals.len().max(1);
	(1..=bpf).map(|j| vals[((n - 1) as f64 * j as f64 / (bpf + 1) as f64) as usize]).collect()
}

/// Per-feature ladders: the run's own thresholds, except `refit` features, which are
/// quantile-refitted on the given training rows (a variant's changed columns).
fn ladders(c: &WnnController, rows: &[Vec<f64>], tr: &[usize], refit: &[usize]) -> Vec<Vec<f64>>
{
	let (th, nf) = (c.thresholds_ref(), c.num_features());
	let bpf = th.len() / nf;
	(0..nf)
		.map(|f| {
			if refit.contains(&f)
			{
				quantile_ladder(tr.iter().map(|&i| rows[i][f]).collect(), bpf)
			}
			else
			{
				th[f * bpf..(f + 1) * bpf].iter().map(|&v| v as f64).collect()
			}
		})
		.collect()
}

fn bits(x: f64, lad: &[f64]) -> impl Iterator<Item = f64> + '_
{
	lad.iter().map(move |&t| if x >= t { 1.0 } else { 0.0 })
}

/// Feature vector → row. `raw[f]` = keep feature f continuous; else thermometer bits.
fn encode(v: &[f64], lad: &[Vec<f64>], raw: &dyn Fn(usize) -> bool) -> Vec<f64>
{
	let mut r = Vec::with_capacity(v.len() * 8 + 1);
	for (f, &x) in v.iter().enumerate()
	{
		if raw(f)
		{
			r.push(x);
		}
		else
		{
			r.extend(bits(x, &lad[f]));
		}
	}
	r.push(1.0);
	r
}

// ---- strata -----------------------------------------------------------------

struct Stratum<'a>
{
	name: &'static str,
	trp: &'a Pool,
	tr: Vec<usize>,
	tep: &'a Pool,
	te: Vec<usize>,
}

impl Stratum<'_>
{
	fn ytr(&self) -> Vec<[f64; 4]>
	{
		self.tr.iter().map(|&i| self.trp.y[i]).collect()
	}

	fn yte(&self) -> Vec<[f64; 4]>
	{
		self.te.iter().map(|&i| self.tep.y[i]).collect()
	}

	/// Ridge train → test with a row builder `row(pool, i)`; R² per bank.
	fn r2(&self, row: &dyn Fn(&Pool, usize) -> Vec<f64>) -> [f64; 4]
	{
		let xtr: Vec<Vec<f64>> = self.tr.iter().map(|&i| row(self.trp, i)).collect();
		let xte: Vec<Vec<f64>> = self.te.iter().map(|&i| row(self.tep, i)).collect();
		let s = score(&ls(&xtr, &self.ytr(), &xte), &self.yte());
		std::array::from_fn(|m| s[m].2)
	}
}

fn strata<'a>(train: &'a Pool, ho30: &'a Pool, train5: &'a Pool, ho5: &'a Pool, ctx: &Ctx) -> Vec<Stratum<'a>>
{
	let hold = |p: &Pool| (0..p.y.len()).filter(|&i| p.step[i] as f64 * ctx.t_rec > HOLD_T).collect::<Vec<_>>();
	let trans: Vec<usize> = (0..ho30.y.len())
		.filter(|&i| (ho30.step[i] as f64 * ctx.t_rec) < HOLD_T || ho30.tilt_deg(i) > TRANS_TILT_DEG)
		.collect();
	vec![
		Stratum { name: "ALL", trp: train, tr: (0..train.y.len()).collect(), tep: ho30, te: (0..ho30.y.len()).collect() },
		Stratum { name: "TRANSIENT", trp: train, tr: (0..train.y.len()).collect(), tep: ho30, te: trans },
		Stratum { name: "HOLD", trp: train5, tr: hold(train5), tep: ho5, te: hold(ho5) },
	]
}

fn fmt(r: &[f64; 4]) -> String
{
	format!("T:{:.3} p:{:.3} q:{:.3} r:{:.3}", r[0], r[1], r[2], r[3])
}

// ---- the checks ---------------------------------------------------------------

/// p, q of an R² row (the roll/pitch banks the decision tree reads).
fn pq(r: &[f64; 4]) -> [f64; 2]
{
	[r[1], r[2]]
}

type Summary = Vec<(String, [f64; 2])>;

struct Ladders
{
	prod: Vec<Vec<f64>>,     // the run's thresholds (production or A)
	base_refit: Vec<Vec<f64>>, // + roll/pitch attitude columns refitted on this stratum's train
	e1: Vec<Vec<f64>>,
	tru: Vec<Vec<f64>>,
}

fn attitude_cols(l: &Layout, with_targets: bool) -> Vec<usize>
{
	let mut v: Vec<usize> = ["roll_err", "pitch_err", "roll_i", "pitch_i"].iter().map(|n| l.idx(n)).collect();
	if with_targets
	{
		v.extend([l.idx("tgt_r"), l.idx("tgt_p")]);
	}
	v
}

fn stratum_ladders(c: &WnnController, l: &Layout, s: &Stratum) -> Ladders
{
	let att = attitude_cols(l, false);
	let att_t = attitude_cols(l, true);
	Ladders {
		prod: ladders(c, &s.trp.base, &[], &[]),
		base_refit: ladders(c, &s.trp.base, &s.tr, &att),
		e1: ladders(c, &s.trp.e1, &s.tr, &att_t),
		tru: ladders(c, &s.trp.tru, &s.tr, &att),
	}
}

/// D0 + baselines + D6 + D7: raw / thermometer rows under each variant.
fn check_variants(c: &WnnController, s: &Stratum, lad: &Ladders, sum: &mut Summary, tag: &str)
{
	let no = |_f: usize| false;
	let all = |_f: usize| true;
	let mem = score(&s.te.iter().map(|&i| memoriser(c, s.tep.cap.label[i])).collect::<Vec<_>>(), &s.yte());
	let rows: [(&str, Var, Option<&Vec<Vec<f64>>>); 6] = [
		("raw", Var::Base, None),
		("thermo", Var::Base, Some(&lad.prod)),
		("thermo_refit", Var::Base, Some(&lad.base_refit)),
		("D6_raw_true", Var::True, None),
		("D6_thermo_true", Var::True, Some(&lad.tru)),
		("D7_raw_e1", Var::E1, None),
	];
	let mut parts = vec![format!("D0 memoriser {}", fmt(&std::array::from_fn(|m| mem[m].2)))];
	for (name, v, ld) in rows.iter().chain(std::iter::once(&("D7_thermo_e1", Var::E1, Some(&lad.e1))))
	{
		let r = match ld
		{
			Some(ld) => s.r2(&|p: &Pool, i| encode(&p.rows(*v)[i], ld, &no)),
			None => s.r2(&|p: &Pool, i| encode(&p.rows(*v)[i], &lad.prod, &all)),
		};
		parts.push(format!("{name} {}", fmt(&r)));
		sum.push((name.to_string(), pq(&r)));
	}
	eprintln!("[diag] {tag} {} VARIANTS {}", s.name, parts.join(" | "));
}

/// D6 side table: the attitude estimator vs the TRUE Euler angles on the test rows.
fn check_estimator(l: &Layout, s: &Stratum, tag: &str)
{
	let (re, pe) = (l.idx("roll_err"), l.idx("pitch_err"));
	let n = s.te.len().max(1) as f64;
	let stat = |col: usize, a: usize| -> (f64, f64, f64) {
		let (mut se, mut st, mut sx, mut sy, mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
		for &i in &s.te
		{
			let (est, tru) = (-s.tep.base[i][col], s.tep.cap.tx[i][a]);
			se += (est - tru).powi(2);
			st += tru.powi(2);
			sx += est;
			sy += tru;
			sxy += est * tru;
			sxx += est * est;
			syy += tru * tru;
		}
		let cov = sxy / n - sx * sy / n / n;
		let corr = cov / ((sxx / n - (sx / n).powi(2)).max(1e-30) * (syy / n - (sy / n).powi(2)).max(1e-30)).sqrt();
		((se / n).sqrt().to_degrees(), (st / n).sqrt().to_degrees(), corr)
	};
	let (r, p) = (stat(re, 0), stat(pe, 1));
	eprintln!(
		"[diag] {tag} {} ESTIMATOR rms(est−true) roll {:.3}° pitch {:.3}° | rms(true) roll {:.3}° pitch {:.3}° | corr {:.4}/{:.4}",
		s.name, r.0, p.0, r.1, p.1, r.2, p.2
	);
}

/// D1: thermometer everywhere except ONE roll/pitch-family feature kept raw.
fn check_d1(c: &WnnController, l: &Layout, s: &Stratum, lad: &Ladders, sum: &mut Summary, tag: &str)
{
	let no = |_f: usize| false;
	let base = pq(&s.r2(&|p: &Pool, i| encode(&p.base[i], &lad.prod, &no)));
	let mut fam = l.family(1);
	fam.extend(l.family(2));
	let mut parts = Vec::new();
	let mut best = (String::new(), [f64::MIN; 2]);
	for f in fam
	{
		let r = pq(&s.r2(&|p: &Pool, i| encode(&p.base[i], &lad.prod, &|g| g == f)));
		let d = [r[0] - base[0], r[1] - base[1]];
		if d[0] + d[1] > best.1[0] + best.1[1]
		{
			best = (l.names[f].clone(), d);
		}
		parts.push(format!("{} Δp {:+.3} Δq {:+.3}", l.names[f], d[0], d[1]));
	}
	let _ = c;
	eprintln!("[diag] {tag} {} D1 hybrid(f raw) − thermo: {}", s.name, parts.join(" | "));
	sum.push((format!("D1_best:{}", best.0), best.1));
}

/// D2: amplification of the raw linear map, and thermo + the thermometer-coded ŷ.
fn check_d2(l: &Layout, s: &Stratum, lad: &Ladders, sum: &mut Summary, tag: &str)
{
	let raw = |v: &Vec<f64>| v.iter().copied().chain(std::iter::once(1.0)).collect::<Vec<f64>>();
	let xtr: Vec<Vec<f64>> = s.tr.iter().map(|&i| raw(&s.trp.base[i])).collect();
	let y32: Vec<[f32; 4]> = s.ytr().iter().map(|v| v.map(|x| x as f32)).collect();
	let w = crate::imitation_probe::ridge(&xtr, &y32, 4);
	let pred = |v: &Vec<f64>| -> [f64; 4] { let r = raw(v); std::array::from_fn(|m| r.iter().zip(&w[m]).map(|(a, b)| a * b).sum()) };
	let amp = amplification(l, s, &w);
	let yh_tr: Vec<[f64; 4]> = s.tr.iter().map(|&i| pred(&s.trp.base[i])).collect();
	let yl: Vec<Vec<f64>> = (0..4).map(|m| quantile_ladder(yh_tr.iter().map(|v| v[m]).collect(), 8)).collect();
	let no = |_f: usize| false;
	let r = s.r2(&|p: &Pool, i| {
		let mut row = encode(&p.base[i], &lad.prod, &no);
		let yh = pred(&p.base[i]);
		for m in 0..4
		{
			row.extend(bits(yh[m], &yl[m]));
		}
		row
	});
	eprintln!("[diag] {tag} {} D2 amplification A p {:.2} q {:.2} | thermo+ŷ_thermo {}", s.name, amp[0], amp[1], fmt(&r));
	sum.push(("D2_amp".into(), amp));
	sum.push(("D2_thermo_yhat".into(), pq(&r)));
}

/// A_m = Σ_f β_f²·Var(x_f) / Var(y_m) on the test rows (p, q).
fn amplification(l: &Layout, s: &Stratum, w: &[Vec<f64>]) -> [f64; 2]
{
	let nf = l.names.len();
	let var = |xs: Vec<f64>| { let n = xs.len().max(1) as f64; let mu = xs.iter().sum::<f64>() / n; xs.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / n };
	let vx: Vec<f64> = (0..nf).map(|f| var(s.te.iter().map(|&i| s.tep.base[i][f]).collect())).collect();
	std::array::from_fn(|k| {
		let m = k + 1;
		let vy = var(s.te.iter().map(|&i| s.tep.y[i][m]).collect());
		(0..nf).map(|f| w[m][f].powi(2) * vx[f]).sum::<f64>() / vy.max(1e-30)
	})
}

/// D3: raw + pairwise products within the roll family and within the pitch family.
fn check_d3(l: &Layout, s: &Stratum, sum: &mut Summary, tag: &str)
{
	let fams = [l.family(1), l.family(2)];
	let r = s.r2(&|p: &Pool, i| {
		let v = &p.base[i];
		let mut row: Vec<f64> = v.clone();
		for fam in &fams
		{
			for a in 0..fam.len()
			{
				for b in a + 1..fam.len()
				{
					row.push(v[fam[a]] * v[fam[b]]);
				}
			}
		}
		row.push(1.0);
		row
	});
	eprintln!("[diag] {tag} {} D3 raw+within-family products {}", s.name, fmt(&r));
	sum.push(("D3_raw_products".into(), pq(&r)));
}

/// D5: the teacher's hidden plan state (ua, I, Fh, d̂), raw and thermometer-coded.
fn check_d5(s: &Stratum, lad: &Ladders, sum: &mut Summary, tag: &str)
{
	let hl: Vec<Vec<f64>> = (0..12).map(|k| quantile_ladder(s.tr.iter().map(|&i| s.trp.cap.hidden[i][k]).collect(), 8)).collect();
	let no = |_f: usize| false;
	let mut parts = Vec::new();
	for (name, ks) in [("ua", 0..3), ("I", 3..6), ("Fh", 6..9), ("dhat", 9..12), ("all", 0..12)]
	{
		let ks: Vec<usize> = ks.collect();
		let r = s.r2(&|p: &Pool, i| {
			let mut row = encode(&p.base[i], &lad.prod, &no);
			for &k in &ks
			{
				row.extend(bits(p.cap.hidden[i][k], &hl[k]));
			}
			row
		});
		parts.push(format!("thermo+{name}_thermo {}", fmt(&r)));
		sum.push((format!("D5_thermo_{name}"), pq(&r)));
	}
	let all = |_f: usize| true;
	let r = s.r2(&|p: &Pool, i| {
		let mut row = encode(&p.base[i], &lad.prod, &all);
		row.extend_from_slice(&p.cap.hidden[i]);
		row
	});
	parts.push(format!("raw+all {}", fmt(&r)));
	sum.push(("D5_raw_all".into(), pq(&r)));
	eprintln!("[diag] {tag} {} D5 hidden {}", s.name, parts.join(" | "));
}

// ---- D4: the majority oracle on synthetic connectivity draws -----------------------

/// Thermometer frame bits of rows `idx` of pool `p` under a variant + ladder.
fn frames(p: &Pool, idx: &[usize], v: Var, lad: &[Vec<f64>]) -> Vec<Vec<bool>>
{
	let no = |_f: usize| false;
	idx.iter()
		.map(|&i| {
			let r = encode(&p.rows(v)[i], lad, &no);
			r[..r.len() - 1].iter().map(|&b| b > 0.5).collect()
		})
		.collect()
}

/// The frames + target bits a D4 oracle reads (train side subsampled to `cap`).
struct OracleSet
{
	tr_f: Vec<Vec<bool>>,
	tr_t: Vec<Vec<bool>>,
	te_f: Vec<Vec<bool>>,
	te_t: Vec<Vec<bool>>,
}

fn oracle_set(c: &WnnController, s: &Stratum, v: Var, lad: &[Vec<f64>], cap: usize) -> OracleSet
{
	let stride = (s.tr.len() / cap.max(1)).max(1);
	let tr: Vec<usize> = s.tr.iter().copied().step_by(stride).collect();
	let tb = |p: &Pool, idx: &[usize]| idx.iter().map(|&i| c.diag_targets(p.cap.label[i]).1).collect::<Vec<_>>();
	OracleSet {
		tr_f: frames(s.trp, &tr, v, lad),
		tr_t: tb(s.trp, &tr),
		te_f: frames(s.tep, &s.te, v, lad),
		te_t: tb(s.tep, &s.te),
	}
}

/// `n` neurons × `width` distinct frame bits; axis-RF (bank n / L, cross 0.2) when `rf`.
fn draw_maps(n: usize, nfb: usize, width: usize, rf: Option<(&Layout, usize, usize)>, seed: u64) -> Vec<Vec<usize>>
{
	let mut rng = SmallRng::seed_from_u64(seed);
	let mut flat: Vec<i64> = Vec::with_capacity(n * width);
	for _ in 0..n
	{
		let mut pool: Vec<usize> = (0..nfb).collect();
		for t in 0..width
		{
			let k = rng.gen_range(t..nfb);
			pool.swap(t, k);
			flat.push(pool[t] as i64);
		}
	}
	if let Some((l, levels, bpf)) = rf
	{
		let offsets: Vec<usize> = (0..=n).map(|i| i * width).collect();
		let layout = crate::receptive_fields::RfLayout {
			space: nfb,
			frame_bits: nfb,
			bpf,
			levels,
			bank_family: &[8, 1, 2, 4],
			feature_family: &l.fam,
			max_cross: (0.2 * width as f64 + 1e-9).floor() as usize,
		};
		let draw = crate::receptive_fields::RfDraw { seed: seed ^ 0x4F, generation: 0, genome: 0, layer: 0 };
		crate::receptive_fields::project_receptive_fields(&mut flat, &offsets, &layout, draw).expect("rf projection");
	}
	flat.chunks(width).map(|ch| ch.iter().map(|&b| b as usize).collect()).collect()
}

/// Per-neuron MAJORITY cells counted on (frames, target bits), read on `read`:
/// SELF (count = read = test) is the aliasing floor of any memoryless lookup with
/// these maps; GEN (count = train, read = test, an unseen address = the untrained
/// cell) is what such a lookup generalises to. Decoded per bank; R² vs the label.
fn oracle_r2(c: &WnnController, s: &Stratum, count: (&[Vec<bool>], &[Vec<bool>]), read: &[Vec<bool>], maps: &[Vec<usize>]) -> [f64; 4]
{
	let addr = |f: &Vec<bool>, n: usize| maps[n].iter().enumerate().fold(0u64, |a, (j, &b)| a | ((f[b] as u64) << j));
	let nn = maps.len();
	let mut cnt: Vec<HashMap<u64, (u32, u32)>> = (0..nn).map(|_| HashMap::new()).collect();
	for (f, t) in count.0.iter().zip(count.1)
	{
		for n in 0..nn
		{
			let e = cnt[n].entry(addr(f, n)).or_insert((0, 0));
			if t[n] { e.0 += 1 } else { e.1 += 1 }
		}
	}
	let mode = c.memory_mode_u8();
	let (tc, fc, ec) = (crate::cell_mode::true_cell(mode), crate::cell_mode::false_cell(mode), crate::cell_mode::canonical_default_cell(mode));
	let lv = c.levels_per_motor();
	let pred: Vec<[f64; 4]> = read
		.iter()
		.map(|f| {
			let cells: Vec<u8> = (0..nn)
				.map(|n| match cnt[n].get(&addr(f, n)) { Some(&(t, fl)) => if t >= fl { tc } else { fc }, None => ec })
				.collect();
			std::array::from_fn(|m| phys(c, m, crate::cell_mode::decode_motor_cells(&cells[m * lv..(m + 1) * lv], mode, c.output_decode_u8())))
		})
		.collect();
	let sc = score(&pred, &s.yte());
	std::array::from_fn(|m| sc[m].2)
}

fn check_d4(c: &WnnController, l: &Layout, s: &Stratum, lad: &Ladders, sum: &mut Summary, tag: &str)
{
	let nf = l.names.len();
	let bpf = c.thresholds_ref().len() / nf;
	let (nn, lv) = (4 * c.levels_per_motor(), c.levels_per_motor());
	let cap = env_usize("WNN_IMIT_DIAG_D4_TRAIN", 12000);
	let ob = oracle_set(c, s, Var::Base, &lad.prod, cap);
	let oe = oracle_set(c, s, Var::E1, &lad.e1, cap);
	let mut parts = Vec::new();
	for (width, draws) in [(24usize, env_usize("WNN_IMIT_DIAG_D4_DRAWS", 8) as u64), (48, 4)]
	{
		for (name, rf, o) in [("spread", false, &ob), ("axisRF", true, &ob), ("axisRF+E1", true, &oe), ("spread+E1", false, &oe)]
		{
			let (mut gen, mut slf) = ([0.0; 4], [0.0; 4]);
			for d in 0..draws
			{
				let maps = draw_maps(nn, nf * bpf, width, rf.then_some((l, lv, bpf)), 0x4D4 + d * 7919 + width as u64);
				let g = oracle_r2(c, s, (&o.tr_f, &o.tr_t), &o.te_f, &maps);
				let sf = oracle_r2(c, s, (&o.te_f, &o.te_t), &o.te_f, &maps);
				for m in 0..4
				{
					gen[m] += g[m] / draws as f64;
					slf[m] += sf[m] / draws as f64;
				}
			}
			parts.push(format!("w{width} {name} GEN {} SELF {}", fmt(&gen), fmt(&slf)));
			sum.push((format!("D4gen_w{width}_{name}"), pq(&gen)));
			sum.push((format!("D4self_w{width}_{name}"), pq(&slf)));
		}
	}
	eprintln!("[diag] {tag} {} D4 oracle(majority, synthetic maps) {}", s.name, parts.join(" | "));
}

// ---- entry ----------------------------------------------------------------------------

fn env_usize(k: &str, d: usize) -> usize
{
	std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn context<'a>(c: &WnnController, cfg: &RewardGatedConfigPacked, lay: &'a Layout) -> Ctx<'a>
{
	let p = c.obs_params();
	let (leak, scale, dt) = (p.6 as f64, p.7 as f64, p.12 as f64);
	let ar = c.action_repeat_n();
	Ctx {
		lay,
		pos: crate::episode_regime::pos_loop_for(cfg).expect("E1 preview needs the horizontal channel"),
		leak_ar: leak.powi(ar as i32),
		sum_ar: (0..ar).map(|j| leak.powi(j as i32)).sum(),
		scale,
		t_rec: ar as f64 * dt,
	}
}

/// The full D0-D7 diagnosis on one recipe fixture; one SUMMARY line per stratum.
pub(crate) fn diagnose(c0: &WnnController, cfg: &RewardGatedConfigPacked, target: [f32; 3], seed: u64, lay: &Layout, tag: &str)
{
	use crate::dagger_train::{sim_default, AirframeRs, TeacherBank};
	assert_eq!(lay.names.len(), c0.num_features(), "layout file does not match the fixture's feature count");
	let mut cfg_bc = cfg.clone();
	cfg_bc.expert_drives = true;
	let mut rig = Rig { cfg_bc, teachers: TeacherBank::new(AirframeRs::from_cfg(cfg)), sim: sim_default(cfg), target };
	let mut c = c0.clone();
	c.set_observer_motor_lag_core(rig.sim.motor_lag());
	let ctx = context(&c, cfg, lay);
	let rounds = env_usize("WNN_IMIT_DIAG_ROUNDS", cfg.num_rounds);
	let plan_tr: Vec<(usize, f64)> = (0..rounds).flat_map(|it| (0..cfg.episodes_per_round).map(move |_| it)).map(|it| (it, cfg.round_tilt_rad(it))).collect();
	let n30 = env_usize("WNN_IMIT_DIAG_HO30", 12);
	let n5 = env_usize("WNN_IMIT_DIAG_EPS5", 40);
	let full = vec![(0usize, cfg.full_tilt_deg.to_radians()); n30];
	let five = vec![(0usize, 5f64.to_radians()); n5];
	let mut pool = |plan: &[(usize, f64)], sd: u64| {
		let cs = capture(&mut rig, &mut c, plan, sd);
		make_pool(&ctx, &c, cs)
	};
	let train = pool(&plan_tr, seed ^ 0xB0C);
	let ho30 = pool(&full, seed ^ 0x4E1D);
	let train5 = pool(&five, seed ^ 0x5A5);
	let ho5 = pool(&five, seed ^ 0x5B5);
	eprintln!("[diag] {tag} pools train {} ho30 {} train5 {} ho5 {} records (t_rec {:.3}s)", train.y.len(), ho30.y.len(), train5.y.len(), ho5.y.len(), ctx.t_rec);
	for s in strata(&train, &ho30, &train5, &ho5, &ctx)
	{
		run_stratum(c0, lay, &s, tag);
	}
}

fn run_stratum(c: &WnnController, lay: &Layout, s: &Stratum, tag: &str)
{
	let lad = stratum_ladders(c, lay, s);
	let mut sum: Summary = Vec::new();
	eprintln!("[diag] {tag} {} n_train {} n_test {}", s.name, s.tr.len(), s.te.len());
	check_variants(c, s, &lad, &mut sum, tag);
	check_estimator(lay, s, tag);
	check_d1(c, lay, s, &lad, &mut sum, tag);
	check_d2(lay, s, &lad, &mut sum, tag);
	check_d3(lay, s, &mut sum, tag);
	check_d5(s, &lad, &mut sum, tag);
	if std::env::var("WNN_IMIT_DIAG_SKIP_D4").is_err()
	{
		check_d4(c, lay, s, &lad, &mut sum, tag);
	}
	let body: Vec<String> = sum.iter().map(|(k, v)| format!("{k}={:.4},{:.4}", v[0], v[1])).collect();
	eprintln!("[diag] SUMMARY tag={tag} stratum={} {}", s.name, body.join(" "));
}

#[cfg(test)]
mod tests
{
	/// CTRL-49 diagnosis: WNN_IMIT_DIAG=1 WNN_IMIT_FIXTURE=<fixture> WNN_IMIT_DIAG_LAYOUT=<layout>
	/// [WNN_IMIT_DIAG_TAG=<tag>] cargo test --release ... diag_ctrl49 -- --ignored --nocapture
	#[test]
	#[ignore]
	fn diag_ctrl49_input_representation()
	{
		if std::env::var("WNN_IMIT_DIAG").ok().as_deref() != Some("1")
		{
			eprintln!("[diag] WNN_IMIT_DIAG != 1 — skipped");
			return;
		}
		let (c, cfg, target, seeds) = crate::imitation_probe::from_fixture(&std::env::var("WNN_IMIT_FIXTURE").expect("WNN_IMIT_FIXTURE"));
		let lay = super::Layout::load(&std::env::var("WNN_IMIT_DIAG_LAYOUT").expect("WNN_IMIT_DIAG_LAYOUT"));
		let tag = std::env::var("WNN_IMIT_DIAG_TAG").unwrap_or_else(|_| "diag".into());
		super::diagnose(&c, &cfg, target, seeds[0], &lay, &tag);
	}
}
