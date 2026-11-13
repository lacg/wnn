//! IDS-56 MIN-SHARE vote (`ids_vote_share` θ, worker ABI 28). Child module of
//! `vote_writer_tests` (shares its fixtures / trainers / cell readers).
//!
//! A TRUE-direction (own-class) visit adds +(1−θ)·w, any other −θ·w, held as
//! the exact reduced integer ratio own:other (`VoteShare`); the binning is the
//! plain vote's sign rule. θ = 0.5 is 1:1 = today's vote, pinned bit-for-bit
//! against a digest recorded on the unmodified ABI-27 tree (51783592).

use super::*;

fn share(theta: f64) -> VoteShare
{
	VoteShare::from_theta(theta).expect("θ on the grid")
}

fn min_share(mode: u8, theta: f64) -> WriteRule
{
	WriteRule::resolve(mode, Some(share(theta)))
}

fn fnv_cells(h: &mut u64, cells: &[i64])
{
	for c in cells
	{
		for b in c.to_le_bytes()
		{
			*h ^= b as u64;
			*h = h.wrapping_mul(0x0100_0000_01b3);
		}
	}
}

/// FNV-1a over every exported cell of every fixture × vote mode, weighted
/// (cw 1/2/3) and unweighted, under `rule_of(mode)`. CPU, or the Metal
/// trainer when `gpu` (None without a device).
fn vote_digest(rule_of: impl Fn(u8) -> WriteRule, gpu: bool) -> Option<u64>
{
	let mut rng = SmallRng::seed_from_u64(0x1056);
	let mut h = 0xcbf2_9ce4_8422_2325u64;
	for (g, data) in fixtures(&mut rng).iter()
	{
		for mode in VOTE_MODES
		{
			for cw in [Some(&[1u32, 2, 3][..]), None]
			{
				let export = if gpu
				{
					gpu_train(g, data, mode, rule_of(mode), cw)?
				}
				else
				{
					cpu_train(g, data, mode, rule_of(mode), cw).1
				};
				fnv_cells(&mut h, &cells(&export, mode));
			}
		}
	}
	Some(h)
}

/// The ABI-27 plain vote's digest, recorded on the unmodified tree
/// (51783592, `WriteRule::resolve(mode, true)`) with this exact
/// `vote_digest` body — CPU and Metal agree on it.
const ABI27_VOTE_DIGEST: u64 = 0x8f40_9dc8_b487_3024;

/// θ = 0.5 (`VoteShare::MAJORITY`, also what `from_theta(0.5)` returns) trains
/// EXACTLY the ABI-27 vote: same digest on CPU and on Metal.
#[test]
fn min_share_half_is_bit_identical_to_abi27_vote()
{
	assert_eq!(share(0.5), VoteShare::MAJORITY);
	let cpu = vote_digest(|m| min_share(m, 0.5), false).unwrap();
	eprintln!("[vote_share] θ=0.5 CPU digest {cpu:#018x}");
	assert_eq!(cpu, ABI27_VOTE_DIGEST, "θ=0.5 CPU cells drifted from the ABI-27 vote");
	if let Some(gpu) = vote_digest(|m| min_share(m, 0.5), true)
	{
		eprintln!("[vote_share] θ=0.5 GPU digest {gpu:#018x}");
		assert_eq!(gpu, ABI27_VOTE_DIGEST, "θ=0.5 Metal cells drifted from the ABI-27 vote");
	}
}

/// θ = 0.1 (own 9 : other 1), binary single-cluster, unit weights:
///   addr 5: 1 attack + 3 normals  → +9 −3  = +6 → TRUE  (plain vote: FALSE)
///   addr 9: 1 attack + 9 normals  → +9 −9  =  0 → untrained (tie)
///   addr 3: 1 attack + 10 normals → +9 −10 = −1 → FALSE
///   addr 7: 2 normals             → −2         → FALSE
///   addr 0: untouched                          → untrained
#[test]
fn min_share_hand_case_single_cluster()
{
	let mut rows = vec![(5, 1)];
	rows.extend([(5, 0); 3]);
	rows.push((9, 1));
	rows.extend([(9, 0); 9]);
	rows.push((3, 1));
	rows.extend([(3, 0); 10]);
	rows.extend([(7, 0); 2]);
	let data = rows_at(&rows, 1);
	let g = Genome::identity(4);
	for mode in VOTE_MODES
	{
		let untrained = CellMode::from_u8(mode).unwrap().default_cell(ram_core::cell_mode::UnwrittenLean::False) as i64;
		let (plain, _) = cpu_train(&g, &data, mode, vote(mode), None);
		assert_eq!(plain[0].read(0, 5), FALSE, "mode {mode}: plain vote 1 vs 3");
		let (mem, cexp) = cpu_train(&g, &data, mode, min_share(mode, 0.1), None);
		assert_eq!(mem[0].read(0, 5), TRUE, "mode {mode}: θ=0.1 +9 −3");
		assert_eq!(mem[0].read(0, 9), untrained, "mode {mode}: θ=0.1 +9 −9 tie");
		assert_eq!(mem[0].read(0, 3), FALSE, "mode {mode}: θ=0.1 +9 −10");
		assert_eq!(mem[0].read(0, 7), FALSE, "mode {mode}: θ=0.1 normals only");
		assert_eq!(mem[0].read(0, 0), untrained, "mode {mode}: untouched");
		if let Some(gexp) = gpu_train(&g, &data, mode, min_share(mode, 0.1), None)
		{
			assert_eq!(cells(&gexp, mode), cells(&cexp, mode), "mode {mode}: θ=0.1 GPU vs CPU hand case");
		}
	}
}

/// Class-imbalanced weights scale each side BEFORE the share:
///   cw = [normal 3, attack 1], θ = 0.1:
///     addr 5: 1 att + 3 nor → +9·1 − 1·3·3 =  0 → untrained
///     addr 9: 1 att + 4 nor → +9   − 12    = −3 → FALSE
///     addr 3: 2 att + 5 nor → +18  − 15    = +3 → TRUE
///   same rows, θ = 0.25 (3:1): addr 3 → +6 − 15 → FALSE
///   cw = [normal 1, attack 4], θ = 0.1: 1 att + 36 nor → +36 − 36 = 0 (tie),
///     1 att + 37 nor → −1 → FALSE.
#[test]
fn min_share_class_weights_scale_before_the_share()
{
	let g = Genome::identity(4);
	let mut rows = vec![(5, 1)];
	rows.extend([(5, 0); 3]);
	rows.push((9, 1));
	rows.extend([(9, 0); 4]);
	rows.extend([(3, 1); 2]);
	rows.extend([(3, 0); 5]);
	let data = rows_at(&rows, 1);
	let mut heavy = vec![(11, 1)];
	heavy.extend([(11, 0); 36]);
	heavy.push((12, 1));
	heavy.extend([(12, 0); 37]);
	let heavy = rows_at(&heavy, 1);
	for mode in VOTE_MODES
	{
		let untrained = CellMode::from_u8(mode).unwrap().default_cell(ram_core::cell_mode::UnwrittenLean::False) as i64;
		let cw = [3u32, 1];
		let (mem, cexp) = cpu_train(&g, &data, mode, min_share(mode, 0.1), Some(&cw));
		assert_eq!(mem[0].read(0, 5), untrained, "mode {mode}: +9 −9");
		assert_eq!(mem[0].read(0, 9), FALSE, "mode {mode}: +9 −12");
		assert_eq!(mem[0].read(0, 3), TRUE, "mode {mode}: +18 −15");
		let (quarter, _) = cpu_train(&g, &data, mode, min_share(mode, 0.25), Some(&cw));
		assert_eq!(quarter[0].read(0, 3), FALSE, "mode {mode}: θ=0.25 +6 −15");
		let (h, hexp) = cpu_train(&g, &heavy, mode, min_share(mode, 0.1), Some(&[1, 4]));
		assert_eq!(h[0].read(0, 11), untrained, "mode {mode}: +36 −36");
		assert_eq!(h[0].read(0, 12), FALSE, "mode {mode}: +36 −37");
		if let Some(gexp) = gpu_train(&g, &data, mode, min_share(mode, 0.1), Some(&cw))
		{
			assert_eq!(cells(&gexp, mode), cells(&cexp, mode), "mode {mode}: weighted GPU vs CPU");
		}
		if let Some(gexp) = gpu_train(&g, &heavy, mode, min_share(mode, 0.1), Some(&[1, 4]))
		{
			assert_eq!(cells(&gexp, mode), cells(&hexp, mode), "mode {mode}: heavy GPU vs CPU");
		}
	}
}

/// Multi-cluster (all other classes negative), θ = 0.1, unit weights. K = 3,
/// address 5 sees 1 × c0, 4 × c1, 5 × c2:
///   c0: +9 −(4+5)   =  0 → untrained   (plain vote: +1 −9 → FALSE)
///   c1: +36 −(1+5)  = +30 → TRUE
///   c2: +45 −(1+4)  = +40 → TRUE
/// Address 6 sees 1 × c0, 10 × c1: c0 +9 −10 → FALSE, c1 +90 −1 → TRUE,
/// c2 −11 → FALSE.
#[test]
fn min_share_multi_cluster_hand_case()
{
	let mut rows = vec![(5, 0)];
	rows.extend([(5, 1); 4]);
	rows.extend([(5, 2); 5]);
	rows.push((6, 0));
	rows.extend([(6, 1); 10]);
	let data = rows_at(&rows, 3);
	let g = Genome { bits: vec![4; 3], npc: vec![1; 3], conns: (0..3).flat_map(|_| 0..4i64).collect() };
	for mode in VOTE_MODES
	{
		let untrained = CellMode::from_u8(mode).unwrap().default_cell(ram_core::cell_mode::UnwrittenLean::False) as i64;
		let (mem, cexp) = cpu_train(&g, &data, mode, min_share(mode, 0.1), None);
		let s = cexp.groups[0].neurons;
		let at = |a: usize| [mem[0].read(0, a), mem[0].read(s, a), mem[0].read(2 * s, a)];
		assert_eq!(at(5), [untrained, TRUE, TRUE], "mode {mode}: addr 5");
		assert_eq!(at(6), [FALSE, TRUE, FALSE], "mode {mode}: addr 6");
		let (plain, _) = cpu_train(&g, &data, mode, vote(mode), None);
		assert_eq!(plain[0].read(0, 5), FALSE, "mode {mode}: plain vote +1 −9");
		if let Some(gexp) = gpu_train(&g, &data, mode, min_share(mode, 0.1), None)
		{
			assert_eq!(cells(&gexp, mode), cells(&cexp, mode), "mode {mode}: multi-cluster GPU vs CPU");
		}
	}
}

/// The reference tally per neuron: address → Σ signed weights (i64, no
/// saturation). Single-cluster: label 1 is the TRUE direction; K > 1: an
/// example's own cluster gets +own·w, every other cluster −other·w (w = the
/// example's class weight on both sides, as the trainers do).
fn reference_tallies(g: &Genome, data: &Data, k: usize, n: usize, bits: usize, s: VoteShare, cw: &[u32])
	-> Vec<std::collections::HashMap<u64, i64>>
{
	let mut out = vec![std::collections::HashMap::new(); k * n];
	for (ex, &y) in data.targets.iter().enumerate()
	{
		let w = cw[y as usize] as i64;
		for (neuron, tally) in out.iter_mut().enumerate()
		{
			let conns = &g.conns[neuron * bits..(neuron + 1) * bits];
			let a = ram_core::neuron_memory::compute_address_packed_bytes(data.train.packed_row(ex), conns, bits) as u64;
			let own = if k == 1 { y == 1 } else { (neuron / n) as i64 == y };
			let delta = if own { s.own() as i64 * w } else { -(s.other() as i64 * w) };
			*tally.entry(a).or_insert(0) += delta;
		}
	}
	out
}

/// Every cell equals an independent i64 reference tally: Σ own·w over
/// own-class visits − Σ other·w over the rest, sign-binned (tie → untrained),
/// for θ ∈ {0.1, 0.25, 0.5}, single-cluster binary and K = 3, dense and sparse.
#[test]
fn min_share_matches_reference_tally()
{
	let mut rng = SmallRng::seed_from_u64(0x1057);
	for (k, n, bits) in [(1usize, 3usize, 10usize), (1, 3, 14), (3, 2, 10), (3, 2, 13)]
	{
		let g = Genome::random(&vec![n; k], &vec![bits; k * n], &mut rng);
		let data = Data::random(1500, k, &mut rng);
		let cw = [2u32, 1, 3];
		for theta in [0.1, 0.25, 0.5]
		{
			let s = share(theta);
			let reference = reference_tallies(&g, &data, k, n, bits, s, &cw);
			for mode in VOTE_MODES
			{
				let untrained = CellMode::from_u8(mode).unwrap().default_cell(ram_core::cell_mode::UnwrittenLean::False) as i64;
				let (_, export) = cpu_train(&g, &data, mode, min_share(mode, theta), Some(&cw));
				let stride = export.groups[0].neurons;
				for (neuron, tallies) in reference.iter().enumerate()
				{
					let local = (neuron / n) * stride + neuron % n;
					for a in 0..(1u64 << bits)
					{
						let want = match tallies.get(&a).map(|t| t.signum())
						{
							Some(1) => TRUE,
							Some(-1) => FALSE,
							_ => untrained,
						};
						assert_eq!(
							export.read_cell_at(0, local, a, untrained as u8),
							want,
							"k {k} bits {bits} θ {theta} mode {mode}: neuron {neuron} addr {a}"
						);
					}
				}
			}
		}
	}
}

/// θ = 0.1 is order-independent on CPU and Metal, CPU == Metal (cells and
/// exact scores), and genuinely differs from the plain vote on every fixture
/// (so the parity is not vacuous).
#[test]
fn min_share_cpu_gpu_parity_and_order_independence()
{
	let mut rng = SmallRng::seed_from_u64(0x1058);
	let cw: &[u32] = &[1, 2, 3];
	let fx = fixtures(&mut rng);
	let mut differs = 0usize;
	for (i, (g, data)) in fx.iter().enumerate()
	{
		for mode in VOTE_MODES
		{
			let rule = min_share(mode, 0.1);
			let (_, cpu) = cpu_train(g, data, mode, rule, Some(cw));
			let (_, rev) = cpu_train(g, &data.reversed(), mode, rule, Some(cw));
			assert_eq!(cells(&cpu, mode), cells(&rev, mode), "fixture {i} mode {mode}: CPU reversed");
			let (_, plain) = cpu_train(g, data, mode, vote(mode), Some(cw));
			differs += (cells(&cpu, mode) != cells(&plain, mode)) as usize;
			if let Some(gpu) = gpu_train(g, data, mode, rule, Some(cw))
			{
				assert_eq!(cells(&cpu, mode), cells(&gpu, mode), "fixture {i} mode {mode}: CPU vs GPU cells");
				assert_eq!(scores(&cpu, data, mode, false), scores(&gpu, data, mode, false), "fixture {i} mode {mode}: scores");
				let shuffled = gpu_train(g, &data.shuffled(&mut rng), mode, rule, Some(cw)).unwrap();
				assert_eq!(cells(&gpu, mode), cells(&shuffled, mode), "fixture {i} mode {mode}: GPU shuffled");
			}
			else
			{
				eprintln!("[vote_share] no Metal device — GPU half skipped");
			}
		}
	}
	assert_eq!(differs, fx.len() * VOTE_MODES.len(), "θ=0.1 trained the plain vote somewhere");
}
