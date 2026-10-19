//! Crossover bit-width rule (flow param `ga_crossover_bits`, 19/10/2026).
//!
//! The phase crossovers move each neuron WITH its own width, so a 34b x 32b
//! cross yields a cluster that is part 34b, part 32b (`Inherit`, the default and
//! the original behaviour). The two blend rules instead draw the child's width
//! from the integer range spanned by the parents (BLX-0), so 34 x 32 can give 33:
//!
//! - `BlendNeuron`: per neuron position i of cluster c, width ~ U[min, max] of
//!   the parents' widths at (c, i). Mixed parents can give 3+ widths per cluster.
//! - `BlendCluster`: per cluster c, ONE width ~ U[min, max] of every width in
//!   either parent's cluster c; every child neuron in c is refit to it.
//!
//! Applied as a post-pass on each crossover child, so it covers every operator
//! (neurons / bits / connections / cluster, uniform and pool-shuffle). Pairing
//! is positional: for pool-shuffle the pool is already shuffled, so position is
//! a random pairing. A position only one parent has keeps its width.
//!
//! Refit when a neuron's width changes: shrinking keeps a random subset of its
//! connections in order (same Fisher-Yates as the bits mutation); growing keeps
//! all of them and tops up first from the partner neurons at that position (not
//! already present), then with random inputs. `Inherit` draws no RNG, so default
//! offspring stay bit-identical (golden digests).

use rand::Rng;

/// Width rule applied to a crossover child.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossoverBits
{
	Inherit,
	BlendNeuron,
	BlendCluster,
}

impl CrossoverBits
{
	/// Flow-param value → rule. Unknown values are an error (no silent default).
	pub fn parse(s: &str) -> Result<Self, String>
	{
		match s
		{
			"inherit" => Ok(CrossoverBits::Inherit),
			"blend_neuron" => Ok(CrossoverBits::BlendNeuron),
			"blend_cluster" => Ok(CrossoverBits::BlendCluster),
			other => Err(format!("ga_crossover_bits: unknown rule {other:?} (inherit|blend_neuron|blend_cluster)")),
		}
	}
}

type Parent = (Vec<usize>, Vec<usize>, Vec<i64>, f64);
type Child = (Vec<usize>, Vec<usize>, Vec<i64>);

/// One parent's view: neuron offsets per cluster and connection offsets per neuron.
struct View<'a>
{
	g: &'a Parent,
	neuron_off: Vec<usize>,
	conn_off: Vec<usize>,
}

impl<'a> View<'a>
{
	fn new(g: &'a Parent) -> Self
	{
		View { g, neuron_off: prefix(&g.1), conn_off: prefix(&g.0) }
	}

	/// Width of the neuron at (cluster, local), if this parent has it.
	fn bits_at(&self, c: usize, local: usize) -> Option<usize>
	{
		(c < self.g.1.len() && local < self.g.1[c]).then(|| self.g.0[self.neuron_off[c] + local])
	}

	/// Connections of the neuron at (cluster, local); empty if absent or connection-less.
	fn conns_at(&self, c: usize, local: usize) -> &'a [i64]
	{
		if self.g.2.is_empty() || self.bits_at(c, local).is_none()
		{
			return &[];
		}
		let n = self.neuron_off[c] + local;
		&self.g.2[self.conn_off[n]..self.conn_off[n + 1]]
	}

	/// Every width in cluster c.
	fn cluster_bits(&self, c: usize) -> &'a [usize]
	{
		if c >= self.g.1.len()
		{
			return &[];
		}
		&self.g.0[self.neuron_off[c]..self.neuron_off[c + 1]]
	}
}

fn prefix(counts: &[usize]) -> Vec<usize>
{
	let mut off = Vec::with_capacity(counts.len() + 1);
	off.push(0);
	for &n in counts
	{
		off.push(off.last().unwrap() + n);
	}
	off
}

/// Apply `rule` to a crossover child of (p1, p2). `Inherit` returns it unchanged.
pub fn apply(rule: CrossoverBits, child: Child, p1: &Parent, p2: &Parent, total_input_bits: usize, rng: &mut impl Rng) -> Child
{
	if rule == CrossoverBits::Inherit
	{
		return child;
	}
	let (v1, v2) = (View::new(p1), View::new(p2));
	let cluster_w: Vec<Option<usize>> = (0..child.1.len())
		.map(|c| if rule == CrossoverBits::BlendCluster { cluster_width(&v1, &v2, c, rng) } else { None })
		.collect();
	refit_child(&child, &v1, &v2, &cluster_w, total_input_bits, rng)
}

/// BlendCluster: one width ~ U[min, max] over both parents' cluster c (None if both empty).
fn cluster_width(v1: &View, v2: &View, c: usize, rng: &mut impl Rng) -> Option<usize>
{
	let all = v1.cluster_bits(c).iter().chain(v2.cluster_bits(c));
	let (lo, hi) = all.fold((usize::MAX, 0), |(lo, hi), &b| (lo.min(b), hi.max(b)));
	(lo <= hi).then(|| draw(lo, hi, rng))
}

fn draw(lo: usize, hi: usize, rng: &mut impl Rng) -> usize
{
	if lo == hi { lo } else { rng.gen_range(lo..=hi) }
}

/// Rebuild the child neuron by neuron with each neuron's target width.
fn refit_child(child: &Child, v1: &View, v2: &View, cluster_w: &[Option<usize>], tib: usize, rng: &mut impl Rng) -> Child
{
	let has_conns = !child.2.is_empty();
	let conn_off = prefix(&child.0);
	let (mut bits, mut conns) = (Vec::with_capacity(child.0.len()), Vec::with_capacity(child.2.len()));
	let mut n = 0;
	for (c, &count) in child.1.iter().enumerate()
	{
		for local in 0..count
		{
			let own_b = child.0[n];
			let w = cluster_w[c].unwrap_or_else(|| neuron_width(own_b, v1.bits_at(c, local), v2.bits_at(c, local), rng));
			bits.push(w);
			if has_conns
			{
				let own = &child.2[conn_off[n]..conn_off[n + 1]];
				let partners = [v1.conns_at(c, local), v2.conns_at(c, local)];
				conns.extend(refit(own, w, &partners, tib, rng));
			}
			n += 1;
		}
	}
	(bits, child.1.clone(), conns)
}

/// BlendNeuron: width ~ U[min, max] of the two parents at this position; a
/// position only one parent has keeps the child's own width.
fn neuron_width(own: usize, b1: Option<usize>, b2: Option<usize>, rng: &mut impl Rng) -> usize
{
	match (b1, b2)
	{
		(Some(a), Some(b)) => draw(a.min(b), a.max(b), rng),
		_ => own,
	}
}

/// Resize one neuron's connections to `w` (see module doc for the rule).
fn refit(own: &[i64], w: usize, partners: &[&[i64]; 2], tib: usize, rng: &mut impl Rng) -> Vec<i64>
{
	if w <= own.len()
	{
		return keep_subset(own, w, rng);
	}
	let mut out = own.to_vec();
	for &p in partners.iter().flat_map(|p| p.iter())
	{
		if out.len() == w
		{
			break;
		}
		if !out.contains(&p)
		{
			out.push(p);
		}
	}
	while out.len() < w
	{
		out.push(rng.gen_range(0..tib as i64));
	}
	out
}

/// Keep `w` of `own` chosen at random, original order preserved.
fn keep_subset(own: &[i64], w: usize, rng: &mut impl Rng) -> Vec<i64>
{
	if w == own.len()
	{
		return own.to_vec();
	}
	let mut idx: Vec<usize> = (0..own.len()).collect();
	for i in 0..w
	{
		let j = rng.gen_range(i..own.len());
		idx.swap(i, j);
	}
	let mut kept = idx[..w].to_vec();
	kept.sort_unstable();
	kept.into_iter().map(|i| own[i]).collect()
}

#[cfg(test)]
#[path = "crossover_bits_tests.rs"]
mod tests;
