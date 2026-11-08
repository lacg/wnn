//! IDS receptive fields (flow param `ids_receptive_fields="family"`, worker ABI 27;
//! the IDS port of the controller's CTRL-45, docs/ids_receptive_fields_design.md).
//!
//! A CONSTRAINT on a genome's connectivity map, not a new neuron type: every
//! encoded input bit belongs to one feature FAMILY (a declared group of features
//! with one mechanism — TTL/state, handshake timing, volume, ...), and a neuron
//! may hold at most `budget(w) = floor(cross * w)` of its `w` connections outside
//! its own family. It is ONE idempotent re-projection applied to the output of
//! every genome operator — the `neuron_budget::fit_genome` pattern — so no
//! operator has to know about families.
//!
//! IDS has ONE decision (attack or not), so there is no output bank to key a
//! family on (the controller's choice). A neuron's family is DERIVED from its own
//! connections, never stored and never keyed on its flat position:
//!   - HELD: the eligible family holding a strict MAJORITY of its connections.
//!     After projection the family holds `w - budget(w) > w / 2` (cross < 0.5), so
//!     a projected neuron keeps its family through crossover (whole neurons move),
//!     a few ±2 connection drifts, BITS grow/shrink — idempotent and stable.
//!   - PICKED: a neuron with no majority family (a fresh uniform neuron, or one
//!     scrambled by a per-connection crossover) draws one, weighted by how many of
//!     its own connections each eligible family holds. For a fresh uniform neuron
//!     that is "my first connection picks the family" (Luiz): P(family) = the
//!     family's share of the encoded bits, so the cluster's aggregate wiring per
//!     family matches `spread` — only its CONCENTRATION changes.
//!
//! ELIGIBLE: family f can host a w-wide neuron iff it has at least `w - budget(w)`
//! distinct encoded bits. A smaller family is reachable only through the cross
//! budget at that width (the controller's small-family corner, made explicit:
//! no out-of-family "deficit" that would break the majority rule). A neuron with
//! NO eligible family at its width is EXEMPT: left as it is, drawing nothing.
//!
//! Host-side GA operator: the runtime (CPU trainer, Metal kernels, exporters)
//! only ever sees connection INDICES, so there is no GPU twin to keep in parity.
//! `None` (receptive fields off) never reaches this module: no RNG draw, the
//! operators stay bit-identical (GOLDEN_SEARCH / GOLDEN_OPERATORS).

use rand::Rng;

/// The family map over the encoded input bits + the cross budget.
#[derive(Clone, Debug)]
pub struct ReceptiveFields
{
	bit_family: Vec<u8>,
	family_bits: Vec<Vec<i64>>,
	cross: f64,
}

/// What a neuron's connections say about its family (inspection / tests).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeuronField
{
	/// The held (majority, eligible) family; None = no majority or exempt.
	pub family: Option<u8>,
	/// Connections inside `family` (0 when None).
	pub in_family: usize,
	/// Out-of-family connections allowed at this width.
	pub budget: usize,
	/// True when no family can host this width (left untouched).
	pub exempt: bool,
}

impl ReceptiveFields
{
	/// `bit_family[b]` = family index of encoded bit b; `cross` in [0, 0.5).
	pub fn new(bit_family: Vec<u8>, cross: f64) -> Result<Self, String>
	{
		if !(0.0..0.5).contains(&cross)
		{
			return Err(format!("receptive fields: rf_cross={cross} must be in [0, 0.5) (the majority rule)"));
		}
		if bit_family.is_empty()
		{
			return Err("receptive fields: empty bit->family map".into());
		}
		let n_families = *bit_family.iter().max().unwrap() as usize + 1;
		let mut family_bits = vec![Vec::new(); n_families];
		for (bit, &f) in bit_family.iter().enumerate()
		{
			family_bits[f as usize].push(bit as i64);
		}
		Ok(Self { bit_family, family_bits, cross })
	}

	pub fn total_input_bits(&self) -> usize
	{
		self.bit_family.len()
	}

	pub fn num_families(&self) -> usize
	{
		self.family_bits.len()
	}

	pub fn family_size(&self, family: usize) -> usize
	{
		self.family_bits[family].len()
	}

	/// Out-of-family connections a `width`-wide neuron may hold.
	pub fn budget(&self, width: usize) -> usize
	{
		(self.cross * width as f64 + 1e-9).floor() as usize
	}

	fn eligible(&self, family: usize, width: usize) -> bool
	{
		self.family_bits[family].len() >= width - self.budget(width)
	}

	fn family_of(&self, bit: i64) -> Option<usize>
	{
		usize::try_from(bit).ok().and_then(|b| self.bit_family.get(b)).map(|&f| f as usize)
	}

	/// Connections per family (an out-of-range index counts for no family).
	fn counts(&self, slots: &[i64]) -> Vec<usize>
	{
		let mut counts = vec![0usize; self.num_families()];
		slots.iter().filter_map(|&b| self.family_of(b)).for_each(|f| counts[f] += 1);
		counts
	}

	/// The eligible family holding a strict majority of `slots`, if any.
	fn held(&self, slots: &[i64], counts: &[usize]) -> Option<usize>
	{
		(0..counts.len()).find(|&f| 2 * counts[f] > slots.len() && self.eligible(f, slots.len()))
	}

	/// Read-only view of one neuron (no RNG).
	pub fn field(&self, slots: &[i64]) -> NeuronField
	{
		let counts = self.counts(slots);
		let held = self.held(slots, &counts);
		NeuronField {
			family: held.map(|f| f as u8),
			in_family: held.map_or(0, |f| counts[f]),
			budget: self.budget(slots.len()),
			exempt: !(0..self.num_families()).any(|f| self.eligible(f, slots.len())),
		}
	}

	/// True when the neuron already satisfies the constraint (or is exempt).
	pub fn satisfied(&self, slots: &[i64]) -> bool
	{
		let field = self.field(slots);
		field.exempt || field.family.is_some_and(|_| slots.len() - field.in_family <= field.budget)
	}

	/// Held family, else a draw over the eligible families weighted by the
	/// neuron's own connections in each (by family size if it holds none).
	fn pick_family(&self, slots: &[i64], rng: &mut impl Rng) -> Option<usize>
	{
		let counts = self.counts(slots);
		if let Some(f) = self.held(slots, &counts)
		{
			return Some(f);
		}
		let eligible: Vec<usize> = (0..counts.len()).filter(|&f| self.eligible(f, slots.len())).collect();
		let own: Vec<usize> = eligible.iter().map(|&f| counts[f]).collect();
		let weights = if own.iter().any(|&c| c > 0) { own } else { eligible.iter().map(|&f| self.family_size(f)).collect() };
		weighted_pick(&eligible, &weights, rng)
	}

	/// Re-project every neuron of a flat genome in place; returns how many
	/// neurons changed. Neurons are laid out by `bits_per_neuron`.
	pub fn project(&self, bits_per_neuron: &[usize], connections: &mut [i64], rng: &mut impl Rng) -> Result<usize, String>
	{
		let total: usize = bits_per_neuron.iter().sum();
		if total != connections.len()
		{
			return Err(format!("receptive fields: {} connections for {total} bits", connections.len()));
		}
		let mut start = 0;
		let mut changed = 0;
		for &width in bits_per_neuron
		{
			changed += self.project_neuron(&mut connections[start..start + width], rng) as usize;
			start += width;
		}
		Ok(changed)
	}

	/// One neuron: keep a uniform `budget` of its out-of-family connections and
	/// redraw the rest, in the same slots, from the family's bits it does not
	/// hold. A satisfied (or exempt) neuron draws nothing beyond the family pick.
	fn project_neuron(&self, slots: &mut [i64], rng: &mut impl Rng) -> bool
	{
		if slots.is_empty() || self.satisfied(slots)
		{
			return false;
		}
		let Some(family) = self.pick_family(slots, rng) else { return false };
		let mut out: Vec<usize> = (0..slots.len()).filter(|&j| self.family_of(slots[j]) != Some(family)).collect();
		let budget = self.budget(slots.len());
		if out.len() <= budget
		{
			return false;
		}
		shuffle_prefix(&mut out, budget, rng);
		let fresh = self.fresh_bits(slots, family, out.len() - budget, rng);
		out[budget..].iter().zip(fresh).for_each(|(&slot, bit)| slots[slot] = bit);
		true
	}

	/// `k` distinct family bits the neuron does not hold, uniform without
	/// replacement. Eligibility guarantees the pool is large enough:
	/// pool - held_in >= (w - budget) - (w - out) = out - budget = k.
	fn fresh_bits(&self, slots: &[i64], family: usize, k: usize, rng: &mut impl Rng) -> Vec<i64>
	{
		let held: std::collections::HashSet<i64> = slots.iter().copied().collect();
		let mut pool: Vec<i64> = self.family_bits[family].iter().copied().filter(|b| !held.contains(b)).collect();
		debug_assert!(pool.len() >= k, "eligible family pool {} < {k}", pool.len());
		shuffle_prefix(&mut pool, k, rng);
		pool.truncate(k);
		pool
	}
}

/// Partial Fisher-Yates: afterwards v[..k] is a uniform k-subset of v.
fn shuffle_prefix<T>(v: &mut [T], k: usize, rng: &mut impl Rng)
{
	for t in 0..k.min(v.len())
	{
		let pick = rng.gen_range(t..v.len());
		v.swap(t, pick);
	}
}

/// One item drawn with probability proportional to its weight (None if empty).
fn weighted_pick(items: &[usize], weights: &[usize], rng: &mut impl Rng) -> Option<usize>
{
	let total: usize = weights.iter().sum();
	if total == 0
	{
		return None;
	}
	let mut ticket = rng.gen_range(0..total);
	for (&item, &w) in items.iter().zip(weights)
	{
		if ticket < w
		{
			return Some(item);
		}
		ticket -= w;
	}
	unreachable!("ticket < total")
}

#[cfg(test)]
#[path = "receptive_fields_tests.rs"]
mod tests;
