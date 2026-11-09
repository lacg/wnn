//! `VoteShare` — the MIN-SHARE weights of `WriteRule::Vote` (IDS-56,
//! docs/ids55_family_fix_decomposition.md "Lever B").
//!
//! The vote writer keeps a signed per-(neuron, address) tally. With share θ an
//! own-class (TRUE-direction) visit adds `+(1−θ)·w` and any other visit
//! `−θ·w` (w = the example's class-frequency weight, exactly the weight the
//! plain vote already uses). Commit is unchanged: tally > 0 → TRUE, < 0 →
//! FALSE, 0 → the mode's untrained cell (`neuron_memory::vote_bin_to_cell`).
//! So a cell commits TRUE iff the own class is NOT a clear minority of the
//! weighted visits: p_own > θ.
//!
//! θ is held as the EXACT reduced integer ratio `own : other = (1−θ) : θ` on a
//! 1/100 grid, so the tally is an integer sum (order-independent, identical on
//! CPU and GPU — both add the same `w·own` / `w·other` integers into the same
//! OI packed counter). θ = 0.5 reduces to 1 : 1 = `MAJORITY`, which is
//! literally today's ±w vote — bit-identical by construction, not by rounding.
//!
//! Saturation: the counter `net` is a 30-bit signed field (`OI_NET_MAX` ≈
//! 5.4e8). A cell's |tally| ≤ visits · max_class_weight · max(own, other), so
//! the coarse 1/100 grid (own ≤ 99) is deliberate: it bounds the extra headroom
//! the share costs over the plain vote to ≤ 99× (9× at θ = 0.1).

/// θ is accepted only on this grid (multiples of 1/100). Off-grid values are
/// REJECTED, never rounded, so a flow param means exactly what it says.
pub const VOTE_SHARE_GRID: u32 = 100;

/// The integer weights of one vote visit: `own` per unit class weight on a
/// TRUE-direction visit, `other` on a FALSE-direction visit. Always reduced
/// (gcd 1) and `own >= other >= 1` (θ ∈ (0, 0.5]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoteShare
{
	own: u32,
	other: u32,
}

impl VoteShare
{
	/// θ = 0.5: the plain majority vote (±w), today's `ids_write_rule="vote"`.
	pub const MAJORITY: VoteShare = VoteShare { own: 1, other: 1 };

	/// The share for θ ∈ (0, 0.5] on the 1/100 grid. Anything else (≤ 0,
	/// > 0.5, non-finite, off-grid) is an error.
	pub fn from_theta(theta: f64) -> Result<VoteShare, String>
	{
		if !theta.is_finite() || theta <= 0.0 || theta > 0.5
		{
			return Err(format!("vote share θ={theta} must be in (0, 0.5]"));
		}
		let scaled = theta * VOTE_SHARE_GRID as f64;
		let steps = scaled.round();
		if (scaled - steps).abs() > 1e-6
		{
			return Err(format!("vote share θ={theta} must be a multiple of 1/{VOTE_SHARE_GRID}"));
		}
		let other = steps as u32;
		let own = VOTE_SHARE_GRID - other;
		let g = gcd(own, other);
		Ok(VoteShare { own: own / g, other: other / g })
	}

	/// Weight per unit class weight of a TRUE-direction (own-class) visit.
	#[inline]
	pub fn own(self) -> u32
	{
		self.own
	}

	/// Weight per unit class weight of a FALSE-direction (other-class) visit.
	#[inline]
	pub fn other(self) -> u32
	{
		self.other
	}

	/// θ = other / (own + other) (exact on the grid, for display / logging).
	pub fn theta(self) -> f64
	{
		self.other as f64 / (self.own + self.other) as f64
	}

	/// The tally weight of one visit carrying class weight `weight`.
	#[inline]
	pub fn weight(self, toward_true: bool, weight: u32) -> u32
	{
		weight.saturating_mul(if toward_true { self.own } else { self.other })
	}

	/// The signed tally delta of one visit (the value fed to
	/// `oi_apply_nudge`). GPU twin: marker_train.metal's WNN_WRITE_VOTE arm.
	#[inline]
	pub fn delta(self, toward_true: bool, weight: u32) -> i32
	{
		let magnitude = self.weight(toward_true, weight).min(i32::MAX as u32) as i32;
		if toward_true
		{
			magnitude
		}
		else
		{
			-magnitude
		}
	}
}

fn gcd(a: u32, b: u32) -> u32
{
	if b == 0
	{
		a
	}
	else
	{
		gcd(b, a % b)
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn half_is_the_plain_majority_vote()
	{
		assert_eq!(VoteShare::from_theta(0.5), Ok(VoteShare::MAJORITY));
		assert_eq!(VoteShare::MAJORITY.delta(true, 7), 7);
		assert_eq!(VoteShare::MAJORITY.delta(false, 7), -7);
	}

	#[test]
	fn shares_are_exact_reduced_ratios()
	{
		let s = VoteShare::from_theta(0.1).unwrap();
		assert_eq!((s.own(), s.other()), (9, 1));
		let s = VoteShare::from_theta(0.25).unwrap();
		assert_eq!((s.own(), s.other()), (3, 1));
		let s = VoteShare::from_theta(0.3).unwrap();
		assert_eq!((s.own(), s.other()), (7, 3));
		let s = VoteShare::from_theta(0.01).unwrap();
		assert_eq!((s.own(), s.other()), (99, 1));
		assert!((VoteShare::from_theta(0.3).unwrap().theta() - 0.3).abs() < 1e-12);
	}

	#[test]
	fn out_of_range_or_off_grid_is_rejected()
	{
		for bad in [0.0, -0.1, 0.51, 1.0, f64::NAN, f64::INFINITY, 0.123, 0.005]
		{
			assert!(VoteShare::from_theta(bad).is_err(), "θ={bad} must be rejected");
		}
	}

	#[test]
	fn deltas_scale_by_class_weight()
	{
		let s = VoteShare::from_theta(0.1).unwrap();
		assert_eq!(s.delta(true, 3), 27);
		assert_eq!(s.delta(false, 3), -3);
		assert_eq!(s.weight(true, u32::MAX), u32::MAX);
		assert_eq!(s.delta(true, u32::MAX), i32::MAX);
	}
}
