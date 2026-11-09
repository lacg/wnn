//! CTRL-46 item 1 (ABI 46, `--write-rule vote [--vote-share θ]`): the storage of the
//! controller's DAgger vote tally. The RULE is ram_core's — this file holds only the
//! per-round (neuron, address) → tally map and the walk-visit bookkeeping.
//!
//! WHY. The non-QUAD controller modes (BINARY / TERNARY / PLN) train by
//! `WriteRule::LastWrite` (cell_mode::nudge_cell). In the hold the small roll/pitch
//! command lands on single output neurons whose 24-bit addresses are scattered by
//! bits unrelated to the label, so a cell keeps the sign of the LAST episode that
//! wrote it: sign agreement 0.48-0.60 (docs/ctrl_steady_gap_student.md §5.1). A
//! tally keeps the evidence of every visit instead.
//!
//! SHARED SEMANTICS (ram_core, the IDS-56 path):
//! * the rule is `WriteRule::Vote(share)` from `WriteRule::resolve_controller`
//!   (QUAD family refused there — its nudge lattice already tallies);
//! * a visit's weight is `VoteShare::delta(toward_true, w)`: +own·w TRUE-direction,
//!   −other·w otherwise, own : other = (1−θ) : θ exact on the 1/100 grid. The DAgger
//!   walk does not weight visits, so w = 1;
//! * the commit is `vote_share::vote_net_to_cell(tally, tie)` — the same function
//!   the IDS commit (`neuron_memory::vote_bin_to_cell`) calls — with `tie` from
//!   `CounterBin::for_training(rule, mode)` (FALSE for BINARY, EMPTY for TERNARY/PLN).
//! The IDS keeps the tally in the OI packed counter's 30-bit `net`; the controller
//! keeps an i64 per key (no saturation bound to reason about).
//!
//! SCOPE = ONE DAgger ROUND. The tally collects every visit of the round's gated
//! episodes, then commits once (before the round's closed-loop eval, so the next
//! round's rollouts fly the committed memory) and is dropped. A cell the round did
//! not visit keeps its value; a visited cell takes the round's verdict — the
//! round-granular analogue of last-write-wins' "later evidence corrects earlier".
//! A cross-round tally was rejected: its memory grows with every round × fold
//! (O(distinct visited addresses), unbounded across the warm-start chain), while one
//! round is bounded by episodes × decisions × neurons.
//!
//! ORDER-INDEPENDENT by construction: the tally is an integer sum keyed by
//! (layer, neuron, address); the commit writes distinct keys. During a round the
//! memory is FROZEN (the walk records visits, it writes nothing), so a record's
//! address / solve never depends on another record's commit — trajectories,
//! windows and records may be visited in any order, on the CPU or the GPU walk, and
//! produce the same cells.

use ram_core::cell_mode::{CounterBin, WriteRule};
use ram_core::sparse_memory::SparseLayerMemory;
use ram_core::vote_share::{vote_net_to_cell, VoteShare};
use rustc_hash::FxHashMap;

#[derive(Clone, Debug)]
pub(crate) struct VoteTally
{
	share: VoteShare,
	/// The mode's untrained cell a tie commits to (ram_core CounterBin::Vote.tie).
	tie: u8,
	/// (neuron, address) → signed tally, output layer.
	out: FxHashMap<(u32, u64), i64>,
	/// Same for the state layer (sn > 0).
	state: FxHashMap<(u32, u64), i64>,
}

impl VoteTally
{
	/// The tally of a `WriteRule::Vote(share)` controller in `memory_mode`; refuses a
	/// mode whose shared resolution is not the vote writer (the QUAD family).
	pub(crate) fn for_mode(memory_mode: u8, share: VoteShare) -> Result<Self, String>
	{
		let rule = WriteRule::resolve_controller(memory_mode, Some(share))?;
		match CounterBin::for_training(rule, memory_mode)
		{
			Some(CounterBin::Vote { tie }) => Ok(VoteTally { share, tie, out: FxHashMap::default(), state: FxHashMap::default() }),
			other => Err(format!("write rule {rule:?} for memory_mode {memory_mode} has no vote counter ({other:?})")),
		}
	}


	/// One visit of `neuron` at `address` toward `toward_true` (w = 1).
	#[inline]
	pub(crate) fn visit(&mut self, state_layer: bool, neuron: usize, address: u64, toward_true: bool)
	{
		let d = self.share.delta(toward_true, 1) as i64;
		let map = if state_layer { &mut self.state } else { &mut self.out };
		*map.entry((neuron as u32, address)).or_insert(0) += d;
	}

	/// A state-layer visit toward a mode-native cell VALUE: the TRUE cell → +, the
	/// FALSE cell → −; any other target (an untrained-cell override) casts no vote.
	#[inline]
	pub(crate) fn visit_state_value(&mut self, neuron: usize, address: u64, target: u8, mode: u8)
	{
		if target == crate::cell_mode::true_cell(mode)
		{
			self.visit(true, neuron, address, true);
		}
		else if target == crate::cell_mode::false_cell(mode)
		{
			self.visit(true, neuron, address, false);
		}
	}

	/// Visited (neuron, address) keys so far (both layers).
	#[cfg(test)]
	pub(crate) fn len(&self) -> usize
	{
		self.out.len() + self.state.len()
	}

	/// Commit every visited key into the two layers. Returns (state cells changed,
	/// output cells changed), "changed" = what a read (miss → the layer's default)
	/// returns differs from the verdict.
	pub(crate) fn commit(self, state_mem: &SparseLayerMemory, out_mem: &SparseLayerMemory) -> (usize, usize)
	{
		(Self::commit_layer(&self.state, state_mem, self.tie), Self::commit_layer(&self.out, out_mem, self.tie))
	}

	fn commit_layer(map: &FxHashMap<(u32, u64), i64>, mem: &SparseLayerMemory, tie: u8) -> usize
	{
		let mut changed = 0usize;
		for (&(n, addr), &t) in map
		{
			let v = vote_net_to_cell(t, tie);
			if mem.read_cell_or_default(n as usize, addr) != v
			{
				mem.write_cell(n as usize, addr, v, true);
				changed += 1;
			}
		}
		changed
	}

	/// Sorted (layer, neuron, address, tally) — test/parity view.
	#[cfg(test)]
	pub(crate) fn sorted(&self) -> Vec<(bool, u32, u64, i64)>
	{
		let mut v: Vec<(bool, u32, u64, i64)> = self
			.out
			.iter()
			.map(|(&(n, a), &t)| (false, n, a, t))
			.chain(self.state.iter().map(|(&(n, a), &t)| (true, n, a, t)))
			.collect();
		v.sort_unstable();
		v
	}
}

#[cfg(test)]
mod tests
{
	use super::*;
	use ram_core::neuron_memory::{BINARY, EMPTY_U8, FALSE_U8, PLN, QSR, QUAD_BINARY, QUAD_WEIGHTED, TERNARY, TRUE_U8};

	fn mem(mode: u8) -> SparseLayerMemory
	{
		SparseLayerMemory::new_with_default(2, 8, crate::cell_mode::canonical_default_cell(mode))
	}

	/// Hand case at θ = 0.25 (own 3 : other 1): 1 TRUE vs 2 FALSE → +3 − 2 = +1 → TRUE
	/// (p_own = 1/3 > θ); 1 TRUE vs 3 FALSE → 0 → untrained; 1 TRUE vs 4 FALSE → −1 → FALSE.
	#[test]
	fn min_share_hand_cases()
	{
		let s = VoteShare::from_theta(0.25).unwrap();
		for (falses, want) in [(2usize, TRUE_U8), (3, EMPTY_U8), (4, FALSE_U8)]
		{
			let mut t = VoteTally::for_mode(TERNARY, s).unwrap();
			t.visit(false, 1, 0x5a, true);
			for _ in 0..falses
			{
				t.visit(false, 1, 0x5a, false);
			}
			let m = mem(TERNARY);
			let (sc, oc) = t.commit(&mem(TERNARY), &m);
			assert_eq!(sc, 0);
			assert_eq!(m.read_cell(1, 0x5a), want, "{falses} FALSE visits");
			assert_eq!(oc, (want != EMPTY_U8) as usize);
		}
	}

	/// θ = 0.5 is the plain ±1 majority; BINARY's untrained and FALSE are both the
	/// deleted cell, so a tie and a FALSE majority both erase a stale TRUE.
	#[test]
	fn majority_and_binary_untrained()
	{
		let m = mem(BINARY);
		m.write_cell(0, 7, TRUE_U8, true);
		m.write_cell(0, 9, TRUE_U8, true);
		let mut t = VoteTally::for_mode(BINARY, VoteShare::from_theta(0.5).unwrap()).unwrap();
		t.visit(false, 0, 7, true);
		t.visit(false, 0, 7, false); // tie → untrained (FALSE for BINARY)
		t.visit(false, 0, 9, true);
		t.visit(false, 0, 9, true);
		t.visit(false, 0, 9, false); // +1 → TRUE (unchanged)
		t.visit(false, 0, 11, true); // fresh TRUE
		let (_, oc) = t.commit(&mem(BINARY), &m);
		assert_eq!(m.read_cell_or_default(0, 7), FALSE_U8);
		assert_eq!(m.read_cell_or_default(0, 9), TRUE_U8);
		assert_eq!(m.read_cell_or_default(0, 11), TRUE_U8);
		assert_eq!(oc, 2, "7 erased + 11 set; 9 unchanged");
	}

	/// Any permutation of the same visits yields the same tally and cells.
	#[test]
	fn order_independent()
	{
		let s = VoteShare::from_theta(0.3).unwrap();
		let mut visits: Vec<(usize, u64, bool)> = (0..400u64).map(|i| ((i % 2) as usize, i % 13, (i * 7919) % 5 < 2)).collect();
		let mut a = VoteTally::for_mode(BINARY, s).unwrap();
		for &(n, ad, tt) in &visits
		{
			a.visit(false, n, ad, tt);
		}
		visits.reverse();
		visits.rotate_left(97);
		let mut b = VoteTally::for_mode(BINARY, s).unwrap();
		for &(n, ad, tt) in &visits
		{
			b.visit(false, n, ad, tt);
		}
		assert_eq!(a.sorted(), b.sorted());
		let (ma, mb) = (mem(BINARY), mem(BINARY));
		a.commit(&mem(BINARY), &ma);
		b.commit(&mem(BINARY), &mb);
		for n in 0..2
		{
			for ad in 0..13
			{
				assert_eq!(ma.read_cell(n, ad), mb.read_cell(n, ad));
			}
		}
	}

	/// The QUAD family has no controller vote writer (refused in ram_core).
	#[test]
	fn quad_modes_are_refused()
	{
		for m in [QUAD_WEIGHTED, QUAD_BINARY, QSR]
		{
			assert!(VoteTally::for_mode(m, VoteShare::MAJORITY).is_err());
		}
		for m in [BINARY, TERNARY, PLN]
		{
			assert!(VoteTally::for_mode(m, VoteShare::MAJORITY).is_ok());
		}
	}

	/// An override target that is neither the TRUE nor the FALSE cell casts no vote.
	#[test]
	fn untrained_state_override_casts_no_vote()
	{
		let mut t = VoteTally::for_mode(TERNARY, VoteShare::MAJORITY).unwrap();
		t.visit_state_value(0, 3, EMPTY_U8, TERNARY);
		assert_eq!(t.len(), 0);
		t.visit_state_value(0, 3, TRUE_U8, TERNARY);
		assert_eq!(t.sorted(), vec![(true, 0, 3, 1)]);
	}
}
