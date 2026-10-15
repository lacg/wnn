//! One stratified train fold of an `IDSCache`, stored as ROW INDICES into the
//! cache's full train matrix (13/10/2026 memory fix).
//!
//! Before, the cache materialised every fold's packed rows AND the full train
//! matrix — the train set held twice (8.6 GB + 8.6 GB at 46M rows). A fold now
//! keeps only which rows it owns (in fold order) plus its own targets and
//! negatives (the negatives are seeded per fold, so they are kept verbatim);
//! the packed rows are gathered from the full matrix on demand, byte-identical
//! to the old per-fold copy.

use ram_core::packed_bits::PackedBits;

/// Rows of the full train matrix (fold order) + their labels and negatives.
#[derive(Clone)]
pub struct IDSFold
{
	/// Full-train row index of each fold example, in fold order.
	pub rows: Vec<u32>,
	/// Target class per fold example: [num_examples].
	pub targets: Vec<i64>,
	/// Negative classes per fold example: [num_examples * num_negatives].
	pub negatives: Vec<i64>,
}

impl IDSFold
{
	/// Number of examples in the fold.
	pub fn num_examples(&self) -> usize
	{
		self.rows.len()
	}

	/// Concatenate `folds` (in order) into one row list + labels + negatives —
	/// the k-fold training set's layout, exactly the old merged-subset order.
	pub fn concat<'a, I: Iterator<Item = &'a IDSFold> + Clone>(folds: I) -> IDSFold
	{
		IDSFold {
			rows: folds.clone().flat_map(|f| f.rows.iter().copied()).collect(),
			targets: folds.clone().flat_map(|f| f.targets.iter().copied()).collect(),
			negatives: folds.flat_map(|f| f.negatives.iter().copied()).collect(),
		}
	}

	/// The fold's packed input rows, gathered from the full train matrix.
	pub fn gather_input(&self, full_train: &PackedBits) -> PackedBits
	{
		full_train.gather_rows(self.rows.len(), |k| self.rows[k] as usize)
	}
}
