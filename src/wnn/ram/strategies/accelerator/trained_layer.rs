//! `TrainedLayer`: a genome trained EXACTLY as final validation trains it,
//! returned as deployable physical units (IDS-37 stage 7, the FPGA export).
//!
//! Built by `ids_cache::export_trained_ids_cached` through the same training
//! function as `evaluate_at_thresholds` (`train_genome_export`: the GPU
//! training factory, so the multiplicity layer at rate >= 1.0) and scored by
//! the production scorer — so the exported tables ARE the validated model and
//! the returned scores let the exporter prove it against the DB row.

use crate::adaptive::{cell_weight_quarters, GenomeExport, UnitTable};
use ram_core::footprint::Footprint;
use ram_core::super_genome::FlatGenome;
use ram_core::unit_grouping::UnitGrouping;

pub struct TrainedLayer
{
	/// Logical neuron -> unit
	pub grouping: UnitGrouping,
	/// One table per unit, unit order
	pub units: Vec<UnitTable>,
	/// The cell an absent sparse key reads (mode default, 0 when coverage-aware)
	pub miss_cell: u8,
	/// Integer adder weight per cell code (quarters); None = stochastic mode
	pub cell_weight_quarters: Option<[u32; 4]>,
	pub footprint: Footprint,
	pub eval_scores: Vec<f64>,
	pub val_scores: Option<Vec<f64>>,
}

impl TrainedLayer
{
	pub fn from_export(
		export: &GenomeExport,
		logical: FlatGenome,
		settings: ram_core::neuron_memory::EvalSettings,
		eval_scores: Vec<f64>,
		val_scores: Option<Vec<f64>>,
	) -> Self
	{
		let units = export.unit_tables(logical, settings.coverage_aware);
		Self {
			grouping: Self::grouping_of(export, units.len()),
			units,
			miss_cell: ram_core::metal_sparse::default_cell_for_coverage(
				settings.memory_mode,
				settings.coverage_aware,
			) as u8,
			cell_weight_quarters: cell_weight_quarters(settings.memory_mode, settings.empty_value),
			footprint: export.footprint(),
			eval_scores,
			val_scores,
		}
	}

	/// The export's grouping, or the identity over its units (a plain export
	/// may carry an identity grouping of any size).
	fn grouping_of(export: &GenomeExport, units: usize) -> UnitGrouping
	{
		if export.grouping.is_identity()
		{
			UnitGrouping::identity(units)
		}
		else
		{
			export.grouping.clone()
		}
	}
}
